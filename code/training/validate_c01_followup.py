"""Paired checkpoint/condition evaluation and controlled input interventions.

Read-only with respect to data, checkpoints and training code. Saved predictions
are keyed by prepared-data global rows. No model selection or retraining here.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from models.rf_model import RFModel
from utils.common import TARGETS, model_bundle, read_json, write_json
from utils.metrics import classification_metrics

FEATURES = ['log_rms', 'lag1_coherence', 'spectral_flatness',
            'spectral_effective_bin_fraction', 'dc_fraction', 'high_frequency_power_fraction']
CHECKPOINTS = {
    'pyramid21m': 'logs/supervised_main/20261112_171823_031143/best.pt',
    'finetune96m': 'logs/finetune_v2/20261112_145913_822471/best.pt',
}
VARIANTS = ['original', 'gain_0.25', 'gain_4', 'grid_x2', 'grid_x4',
            'grid_x8', 'grid_x16', 'time_permutation']
SEED = 20261113
DELTA = 1 / 32768


def transform(iq, lengths, rows, variant):
    """All grid/permutation changes retain per-record RMS and padding/length."""
    if variant == 'original':
        return iq.copy()
    valid = np.arange(iq.shape[1])[None] < lengths[:, None]
    x = iq.astype(np.float64)
    if variant.startswith('gain_'):
        return (x * float(variant[5:])).astype(np.float32)
    if variant.startswith('grid_x'):
        step = DELTA * int(variant[6:])
        x = np.rint(x / step) * step
    elif variant == 'time_permutation':
        for j, (row, length) in enumerate(zip(rows, lengths)):
            rng = np.random.default_rng(SEED + int(row))
            x[j, :length] = x[j, rng.permutation(int(length))]
    else:
        raise ValueError(variant)
    x *= valid[..., None]
    before = np.square(iq.astype(np.float64)).sum((1, 2))
    after = np.square(x).sum((1, 2))
    if np.any(after <= 0):
        raise ValueError('Intervention produced an all-zero record')
    x *= np.sqrt(before / after)[:, None, None]
    return x.astype(np.float32)


def batch_rows(rows, meta, batch_size):
    # Native-length batches avoid padding-dependent artifacts.
    for length in np.unique(meta['length'][rows]):
        selected = rows[meta['length'][rows] == length]
        for begin in range(0, len(selected), batch_size):
            yield selected[begin:begin + batch_size], int(length)


@torch.inference_mode()
def predict(model, rows, iq, meta, scaler, device, batch_size, description, variant='original'):
    chunks = {name: [] for name in ['rows', 'modulation', 'protocol', 'parameters',
                                     'relative_l2', 'rms_relative_change']}
    batches = list(batch_rows(rows, meta, batch_size))
    for selected, length in tqdm(batches, desc=description, mininterval=5):
        raw = np.array(iq[selected, :length], copy=True)
        lengths = meta['length'][selected].astype(np.int64)
        x = transform(raw, lengths, selected, variant)
        delta = (x.astype(np.float64) - raw).reshape(len(x), -1)
        energy = np.square(raw.astype(np.float64)).sum((1, 2))
        relative = np.sqrt(np.square(delta).sum(1) / energy)
        rms_change = np.sqrt(np.square(x.astype(np.float64)).sum((1, 2)) / energy) - 1
        output = model(torch.from_numpy(x).to(device), torch.from_numpy(lengths).to(device))
        values = {name: output[name].float().cpu().numpy() for name in ['modulation', 'protocol', 'parameters']}
        values['parameters'] = values['parameters'].astype(np.float64) * np.asarray(scaler['std']) + np.asarray(scaler['mean'])
        if not all(np.isfinite(v).all() for v in values.values()):
            raise FloatingPointError('Nonfinite checkpoint predictions')
        chunks['rows'].append(selected)
        chunks['relative_l2'].append(relative)
        chunks['rms_relative_change'].append(rms_change)
        for name, value in values.items():
            chunks[name].append(value)
    result = {key: np.concatenate(value) for key, value in chunks.items()}
    order = np.argsort(result['rows'])
    return {key: value[order] for key, value in result.items()}


def save_csv(path, records):
    if not records:
        return
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def class_result(logits, y, names, keep=None):
    keep = y >= 0 if keep is None else ((y >= 0) & keep)
    y = y[keep].astype(np.int64)
    z = torch.from_numpy(logits[keep]).float()
    if not len(y):
        return {'support': 0}
    prediction = z.argmax(-1).numpy()
    n = len(names)
    cm = np.bincount(y * n + prediction, minlength=n*n).reshape(n, n)
    result = classification_metrics(cm, names)
    # Conditional slices average F1 over classes with true support in that slice.
    supported = cm.sum(1) > 0
    result['macro_f1'] = float(np.asarray(result['f1'])[supported].mean())
    result['macro_recall'] = float(np.asarray(result['recall'])[supported].mean())
    truth = torch.from_numpy(y)
    logp = z.log_softmax(-1)
    confidence = z.softmax(-1).max(-1).values.numpy()
    ptrue = logp[torch.arange(len(y)), truth]
    wrong = z.clone()
    wrong[torch.arange(len(y)), truth] = -torch.inf
    margin = (z[torch.arange(len(y)), truth] - wrong.max(-1).values).numpy()
    correct = prediction == y
    result.update(cross_entropy=float(-ptrue.mean()), mean_confidence=float(confidence.mean()),
                  confidence_wrong=float(confidence[~correct].mean()) if (~correct).any() else None,
                  margin_median=float(np.median(margin)), margin_p10=float(np.quantile(margin, .1)))
    return result


def task_metrics(pred, meta, schema):
    rows = pred['rows']
    result = {'rows': len(rows), 'classification': {}, 'parameters': {}}
    for task in ['modulation', 'protocol']:
        result['classification'][task] = class_result(pred[task], meta[task][rows], schema['classification_labels'][task])
    result['classification']['first14'] = class_result(pred['modulation'], meta['modulation'][rows],
        schema['classification_labels']['modulation'], meta['modulation'][rows] < 14)
    result['classification']['remaining9'] = class_result(pred['modulation'], meta['modulation'][rows],
        schema['classification_labels']['modulation'], meta['modulation'][rows] >= 14)
    for j, name in enumerate(TARGETS):
        valid = meta['target_valid'][rows, j]
        errors = pred['parameters'][valid, j] - meta['targets'][rows[valid], j]
        result['parameters'][name] = {'support': len(errors),
            'rmse': float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            'mae': float(np.mean(np.abs(errors))) if len(errors) else None,
            'absolute_error_p95': float(np.quantile(np.abs(errors), .95)) if len(errors) else None}
    return result


def prepare(output, data, feature_path, meta):
    path = output / 'selection.npz'
    if path.exists():
        return dict(np.load(path, allow_pickle=False))
    table = pq.read_table(feature_path, columns=['sample_id'] + FEATURES).to_pandas()
    ids = pd.Index(table['sample_id'])
    if not ids.is_unique:
        raise ValueError('Raw sample IDs are not unique')
    locations = ids.get_indexer(meta['sample_id'].astype(str))
    if np.any(locations < 0):
        raise ValueError('Raw feature/cache row join failed')
    values = table[FEATURES].to_numpy()[locations]
    rng = np.random.default_rng(SEED)
    groups = np.full((len(locations), len(FEATURES)), -1, dtype=np.int8)
    thresholds = {}
    train_rows, panel_train, panel_validation = [], [], []
    for label in range(23):
        train = np.flatnonzero((meta['split'] == 0) & (meta['modulation'] == label))
        all_label = np.flatnonzero(meta['modulation'] == label)
        for j, name in enumerate(FEATURES):
            edges = np.quantile(values[train, j], [.25, .5, .75])
            thresholds.setdefault(name, {})[str(label)] = edges.tolist()
            groups[all_label, j] = np.searchsorted(edges, values[all_label, j], side='right')
        for q in range(4):
            pool = train[groups[train, 0] == q]
            sample = rng.choice(pool, 32, replace=False)
            train_rows.extend(sample)
            panel_train.extend(sample[:16])
            pool_val = np.flatnonzero((meta['split'] == 1) & (meta['modulation'] == label) & (groups[:, 0] == q))
            panel_validation.extend(rng.choice(pool_val, 16, replace=False))
    for label in range(8):
        pool = np.flatnonzero((meta['split'] == 0) & (meta['protocol'] == label))
        train_rows.extend(rng.choice(pool, 128, replace=False))
    for length in [512, 1024]:
        pool = np.flatnonzero((meta['split'] == 0) & meta['target_valid'].any(1) & (meta['length'] == length))
        train_rows.extend(rng.choice(pool, 512, replace=False))
    result = dict(validation=np.flatnonzero(meta['split'] == 1), train_sample=np.sort(np.unique(train_rows)),
                  panel_train=np.sort(panel_train), panel_validation=np.sort(panel_validation),
                  features=values, condition_groups=groups)
    np.savez_compressed(path, **result)
    write_json(output / 'training_thresholds.json', thresholds)
    return result


def paired(a, b, y, keep):
    keep = keep & (y >= 0)
    ca = a[keep].argmax(-1) == y[keep]
    cb = b[keep].argmax(-1) == y[keep]
    diff = cb.astype(float) - ca.astype(float)
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.
    return dict(support=len(diff), both_correct=int((ca & cb).sum()), both_wrong=int((~ca & ~cb).sum()),
                only_a_correct=int((ca & ~cb).sum()), only_b_correct=int((~ca & cb).sum()),
                accuracy_a=float(ca.mean()), accuracy_b=float(cb.mean()), difference=float(diff.mean()),
                difference_ci95_normal=[float(diff.mean()-1.96*se), float(diff.mean()+1.96*se)])


def summarize(output, meta, schema, selection):
    all_metrics, slices, paired_results = {}, [], {}
    predictions = {}
    for name in CHECKPOINTS:
        all_metrics[name] = {}
        for split in ['validation', 'train_sample']:
            path = output / f'{name}_{split}.npz'
            if not path.exists():
                continue
            pred = dict(np.load(path, allow_pickle=False))
            predictions[name, split] = pred
            all_metrics[name][split] = task_metrics(pred, meta, schema)
            rows = pred['rows']; y = meta['modulation'][rows]
            for j, feature in enumerate(FEATURES):
                for q in range(4):
                    keep = (y >= 0) & (y < 14) & (selection['condition_groups'][rows, j] == q)
                    m = class_result(pred['modulation'], y, schema['classification_labels']['modulation'], keep)
                    slices.append(dict(model=name, split=split, feature=feature, quartile=q+1,
                        **{k: m.get(k) for k in ['support', 'accuracy', 'macro_f1', 'weighted_f1', 'cross_entropy',
                                               'mean_confidence', 'confidence_wrong', 'margin_median']}))
    for split in ['validation', 'train_sample']:
        if ('pyramid21m', split) not in predictions or ('finetune96m', split) not in predictions:
            continue
        a, b = predictions['pyramid21m', split], predictions['finetune96m', split]
        assert np.array_equal(a['rows'], b['rows'])
        y = meta['modulation'][a['rows']]
        paired_results[split] = {group: paired(a['modulation'], b['modulation'], y, keep)
            for group, keep in [('all', y >= 0), ('first14', (y >= 0) & (y < 14)), ('remaining9', y >= 14)]}
        if split == 'validation':
            ap, bp = a['modulation'].argmax(1), b['modulation'].argmax(1)
            keep = (y >= 0) & (ap != y) & (bp != y)
            common = []
            for i in np.flatnonzero(keep):
                row = a['rows'][i]
                common.append(dict(row=int(row), sample_id=str(meta['sample_id'][row]), label=schema['classification_labels']['modulation'][y[i]],
                    pyramid_prediction=schema['classification_labels']['modulation'][ap[i]],
                    finetune_prediction=schema['classification_labels']['modulation'][bp[i]],
                    rms=float(np.exp(selection['features'][row, 0])), rms_quartile=int(selection['condition_groups'][row, 0])+1,
                    lag1_coherence=float(selection['features'][row, 1])))
            save_csv(output / 'common_modulation_errors.csv', common)
    interventions = []
    for name in CHECKPOINTS:
        for split in ['panel_train', 'panel_validation']:
            base_path = output / f'{name}_{split}_original.npz'
            if not base_path.exists():
                continue
            base = dict(np.load(base_path, allow_pickle=False))
            rows = base['rows']; y = meta['modulation'][rows]
            original_correct = base['modulation'].argmax(1) == y
            q = selection['condition_groups'][rows, 0]
            for variant in VARIANTS:
                path = output / f'{name}_{split}_{variant}.npz'
                if not path.exists():
                    continue
                pred = dict(np.load(path, allow_pickle=False))
                assert np.array_equal(rows, pred['rows'])
                for group, keep in [('first14', y < 14), ('remaining9', y >= 14),
                    ('first14_q1', (y < 14) & (q == 0)), ('first14_q4', (y < 14) & (q == 3))]:
                    m = class_result(pred['modulation'], y, schema['classification_labels']['modulation'], keep)
                    changed_pred = pred['modulation'].argmax(1)
                    eligible = keep & original_correct
                    interventions.append(dict(model=name, split=split, variant=variant, group=group,
                        **{k: m[k] for k in ['support', 'accuracy', 'macro_f1', 'cross_entropy', 'margin_median']},
                        flip_rate=float((changed_pred[keep] != base['modulation'][keep].argmax(1)).mean()),
                        originally_correct_support=int(eligible.sum()),
                        originally_correct_retention=float((changed_pred[eligible] == y[eligible]).mean()) if eligible.any() else None,
                        relative_l2_median=float(np.median(pred['relative_l2'][keep])),
                        relative_l2_p95=float(np.quantile(pred['relative_l2'][keep], .95)),
                        rms_relative_change_max=float(np.max(np.abs(pred['rms_relative_change'][keep])))))
    write_json(output / 'metrics.json', all_metrics)
    write_json(output / 'paired_comparison.json', paired_results)
    save_csv(output / 'condition_metrics.csv', slices)
    save_csv(output / 'intervention_metrics.csv', interventions)
    # Full validation per-class details and parameter strata in original units.
    class_records, parameter_records = [], []
    for name, splits in all_metrics.items():
        if 'validation' not in splits:
            continue
        for task in ['modulation', 'protocol']:
            m = splits['validation']['classification'][task]
            for j, label in enumerate(m['labels']):
                class_records.append(dict(model=name, task=task, label=label, support=m['class_support'][j],
                    precision=m['precision'][j], recall=m['recall'][j], f1=m['f1'][j]))
        pred = predictions[name, 'validation']; rows = pred['rows']
        for j, target in enumerate(TARGETS):
            valid = meta['target_valid'][rows, j]
            truth = meta['targets'][rows, j]
            # Fixed train-side bins for continuous fields, exact values for discrete fields.
            train_truth = meta['targets'][(meta['split'] == 0) & meta['target_valid'][:, j], j]
            unique = np.unique(train_truth)
            edges = np.quantile(train_truth, [.25, .5, .75])
            group = truth if len(unique) <= 32 else np.searchsorted(edges, truth, side='right')
            for value in np.unique(group[valid]):
                keep = valid & (group == value)
                error = pred['parameters'][keep, j] - truth[keep]
                parameter_records.append(dict(model=name, target=target, grouping='exact_value' if len(unique)<=32 else 'train_quartile',
                    group=float(value), support=int(keep.sum()), rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(np.abs(error)))))
    save_csv(output / 'per_class_metrics.csv', class_records)
    save_csv(output / 'parameter_condition_metrics.csv', parameter_records)
    print(json.dumps({'metrics': {name: {split: {task: round(m['accuracy'], 5) for task, m in value['classification'].items()}
        for split, value in splits.items()} for name, splits in all_metrics.items()}}, ensure_ascii=False), flush=True)


def self_test():
    rng = np.random.default_rng(SEED)
    x = (rng.integers(-100, 100, (3, 1024, 2)) * DELTA).astype(np.float32)
    lengths = np.array([512, 1024, 512]); rows = np.array([1, 2, 3])
    x[np.arange(1024)[None] >= lengths[:, None]] = 0
    for variant in VARIANTS:
        y = transform(x, lengths, rows, variant)
        assert np.isfinite(y).all()
        assert np.all(y[np.arange(1024)[None] >= lengths[:, None]] == 0)
        assert np.array_equal(y, transform(x, lengths, rows, variant))
        if not variant.startswith('gain_'):
            np.testing.assert_allclose(np.square(x.astype(float)).sum((1,2)), np.square(y.astype(float)).sum((1,2)), rtol=2e-7)
    m = class_result(np.array([[4., 0., -1.], [4., 0., -1.]], np.float32), np.array([0, 1]), ['a','b','c'])
    assert m['accuracy'] == .5 and abs(m['macro_f1'] - 1/3) < 1e-12
    print('Self-test passed: RMS/padding preservation, deterministic transforms, supported-class F1.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'artifacts/data_full')
    p.add_argument('--features', type=Path, default=ROOT/'artifacts/c01_distribution/20261113/features.parquet')
    p.add_argument('--output', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--phase', choices=['all', 'evaluate', 'intervene', 'summarize'], default='all')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        self_test(); return
    output = args.output or ROOT/'artifacts/c01_followup_validation'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    info = read_json(args.data/'dataset.json')
    with np.load(args.data/'metadata.npz', allow_pickle=False) as f:
        meta = {k: f[k] for k in f.files}
    selection = prepare(output, args.data, args.features, meta)
    protocol = dict(dataset_fingerprint=info['fingerprint'], checkpoints=CHECKPOINTS, seed=SEED,
        device=args.device, precision='FP32, matmul TF32 disabled', batch_size=args.batch_size,
        validation='full unchanged held-out split; native-length batches',
        train_sample='32 per class per RMS quartile + 128 per protocol class + 512 per parameter family; not full train metrics',
        panel='16 per modulation class per train-defined RMS quartile; fixed before predictions',
        threshold_fitting='entire training side only', grid_step=DELTA, variants=VARIANTS,
        limits=['Different architectures/budgets: not causal pretraining ablation',
                'Observational features are not SNR; RMS and relative grid spacing are tied',
                'Artificial grid/permutation sensitivity is not a proof of original-data Bayes ceiling',
                'Conditional macro-F1 averages only true-supported classes; inference always retains all 23 logits'])
    write_json(output/'protocol.json', protocol)
    if args.phase != 'summarize':
        iq = np.load(args.data/'iq.npy', mmap_mode='r')
        for name, checkpoint_path in CHECKPOINTS.items():
            checkpoint = torch.load(ROOT/checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
            if checkpoint['dataset_fingerprint'] != info['fingerprint']:
                raise ValueError('Checkpoint/data fingerprint mismatch')
            model = RFModel(**model_bundle(checkpoint)).to(args.device).eval()
            model.load_state_dict(checkpoint['model'], strict=True)
            print(name, 'parameters', sum(v.numel() for v in model.parameters()), 'epoch', checkpoint['epoch'], flush=True)
            if args.phase in ['all', 'evaluate']:
                for split in ['validation', 'train_sample']:
                    path = output/f'{name}_{split}.npz'
                    if not path.exists():
                        result = predict(model, selection[split], iq, meta, checkpoint['target_scaler'], args.device,
                                         args.batch_size, f'{name}/{split}')
                        np.savez_compressed(path, **result)
                        print(name, split, task_metrics(result, meta, info['schema'])['classification']['modulation']['accuracy'], flush=True)
            if args.phase in ['all', 'intervene']:
                for split in ['panel_train', 'panel_validation']:
                    for variant in VARIANTS:
                        path = output/f'{name}_{split}_{variant}.npz'
                        if not path.exists():
                            result = predict(model, selection[split], iq, meta, checkpoint['target_scaler'], args.device,
                                             args.batch_size, f'{name}/{split}/{variant}', variant)
                            np.savez_compressed(path, **result)
            del model, checkpoint
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()
    summarize(output, meta, info['schema'], selection)
    print('Output:', output, flush=True)


if __name__ == '__main__':
    main()
