"""Reproducible complete-validation tables and paired, per-record comparisons."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from training.validate_c01_followup import paired


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def label(identity):
    run, arm = identity.split('/')
    if run.startswith('calibrated_scaled'):
        return 'Calibrated + frozen + token scale'
    if run.startswith('calibrated_frozen'):
        return 'Calibrated + frozen'
    if run.startswith('calibrated_'):
        return 'Calibrated + joint training'
    return dict(control='Control',fixed_reception='Fixed receiver',learned_reception='Learned receiver',
        raw_front='Extra raw input',lag1_front='Lag-1 carrier input',m4_front='Gated M4 carrier input')[arm]


def plots(output, differences, parameters):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    identities = list(dict.fromkeys(r['experiment'] for r in differences))
    fig, axes = plt.subplots(1,2,figsize=(12,6),sharey=True)
    for axis, task in zip(axes,['modulation','first14']):
        for estimator, offset, colour in [('last',-.13,'#2166ac'),('ema',.13,'#b35806')]:
            chosen = {(r['experiment']):r for r in differences if r['task']==task and r['estimator']==estimator}
            x = np.asarray([100*chosen[k]['difference'] for k in identities])
            lo = np.asarray([100*chosen[k]['ci95_lower'] for k in identities])
            hi = np.asarray([100*chosen[k]['ci95_upper'] for k in identities])
            axis.errorbar(x,np.arange(len(identities))+offset,xerr=np.stack([x-lo,hi-x]),fmt='o',
                capsize=3,color=colour,label=estimator)
        axis.axvline(0,color='grey',linewidth=1)
        axis.set_title(task);axis.set_xlabel('Accuracy change vs matched control (percentage points)')
        axis.grid(axis='x',alpha=.2);axis.legend()
    axes[0].set_yticks(np.arange(len(identities)),[label(k) for k in identities]);axes[0].invert_yaxis()
    fig.suptitle('Paired record intervals on one fixed validation set; one training seed')
    fig.tight_layout();fig.savefig(output/'paired_accuracy.png',dpi=160);fig.savefig(output/'paired_accuracy.pdf');plt.close(fig)
    names = list(dict.fromkeys(r['parameter'] for r in parameters))
    lookup = {(r['experiment'],r['parameter']):r for r in parameters if r['estimator']=='last'}
    matrix = np.asarray([[lookup[k,name]['rmse']/lookup[identities[0],name]['rmse'] for name in names] for k in identities])
    fig,axis = plt.subplots(figsize=(11,6));im=axis.imshow(matrix,cmap='RdBu_r',vmin=.9,vmax=1.1,aspect='auto')
    for i in range(len(identities)):
        for j in range(len(names)):
            axis.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',fontsize=9)
    axis.set_yticks(np.arange(len(identities)),[label(k) for k in identities]);axis.set_xticks(np.arange(len(names)),names,rotation=30,ha='right')
    axis.set_title('Last-weight parameter RMSE / control RMSE (lower is better; original units in CSV)')
    fig.colorbar(im,ax=axis);fig.tight_layout();fig.savefig(output/'parameter_rmse_ratio.png',dpi=160);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = np.load(ROOT / 'artifacts/data_full/metadata.npz', allow_pickle=False)
    classification, parameters, differences, protocols, classes = [], [], [], [], []
    conditions = np.load(ROOT / 'artifacts/c01_followup_validation/20261113_224434/selection.npz')['condition_groups']
    reference = args.runs[0]
    ref_protocol = json.loads((reference / 'protocol.json').read_text())
    reference_preds = {}
    for estimator in ['last', 'ema']:
        reference_preds[estimator] = dict(np.load(reference / f'control_{estimator}_validation.npz'))
    for run in args.runs:
        protocol = json.loads((run / 'protocol.json').read_text())
        for key in ['source_sha256', 'training_stream_sha256', 'steps', 'batch_size', 'seed']:
            if protocol[key] != ref_protocol[key]:
                raise ValueError(f'Unmatched experiment {run}: {key}')
        protocols.append(dict(directory=str(run), protocol=protocol))
        results = json.loads((run / 'results.json').read_text())
        for arm, result in results.items():
            identity = run.name + '/' + arm
            for estimator, metric in result['metrics'].items():
                for task, values in {**metric['classification'], **metric['conditions']}.items():
                    classification.append(dict(experiment=identity, estimator=estimator, task=task,
                        support=values['support'], accuracy=values['accuracy'],
                        macro_f1=values['macro_f1'], weighted_f1=values['weighted_f1']))
                for task in ['modulation', 'protocol']:
                    values = metric['classification'][task]
                    # Label order is preserved by the source schema; no
                    # averaging away of the difficult modulation classes.
                    for j, (precision, recall, f1) in enumerate(zip(values['precision'], values['recall'], values['f1'])):
                        classes.append(dict(experiment=identity, estimator=estimator, task=task,
                            label_index=j, label=values['labels'][j], support=values['class_support'][j],
                            precision=precision, recall=recall, f1=f1))
                for task, values in metric['parameters'].items():
                    parameters.append(dict(experiment=identity, estimator=estimator, parameter=task, **values))
                pred = dict(np.load(run / f'{arm}_{estimator}_validation.npz'))
                ref = reference_preds[estimator]
                assert np.array_equal(pred['rows'], ref['rows'])
                rows = pred['rows']
                masks = [('modulation', np.ones(len(rows), bool)),
                                   ('first14', meta['modulation'][rows] < 14),
                                   ('remaining9', meta['modulation'][rows] >= 14),
                                   ('protocol', np.ones(len(rows), bool))]
                masks.extend((f'first14_rms_q{q+1}', (meta['modulation'][rows] < 14) & (conditions[rows,0] == q)) for q in range(4))
                for task, keep in masks:
                    key = 'protocol' if task == 'protocol' else 'modulation'
                    comparison = paired(ref[key], pred[key], meta[key][rows], keep)
                    lo, hi = comparison.pop('difference_ci95_normal')
                    differences.append(dict(experiment=identity, estimator=estimator, task=task,
                        **comparison, ci95_lower=lo, ci95_upper=hi))
    write_csv(args.output / 'classification.csv', classification)
    write_csv(args.output / 'per_class.csv', classes)
    write_csv(args.output / 'parameters.csv', parameters)
    write_csv(args.output / 'paired_control.csv', differences)
    front_comparisons = []
    for run in args.runs:
        if not (run / 'raw_front_last_validation.npz').exists():
            continue
        for estimator in ['last', 'ema']:
            ref = dict(np.load(run / f'raw_front_{estimator}_validation.npz'))
            for arm in ['lag1_front', 'm4_front']:
                pred = dict(np.load(run / f'{arm}_{estimator}_validation.npz'))
                assert np.array_equal(pred['rows'], ref['rows'])
                y = meta['modulation'][pred['rows']]
                for task, keep in [('modulation', y >= 0), ('first14', y < 14), ('remaining9', y >= 14)]:
                    comparison = paired(ref['modulation'], pred['modulation'], y, keep)
                    lo, hi = comparison.pop('difference_ci95_normal')
                    front_comparisons.append(dict(experiment=run.name+'/'+arm, estimator=estimator, task=task,
                        **comparison, ci95_lower=lo, ci95_upper=hi))
    write_csv(args.output / 'paired_input_projection.csv', front_comparisons)
    plots(args.output, differences, parameters)
    (args.output / 'protocols.json').write_text(json.dumps(protocols, ensure_ascii=False, indent=2))
    # Normal paired CI measures validation-record uncertainty, not seed-to-seed
    # variation, family-wise significance, or hidden-test generalization.
    record = dict(reference_directory=str(reference), matched_estimator=True,
        confidence_interval='Paired normal interval on correctness differences, no multiple-comparison adjustment',
        limitation='Single shared training seed; intervals do not quantify optimization-seed variation',
        validation_rows=len(reference_preds['last']['rows']), runs=[str(p) for p in args.runs])
    (args.output / 'comparison_protocol.json').write_text(json.dumps(record, indent=2))
    lines = ['| 实验 | 权重 | 调制准确率 | 调制Macro F1 | 调制Weighted F1 | 前14类准确率 | 协议准确率 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    lookup = {(r['experiment'], r['estimator'], r['task']): r for r in classification}
    for identity, estimator in dict.fromkeys((r['experiment'], r['estimator']) for r in classification):
        mod = lookup[identity, estimator, 'modulation']
        first = lookup[identity, estimator, 'first14']
        proto = lookup[identity, estimator, 'protocol']
        lines.append(f'| {identity} | {estimator} | {100*mod["accuracy"]:.3f}% | '
            f'{100*mod["macro_f1"]:.3f}% | {100*mod["weighted_f1"]:.3f}% | '
            f'{100*first["accuracy"]:.3f}% | {100*proto["accuracy"]:.3f}% |')
    (args.output / 'metrics_table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    snapshot = args.output / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    hashes = {}
    for relative in ['models/reception.py', 'models/carrier_input.py', 'models/carrier_recovery.py',
                     'training/explore_reception.py', 'training/precalibrate_reception.py',
                     'training/summarize_reception.py', 'training/test_reception.py',
                     'training/inspect_public_mod_metadata.py', 'training/inspect_reception_branch.py',
                     'training/test_carrier_recovery.py', 'training/inspect_reception_calibration.py',
                     'training/fold_raw_input_projection.py', 'training/validate_folded_reception.py']:
        source = ROOT / 'code' / relative
        data = source.read_bytes()
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[relative] = hashlib.sha256(data).hexdigest()
    (snapshot / 'sha256.json').write_text(json.dumps(hashes, indent=2))


if __name__ == '__main__':
    main()
