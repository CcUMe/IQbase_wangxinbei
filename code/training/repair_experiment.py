"""Controlled real-data comparisons for the observed attention saturation."""
import argparse
from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models.rf_model import RFModel
from utils.augmentation import augment_supervised_iq
from training.train import build_optimizer
from utils.common import log_json, model_bundle, runtime, save_checkpoint, seed_all, write_json
from utils.config import make_scheduler, resolve_config, training_options
from utils.dataset import BalancedBatchSampler, PreparedDataset, collate, to_device
from utils.engine import autocast, evaluate, supervised_loss
from utils.head_optimizer import active_heads, clear_inactive_head_gradients


VARIANTS = {
    'control': (False, False, False, 3e-4),
    'qk_only': (True, False, False, 3e-4),
    'qk_guard': (True, False, True, 3e-4),
    'qk_guard_low': (True, False, True, 1e-4),
    'fusion_only': (False, True, False, 3e-4),
    'normalized': (True, True, False, 3e-4),
    'fixed': (True, True, True, 3e-4),
    'lower_lr': (False, False, False, 1e-4),
    'source': (None, None, True, 1e-4),
}


def validation_rows(dataset, seed=984):
    rng = np.random.default_rng(seed)
    rows = dataset.indices
    selected = []
    for task in ['modulation', 'protocol']:
        labels = dataset.meta[task][rows]
        for label in np.unique(labels[labels >= 0]):
            eligible = rows[labels == label]
            selected.extend(rng.choice(eligible, min(128, len(eligible)), replace=False).tolist())
    for length in [512, 1024]:
        eligible = rows[(dataset.meta['task'][rows] == 2) & (dataset.meta['length'][rows] == length)]
        selected.extend(rng.choice(eligible, min(2048, len(eligible)), replace=False).tolist())
    lookup = {int(row): pos for pos, row in enumerate(rows)}
    return [lookup[row] for row in selected], selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', choices=list(VARIANTS), required=True)
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--evaluate-every', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', default='auto')
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--phase-rotation-probability', type=float, default=0.)
    p.add_argument('--label-smoothing', type=float, default=0.)
    p.add_argument('--modulation-phase-probability', type=float, default=0.)
    p.add_argument('--modulation-shift-probability', type=float, default=0.)
    p.add_argument('--selection-metric', choices=['proxy_score', 'modulation_accuracy'], default='proxy_score')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device, _, world = runtime(args.device)
    if world != 1:
        raise ValueError('Each controlled comparison uses one GPU')
    if args.steps < 0 or args.evaluate_every < 1 or args.batch_size < 1:
        p.error('steps must be nonnegative; evaluate-every and batch-size must be positive')
    seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    train, val = PreparedDataset(args.data), PreparedDataset(args.data, 'validation')
    if train.info['fingerprint'] != source['dataset_fingerprint']:
        raise ValueError('Different data split')
    indices, original_rows = validation_rows(val)
    write_json(args.output / 'validation_rows.json', {'fingerprint': train.info['fingerprint'], 'rows': original_rows})
    val_loader = DataLoader(Subset(val, indices), batch_size=128, collate_fn=collate, num_workers=2, pin_memory=device.type == 'cuda')
    config = resolve_config(source['training_config'])
    qk, fusion, skip, lr = VARIANTS[args.variant]
    if qk is None:
        qk = config['model']['encoder']['qk_normalization']
        fusion = config['model']['signal']['normalize_fusion']
    lr = args.learning_rate if args.learning_rate is not None else lr
    config['experiment'] = {'name': args.variant, 'description': 'Controlled repair from identical checkpoint; fixed validation subset.'}
    config['model']['encoder']['qk_normalization'] = qk
    config['model']['signal']['normalize_fusion'] = fusion
    config['optimizer'].update(learning_rate=lr, skip_inactive_heads=skip)
    config['data']['augmentation']['phase_rotation_probability'] = args.phase_rotation_probability
    config['data']['augmentation']['modulation_only'].update(phase_rotation_probability=args.modulation_phase_probability,
                                                            circular_shift_probability=args.modulation_shift_probability)
    config['loss']['supervised']['label_smoothing'] = args.label_smoothing
    config['evaluation']['selection_metric'] = args.selection_metric
    config['training'].update(epochs=1, max_steps=0, gradient_accumulation_steps=1)
    config['data'].update(batch_size=args.batch_size, steps_per_epoch=args.steps)
    config['scheduler'].update(name='constant', warmup_steps=0, warmup_ratio=0.)
    config['runtime'].update(log_every=50, precision='bf16', seed=args.seed)
    options = training_options(config)
    bundle = model_bundle(source)
    bundle['model_config'] = {**bundle['model_config'], 'qk_normalization': qk, 'normalize_fusion': fusion}
    model = RFModel(**bundle).to(device)
    model.load_state_dict(source['model'], strict=True)
    optimizer = build_optimizer(model, config['optimizer'])
    scheduler, _ = make_scheduler(optimizer, config['scheduler'], max(1, args.steps))
    write_json(args.output / 'resolved_config.json', config)
    sampler = BalancedBatchSampler(train, args.batch_size, max(1, args.steps), seed=args.seed)
    loader = DataLoader(train, batch_sampler=sampler, collate_fn=collate, num_workers=2, pin_memory=device.type == 'cuda')
    best = -float('inf')
    norms, losses, history = [], [], []
    started = time.monotonic()

    def assess(step):
        nonlocal best
        metrics = evaluate(model, val_loader, device, bundle['target_scaler'], bundle['schema'], 'fp32')
        score = metrics['proxy_score']
        selection = metrics['classification']['modulation']['accuracy'] if args.selection_metric == 'modulation_accuracy' else score
        row = {'event': 'validation', 'step': step, 'score': score,
               'selection_value': selection, 'selection_metric': args.selection_metric,
               'modulation': metrics['classification']['modulation']['accuracy'],
               'protocol': metrics['classification']['protocol']['accuracy'], 'parameter_proxy': metrics['parameter_proxy'],
               'mean_train_loss': float(np.mean(losses)) if losses else None,
               'gradient_median': float(np.median(norms)) if norms else None,
               'gradient_p95': float(np.quantile(norms, .95)) if norms else None,
               'elapsed_seconds': time.monotonic() - started}
        history.append(row)
        log_json(args.output / 'comparison.log', row)
        payload = dict(model=model.state_dict(), model_config=model.config, schema=bundle['schema'],
            target_scaler=bundle['target_scaler'], parameter_grids=bundle['parameter_grids'],
            dataset_fingerprint=train.info['fingerprint'], stage='supervised', optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), grad_scaler={}, training_config=config, world_size=1,
            codebook_signature=None, epoch=0, batch_index=step - 1, epoch_complete=step == args.steps,
            global_step=step, best_score=max(selection, best), metrics=metrics,
            repair_source_checkpoint=str(args.checkpoint.resolve()), repair_variant=args.variant,
            validation_subset=True, repair_source_steps=source['global_step'] + source.get('repair_source_steps', source.get('initialization_source_steps', 0)))
        save_checkpoint(args.output / 'last.pt', payload)
        if selection > best:
            save_checkpoint(args.output / 'best.pt', payload)
            best = selection
        write_json(args.output / 'history.json', history)
        model.train()
        norms.clear()
        losses.clear()

    assess(0)
    if args.steps == 0:
        return
    for step, cpu_batch in enumerate(loader, 1):
        active = active_heads(cpu_batch, options) if skip else None
        batch = to_device(cpu_batch, device)
        torch.manual_seed(args.seed + step - 1)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, 'bf16'):
            training_iq = augment_supervised_iq(batch, options['augmentation'])
            output = model(training_iq, batch['lengths'])
            loss, parts = supervised_loss(output, batch, bundle['target_scaler'], bundle['parameter_grids'], options)
        if active is not None and parts.get('supcon_coverage', 0) > 0:
            active.add('aux.contrastive.')
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'Nonfinite loss at step {step}')
        loss.backward()
        if skip:
            clear_inactive_head_gradients(model, active, device)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        norms.append(float(norm))
        losses.append(float(loss.detach()))
        if step % 50 == 0:
            log_json(args.output / 'comparison.log', {'event': 'step', 'step': step, 'loss': losses[-1], 'gradient_norm': norms[-1]})
        if step % args.evaluate_every == 0 or step == args.steps:
            assess(step)


if __name__ == '__main__':
    main()
