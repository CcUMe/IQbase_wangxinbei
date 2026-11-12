"""Matched masked-waveform versus phase-invariant local-structure pretraining.

This is an atomic-only diagnostic, not a reproduction of the time-aligned ESU paper.
The codebook is fitted only on training IQ, without modulation labels or fs/SNR.
"""
import argparse
import hashlib
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from models.rf_model import RFModel
from training.build_codebook import cluster
from training.train import build_optimizer
from utils.common import model_bundle, runtime, seed_all, write_json, save_checkpoint, log_json
from utils.config import resolve_config, training_options, make_scheduler
from utils.dataset import PreparedDataset, BalancedBatchSampler, collate, to_device
from utils.engine import autocast, patch_mask, pretraining_loss
from utils.signal_statistics import signal_statistics
from utils.readable_log import ReadableRunLog
from utils.training_display import EpochProgress


@torch.no_grad()
def descriptors(iq, patch=16):
    patches = iq.float().reshape(-1, patch, 2)
    lengths = torch.full((len(patches),), patch, device=iq.device, dtype=torch.long)
    return signal_statistics(patches, lengths)


def quantize(iq, vocabulary, patch=16):
    features = descriptors(iq, patch)
    features = F.normalize(((features - vocabulary['mean']) / vocabulary['std']).clamp(-8, 8), dim=-1)
    labels = (features @ vocabulary['centers'].T).argmax(-1)
    return labels.reshape(len(iq), -1)


def hash_state(model, prefix=None):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if prefix is None or name.startswith(prefix):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_codebook(args, device):
    train = PreparedDataset(args.data)
    _, unique = np.unique(train.meta['hash'][train.indices], return_index=True)
    rows = train.indices[unique]
    buckets = {}
    for task in range(3):
        for length in [512, 1024]:
            selected = unique[(train.meta['task'][rows] == task) & (train.meta['length'][rows] == length)]
            if len(selected):
                buckets[(task, length)] = selected
    rng = np.random.default_rng(args.seed)
    pools, fitted_rows = [], []
    quota = args.max_patches // len(buckets)
    for (task, length), choices in buckets.items():
        positions = rng.choice(choices, min(len(choices), math.ceil(quota / (length // 16))), replace=False)
        loader = DataLoader(Subset(train, positions.tolist()), batch_size=128, collate_fn=collate)
        pool = torch.cat([descriptors(batch['iq'].to(device)).cpu() for batch in loader])
        pools.append(pool[torch.from_numpy(rng.choice(len(pool), min(quota, len(pool)), replace=False))])
        fitted_rows.extend(train.indices[positions].tolist())
    raw = torch.cat(pools)
    mean, std = raw.mean(0), raw.std(0, unbiased=False).clamp_min(1e-4)
    pool = F.normalize(((raw - mean) / std).clamp(-8, 8), dim=-1)
    centers = cluster(pool, args.vocab, 20, device, args.seed)
    labels = (pool @ centers.T).argmax(-1)
    counts = torch.bincount(labels, minlength=args.vocab)
    config = {'kind': 'phase-invariant patch statistics; atomic-only masked prediction',
        'patch_size': 16, 'vocab_size': args.vocab, 'dataset_fingerprint': train.info['fingerprint'],
        'fitted_patches': len(pool), 'fitted_records': len(fitted_rows), 'seed': args.seed,
        'label_usage': counts.tolist(), 'used_codes': int((counts > 0).sum()),
        'fit_split': 'train', 'uses_class_labels': False, 'emergent_units': False,
        'physical_time_alignment': False,
        'fitted_rows_sha256': hashlib.sha256(np.asarray(fitted_rows, dtype='<i8').tobytes()).hexdigest()}
    save_checkpoint(args.output / 'codebook.pt', dict(mean=mean, std=std, centers=centers, config=config))
    write_json(args.output / 'protocol.json', config)
    np.save(args.output / 'fitted_rows.npy', np.asarray(fitted_rows))
    print(config, flush=True)


def train_pretraining(args, device):
    source = torch.load(args.template_checkpoint, map_location='cpu', weights_only=True, mmap=True)
    train, validation = PreparedDataset(args.data), PreparedDataset(args.data, 'validation')
    if source['dataset_fingerprint'] != train.info['fingerprint']:
        raise ValueError('Split mismatch')
    vocabulary = torch.load(args.codebook, map_location='cpu', weights_only=True)
    if vocabulary['config']['dataset_fingerprint'] != train.info['fingerprint']:
        raise ValueError('Codebook split mismatch')
    vocabulary = {k: v.to(device) if torch.is_tensor(v) else v for k, v in vocabulary.items()}
    bundle = model_bundle(source)
    bundle['model_config'] = {**bundle['model_config'], 'qk_normalization': True,
        'pretrain_heads': True, 'atomic_vocab': len(vocabulary['centers']), 'emergent_vocab': 0,
        'contrastive_head': False}
    model = RFModel(**bundle).to(device)
    initial_hash = hash_state(model)
    shared_hash = hash_state(model, 'backbone.')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(('backbone.', 'aux.')))
    config = resolve_config(source['training_config'])
    config['model']['encoder']['qk_normalization'] = True
    config['training'].update(stage='pretrain', epochs=3)
    config['loss']['pretraining']['weights'].update(reconstruction=float(args.objective == 'reconstruction'),
        atomic=float(args.objective == 'structure'), emergent=0., spectrum=0.)
    config['optimizer'].update(learning_rate=3e-4, skip_inactive_heads=True)
    options = training_options(config)
    optimizer = build_optimizer(model, config['optimizer'])
    schedule = dict(name='cosine', warmup_steps=min(100, args.steps // 10), warmup_ratio=0.,
                    warmup_start_factor=0., min_lr_ratio=.05)
    scheduler, _ = make_scheduler(optimizer, schedule, args.steps)
    stream_hash, mask_hash = hashlib.sha256(), hashlib.sha256()
    protocol = {'objective': args.objective, 'steps': args.steps, 'batch_size': args.batch_size,
        'seed': args.seed, 'sampling_seed': 2029, 'initial_state_sha256': initial_hash,
        'initial_backbone_sha256': shared_hash, 'dataset_fingerprint': train.info['fingerprint'],
        'codebook': str(args.codebook.resolve()), 'scheduler': schedule,
        'loss_weights': config['loss']['pretraining']['weights'], 'task_heads_frozen': True,
        'both_heads_in_both_arms': True, 'same_quantization_compute_in_both_arms': True,
        'uses_class_labels': False, 'mask': {'ratio': .3, 'span': 2},
        'atomic_only': True, 'physical_time_alignment': False}
    write_json(args.output / 'protocol.json', protocol)
    write_json(args.output / 'resolved_config.json', config)
    sampler = BalancedBatchSampler(train, args.batch_size, args.steps, seed=2029)
    loader = DataLoader(train, batch_sampler=sampler, collate_fn=collate, num_workers=2, pin_memory=True)
    started = time.monotonic()
    total_parts, completed = {}, 0
    with ReadableRunLog(args.output):
        print(f'训练总览 | {args.objective} 预训练诊断 | {args.steps} steps | batch={args.batch_size}', flush=True)
        print(f'初始化={initial_hash} | backbone={shared_hash} | {model.parameter_report()}', flush=True)
        chunk = max(1, args.steps // 3)
        progress = None
        for step, cpu in enumerate(loader):
            if step % chunk == 0:
                if progress is not None:
                    progress.__exit__(None, None, None)
                progress = EpochProgress(step // chunk, math.ceil(args.steps / chunk), min(chunk, args.steps - step), min_interval=1.)
                progress.__enter__()
            stream_hash.update(np.asarray(cpu['rows'], dtype='<i8').tobytes())
            batch = to_device(cpu, device)
            batch['atomic'] = quantize(batch['iq'], vocabulary)
            mask = patch_mask(batch['lengths'], 16, .3, 2, 12345 + step)
            mask_hash.update(mask.cpu().numpy().tobytes())
            torch.manual_seed(args.seed + 10000000 + step)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, 'bf16'):
                output = model(batch['iq'], batch['lengths'], patch_mask=mask, return_pretrain=True)
                loss, parts = pretraining_loss(output, batch, mask, options)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Nonfinite pretraining loss at {step}')
            loss.backward()
            unused = 'aux.atomic.' if args.objective == 'reconstruction' else 'aux.reconstruction.'
            for name, parameter in model.named_parameters():
                if name.startswith(unused):
                    parameter.grad = None
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            rates = {g['name']: g['lr'] for g in optimizer.param_groups}
            optimizer.step()
            scheduler.step()
            progress.observe(float(loss.detach()), parts, set(parts))
            progress.advance(rates, float(norm))
            completed += 1
            for key, value in parts.items():
                total_parts[key] = total_parts.get(key, 0.) + value
            if completed % 50 == 0:
                log_json(args.output / 'events.jsonl', {'event': 'step', 'step': completed,
                    'loss': float(loss.detach()), 'parts': parts, 'learning_rates': rates}, console=False)
        progress.__exit__(None, None, None)
        model.eval()
        validation_losses = {}
        with torch.inference_mode():
            for split, data in [('train', train), ('validation', validation)]:
                values = []
                samples = BalancedBatchSampler(data, 128, 24, seed=401)
                for index, cpu in enumerate(DataLoader(data, batch_sampler=samples, collate_fn=collate, num_workers=2)):
                    batch = to_device(cpu, device)
                    batch['atomic'] = quantize(batch['iq'], vocabulary)
                    mask = patch_mask(batch['lengths'], 16, .3, 2, 55500 + index)
                    output = model(batch['iq'], batch['lengths'], patch_mask=mask, return_pretrain=True)
                    loss, _ = pretraining_loss(output, batch, mask, options)
                    values.append(float(loss))
                validation_losses[split] = float(np.mean(values))
        save_checkpoint(args.output / 'last.pt', dict(model=model.state_dict(), model_config=model.config,
            **{k: bundle[k] for k in ['schema', 'target_scaler', 'parameter_grids']},
            dataset_fingerprint=train.info['fingerprint'], stage='pretrain', training_config=config,
            global_step=completed, epoch=2, metrics={'pretraining_diagnostic': validation_losses}, resume_allowed=False))
        protocol.update(sample_stream_sha256=stream_hash.hexdigest(), mask_stream_sha256=mask_hash.hexdigest(),
            elapsed_seconds=time.monotonic() - started, mean_training_parts={k:v/completed for k,v in total_parts.items()},
            final_fixed_mask_losses=validation_losses, final_backbone_sha256=hash_state(model, 'backbone.'))
        write_json(args.output / 'protocol.json', protocol)
        print('预训练完成', protocol, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--objective', choices=['fit', 'reconstruction', 'structure'], required=True)
    p.add_argument('--template-checkpoint', type=Path)
    p.add_argument('--codebook', type=Path)
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--vocab', type=int, default=256)
    p.add_argument('--max-patches', type=int, default=65536)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.vocab, args.max_patches) < 1:
        p.error('Positive budgets required')
    args.output.mkdir(parents=True, exist_ok=False)
    device, _, world = runtime(args.device, 4)
    if world != 1:
        raise ValueError('One GPU/process per independent arm')
    seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.objective == 'fit':
        fit_codebook(args, device)
    else:
        if not args.codebook or not args.template_checkpoint:
            p.error('Pretraining requires --codebook and --template-checkpoint')
        train_pretraining(args, device)


if __name__ == '__main__':
    main()
