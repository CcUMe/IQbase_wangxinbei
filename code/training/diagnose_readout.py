"""Freeze an existing RF encoder; compare fresh linear and original-shape MLP readouts."""
import argparse
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.rf_model import RFModel
from utils.common import model_bundle, runtime, seed_all, write_json, read_json, save_checkpoint, log_json
from utils.config import make_scheduler
from utils.dataset import PreparedDataset, collate, to_device
from utils.metrics import classification_metrics
from utils.readable_log import ReadableRunLog
from utils.training_display import EpochProgress, classification_summary
from utils.signal_statistics import signal_statistics, STATISTICS_DIM


@torch.inference_mode()
def feature_metrics(head, features, labels, names, batch_size=4096, input_width=None):
    matrix = torch.zeros(len(names) ** 2, dtype=torch.long, device=features.device)
    ce, confidence = 0., 0.
    predictions = []
    head.eval()
    for begin in range(0, len(labels), batch_size):
        target = labels[begin:begin + batch_size]
        inputs = features[begin:begin + batch_size]
        if input_width is not None:
            inputs = inputs[:, :input_width]
        logits = head(inputs).float()
        prediction = logits.argmax(-1)
        matrix += torch.bincount(target * len(names) + prediction, minlength=len(names) ** 2)
        ce += float(F.cross_entropy(logits, target, reduction='sum'))
        confidence += float(logits.softmax(-1).max(-1).values.sum())
        predictions.append(prediction.cpu().numpy())
    result = classification_metrics(matrix.reshape(len(names), len(names)).cpu().numpy(), names)
    result.update(cross_entropy=ce / len(labels), mean_confidence=confidence / len(labels))
    return result, np.concatenate(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--feature-batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027])
    parser.add_argument('--pooling', choices=['mean', 'mean_std'], default='mean')
    parser.add_argument('--physical-statistics', action='store_true')
    parser.add_argument('--reuse-feature-cache', type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.feature_batch_size) < 1:
        parser.error('Epochs and batch sizes must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    device, _, world = runtime(args.device, 4)
    if world != 1:
        raise ValueError('Frozen readout diagnostic is single-process')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    bundle = model_bundle(source)
    model = RFModel(**bundle).to(device).eval().requires_grad_(False)
    model.load_state_dict(source['model'], strict=True)
    names = source['schema']['classification_labels']['modulation']
    started = time.monotonic()
    cache = {}
    reference = {}
    protocol = {'checkpoint': str(args.checkpoint.resolve()), 'dataset_fingerprint': source['dataset_fingerprint'],
        'pooling': args.pooling,
        'physical_statistics': args.physical_statistics,
        'reused_feature_cache': str(args.reuse_feature_cache.resolve()) if args.reuse_feature_cache else None,
        'feature_precision': 'fp32', 'encoder_frozen': True, 'epochs': args.epochs, 'seeds': args.seeds,
        'device': str(device),
        'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
        'batch_size': args.batch_size, 'feature_normalization': 'train-only mean/std, shared by both readout types',
        'selection': 'fixed final epoch; validation does not select hyperparameters or stopping time',
        'heads': ['linear', 'MLP with same hidden width/activation as original'],
        'note': 'Reference original head uses raw features. Both fresh heads use identically standardized features.'}
    write_json(args.output / 'protocol.json', protocol)
    with ReadableRunLog(args.output):
        print(f'训练总览 | 诊断二 | 冻结基座读出对照 | source={args.checkpoint}', flush=True)
        print(f'方案 | FP32 特征缓存 | Linear 与 MLP 均训练 {args.epochs} epochs | seeds={args.seeds}')
        print('比较口径 | 两个新头使用相同训练侧标准化、批次顺序、优化器和预算；最后一轮为主结果。')
        for split in ['train', 'validation']:
            dataset = PreparedDataset(args.data, split)
            if dataset.info['fingerprint'] != source['dataset_fingerprint']:
                raise ValueError('Dataset fingerprint mismatch')
            rows = dataset.indices
            rows = rows[dataset.meta['modulation'][rows] >= 0]
            dataset.indices = rows[np.argsort(dataset.meta['length'][rows], kind='stable')]
            labels_np = dataset.meta['modulation'][dataset.indices].astype(np.int64)
            np.save(args.output / f'{split}_rows.npy', dataset.indices)
            np.save(args.output / f'{split}_labels.npy', labels_np)
            width = model.config['width'] * (2 if args.pooling == 'mean_std' else 1)
            width += STATISTICS_DIM if args.physical_statistics else 0
            cached = None
            if args.reuse_feature_cache:
                previous = read_json(args.reuse_feature_cache / 'protocol.json')
                if (Path(previous['checkpoint']).resolve() != args.checkpoint.resolve() or
                    previous['dataset_fingerprint'] != source['dataset_fingerprint'] or
                    previous.get('pooling', 'mean') != args.pooling or previous.get('physical_statistics', False)):
                    raise ValueError('Cached features do not match this base and pooling')
                if not np.array_equal(np.load(args.reuse_feature_cache / f'{split}_rows.npy'), dataset.indices):
                    raise ValueError('Cached feature row order differs')
                cached = np.load(args.reuse_feature_cache / f'{split}_features.npy', mmap_mode='r')
            path = args.output / f'{split}_features.npy'
            features = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(dataset), width))
            loader = DataLoader(dataset, batch_size=args.feature_batch_size, collate_fn=collate, num_workers=2, pin_memory=True)
            offset = 0
            with torch.inference_mode():
                for index, cpu in enumerate(loader):
                    batch = to_device(cpu, device)
                    size = len(batch['iq'])
                    if cached is not None:
                        pooled = torch.from_numpy(np.array(cached[offset:offset + size])).to(device)
                    else:
                        h, valid, _ = model.encode(batch['iq'], batch['lengths'])
                        pooled = h.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
                        if args.pooling == 'mean_std':
                            variance = ((h - pooled[:, None]).square() * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
                            pooled = torch.cat((pooled, variance.clamp_min(1e-8).sqrt()), -1)
                    if args.physical_statistics:
                        student_iq, _, _, _ = model.backbone.preprocess(batch['iq'], batch['lengths'])
                        pooled = torch.cat((pooled, signal_statistics(student_iq, batch['lengths'])), -1)
                    features[offset:offset + size] = pooled.cpu().numpy()
                    offset += size
                    if index % 100 == 0 or offset == len(dataset):
                        print(f'特征提取 | {split} | {offset:,}/{len(dataset):,}', flush=True)
            features.flush()
            x = torch.from_numpy(np.array(features)).to(device)
            y = torch.from_numpy(labels_np).to(device)
            cache[split] = (x, y)
            reference[split], predictions = feature_metrics(model.heads['modulation'], x, y, names,
                                                            input_width=model.heads['modulation'][0].in_features)
            np.save(args.output / f'reference_{split}_predictions.npy', predictions)
        # Confirm the frozen encoder remained bitwise unchanged, not just without optimizer membership.
        for name, value in model.backbone.state_dict().items():
            if not torch.equal(value.cpu(), source['model']['backbone.' + name]):
                raise AssertionError(f'Frozen encoder changed: {name}')
        protocol['backbone_verified_unchanged'] = True
        write_json(args.output / 'reference.json', reference)
        print('原检查点分类头：', flush=True)
        classification_summary({'classification': {'modulation': reference['validation']}})
        hidden = model.heads['modulation'][0].out_features
        del model
        train_x, train_y = cache['train']
        val_x, val_y = cache['validation']
        mean = train_x.mean(0)
        std = train_x.std(0, unbiased=False).clamp_min(1e-4)
        train_x = (train_x - mean) / std
        val_x = (val_x - mean) / std
        total_steps = math.ceil(len(train_y) / args.batch_size) * args.epochs
        results = []
        for kind in ['linear', 'mlp']:
            for seed in args.seeds:
                seed_all(seed)
                head = (nn.Linear(width, len(names)) if kind == 'linear' else nn.Sequential(
                    nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, len(names)))).to(device)
                optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                scheduler, _ = make_scheduler(optimizer, dict(name='cosine', warmup_steps=0, warmup_ratio=0.,
                    warmup_start_factor=0., min_lr_ratio=.05), total_steps)
                case = args.output / f'{kind}_seed{seed}'
                case.mkdir()
                history, global_step = [], 0
                print(f'\n分类头训练 | {kind} | seed={seed} | 参数={sum(p.numel() for p in head.parameters()):,}', flush=True)
                for epoch in range(args.epochs):
                    head.train()
                    generator = torch.Generator(device=device).manual_seed(seed + epoch * 100003)
                    order = torch.randperm(len(train_y), device=device, generator=generator)
                    steps = math.ceil(len(order) / args.batch_size)
                    progress = EpochProgress(epoch, args.epochs, steps, min_interval=1.)
                    with progress:
                        for begin in range(0, len(order), args.batch_size):
                            index = order[begin:begin + args.batch_size]
                            optimizer.zero_grad(set_to_none=True)
                            logits = head(train_x[index])
                            loss = F.cross_entropy(logits, train_y[index])
                            if not bool(torch.isfinite(loss)):
                                raise FloatingPointError('Nonfinite readout loss')
                            loss.backward()
                            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                            rate = optimizer.param_groups[0]['lr']
                            optimizer.step()
                            scheduler.step()
                            global_step += 1
                            progress.observe(float(loss.detach()), {'modulation': float(loss.detach())}, {'modulation'})
                            progress.advance({'heads': rate}, float(norm))
                            if global_step % 50 == 0:
                                log_json(args.output / 'events.jsonl', {'event': 'step', 'head': kind, 'seed': seed,
                                    'epoch': epoch + 1, 'step': global_step, 'loss': float(loss.detach()), 'learning_rate': rate}, console=False)
                    print(f'Epoch {epoch + 1:02d}/{args.epochs:02d} | {kind} seed={seed} | train_CE={progress.summary()["loss"]:.5f}', flush=True)
                    if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
                        train_metrics, _ = feature_metrics(head, train_x, train_y, names)
                        val_metrics, predictions = feature_metrics(head, val_x, val_y, names)
                        print(f'训练集 | Accuracy={train_metrics["accuracy"]:.2%} | CE={train_metrics["cross_entropy"]:.5f}', flush=True)
                        classification_summary({'classification': {'modulation': val_metrics}})
                        record = {'epoch': epoch + 1, 'train': train_metrics, 'validation': val_metrics}
                        history.append(record)
                        log_json(args.output / 'events.jsonl', {'event': 'validation', 'head': kind, 'seed': seed, **record}, console=False)
                        write_json(case / 'history.json', history)
                save_checkpoint(case / 'head.pt', {'head': head.state_dict(), 'head_type': kind, 'feature_mean': mean,
                    'feature_std': std, 'source_checkpoint': str(args.checkpoint.resolve()), 'stage': 'frozen_readout_diagnostic',
                    'dataset_fingerprint': source['dataset_fingerprint'], 'seed': seed, 'epochs': args.epochs})
                np.save(case / 'validation_predictions.npy', predictions)
                result = {'head': kind, 'seed': seed, 'train': train_metrics, 'validation': val_metrics}
                results.append(result)
                write_json(case / 'metrics.json', result)
        protocol['elapsed_seconds'] = time.monotonic() - started
        write_json(args.output / 'protocol.json', protocol)
        write_json(args.output / 'summary.json', {'protocol': protocol, 'reference': reference, 'results': results})
        print(f'\n冻结基座诊断结束 | {protocol["elapsed_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
