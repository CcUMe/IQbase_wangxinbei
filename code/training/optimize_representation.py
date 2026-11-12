"""Matched CE versus cross-record SupCon optimization from one shared RF checkpoint."""
import argparse
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.rf_model import RFModel
from training.train import build_optimizer
from utils.augmentation import augment_supervised_iq
from utils.common import model_bundle, runtime, seed_all, save_checkpoint, write_json, log_json
from utils.config import load_config, training_options, make_scheduler
from utils.contrastive import load_with_training_aux
from utils.dataset import BalancedBatchSampler, PreparedDataset, collate, to_device
from utils.engine import autocast, evaluate, supervised_loss
from utils.head_optimizer import active_heads, clear_inactive_head_gradients
from utils.readable_log import ReadableRunLog
from utils.training_display import EpochProgress, active_loss_parts, epoch_summary, classification_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('code/configs/supervised/main.json'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--contrastive-weight', type=float, required=True)
    p.add_argument('--temperature', type=float, default=.1)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--epoch-steps', type=int, default=1500)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    args = p.parse_args()
    if min(args.steps, args.epoch_steps, args.batch_size) < 1 or args.steps % args.epoch_steps:
        p.error('Positive budgets required; steps must be divisible by epoch-steps')
    if args.contrastive_weight < 0 or args.temperature <= 0:
        p.error('Invalid contrastive weight or temperature')
    args.output.mkdir(parents=True, exist_ok=False)
    device, _, world = runtime('auto', 4)
    if world != 1:
        raise ValueError('Use one independent GPU per matched trial')
    seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    train, val = PreparedDataset(args.data), PreparedDataset(args.data, 'validation')
    if train.info['fingerprint'] != source['dataset_fingerprint']:
        raise ValueError('Dataset fingerprint mismatch')
    config = load_config(args.config)
    bundle = model_bundle(source)
    # Keep the source signal/encoder semantics; change only the loss between arms.
    # Older checkpoints predate these fields. Do not inherit newly enabled
    # main.json features when reproducing a trial from an older source model.
    source_model = {'classification_pooling': 'mean', 'physical_statistics': False,
                    'classification_fp32': False, 'local_refinement': False,
                    'contrastive_projection_dim': 128, **bundle['model_config']}
    for group in config['model'].values():
        for key in group:
            if key in source_model:
                group[key] = source_model[key]
    config['loss']['supervised']['contrastive'].update(weight=args.contrastive_weight, temperature=args.temperature)
    config['optimizer'].update(learning_rate=args.learning_rate, skip_inactive_heads=True)
    config['training'].update(stage='supervised', epochs=args.steps // args.epoch_steps, max_steps=0, gradient_accumulation_steps=1)
    config['data'].update(batch_size=args.batch_size, steps_per_epoch=args.epoch_steps, merge_paired=False)
    config['data']['sampling'].update(mode='balanced', task_probabilities={'modulation': 1., 'protocol': 1., 'parameter': 1.})
    config['scheduler'].update(name='cosine', warmup_steps=min(200, args.steps // 10), warmup_ratio=0., min_lr_ratio=.05)
    config['runtime'].update(seed=args.seed, precision='bf16', log_every=50)
    config['evaluation'].update(max_batches=0, batch_size=256, precision='fp32', selection_metric='modulation_accuracy')
    options = training_options(config)
    bundle['model_config'] = {**options['model'], 'pretrain_heads': False, 'contrastive_head': True}
    model = RFModel(**bundle).to(device)
    load_with_training_aux(model, source['model'], source['model_config'])
    state_hash = hashlib.sha256()
    for name, value in model.state_dict().items():
        state_hash.update(name.encode())
        state_hash.update(value.detach().cpu().numpy().tobytes())
    optimizer = build_optimizer(model, config['optimizer'])
    scheduler, warmup = make_scheduler(optimizer, config['scheduler'], args.steps)
    sampler = BalancedBatchSampler(train, args.batch_size, args.epoch_steps, args.seed)
    loader = DataLoader(train, batch_sampler=sampler, collate_fn=collate, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val, batch_size=256, collate_fn=collate, num_workers=2, pin_memory=True)
    write_json(args.output / 'resolved_config.json', config)
    protocol = {'source_checkpoint': str(args.checkpoint.resolve()), 'source_steps': source['global_step'],
        'contrastive_weight': args.contrastive_weight, 'temperature': args.temperature, 'initial_state_sha256': state_hash.hexdigest(),
        'dataset_fingerprint': train.info['fingerprint'], 'steps': args.steps, 'batch_size': args.batch_size,
        'seed': args.seed, 'projection_in_both_arms': True, 'sample_stream_sha256': None,
        'augmentation': config['data']['augmentation'], 'encoder': config['model']['encoder']}
    write_json(args.output / 'protocol.json', protocol)
    sample_hash = hashlib.sha256()
    history, best_mod, best_overall = [], -1., -float('inf')
    started, global_step = time.monotonic(), 0
    with ReadableRunLog(args.output):
        print(f'训练总览 | 表征优化 | SupCon={args.contrastive_weight:g} | temperature={args.temperature:g}')
        print(f'模型 | {model.parameter_report()} | source={args.checkpoint}')
        print(f'预算 | {config["training"]["epochs"]} epochs × {args.epoch_steps} steps | batch={args.batch_size} | cosine warmup={warmup}')
        print(f'增强 | {config["data"]["augmentation"]} | 初始权重 SHA256={state_hash.hexdigest()}')
        print('验证 | 全部 118,182 条；最佳调制与最佳综合分检查点分别保存。')

        def assess(epoch, training=None):
            nonlocal best_mod, best_overall
            metrics = evaluate(model, val_loader, device, bundle['target_scaler'], bundle['schema'], 'fp32')
            score = metrics['classification']['modulation']['accuracy']
            improved = score > best_mod
            overall_improved = metrics['proxy_score'] > best_overall
            best_mod, best_overall = max(best_mod, score), max(best_overall, metrics['proxy_score'])
            payload = dict(model=model.state_dict(), model_config=model.config, schema=bundle['schema'],
                target_scaler=bundle['target_scaler'], parameter_grids=bundle['parameter_grids'],
                dataset_fingerprint=train.info['fingerprint'], stage='supervised', training_config=config,
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), grad_scaler={}, codebook_signature=None,
                world_size=1, epoch=max(0, epoch), batch_index=args.epoch_steps - 1 if epoch >= 0 else -1, epoch_complete=epoch >= 0,
                global_step=global_step, best_score=best_mod, metrics=metrics,
                initialization_source_steps=source['global_step'] + source.get('repair_source_steps', source.get('initialization_source_steps', 0)))
            save_checkpoint(args.output / 'last.pt', payload)
            if improved:
                save_checkpoint(args.output / 'best.pt', payload)
                write_json(args.output / 'best_metrics.json', metrics)
            if overall_improved:
                save_checkpoint(args.output / 'best_overall.pt', payload)
            write_json(args.output / 'metrics.json', metrics)
            write_json(args.output / f'metrics_step_{global_step:06d}.json', metrics)
            record = {'event': 'validation', 'epoch': epoch + 1, 'step': global_step, 'metrics': metrics,
                      'sample_stream_sha256': sample_hash.hexdigest(), 'training': training}
            history.append(record)
            write_json(args.output / 'history.json', history)
            log_json(args.output / 'events.jsonl', record, console=False)
            if training:
                epoch_summary(epoch, config['training']['epochs'], training, metrics, True, improved,
                              ['last.pt'] + (['best.pt'] if improved else []), True)
            else:
                print('起点验证：')
                classification_summary(metrics)
        assess(-1)
        for epoch in range(config['training']['epochs']):
            sampler.set_epoch(epoch)
            model.train()
            coverage, positives = [], []
            progress = EpochProgress(epoch, config['training']['epochs'], args.epoch_steps, min_interval=1.)
            with progress:
                for cpu in loader:
                    sample_hash.update(np.asarray(cpu['rows'], dtype='<i8').tobytes())
                    active = active_heads(cpu, options)
                    batch = to_device(cpu, device)
                    torch.manual_seed(args.seed + global_step)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(device, 'bf16'):
                        output = model(augment_supervised_iq(batch, options['augmentation']), batch['lengths'])
                        loss, parts = supervised_loss(output, batch, bundle['target_scaler'], bundle['parameter_grids'], options)
                    if parts.get('supcon_coverage', 0) > 0:
                        active.add('aux.contrastive.')
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f'Nonfinite loss at step {global_step}')
                    loss.backward()
                    clear_inactive_head_gradients(model, active, device)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    rates = {g['name']: g['lr'] for g in optimizer.param_groups}
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                    progress.observe(float(loss.detach()), parts, active_loss_parts(cpu, options))
                    progress.advance(rates, float(norm))
                    if bool((cpu['modulation'] >= 0).any()) and 'supcon_coverage' in parts:
                        coverage.append(parts['supcon_coverage'])
                        positives.append(parts['supcon_positives'])
                    if global_step % 50 == 0:
                        log_json(args.output / 'events.jsonl', {'event': 'step', 'step': global_step,
                            'parts': parts, 'loss': float(loss.detach()), 'learning_rates': rates, 'gradient_norm': float(norm)}, console=False)
            training = progress.summary()
            if coverage:
                training['supcon_coverage'] = float(np.mean(coverage))
                training['supcon_positives'] = float(np.mean(positives))
                print(f'跨记录正样本 | 有效锚点比例={np.mean(coverage):.2%} | 每锚点正样本数={np.mean(positives):.2f}')
            print(f'全量验证中 | Epoch {epoch + 1} ...', flush=True)
            assess(epoch, training)
        protocol['sample_stream_sha256'] = sample_hash.hexdigest()
        protocol['elapsed_seconds'] = time.monotonic() - started
        write_json(args.output / 'protocol.json', protocol)
        print(f'优化实验结束 | {protocol["elapsed_seconds"]:.1f}s | 样本流 SHA256={sample_hash.hexdigest()}', flush=True)


if __name__ == '__main__':
    main()
