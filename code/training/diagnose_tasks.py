"""Matched modulation exposure: fresh shared model, joint versus modulation-only updates."""
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
from utils.common import model_bundle, log_json, runtime, seed_all, save_checkpoint, write_json, read_json
from utils.config import resolve_config, training_options, make_scheduler
from utils.dataset import BalancedBatchSampler, PreparedDataset, collate, to_device
from utils.engine import autocast, evaluate, supervised_loss
from utils.head_optimizer import active_heads, clear_inactive_head_gradients
from utils.readable_log import ReadableRunLog
from utils.training_display import EpochProgress, active_loss_parts, epoch_summary


class MatchedSampler:
    def __init__(self, dataset, cycles, batch_size, seed, mode, start_cycle=0):
        self.base = BalancedBatchSampler(dataset, batch_size, cycles * 3, seed=seed)
        if list(self.base.groups) != [0, 1, 2]:
            raise ValueError('This diagnostic requires all three task groups in their original order')
        self.mode, self.cycles = mode, cycles
        self.start_cycle = start_cycle

    def __len__(self):
        return (self.cycles - self.start_cycle) * (3 if self.mode == 'joint' else 1)

    def __iter__(self):
        for index, batch in enumerate(self.base):
            if index < self.start_cycle * 3:
                continue
            if self.mode == 'joint' or index % 3 == 0:
                yield batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--template-checkpoint', type=Path, required=True, help='Architecture/schema only; weights are NOT loaded')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['joint', 'modulation_only'], required=True)
    p.add_argument('--cycles', type=int, default=4800)
    p.add_argument('--epoch-cycles', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--contrastive-weight', type=float, default=0.)
    p.add_argument('--local-refinement', action='store_true')
    p.add_argument('--drop-scale-embedding', action='store_true')
    p.add_argument('--drop-global-spectrum', action='store_true')
    p.add_argument('--multiscale-frontend', action='store_true')
    p.add_argument('--pretrained-backbone', type=Path, help='Optional training-only pretrained backbone; task heads remain fresh')
    p.add_argument('--resume', type=Path, help='Resume a checkpoint containing optimizer and cycle-scheduler state')
    p.add_argument('--stop-after-epochs', type=int, default=0, help='Execution check: stop at an epoch boundary without changing the planned schedule')
    args = p.parse_args()
    if args.cycles < 1 or args.epoch_cycles < 1 or args.batch_size < 1:
        p.error('cycles, epoch-cycles and batch-size must be positive')
    args.output.mkdir(parents=True, exist_ok=bool(args.resume))
    device, _, world = runtime('auto', 4)
    if world != 1:
        raise ValueError('Use one independent GPU per diagnostic arm')
    seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = torch.load(args.template_checkpoint, map_location='cpu', weights_only=True, mmap=True)
    train, val = PreparedDataset(args.data), PreparedDataset(args.data, 'validation')
    if train.info['fingerprint'] != source['dataset_fingerprint']:
        raise ValueError('Dataset fingerprint mismatch')
    config = resolve_config(source['training_config'])
    config['optimizer'].update(learning_rate=args.learning_rate, skip_inactive_heads=True)
    config['model']['encoder']['qk_normalization'] = True
    config['data']['augmentation'] = {'phase_rotation_probability': 0., 'modulation_only': {
        'phase_rotation_probability': 0., 'circular_shift_probability': 0.}}
    config['loss']['supervised']['label_smoothing'] = 0.
    config['loss']['supervised']['contrastive']['weight'] = args.contrastive_weight
    config['model']['signal']['local_refinement'] = args.local_refinement
    config['model']['signal'].update(scale_embedding=not args.drop_scale_embedding,
        global_spectrum_input=not args.drop_global_spectrum, multiscale_frontend=args.multiscale_frontend)
    config['runtime']['precision'] = 'bf16'
    options = training_options(config)
    bundle = model_bundle(source)
    bundle['model_config'] = {**bundle['model_config'], 'qk_normalization': True,
        'contrastive_head': args.contrastive_weight > 0, 'local_refinement': args.local_refinement,
        'scale_embedding': not args.drop_scale_embedding, 'global_spectrum_input': not args.drop_global_spectrum,
        'multiscale_frontend': args.multiscale_frontend, 'pretrain_heads': False, 'atomic_vocab': 0, 'emergent_vocab': 0}
    model = RFModel(**bundle).to(device)
    initial = hashlib.sha256()
    reference_initial = hashlib.sha256()
    for name, value in model.state_dict().items():
        initial.update(name.encode())
        initial.update(value.detach().cpu().numpy().tobytes())
        if not name.startswith(('aux.contrastive.', 'backbone.time.refinement.', 'backbone.time.multiscale.')):
            reference_initial.update(name.encode())
            reference_initial.update(value.detach().cpu().numpy().tobytes())
    random_initial_hash = initial.hexdigest()
    pretraining_steps = 0
    if args.pretrained_backbone:
        pretrained = torch.load(args.pretrained_backbone, map_location='cpu', weights_only=True, mmap=True)
        if pretrained['dataset_fingerprint'] != train.info['fingerprint']:
            raise ValueError('Pretraining split mismatch')
        for key in ['width', 'depth', 'heads', 'patch_size', 'stft_window']:
            if model.config[key] != pretrained['model_config'][key]:
                raise ValueError(f'Pretrained backbone mismatch: {key}')
        shared = {k.removeprefix('backbone.'): v for k, v in pretrained['model'].items() if k.startswith('backbone.')}
        model.backbone.load_state_dict(shared, strict=True)
        pretraining_steps = pretrained['global_step']
        initial = hashlib.sha256()
        for name, value in model.state_dict().items():
            initial.update(name.encode())
            initial.update(value.detach().cpu().numpy().tobytes())
    optimizer = build_optimizer(model, config['optimizer'])
    warmup = min(167, max(0, args.cycles // 10))
    schedule_options = dict(name='cosine', warmup_steps=warmup, warmup_ratio=0., warmup_start_factor=0., min_lr_ratio=.05)
    scheduler, _ = make_scheduler(optimizer, schedule_options, args.cycles)
    protocol = {'mode': args.mode, 'initialization': 'random; template weights not loaded',
        'initial_state_sha256': initial.hexdigest(), 'dataset_fingerprint': train.info['fingerprint'],
        'reference_initial_state_sha256': reference_initial.hexdigest(),
        'contrastive_weight': args.contrastive_weight, 'local_refinement': args.local_refinement,
        'scale_embedding': not args.drop_scale_embedding, 'global_spectrum_input': not args.drop_global_spectrum,
        'multiscale_frontend': args.multiscale_frontend, 'random_initial_state_sha256': random_initial_hash,
        'pretrained_backbone': str(args.pretrained_backbone.resolve()) if args.pretrained_backbone else None,
        'pretraining_steps': pretraining_steps,
        'batch_size': args.batch_size, 'modulation_cycles': args.cycles,
        'modulation_exposures': args.cycles * args.batch_size, 'seed': args.seed,
        'learning_rate': args.learning_rate, 'scheduler_unit': 'modulation cycle', 'scheduler': schedule_options,
        'schedule_note': 'Every cycle has one identical modulation batch; joint adds protocol and parameter updates at the same cycle LR.',
        'optimizer_note': 'Joint has three backbone AdamW updates per cycle; single has one. This tests the net effect of auxiliary-task updates, not isolated gradient conflict.',
        'template_checkpoint': str(args.template_checkpoint.resolve()), 'model_config': model.config,
        'modulation_batch_stream_sha256': None, 'training_config_reference': config}
    start_epoch, start_cycle = 0, 0
    history, best = [], -1.
    stream_hash = hashlib.sha256()
    if args.resume:
        previous_protocol = read_json(args.output / 'protocol.json')
        for key in protocol:
            if key != 'modulation_batch_stream_sha256' and previous_protocol[key] != protocol[key]:
                raise ValueError(f'Resume protocol changed: {key}')
        previous = torch.load(args.resume, map_location='cpu', weights_only=True, mmap=True)
        if 'optimizer' not in previous or 'scheduler' not in previous:
            raise ValueError('Older diagnostic weights have no optimizer state; restart the controlled trial')
        model.load_state_dict(previous['model'], strict=True)
        optimizer.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler'])
        start_epoch, start_cycle = previous['epoch'] + 1, previous['modulation_updates']
        history = previous['history'] if 'history' in previous else read_json(args.output / 'history.json')[:start_epoch]
        best = previous['best_score']
        for index, positions in enumerate(MatchedSampler(train, args.cycles, args.batch_size, args.seed, args.mode)):
            if index >= start_cycle * (3 if args.mode == 'joint' else 1):
                break
            if args.mode != 'joint' or index % 3 == 0:
                stream_hash.update(np.asarray(train.indices[positions], dtype='<i8').tobytes())
        if stream_hash.hexdigest() != previous['modulation_batch_stream_sha256']:
            raise ValueError('Resume sample stream differs')
    write_json(args.output / 'protocol.json', protocol)
    train_loader = DataLoader(train, batch_sampler=MatchedSampler(train, args.cycles, args.batch_size, args.seed, args.mode, start_cycle),
                              collate_fn=collate, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val, batch_size=256, collate_fn=collate, num_workers=2, pin_memory=True)
    iterator = iter(train_loader)
    epochs = math.ceil(args.cycles / args.epoch_cycles)
    slots = 3 if args.mode == 'joint' else 1
    started = time.monotonic()
    with ReadableRunLog(args.output):
        print(f'训练总览 | 诊断一 | {args.mode} | 随机初始化 | 参数={model.parameter_report()["final_parameters"]:,}')
        print(f'数据 | 训练={len(train):,} 验证={len(val):,} | 每批={args.batch_size} | 调制更新={args.cycles:,}')
        print(f'预算 | {epochs} epochs | joint 每调制更新附加两次其他任务更新；调制批次及其 LR、Dropout seed 匹配')
        print(f'初始化 SHA256 | {initial.hexdigest()}\n输出 | {args.output}')
        if args.mode == 'modulation_only':
            print('诊断口径 | 协议及参数头没有训练，其指标只作记录，不作为完整比赛模型表现。')
        for epoch in range(start_epoch, epochs):
            model.train()
            begin = epoch * args.epoch_cycles
            end = min(args.cycles, begin + args.epoch_cycles)
            progress = EpochProgress(epoch, epochs, (end - begin) * slots, min_interval=1.)
            with progress:
                for cycle in range(begin, end):
                    for slot in range(slots):
                        cpu = next(iterator)
                        if slot == 0:
                            if not bool((cpu['modulation'] >= 0).all()):
                                raise ValueError('Common batch is not modulation-only')
                            stream_hash.update(np.asarray(cpu['rows'], dtype='<i8').tobytes())
                        active = active_heads(cpu, options)
                        batch = to_device(cpu, device)
                        torch.manual_seed(args.seed + cycle * 3 + slot)
                        optimizer.zero_grad(set_to_none=True)
                        with autocast(device, 'bf16'):
                            output = model(batch['iq'], batch['lengths'])
                            loss, parts = supervised_loss(output, batch, bundle['target_scaler'], bundle['parameter_grids'], options)
                        if parts.get('supcon_coverage', 0) > 0:
                            active.add('aux.contrastive.')
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError(f'Nonfinite loss at cycle {cycle}, slot {slot}')
                        loss.backward()
                        clear_inactive_head_gradients(model, active, device)
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        rates = {g['name']: g['lr'] for g in optimizer.param_groups}
                        optimizer.step()
                        progress.observe(float(loss.detach()), parts, active_loss_parts(cpu, options))
                        progress.advance(rates, float(norm))
                        if cycle % 50 == 0:
                            log_json(args.output / 'events.jsonl', {'event': 'step', 'cycle': cycle + 1, 'slot': slot,
                                'loss': float(loss.detach()), 'parts': parts, 'gradient_norm': float(norm), 'learning_rates': rates}, console=False)
                    scheduler.step()
            print(f'验证中 | Epoch {epoch + 1}/{epochs} | 全部 {len(val):,} 条 ...', flush=True)
            metrics = evaluate(model, val_loader, device, bundle['target_scaler'], bundle['schema'], 'fp32')
            metrics['diagnostic_context'] = {'mode': args.mode, 'auxiliary_heads_trained': args.mode == 'joint'}
            score = metrics['classification']['modulation']['accuracy']
            improved = score > best
            best = max(best, score)
            updates = end * slots
            payload = {'model': model.state_dict(), **{k: bundle[k] for k in ['schema', 'target_scaler', 'parameter_grids']},
                'model_config': model.config, 'dataset_fingerprint': train.info['fingerprint'],
                'stage': 'supervised' if args.mode == 'joint' else 'diagnostic',
                'training_config': config, 'epoch': epoch, 'global_step': updates, 'modulation_updates': end,
                'metrics': metrics, 'diagnostic_mode': args.mode}
            payload['initialization_source_steps'] = pretraining_steps
            payload.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), best_score=best,
                modulation_batch_stream_sha256=stream_hash.hexdigest(), history=history + [{
                    'event': 'epoch', 'epoch': epoch + 1, 'modulation_updates': end,
                    'optimizer_updates': updates, 'modulation_exposures': end * args.batch_size,
                    'training': progress.summary(), 'metrics': metrics,
                    'modulation_batch_stream_sha256': stream_hash.hexdigest()}])
            save_checkpoint(args.output / 'last.pt', payload)
            if improved:
                save_checkpoint(args.output / 'best.pt', payload)
            write_json(args.output / 'metrics.json', metrics)
            write_json(args.output / f'metrics_epoch_{epoch + 1:02d}.json', metrics)
            training = progress.summary()
            epoch_summary(epoch, epochs, training, metrics, True, improved, ['last.pt'] + (['best.pt'] if improved else []), True)
            row = {'event': 'epoch', 'epoch': epoch + 1, 'modulation_updates': end, 'optimizer_updates': updates,
                   'modulation_exposures': end * args.batch_size, 'training': training, 'metrics': metrics,
                   'modulation_batch_stream_sha256': stream_hash.hexdigest()}
            history.append(row)
            log_json(args.output / 'events.jsonl', row, console=False)
            write_json(args.output / 'history.json', history)
            if args.stop_after_epochs and epoch + 1 >= args.stop_after_epochs:
                break
        protocol['modulation_batch_stream_sha256'] = stream_hash.hexdigest()
        protocol['elapsed_seconds'] = time.monotonic() - started
        write_json(args.output / 'protocol.json', protocol)
        print(f'诊断训练结束 | {protocol["elapsed_seconds"]:.1f}s | 调制流 SHA256={stream_hash.hexdigest()}', flush=True)


if __name__ == '__main__':
    main()
