"""Supervised or self-supervised training, single GPU / torchrun DDP."""
import argparse
import math
import hashlib
import json
import inspect
import shlex
from pathlib import Path
import sys
import time
from contextlib import nullcontext

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from models.rf_model import RFModel
from utils.augmentation import augment_supervised_iq
from utils.common import load_checkpoint, log_json, read_json, runtime, save_checkpoint, seed_all, write_json
from utils.dataset import BalancedBatchSampler, PreparedDataset, collate, to_device
from utils.engine import autocast, evaluate, evaluate_pretraining, patch_mask, pretraining_loss, supervised_loss
from utils.config import load_config, make_scheduler, resolve_config, resume_comparison_config, training_options, validate_config
from utils.training_display import EpochProgress, active_loss_parts, epoch_summary, training_overview, classification_summary
from utils.run_directory import training_run_directory
from utils.head_optimizer import active_heads, clear_inactive_head_gradients
from utils.readable_log import ReadableRunLog
from utils.contrastive import load_with_training_aux
from utils.command_config import checkpoint_reference, configured_launch
from utils.position import POSITION_DEFAULTS
from utils.finetuning import FineTuneController


def build_optimizer(model, settings, finetuning=None):
    groups = {}
    layerwise = finetuning and (finetuning['layer_lr_decay'] != 1 or finetuning['frontend_lr_factor'] != 1)
    depth = len(model.backbone.blocks) if layerwise else 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        backbone = name.startswith('backbone.')
        no_decay = settings['exclude_bias_and_norm_from_decay'] and (parameter.ndim < 2 or name.endswith('.bias'))
        key = ('backbone' if backbone else 'heads') + ('_no_decay' if no_decay else '_decay')
        multiplier = settings['backbone_lr_factor'] if backbone else 1.
        if layerwise and backbone:
            if name.startswith('backbone.blocks.'):
                index = int(name.split('.')[2])
                multiplier *= finetuning['layer_lr_decay'] ** (depth - 1 - index)
                key = f'backbone_layer_{index}_' + ('no_decay' if no_decay else 'decay')
            elif name.startswith('backbone.norm.'):
                key = 'backbone_norm_' + ('no_decay' if no_decay else 'decay')
            else:
                multiplier *= finetuning['layer_lr_decay'] ** depth * finetuning['frontend_lr_factor']
                key = 'backbone_frontend_' + ('no_decay' if no_decay else 'decay')
        if key not in groups:
            groups[key] = {'params': [], 'name': key,
                           'lr': settings['learning_rate'] * multiplier,
                           'weight_decay': 0. if no_decay else settings['weight_decay']}
        groups[key]['params'].append(parameter)
    return torch.optim.AdamW(list(groups.values()), betas=tuple(settings['betas']), eps=settings['eps'])


def loader_options(settings, device):
    result = dict(num_workers=settings['num_workers'], pin_memory=settings['pin_memory'] and device.type == 'cuda')
    if settings['num_workers']:
        result.update(persistent_workers=settings['persistent_workers'], prefetch_factor=settings['prefetch_factor'])
    return result


def selection_score(metrics, name, stage):
    name = ('pretraining_loss' if stage == 'pretrain' else 'proxy_score') if name == 'auto' else name
    if name == 'pretraining_loss':
        return -metrics['pretraining_validation_loss']
    if name.endswith('_accuracy'):
        return metrics['classification'][name.removesuffix('_accuracy')]['accuracy']
    return metrics[name]


def configure_checkpoint_start(args, settings):
    """Explicit CLI initialization takes priority over configured resume_from."""
    if not (args.resume or args.no_resume or args.init_checkpoint or args.warmstart_checkpoint):
        configured = settings['checkpoint']['resume_from']
        if configured:
            args.resume = Path(configured)
        elif settings['run']['init_checkpoint']:
            args.init_checkpoint = checkpoint_reference(settings['run']['init_checkpoint'])
        elif settings['run']['warmstart_checkpoint']:
            args.warmstart_checkpoint = checkpoint_reference(settings['run']['warmstart_checkpoint'])
    for name in ['init_checkpoint', 'warmstart_checkpoint']:
        if getattr(args, name):
            setattr(args, name, checkpoint_reference(getattr(args, name)))
    if args.resume:
        args.resume = args.resume.expanduser().resolve()
        if not args.resume.is_file():
            raise ValueError(f'Resume checkpoint does not exist: {args.resume}')
    settings['checkpoint']['resume_from'] = str(args.resume) if args.resume else ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Parent directory for timestamped runs; optional with --resume')
    parser.add_argument('--stage', choices=['supervised', 'pretrain'])
    parser.add_argument('--codebook', type=Path)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument('--init-checkpoint', type=Path, help='Load shared backbone only; overrides configured resume_from')
    initialization.add_argument('--warmstart-checkpoint', type=Path, help='Load all weights with a new optimizer; overrides configured resume_from')
    initialization.add_argument('--resume', type=Path, help='Restore full training state; overrides checkpoint.resume_from')
    initialization.add_argument('--no-resume', action='store_true', help='Ignore configured resume_from and start a new run')
    parser.add_argument('--device')
    parser.add_argument('--gpus', help='Override runtime.gpus')
    parser.add_argument('--processes', type=int, help='Override automatic process count')
    parser.add_argument('--max-steps', type=int, help='Optimizer-update cap; does not rescale the planned LR schedule')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int, help='Per-process batch size')
    parser.add_argument('--merge-paired', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    explicit_output = args.output is not None
    resolved = load_config(args.config)
    if not resolved['optimizer']['skip_inactive_heads']:
        parser.error('train.py requires optimizer.skip_inactive_heads=true to prevent inactive AdamW head updates; '
                     'false is reserved for historical diagnostic runners')
    for key in ['gpus', 'processes']:
        if getattr(args, key) is not None:
            resolved['runtime'][key] = getattr(args, key)
    for name in ['data', 'output', 'codebook']:
        if getattr(args, name) is None and resolved['run'][name]:
            setattr(args, name, Path(resolved['run'][name]).expanduser())
    if args.data is None:
        parser.error('Set run.data in the config or pass --data')
    try:
        configure_checkpoint_start(args, resolved)
    except ValueError as exc:
        parser.error(str(exc))
    if args.resume and not explicit_output:
        args.output = None  # Resume owns its original directory, not a new output default.
    if args.output is None and not args.resume:
        parser.error('--output is required unless --resume or checkpoint.resume_from is provided')
    configured_launch(resolved['runtime']['gpus'], resolved['runtime']['processes'],
                      args.device or resolved['runtime']['device'], distributed=True)
    for name in ['data', 'output', 'codebook', 'init_checkpoint', 'warmstart_checkpoint']:
        value = getattr(args, name)
        resolved['run'][name] = str(value) if value is not None else ''
    if args.stage is not None:
        resolved['training']['stage'] = args.stage
    args.stage = resolved['training']['stage']
    if args.merge_paired is not None:
        resolved['data']['merge_paired'] = args.merge_paired
    if args.epochs is not None:
        resolved['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        resolved['data']['batch_size'] = args.batch_size
    if args.device is not None:
        resolved['runtime']['device'] = args.device
    validate_config(resolved)
    config = training_options(resolved)
    cap = config['max_steps'] if args.max_steps is None else args.max_steps
    if cap < 0:
        parser.error('max-steps must be nonnegative')
    device, rank, world = runtime(config['device'], config['threads'])
    seed_all(config['seed'])
    torch.use_deterministic_algorithms(config['deterministic'])
    torch.backends.cuda.matmul.allow_tf32 = config['allow_tf32']
    torch.backends.cudnn.allow_tf32 = config['allow_tf32']
    torch.backends.cudnn.deterministic = config['deterministic']
    train_data = PreparedDataset(args.data, 'train', args.codebook, config['merge_paired'])
    val_data = PreparedDataset(args.data, 'validation', args.codebook)
    info = train_data.info
    codebook = read_json(args.codebook / 'codebook.json') if args.codebook else None
    codebook_signature = hashlib.sha256(json.dumps(codebook, sort_keys=True).encode()).hexdigest() if codebook else None
    model_config = dict(config['model'])
    model_config['pretrain_heads'] = args.stage == 'pretrain'
    model_config['contrastive_head'] = args.stage == 'supervised' and config['contrastive_weight'] > 0
    model_config['atomic_vocab'] = codebook['atomic_vocab'] if codebook else 0
    model_config['emergent_vocab'] = codebook['emergent_vocab'] if codebook else 0
    if model_config.get('patch_size', 16) != info['patch_size']:
        raise ValueError('Model/cache patch size mismatch')
    if args.stage == 'pretrain':
        active = config['reconstruction_weight'] + config['spectral_loss_weight']
        if codebook:
            active += config['atomic_weight'] + config['emergent_weight']
        if active <= 0:
            raise ValueError('No active pretraining objective; structural prediction requires --codebook')
    model = RFModel(model_config, info['schema'], info['target_scaler'], info['parameter_grids']).to(device)
    initialization_source_steps = 0
    if args.warmstart_checkpoint:
        initial = torch.load(args.warmstart_checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if initial['dataset_fingerprint'] != info['fingerprint'] or initial['stage'] != args.stage:
            raise ValueError('Full-model warmstart requires identical dataset split and training stage')
        if initial.get('codebook_signature') != codebook_signature:
            raise ValueError('Full-model warmstart requires the same vocabulary; use init-checkpoint for a new vocabulary')
        if any(initial[key] != info[key] for key in ['schema', 'target_scaler', 'parameter_grids']):
            raise ValueError('Full-model warmstart requires identical label schema, target scales and grids')
        load_with_training_aux(model, initial['model'], initial['model_config'])
        initialization_source_steps = initial['global_step'] + initial.get('repair_source_steps', initial.get('initialization_source_steps', 0))
    if args.init_checkpoint:
        initial = load_checkpoint(args.init_checkpoint)
        for key in ['patch_size', 'stft_window', 'position']:
            if initial['model_config'].get(key) != model.config.get(key):
                raise ValueError(f'Pretrained backbone config mismatch: {key}')
        previous_position = {**POSITION_DEFAULTS, **initial['model_config'].get('position_options', {})}
        if previous_position != model.config['position_options']:
            raise ValueError('Pretrained backbone position_options mismatch; keep pretraining and adaptation coordinates identical')
        shared = {k[len('backbone.'):]: v for k, v in initial['model'].items() if k.startswith('backbone.')}
        loaded = model.backbone.load_state_dict(shared, strict=False)
        allowed = ('time.refinement.',) if model.config.get('local_refinement', False) else ()
        if model.config.get('multiscale_frontend', False) and not initial['model_config'].get('multiscale_frontend', False):
            allowed += ('time.multiscale.',)
        if loaded.unexpected_keys or any(not key.startswith(allowed) for key in loaded.missing_keys):
            raise ValueError(f'Incompatible pretrained backbone: {loaded}')
    report = model.parameter_report()
    if report['final_parameters'] > 1_000_000_000 or report['shared_ratio'] <= .8:
        raise ValueError(f'Model violates contest parameter constraint: {report}')
    # Construct groups before freezing: future unfrozen parameters retain their
    # group positions and checkpoint-compatible optimizer state.
    optimizer = build_optimizer(model, resolved['optimizer'], resolved['finetuning'] if args.stage == 'supervised' else None)
    accumulation = config['gradient_accumulation_steps']
    steps = config['steps_per_epoch'] or math.ceil(len(train_data) / (config['batch_size'] * world * accumulation))
    total = config['epochs'] * steps
    scheduler, warmup = make_scheduler(optimizer, resolved['scheduler'], total)
    grad_scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and config['precision'] == 'fp16')
    epoch_start, batch_start, global_step, best, metrics = 0, 0, 0, -float('inf'), {}
    validation_global_step = None
    previous = None
    best_metrics = {'modulation': -float('inf'), 'proxy': -float('inf')}
    if args.resume:
        previous = load_checkpoint(args.resume)
        if previous.get('resume_allowed') is False:
            raise ValueError('This checkpoint contains promoted model weights; use --warmstart-checkpoint to train it with a new optimizer')
        if previous['dataset_fingerprint'] != info['fingerprint'] or previous['stage'] != args.stage:
            raise ValueError('Resume stage/dataset mismatch')
        if previous.get('codebook_signature') != codebook_signature:
            raise ValueError('Resume vocabulary mismatch; use init-checkpoint for a new vocabulary iteration')
        if previous['training_config'].get('schema_version') != 2:
            raise ValueError('Legacy checkpoints use a different LR schedule; use --init-checkpoint to start a version-2 experiment')
        saved_config = resolve_config(previous['training_config'])
        previous_model_config = {**model.config, **previous['model_config']}
        if (previous_model_config != model.config or
                resume_comparison_config(saved_config) != resume_comparison_config(resolved) or
                previous.get('world_size', 1) != world):
            raise ValueError('Resume requires identical model/training config and world size; '
                             'use --warmstart-checkpoint for a new optimizer/schedule (or --init-checkpoint for backbone only)')
        model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler'])
        grad_scaler.load_state_dict(previous['grad_scaler'])
        global_step, best = previous['global_step'], previous['best_score']
        initialization_source_steps = previous.get('initialization_source_steps', previous.get('repair_source_steps', 0))
        metrics = previous['metrics']
        best_metrics = previous.get('best_metrics', best_metrics)
        validation_global_step = previous.get('validation_global_step')
        epoch_start = previous['epoch'] + int(previous['epoch_complete'])
        batch_start = 0 if previous['epoch_complete'] else previous['batch_index'] + 1
        if cap and cap <= global_step:
            raise ValueError('max-steps must exceed the saved global step when resuming')
    fine = None
    if args.stage == 'supervised':
        if ((resolved['finetuning']['strategy'] == 'gradual' or resolved['finetuning']['anchor_weight'])
                and not (args.init_checkpoint or args.warmstart_checkpoint or args.resume)):
            raise ValueError('Gradual fine-tuning / pretraining anchor requires an initialization or resume checkpoint')
        if (resolved['finetuning']['strategy'] == 'gradual' and
                resolved['finetuning']['head_warmup_steps'] + resolved['finetuning']['upper_layers_steps'] >= total):
            raise ValueError('Fine-tuning frozen phases must leave updates for the full stage')
        anchor = previous.get('finetune_anchor') if previous else None
        if previous and resolved['finetuning']['anchor_weight'] and anchor is None:
            raise ValueError('Resume checkpoint is missing its original pretraining anchor')
        fine = FineTuneController(model, resolved['finetuning'], anchor)
        fine.apply(global_step)
    sampling = resolved['data']['sampling']
    sampler = BalancedBatchSampler(train_data, config['batch_size'], steps * accumulation, config['seed'], rank,
                                   sampling['mode'], sampling['task_probabilities'])
    loader = DataLoader(train_data, batch_sampler=sampler, collate_fn=collate, **loader_options(resolved['data']['loader'], device))
    val_loader = DataLoader(val_data, batch_size=resolved['evaluation']['batch_size'], collate_fn=collate,
                            **loader_options(resolved['data']['loader'], device))
    module = model
    def wrap_ddp():
        if world == 1:
            return module
        ddp_options = dict(device_ids=[device.index] if device.type == 'cuda' else None, find_unused_parameters=False)
        signature = inspect.signature(DistributedDataParallel.__init__).parameters
        ddp_options['forward_sync_buffers' if 'forward_sync_buffers' in signature else 'broadcast_buffers'] = False
        return DistributedDataParallel(module, **ddp_options)
    model = wrap_ddp()
    args.output = training_run_directory(args.output, args.resume, rank, world)
    run_log = ReadableRunLog(args.output) if rank == 0 else None
    if rank == 0:
        write_json(args.output / 'resolved_config.json', resolved)
        write_json(args.output / 'parameter_report.json', report)
        log_json(args.output / 'events.jsonl', dict(event='start', stage=args.stage, device=str(device), world_size=world,
                    train_rows=len(train_data), validation_rows=len(val_data), steps_per_epoch=steps, parameters=report,
                    gradient_accumulation_steps=accumulation, effective_batch_size=config['batch_size'] * world * accumulation,
                    scheduler=resolved['scheduler'], resolved_warmup_steps=warmup, planned_optimizer_updates=total, max_steps=cap,
                    finetuning_state=fine.state(global_step) if fine else None,
                    initialization={'resume': str(args.resume) if args.resume else None,
                                    'backbone': str(args.init_checkpoint) if args.init_checkpoint else None,
                                    'full_model': str(args.warmstart_checkpoint) if args.warmstart_checkpoint else None}), console=False)
        training_overview(resolved, args, device, world, len(train_data), len(val_data), info,
                          report, steps, warmup, cap, global_step, epoch_start, codebook)
        if fine:
            print(f'微调阶段 | step {global_step:,} | {fine.phase} | 可训练参数 '
                  f'{fine.state(global_step)["trainable_parameters"]:,}', flush=True)
    def checkpoint_payload(epoch, batch_index, epoch_complete):
        return dict(model=module.state_dict(), model_config=module.config, schema=info['schema'],
            target_scaler=info['target_scaler'], parameter_grids=info['parameter_grids'],
            dataset_fingerprint=info['fingerprint'], stage=args.stage, optimizer=optimizer.state_dict(),
            codebook_signature=codebook_signature, scheduler=scheduler.state_dict(),
            grad_scaler=grad_scaler.state_dict(), training_config=resolved, world_size=world,
            epoch=epoch, batch_index=batch_index, epoch_complete=epoch_complete, global_step=global_step,
            best_score=best, metrics=metrics, validation_global_step=validation_global_step,
            best_metrics=best_metrics,
            finetuning_state=fine.state(global_step) if fine else None,
            finetune_anchor=fine.anchor_state() if fine else None,
            initialization_source_steps=initialization_source_steps)

    last_assessment = {'improved': False, 'saved': []}
    def assess(epoch, batch_index, complete, trigger):
        nonlocal metrics, best, validation_global_step, last_assessment
        if world > 1:
            dist.barrier()
        if rank == 0:
            print(f'验证中 | epoch {epoch + 1} | step {global_step:,} | {trigger}', flush=True)
            if args.stage == 'pretrain':
                value = evaluate_pretraining(module, val_loader, device, config, codebook, resolved['evaluation']['max_batches'])
                metrics = {'pretraining_validation_loss': value}
            else:
                metrics = evaluate(module, val_loader, device, info['target_scaler'], info['schema'],
                    resolved['evaluation']['precision'], resolved['evaluation']['max_batches'],
                    resolved['evaluation']['proxy_task_weights'])
            score = selection_score(metrics, resolved['evaluation']['selection_metric'], args.stage)
            improved = score > best
            best = max(best, score)
            wins = {}
            if args.stage == 'supervised':
                for key, value in [('modulation', metrics['classification']['modulation']['accuracy']),
                                   ('proxy', metrics['proxy_score'])]:
                    wins[key] = value > best_metrics[key]
                    best_metrics[key] = max(best_metrics[key], value)
            validation_global_step = global_step
            payload = checkpoint_payload(epoch, batch_index, complete)
            saved = []
            targets = [('last.pt', resolved['checkpoint']['save_last']),
                       ('best.pt', improved and resolved['checkpoint']['save_best']),
                       ('best_modulation.pt', wins.get('modulation', False) and resolved['checkpoint']['save_best_modulation']),
                       ('best_proxy.pt', wins.get('proxy', False) and resolved['checkpoint']['save_best_proxy'])]
            for name, enabled in targets:
                if enabled:
                    save_checkpoint(args.output / name, payload)
                    saved.append(name)
            write_json(args.output / 'metrics.json', metrics)
            summary = ({'pretraining_validation_loss': metrics['pretraining_validation_loss']} if args.stage == 'pretrain'
                else {'classification_accuracy': {k:v['accuracy'] for k,v in metrics['classification'].items()},
                      'classification_f1': {k:{'macro_f1':v['macro_f1'],'weighted_f1':v['weighted_f1']} for k,v in metrics['classification'].items()},
                      'parameter_rmse': {k:v['rmse'] for k,v in metrics['parameters'].items()}})
            log_json(args.output / 'events.jsonl', dict(event='validation', epoch=epoch, step=global_step,
                trigger=trigger, score=score, best=best, improved=improved,
                summary=summary, finetuning_state=fine.state(global_step) if fine else None), console=False)
            if args.stage == 'supervised' and trigger != 'epoch':
                classification_summary(metrics)
                print('  参数 RMSE | ' + ' | '.join(f'{k}={v["rmse"]:.3e}' for k,v in metrics['parameters'].items()
                                                   if v['rmse'] is not None), flush=True)
            last_assessment = {'improved': improved, 'saved': saved}
        validation_global_step = global_step  # All ranks must agree about duplicate validation.
        if world > 1:
            dist.barrier()
        if fine:
            fine.train_mode()
        else:
            module.train()

    started, stop = time.monotonic(), False
    for epoch in range(epoch_start, config['epochs']):
        sampler.set_epoch(epoch)
        model.train()
        if fine:
            fine.train_mode()
        last_batch = batch_start - 1 if epoch == epoch_start else -1
        update_loss, update_parts = 0., {}
        initial = batch_start // accumulation if epoch == epoch_start else 0
        progress = EpochProgress(epoch, config['epochs'], steps, initial,
                                 enabled=rank == 0 and config['progress_bar'], min_interval=config['progress_min_interval'],
                                 gradient_clip=config['gradient_clip'])
        progress.log_every = config['log_every']
        with progress:
            for batch_index, batch in enumerate(loader):
                if epoch == epoch_start and batch_index < batch_start:
                    continue
                if cap and global_step >= cap:
                    stop = True
                    break
                if fine and batch_index % accumulation == 0:
                    changed = fine.apply(global_step)
                    if changed:
                        if world > 1:
                            dist.barrier()
                            if not hasattr(model, '_remove_autograd_hooks'):
                                raise RuntimeError('This PyTorch DDP cannot safely rebuild gradual-unfreezing hooks')
                            model._remove_autograd_hooks()
                            del model
                            model = wrap_ddp()
                        if rank == 0:
                            print(f'微调阶段 | step {global_step:,} | {fine.phase} | 可训练参数 '
                                  f'{fine.state(global_step)["trainable_parameters"]:,}', flush=True)
                            log_json(args.output / 'events.jsonl', dict(event='finetuning_phase',
                                step=global_step, **fine.state(global_step)), console=False)
                last_batch = batch_index
                active = active_loss_parts(batch, config) if rank == 0 else set()
                if batch_index % accumulation == 0:
                    update_active_heads = set()
                if config['skip_inactive_heads']:
                    update_active_heads.update(active_heads(batch, config))
                batch = to_device(batch, device)
                seed = config['seed'] + (global_step * accumulation + batch_index % accumulation) * world + rank
                torch.manual_seed(seed)
                if batch_index % accumulation == 0:
                    optimizer.zero_grad(set_to_none=True)
                    update_loss, update_parts = 0., {}
                boundary = (batch_index + 1) % accumulation == 0
                sync = model.no_sync() if world > 1 and not boundary else nullcontext()
                with sync:
                    with autocast(device, config['precision']):
                        if args.stage == 'pretrain':
                            mask = patch_mask(batch['lengths'], module.config['patch_size'], config['mask_ratio'], config['mask_span'],
                                              seed, batch['iq'], config['mask_strategy'], config['energy_fraction'], config['energy_floor'])
                            output = model(batch['iq'], batch['lengths'], patch_mask=mask, return_pretrain=True)
                            loss, parts = pretraining_loss(output, batch, mask, config, codebook)
                        else:
                            training_iq = augment_supervised_iq(batch, config['augmentation'])
                            output = model(training_iq, batch['lengths'])
                            loss, parts = supervised_loss(output, batch, info['target_scaler'], info['parameter_grids'], config)
                            anchor_loss = fine.anchor_loss() if fine else None
                            if anchor_loss is not None:
                                loss = loss + resolved['finetuning']['anchor_weight'] * anchor_loss
                                parts['anchor'] = float(anchor_loss.detach())
                                if rank == 0:
                                    active.add('anchor')
                        if config['skip_inactive_heads'] and parts.get('supcon_coverage', 0) > 0:
                            update_active_heads.add('aux.contrastive.')
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f'Nonfinite loss at step {global_step}: {parts}')
                    grad_scaler.scale(loss / accumulation).backward()
                loss_value = float(loss.detach())
                if rank == 0:
                    if parts.get('supcon_coverage') == 0:
                        active.discard('supcon')
                    progress.observe(loss_value, parts, active)
                update_loss += loss_value / accumulation
                for key, value in parts.items():
                    update_parts[key] = update_parts.get(key, 0.) + value / accumulation
                if not boundary:
                    continue
                grad_scaler.unscale_(optimizer)
                if config['skip_inactive_heads']:
                    clear_inactive_head_gradients(module, update_active_heads, device, world)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'] or float('inf'), error_if_nonfinite=True)
                used_rates = {group['name']: group['lr'] for group in optimizer.param_groups}
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()
                global_step += 1
                if rank == 0:
                    progress.advance(used_rates, float(norm))
                    if global_step == 1 or global_step % config['log_every'] == 0:
                        log_json(args.output / 'events.jsonl', dict(event='step', epoch=epoch, step=global_step, loss=update_loss,
                            parts=update_parts, gradient_norm=float(norm), lr=next(iter(used_rates.values())), learning_rates=used_rates,
                            elapsed_seconds=time.monotonic() - started), console=False)
                every = resolved['finetuning']['validation_every_steps'] if fine else 0
                phase_end = fine and resolved['finetuning']['validate_stage_boundaries'] and fine.is_boundary(global_step)
                step_validated = False
                if batch_index + 1 < steps * accumulation and (phase_end or (every and global_step % every == 0)):
                    assess(epoch, batch_index, False, 'phase_boundary' if phase_end else 'steps')
                    step_validated = True
                save_steps = resolved['checkpoint']['save_every_steps']
                if (resolved['checkpoint']['save_last'] and save_steps and global_step % save_steps == 0
                        and batch_index + 1 < steps * accumulation and not step_validated):
                    if rank == 0:
                        # Only save at optimizer boundaries; partial accumulated gradients
                        # are never needed to restart this checkpoint.
                        save_checkpoint(args.output / 'last.pt', checkpoint_payload(epoch, batch_index, False))
                        log_json(args.output / 'events.jsonl', dict(event='checkpoint', kind='periodic',
                            epoch=epoch, step=global_step, path=str(args.output / 'last.pt'),
                            validation_global_step=validation_global_step), console=False)
                        print(f'定期保存 | step {global_step:,} | last.pt', flush=True)
                    if world > 1:
                        dist.barrier()
        complete = last_batch == steps * accumulation - 1
        finished = epoch + 1 == config['epochs'] or stop or bool(cap and global_step >= cap)
        should_evaluate = finished or (epoch + 1) % resolved['evaluation']['every_epochs'] == 0
        if should_evaluate and validation_global_step != global_step:
            assess(epoch, last_batch, complete, 'epoch')
        if rank == 0:
            improved = last_assessment['improved'] if should_evaluate else False
            payload = checkpoint_payload(epoch, last_batch, complete)
            saved = list(last_assessment['saved']) if should_evaluate else []
            if resolved['checkpoint']['save_last']:
                save_checkpoint(args.output / 'last.pt', payload)
                if 'last.pt' not in saved:
                    saved.append('last.pt')
            interval = resolved['checkpoint']['save_every_epochs']
            if interval and complete and (epoch + 1) % interval == 0:
                save_checkpoint(args.output / f'epoch_{epoch + 1:04d}.pt', payload)
                saved.append(f'epoch_{epoch + 1:04d}.pt')
            training_summary = progress.summary()
            log_json(args.output / 'events.jsonl', dict(event='epoch', epoch=epoch, step=global_step,
                     complete=complete, training=training_summary, saved=saved), console=False)
            epoch_summary(epoch, config['epochs'], training_summary, metrics, should_evaluate, improved, saved, complete)
        if world > 1:
            dist.barrier()
        batch_start = 0
        if stop or (cap and global_step >= cap):
            break
    if rank == 0:
        print(f"训练结束 | 总更新 {global_step:,}/{total:,} | 用时 {time.monotonic() - started:.1f}s | 输出 {args.output}", flush=True)
        if args.stage == 'supervised' and (args.output / 'best.pt').is_file():
            print('导出本次最佳模型供推理使用：\n' + shlex.join([
                sys.executable, str(Path(__file__).with_name('export.py')),
                '--checkpoint', str(args.output / 'best.pt'), '--output', 'weights']), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
    if run_log:
        run_log.close()


if __name__ == '__main__':
    main()
