"""One resumable shared-model run: local-offset adapter and paired-view learning.

Run from anywhere with --config; config paths are relative to the repository.
No teacher, validation fitting, prediction ensemble, or task-specific backbone.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
import numpy as np
import torch
from tqdm import tqdm
from models.rf_model import RFModel
from models.patch_shift import PATCH_SHIFT_DEFAULTS, patch_shift_options
from utils.common import (read_config_json, write_json, model_bundle, save_checkpoint,
                          seed_all, save_safetensors, load_safetensors)
from utils.config import merge_known, resolve_config, training_options, make_scheduler
from utils.contrastive import load_with_training_aux
from utils.dataset import PreparedDataset, BalancedBatchSampler, collate, to_device
from utils.engine import autocast, supervised_loss
from utils.scoring import add_composite_score, selection_definition, TASK_WEIGHTS
from utils.checkpoint_selection import migrate_best_selection
from utils.head_optimizer import active_heads, clear_inactive_head_gradients
from utils.process_lifecycle import managed_entrypoint
from utils.tensorboard_log import TensorBoardLog
from utils.view_consistency import eligible_modulation, augmented_view, paired_correction
from training.validate_c01_followup import predict, task_metrics, class_result


DEFAULTS = {
    'run': {'source': '', 'data': 'artifacts/data_full', 'output': 'logs/supervised_shift_consistent'},
    'model': {'patch_shift': {**PATCH_SHIFT_DEFAULTS, 'enabled': True}},
    'training': {'steps': 24000, 'batch_size': 256, 'seed': 2026111427,
                 'precision': 'bf16', 'gradient_clip': 1., 'ema_decay': .995},
    'optimizer': {'learning_rate': 2e-5, 'adapter_learning_rate': 2e-4, 'weight_decay': .01},
    'scheduler': {'warmup_steps': 500, 'minimum_ratio': .2},
    'views': {'class_indices': list(range(14)), 'phase_probability': 1., 'shift_probability': 1.,
              'augmented_weight': .5, 'consistency_weight': .05, 'ramp_steps': 2000},
    'evaluation': {'every_steps': 2400, 'batch_size': 128, 'protocol_max_drop': .003,
                   'qam_max_drop': .005, 'macro_f1_max_drop': .003, 'parameter_max_rmse_increase': .05},
    'checkpoint': {'every_steps': 500},
    'runtime': {'device': 'cuda:0', 'threads': 4, 'log_every': 50, 'tensorboard': True},
}


def path(value):
    return (ROOT/value).resolve()


def read_settings(filename):
    config = merge_known(DEFAULTS, read_config_json(filename))
    patch_shift_options(config['model']['patch_shift'])
    t, v, e, runtime = (config[k] for k in ['training', 'views', 'evaluation', 'runtime'])
    def require(ok, message):
        if not ok:
            raise ValueError(message)
    for name, value in [('steps', t['steps']), ('batch_size', t['batch_size']),
                        ('every_steps', e['every_steps']), ('evaluation.batch_size', e['batch_size']),
                        ('checkpoint.every_steps', config['checkpoint']['every_steps']),
                        ('threads', runtime['threads']), ('log_every', runtime['log_every']),
                        ('ramp_steps', v['ramp_steps'])]:
        require(type(value) is int and value > 0, f'{name} must be a positive integer')
    require(t['precision'] in ['bf16', 'fp32', 'fp16'], 'Invalid precision')
    require(0 <= t['ema_decay'] < 1 and t['gradient_clip'] > 0, 'Invalid EMA/clip')
    require(0 <= v['augmented_weight'] <= 1 and v['consistency_weight'] >= 0, 'Invalid paired loss')
    require(all(0 <= v[k] <= 1 for k in ['phase_probability', 'shift_probability']), 'Invalid view probability')
    require(isinstance(v['class_indices'], list) and v['class_indices'] and
            all(type(i) is int and i >= 0 for i in v['class_indices']) and
            len(set(v['class_indices'])) == len(v['class_indices']), 'Invalid view class indices')
    require(all(config['optimizer'][k] > 0 for k in ['learning_rate', 'adapter_learning_rate']), 'Learning rates must be positive')
    require(config['optimizer']['weight_decay'] >= 0, 'Weight decay must be nonnegative')
    require(type(config['scheduler']['warmup_steps']) is int and 0 <= config['scheduler']['warmup_steps'] < t['steps'], 'Invalid warmup')
    require(0 < config['scheduler']['minimum_ratio'] <= 1, 'Invalid minimum LR')
    require(all(e[k] >= 0 for k in ['protocol_max_drop', 'qam_max_drop', 'macro_f1_max_drop', 'parameter_max_rmse_increase']), 'Invalid selection guard')
    require(all(isinstance(value, str) and value for value in config['run'].values()), 'Run paths must be nonempty strings')
    # JSON does not forbid NaN in Python; reject it before touching any output.
    json.dumps(config, allow_nan=False)
    return config


def signature(config):
    value = deepcopy(config)
    value['run'].pop('output')
    for name in ['device', 'threads', 'log_every', 'tensorboard']:
        value['runtime'].pop(name)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def optimizer_for(model, config):
    groups = {}
    for name, parameter in model.named_parameters():
        adapter = name.startswith('backbone.patch_shift.')
        lr = config['adapter_learning_rate' if adapter else 'learning_rate']
        decay = 0. if parameter.ndim < 2 or name.endswith('.bias') else config['weight_decay']
        key = ('adapter' if adapter else 'original', decay)
        if key not in groups:
            groups[key] = dict(params=[], lr=lr, weight_decay=decay, name=key[0]+('_decay' if decay else '_no_decay'))
        groups[key]['params'].append(parameter)
    return torch.optim.AdamW(list(groups.values()), betas=(.9, .999), eps=1e-8)


def evaluate_model(model, bundle, dataset, rows, output, name, device, batch_size):
    model.eval()
    if dataset.iq is None:
        dataset.iq = np.load(dataset.directory/'iq.npy', mmap_mode='r')
    prediction = predict(model, rows, dataset.iq, dataset.meta, bundle['target_scaler'],
                         device, batch_size, name)
    metric = task_metrics(prediction, dataset.meta, bundle['schema'])
    add_composite_score(metric, bundle['target_scaler'])
    y = dataset.meta['modulation'][prediction['rows']]
    metric['classification']['qam'] = class_result(prediction['modulation'], y,
        bundle['schema']['classification_labels']['modulation'], np.isin(y, [4, 5, 6, 7, 12]))
    np.savez_compressed(output/(name+'.npz'), **prediction)
    write_json(output/(name+'_metrics.json'), metric)
    return metric


def selection(metric, baseline, settings):
    violations = []
    for name, allowance in [('protocol', settings['protocol_max_drop']), ('qam', settings['qam_max_drop'])]:
        if metric['classification'][name]['accuracy'] < baseline['classification'][name]['accuracy']-allowance:
            violations.append(name+'_accuracy')
    if metric['classification']['modulation']['macro_f1'] < baseline['classification']['modulation']['macro_f1']-settings['macro_f1_max_drop']:
        violations.append('modulation_macro_f1')
    for field, original in baseline['parameters'].items():
        if original['rmse'] is not None and metric['parameters'][field]['rmse'] > original['rmse']*(1+settings['parameter_max_rmse_increase']):
            violations.append(field+'_rmse')
    return not violations, violations


def export_best(checkpoint, destination):
    destination.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    bundle = model_bundle(state)
    exported = RFModel(**bundle).eval()
    exported.load_state_dict(state['model'], strict=True)
    filename = destination/'model.safetensors'
    save_safetensors(filename, exported.state_dict())
    restored = load_safetensors(filename)
    if set(restored) != set(exported.state_dict()) or any(not torch.equal(restored[k], v.cpu()) for k, v in exported.state_dict().items()):
        raise ValueError('Export tensor round-trip failed')
    write_json(destination/'config.json', {**bundle, 'format_version': 1,
        'parameters': exported.parameter_report(), 'source_checkpoint': str(checkpoint),
        'trained_steps': state['global_step']+state['initialization_source_steps'],
        'validation_metrics': state['metrics'], 'note': 'Local development metrics, not official contest score.'})


def smoke_validation(dataset):
    meta, rows = dataset.meta, dataset.indices
    selected = []
    for task in ['modulation', 'protocol']:
        for label in np.unique(meta[task][rows]):
            if label >= 0:
                selected.extend(rows[meta[task][rows] == label][:2])
    for j in range(7):
        selected.extend(rows[meta['target_valid'][rows, j]][:4])
    return np.unique(selected)


@managed_entrypoint('Patch偏移一致性训练', shutdown_timeout=30.)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'code/configs/supervised/shift_consistent.json5')
    parser.add_argument('--resume', type=Path, help='Resume THIS trainer last.pt, retaining original output directory')
    parser.add_argument('--max-steps', type=int, default=0, help='Stop cap; does not alter the planned cosine schedule')
    parser.add_argument('--smoke', action='store_true', help='Six tiny steps and stratified subset validation; NOT a score run')
    parser.add_argument('--source', type=Path, help='Override source checkpoint, recorded in protocol')
    parser.add_argument('--output', type=Path, help='New run parent directory')
    parser.add_argument('--device', help='cuda:0 or cpu, one process only')
    args = parser.parse_args()
    c = read_settings(args.config)
    if args.source:
        c['run']['source'] = str(args.source.resolve())
    if args.output:
        c['run']['output'] = str(args.output.resolve())
    if args.device:
        c['runtime']['device'] = args.device
    if args.smoke:
        c['training'].update(steps=6, batch_size=4)
        c['scheduler']['warmup_steps'] = 1
        c['evaluation'].update(every_steps=3, batch_size=4)
        c['checkpoint']['every_steps'] = 3
        c['runtime'].update(log_every=1, tensorboard=False)
    if args.max_steps < 0:
        parser.error('--max-steps must be nonnegative')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This trainer uses one shared model on one process/device')
    device = torch.device(c['runtime']['device'])
    if device.type == 'cpu' and c['training']['precision'] == 'fp16':
        raise ValueError('Use FP32 or BF16 for CPU')
    if device.type == 'cuda':
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if len(visible.split(',')) != 1 or not visible or visible == '-1':
            raise ValueError('Set CUDA_VISIBLE_DEVICES to exactly one allocated GPU')
        torch.cuda.set_device(device)
    torch.set_num_threads(c['runtime']['threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(c['training']['seed'])
    previous = torch.load(args.resume, map_location='cpu', weights_only=True, mmap=True) if args.resume else None
    resumed_metrics = previous['metrics'] if previous else None
    if previous and (previous.get('trainer') != 'shift_consistent_v1' or previous['config_signature'] != signature(c)
                     or previous['smoke'] != args.smoke):
        raise ValueError('Resume requires this trainer, identical recipe and smoke mode')
    if previous and not all(k in previous for k in ['optimizer', 'scheduler', 'grad_scaler', 'ema']):
        raise ValueError('Resume last.pt with optimizer state; best.pt and *_model.pt are inference/warmstart weights')
    train = PreparedDataset(path(c['run']['data']), 'train')
    val = PreparedDataset(path(c['run']['data']), 'validation')
    if not len(train) or not len(val):
        raise ValueError('Both training and validation records are required')
    source_path = path(c['run']['source'])
    source = previous or torch.load(source_path, map_location='cpu', weights_only=True, mmap=True)
    if source['stage'] != 'supervised' or source['dataset_fingerprint'] != train.info['fingerprint']:
        raise ValueError('Source must be a supervised checkpoint from the same grouped split')
    if any(source[k] != train.info[k] for k in ['schema', 'target_scaler', 'parameter_grids']):
        raise ValueError('Source schema/scales/grids do not match the prepared dataset')
    if max(c['views']['class_indices']) >= len(source['schema']['classification_labels']['modulation']):
        raise ValueError('View class index is outside the source schema')
    bundle = model_bundle(source)
    bundle['model_config'] = {**bundle['model_config'], 'pretrain_heads': False,
        'contrastive_head': False, 'patch_shift': c['model']['patch_shift']}
    model = RFModel(**bundle)
    load_with_training_aux(model, source['model'], source['model_config'])
    report = model.parameter_report()
    if report['shared_ratio'] <= .8 or report['final_parameters'] > 1_000_000_000:
        raise ValueError(f'Contest parameter constraint failed: {report}')
    training_config = resolve_config(source['training_config'])
    training_config['model']['patch_shift'] = model.config['patch_shift']
    options = training_options(training_config)
    options['stage'] = 'supervised'
    if options['contrastive_weight']:
        raise ValueError('Use a source without active contrastive loss for this continuation')
    model = model.to(device)
    opt = optimizer_for(model, c['optimizer'])
    scheduler, _ = make_scheduler(opt, dict(name='cosine', warmup_steps=c['scheduler']['warmup_steps'],
        warmup_ratio=0., warmup_start_factor=.1, min_lr_ratio=c['scheduler']['minimum_ratio']), c['training']['steps'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and c['training']['precision'] == 'fp16')
    step = source['global_step'] if previous else 0
    if args.max_steps and args.max_steps <= step:
        raise ValueError('--max-steps must exceed the resumed step')
    if previous:
        out = args.resume.resolve().parent
        c['run']['output'] = str(out.parent)
        opt.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler'])
        scaler.load_state_dict(previous['grad_scaler'])
        ema = {k: v.to(device) for k, v in previous['ema'].items()}
        baseline, best_score = previous['baseline_metrics'], previous['best_score']
        last_metrics_step = previous.get('metrics_validation_step', 0)
        origin_steps, source_hash = previous['initialization_source_steps'], previous['source_sha256']
    else:
        out = path(c['run']['output'])/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        out.mkdir(parents=True, exist_ok=False)
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        origin_steps = source['global_step']+source.get('initialization_source_steps', 0)
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        baseline, best_score = None, -math.inf
        last_metrics_step = 0
        for folder in ['models', 'utils']:
            for filename in (ROOT/'code'/folder).glob('*.py'):
                destination = out/'source_snapshot'/folder/filename.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(filename, destination)
        for filename in [Path(__file__), ROOT/'code/training/validate_c01_followup.py']:
            shutil.copy2(filename, out/'source_snapshot'/filename.name)
    composite_evaluation = {'selection_metric': 'proxy_score', 'proxy_task_weights': TASK_WEIGHTS}
    if previous:
        migration = migrate_best_selection(out, previous, composite_evaluation, bundle['target_scaler'])
        if migration:
            best_score = migration['best_score'] if migration['best_score'] is not None else -math.inf
        add_composite_score(baseline, bundle['target_scaler'])
    rows = smoke_validation(val) if args.smoke else val.indices
    stop = min(c['training']['steps'], args.max_steps or c['training']['steps'])
    write_json(out/'resolved_config.json', c)
    write_json(out/'parameter_report.json', report)
    write_json(out.parent/'latest_run.json', dict(run_directory=str(out), timezone='Asia/Shanghai'))
    del source, previous

    def payload(metric, estimator, resumable=False):
        value = {**bundle, 'model_config': model.config,
            'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'training_config': training_config, 'optimization_config': c,
            'trainer': 'shift_consistent_v1', 'config_signature': signature(c),
            'stage': 'supervised', 'dataset_fingerprint': train.info['fingerprint'],
            'global_step': step, 'initialization_source_steps': origin_steps,
            'metrics': metric, 'baseline_metrics': baseline, 'best_score': best_score,
            'selection': selection_definition('supervised', composite_evaluation),
            'metrics_validation_step': last_metrics_step if resumable else step,
            'source_sha256': source_hash, 'weight_estimator': estimator, 'smoke': args.smoke,
            'resume_allowed': False, 'resume_entrypoint': 'code/training/train_shift_consistent.py'}
        if resumable:
            value.update(optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                         grad_scaler=scaler.state_dict(), ema={k: v.detach().cpu() for k, v in ema.items()})
        return value

    with (out/'train.log').open('a', buffering=1) as logfile, (out/'events.jsonl').open('a', buffering=1) as events, TensorBoardLog(
            out, options={'enabled': c['runtime']['tensorboard']}, log_every=c['runtime']['log_every'],
            resume_step=step if args.resume else None) as tb:
        def record(event):
            events.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
        def say(message):
            print(message, flush=True)
            logfile.write(message+'\n')
        say(f'RUN {out} | device={device} | steps={step}/{c["training"]["steps"]} | batch={c["training"]["batch_size"]}')
        say(f'参数 {report["final_parameters"]:,} | 共享 {100*report["shared_ratio"]:.2f}% | 仅监督训练侧目标类使用成对增强')
        tb.metadata(c, report, step)
        if baseline is None:
            baseline = evaluate_model(model, bundle, val, rows, out, 'source_validation', device, c['evaluation']['batch_size'])
            best_score = baseline['proxy_score']
            save_checkpoint(out/'best.pt', payload(baseline, 'initial'))
            say(f'起点本地综合分={best_score:.6f}/100（非官方） | full_validation={not args.smoke}; 未提升时保留此起点')
        last_metrics = resumed_metrics or baseline

        def assess():
            nonlocal best_score, last_metrics, last_metrics_step
            original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            evaluation_dir = out/'evaluations'/f'step_{step:08d}'
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            try:
                for estimator, state in [('last', original), ('ema', ema)]:
                    model.load_state_dict(state, strict=True)
                    metric = evaluate_model(model, bundle, val, rows, evaluation_dir, estimator, device, c['evaluation']['batch_size'])
                    score = metric['proxy_score']
                    allowed, violations = selection(metric, baseline, c['evaluation'])
                    improved = score > best_score  # Guards remain diagnostics; primary best is composite.
                    if improved:
                        best_score = score
                        save_checkpoint(out/'best.pt', payload(metric, estimator))
                    save_checkpoint(out/(estimator+'_model.pt'), payload(metric, estimator))
                    record(dict(event='validation', step=step, estimator=estimator,
                                accuracy=metric['classification']['modulation']['accuracy'],
                                score=score, selection_metric='proxy_score',
                                eligible=allowed, violations=violations, improved=improved, best_score=best_score))
                    for task, values in metric['classification'].items():
                        for key in ['accuracy', 'macro_f1', 'weighted_f1', 'cross_entropy']:
                            tb.scalar(f'validation/{estimator}/{task}/{key}', values.get(key), step)
                    for field, values in metric['parameters'].items():
                        for key in ['rmse', 'mae', 'absolute_error_p95']:
                            tb.scalar(f'validation/{estimator}/parameters/{field}/{key}', values.get(key), step)
                    tb.scalar(f'validation/{estimator}/local_composite', score, step)
                    say(f'验证 step={step} {estimator}: 本地综合分={score:.6f}（非官方）'
                        f' | 调制={100*metric["classification"]["modulation"]["accuracy"]:.4f}%'
                        f' | 前14={100*metric["classification"]["first14"]["accuracy"]:.3f}%'
                        f' | QAM={100*metric["classification"]["qam"]["accuracy"]:.3f}% | 保护项诊断={allowed} | best={best_score:.6f} | {violations}')
                    if estimator == 'last':
                        last_metrics, last_metrics_step = metric, step
            finally:
                model.load_state_dict(original, strict=True)
                model.train()
            save_checkpoint(out/'last.pt', payload(last_metrics, 'last', resumable=True))

        if not math.isfinite(best_score):
            assess()  # Legacy resume had no weight/metric-aligned candidate to reselect.
        sampler = BalancedBatchSampler(train, c['training']['batch_size'], c['training']['steps'],
                                       c['training']['seed'], mode='balanced')
        started = time.monotonic()
        progress = tqdm(total=stop, initial=step, desc='shared paired-view training', mininterval=5.)
        try:
            for index, positions in enumerate(sampler):
                if index < step:
                    continue
                if step >= stop:
                    break
                seed_all(c['training']['seed']+index+100000)
                cpu = collate([train[int(position)] for position in positions])
                active = active_heads(cpu, options)
                batch = to_device(cpu, device)
                model.train()
                opt.zero_grad(set_to_none=True)
                with autocast(device, c['training']['precision']):
                    output = model(batch['iq'], batch['lengths'])
                    loss, parts = supervised_loss(output, batch, bundle['target_scaler'], bundle['parameter_grids'], options, distributed=False)
                    selected = eligible_modulation(batch, c['views']['class_indices'])
                    if bool(selected.any()):
                        view = augmented_view(batch['iq'][selected], batch['lengths'][selected],
                            c['training']['seed']+index+700000, c['views']['phase_probability'], c['views']['shift_probability'])
                        other = model(view, batch['lengths'][selected])
                        delta, view_parts = paired_correction(output['modulation'], other['modulation'], batch['modulation'], selected,
                            augmented_weight=c['views']['augmented_weight'], consistency_weight=c['views']['consistency_weight'],
                            ramp=min(1., (index+1)/c['views']['ramp_steps']))
                        loss = loss+options['task_weights'][0]*delta
                        parts.update(view_parts)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f'Nonfinite loss at step {step}: {parts}')
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                clear_inactive_head_gradients(model, active, device)
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), c['training']['gradient_clip'], error_if_nonfinite=True))
                scaler.step(opt)
                scaler.update()
                scheduler.step()
                with torch.no_grad():
                    for key, value in model.state_dict().items():
                        if value.is_floating_point():
                            ema[key].lerp_(value.detach(), 1-c['training']['ema_decay'])
                        else:
                            ema[key].copy_(value)
                step = index+1
                if step == 1 or step % c['runtime']['log_every'] == 0:
                    event = dict(event='step', step=step, loss=float(loss.detach()), parts=parts, gradient_norm=norm,
                                 learning_rates={g['name']: g['lr'] for g in opt.param_groups}, elapsed_seconds=time.monotonic()-started)
                    if hasattr(model.backbone, 'patch_shift'):
                        event['patch_update_ratio'] = float(model.backbone.patch_shift.last_update_ratio)
                    record(event)
                    tb.scalar('train/loss', event['loss'], step)
                    tb.scalar('train/gradient_norm', norm, step)
                    tb.scalar('train/patch_update_ratio', event.get('patch_update_ratio'), step)
                    for key, value in parts.items():
                        tb.scalar('train/'+key, value, step)
                    for key, value in event['learning_rates'].items():
                        tb.scalar('learning_rate/'+key, value, step)
                    say(f'step={step}/{c["training"]["steps"]} loss={event["loss"]:.5f} grad={norm:.3f} patch_ratio={event.get("patch_update_ratio", 0):.5f}')
                progress.update(1)
                if step % c['evaluation']['every_steps'] == 0 or step == stop:
                    assess()
                elif step % c['checkpoint']['every_steps'] == 0:
                    save_checkpoint(out/'last.pt', payload(last_metrics, 'last', resumable=True))
        finally:
            progress.close()
        export_best(out/'best.pt', out/'exported_best')
        write_json(out/'completion.json', dict(status='complete' if step == c['training']['steps'] else 'stopped_at_cap',
            steps=step, best_score=best_score, selection_metric='proxy_score', source_score=baseline['proxy_score'],
            source_accuracy=baseline['classification']['modulation']['accuracy'],
            improved=best_score > baseline['proxy_score'], smoke=args.smoke,
            inference_weights=str(out/'exported_best'), resume_checkpoint=str(out/'last.pt')))
        say(f'完成 | 本地综合分best={best_score:.6f}/100（非官方） | 导出={out/"exported_best"} | smoke={args.smoke}')


if __name__ == '__main__':
    main()
