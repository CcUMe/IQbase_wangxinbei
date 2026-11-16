"""Train the pinned official IQFormer on C01 modulation labels only."""
import argparse
import json
import math
import shutil
import sys
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[1] / 'code'))

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from c01.runtime import (C01Modulation, autocast, build_model, evaluate, load_config,
                         make_loader, path_from_root, print_metrics, restore_rng,
                         resume_config, rng_state, save_metrics, setup_runtime, source_identity)
from utils.common import save_checkpoint, write_json
from utils.readable_log import ReadableRunLog
from utils.tensorboard_log import TensorBoardLog


def scheduler_for(optimizer, config):
    s, total = config['scheduler'], config['training']['epochs']
    if s['name'] == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=s['factor'], patience=s['patience'],
            threshold=s['threshold'], min_lr=s['min_lr'])
    warm, floor = s['warmup_epochs'], s['min_lr'] / config['optimizer']['learning_rate']
    def factor(epoch):
        if epoch < warm:
            return (epoch + 1) / warm
        progress = min(1., max(0., (epoch - warm) / (total - warm)))
        return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE / 'c01.json5')
    parser.add_argument('--device', help='auto, cpu, or cuda:0 (visible device index)')
    parser.add_argument('--output', type=Path, help='Parent folder; a timestamp is always appended')
    parser.add_argument('--resume', type=Path, help='Completed-epoch checkpoint; creates a new timestamped continuation')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--max-train-batches', type=int)
    parser.add_argument('--max-val-batches', type=int)
    parser.add_argument('--stop-after-epochs', type=int, default=0,
                        help='Stop this invocation early without changing the planned scheduler')
    args = parser.parse_args()
    config = load_config(args.config)
    for value, section, key in [(args.epochs, 'training', 'epochs'),
                                (args.num_workers, 'data', 'num_workers'),
                                (args.max_train_batches, 'training', 'max_train_batches'),
                                (args.max_val_batches, 'training', 'max_val_batches')]:
        if value is not None:
            if value < (1 if key == 'epochs' else 0):
                parser.error(f'Invalid --{key.replace("_", "-")}')
            config[section][key] = value
    if args.output:
        config['run']['output'] = str(path_from_root(args.output))
    if args.device:
        config['runtime']['device'] = args.device
    if args.stop_after_epochs < 0 or config['scheduler']['warmup_epochs'] >= config['training']['epochs']:
        parser.error('Invalid stop limit or warmup/total epochs')
    device = setup_runtime(config)
    train = C01Modulation(config['run']['data'], 'train')
    val = C01Modulation(config['run']['data'], 'validation')
    if train.info['fingerprint'] != val.info['fingerprint'] or train.labels != val.labels:
        raise ValueError('Train/validation schema mismatch')
    train_loader, train_sampler = make_loader(train, config, True)
    val_loader, _ = make_loader(val, config, False)
    model = build_model(config, train.labels).to(device)
    settings = config['optimizer']
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                 weight_decay=settings['weight_decay'], betas=settings['betas'], eps=settings['eps'])
    scheduler = scheduler_for(optimizer, config)
    identity = source_identity()
    start_epoch, global_step, best, patience_best, bad_epochs = 0, 0, -1., -1., 0
    inherited_best = None
    if args.resume:
        previous = torch.load(args.resume, map_location='cpu', weights_only=True, mmap=True)
        if previous.get('format') != 'iqformer-c01-v1' or not previous.get('epoch_complete'):
            raise ValueError('Resume requires a completed-epoch C01 checkpoint')
        if (previous['dataset_fingerprint'] != train.info['fingerprint'] or previous['labels'] != train.labels
                or resume_config(previous['config']) != resume_config(config)
                or previous['source'] != identity):
            raise ValueError('Resume architecture/data/recipe/source mismatch; keep the original config')
        model.load_state_dict(previous['model'], strict=True)
        optimizer.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler'])
        start_epoch, global_step = previous['epoch'] + 1, previous['global_step']
        best, patience_best, bad_epochs = previous['best'], previous['patience_best'], previous['bad_epochs']
        restore_rng(previous['rng'])
        for candidate in dict.fromkeys([args.resume, args.resume.parent / 'best.pt']):
            if not candidate.is_file():
                continue
            saved = torch.load(candidate, map_location='cpu', weights_only=True, mmap=True)
            if (saved.get('format') == 'iqformer-c01-v1' and saved['source'] == identity
                    and saved['dataset_fingerprint'] == train.info['fingerprint']
                    and saved['global_step'] <= global_step
                    and saved['metrics']['classification']['modulation']['accuracy'] == best):
                inherited_best = (candidate, saved['metrics'])
                del saved
                break
            del saved
        del previous
    if start_epoch >= config['training']['epochs']:
        raise ValueError('Checkpoint already completed all planned epochs')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    run = Path(config['run']['output']) / stamp
    run.mkdir(parents=True, exist_ok=False)
    if inherited_best:
        shutil.copy2(inherited_best[0], run / 'best.pt')
        write_json(run / 'best_metrics.json', inherited_best[1])
    write_json(run / 'resolved_config.json', config)
    write_json(run / 'source_info.json', identity)
    write_json(run.parent / 'latest_run.json', {'run_directory': str(run)})
    count = sum(p.numel() for p in model.parameters())
    write_json(run / 'parameter_report.json', {'parameters': count, 'classes': train.labels,
                                              'task': 'modulation-only diagnostic baseline'})
    total_epochs = config['training']['epochs']
    train_limit, val_limit = [config['training'][key] for key in ['max_train_batches', 'max_val_batches']]
    invocation_epochs = 0
    with ExitStack() as stack:
        stack.enter_context(ReadableRunLog(run))
        tb = stack.enter_context(TensorBoardLog(
            run, {'enabled': config['logging']['tensorboard']},
            log_every=config['logging']['log_every_steps'], resume_step=global_step,
            gradient_clip=config['training']['gradient_clip']))
        event_file = stack.enter_context((run / 'events.jsonl').open('a', buffering=1))
        def event(value, metrics=None):
            event_file.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
            tb.record(value, metrics)
        print(f'训练总览 | 官方IQFormer / C01调制识别 | 参数={count:,} | 类别={len(train.labels)}')
        print(f'数据 | train={len(train):,} | validation={len(val):,} | fingerprint={train.info["fingerprint"]}')
        print(f'批次 | batch={config["data"]["batch_size"]} | 原长度512/1024分桶 | '
              f'每轮{len(train_loader):,}步 | repeats={config["data"]["repeats_per_epoch"]}')
        print(f'模型 | layers={config["model"]["layers"]} | widths={config["model"]["embed_dims"]} | '
              '官方双层BiLSTM＋原分类头')
        print(f'输入 | normalization={config["input"]["normalization"]} | I通道STFT实部 | '
              f'Blackman={config["input"]["stft_window"]}, FFT=128, hop=1, bins=32')
        print(f'优化 | AdamW lr={settings["learning_rate"]:g}, wd={settings["weight_decay"]:g} | '
              f'{config["scheduler"]["name"]} | device={device} | {config["runtime"]["precision"]}')
        print(f'轮次 | {start_epoch + 1}—{total_epochs} | seed={config["runtime"]["seed"]} | '
              f'clip={config["training"]["gradient_clip"]:g} | '
              f'early_stop={config["training"]["early_stopping_patience"]}')
        print('比较口径 | 一个epoch是一次调制样本遍历，与主模型多任务epoch的调制曝光量不同。')
        if train_limit or val_limit:
            print(f'有限批次执行检查 | train_limit={train_limit} | val_limit={val_limit}；不代表全量准确率。')
        print(f'保存目录 | {run}', flush=True)
        tb.metadata(config, {'parameters': count}, global_step)
        event({'event': 'start', 'stage': 'supervised', 'epoch': start_epoch,
               'step': global_step, 'world_size': 1, 'train_rows': len(train),
               'validation_rows': len(val), 'effective_batch_size': config['data']['batch_size'],
               'source': identity, 'resume': str(args.resume) if args.resume else None})
        for epoch in range(start_epoch, total_epochs):
            started = time.monotonic()
            train_sampler.epoch = epoch
            train_loader.generator.manual_seed(config['runtime']['seed'] + epoch)
            model.train()
            loss_sum, correct, seen, norms = 0., 0, 0, []
            steps = min(len(train_loader), train_limit or len(train_loader))
            used_lr = optimizer.param_groups[0]['lr']
            bar = tqdm(train_loader, total=steps, desc=f'Epoch {epoch+1}/{total_epochs}',
                       disable=not config['logging']['progress_bar'], dynamic_ncols=True)
            for index, batch in enumerate(bar):
                if index >= steps:
                    break
                iq, spectrum, y = [batch[k].to(device, non_blocking=True) for k in ['iq', 'stft', 'target']]
                optimizer.zero_grad(set_to_none=True)
                with autocast(config, device):
                    logits = model(iq, spectrum)
                    loss = F.cross_entropy(logits.float(), y,
                                           label_smoothing=config['training']['label_smoothing'])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config['training']['gradient_clip'] or float('inf'), error_if_nonfinite=True))
                optimizer.step()
                global_step += 1
                value = float(loss.detach())
                loss_sum += value * len(y)
                correct += int((logits.argmax(-1) == y).sum())
                seen += len(y)
                norms.append(norm)
                bar.set_postfix(ce=f'{loss_sum/seen:.4f}', acc=f'{correct/seen:.2%}', lr=f'{used_lr:.2e}', grad=f'{norm:.2f}')
                if global_step == 1 or global_step % config['logging']['log_every_steps'] == 0:
                    event({'event': 'step', 'epoch': epoch, 'step': global_step, 'loss': value,
                           'parts': {'modulation': value}, 'active_parts': ['modulation'],
                           'gradient_norm': norm, 'learning_rates': {'model': used_lr},
                           'samples_seen_this_epoch': seen})
                    bar.write(f'  step={global_step} | CE={loss_sum/seen:.4f} | trainAcc={correct/seen:.2%} | '
                              f'grad={norm:.3f} | lr={used_lr:.3e}')
            metrics, _ = evaluate(model, val_loader, config, device, val_limit)
            print_metrics(metrics)
            m = metrics['classification']['modulation']
            score = m['accuracy']
            improved = score > best
            best = max(best, score)
            if score > patience_best + config['training']['early_stopping_min_delta']:
                patience_best, bad_epochs = score, 0
            else:
                bad_epochs += 1
            if config['scheduler']['name'] == 'plateau':
                scheduler.step(m['cross_entropy'])
            else:
                scheduler.step()
            gradients = dict(median=float(np.median(norms)), p95=float(np.quantile(norms, .95)),
                             max=max(norms), clipped_fraction=float(np.mean(np.asarray(norms) > config['training']['gradient_clip']))
                             if config['training']['gradient_clip'] else 0.)
            summary = dict(loss=loss_sum/seen, accuracy=correct/seen, parts={'modulation': loss_sum/seen},
                           samples=seen, optimizer_updates=len(norms), gradient_norm=gradients,
                           learning_rates={'model': used_lr}, next_lr=optimizer.param_groups[0]['lr'],
                           elapsed_seconds=time.monotonic()-started)
            metrics.update(epoch=epoch+1, step=global_step, best_accuracy=best)
            save_metrics(run, metrics)
            payload = dict(format='iqformer-c01-v1', model=model.state_dict(), config=config,
                           labels=train.labels, dataset_fingerprint=train.info['fingerprint'], source=identity,
                           optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state(),
                           epoch=epoch, epoch_complete=True, global_step=global_step, metrics=metrics,
                           best=best, patience_best=patience_best, bad_epochs=bad_epochs)
            save_checkpoint(run / 'last.pt', payload)
            if improved:
                save_checkpoint(run / 'best.pt', payload)
                write_json(run / 'best_metrics.json', metrics)
            event({'event': 'validation', 'epoch': epoch, 'step': global_step, 'score': score,
                   'best': best, 'improved': improved, 'complete': metrics['complete'],
                   'summary': {'classification_accuracy': {'modulation': score},
                               'classification_f1': {'modulation': {k: m[k] for k in ['macro_f1', 'weighted_f1']}}},
                   'groups': metrics['groups'], 'early_stopping': {'bad_validations': bad_epochs}}, metrics)
            for name, values in metrics['groups'].items():
                for key, value in values.items():
                    tb.scalar(f'validation/modulation_groups/{name}/{key}', value, global_step)
            event({'event': 'epoch', 'epoch': epoch, 'step': global_step,
                   'complete': seen == len(train)*config['data']['repeats_per_epoch'], 'training': summary})
            print(f'第{epoch+1}轮完成 | trainCE={loss_sum/seen:.4f} | valAcc={score:.4%} | '
                  f'best={best:.4%} | grad median={gradients["median"]:.3f}, p95={gradients["p95"]:.3f} | '
                  f'下轮lr={optimizer.param_groups[0]["lr"]:.3e} | {summary["elapsed_seconds"]:.1f}s', flush=True)
            invocation_epochs += 1
            if args.stop_after_epochs and invocation_epochs >= args.stop_after_epochs:
                print('按本次执行轮数限制停止；完整计划与调度未修改，可从last.pt续训。')
                break
            if config['training']['early_stopping_patience'] and bad_epochs >= config['training']['early_stopping_patience']:
                print('达到early stopping条件。')
                break
        print(f'训练结束 | bestAcc={best:.4%} | 检查点={run}/last.pt | 本轮最优若有更新保存在best.pt', flush=True)


if __name__ == '__main__':
    main()
