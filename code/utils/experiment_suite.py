"""Frozen, sequential experiments through the single public training CLI."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import torch
from utils.common import read_config_json, read_json, write_json
from utils.config import load_config
from utils.training_workflow import (ROOT, Pipeline, PipelineStop, absolute, digest,
                                     stamp, validate_devices)


def prepare_experiments(settings):
    if set(settings) != {'workflow', 'run', 'runtime', 'experiments'}:
        raise ValueError('Suite requires workflow/run/runtime/experiments')
    if settings['workflow'] != 'experiment_suite':
        raise ValueError('Expected experiment_suite')
    if set(settings['runtime']) != {'gpus'} or set(settings['run']) != {'output'}:
        raise ValueError('Suite runtime specifies allowed gpus; run specifies output')
    allowed = settings['runtime']['gpus'].split(',')
    validate_devices(settings['runtime']['gpus'], len(allowed), 'auto')
    if not isinstance(settings['experiments'], list) or not settings['experiments']:
        raise ValueError('Suite needs at least one experiment')
    configs = {}
    for entry in settings['experiments']:
        if set(entry) != {'name', 'config'}:
            raise ValueError('Each experiment requires name/config')
        name = entry['name']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_]+', name) or name in configs:
            raise ValueError('Experiment names must be unique plain identifiers')
        c = load_config(absolute(entry['config']))
        gpu, processes = c['runtime']['gpus'], c['runtime']['processes']
        cards = validate_devices(gpu, processes, c['runtime']['device'])
        if not set(cards).issubset(allowed):
            raise ValueError('Experiment GPU is outside suite runtime.gpus')
        if (c['training']['max_steps'] or c['checkpoint']['resume_from'] or
                c['evaluation']['early_stopping']['patience']):
            raise ValueError('Suite uses full fixed budgets; resume via the suite directory')
        if not c['checkpoint']['save_last'] or not c['checkpoint']['save_best']:
            raise ValueError('Suite needs last/best checkpoints')
        if c['evaluation']['max_batches']:
            raise ValueError('Suite requires the full fixed validation set')
        for key in ['data', 'codebook', 'init_checkpoint', 'warmstart_checkpoint']:
            if c['run'][key]:
                c['run'][key] = absolute(c['run'][key])
        configs[name] = c
    return configs


def write_comparison(directory, manifest):
    rows = []
    for name, stage in manifest['stages'].items():
        if not stage.get('complete'):
            continue
        checkpoint = torch.load(stage['best_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
        metrics = checkpoint['metrics']
        row = {'name': name, 'checkpoint': stage['best_checkpoint'],
               'checkpoint_sha256': digest(stage['best_checkpoint']),
               'epoch_1based': checkpoint['epoch'] + 1, 'step': checkpoint['global_step'],
               'metrics': metrics}
        mod = metrics.get('classification', {}).get('modulation')
        if mod:
            cm = mod['confusion_matrix']
            row['modulation_groups'] = {}
            for group, ids in [('first14', range(14)), ('last9', range(14, 23)),
                               ('qam', [4, 5, 6, 7, 12]), ('psk_related', [1, 2, 3, 8, 13])]:
                support = sum(sum(cm[i]) for i in ids)
                row['modulation_groups'][group] = {
                    'support': support, 'accuracy': sum(cm[i][i] for i in ids) / max(support, 1)}
        rows.append(row)
    write_json(directory / 'comparison.json', {
        'note': 'Matched supervised screening; single training seed, local development set, not official scores.',
        'completed': len(rows), 'experiments': rows})
    lines = ['# 独立精度实验', '', '各组依次执行；完整结果见comparison.json。', '',
             '| 实验 | 最佳轮次 | 调制Acc | Macro-F1 | 前14类 | QAM | 协议Acc | 本地代理 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m = row['metrics']
        if 'classification' not in m:
            continue
        mod, pro = m['classification']['modulation'], m['classification']['protocol']
        groups = row['modulation_groups']
        lines.append(f"| {row['name']} | {row['epoch_1based']} | {mod['accuracy']:.4%} | "
                     f"{mod['macro_f1']:.4%} | {groups['first14']['accuracy']:.4%} | "
                     f"{groups['qam']['accuracy']:.4%} | {pro['accuracy']:.4%} | {m['proxy_score']:.4f} |")
    lines += ['', '未完成组不作精度结论；best/last和全部七项参数误差保留在各组运行目录。']
    (directory / 'comparison.md').write_text('\n'.join(lines) + '\n')


def run_suite():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if 'LOCAL_RANK' in os.environ or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Start one suite coordinator; each config chooses its own process count')
    if args.resume:
        directory = args.resume.resolve()
        manifest = read_json(directory / 'manifest.json')
        if manifest.get('workflow') != 'experiment_suite':
            raise ValueError('Not an experiment suite')
        for relative, expected in read_json(directory / 'source_snapshot/sha256.json').items():
            if digest(Path(manifest['code_snapshot']) / relative) != expected:
                raise ValueError('Frozen source changed: ' + relative)
        for name, expected in manifest['config_hashes'].items():
            if digest(directory / (name + '_config.json')) != expected:
                raise ValueError('Frozen experiment config changed: ' + name)
        for path, expected in manifest['initialization_hashes'].items():
            if digest(path) != expected:
                raise ValueError('Frozen initialization changed: ' + path)
    else:
        settings = read_config_json(args.config)
        configs = prepare_experiments(settings)
        directory = Path(absolute(settings['run']['output'])) / stamp()
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / 'suite_config.json', settings)
        snapshot = directory / 'source_snapshot/code'
        shutil.copytree(ROOT / 'code', snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        write_json(directory / 'source_snapshot/sha256.json', {
            str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*') if p.is_file()})
        config_hashes, initializations = {}, {}
        for name, config in configs.items():
            config['run']['output'] = str(directory / name)
            path = directory / (name + '_config.json')
            write_json(path, config)
            config_hashes[name] = digest(path)
            for key in ['init_checkpoint', 'warmstart_checkpoint']:
                if config['run'][key]:
                    initializations[config['run'][key]] = digest(config['run'][key])
        manifest = {'workflow': 'experiment_suite', 'format_version': 1,
                    'gpu': settings['runtime']['gpus'], 'device': 'auto', 'processes': 1,
                    'code_snapshot': str(snapshot), 'stages': {}, 'order': list(configs),
                    'config_hashes': config_hashes, 'initialization_hashes': initializations,
                    'stream_console': False}
    pipeline = Pipeline(directory, manifest)
    pipeline.save()
    write_json(directory.parent / 'latest_suite.json', {'suite_directory': str(directory), 'pid': os.getpid()})
    pipeline.status('suite_starting', directory=str(directory), allowed_gpus=manifest['gpu'])
    try:
        for name in manifest['order']:
            config = read_json(directory / (name + '_config.json'))
            # No extra distributed communicator is needed for serial single-GPU
            # screening. Explicit per-stage devices remain checked above.
            pipeline.env['CUDA_VISIBLE_DEVICES'] = config['runtime']['gpus'] if config['runtime']['device'] != 'cpu' else ''
            pipeline.manifest['processes'] = config['runtime']['processes']
            pipeline.status('experiment_starting', experiment=name, gpus=config['runtime']['gpus'])
            pipeline.train(name, deepcopy(config))
            write_comparison(directory, manifest)
        pipeline.status('suite_complete', comparison=str(directory / 'comparison.json'))
    except PipelineStop as stop:
        pipeline.finish_stop(stop.signum)
    except Exception as error:
        pipeline.status('failed', error=repr(error))
        raise
