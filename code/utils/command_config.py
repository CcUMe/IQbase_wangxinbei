"""Optional grouped JSON defaults for data preparation and vocabulary commands."""
import argparse
import os
from pathlib import Path
import sys

from utils.common import read_config_json


def command_defaults(parser, mapping, argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--config')
    known, _ = preliminary.parse_known_args(argv)
    if not known.config:
        return
    raw = read_config_json(known.config)
    defaults = {}
    for group, options in raw.items():
        if group not in mapping or not isinstance(options, dict):
            parser.error(f'Unknown or invalid configuration group: {group}')
        for name, value in options.items():
            if name not in mapping[group]:
                parser.error(f'Unknown configuration field: {group}.{name}')
            defaults[mapping[group][name]] = value
    for action in parser._actions:
        if action.type is Path and defaults.get(action.dest, object()) in [None, '']:
            defaults.pop(action.dest, None)
    parser.set_defaults(**defaults)
    # argparse does not consider a configured default sufficient for required=True.
    for action in parser._actions:
        if action.dest in defaults and defaults[action.dest] not in [None, '']:
            action.required = False


def checkpoint_reference(value):
    """Resolve exact weights, completed two-round workflows, or a run parent.

    Pending round2 is never silently replaced with round1. Follow completion
    pointers before active-workflow pointers, and reject cyclic pointers.
    """
    from utils.common import read_json
    from utils.run_directory import latest_run
    visited=set()
    def resolve(reference):
        path=Path(reference).expanduser().resolve()
        if path in visited:raise ValueError('Cyclic checkpoint reference: '+str(path))
        visited.add(path)
        if path.is_file():return path
        if (path/'best.pt').is_file():return path/'best.pt'
        if (path/'manifest.json').is_file():
            stages=read_json(path/'manifest.json')['stages']
            name='finetune' if 'finetune' in stages else 'pretrain_round2'
            stage=stages.get(name,{})
            if not stage.get('complete'):
                raise ValueError('流程尚未完成 '+name+'；如需中间权重，请显式指定对应.pt文件：'+str(path))
            checkpoint=Path(stage['best_checkpoint']).expanduser().resolve()
            if not checkpoint.is_file():raise ValueError('Checkpoint does not exist: '+str(checkpoint))
            return checkpoint
        for name in ['latest_pretrained.json','latest_pipeline.json']:
            if (path/name).is_file():
                pointer=read_json(path/name)
                checkpoint=resolve(pointer['pipeline_directory'])
                if name=='latest_pretrained.json' and checkpoint!=Path(pointer['checkpoint']).resolve():
                    raise ValueError('Completed-pretraining pointer does not match its pipeline')
                return checkpoint
        try:checkpoint=latest_run(path)/'best.pt'
        except FileNotFoundError:raise ValueError('Checkpoint does not exist: '+str(path)) from None
        if not checkpoint.is_file():raise ValueError('Checkpoint does not exist: '+str(checkpoint))
        return checkpoint
    return resolve(value)


def configured_launch(gpus='', processes=1, device='auto', distributed=False):
    """Select devices before CUDA initialization; direct training may launch torchrun."""
    if 'LOCAL_RANK' in os.environ or int(os.environ.get('WORLD_SIZE', '1')) > 1:
        return  # An explicit external torchrun command owns its GPU assignment.
    if type(processes) is not int or processes not in [1, 2]:
        raise ValueError('runtime.processes must be 1 or 2')
    if not isinstance(gpus, str):
        raise ValueError('runtime.gpus must be a comma-separated string, e.g. "2,3"')
    visible = [x.strip() for x in gpus.split(',')] if gpus else []
    if visible and (any(not x.isdigit() for x in visible) or len(set(visible)) != len(visible)):
        raise ValueError('runtime.gpus must contain distinct GPU indices')
    if processes == 1 and len(visible) > 1:
        raise ValueError('One process requires one GPU index; use processes=2 for two GPUs')
    if processes == 2:
        if not distributed:
            raise ValueError('This command supports one process only')
        if device != 'cpu' and (len(visible) != 2 or set(visible) not in [{'0','1'}, {'2','3'}, {'4','5'}]):
            raise ValueError('Two-GPU training requires gpus="0,1", "2,3" or "4,5"')
    if visible:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(visible)
    if processes > 1:
        # Supported by this host's torchrun; preserve explicit user overrides.
        os.environ.setdefault('TORCH_ELASTIC_SHUTDOWN_TIMEOUT', '10')
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   f'--nproc_per_node={processes}', *sys.argv]
        # Replace this wrapper instead of creating an extra launcher parent.
        # IDE stop / terminal SIGINT now reaches the torchrun agent directly.
        os.execvpe(sys.executable, command, os.environ.copy())
