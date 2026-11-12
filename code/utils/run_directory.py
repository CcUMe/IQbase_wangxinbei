"""Timestamped training runs; rank zero chooses one shared directory for DDP."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.common import read_json, write_json


def latest_run(directory):
    """Resolve a new run root, or a legacy directory containing last.pt."""
    directory = Path(directory).expanduser().resolve()
    pointer = directory / 'latest_run.json'
    if pointer.is_file():
        run = Path(read_json(pointer)['run_directory']).resolve()
        if run.parent != directory or not run.is_dir():
            raise ValueError(f'Invalid run directory in {pointer}')
        return run
    if (directory / 'last.pt').is_file():
        return directory
    raise FileNotFoundError(f'No training run found in {directory}')


def _choose_directory(output, resume):
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    if resume:
        checkpoint = Path(resume).expanduser().resolve(strict=True)
        if not checkpoint.is_file():
            raise ValueError(f'Resume checkpoint is not a file: {checkpoint}')
        run = checkpoint.parent
        info_path = run / 'run_info.json'
        info = read_json(info_path) if info_path.exists() else {
            'run_directory': str(run), 'output_root': str(run.parent),
            'created_at': None, 'timezone': 'Asia/Shanghai', 'resume_history': []}
        allowed = {run, run.parent, Path(info['output_root']).resolve()}
        if output is not None and Path(output).expanduser().resolve() not in allowed:
            raise ValueError('--output must be the original run directory or its parent when resuming; '
                             'omit --output to use the checkpoint directory automatically')
        info.setdefault('resume_history', []).append({'resumed_at': now.isoformat(timespec='seconds'),
                                                     'checkpoint': str(checkpoint)})
        write_json(info_path, info)
        return run
    if output is None:
        raise ValueError('--output is required for a new training run')
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime('%Y%m%d_%H%M%S_%f')
    for suffix in range(1000):
        run = root / (stamp if suffix == 0 else f'{stamp}_{suffix:03d}')
        try:
            run.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError(f'Cannot create a unique timestamped run in {root}')
    info = {'run_directory': str(run), 'output_root': str(root),
            'created_at': now.isoformat(timespec='microseconds'), 'timezone': 'Asia/Shanghai', 'resume_history': []}
    write_json(run / 'run_info.json', info)
    # Atomic replacement; concurrent launches cannot produce a partial pointer.
    temporary = root / f'.latest_run_{run.name}.json.tmp'
    write_json(temporary, {key: info[key] for key in ['run_directory', 'created_at', 'timezone']})
    temporary.replace(root / 'latest_run.json')
    return run


def training_run_directory(output, resume=None, rank=0, world=1):
    result = [None]
    if rank == 0:
        try:
            result[0] = {'directory': str(_choose_directory(output, resume))}
        except (OSError, ValueError, KeyError) as exc:
            result[0] = {'error': str(exc)}
    if world > 1:
        import torch.distributed as dist
        dist.broadcast_object_list(result, src=0)
    if 'error' in result[0]:
        raise ValueError(result[0]['error'])
    return Path(result[0]['directory'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print the latest timestamped training directory.')
    parser.add_argument('--latest', type=Path, required=True)
    print(latest_run(parser.parse_args().latest))
