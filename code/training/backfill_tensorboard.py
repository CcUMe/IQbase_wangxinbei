"""Import or follow existing events.jsonl without touching checkpoints or training."""
from contextlib import ExitStack
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.tensorboard_log import TensorBoardLog
from utils.process_lifecycle import managed_entrypoint


def matching_metrics(run, event):
    """Only the latest matching validation snapshot can supply missing details."""
    try:
        metrics = json.loads((run / 'metrics.json').read_text())
    except (OSError, json.JSONDecodeError):
        return None
    summary = event.get('summary', {})
    if 'pretraining_validation_loss' in summary:
        return metrics if metrics.get('pretraining_validation_loss') == summary['pretraining_validation_loss'] else None
    if not summary.get('classification_accuracy'):
        return None
    for task, accuracy in summary['classification_accuracy'].items():
        values = metrics.get('classification', {}).get(task, {})
        if values.get('accuracy') != accuracy:
            return None
        if any(values.get(k) != v for k, v in summary.get('classification_f1', {}).get(task, {}).items()):
            return None
    for field, rmse in summary.get('parameter_rmse', {}).items():
        if metrics.get('parameters', {}).get(field, {}).get('rmse') != rmse:
            return None
    return metrics


def import_history(run, output=None, *, follow=False, poll_seconds=5.):
    run = Path(run).expanduser().resolve()
    source = run / 'events.jsonl'
    if not source.is_file():
        raise ValueError(f'Missing training events: {source}')
    output = Path(output).expanduser().resolve() if output else run / 'tensorboard_history'
    if output == run or output == run / 'tensorboard':
        raise ValueError('History import must use its own directory, distinct from the native tensorboard writer')
    if poll_seconds <= 0 or not poll_seconds < float('inf'):
        raise ValueError('poll_seconds must be finite and positive')
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'history_import.json'
    config = json.loads((run / 'resolved_config.json').read_text())
    # Replay every available source observation, without inventing missing steps.
    options = {**config.get('logging', {}).get('tensorboard', {}), 'enabled': True, 'log_every_steps': 1}
    with (output / '.history.lock').open('a') as lock, source.open('rb') as stream, ExitStack() as resources:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'Another history importer owns {output}') from exc
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['source'] != str(source):
                raise ValueError('History output belongs to a different source')
        else:
            if any(output.glob('events.out.tfevents.*')):
                raise ValueError('Existing TensorBoard files have no importer state; choose a separate empty --output')
            state = {'source': str(source), 'offset': 0, 'events': 0, 'prefix_sha256': ''}
        first = stream.readline(); stream.seek(0)
        prefix = hashlib.sha256(first).hexdigest() if first.endswith(b'\n') else ''
        if state.get('prefix_sha256') and prefix != state['prefix_sha256']:
            raise ValueError('Source events were replaced; choose a new history output')
        if state['offset'] > source.stat().st_size:
            raise ValueError('Source events were truncated; choose a new history output')
        if not follow and state['events'] and state['offset'] == source.stat().st_size:
            print(f'TensorBoard历史 | 无新增事件 | 累计={state["events"]} | {output}', flush=True)
            return {'output': str(output), 'new_events': 0, 'total_events': state['events']}
        stream.seek(state['offset'])
        gradient_clip = config.get('training', {}).get('gradient_clip', 0.)
        def writer(resume_step=None):
            return resources.enter_context(TensorBoardLog(run, options, log_dir=output,
                resume_step=resume_step, gradient_clip=gradient_clip))
        tb = writer()
        if not state['events']:
            tb.metadata(config)
        while True:
            count = 0
            while True:
                before = stream.tell(); line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    stream.seek(before)
                    break
                if not line.strip():
                    state['offset'] = stream.tell()
                    continue
                event = json.loads(line)
                if event.get('event') == 'start' and event.get('initialization', {}).get('resume'):
                    resumed = event.get('step')
                    if resumed is None:
                        resumed = (event.get('finetuning_state') or {}).get('next_step')
                    if resumed is not None:
                        tb.close(); tb = writer(resumed)
                    else:
                        print('旧恢复记录缺少global step：保留原始事件顺序，无法精确清理旧分支尾部。', file=sys.stderr)
                metrics = matching_metrics(run, event) if event.get('event') == 'validation' else None
                tb.record(event, metrics)
                state['offset'] = stream.tell(); state['events'] += 1; count += 1
            tb.flush()
            if not state['prefix_sha256']:
                position = stream.tell(); stream.seek(0); first = stream.readline(); stream.seek(position)
                if first.endswith(b'\n'):
                    state['prefix_sha256'] = hashlib.sha256(first).hexdigest()
            temporary = state_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
            temporary.replace(state_path)
            if count or not follow:
                print(f'TensorBoard历史 | 新增事件={count} | 累计={state["events"]} | {output}', flush=True)
            if not follow:
                return {'output': str(output), 'new_events': count, 'total_events': state['events']}
            time.sleep(poll_seconds)


@managed_entrypoint('TensorBoard历史补录')
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Exact timestamped training directory')
    parser.add_argument('--output', type=Path, help='Defaults to run/tensorboard_history, separate from native events')
    parser.add_argument('--follow', action='store_true', help='Tail a running old process without restarting training')
    parser.add_argument('--poll-seconds', type=float, default=5.)
    args = parser.parse_args()
    import_history(args.run, args.output, follow=args.follow, poll_seconds=args.poll_seconds)


if __name__ == '__main__':
    main()
