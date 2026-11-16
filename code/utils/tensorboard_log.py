"""Rank-zero scalar summaries, shared by live training and JSONL history import."""
import json
import math
from numbers import Real
from pathlib import Path
import re
import sys


TENSORBOARD_DEFAULTS = dict(enabled=True, log_every_steps=0, flush_secs=30,
                            max_queue=10, classwise_metrics=True)


def tensorboard_options(options=None):
    if options is not None and (not isinstance(options, dict) or set(options) - set(TENSORBOARD_DEFAULTS)):
        raise ValueError('Unknown logging.tensorboard option')
    result = {**TENSORBOARD_DEFAULTS, **(options or {})}
    for key in ['enabled', 'classwise_metrics']:
        if type(result[key]) is not bool:
            raise ValueError(f'logging.tensorboard.{key} must be boolean')
    for key in ['log_every_steps', 'flush_secs', 'max_queue']:
        if type(result[key]) is not int or result[key] < (0 if key == 'log_every_steps' else 1):
            raise ValueError(f'Invalid logging.tensorboard.{key}')
    return result


def _summary_writer():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError('TensorBoard is enabled but unavailable. Install with '
                           '`python -m pip install tensorboard` or set logging.tensorboard.enabled=false') from exc
    return SummaryWriter


class TensorBoardLog:
    def __init__(self, run_directory, options=None, *, log_every=50, rank=0,
                 resume_step=None, gradient_clip=0., log_dir=None):
        self.options = tensorboard_options(options)
        self.interval = self.options['log_every_steps'] or log_every
        if type(self.interval) is not int or self.interval < 1:
            raise ValueError('TensorBoard log interval must be positive')
        self.directory = Path(log_dir) if log_dir is not None else Path(run_directory) / 'tensorboard'
        self.gradient_clip = gradient_clip
        self.writer = None
        if rank == 0 and self.options['enabled']:
            writer = _summary_writer()
            self.writer = writer(log_dir=str(self.directory), max_queue=self.options['max_queue'],
                                 flush_secs=self.options['flush_secs'],
                                 purge_step=None if resume_step is None else resume_step + 1)

    @property
    def enabled(self):
        return self.writer is not None

    def should_log_step(self, step):
        return self.enabled and (step == 1 or step % self.interval == 0)

    def scalar(self, tag, value, step):
        if self.enabled and isinstance(value, Real) and math.isfinite(value):
            self.writer.add_scalar(tag, float(value), int(step))

    def metadata(self, config, parameters=None, step=0):
        if not self.enabled:
            return
        self.writer.add_text('run/config', '```json\n' + json.dumps(config, ensure_ascii=False, indent=2) + '\n```', step)
        if parameters:
            for key, value in parameters.items():
                self.scalar('run/parameters/' + key, value, step)

    def record(self, event, metrics=None):
        if not self.enabled:
            return
        kind, step = event.get('event'), event.get('step', 0)
        if kind == 'start':
            for key in ['world_size', 'train_rows', 'validation_rows', 'effective_batch_size']:
                self.scalar('run/' + key, event.get(key), step)
            self.writer.add_text('run/stage', str(event.get('stage', 'unknown')), step)
        elif kind == 'step':
            if not self.should_log_step(step):
                return
            self.scalar('train/step_loss/total', event.get('loss'), step)
            parts = event.get('parts', {})
            active = event.get('active_parts')
            if active is None:
                # Old JSONL has zero placeholders for inactive tasks. Genuine
                # active zero losses are retained by the new explicit metadata.
                active = [key for key, value in parts.items() if value != 0]
            for key in active:
                self.scalar('train/step_loss/' + key, parts.get(key), step)
            for key in ['supcon_coverage', 'supcon_positives']:
                self.scalar('train/step_statistics/' + key, parts.get(key), step)
            for name, value in event.get('learning_rates', {}).items():
                self.scalar('train/learning_rate/' + name, value, step)
            norm = event.get('gradient_norm')
            self.scalar('train/gradient/norm_pre_clip', norm, step)
            if isinstance(norm, Real) and math.isfinite(norm):
                self.scalar('train/gradient/clipped', int(bool(self.gradient_clip and norm > self.gradient_clip)), step)
            self.scalar('train/epoch/number', event.get('epoch', 0) + 1, step)
        elif kind == 'epoch':
            summary = event.get('training', {})
            self.scalar('train/epoch_loss/total', summary.get('loss'), step)
            for name, value in summary.get('parts', {}).items():
                self.scalar('train/epoch_loss/' + name, value, step)
            for name, value in summary.get('gradient_norm', {}).items():
                self.scalar('train/epoch_gradient/' + name, value, step)
            for name in ['elapsed_seconds', 'optimizer_updates']:
                self.scalar('train/epoch_performance/' + name, summary.get(name), step)
            self.scalar('train/epoch/number', event.get('epoch', 0) + 1, step)
            self.scalar('train/epoch/complete', int(event.get('complete', False)), step)
            self.flush()
        elif kind == 'validation':
            self.validation(metrics if metrics is not None else self.summary_metrics(event.get('summary', {})), step)
            self.scalar('validation/selection/score', event.get('score'), step)
            self.scalar('validation/selection/best', event.get('best'), step)
            self.scalar('validation/selection/early_stop_bad_validations', event.get('early_stopping', {}).get('bad_validations'), step)
            self.flush()
        elif kind == 'finetuning_phase':
            self.scalar('run/finetuning/trainable_parameters', event.get('trainable_parameters'), step)
            self.writer.add_text('run/finetuning/phase', str(event.get('applied_phase', event.get('phase', 'unknown'))), step)

    @staticmethod
    def summary_metrics(summary):
        result = {'classification': {}, 'parameters': {}}
        for task, accuracy in summary.get('classification_accuracy', {}).items():
            result['classification'][task] = {'accuracy': accuracy, **summary.get('classification_f1', {}).get(task, {})}
        for name, value in summary.get('parameter_rmse', {}).items():
            result['parameters'][name] = {'rmse': value}
        if 'pretraining_validation_loss' in summary:
            result['pretraining_validation_loss'] = summary['pretraining_validation_loss']
        return result

    def validation(self, metrics, step):
        self.scalar('validation/pretraining/loss', metrics.get('pretraining_validation_loss'), step)
        self.scalar('validation/rows', metrics.get('rows'), step)
        for key in ['proxy_score', 'parameter_proxy']:
            self.scalar('validation/local_proxy/' + key, metrics.get(key), step)
        for task, values in metrics.get('classification', {}).items():
            for key in ['accuracy', 'micro_f1', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1',
                        'balanced_accuracy', 'cross_entropy', 'mean_confidence', 'support']:
                self.scalar(f'validation/{task}/{key}', values.get(key), step)
            if self.options['classwise_metrics']:
                for i, label in enumerate(values.get('labels', [])):
                    suffix = f'{i:02d}_' + re.sub(r'[^A-Za-z0-9_.+-]+', '_', label).strip('_')
                    for key in ['recall', 'precision', 'f1', 'class_support']:
                        items = values.get(key, [])
                        if i < len(items):
                            self.scalar(f'validation/{task}_class_{key}/{suffix}', items[i], step)
        for field, values in metrics.get('parameters', {}).items():
            for key in ['rmse', 'mae', 'absolute_error_p95', 'count']:
                self.scalar(f'validation/parameters/{field}/{key}', values.get(key), step)

    def flush(self):
        if self.enabled:
            self.writer.flush()

    def close(self):
        writer, self.writer = self.writer, None
        if writer is not None:
            writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        try:
            self.close()
        except Exception as exc:
            if exc_type is None:
                raise
            print(f'TensorBoard关闭失败，保留原异常：{exc}', file=sys.stderr)
