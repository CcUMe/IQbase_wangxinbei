"""C01 preliminary-round local composite, explicitly NOT an official score.

The task PDF (printed pp. 7-8) gives 30/30/40 task weights and
P=1/(1+RMSE/reference_scale), but does not disclose the fixed reference
scales or how seven parameter fields are aggregated. This implementation
uses training-side std per field, then 1/(1+mean(normalized RMSE)).
"""
import math

from utils.common import TARGETS


SCORE_VERSION = 'c01_local_composite_v1'
TASK_WEIGHTS = {'protocol': 30.0, 'modulation': 30.0, 'parameter': 40.0}
SCORE_NOTE = ('Local composite, NOT official: 30/30/40 default task weights; '
              'parameter_precision=1/(1+mean_j(RMSE_j/train_std_j)). '
              'Official reference scales and seven-field aggregation are undisclosed.')


def add_composite_score(metrics, scaler, weights=None):
    """Attach one auditable score to full or diagnostic subset metrics.

    Missing fields are excluded from the diagnostic parameter mean, as in
    the legacy proxy. Coverage is explicit; a missing task cannot silently
    become an eligible three-task best checkpoint.
    """
    weights = dict(TASK_WEIGHTS if weights is None else weights)
    if (set(weights) != set(TASK_WEIGHTS) or
            any(not math.isfinite(v) or v < 0 for v in weights.values()) or
            not math.isclose(sum(weights.values()), 100.0, abs_tol=1e-8)):
        raise ValueError('Composite task weights must be finite, nonnegative and sum to 100')
    scales = scaler['std']
    if len(scales) != len(TARGETS) or any(not math.isfinite(v) or v <= 0 for v in scales):
        raise ValueError('Composite normalization requires seven fixed positive training std values')
    normalized, missing = {}, []
    for name, scale in zip(TARGETS, scales):
        field = metrics['parameters'].get(name, {})
        value = field.get('rmse')
        if not field.get('count', field.get('support', 0)) or value is None:
            missing.append(name)
            continue
        if not math.isfinite(value) or value < 0:
            raise FloatingPointError(f'Invalid parameter RMSE: {name}={value}')
        normalized[name] = float(value / scale)
    ratio = math.fsum(normalized.values()) / len(normalized) if normalized else None
    parameter = 1.0 / (1.0 + ratio) if ratio is not None else 0.0
    components = {'parameter': weights['parameter'] * parameter}
    missing_tasks = []
    for task in ['protocol', 'modulation']:
        values = metrics['classification'][task]
        value = values['accuracy']
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise FloatingPointError(f'Invalid {task} accuracy: {value}')
        if not values.get('support', 0):
            missing_tasks.append(task)
        components[task] = weights[task] * value
    metrics.update(parameter_proxy=parameter, proxy_score=math.fsum(components.values()),
                   proxy_task_weights=weights, score_note=SCORE_NOTE,
                   local_composite={
                       'version': SCORE_VERSION, 'official': False,
                       'scale_source': 'training_target_scaler.std',
                       'reference_scales': dict(zip(TARGETS, map(float, scales))),
                       'parameter_aggregation': 'inverse_one_plus_mean_normalized_rmse',
                       'normalized_rmse': normalized, 'missing_parameter_fields': missing,
                       'missing_classification_tasks': missing_tasks,
                       'complete_task_coverage': not missing and not missing_tasks,
                       'task_weights': weights, 'components': components})
    return metrics


def selection_score(metrics, name, stage):
    """Primary best is composite for supervised/fine; structural loss for pretrain."""
    if stage == 'pretrain':
        name = 'pretraining_loss' if name == 'auto' else name
        if name not in ['pretraining_loss', 'pretraining_c01_first14_loss']:
            raise ValueError('Pretraining selects structural validation loss, not untrained task heads')
        return -metrics['pretraining_validation_loss' if name == 'pretraining_loss' else name]
    if name not in ['auto', 'proxy_score']:
        raise ValueError('Supervised/finetuning best.pt must select the three-task proxy_score')
    details = metrics.get('local_composite')
    if details and not details['complete_task_coverage']:
        raise ValueError('Cannot select a three-task best: validation lacks a task or parameter field; '
                         'use full/stratified validation, not a single-task prefix')
    score = float(metrics['proxy_score'])
    if not math.isfinite(score):
        raise FloatingPointError('Nonfinite local composite score')
    return score


def selection_definition(stage, evaluation):
    name = evaluation.get('selection_metric', 'auto')
    if stage == 'pretrain':
        return {'metric': 'pretraining_loss' if name == 'auto' else name}
    return {'metric': 'proxy_score', 'score_version': SCORE_VERSION,
            'task_weights': dict(evaluation.get('proxy_task_weights', TASK_WEIGHTS)),
            'official': False}
