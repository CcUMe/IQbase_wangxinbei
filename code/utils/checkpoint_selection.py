"""Rebase checkpoint selection on resume without mixing score units or deleting history."""
from copy import deepcopy
from datetime import datetime
import math
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import torch

from utils.common import save_checkpoint, write_json
from utils.scoring import TASK_WEIGHTS, add_composite_score, selection_definition, selection_score


def saved_selection_definition(checkpoint):
    if checkpoint.get('selection'):
        return checkpoint['selection']
    evaluation = checkpoint.get('training_config', {}).get('evaluation', {})
    name = evaluation.get('selection_metric', 'auto')
    if checkpoint.get('trainer') == 'shift_consistent_v1':
        name = 'modulation_accuracy'  # That historical trainer overrode its inherited config.
    return {'metric': 'proxy_score' if name == 'auto' else name,
            'task_weights': evaluation.get('proxy_task_weights', TASK_WEIGHTS)}


def migrate_best_selection(directory, previous, evaluation, scaler):
    """Only called when explicitly resuming with the current training entrypoint.

    Keep model/AdamW/LR state at the resumed step; reselect best from available
    validated weights, not from a historical metric for which weights are gone.
    Old primary files are preserved by hard link (copy fallback) before updates.
    """
    target = selection_definition('supervised', evaluation)
    source = saved_selection_definition(previous)
    if (source['metric'] == target['metric'] and
            source.get('score_version', target['score_version']) == target['score_version'] and
            source.get('task_weights', TASK_WEIGHTS) == target['task_weights']):
        return None
    directory = Path(directory)
    archive = directory / 'selection_history' / datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    archive.mkdir(parents=True, exist_ok=False)
    for name in ['best.pt', 'best_proxy.pt', 'resolved_config.json']:
        path = directory / name
        if path.is_file():
            if path.suffix != '.pt':
                shutil.copy2(path, archive / name)
                continue
            try:
                os.link(path, archive / name)
            except OSError:
                shutil.copy2(path, archive / name)
    candidates, winner, winner_score, winner_path = [], None, -math.inf, None
    for path in sorted(directory.glob('*.pt')):
        if path.name.endswith('.tmp.pt'):
            continue
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            if (checkpoint.get('stage') != 'supervised' or
                    checkpoint.get('dataset_fingerprint') != previous['dataset_fingerprint'] or
                    checkpoint.get('model_config') != previous['model_config'] or
                    checkpoint.get('target_scaler') != scaler):
                continue
            # Periodic last/step snapshots can contain weights newer than their
            # last metrics. Such stale metrics must never select those weights.
            validated = checkpoint.get('validation_global_step', checkpoint.get('metrics_validation_step'))
            if validated is not None:
                aligned = validated == checkpoint['global_step']
            else:
                aligned = path.name.startswith('best')
            if not aligned:
                candidates.append({'path': str(path), 'eligible': False, 'reason': 'metrics/weights step mismatch'})
                continue
            metrics = deepcopy(checkpoint.get('metrics', {}))
            if 'classification' not in metrics or 'parameters' not in metrics:
                continue
            add_composite_score(metrics, scaler, target['task_weights'])
            score = selection_score(metrics, 'proxy_score', 'supervised')
            candidates.append({'path': str(path), 'eligible': True, 'score': score,
                               'step': checkpoint['global_step']})
            if score > winner_score:
                winner, winner_score, winner_path = checkpoint, score, path
                winner['metrics'] = metrics
        except (ValueError, KeyError, FloatingPointError) as error:
            candidates.append({'path': str(path), 'eligible': False, 'reason': str(error)})
    report = {'previous_selection': source, 'selection': target, 'archive': str(archive),
              'candidates': candidates, 'selected_source': str(winner_path) if winner_path else None,
              'best_score': winner_score if winner is not None else None,
              'historical_best_score': previous['best_score'] if math.isfinite(previous['best_score']) else None,
              'note': 'Available validated weights only; missing historical epochs cannot be reconstructed.'}
    write_json(archive / 'migration.json', report)
    if winner is not None:
        winner['selection'] = target
        winner['selection_migration'] = str(archive / 'migration.json')
        winner['best_score'] = winner_score
        winner['best_metrics'] = {
            'modulation': winner['metrics']['classification']['modulation']['accuracy'],
            **winner.get('best_metrics', {}), 'proxy': winner_score}
        winner['early_stopping'] = {'best': winner_score, 'bad_validations': 0}
        winner['validation_global_step'] = winner['global_step']
        winner['training_config'] = deepcopy(winner['training_config'])
        winner['training_config']['evaluation'].update(
            selection_metric='proxy_score', proxy_task_weights=target['task_weights'])
        save_checkpoint(directory / 'best.pt', winner)
        if (directory / 'best_proxy.pt').is_file():
            save_checkpoint(directory / 'best_proxy.pt', winner)
    return report
