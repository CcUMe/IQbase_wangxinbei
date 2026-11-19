"""C01 composite formula, cross-task selection and safe legacy-resume migration."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from utils.common import TARGETS, save_checkpoint
from utils.config import DEFAULTS, load_config, resolve_config, resume_comparison_config
from utils.scoring import add_composite_score, selection_definition, selection_score
from utils.checkpoint_selection import migrate_best_selection


SCALER = {'mean': [0.] * 7, 'std': [float(i + 1) for i in range(7)]}


def metrics(mod=.70, pro=.97, ratio=.2):
    return add_composite_score({
        'classification': {'modulation': {'accuracy': mod, 'support': 100},
                           'protocol': {'accuracy': pro, 'support': 100}},
        'parameters': {name: {'rmse': ratio * scale, 'count': 10}
                       for name, scale in zip(TARGETS, SCALER['std'])}}, SCALER)


def checkpoint(weight, score_metrics, step, validation_step=None):
    config = deepcopy(DEFAULTS)
    config['evaluation']['selection_metric'] = 'modulation_accuracy'
    return {'model': {'weight': torch.tensor([float(weight)])},
            'model_config': {'width': 8}, 'training_config': config,
            'stage': 'supervised', 'dataset_fingerprint': 'unchanged-c01',
            'target_scaler': SCALER, 'global_step': step,
            'validation_global_step': step if validation_step is None else validation_step,
            'epoch': 0, 'epoch_complete': True,
            'metrics': score_metrics, 'best_score': .73,
            'early_stopping': {'best': .73, 'bad_validations': 5}}


class CompositeTests(unittest.TestCase):
    def test_preliminary_weights_and_training_scales(self):
        m = metrics(.8, .9, .2)
        self.assertAlmostEqual(m['parameter_proxy'], 1 / 1.2)
        self.assertAlmostEqual(m['proxy_score'], 30 * .8 + 30 * .9 + 40 / 1.2)
        self.assertEqual(m['local_composite']['reference_scales'], dict(zip(TARGETS, SCALER['std'])))
        self.assertAlmostEqual(sum(m['local_composite']['components'].values()), m['proxy_score'])
        self.assertFalse(m['local_composite']['official'])

    def test_mean_normalized_rmse_precedes_inverse_not_inverse_before_mean(self):
        m = metrics()
        ratios = [0, 1, 2, 3, 4, 5, 6]
        for j, name in enumerate(TARGETS):
            m['parameters'][name]['rmse'] = ratios[j] * SCALER['std'][j]
        add_composite_score(m, SCALER)
        self.assertEqual(m['parameter_proxy'], .25)

    def test_best_cannot_trade_large_parameter_loss_for_tiny_modulation_gain(self):
        balanced = metrics(.70, .97, .2)
        higher_mod_worse_params = metrics(.71, .965, .4)
        self.assertGreater(selection_score(balanced, 'proxy_score', 'supervised'),
                           selection_score(higher_mod_worse_params, 'auto', 'supervised'))
        with self.assertRaisesRegex(ValueError, 'three-task'):
            selection_score(balanced, 'modulation_accuracy', 'supervised')

    def test_missing_fields_and_single_task_prefix_cannot_select_three_task_best(self):
        for field in ['parameter', 'protocol']:
            m = metrics()
            if field == 'parameter':
                m['parameters']['frequency_shift'] = {'count': 0, 'rmse': None}
            else:
                m['classification']['protocol']['support'] = 0
            add_composite_score(m, SCALER)
            with self.assertRaisesRegex(ValueError, 'lacks a task'):
                selection_score(m, 'proxy_score', 'supervised')

    def test_true_zero_error_is_included_and_bad_values_rejected(self):
        self.assertEqual(metrics(ratio=0)['parameter_proxy'], 1.)
        for scale in [0., float('nan'), float('inf')]:
            scaler = deepcopy(SCALER)
            scaler['std'][0] = scale
            with self.assertRaises(ValueError):
                add_composite_score(metrics(), scaler)
        m = metrics()
        m['parameters']['symbol_rate']['rmse'] = float('nan')
        with self.assertRaises(FloatingPointError):
            add_composite_score(m, SCALER)

    def test_legacy_prediction_reports_use_support_instead_of_count(self):
        m = metrics()
        expected = m['proxy_score']
        for values in m['parameters'].values():
            values['support'] = values.pop('count')
        add_composite_score(m, SCALER)
        self.assertEqual(m['proxy_score'], expected)
        self.assertTrue(m['local_composite']['complete_task_coverage'])

    def test_pretraining_keeps_structure_target(self):
        m = {'pretraining_validation_loss': 7.2, 'pretraining_c01_first14_loss': 3.4}
        self.assertEqual(selection_score(m, 'auto', 'pretrain'), -7.2)
        self.assertEqual(selection_score(m, 'pretraining_c01_first14_loss', 'pretrain'), -3.4)

    def test_legacy_configs_upgrade_without_mutating_history(self):
        raw = deepcopy(DEFAULTS)
        raw['evaluation']['selection_metric'] = 'modulation_accuracy'
        self.assertEqual(resolve_config(raw)['evaluation']['selection_metric'], 'proxy_score')
        self.assertEqual(raw['evaluation']['selection_metric'], 'modulation_accuracy')
        current = deepcopy(raw)
        current['evaluation']['selection_metric'] = 'proxy_score'
        self.assertEqual(resume_comparison_config(raw), resume_comparison_config(current))
        current['optimizer']['learning_rate'] *= 2
        self.assertNotEqual(resume_comparison_config(raw), resume_comparison_config(current))

    def test_all_public_three_task_configs_select_composite(self):
        count = 0
        for path in (Path(__file__).resolve().parents[1] / 'configs').rglob('*'):
            if path.suffix not in ['.json', '.json5']:
                continue
            from utils.common import read_config_json
            raw = read_config_json(path)
            if raw.get('schema_version') != 2:
                continue
            config = load_config(path)
            if config['training']['stage'] == 'supervised':
                count += 1
                self.assertIn(config['evaluation']['selection_metric'], ['auto', 'proxy_score'], str(path))
        self.assertGreater(count, 20)

    def test_resume_reselects_available_weights_and_preserves_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            high_mod = checkpoint(1, metrics(.73, .95, .6), 1)
            balanced = checkpoint(2, metrics(.70, .98, .1), 2)
            stale = checkpoint(3, metrics(.99, .99, 0), 4, validation_step=3)
            for name, value in [('best.pt', high_mod), ('best_proxy.pt', balanced), ('last.pt', stale)]:
                save_checkpoint(path / name, value)
            original = hashlib.sha256((path / 'best.pt').read_bytes()).hexdigest()
            (path / 'resolved_config.json').write_text('original configuration')
            result = migrate_best_selection(path, stale, DEFAULTS['evaluation'], SCALER)
            chosen = torch.load(path / 'best.pt', weights_only=True)
            self.assertEqual(chosen['model']['weight'].item(), 2.)
            self.assertEqual(chosen['best_score'], balanced['metrics']['proxy_score'])
            self.assertEqual(chosen['selection']['metric'], 'proxy_score')
            self.assertIn('modulation', chosen['best_metrics'])
            self.assertEqual(chosen['early_stopping']['bad_validations'], 0)
            archive = Path(result['archive'])
            self.assertEqual(hashlib.sha256((archive / 'best.pt').read_bytes()).hexdigest(), original)
            (path / 'resolved_config.json').write_text('new configuration')
            self.assertEqual((archive / 'resolved_config.json').read_text(), 'original configuration')
            self.assertTrue(any(not c['eligible'] and 'mismatch' in c['reason'] for c in result['candidates']))
            self.assertIsNone(migrate_best_selection(path, chosen, DEFAULTS['evaluation'], SCALER))

    def test_resume_without_validated_weights_requests_fresh_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            stale = checkpoint(3, metrics(), 4, validation_step=3)
            stale['best_score'] = -float('inf')  # Old runs may stop before their first validation.
            save_checkpoint(path / 'last.pt', stale)
            result = migrate_best_selection(path, stale, DEFAULTS['evaluation'], SCALER)
            self.assertIsNone(result['best_score'])
            self.assertIsNone(result['selected_source'])

    def test_changed_score_weights_are_rebased_not_compared_to_old_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            old = checkpoint(1, metrics(), 1)
            old['selection'] = selection_definition('supervised', DEFAULTS['evaluation'])
            save_checkpoint(path / 'best.pt', old)
            evaluation = deepcopy(DEFAULTS['evaluation'])
            evaluation['proxy_task_weights'] = {'modulation': 20., 'protocol': 30., 'parameter': 50.}
            result = migrate_best_selection(path, old, evaluation, SCALER)
            self.assertAlmostEqual(result['best_score'], 20 * .70 + 30 * .97 + 50 / 1.2)

    def test_current_cli_resumes_legacy_selector_without_restarting_optimizer_or_schedule(self):
        root = Path(__file__).resolve().parents[2]
        pointer = root / 'artifacts/iterative_pipeline_checks/latest.json'
        if not pointer.is_file():
            self.skipTest('Run test_iterative_pipeline.py first to prepare the tiny real-data fixture')
        fixture = Path(json.loads(pointer.read_text())['directory'])
        base = json.loads((fixture / 'finetune.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            base['run'].update(output=str(directory / 'runs'), init_checkpoint='', warmstart_checkpoint='')
            base['checkpoint']['resume_from'] = ''
            base['training'].update(epochs=2, max_steps=1)
            base['data'].update(steps_per_epoch=2, batch_size=4)
            base['data']['loader']['num_workers'] = 0
            base['runtime'].update(device='cpu', gpus='', processes=1, threads=2, precision='fp32')
            base['logging']['tensorboard']['enabled'] = False
            base['evaluation'].update(max_batches=0, selection_metric='proxy_score')
            config = directory / 'config.json'
            def run(extra):
                config.write_text(json.dumps(base))
                result = subprocess.run([sys.executable, str(root / 'code/training/train.py'),
                    '--config', str(config), *extra], cwd=root, env={**os.environ, 'OMP_NUM_THREADS': '2'},
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout[-10000:])
            run([])
            run_directory = Path(json.loads((directory / 'runs/latest_run.json').read_text())['run_directory'])
            for name in ['last.pt', 'best.pt']:
                state = torch.load(run_directory / name, map_location='cpu', weights_only=True)
                state.pop('selection', None)
                state['training_config']['evaluation']['selection_metric'] = 'modulation_accuracy'
                state['best_score'] = state['metrics']['classification']['modulation']['accuracy']
                state['early_stopping'] = {'best': state['best_score'], 'bad_validations': 5}
                save_checkpoint(run_directory / name, state)
            original = hashlib.sha256((run_directory / 'best.pt').read_bytes()).hexdigest()
            base['training']['max_steps'] = 3
            run(['--resume', str(run_directory / 'last.pt')])
            last = torch.load(run_directory / 'last.pt', map_location='cpu', weights_only=True)
            best = torch.load(run_directory / 'best.pt', map_location='cpu', weights_only=True)
            self.assertEqual(last['global_step'], 3)
            self.assertEqual(last['scheduler']['last_epoch'], 3)
            self.assertEqual(best['selection']['metric'], 'proxy_score')
            self.assertAlmostEqual(best['best_score'], best['metrics']['proxy_score'])
            self.assertLess(last['early_stopping']['bad_validations'], 5)
            history = [json.loads(x) for x in (run_directory / 'events.jsonl').read_text().splitlines()]
            migrations = [x for x in history if x['event'] == 'selection_migration']
            self.assertEqual(len(migrations), 1)
            archive = Path(migrations[0]['archive'])
            self.assertEqual(hashlib.sha256((archive / 'best.pt').read_bytes()).hexdigest(), original)
            scores = [x['score'] for x in history if x['event'] == 'validation']
            self.assertAlmostEqual(best['best_score'], max(scores))


if __name__ == '__main__':
    unittest.main()
