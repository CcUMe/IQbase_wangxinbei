"""Read real event files to verify scalar semantics, rollback and history import."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from training.backfill_tensorboard import import_history
from utils.config import DEFAULTS, resolve_config, resume_comparison_config
from utils.tensorboard_log import TensorBoardLog


def read(directory):
    return EventAccumulator(str(directory), size_guidance={'scalars': 0}).Reload()


def step(number, *, parts=None, active=None):
    result = {'event': 'step', 'step': number, 'epoch': 0, 'loss': 2.0,
              'parts': parts or {'modulation': 2., 'protocol': 0., 'parameters': 0.},
              'gradient_norm': 2., 'learning_rates': {'backbone_decay': .0003}}
    if active is not None:
        result['active_parts'] = active
    return result


class TensorBoardTests(unittest.TestCase):
    def test_actual_scalars_keep_active_zero_and_skip_inactive_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            with TensorBoardLog(directory, {'log_every_steps': 1}, gradient_clip=1.) as writer:
                writer.metadata({'training': {'stage': 'supervised'}}, {'final_parameters': 100})
                writer.record(step(1, active=['modulation']))
                writer.record(step(2, parts={'modulation': 0., 'protocol': 0.}, active=['modulation']))
                writer.record({'event': 'epoch', 'epoch': 0, 'step': 2, 'complete': True,
                               'training': {'loss': 1., 'parts': {'modulation': 1.},
                                            'gradient_norm': {'median': 2., 'p95': 3., 'clipped_fraction': .5}}})
                writer.record({'event': 'validation', 'step': 2, 'score': .5, 'best': .5}, {
                    'rows': 4, 'classification': {'modulation': {'accuracy': .5, 'macro_f1': .4,
                        'weighted_f1': .45, 'labels': ['Radar/Code'], 'recall': [.5],
                        'precision': [.6], 'f1': [.55], 'class_support': [4]}},
                    'parameters': {'frequency_shift': {'rmse': 2., 'mae': 1., 'count': 4}}})
            events = read(Path(directory) / 'tensorboard'); tags = events.Tags()['scalars']
            losses = events.Scalars('train/step_loss/modulation')
            self.assertEqual([x.step for x in losses], [1, 2]); self.assertEqual(losses[-1].value, 0.)
            self.assertNotIn('train/step_loss/protocol', tags)
            self.assertEqual(events.Scalars('train/gradient/clipped')[0].value, 1.)
            self.assertAlmostEqual(events.Scalars('validation/modulation/macro_f1')[0].value, .4, places=6)
            self.assertIn('validation/modulation_class_f1/00_Radar_Code', tags)
            self.assertEqual(events.Scalars('validation/parameters/frequency_shift/rmse')[0].value, 2.)
            self.assertEqual(events.Scalars('train/epoch_gradient/p95')[0].value, 3.)

    def test_disabled_and_nonzero_rank_never_import_or_create_writer(self):
        with tempfile.TemporaryDirectory() as root, patch('utils.tensorboard_log._summary_writer') as factory:
            for index, options, rank in [(0, {'enabled': False}, 0), (1, {'enabled': True}, 1)]:
                directory = Path(root) / str(index)
                with TensorBoardLog(directory, options, rank=rank) as writer:
                    writer.record(step(1)); writer.metadata({})
                    self.assertFalse(writer.enabled)
                self.assertFalse(directory.exists())
            factory.assert_not_called()

    def test_independent_interval_logs_first_and_selected_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            with TensorBoardLog(directory, {'log_every_steps': 5}) as writer:
                for number in range(1, 8): writer.record(step(number))
            self.assertEqual([x.step for x in read(Path(directory) / 'tensorboard').Scalars('train/step_loss/total')], [1, 5])

    def test_resume_purges_future_but_retains_checkpoint_step(self):
        with tempfile.TemporaryDirectory() as directory:
            options = {'log_every_steps': 1}
            with TensorBoardLog(directory, options) as writer:
                for number in [1, 2, 9]: writer.record(step(number))
            with TensorBoardLog(directory, options, resume_step=2) as writer:
                writer.record(step(3))
            values = read(Path(directory) / 'tensorboard').Scalars('train/step_loss/total')
            self.assertEqual([x.step for x in values], [1, 2, 3])

    def test_context_flushes_on_error_and_preserves_original_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'training failure'):
                with TensorBoardLog(directory) as writer:
                    writer.record(step(1)); raise ValueError('training failure')
            self.assertEqual(len(read(Path(directory) / 'tensorboard').Scalars('train/step_loss/total')), 1)

    def test_pretraining_has_component_and_validation_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            with TensorBoardLog(directory) as writer:
                writer.record(step(1, parts={'atomic': .2, 'emergent': .3, 'reconstruction': 0.},
                                   active=['atomic', 'emergent']))
                writer.record({'event': 'validation', 'step': 2, 'summary': {'pretraining_validation_loss': .4}})
            events = read(Path(directory) / 'tensorboard')
            self.assertAlmostEqual(events.Scalars('train/step_loss/atomic')[0].value, .2, places=6)
            self.assertAlmostEqual(events.Scalars('validation/pretraining/loss')[0].value, .4, places=6)
            self.assertNotIn('train/step_loss/reconstruction', events.Tags()['scalars'])

    def test_logging_controls_do_not_break_strict_resume_recipe(self):
        original = resolve_config(deepcopy(DEFAULTS)); changed = deepcopy(original)
        changed['logging']['tensorboard'].update(enabled=False, log_every_steps=7)
        self.assertEqual(resume_comparison_config(original), resume_comparison_config(resolve_config(changed)))
        for key, value in [('enabled', 1), ('flush_secs', 0), ('max_queue', 0),
                           ('log_every_steps', -1), ('classwise_metrics', 'yes')]:
            broken = deepcopy(original); broken['logging']['tensorboard'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'logging.tensorboard'):
                resolve_config(broken)

    def history(self, directory):
        run = Path(directory)
        (run / 'resolved_config.json').write_text(json.dumps(DEFAULTS))
        return run

    def test_history_import_is_incremental_and_waits_for_complete_line(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.history(directory); source = run / 'events.jsonl'
            validation = {'event': 'validation', 'step': 2, 'summary': {
                'classification_accuracy': {'modulation': .5},
                'classification_f1': {'modulation': {'macro_f1': .4, 'weighted_f1': .45}},
                'parameter_rmse': {'frequency_shift': 2.}}}
            metrics = {'classification': {'modulation': {'accuracy': .5, 'macro_f1': .4, 'weighted_f1': .45,
                       'labels': ['A'], 'recall': [.5], 'f1': [.4]}},
                       'parameters': {'frequency_shift': {'rmse': 2.}}}
            (run / 'metrics.json').write_text(json.dumps(metrics))
            source.write_text(json.dumps(step(1)) + '\n' + json.dumps(validation) + '\n')
            first = import_history(run); self.assertEqual(first['new_events'], 2)
            self.assertEqual(import_history(run)['new_events'], 0)
            with source.open('a') as stream: stream.write('{"event":"step","step":3')
            self.assertEqual(import_history(run)['new_events'], 0)
            with source.open('a') as stream: stream.write(',"loss":1,"parts":{"modulation":1}}\n')
            self.assertEqual(import_history(run)['new_events'], 1)
            events = read(run / 'tensorboard_history')
            self.assertEqual([x.step for x in events.Scalars('train/step_loss/total')], [1, 3])
            self.assertEqual([x.step for x in events.Scalars('validation/modulation/accuracy')], [2])
            self.assertIn('validation/modulation_class_f1/00_A', events.Tags()['scalars'])
            self.assertNotIn('train/step_loss/protocol', events.Tags()['scalars'])

    def test_history_resume_clears_abandoned_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.history(directory); source = run / 'events.jsonl'
            source.write_text(''.join(json.dumps(step(n)) + '\n' for n in [1, 3, 9]))
            import_history(run)
            with source.open('a') as stream:
                stream.write(json.dumps({'event': 'start', 'step': 1, 'initialization': {'resume': 'last.pt'}}) + '\n')
                stream.write(json.dumps(step(2)) + '\n')
            import_history(run)
            values = read(run / 'tensorboard_history').Scalars('train/step_loss/total')
            self.assertEqual([x.step for x in values], [1, 2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
