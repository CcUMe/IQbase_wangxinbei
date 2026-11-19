"""Device restrictions and immutable-budget validation for sequential suites."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.common import write_json
from utils.config import DEFAULTS
from utils.experiment_suite import prepare_experiments
from utils.training_workflow import training_budget_complete
from utils.metrics import classification_group_metrics


class SuiteValidationTests(unittest.TestCase):
    def settings(self, path):
        return {'workflow': 'experiment_suite', 'run': {'output': 'unused'},
                'runtime': {'gpus': '2,3'}, 'experiments': [{'name': 'candidate', 'config': str(path)}]}

    def test_device_and_fixed_budget_restrictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            config = deepcopy(DEFAULTS)
            config['runtime'].update(gpus='2', processes=1)
            config['run']['data'] = directory
            write_json(path, config)
            self.assertEqual(prepare_experiments(self.settings(path))['candidate']['runtime']['gpus'], '2')
            config['runtime']['gpus'] = '4'
            write_json(path, config)
            with self.assertRaisesRegex(ValueError, 'outside'):
                prepare_experiments(self.settings(path))
            config['runtime'].update(gpus='2,3', processes=2)
            write_json(path, config)
            self.assertEqual(prepare_experiments(self.settings(path))['candidate']['runtime']['processes'], 2)
            capped=deepcopy(config);capped['training']['max_steps']=3
            write_json(path,capped)
            self.assertEqual(prepare_experiments(self.settings(path))['candidate']['training']['max_steps'],3)
            for branch, key, value in [('checkpoint','resume_from','/unused'),('evaluation', 'max_batches', 1)]:
                changed = deepcopy(config); changed[branch][key] = value
                write_json(path, changed)
                with self.assertRaises(ValueError):
                    prepare_experiments(self.settings(path))

    def test_stage_name_cannot_escape_output(self):
        settings = self.settings('/unused')
        settings['experiments'][0]['name'] = '../outside'
        with self.assertRaisesRegex(ValueError, 'identifiers'):
            prepare_experiments(settings)

    def test_capped_budget_preserves_full_epoch_plan(self):
        config={'training':{'epochs':20,'max_steps':10}}
        self.assertFalse(training_budget_complete(config,{'global_step':9,'epoch':1,'epoch_complete':False}))
        self.assertTrue(training_budget_complete(config,{'global_step':10,'epoch':1,'epoch_complete':False}))
        self.assertEqual(config['training']['epochs'],20)
        config['training']['max_steps']=0
        self.assertFalse(training_budget_complete(config,{'global_step':10,'epoch':1,'epoch_complete':True}))
        self.assertTrue(training_budget_complete(config,{'global_step':100,'epoch':19,'epoch_complete':True}))

    def test_group_metrics_keep_outside_predictions_as_errors(self):
        metrics=classification_group_metrics([[2,0,2],[1,3,0],[0,0,8]],[0,1])
        self.assertEqual(metrics['support'],8)
        self.assertAlmostEqual(metrics['accuracy'],5/8)
        self.assertAlmostEqual(metrics['macro_f1'],(4/7+6/7)/2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
