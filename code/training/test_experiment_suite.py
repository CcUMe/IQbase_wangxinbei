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
            for branch, key, value in [('training', 'max_steps', 3), ('evaluation', 'max_batches', 1)]:
                changed = deepcopy(config); changed[branch][key] = value
                write_json(path, changed)
                with self.assertRaises(ValueError):
                    prepare_experiments(self.settings(path))

    def test_stage_name_cannot_escape_output(self):
        settings = self.settings('/unused')
        settings['experiments'][0]['name'] = '../outside'
        with self.assertRaisesRegex(ValueError, 'identifiers'):
            prepare_experiments(settings)


if __name__ == '__main__':
    unittest.main(verbosity=2)
