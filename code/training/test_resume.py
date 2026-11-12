"""Resume initialization precedence and strict training-config compatibility."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.train import configure_checkpoint_start
from utils.config import DEFAULTS, resolve_config, resume_comparison_config


class ResumeConfigurationTests(unittest.TestCase):
    def test_config_resume_and_explicit_initialization_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / 'last.pt'
            saved.touch()
            alternate = Path(directory) / 'other.pt'
            alternate.touch()
            for override, expected in [({}, saved), ({'resume': alternate}, alternate),
                ({'no_resume': True}, None), ({'init_checkpoint': alternate}, None),
                ({'warmstart_checkpoint': alternate}, None)]:
                config = deepcopy(DEFAULTS)
                config['checkpoint']['resume_from'] = str(saved)
                args = argparse.Namespace(resume=None, no_resume=False, init_checkpoint=None,
                                          warmstart_checkpoint=None)
                for key, value in override.items():
                    setattr(args, key, value)
                configure_checkpoint_start(args, config)
                self.assertEqual(args.resume, expected)
                self.assertEqual(config['checkpoint']['resume_from'], str(expected) if expected else '')

    def test_saving_changes_are_allowed_but_training_changes_still_rejected(self):
        original = deepcopy(DEFAULTS)
        loading = deepcopy(DEFAULTS)
        loading['checkpoint'].update(resume_from='some/run/last.pt', save_every_steps=20,
                                     save_every_epochs=2, save_best=False)
        loading['runtime'].update(progress_bar=False, progress_min_interval=2.)
        loading['training']['max_steps'] = 1000
        self.assertEqual(resume_comparison_config(original), resume_comparison_config(loading))
        self.assertEqual(original['checkpoint']['resume_from'], '')
        loading['data']['batch_size'] *= 2
        self.assertNotEqual(resume_comparison_config(original), resume_comparison_config(loading))
        # Checkpoints saved before this feature normalize to the new defaults.
        legacy = deepcopy(original)
        del legacy['checkpoint']['resume_from']
        del legacy['checkpoint']['save_every_steps']
        self.assertEqual(resolve_config(legacy), original)

    def test_invalid_paths_and_resume_options_fail_early(self):
        args = argparse.Namespace(resume=None, no_resume=False, init_checkpoint=None, warmstart_checkpoint=None)
        with tempfile.TemporaryDirectory() as directory:
            config = deepcopy(DEFAULTS)
            config['checkpoint']['resume_from'] = str(Path(directory) / 'missing.pt')
            with self.assertRaisesRegex(ValueError, 'does not exist'):
                configure_checkpoint_start(args, config)
        for field, value in [('resume_from', None), ('resume_from', 1), ('save_every_steps', -1),
                              ('save_every_steps', 1.5)]:
            config = deepcopy(DEFAULTS)
            config['checkpoint'][field] = value
            with self.assertRaises(ValueError):
                resolve_config(config)


if __name__ == '__main__':
    unittest.main(verbosity=2)
