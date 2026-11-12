"""Confusion-matrix F1 and readable console/log separation regression checks."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from utils.metrics import classification_metrics
from utils.readable_log import ReadableRunLog
from training.diagnose_tasks import MatchedSampler
from types import SimpleNamespace


class ReportingTests(unittest.TestCase):
    def test_f1_uses_false_positives_and_support_weights(self):
        m = classification_metrics([[3, 1], [2, 4]], ['a', 'b'])
        self.assertAlmostEqual(m['accuracy'], .7)
        self.assertAlmostEqual(m['f1'][0], 6 / 9)
        self.assertAlmostEqual(m['f1'][1], 8 / 11)
        self.assertAlmostEqual(m['weighted_f1'], ((6 / 9) * 4 + (8 / 11) * 6) / 10)
        z = classification_metrics([[0, 0], [1, 0]])
        self.assertEqual(z['macro_f1'], 0.)

    def test_log_mirrors_text_migrates_json_and_removes_tqdm_redraws(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = json.dumps({'event': 'step', 'loss': .2}) + '\n'
            (root / 'train.log').write_text(legacy)
            console = io.StringIO()
            with patch('sys.stdout', console), patch('sys.stderr', console):
                with ReadableRunLog(root):
                    print('训练总览')
                    sys.stdout.write('\rdiscarded progress\rfinal progress\n')
                    sys.stdout.log_snapshot('Epoch 1 | step 50 | loss=0.2')
                    print('F1=70%', file=sys.stderr)
            saved = (root / 'train.log').read_text()
            self.assertIn('训练总览', saved)
            self.assertIn('F1=70%', saved)
            self.assertIn('final progress', saved)
            self.assertNotIn('discarded progress', saved)
            self.assertNotIn('\r', saved)
            self.assertEqual((root / 'train_legacy.jsonl').read_text(), legacy)

    def test_single_and_joint_have_identical_modulation_streams(self):
        dataset = SimpleNamespace(indices=np.arange(12), meta={
            'modulation': np.array([0] * 4 + [-1] * 8), 'protocol': np.array([-1] * 4 + [0] * 4 + [-1] * 4),
            'target_valid': np.array([[False] * 7] * 8 + [[True] * 7] * 4),
            'length': np.array([512, 1024] * 6)})
        joint = list(MatchedSampler(dataset, 5, 3, 2026, 'joint'))
        single = list(MatchedSampler(dataset, 5, 3, 2026, 'modulation_only'))
        self.assertEqual(joint[::3], single)


if __name__ == '__main__':
    unittest.main(verbosity=2)
