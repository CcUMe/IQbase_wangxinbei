"""Functional and parity checks; CPU only, no benchmark accuracy claims."""
import ast
import hashlib
import sys
import unittest
import warnings
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[1] / 'code'))

import numpy as np
import torch
from scipy.signal import stft
from torch.utils.data import Dataset

from c01.data import C01Modulation, LengthBatches, SignalViews
from c01.runtime import build_model, load_config
from utils.common import read_json


class C01Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = load_config(BASE / 'c01.json5')

    def test_official_source_only_three_edits(self):
        info = read_json(BASE / 'source_info.json')
        source = BASE / 'upstream/model/IQFormer.py'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), info['official_model_sha256'])
        expected = source.read_text().replace('from timm.models.layers import DropPath, trunc_normal_',
                                              'from .compat import DropPath, trunc_normal_')
        expected = expected.replace('from timm.models.registry import register_model',
                                     'from .compat import register_model')
        expected = expected.replace('torch.squeeze(self.patch_embedSTFT(stft))',
                                     'self.patch_embedSTFT(stft).squeeze(2)')
        self.assertEqual(expected, (BASE / 'c01/official_model.py').read_text())

    def test_views_exactly_match_official_dataset(self):
        tree = ast.parse((BASE / 'upstream/dataset/dataset.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RMLgeneral')
        namespace = dict(Dataset=Dataset, np=np, torch=torch, stft=stft)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<upstream RMLgeneral>', 'exec'), namespace)
        for length in [512, 1024]:
            raw = np.random.default_rng(length).normal(size=(2, length, 2)).astype(np.float32)
            dataset = namespace['RMLgeneral'](raw.transpose(0, 2, 1), [0, 1], [0, 0])
            adapted = SignalViews(self.config['input'])([
                {'iq': raw[i], 'sample_id': str(i), 'target': i} for i in range(2)])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for i in range(2):
                    iq, spectrum, _, _ = dataset[i]
                    torch.testing.assert_close(adapted['iq'][i], iq, rtol=0, atol=0)
                    torch.testing.assert_close(adapted['stft'][i], spectrum, rtol=0, atol=0)
            self.assertEqual(tuple(adapted['stft'].shape), (2, 1, 32, length))

    def test_model_parity_batch_one_and_backward(self):
        source = (BASE / 'upstream/model/IQFormer.py').read_text()
        source = source.replace('from timm.models.layers import DropPath, trunc_normal_',
                                'from c01.compat import DropPath, trunc_normal_')
        source = source.replace('from timm.models.registry import register_model',
                                'from c01.compat import register_model')
        namespace = {}
        exec(compile(source, '<upstream IQFormer>', 'exec'), namespace)
        torch.manual_seed(2026)
        reference = namespace['IQFormer'](**self.config['model'], num_classes=23, fork_feat=False).eval()
        torch.manual_seed(2026)
        adapted = build_model(self.config, list(range(23))).eval()
        for key, tensor in reference.state_dict().items():
            torch.testing.assert_close(tensor, adapted.state_dict()[key], rtol=0, atol=0)
        for length in [512, 1024]:
            batch = SignalViews(self.config['input'])([
                {'iq': np.random.default_rng(i).normal(size=(length, 2)).astype(np.float32),
                 'sample_id': str(i)} for i in range(2)])
            with torch.inference_mode():
                original = reference(batch['iq'], batch['stft'])
                actual = adapted(batch['iq'], batch['stft'])
                torch.testing.assert_close(actual, original, rtol=0, atol=0)
                single = adapted(batch['iq'][:1], batch['stft'][:1])
                self.assertEqual(tuple(single.shape), (1, 23))
                torch.testing.assert_close(single[0], actual[0], rtol=1e-4, atol=1e-5)
        adapted.train()
        logits = adapted(batch['iq'], batch['stft'])
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        for name in ['patch_embedIQ.0.weight', 'patch_embedSTFT.0.weight', 'patch_LSTM.weight_ih_l0', 'head.weight']:
            gradient = dict(adapted.named_parameters())[name].grad
            self.assertTrue(bool(torch.isfinite(gradient).all()), name)
            self.assertGreater(float(gradient.norm()), 0, name)

    def test_sampler_complete_no_padding(self):
        lengths = np.array([512] * 5 + [1024] * 7)
        sampler = LengthBatches(lengths, 3, True, 2026, repeats=2)
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual(sorted(i for batch in batches for i in batch), sorted(list(range(12)) * 2))
        self.assertTrue(all(len(np.unique(lengths[batch])) == 1 for batch in batches))
        self.assertEqual(batches, list(sampler))
        sampler.epoch = 1
        self.assertNotEqual(batches, list(sampler))

    def test_real_c01_split_and_views(self):
        train = C01Modulation(self.config['run']['data'], 'train')
        val = C01Modulation(self.config['run']['data'], 'validation')
        self.assertEqual(len(train), 186113)
        self.assertEqual(len(val), 20885)
        self.assertEqual(len(train.labels), 23)
        self.assertEqual(len(np.intersect1d(train.rows, val.rows)), 0)
        for length in [512, 1024]:
            row = val[int(np.flatnonzero(val.lengths == length)[0])]
            batch = SignalViews(self.config['input'])([row])
            self.assertEqual(tuple(batch['iq'].shape), (1, 2, length))
            self.assertEqual(tuple(batch['stft'].shape), (1, 1, 32, length))
            torch.testing.assert_close(batch['iq'][0].T, torch.from_numpy(row['iq']), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
