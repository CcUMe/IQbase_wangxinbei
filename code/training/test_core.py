"""Focused regression checks for data/loss leakage, BPE and export semantics.

Uses Python unittest, so no pytest dependency is required.
"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from models.rf_model import RFModel
from utils.common import load_safetensors, save_safetensors, runtime
from utils.engine import patch_mask, supervised_loss
from utils.structure import StructuralTokenizer, learn_merges
from utils.dataset import BalancedBatchSampler


def fixture():
    schema = {'classification_labels': {'modulation': ['a', 'b'], 'protocol': ['x', 'y']}}
    scaler = {'mean': [0.] * 7, 'std': [1.] * 7}
    config = dict(width=32, depth=1, heads=4, patch_size=16, stft_window=64,
                  dropout=0., phase_features=True, pretrain_heads=True)
    return RFModel(config, schema, scaler, [[] for _ in range(7)]).eval(), scaler


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_masked_targets_cannot_leak_through_either_stream(self):
        model, _ = fixture()
        x = torch.randn(2, 512, 2)
        lengths = torch.tensor([512, 512])
        mask = patch_mask(lengths, 16, .5, 3, 12)
        changed = x.clone()
        changed[mask.repeat_interleave(16, 1)] += 1000
        with torch.no_grad():
            a = model(x, lengths, patch_mask=mask, return_pretrain=True)
            b = model(changed, lengths, patch_mask=mask, return_pretrain=True)
        for key in ['tokens', 'modulation', 'protocol', 'parameters']:
            self.assertTrue(torch.equal(a[key], b[key]), key)
        self.assertFalse(torch.equal(a['reconstruction_target'], b['reconstruction_target']))

    def test_invalid_nan_targets_are_ignored_but_real_zero_is_scored(self):
        output = {'modulation': torch.zeros(2, 2, requires_grad=True), 'protocol': torch.zeros(2, 2, requires_grad=True),
                  'parameters': torch.ones(2, 7, requires_grad=True), 'grid_logits': {}}
        targets = torch.full((2, 7), float('nan'))
        targets[0, 2] = 0
        valid = torch.zeros(2, 7, dtype=torch.bool)
        valid[0, 2] = True
        batch = {'iq': torch.zeros(2, 512, 2), 'modulation': torch.tensor([-1, -1]),
                 'protocol': torch.tensor([-1, -1]), 'targets': targets, 'target_valid': valid}
        loss, parts = supervised_loss(output, batch, {'mean': [0.] * 7, 'std': [1.] * 7}, [[]] * 7, {})
        self.assertEqual(parts['parameters'], 1.0)
        loss.backward()
        gradient = output['parameters'].grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertEqual(int((gradient != 0).sum()), 1)
        self.assertEqual(float(gradient[0, 2]), 2.)

    def test_bpe_backfill_matches_ordered_reference(self):
        rng = np.random.default_rng(11)
        sequences = [rng.integers(0, 4, size=32).tolist() for _ in range(40)]
        sequences += [[1] * 32, [0, 1, 0, 1, 0, 1, 2, 3]]
        rules, lengths = learn_merges(sequences, 4, 24, 5)
        tokenizer = StructuralTokenizer(rules)
        self.assertLessEqual(max(lengths), 5)
        for seq in sequences:
            symbols = [(v, 1) for v in seq]
            for a, b, new in rules:
                merged, i = [], 0
                while i < len(symbols):
                    if i + 1 < len(symbols) and symbols[i][0] == a and symbols[i + 1][0] == b:
                        merged.append((new, symbols[i][1] + symbols[i + 1][1]))
                        i += 2
                    else:
                        merged.append(symbols[i])
                        i += 1
                symbols = merged
            expected = [token for token, length in symbols for _ in range(length)]
            np.testing.assert_array_equal(tokenizer.encode(seq), expected)

    def test_export_uses_standard_tensor_shapes_and_bytes(self):
        tensors = {'scalar': torch.tensor(3.), 'matrix': torch.randn(3, 5).T,
                   'integer': torch.arange(4), 'boolean': torch.tensor([True, False])}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            save_safetensors(path, tensors)
            restored = load_safetensors(path)
            self.assertEqual(set(restored), set(tensors))
            for name in tensors:
                self.assertTrue(torch.equal(restored[name], tensors[name]))

    def test_fft_and_portable_matmul_forward_agree(self):
        model, _ = fixture()
        x = torch.randn(1, 512, 2)
        with torch.no_grad():
            a = model(x, torch.tensor([512]))
            model.config['spectral_backend'] = 'matmul'
            b = model(x, torch.tensor([512]))
        for key in ['modulation', 'protocol', 'parameters']:
            torch.testing.assert_close(a[key], b[key], atol=3e-4, rtol=3e-4)

    def test_timing_fields_are_not_quantized_on_tiny_datasets(self):
        model, scaler = fixture()
        grids = [[0., 1.]] * 7
        cfg = {**model.config, 'parameter_mode': 'grid_residual'}
        model = RFModel(cfg, model.schema, scaler, grids)
        self.assertEqual(set(model.grids), {'0', '1', '2', '6'})

    def test_energy_mask_has_exact_budget_and_keeps_context(self):
        lengths = torch.tensor([512, 1024])
        iq = torch.randn(2, 1024, 2)
        for ratio in [.01, .3, .5]:
            mask = patch_mask(lengths, 16, ratio, 2, 8, iq, 'energy_mixed')
            expected = [max(1, min(n - 1, int(round(n * ratio)))) for n in [32, 64]]
            self.assertEqual(mask.sum(1).tolist(), expected)
            self.assertFalse(mask[0, 32:].any())

    def test_distributed_rejects_cross_group_gpu_pair(self):
        with patch.dict('os.environ', {'WORLD_SIZE': '2', 'LOCAL_RANK': '0', 'RANK': '0', 'CUDA_VISIBLE_DEVICES': '0,2'}):
            with self.assertRaisesRegex(ValueError, 'supports GPU pairs'):
                runtime('cuda')

    def test_short_epochs_do_not_starve_parameter_task(self):
        data = SimpleNamespace(indices=np.arange(3), meta={'modulation': np.array([0, -1, -1]),
            'protocol': np.array([-1, 0, -1]), 'target_valid': np.array([[False] * 7, [False] * 7, [True] * 7]),
            'length': np.array([512, 512, 512])})
        sampler = BalancedBatchSampler(data, batch_size=1, steps=2)
        sampler.set_epoch(0)
        first = [batch[0] for batch in sampler]
        sampler.set_epoch(1)
        second = [batch[0] for batch in sampler]
        self.assertEqual(first + second, [0, 1, 2, 0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
