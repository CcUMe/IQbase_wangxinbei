"""Behavioral checks for paired loss, task protection, compatibility and resume."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
import numpy as np
import torch
from torch.nn import functional as F
from models.rf_model import RFModel
from models.patch_shift import PatchShiftAdapter, patch_shift_options
from utils.common import model_bundle, save_checkpoint, write_json
from utils.contrastive import load_with_training_aux
from utils.view_consistency import eligible_modulation, augmented_view, paired_correction
from utils.config import DEFAULTS as STANDARD_DEFAULTS, resolve_config
from training.train_shift_consistent import DEFAULTS, selection


class ShiftConsistentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_two_view_loss_preserves_raw_and_untouched_record_denominators(self):
        torch.manual_seed(9)
        a = torch.randn(4, 23, requires_grad=True)
        b = torch.randn(2, 23, requires_grad=True)
        y, keep = torch.tensor([0, 4, 14, 20]), torch.tensor([True, True, False, False])
        correction, _ = paired_correction(a, b, y, keep, consistency_weight=0.)
        expected = (F.cross_entropy(a[:2], y[:2], reduction='sum')*.5 +
                    F.cross_entropy(b, y[:2], reduction='sum')*.5 +
                    F.cross_entropy(a[2:], y[2:], reduction='sum'))/4
        torch.testing.assert_close(F.cross_entropy(a, y)+correction, expected)
        (F.cross_entropy(a, y)+correction).backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertTrue(torch.isfinite(b.grad).all())
        self.assertGreater(float(a.grad.abs().sum()), 0)
        self.assertGreater(float(b.grad.abs().sum()), 0)
        extreme = torch.tensor([[1e4, -1e4]], requires_grad=True)
        other = -extreme.detach().clone().requires_grad_()
        loss, stats = paired_correction(extreme, other, torch.tensor([0]), torch.tensor([True]))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(stats['view_js'], 0)

    def test_tasks_and_padding_are_protected_and_transforms_reproduce(self):
        batch = {'iq': torch.randn(4, 64, 2), 'lengths': torch.tensor([64, 32, 64, 64]),
                 'modulation': torch.tensor([0, 3, 4, 15]), 'protocol': torch.tensor([-1, -1, 0, -1]),
                 'target_valid': torch.zeros(4, 7, dtype=torch.bool)}
        batch['target_valid'][1, 0] = True
        self.assertEqual(eligible_modulation(batch, list(range(14))).tolist(), [True, False, False, False])
        batch['iq'][1, 32:] = 100.
        a = augmented_view(batch['iq'], batch['lengths'], 17)
        torch.testing.assert_close(a, augmented_view(batch['iq'], batch['lengths'], 17), rtol=0, atol=0)
        self.assertEqual(float(a[1, 32:].abs().max()), 0.)
        for i, length in enumerate(batch['lengths']):
            torch.testing.assert_close(a[i, :length].square().sum(), batch['iq'][i, :length].square().sum())

    def test_adapter_reuses_geometry_and_ignores_hidden_padding(self):
        torch.manual_seed(19)
        projection = torch.nn.Conv1d(8, 16, 16, stride=16)
        hidden = torch.randn(2, 8, 64)
        original = projection(hidden).transpose(1, 2)
        adapter = PatchShiftAdapter(16, 16, dict(enabled=True, bottleneck=8, dropout=0.))
        valid = torch.ones(2, 64, dtype=torch.bool)
        self.assertEqual(float(adapter(hidden, original, projection, valid).detach().abs().max()), 0.)
        with torch.no_grad():
            adapter.up.weight.normal_(0, .02)
        delta = adapter(hidden, original, projection, valid)
        padded = F.pad(hidden, (0, 64), value=999.)
        mask = torch.arange(128)[None].expand(2, -1) < 64
        result = adapter(padded, F.pad(original, (0, 0, 0, 4)), projection, mask)
        torch.testing.assert_close(delta, result[:, :4], rtol=1e-5, atol=1e-5)
        self.assertEqual(float(result[:, 4:].detach().abs().max()), 0.)
        delta.square().mean().backward()
        self.assertGreater(float(adapter.down.weight.grad.norm()), 0.)
        with self.assertRaises(ValueError):
            patch_shift_options(dict(enabled=True, stride=3))

    def test_warmstart_preserves_all_three_tasks_and_cannot_reinterpret_adapter(self):
        torch.manual_seed(3)
        info = json.loads((ROOT/'artifacts/data_full/dataset.json').read_text())
        old = RFModel(dict(width=64, depth=2, heads=4, qk_normalization=True,
                          dropout=0., phase_features=True, parameter_mode='grid_residual'),
                      info['schema'], info['target_scaler'], info['parameter_grids']).eval()
        bundle = dict(model_config={**old.config, 'patch_shift': {'enabled': True, 'bottleneck': 16}},
                      schema=info['schema'], target_scaler=info['target_scaler'], parameter_grids=info['parameter_grids'])
        model = RFModel(**bundle).eval()
        load_with_training_aux(model, old.state_dict(), old.config)
        for length in [512, 1024]:
            x, lengths = torch.randn(2, length, 2), torch.full((2,), length)
            with torch.inference_mode():
                a, b = old(x, lengths), model(x, lengths)
            for key in ['modulation', 'protocol', 'parameters']:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        with torch.no_grad():
            model.backbone.patch_shift.up.weight.normal_(0, .01)
        learned = {k: v.clone() for k, v in model.state_dict().items()}
        load_with_training_aux(model, learned)
        torch.testing.assert_close(model.backbone.patch_shift.up.weight, learned['backbone.patch_shift.up.weight'])
        load_with_training_aux(model, old.state_dict(), old.config)
        self.assertEqual(float(model.backbone.patch_shift.up.weight.detach().abs().max()), 0.)
        changed = deepcopy(model.config)
        changed['patch_shift']['stride'] = 8
        with self.assertRaises(ValueError):
            load_with_training_aux(model, learned, changed)
        self.assertGreater(model.parameter_report()['shared_ratio'], .8)

    def test_selection_rejects_qam_or_parameter_damage(self):
        baseline = {'classification': {'modulation': {'accuracy': .67, 'macro_f1': .66},
                     'protocol': {'accuracy': .97}, 'qam': {'accuracy': .43}},
                    'parameters': {'symbol_rate': {'rmse': 100.}}}
        candidate = deepcopy(baseline)
        candidate['classification']['modulation']['accuracy'] = .69
        candidate['classification']['qam']['accuracy'] = .41
        allowed, issues = selection(candidate, baseline, DEFAULTS['evaluation'])
        self.assertFalse(allowed)
        self.assertIn('qam_accuracy', issues)
        candidate = deepcopy(baseline)
        candidate['parameters']['symbol_rate']['rmse'] = 110.
        self.assertFalse(selection(candidate, baseline, DEFAULTS['evaluation'])[0])


def make_fixture(folder):
    """Small REAL data subset, same label schema; only for integration tests."""
    folder = Path(folder)
    info = json.loads((ROOT/'artifacts/data_full/dataset.json').read_text())
    meta = dict(np.load(ROOT/'artifacts/data_full/metadata.npz', allow_pickle=False))
    chosen = []
    for split in [0, 1]:
        for task in ['modulation', 'protocol']:
            for label in np.unique(meta[task]):
                if label >= 0:
                    chosen.extend(np.flatnonzero((meta['split'] == split) & (meta[task] == label))[:2])
        for j in range(7):
            chosen.extend(np.flatnonzero((meta['split'] == split) & meta['target_valid'][:, j])[:4])
    chosen = np.unique(chosen)
    data = folder/'data'
    data.mkdir(parents=True)
    source_iq = np.load(ROOT/'artifacts/data_full/iq.npy', mmap_mode='r')
    np.save(data/'iq.npy', np.array(source_iq[chosen]))
    np.savez(data/'metadata.npz', **{k: v[chosen] for k, v in meta.items()})
    info.update(rows=len(chosen), train_rows=int((meta['split'][chosen] == 0).sum()),
                validation_rows=int((meta['split'][chosen] == 1).sum()), fingerprint='shift_consistent_test_fixture')
    write_json(data/'dataset.json', info)
    settings = deepcopy(STANDARD_DEFAULTS)
    settings['model']['encoder'].update(width=64, depth=2, heads=4, dropout=0., attention_dropout=0., qk_normalization=True)
    settings['model']['signal']['phase_features'] = True
    settings['model']['heads']['parameter_mode'] = 'grid_residual'
    settings = resolve_config(settings)
    torch.manual_seed(77)
    from utils.config import model_options
    model = RFModel(model_options(settings), info['schema'], info['target_scaler'], info['parameter_grids'])
    save_checkpoint(folder/'source.pt', dict(model_config=model.config, schema=info['schema'],
        target_scaler=info['target_scaler'], parameter_grids=info['parameter_grids'], model=model.state_dict(),
        stage='supervised', global_step=1, dataset_fingerprint=info['fingerprint'], training_config=settings))
    config = deepcopy(DEFAULTS)
    config['run'].update(source=str(folder/'source.pt'), data=str(data), output=str(folder/'runs'))
    config['training'].update(steps=6, batch_size=4, precision='fp32')
    config['model']['patch_shift']['bottleneck'] = 16
    config['scheduler']['warmup_steps'] = 1
    config['evaluation'].update(every_steps=3, batch_size=8)
    config['checkpoint']['every_steps'] = 3
    config['runtime'].update(device='cpu', threads=2, log_every=1, tensorboard=False)
    write_json(folder/'config.json', config)
    return folder/'config.json'


if __name__ == '__main__':
    unittest.main(verbosity=2)
