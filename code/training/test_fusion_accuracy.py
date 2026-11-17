"""Gate-only normalization and long-window three-stage contracts (CPU)."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from models.rf_model import RFModel
from utils.config import DEFAULTS, resolve_config
from utils.contrastive import load_with_training_aux
from utils.initialization import load_pretrained_backbone
from utils.engine import patch_mask


def build(gate=False, window=64, pretrain=False):
    return RFModel(dict(width=32, depth=2, heads=4, dropout=0., attention_dropout=0.,
        patch_size=8, stft_window=window, phase_features=True, qk_normalization=True,
        gate_input_normalization=gate, pretrain_heads=pretrain,
        atomic_vocab=8 if pretrain else 0, emergent_vocab=12 if pretrain else 0,
        iq_geometry={'enabled': True, 'bottleneck': 8}),
        {'classification_labels': {'modulation': ['a', 'b', 'c'], 'protocol': ['x', 'y']}},
        {'mean': [0.] * 7, 'std': [1.] * 7}, [[] for _ in range(7)])


class FusionAccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_only_gate_conditions_are_normalized(self):
        torch.manual_seed(19)
        base, candidate = build(), build(True)
        candidate.load_state_dict(base.state_dict(), strict=True)
        base.eval(); candidate.eval()
        with torch.no_grad():
            base.backbone.gate.weight.zero_(); base.backbone.gate.bias.zero_()
            candidate.backbone.gate.load_state_dict(base.backbone.gate.state_dict())
        seen = []
        handle = candidate.backbone.gate.register_forward_pre_hook(lambda m, args: seen.append(args[0]))
        for length in [512, 1024]:
            x = torch.randn(2, length, 2)
            lens = torch.full((2,), length)
            with torch.no_grad():
                a, b = base(x, lens), candidate(x, lens)
            for key in ['modulation', 'protocol', 'parameters']:
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        handle.remove()
        for conditions in seen:
            for half in conditions.chunk(2, -1):
                torch.testing.assert_close(half.mean(-1), torch.zeros_like(half[..., 0]), atol=1e-6, rtol=0)

    def test_mask_padding_gradients_and_exact_transfer(self):
        for gate, window in [(False, 64), (True, 64), (False, 256)]:
            with self.subTest(gate=gate, window=window):
                torch.manual_seed(23)
                model = build(gate, window, True).eval()
                with torch.no_grad():
                    model.backbone.iq_geometry.up.weight.normal_(0, .03)
                for length in [512, 1024]:
                    x = torch.randn(2, length, 2)
                    lens = torch.full((2,), length)
                    mask = patch_mask(lens, 8, .3, 8, 33)
                    changed = x.clone()
                    changed[mask.repeat_interleave(8, 1)] += 10000
                    with torch.no_grad():
                        a = model(x, lens, patch_mask=mask, return_pretrain=True)
                        b = model(changed, lens, patch_mask=mask, return_pretrain=True)
                    for key in ['tokens', 'modulation', 'protocol', 'parameters']:
                        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
                    for key in a['pretrain']:
                        torch.testing.assert_close(a['pretrain'][key], b['pretrain'][key], atol=0, rtol=0)
                with torch.no_grad():
                    x = torch.randn(2, 512, 2); lens = torch.full((2,), 512)
                    a, b = model(x, lens), model(F.pad(x, (0, 0, 0, 512)), lens)
                for key in ['modulation', 'protocol', 'parameters']:
                    torch.testing.assert_close(a[key], b[key], atol=1e-4, rtol=1e-4)
                model.train()
                output = model(torch.randn(2, 512, 2), lens, patch_mask=patch_mask(lens, 8, .3, 8, 8), return_pretrain=True)
                loss = output['pretrain']['atomic'].square().mean() + output['pretrain']['emergent'].square().mean()
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                for module in [model.backbone.time[0], model.backbone.spectrum[0], model.backbone.gate]:
                    self.assertGreater(float(module.weight.grad.norm()), 0.)
                target = build(gate, window)
                head = target.heads['modulation'][0].weight.detach().clone()
                load_pretrained_backbone(target, {'model': model.state_dict(), 'model_config': model.config})
                for key, value in model.backbone.state_dict().items():
                    torch.testing.assert_close(target.backbone.state_dict()[key], value, atol=0, rtol=0)
                torch.testing.assert_close(target.heads['modulation'][0].weight, head, atol=0, rtol=0)
                self.assertGreater(target.parameter_report()['shared_ratio'], .8)

    def test_semantic_mismatch_rejected_and_legacy_default_preserved(self):
        old, new = build(), build(True)
        with self.assertRaisesRegex(ValueError, 'gate_input_normalization'):
            load_pretrained_backbone(new, {'model': old.state_dict(), 'model_config': old.config})
        with self.assertRaisesRegex(ValueError, 'gate_input_normalization'):
            load_with_training_aux(new, old.state_dict(), old.config)
        with self.assertRaisesRegex(ValueError, 'stft_window'):
            load_pretrained_backbone(build(False, 256), {'model': old.state_dict(), 'model_config': old.config})
        c = deepcopy(DEFAULTS)
        del c['model']['signal']['gate_input_normalization']
        self.assertFalse(resolve_config(c)['model']['signal']['gate_input_normalization'])
        c['model']['signal']['gate_input_normalization'] = 'true'
        with self.assertRaises(ValueError):
            resolve_config(c)


if __name__ == '__main__':
    unittest.main(verbosity=2)
