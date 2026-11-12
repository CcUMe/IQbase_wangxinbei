"""Regression checks for attention saturation and inactive AdamW head updates."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F

from models.rf_model import RFModel, RotaryAttention
from utils.augmentation import augment_iq, augment_supervised_iq, rotate_phase
from training.train import build_optimizer
from utils.config import DEFAULTS, training_options
from utils.engine import patch_mask, supervised_loss
from utils.head_optimizer import active_heads, clear_inactive_head_gradients


def fixture():
    c = deepcopy(DEFAULTS)
    c['model']['encoder'].update(width=32, depth=1, heads=4, dropout=0., attention_dropout=0., qk_normalization=True)
    c['model']['signal'].update(phase_features=True, normalize_fusion=True)
    options = training_options(c)
    scaler = {'mean': [0.] * 7, 'std': [1.] * 7}
    model = RFModel(options['model'], {'classification_labels': {'modulation': ['a', 'b'], 'protocol': ['x', 'y']}}, scaler, [[]] * 7)
    return c, options, model, scaler


class StabilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_qk_logits_remain_bounded_after_projection_weights_grow(self):
        attention = RotaryAttention(32, 4, 0., qk_normalization=True)
        with torch.no_grad():
            attention.qkv.weight.mul_(10000)
            attention.qkv.bias.mul_(10000)
        original = F.scaled_dot_product_attention
        recorded = []
        def inspect(q, k, v, **kwargs):
            recorded.append((q @ k.transpose(-2, -1) / q.shape[-1] ** .5).detach())
            return original(q, k, v, **kwargs)
        x = torch.randn(2, 8, 32, requires_grad=True)
        with patch('torch.nn.functional.scaled_dot_product_attention', side_effect=inspect):
            output = attention(x, torch.ones(2, 8, dtype=torch.bool), torch.arange(8)[None].expand(2, -1))
        self.assertLessEqual(float(recorded[0].abs().max()), 8 ** .5 + 1e-4)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_phase_augmentation_preserves_power_spectrum_and_timing(self):
        iq = torch.randn(3, 512, 2)
        iq[:, :40] = 0
        angle = torch.tensor([.3, 1.7, -2.2])
        rotated = rotate_phase(iq, angle)
        torch.testing.assert_close(rotated.square().sum(-1), iq.square().sum(-1))
        original_fft = torch.fft.fft(torch.view_as_complex(iq.contiguous())).abs()
        rotated_fft = torch.fft.fft(torch.view_as_complex(rotated.contiguous())).abs()
        torch.testing.assert_close(rotated_fft, original_fft, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(rotated[:, :40], iq[:, :40], rtol=0, atol=0)
        self.assertIs(augment_iq(iq, {'phase_rotation_probability': 0.}), iq)
        torch.testing.assert_close(rotate_phase(rotated, -angle), iq, atol=1e-6, rtol=1e-6)

    def test_modulation_shift_never_changes_parameter_protocol_or_paired_rows(self):
        iq = torch.randn(4, 512, 2)
        batch = {'iq': iq, 'lengths': torch.full((4,), 512),
                 'modulation': torch.tensor([0, -1, -1, 0]), 'protocol': torch.tensor([-1, -1, 0, -1]),
                 'target_valid': torch.tensor([[False] * 7, [True] * 7, [False] * 7, [True] * 7])}
        settings = {'phase_rotation_probability': 0., 'modulation_only': {
            'phase_rotation_probability': 1., 'circular_shift_probability': 1.}}
        output = augment_supervised_iq(batch, settings)
        torch.testing.assert_close(output[1:], iq[1:], rtol=0, atol=0)
        torch.testing.assert_close(output[0].square().sum(), iq[0].square().sum())
        self.assertFalse(torch.equal(output[0], iq[0]))

    def test_fusion_input_is_normalized_even_after_branch_scale_growth(self):
        _, _, model, _ = fixture()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.startswith(('backbone.time.', 'backbone.spectrum.', 'backbone.global_spectrum.')):
                    parameter.mul_(100)
        received = []
        handle = model.backbone.blocks[0].register_forward_pre_hook(lambda module, args: received.append(args[0].detach()))
        model(torch.randn(2, 512, 2), torch.tensor([512, 512]))
        handle.remove()
        torch.testing.assert_close(received[0].square().mean(-1), torch.ones(2, 32), atol=2e-4, rtol=2e-4)

    def test_inactive_head_with_adam_momentum_is_unchanged(self):
        c, options, model, scaler = fixture()
        optimizer = build_optimizer(model, c['optimizer'])
        batch = {'iq': torch.randn(2, 512, 2), 'lengths': torch.tensor([512, 512]),
                 'modulation': torch.tensor([0, 1]), 'protocol': torch.tensor([-1, -1]),
                 'targets': torch.zeros(2, 7), 'target_valid': torch.zeros(2, 7, dtype=torch.bool)}
        output = model(batch['iq'], batch['lengths'])
        loss, _ = supervised_loss(output, batch, scaler, [[]] * 7, options)
        loss.backward()
        optimizer.step()
        initial = deepcopy(model.state_dict())
        optimizer_state = deepcopy(optimizer.state_dict())
        changed = []
        batch.update(modulation=torch.tensor([-1, -1]), protocol=torch.tensor([0, 1]))
        for protect in [False, True]:
            model.load_state_dict(initial)
            optimizer.load_state_dict(deepcopy(optimizer_state))
            optimizer.zero_grad(set_to_none=True)
            output = model(batch['iq'], batch['lengths'])
            loss, _ = supervised_loss(output, batch, scaler, [[]] * 7, options)
            loss.backward()
            if protect:
                clear_inactive_head_gradients(model, active_heads(batch, options), torch.device('cpu'))
            optimizer.step()
            changed.append(not torch.equal(model.heads['modulation'][0].weight, initial['heads.modulation.0.weight']))
        self.assertEqual(changed, [True, False])

    def test_stabilization_keeps_masked_targets_out_of_student_inputs(self):
        _, _, model, _ = fixture()
        model.eval()
        x = torch.randn(2, 512, 2)
        lengths = torch.tensor([512, 512])
        mask = patch_mask(lengths, 16, .3, 2, 7)
        changed = x.clone()
        changed[mask.repeat_interleave(16, 1)] += 1000
        with torch.no_grad():
            a = model(x, lengths, patch_mask=mask)
            b = model(changed, lengths, patch_mask=mask)
        for key in ['modulation', 'protocol', 'parameters']:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
