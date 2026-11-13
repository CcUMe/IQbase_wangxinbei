"""Mechanistic checks: retained values, local context and bounded detail use."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn

from infer import Predictor
from models.hierarchical import LengthAwareBiLSTM, EfficientAdditiveAttention
from models.rf_model import RFModel
from models.signal_mechanisms import signal_values, LocalWindowAttention, SignalAwareCrossAttention
from training.test_unified import fixture as previous_fixture
from utils.common import TARGETS, save_safetensors, write_json
from utils.config import resolve_config, training_options
from utils.contrastive import load_with_training_aux
from utils.engine import supervised_loss
from utils.head_optimizer import active_heads, clear_inactive_head_gradients


def fixture():
    _, _, c = previous_fixture()
    c['model']['hierarchy']['mechanism'].update(enabled=True, recurrent_window=32, attention_window=8)
    options = training_options(resolve_config(c))
    schema = {'classification_labels': {'modulation': [f'm{i}' for i in range(23)], 'protocol': [f'p{i}' for i in range(8)]}}
    return RFModel(options['model'], schema, {'mean': [0.] * 7, 'std': [1.] * 7}, [[0., 1.]] * 7), options, c


class SignalMechanismTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(37)

    def test_signal_values_retain_iq_and_relative_phase(self):
        phase = torch.arange(64).float() * .25
        iq = torch.stack((phase.cos(), phase.sin()), -1)[None]
        valid = torch.arange(64)[None] < 48
        values = signal_values(iq, valid)
        torch.testing.assert_close(values[..., :2], iq * valid[..., None], atol=0., rtol=0.)
        torch.testing.assert_close(values[0, 1:48, 4], torch.full((47,), .25).cos(), atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(values[0, 1:48, 5], torch.full((47,), .25).sin(), atol=2e-6, rtol=2e-6)
        self.assertTrue(torch.equal(values[:, 48:], torch.zeros_like(values[:, 48:])))
        self.assertEqual(float(values[0, 0, 3:].abs().sum()), 0.)

    def test_recurrent_context_is_local_and_identity_preserves_amplitude(self):
        recurrent = LengthAwareBiLSTM(8, 4, 2, 0., 'fp32', window_size=16).eval()
        x, lengths, valid = torch.randn(1, 64, 8), torch.tensor([64]), torch.ones(1, 64, dtype=torch.bool)
        changed = x.clone(); changed[:, 32:] += 10
        with torch.no_grad():
            a, b = recurrent(x, lengths, valid), recurrent(changed, lengths, valid)
        torch.testing.assert_close(a[:, :16], b[:, :16], atol=0., rtol=0.)
        with torch.no_grad():
            recurrent.projection.weight.zero_(); recurrent.projection.bias.zero_()
            torch.testing.assert_close(recurrent(x * 3, lengths, valid), x * 3, atol=0., rtol=0.)
        self.assertIsInstance(recurrent.norm, nn.Identity)

    def test_shifted_local_attention_never_wraps_global_context(self):
        attention = LocalWindowAttention(16, window=8, shift=4, dropout=0.).eval()
        x = torch.randn(1, 32, 16)
        valid = torch.ones(1, 32, dtype=torch.bool)
        changed = x.clone(); changed[:, 16:] += 20
        with torch.no_grad():
            a, b = attention(x, valid), attention(changed, valid)
        torch.testing.assert_close(a[:, :4], b[:, :4], atol=0., rtol=0.)
        valid[:, 20:] = False
        with torch.no_grad(): out = attention(x, valid)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(float(out[:, 20:].abs().sum()), 0.)

    def test_physical_values_bypass_semantic_normalization(self):
        attention = SignalAwareCrossAttention(12, 3, 0.).eval()
        query, memory, values = torch.randn(2, 7, 12), torch.randn(2, 9, 12), torch.randn(2, 9, 6)
        valid = torch.ones(2, 9, dtype=torch.bool); valid[0, 5:] = False
        with torch.no_grad():
            semantic, physical = attention(query, memory, valid, signal=values)
            semantic2, physical2 = attention(query, memory, valid, signal=values * 3)
            dirty = values.clone(); dirty[0, 5:] += 1000
            _, masked = attention(query, memory, valid, signal=dirty)
        torch.testing.assert_close(semantic, semantic2, atol=0., rtol=0.)
        torch.testing.assert_close(physical2, physical * 3, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(physical, masked, atol=0., rtol=0.)

    def test_global_spectrum_is_late_and_detail_correction_is_bounded(self):
        model, _, _ = fixture(); model.eval()
        iq, lengths = torch.randn(1, 512, 2), torch.tensor([512])
        stem = model.backbone.stem
        with torch.no_grad():
            before = stem.forward_features(iq, lengths)
            stem.frontend_global[0].weight.add_(torch.randn_like(stem.frontend_global[0].weight))
            after = stem.forward_features(iq, lengths)
        for a, b in zip(before['features'], after['features']): torch.testing.assert_close(a, b, atol=0., rtol=0.)
        self.assertFalse(torch.equal(before['late_global'], after['late_global']))
        with torch.no_grad(): pyramid = model.backbone(iq, lengths, return_pyramid=True)
        head = model.heads['modulation']
        deep, correction = head.descriptors(pyramid)
        self.assertLessEqual(float(correction.detach().abs().max()), .25)
        changed = dict(pyramid)
        changed['summaries'] = [x + 100 for x in pyramid['summaries']]
        deep2, _ = head.descriptors(changed)
        torch.testing.assert_close(deep, deep2, atol=0., rtol=0.)
        self.assertEqual(head.classifier[0].in_features, 16)
        self.assertTrue(all(not isinstance(m, EfficientAdditiveAttention) for m in stem.modules()))
        self.assertTrue(all(isinstance(m, nn.Identity) for m in stem.stage_norms))

    def test_full_pipeline_each_task_padding_and_bfloat16(self):
        model, options, _ = fixture()
        for task in ['modulation', 'protocol', 'parameter']:
            batch = dict(iq=torch.randn(1, 512, 2), modulation=torch.tensor([-1]), protocol=torch.tensor([-1]),
                targets=torch.zeros(1, 7), target_valid=torch.zeros(1, 7, dtype=torch.bool))
            if task == 'parameter': batch['target_valid'][:, 3] = True
            else: batch[task][:] = 0
            model.zero_grad(set_to_none=True)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                out = model(batch['iq'], torch.tensor([512]))
                loss, _ = supervised_loss(out, batch, {'mean': [0.] * 7, 'std': [1.] * 7}, [[0., 1.]] * 7, options)
            loss.backward()
            clear_inactive_head_gradients(model, active_heads(batch, options), torch.device('cpu'))
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual([n for n, p in model.backbone.named_parameters() if p.grad is None], [])
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        model.eval()
        iq, lengths = torch.randn(2, 1024, 2), torch.tensor([512, 1024])
        with torch.no_grad():
            mixed = model(iq, lengths); native = model(iq[:1, :512], lengths[:1])
            changed = iq.clone(); changed[0, 512:] = 1000
            polluted = model(changed, lengths)
            pyramid = model.backbone(iq, lengths, return_pyramid=True)
        self.assertEqual(pyramid['signal_memory'].shape, (2, 1344, 6))
        self.assertEqual(pyramid['signal_features'].shape, (2, 1024, 6))
        for key in ['modulation', 'protocol', 'parameters']:
            torch.testing.assert_close(mixed[key][:1], native[key], atol=4e-5, rtol=4e-5)
            torch.testing.assert_close(mixed[key], polluted[key], atol=0., rtol=0.)

    def test_predictor_roundtrip_and_mode_guard(self):
        model, _, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_safetensors(root / 'model.safetensors', model.state_dict())
            write_json(root / 'config.json', dict(model_config=model.config, schema=model.schema,
                target_scaler={'mean': [0.] * 7, 'std': [1.] * 7}, parameter_grids=[[0., 1.]] * 7))
            predictor = Predictor(root, device='cpu', threads=2)
            result = predictor.predict([dict(task='parameter', sample_id='0', iq=torch.randn(512, 2).tolist())])
            self.assertEqual(set(result[0]['prediction']), set(TARGETS))
        previous, _, _ = previous_fixture()
        with self.assertRaisesRegex(ValueError, 'mechanism'):
            load_with_training_aux(model, previous.state_dict(), previous.config)
        other, _, _ = fixture()
        load_with_training_aux(other, model.state_dict(), model.config)

    def test_invalid_windows_and_correction_bounds_are_rejected(self):
        _, _, base = fixture()
        for options in [dict(recurrent_window=33), dict(attention_window=0),
                        dict(detail_initial_ratio=.5), dict(signal_gain=0)]:
            c = deepcopy(base); c['model']['hierarchy']['mechanism'].update(options)
            with self.assertRaises(ValueError): resolve_config(c)


if __name__ == '__main__': unittest.main(verbosity=2)
