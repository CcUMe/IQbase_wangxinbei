"""Single-frontend graph, geometry, precision and task compatibility checks."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F

from infer import Predictor
from models import hierarchical
from models.rf_model import Backbone, RFModel
from training.test_hybrid import fixture as parallel_fixture
from utils.common import TARGETS, save_safetensors, write_json
from utils.config import resolve_config, training_options
from utils.contrastive import load_with_training_aux
from utils.engine import supervised_loss
from utils.head_optimizer import active_heads, clear_inactive_head_gradients


def fixture():
    _, _, c = parallel_fixture()
    c['model']['hybrid']['topology'] = 'serial'
    c['model']['hierarchy']['spectrum'].update(iq_window=7, centered_windows=True)
    options = training_options(resolve_config(c))
    schema = {'classification_labels': {'modulation': [f'm{i}' for i in range(23)], 'protocol': [f'p{i}' for i in range(8)]}}
    return RFModel(options['model'], schema, {'mean': [0.] * 7, 'std': [1.] * 7}, [[0., 1.]] * 7), options, c


class UnifiedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(2)

    def test_exactly_one_frontend_and_one_global_spectrum(self):
        model, _, _ = fixture()
        counts = {}
        hooks = [module.register_forward_hook(lambda m, a, o, name=name: counts.__setitem__(name, counts.get(name, 0) + 1))
                 for name, module in [('time', model.backbone.stem.time), ('spectrum', model.backbone.stem.spectrum),
                                      ('fusion', model.backbone.stem.fusion)]]
        with patch.object(model.backbone.stem, 'preprocess', wraps=model.backbone.stem.preprocess) as normalize, \
             patch('models.hierarchical.global_log_power', wraps=hierarchical.global_log_power) as global_spectrum:
            model(torch.randn(2, 1024, 2), torch.tensor([512, 1024]))
            self.assertEqual(normalize.call_count, 1)
            self.assertEqual(global_spectrum.call_count, 1)
        for hook in hooks: hook.remove()
        self.assertEqual(counts, {'time': 1, 'spectrum': 1, 'fusion': 1})
        self.assertFalse(hasattr(model.backbone, 'time'))
        self.assertFalse(hasattr(model.backbone, 'detail'))
        self.assertIsNone(model.backbone.stem.readout_fusion)
        self.assertIsNone(model.backbone.stem.global_spectrum)

    def test_mask_geometry_adapter_centers_and_padding(self):
        model, _, _ = fixture()
        model.eval()
        observed = []
        hook = model.backbone.hybrid_fusion.register_forward_pre_hook(lambda m, args: observed.append(args))
        iq, lengths = torch.randn(2, 1024, 2), torch.tensor([512, 1024])
        with torch.no_grad():
            mixed = model(iq, lengths)
            native = model(iq[:1, :512], lengths[:1])
            changed = iq.clone(); changed[0, 512:] += 1000
            polluted = model(changed, lengths)
        hook.remove()
        features, masks, centers = observed[0][:3]
        self.assertEqual([m.sum(1).tolist() for m in masks], [[32, 64], [128, 256], [512, 1024]])
        stride = model.backbone.adapter_stride
        torch.testing.assert_close(centers[0], centers[1].unfold(0, stride, stride).mean(-1), atol=0., rtol=0.)
        self.assertEqual([tuple(h.shape[1:]) for h in features], [(64, 32), (256, 32), (1024, 16)])
        for key in ['modulation', 'protocol', 'parameters']:
            torch.testing.assert_close(mixed[key][:1], native[key], atol=4e-5, rtol=4e-5)
            torch.testing.assert_close(mixed[key], polluted[key], atol=0., rtol=0.)

    def test_both_local_spectral_views_are_centered_on_same_sample(self):
        model, _, _ = fixture()
        spectrum = model.backbone.stem.spectrum
        source = torch.zeros(1, 2, 512); source[0, 0, 256] = 1.
        iw, qw = spectrum.options['i_window'], spectrum.options['iq_window']
        i_view = F.conv1d(source[:, :1], spectrum.i_kernels, padding=iw // 2).square().sum(1)[0]
        iq_view = F.conv1d(source, spectrum.iq_kernels, padding=qw // 2).square().sum(1)[0]
        for view in [i_view, iq_view]:
            self.assertEqual(int(view.argmax()), 256)
            torch.testing.assert_close(view[253:256], view[257:260].flip(0), atol=1e-5, rtol=1e-5)

    def test_every_task_reaches_the_entire_serial_backbone(self):
        model, options, _ = fixture()
        for task in ['modulation', 'protocol', 'parameter']:
            with self.subTest(task=task):
                batch = dict(iq=torch.randn(2, 512, 2), modulation=torch.tensor([-1, -1]), protocol=torch.tensor([-1, -1]),
                    targets=torch.zeros(2, 7), target_valid=torch.zeros(2, 7, dtype=torch.bool))
                if task == 'parameter':
                    batch['target_valid'][:, 3] = True; batch['targets'][:, 3] = 1.
                else:
                    batch[task][:] = torch.tensor([0, 1])
                model.zero_grad(set_to_none=True)
                out = model(batch['iq'], torch.tensor([512, 512]))
                loss, _ = supervised_loss(out, batch, {'mean': [0.] * 7, 'std': [1.] * 7}, [[0., 1.]] * 7, options)
                loss.backward()
                clear_inactive_head_gradients(model, active_heads(batch, options), torch.device('cpu'))
                modules = [model.backbone.stem.time, model.backbone.stem.spectrum, model.backbone.stem.frontend_global,
                    model.backbone.stem.recurrent, *model.backbone.stem.stages, model.backbone.token_projection,
                    model.backbone.blocks, model.backbone.hybrid_fusion]
                for module in modules:
                    gradients = [p.grad for p in module.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
                self.assertEqual([n for n, p in model.backbone.named_parameters() if p.grad is None], [])

    def test_bfloat16_and_zero_signal_are_finite(self):
        model, _, _ = fixture()
        for amplitude in [0., 1e-12, 1.]:
            with self.subTest(amplitude=amplitude):
                model.zero_grad(set_to_none=True)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    out = model(torch.randn(1, 512, 2) * amplitude, torch.tensor([512]))
                    loss = sum(out[k].square().mean() for k in ['modulation', 'protocol', 'parameters'])
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                self.assertEqual(out['modulation'].dtype, torch.float32)

    def test_fft_portability_and_transformer_weight_interface(self):
        model, _, _ = fixture()
        original = Backbone(model.config)
        core = {k: v for k, v in original.state_dict().items() if k.startswith(('blocks.', 'norm.', 'position_projection.'))}
        loaded = model.backbone.load_state_dict(core, strict=False)
        self.assertFalse(loaded.unexpected_keys)
        self.assertTrue(all(k.startswith(('stem.', 'token_projection.', 'token_norm.', 'hybrid_fusion.')) for k in loaded.missing_keys))
        for key, value in core.items(): self.assertTrue(torch.equal(model.backbone.state_dict()[key], value))
        model.eval()
        iq, lengths = torch.randn(1, 512, 2), torch.tensor([512])
        with torch.no_grad():
            fft = model(iq, lengths)
            model.config['spectral_backend'] = 'matmul'
            portable = model(iq, lengths)
        for key in ['modulation', 'protocol', 'parameters']:
            torch.testing.assert_close(fft[key], portable[key], atol=3e-4, rtol=3e-4)

    def test_export_predictor_and_topology_guards(self):
        model, _, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_safetensors(root / 'model.safetensors', model.state_dict())
            write_json(root / 'config.json', dict(model_config=model.config, schema=model.schema,
                target_scaler={'mean': [0.] * 7, 'std': [1.] * 7}, parameter_grids=[[0., 1.]] * 7))
            predictor = Predictor(root, device='cpu', threads=2)
            records = [dict(task=t, sample_id=str(i), iq=torch.randn(512 if i % 2 else 1024, 2).tolist())
                       for i, t in enumerate(['modulation', 'protocol', 'parameter'])]
            out = predictor.predict(records)
        for row, result in zip(records, out):
            self.assertEqual(set(result), {'task', 'sample_id', 'prediction'})
            self.assertEqual((row['task'], row['sample_id']), (result['task'], result['sample_id']))
            if row['task'] == 'parameter':
                self.assertEqual(set(result['prediction']), set(TARGETS))
            else:
                self.assertIn(result['prediction'], model.schema['classification_labels'][row['task']])
        other, _, _ = fixture()
        load_with_training_aux(other, model.state_dict(), model.config)
        parallel, _, _ = parallel_fixture()
        with self.assertRaisesRegex(ValueError, 'topology'):
            load_with_training_aux(model, parallel.state_dict(), parallel.config)

    def test_incompatible_geometry_and_window_configuration_are_rejected(self):
        _, _, base = fixture()
        for mutate in [lambda c: c['model']['hierarchy'].update(downsample_strides=[4, 8]),
                       lambda c: c['model']['hierarchy']['spectrum'].update(centered_windows=False),
                       lambda c: c['model']['hierarchy']['spectrum'].update(iq_window=8),
                       lambda c: c['model']['signal']['position_options'].update(reference_length=512)]:
            c = deepcopy(base); mutate(c)
            with self.assertRaises(ValueError): resolve_config(c)


if __name__ == '__main__':
    unittest.main(verbosity=2)
