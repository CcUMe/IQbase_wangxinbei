"""CPU checks for padding, gradients, task contracts and multi-scale export."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from infer import Predictor
from models.multiresolution import ComplexConv1d
from models.rf_model import RFModel
from training.train import build_optimizer
from utils.common import TARGETS, save_safetensors, write_json
from utils.config import DEFAULTS, load_config, resolve_config, training_options
from utils.contrastive import load_with_training_aux
from utils.early_stopping import EarlyStopping
from utils.engine import supervised_loss
from utils.head_optimizer import active_heads, clear_inactive_head_gradients


def fixture():
    c = deepcopy(DEFAULTS)
    c['model']['encoder'].update(width=32, depth=1, heads=4, dropout=0., attention_dropout=0., qk_normalization=True)
    c['model']['signal']['phase_features'] = True
    c['model']['heads'].update(parameter_mode='grid_residual', classification_pooling='multiscale_attention_stats', classification_fp32=True)
    c['model']['heads']['multiscale_readout'].update(projection_width=16, modulation_hidden=32, protocol_hidden=16, parameter_hidden=16, dropout=0.)
    p = c['model']['pyramid']
    p.update(enabled=True, width=24, heads=3, stem_complex_channels=4, local_blocks=1, transformer_depth=1,
             ffn_hidden=64, dropout=0., attention_dropout=0.)
    p['mid']['groups'], p['fine']['groups'] = 3, 6
    options = training_options(resolve_config(c))
    schema = {'classification_labels': {'modulation': [f'm{i}' for i in range(23)], 'protocol': [f'p{i}' for i in range(8)]}}
    scaler = {'mean': [0.] * 7, 'std': [1.] * 7}
    grids = [[0., 1.]] * 7
    return RFModel(options['model'], schema, scaler, grids), options, c


class MultiResolutionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_complex_convolution_uses_cross_coupled_iq(self):
        layer = ComplexConv1d(1, 1, 1)
        with torch.no_grad():
            layer.real.weight.fill_(2)
            layer.imag.weight.fill_(3)
        out = layer(torch.tensor([[[5.], [7.]]]))
        torch.testing.assert_close(out, torch.tensor([[[-11.], [29.]]]))

    def test_portable_frequency_backend_matches_fft(self):
        model, _, _ = fixture()
        model.eval()
        iq, lengths = torch.randn(1, 512, 2), torch.tensor([512])
        with torch.no_grad():
            fft = model(iq, lengths)
            model.config['spectral_backend'] = 'matmul'
            portable = model(iq, lengths)
        for key in ['modulation', 'protocol', 'parameters']:
            torch.testing.assert_close(fft[key], portable[key], atol=3e-4, rtol=3e-4)

    def test_grids_and_native_vs_padded_predictions(self):
        model, _, _ = fixture()
        model.eval()
        iq = torch.randn(2, 1024, 2)
        lengths = torch.tensor([512, 1024])
        observed = []
        hook = model.backbone.pyramid.register_forward_hook(lambda m, a, o: observed.append(o))
        with torch.no_grad():
            padded = model(iq, lengths)
            native = model(iq[:1, :512], lengths[:1])
            changed = iq.clone()
            changed[0, 512:] = 100 * torch.randn(512, 2)
            pollution = model(changed, lengths)
        hook.remove()
        self.assertEqual([v.sum(1).tolist() for v in observed[0]['valid']], [[32, 64], [64, 128], [128, 256]])
        self.assertEqual(observed[0]['memory_valid'].sum(1).tolist(), [224, 448])
        for key, shape in [('modulation', (2, 23)), ('protocol', (2, 8)), ('parameters', (2, 7))]:
            self.assertEqual(padded[key].shape, shape)
            torch.testing.assert_close(padded[key][:1], native[key], atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(padded[key], pollution[key], atol=0., rtol=0.)
        self.assertEqual(set(padded['grid_logits']), {'0', '1', '2', '6'})

    def test_each_task_backpropagates_through_both_shared_scales(self):
        model, options, _ = fixture()
        batch = dict(iq=torch.randn(2, 512, 2), lengths=torch.tensor([512, 512]),
                     modulation=torch.tensor([-1, -1]), protocol=torch.tensor([-1, -1]),
                     targets=torch.zeros(2, 7), target_valid=torch.zeros(2, 7, dtype=torch.bool))
        for task in ['modulation', 'protocol', 'parameter']:
            with self.subTest(task=task):
                labels = {k: v.clone() for k, v in batch.items()}
                if task == 'parameter':
                    labels['target_valid'][:, 3] = True
                else:
                    labels[task][:] = torch.tensor([0, 1])
                model.zero_grad(set_to_none=True)
                output = model(labels['iq'], labels['lengths'])
                loss, _ = supervised_loss(output, labels, {'mean': [0.] * 7, 'std': [1.] * 7}, [[0., 1.]] * 7, options)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                clear_inactive_head_gradients(model, active_heads(labels, options), torch.device('cpu'))
                for component in [model.backbone.time, model.backbone.pyramid.stem, *model.backbone.pyramid.branches,
                                  model.backbone.pyramid.time_frequency, model.backbone.pyramid.cross_scale]:
                    gradients = [p.grad for p in component.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)

    def test_missing_parameter_fields_do_not_receive_adamw_updates(self):
        model, options, c = fixture()
        optimizer = build_optimizer(model, c['optimizer'])
        before = [deepcopy(h.state_dict()) for h in model.heads['regression']]
        batch = dict(iq=torch.randn(2, 512, 2), modulation=torch.tensor([-1, -1]), protocol=torch.tensor([-1, -1]),
                     targets=torch.full((2, 7), float('nan')), target_valid=torch.zeros(2, 7, dtype=torch.bool))
        batch['target_valid'][:, 3] = True
        batch['targets'][:, 3] = 1.
        out = model(batch['iq'], torch.tensor([512, 512]))
        loss, _ = supervised_loss(out, batch, {'mean': [0.] * 7, 'std': [1.] * 7}, [[0., 1.]] * 7, options)
        loss.backward()
        clear_inactive_head_gradients(model, active_heads(batch, options), torch.device('cpu'))
        optimizer.step()
        for j, head in enumerate(model.heads['regression']):
            equal = all(torch.equal(before[j][k], v) for k, v in head.state_dict().items())
            self.assertEqual(equal, j != 3)
        self.assertTrue(all(p.grad is None for p in model.heads['modulation'].parameters()))
        self.assertTrue(all(p.grad is None for p in model.heads['protocol'].parameters()))

    def test_cpu_bfloat16_backward_and_export_predictor_contract(self):
        model, _, _ = fixture()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = model(torch.randn(1, 512, 2), torch.tensor([512]))
            loss = out['modulation'].square().mean() + out['protocol'].square().mean() + out['parameters'].square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertEqual(out['modulation'].dtype, torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_safetensors(root / 'model.safetensors', model.state_dict())
            write_json(root / 'config.json', dict(model_config=model.config, schema=model.schema,
                target_scaler={'mean': [0.] * 7, 'std': [1.] * 7}, parameter_grids=[[0., 1.]] * 7))
            predictor = Predictor(root, device='cpu', threads=2)
            records = [dict(task=task, sample_id=str(i), iq=torch.randn(512 if i % 2 else 1024, 2).tolist())
                       for i, task in enumerate(['modulation', 'protocol', 'parameter'])]
            results = predictor.predict(records)
        for record, result in zip(records, results):
            self.assertEqual(set(result), {'task', 'sample_id', 'prediction'})
            self.assertEqual(result['task'], record['task'])
            self.assertEqual(result['sample_id'], record['sample_id'])
            if record['task'] == 'parameter':
                self.assertEqual(set(result['prediction']), set(TARGETS))
            else:
                self.assertIn(result['prediction'], model.schema['classification_labels'][record['task']])
        other, _, _ = fixture()
        load_with_training_aux(other, model.state_dict(), model.config)
        self.assertTrue(all(torch.equal(v, other.state_dict()[k]) for k, v in model.state_dict().items()))

    def test_config_rejects_unsupported_modes_and_invalid_scale_layout(self):
        _, _, base = fixture()
        for mutate in [lambda c: c['training'].update(stage='pretrain'),
                       lambda c: c['model']['heads'].update(physical_statistics=True),
                       lambda c: c['model']['pyramid']['mid'].update(stride=16),
                       lambda c: c['model']['pyramid']['fine'].update(groups=7),
                       lambda c: c['run'].update(init_checkpoint='old.pt')]:
            c = deepcopy(base)
            mutate(c)
            with self.assertRaises(ValueError):
                resolve_config(c)

    def test_early_stop_survives_state_restore_and_does_not_hide_small_cumulative_gain(self):
        settings = dict(patience=2, min_delta=.001)
        early = EarlyStopping(settings)
        self.assertFalse(early.update(.6))
        self.assertFalse(early.update(.6006))
        restored = EarlyStopping(settings, early.state_dict())
        self.assertFalse(restored.update(.6012))
        self.assertFalse(restored.update(.6014))
        self.assertTrue(restored.update(.6015))


if __name__ == '__main__':
    unittest.main(verbosity=2)
