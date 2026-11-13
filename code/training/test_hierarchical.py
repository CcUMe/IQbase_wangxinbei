"""CPU regression checks for the high-resolution recurrent/staged backbone."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from infer import Predictor
from models.rf_model import RFModel
from training.train import build_optimizer
from utils.common import TARGETS, save_safetensors, write_json
from utils.config import DEFAULTS, resolve_config, training_options
from utils.contrastive import load_with_training_aux
from utils.engine import supervised_loss
from utils.head_optimizer import active_heads, clear_inactive_head_gradients


def fixture():
    c = deepcopy(DEFAULTS)
    h = c['model']['hierarchy']
    h.update(enabled=True, stage_widths=[24, 32, 48], stage_depths=[1, 1, 1],
             stem_complex_channels=4, conv_kernel=3, lstm_hidden=8, lstm_layers=2,
             fusion_width=24, fusion_heads=3, dropout=0., mlp_ratio=2.)
    h['spectrum'].update(i_window=7, i_nfft=16, i_bins=8, iq_window=8)
    c['model']['heads'].update(parameter_mode='grid_residual', classification_pooling='multiscale_attention_stats', classification_fp32=True)
    c['model']['heads']['multiscale_readout'].update(projection_width=16, modulation_hidden=32, protocol_hidden=16, parameter_hidden=16, dropout=0.)
    options = training_options(resolve_config(c))
    schema = {'classification_labels': {'modulation': [f'm{i}' for i in range(23)], 'protocol': [f'p{i}' for i in range(8)]}}
    scaler, grids = {'mean': [0.] * 7, 'std': [1.] * 7}, [[0., 1.]] * 7
    return RFModel(options['model'], schema, scaler, grids), options, c


class HierarchicalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(23)

    def test_full_resolution_recurrence_stages_and_padding_invariance(self):
        model, _, _ = fixture()
        model.eval()
        iq, lengths = torch.randn(2, 1024, 2), torch.tensor([512, 1024])
        recurrent_shapes, results = [], []
        a = model.backbone.recurrent.rnn.register_forward_pre_hook(lambda m, args: recurrent_shapes.append(tuple(args[0].shape)))
        b = model.backbone.readout_fusion.register_forward_hook(lambda m, args, out: results.append(out))
        with torch.no_grad():
            mixed = model(iq, lengths)
            native = model(iq[:1, :512], lengths[:1])
            changed = iq.clone()
            changed[0, 512:] = torch.randn(512, 2) * 100
            polluted = model(changed, lengths)
        a.remove(); b.remove()
        self.assertEqual(recurrent_shapes[:2], [(1, 512, 24), (1, 1024, 24)])
        self.assertEqual([v.sum(1).tolist() for v in results[0]['valid']], [[512, 1024], [256, 512], [128, 256]])
        self.assertEqual(results[0]['memory_valid'].sum(1).tolist(), [896, 1792])
        for key, shape in [('modulation', (2, 23)), ('protocol', (2, 8)), ('parameters', (2, 7))]:
            self.assertEqual(mixed[key].shape, shape)
            torch.testing.assert_close(mixed[key][:1], native[key], atol=4e-5, rtol=4e-5)
            torch.testing.assert_close(mixed[key], polluted[key], atol=0., rtol=0.)

    def test_each_task_trains_time_frequency_lstm_and_all_stages(self):
        model, options, _ = fixture()
        base = dict(iq=torch.randn(2, 512, 2), modulation=torch.tensor([-1, -1]), protocol=torch.tensor([-1, -1]),
                    targets=torch.zeros(2, 7), target_valid=torch.zeros(2, 7, dtype=torch.bool))
        for task in ['modulation', 'protocol', 'parameter']:
            with self.subTest(task=task):
                batch = {k: v.clone() for k, v in base.items()}
                if task == 'parameter':
                    batch['targets'][:, 3] = 1.
                    batch['target_valid'][:, 3] = True
                else:
                    batch[task][:] = torch.tensor([0, 1])
                model.zero_grad(set_to_none=True)
                out = model(batch['iq'], torch.tensor([512, 512]))
                loss, _ = supervised_loss(out, batch, {'mean': [0.] * 7, 'std': [1.] * 7}, [[0., 1.]] * 7, options)
                loss.backward()
                clear_inactive_head_gradients(model, active_heads(batch, options), torch.device('cpu'))
                for module in [model.backbone.time, model.backbone.spectrum.i_embedding,
                               model.backbone.spectrum.iq_embedding, model.backbone.recurrent, *model.backbone.stages,
                               model.backbone.readout_fusion]:
                    gradients = [p.grad for p in module.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)

    def test_no_unused_shared_parameters_on_joint_loss(self):
        model, _, _ = fixture()
        out = model(torch.randn(1, 512, 2), torch.tensor([512]))
        loss = sum(out[k].square().mean() for k in ['modulation', 'protocol', 'parameters'])
        loss = loss + sum(x.square().mean() for x in out['grid_logits'].values())
        loss.backward()
        self.assertEqual([n for n, p in model.backbone.named_parameters() if p.grad is None], [])

    def test_bfloat16_stages_with_fp32_recurrence(self):
        model, _, _ = fixture()
        for precision in ['fp32', 'autocast']:
            with self.subTest(recurrent_precision=precision):
                model.backbone.recurrent.precision = precision
                model.zero_grad(set_to_none=True)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    out = model(torch.randn(1, 512, 2), torch.tensor([512]))
                    loss = sum(out[k].square().mean() for k in ['modulation', 'protocol', 'parameters'])
                loss.backward()
                self.assertEqual(out['modulation'].dtype, torch.float32)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_field_specific_adamw_updates(self):
        model, options, c = fixture()
        optimizer = build_optimizer(model, c['optimizer'])
        before = [deepcopy(h.state_dict()) for h in model.heads['regression']]
        batch = dict(iq=torch.randn(1, 512, 2), modulation=torch.tensor([-1]), protocol=torch.tensor([-1]),
                     targets=torch.full((1, 7), float('nan')), target_valid=torch.zeros(1, 7, dtype=torch.bool))
        batch['target_valid'][:, 3] = True
        batch['targets'][:, 3] = 1.
        out = model(batch['iq'], torch.tensor([512]))
        loss, _ = supervised_loss(out, batch, {'mean': [0.] * 7, 'std': [1.] * 7}, [[0., 1.]] * 7, options)
        loss.backward()
        clear_inactive_head_gradients(model, active_heads(batch, options), torch.device('cpu'))
        optimizer.step()
        for j, head in enumerate(model.heads['regression']):
            self.assertEqual(all(torch.equal(before[j][k], v) for k, v in head.state_dict().items()), j != 3)

    def test_portable_frequency_backend(self):
        model, _, _ = fixture()
        model.eval()
        iq, lengths = torch.randn(1, 512, 2), torch.tensor([512])
        with torch.no_grad():
            fft = model(iq, lengths)
            model.config['spectral_backend'] = 'matmul'
            portable = model(iq, lengths)
        for key in ['modulation', 'protocol', 'parameters']:
            torch.testing.assert_close(fft[key], portable[key], atol=3e-4, rtol=3e-4)

    def test_safetensors_predictor_and_strict_warmstart(self):
        model, _, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_safetensors(root / 'model.safetensors', model.state_dict())
            write_json(root / 'config.json', dict(model_config=model.config, schema=model.schema,
                target_scaler={'mean': [0.] * 7, 'std': [1.] * 7}, parameter_grids=[[0., 1.]] * 7))
            predictor = Predictor(root, device='cpu', threads=2)
            records = [dict(task=t, sample_id=str(i), iq=torch.randn(512 if i % 2 else 1024, 2).tolist())
                       for i, t in enumerate(['modulation', 'protocol', 'parameter'])]
            outputs = predictor.predict(records)
        for row, out in zip(records, outputs):
            self.assertEqual(set(out), {'task', 'sample_id', 'prediction'})
            self.assertEqual((out['task'], out['sample_id']), (row['task'], row['sample_id']))
            if row['task'] == 'parameter':
                self.assertEqual(set(out['prediction']), set(TARGETS))
            else:
                self.assertIn(out['prediction'], model.schema['classification_labels'][row['task']])
        other, _, _ = fixture()
        load_with_training_aux(other, model.state_dict(), model.config)
        self.assertTrue(all(torch.equal(v, other.state_dict()[k]) for k, v in model.state_dict().items()))
        from training.test_multiresolution import fixture as previous_fixture
        previous, _, _ = previous_fixture()
        with self.assertRaisesRegex(ValueError, 'matching'):
            load_with_training_aux(model, previous.state_dict(), previous.config)

    def test_invalid_configuration_fails_early(self):
        _, _, base = fixture()
        mutations = [lambda c: c['model']['pyramid'].update(enabled=True),
                     lambda c: c['model']['hierarchy'].update(stage_widths=[]),
                     lambda c: c['model']['hierarchy'].update(downsample_strides=[3, 2]),
                     lambda c: c['model']['hierarchy'].update(fusion_heads=5),
                     lambda c: c['model']['hierarchy']['spectrum'].update(i_bins=100),
                     lambda c: c['training'].update(stage='pretrain')]
        for mutate in mutations:
            c = deepcopy(base)
            mutate(c)
            with self.assertRaises(ValueError):
                resolve_config(c)


if __name__ == '__main__':
    unittest.main(verbosity=2)
