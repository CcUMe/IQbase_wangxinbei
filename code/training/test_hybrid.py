"""Checks for an intact Transformer base with a jointly shared detail path."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from infer import Predictor
from models.rf_model import Backbone, RFModel
from training.test_hierarchical import fixture as staged_fixture
from utils.common import TARGETS, save_safetensors, write_json
from utils.config import resolve_config, training_options
from utils.contrastive import load_with_training_aux
from utils.engine import supervised_loss
from utils.head_optimizer import active_heads, clear_inactive_head_gradients


def fixture():
    _, _, c = staged_fixture()
    c['model']['hybrid']['enabled'] = True
    c['model']['encoder'].update(width=32, depth=1, heads=4, dropout=0., attention_dropout=0., qk_normalization=True)
    c['model']['signal']['phase_features'] = True
    c['model']['hierarchy'].update(stage_widths=[16, 24, 32])
    options = training_options(resolve_config(c))
    schema = {'classification_labels': {'modulation': [f'm{i}' for i in range(23)], 'protocol': [f'p{i}' for i in range(8)]}}
    return RFModel(options['model'], schema, {'mean': [0.] * 7, 'std': [1.] * 7}, [[0., 1.]] * 7), options, c


class HybridTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        torch.set_num_threads(2)

    def test_original_base_keys_shapes_weights_and_native_features_are_preserved(self):
        model, _, _ = fixture()
        original = Backbone(model.config).eval()
        result = model.backbone.load_state_dict(original.state_dict(), strict=False)
        self.assertFalse(result.unexpected_keys)
        self.assertTrue(result.missing_keys)
        self.assertTrue(all(k.startswith(('detail.', 'hybrid_fusion.')) for k in result.missing_keys))
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, model.backbone.state_dict()[key]), key)
        model.eval()
        iq, lengths = torch.randn(1, 512, 2), torch.tensor([512])
        with torch.no_grad():
            old = original(iq, lengths)[0]
            retained = model.encode(iq, lengths)[0]
        torch.testing.assert_close(old, retained, atol=1e-6, rtol=1e-6)
        self.assertIsNone(model.backbone.detail.readout_fusion)
        self.assertIsNone(model.backbone.detail.global_spectrum)

    def test_three_resolutions_and_padding(self):
        model, _, _ = fixture()
        model.eval()
        observed = []
        hook = model.backbone.hybrid_fusion.register_forward_hook(lambda m, a, o: observed.append(o))
        iq, lengths = torch.randn(2, 1024, 2), torch.tensor([512, 1024])
        with torch.no_grad():
            mixed = model(iq, lengths)
            native = model(iq[:1, :512], lengths[:1])
            changed = iq.clone(); changed[0, 512:] += 500
            polluted = model(changed, lengths)
        hook.remove()
        self.assertEqual([v.sum(1).tolist() for v in observed[0]['valid']], [[32, 64], [128, 256], [512, 1024]])
        self.assertEqual(observed[0]['memory_valid'].sum(1).tolist(), [672, 1344])
        for key in ['modulation', 'protocol', 'parameters']:
            torch.testing.assert_close(mixed[key][:1], native[key], atol=4e-5, rtol=4e-5)
            torch.testing.assert_close(mixed[key], polluted[key], atol=0., rtol=0.)

    def test_all_tasks_train_the_retained_base_and_shared_detail(self):
        model, options, _ = fixture()
        for task in ['modulation', 'protocol', 'parameter']:
            with self.subTest(task=task):
                batch = dict(iq=torch.randn(2, 512, 2), modulation=torch.tensor([-1, -1]), protocol=torch.tensor([-1, -1]),
                    targets=torch.zeros(2, 7), target_valid=torch.zeros(2, 7, dtype=torch.bool))
                if task == 'parameter':
                    batch['target_valid'][:, 3] = True
                    batch['targets'][:, 3] = 1.
                else:
                    batch[task][:] = torch.tensor([0, 1])
                model.zero_grad(set_to_none=True)
                out = model(batch['iq'], torch.tensor([512, 512]))
                loss, _ = supervised_loss(out, batch, {'mean': [0.] * 7, 'std': [1.] * 7}, [[0., 1.]] * 7, options)
                loss.backward()
                clear_inactive_head_gradients(model, active_heads(batch, options), torch.device('cpu'))
                for module in [model.backbone.time, model.backbone.blocks, model.backbone.global_spectrum,
                               model.backbone.detail.time, model.backbone.detail.spectrum, model.backbone.detail.recurrent,
                               *model.backbone.detail.stages, model.backbone.hybrid_fusion]:
                    gradients = [p.grad for p in module.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
                self.assertEqual([n for n, p in model.backbone.named_parameters() if p.grad is None], [])

    def test_bfloat16_backward(self):
        model, _, _ = fixture()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = model(torch.randn(1, 512, 2), torch.tensor([512]))
            loss = sum(out[k].square().mean() for k in ['modulation', 'protocol', 'parameters'])
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_safetensors_predictor_warmstart_and_architecture_guards(self):
        model, _, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_safetensors(root / 'model.safetensors', model.state_dict())
            write_json(root / 'config.json', dict(model_config=model.config, schema=model.schema,
                target_scaler={'mean': [0.] * 7, 'std': [1.] * 7}, parameter_grids=[[0., 1.]] * 7))
            predictor = Predictor(root, device='cpu', threads=2)
            rows = [dict(task=t, sample_id=str(i), iq=torch.randn(512 if i % 2 else 1024, 2).tolist())
                    for i, t in enumerate(['modulation', 'protocol', 'parameter'])]
            out = predictor.predict(rows)
        for row, result in zip(rows, out):
            self.assertEqual(set(result), {'task', 'sample_id', 'prediction'})
            self.assertEqual((row['task'], row['sample_id']), (result['task'], result['sample_id']))
            if row['task'] == 'parameter':
                self.assertEqual(set(result['prediction']), set(TARGETS))
            else:
                self.assertIn(result['prediction'], model.schema['classification_labels'][row['task']])
        other, _, _ = fixture()
        load_with_training_aux(other, model.state_dict(), model.config)
        old, _, _ = staged_fixture()
        with self.assertRaisesRegex(ValueError, 'matching'):
            load_with_training_aux(model, old.state_dict(), old.config)

    def test_configuration_keeps_original_encoder_controls(self):
        _, _, c = fixture()
        c['model']['encoder'].update(width=64, depth=2)
        options = training_options(resolve_config(c))
        self.assertEqual((options['model']['width'], options['model']['depth']), (64, 2))
        c['model']['hierarchy']['enabled'] = False
        with self.assertRaisesRegex(ValueError, 'requires hierarchy'):
            resolve_config(c)


if __name__ == '__main__':
    unittest.main(verbosity=2)
