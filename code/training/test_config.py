"""Configuration wiring, optimizer groups and learning-rate schedule checks."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from models.rf_model import RFModel
from training.train import build_optimizer
from utils.config import DEFAULTS, load_config, make_scheduler, resolve_config, training_options
from utils.common import read_config_json
from utils.engine import supervised_loss


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_all_shipped_experiments_resolve(self):
        root = Path(__file__).resolve().parents[1] / 'configs'
        paths = [path for path in sorted(root.rglob('*.json'))
                 if read_config_json(path).get('schema_version') == 2]
        self.assertTrue(paths, 'No training configurations discovered')
        for path in paths:
            with self.subTest(config=str(path.relative_to(root))):
                config = load_config(path)
                options = training_options(config)
                self.assertEqual(options['batch_size'], config['data']['batch_size'])
                self.assertEqual(options['model']['width'], config['model']['encoder']['width'])
                self.assertEqual(config['scheduler']['name'], 'cosine')

    def test_comments_preserve_slashes_and_escaped_quotes_inside_strings(self):
        raw = r'''{
          // encoder choices can be commented out without rewriting the file
          "schema_version": 2,
          "experiment": {"description": "https://host/a//b /*keep*/ \"quoted\""},
          /* block comment
             keeps line numbers */
          "model": {"encoder": {"width": 64, "heads": 4}}
        }'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'main.json'
            path.write_text(raw)
            config = load_config(path)
        self.assertEqual(config['model']['encoder']['width'], 64)
        self.assertEqual(config['experiment']['description'], 'https://host/a//b /*keep*/ "quoted"')

    def test_unknown_or_inconsistent_settings_fail(self):
        bad = deepcopy(DEFAULTS)
        bad['scheduler']['min_lr_raito'] = 0.1
        with self.assertRaisesRegex(ValueError, 'min_lr_raito'):
            resolve_config(bad)
        bad = deepcopy(DEFAULTS)
        bad['scheduler']['warmup_ratio'] = 0.1
        with self.assertRaisesRegex(ValueError, 'OR'):
            resolve_config(bad)
        bad = deepcopy(DEFAULTS)
        bad['model']['signal']['position'] = 'physical'
        with self.assertRaisesRegex(ValueError, 'sampling-rate'):
            resolve_config(bad)

    def test_legacy_config_preserves_model_and_loss_options(self):
        config = resolve_config({'model': {'width': 64, 'heads': 4, 'dropout': 0.0}, 'batch_size': 8,
                                 'task_weights': [1., 2., 3.], 'reconstruction_weight': .1})
        options = training_options(config)
        self.assertEqual(options['model']['attention_dropout'], 0.)
        self.assertEqual(options['task_weights'], [1., 2., 3.])
        self.assertEqual(options['reconstruction_weight'], .1)

    def test_cosine_endpoints_group_ratios_and_resume(self):
        a, b = torch.nn.Parameter(torch.ones(())), torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([{'params': [a], 'lr': 0.001}, {'params': [b], 'lr': 0.01}])
        settings = dict(name='cosine', warmup_steps=2, warmup_ratio=0., warmup_start_factor=0., min_lr_ratio=.1)
        scheduler, warmup = make_scheduler(optimizer, settings, 8)
        rates = []
        for _ in range(8):
            rates.append([g['lr'] for g in optimizer.param_groups])
            optimizer.step()
            scheduler.step()
            if len(rates) == 4:
                saved_optimizer, saved_scheduler = deepcopy(optimizer.state_dict()), deepcopy(scheduler.state_dict())
        self.assertEqual(warmup, 2)
        self.assertAlmostEqual(rates[0][1], .005)
        self.assertAlmostEqual(rates[1][1], .01)
        self.assertAlmostEqual(rates[-1][1], .001)
        for x, y in rates:
            self.assertAlmostEqual(x / y, .1)
        self.assertTrue(all(x[1] >= y[1] for x, y in zip(rates[2:], rates[3:])))
        restored = torch.optim.SGD([{'params': [a], 'lr': .001}, {'params': [b], 'lr': .01}])
        restored_schedule, _ = make_scheduler(restored, settings, 8)
        restored.load_state_dict(saved_optimizer)
        restored_schedule.load_state_dict(saved_scheduler)
        for expected in rates[4:]:
            self.assertEqual([g['lr'] for g in restored.param_groups], expected)
            restored.step()
            restored_schedule.step()

    def test_ratio_warmup_and_no_warmup(self):
        for steps, ratio, expected in [(0, .2, 2), (0, 0., 0)]:
            parameter = torch.nn.Parameter(torch.ones(()))
            optimizer = torch.optim.SGD([parameter], lr=.01)
            settings = dict(name='cosine', warmup_steps=steps, warmup_ratio=ratio, warmup_start_factor=0., min_lr_ratio=.05)
            scheduler, warmup = make_scheduler(optimizer, settings, 10)
            self.assertEqual(warmup, expected)
            for _ in range(9):
                optimizer.step()
                scheduler.step()
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .0005)
        settings['warmup_steps'] = 10
        with self.assertRaisesRegex(ValueError, 'planned optimizer updates'):
            make_scheduler(optimizer, settings, 10)
        settings['warmup_steps'] = 2
        schedule, _ = make_scheduler(optimizer, settings, 3)
        for _ in range(2):
            optimizer.step()
            schedule.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .0005)

    def test_optimizer_decay_exclusions_and_model_scales_are_wired(self):
        c = deepcopy(DEFAULTS)
        c['model']['encoder'].update(width=32, depth=1, heads=4, mlp_ratio=2., attention_dropout=.2, rope_base=2000.)
        c['model']['heads'].update(classification_hidden_ratio=.75, parameter_query_temperature=.8, grid_residual_scale=.2)
        c['optimizer'].update(backbone_lr_factor=.1, exclude_bias_and_norm_from_decay=True)
        options = training_options(c)
        model = RFModel(options['model'], {'classification_labels': {'modulation': ['a', 'b'], 'protocol': ['x', 'y']}},
                        {'mean': [0.] * 7, 'std': [1.] * 7}, [[]] * 7)
        optimizer = build_optimizer(model, c['optimizer'])
        self.assertEqual(model.backbone.blocks[0].mlp[0].out_features, 64)
        self.assertEqual(model.heads['modulation'][0].out_features, 24)
        self.assertEqual(model.backbone.blocks[0].attention.dropout, .2)
        assigned = {id(p): group for group in optimizer.param_groups for p in group['params']}
        self.assertEqual(len(assigned), len(list(model.parameters())))
        for name, p in model.named_parameters():
            group = assigned[id(p)]
            self.assertEqual(group['weight_decay'], 0. if p.ndim < 2 or name.endswith('.bias') else .01)
            self.assertAlmostEqual(group['lr'], .00003 if name.startswith('backbone.') else .0003)

    def test_parameter_weights_and_huber_delta_change_actual_loss(self):
        output = {'modulation': torch.zeros(1, 2), 'protocol': torch.zeros(1, 2),
                  'parameters': torch.tensor([[1., 3., 0., 0., 0., 0., 0.]], requires_grad=True), 'grid_logits': {}}
        batch = {'iq': torch.zeros(1, 512, 2), 'modulation': torch.tensor([-1]), 'protocol': torch.tensor([-1]),
                 'targets': torch.zeros(1, 7), 'target_valid': torch.tensor([[True, True, False, False, False, False, False]])}
        scaler = {'mean': [0.] * 7, 'std': [1.] * 7}
        config = {'parameter_weights': [3., 1., 0., 0., 0., 0., 0.], 'regression_loss': 'mse'}
        loss, _ = supervised_loss(output, batch, scaler, [[]] * 7, config)
        self.assertAlmostEqual(float(loss.detach()), 3.)
        config.update(regression_loss='huber', huber_delta=.5)
        loss, _ = supervised_loss(output, batch, scaler, [[]] * 7, config)
        self.assertAlmostEqual(float(loss.detach()), .625)


if __name__ == '__main__':
    unittest.main(verbosity=2)
