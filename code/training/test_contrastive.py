"""Cross-record supervision, no-positive stability, and export/warmstart compatibility."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from models.rf_model import RFModel
from utils.contrastive import supervised_contrastive, load_with_training_aux
from utils.config import DEFAULTS, training_options
from utils.signal_statistics import signal_statistics
from utils.engine import supervised_loss, patch_mask
from utils.head_optimizer import active_heads, clear_inactive_head_gradients


class ContrastiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_class_aligned_features_have_lower_loss_than_mixed_features(self):
        good = torch.tensor([[1., 0.], [1., .01], [0., 1.], [.01, 1.]], requires_grad=True)
        labels, rows = torch.tensor([0, 0, 1, 1]), torch.arange(4)
        aligned, stats = supervised_contrastive(good, labels, rows, gather=False)
        mixed, _ = supervised_contrastive(good[[0, 2, 1, 3]], labels, rows, gather=False)
        self.assertLess(float(aligned.detach()), float(mixed.detach()))
        self.assertEqual(stats['coverage'], 1.)
        aligned.backward()
        self.assertTrue(torch.isfinite(good.grad).all())

    def test_duplicate_record_is_not_a_positive_and_unique_labels_are_safe(self):
        x = torch.randn(3, 4, requires_grad=True)
        loss, stats = supervised_contrastive(x, torch.tensor([0, 0, -1]), torch.tensor([7, 7, 9]), gather=False)
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(stats['coverage'], 0.)
        loss.backward()
        self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))

    def test_warmstart_adds_only_training_head_and_export_keeps_predictions(self):
        config = deepcopy(DEFAULTS)
        config['model']['encoder'].update(width=32, depth=1, heads=4, dropout=0., attention_dropout=0.)
        options = training_options(config)
        args = dict(schema={'classification_labels': {'modulation': ['a', 'b'], 'protocol': ['x', 'y']}},
                    target_scaler={'mean': [0.] * 7, 'std': [1.] * 7}, parameter_grids=[[]] * 7)
        base = RFModel(options['model'], **args).eval()
        training = RFModel({**options['model'], 'contrastive_head': True}, **args).eval()
        load_with_training_aux(training, base.state_dict())
        x, lengths = torch.randn(2, 512, 2), torch.tensor([512, 512])
        with torch.no_grad():
            a, b = base(x, lengths), training(x, lengths)
        for key in ['modulation', 'protocol', 'parameters']:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        self.assertEqual(base.parameter_report()['final_parameters'], training.parameter_report()['final_parameters'])
        load_with_training_aux(base, training.state_dict())
        broken = {k: v for k, v in base.state_dict().items() if k != 'heads.modulation.0.weight'}
        with self.assertRaises(ValueError):
            load_with_training_aux(base, broken)

    def test_statistics_pooling_warmstart_preserves_old_predictions(self):
        config = deepcopy(DEFAULTS)
        config['model']['encoder'].update(width=32, depth=1, heads=4, dropout=0., attention_dropout=0.)
        options = training_options(config)
        args = dict(schema={'classification_labels': {'modulation': ['a', 'b'], 'protocol': ['x', 'y']}},
                    target_scaler={'mean': [0.] * 7, 'std': [1.] * 7}, parameter_grids=[[]] * 7)
        base = RFModel(options['model'], **args).eval()
        richer = RFModel({**options['model'], 'classification_pooling': 'mean_std'}, **args).eval()
        load_with_training_aux(richer, base.state_dict())
        x, lengths = torch.randn(2, 512, 2), torch.tensor([512, 512])
        with torch.no_grad():
            a, b = base(x, lengths), richer(x, lengths)
        for key in ['modulation', 'protocol', 'parameters']:
            torch.testing.assert_close(a[key], b[key], rtol=1e-5, atol=1e-6)

    def test_physical_statistics_are_finite_phase_invariant_and_padding_safe(self):
        torch.manual_seed(91)
        x = torch.randn(2, 512, 2)
        x = x + torch.roll(x, 1, dims=1)
        lengths = torch.tensor([512, 512])
        base = signal_statistics(x, lengths)
        self.assertEqual(base.shape, (2, 32))
        rotated = torch.stack((-x[..., 1], x[..., 0]), -1)
        torch.testing.assert_close(base, signal_statistics(rotated, lengths), atol=3e-4, rtol=3e-4)
        padded = torch.nn.functional.pad(x, (0, 0, 0, 512))
        torch.testing.assert_close(base, signal_statistics(padded, lengths), atol=3e-4, rtol=3e-4)
        self.assertTrue(torch.isfinite(signal_statistics(torch.zeros_like(x), lengths)).all())

    def test_local_refinement_preserves_shared_initialization_and_old_predictions(self):
        config = deepcopy(DEFAULTS)
        config['model']['encoder'].update(width=32, depth=1, heads=4, dropout=0., attention_dropout=0.)
        options = training_options(config)
        args = dict(schema={'classification_labels': {'modulation': ['a', 'b'], 'protocol': ['x', 'y']}},
                    target_scaler={'mean': [0.] * 7, 'std': [1.] * 7}, parameter_grids=[[]] * 7)
        torch.manual_seed(80)
        base = RFModel(options['model'], **args).eval()
        torch.manual_seed(80)
        refined = RFModel({**options['model'], 'local_refinement': True}, **args).eval()
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, refined.state_dict()[key], rtol=0, atol=0)
        x, lengths = torch.randn(2, 512, 2), torch.tensor([512, 512])
        with torch.no_grad():
            a, b = base(x, lengths), refined(x, lengths)
        for key in ['modulation', 'protocol', 'parameters']:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        load_with_training_aux(refined, base.state_dict(), base.config)
        refined(x, lengths)['modulation'].sum().backward()
        self.assertGreater(float(refined.backbone.time.refinement[2].weight.grad.abs().sum()), 0.)

    def test_new_feature_layout_warmstart_and_masking_do_not_leak_targets(self):
        config = deepcopy(DEFAULTS)
        config['model']['encoder'].update(width=32, depth=1, heads=4, dropout=0., attention_dropout=0.)
        options = training_options(config)
        args = dict(schema={'classification_labels': {'modulation': ['a', 'b'], 'protocol': ['x', 'y']}},
                    target_scaler={'mean': [0.] * 7, 'std': [1.] * 7}, parameter_grids=[[]] * 7)
        base = RFModel({**options['model'], 'physical_statistics': True}, **args).eval()
        rich = RFModel({**base.config, 'classification_pooling': 'mean_std'}, **args).eval()
        load_with_training_aux(rich, base.state_dict(), base.config)
        x, lengths = torch.randn(2, 512, 2), torch.tensor([512, 512])
        with torch.no_grad():
            a, b = base(x, lengths), rich(x, lengths)
        torch.testing.assert_close(a['modulation'], b['modulation'], atol=2e-5, rtol=2e-5)
        mask = patch_mask(lengths, 16, .3, 2, 5)
        changed = x.clone()
        changed[mask.repeat_interleave(16, 1)] += 1000
        with torch.no_grad():
            a, b = rich(x, lengths, patch_mask=mask), rich(changed, lengths, patch_mask=mask)
        torch.testing.assert_close(a['modulation'], b['modulation'], atol=0, rtol=0)

    def test_projection_momentum_is_skipped_without_cross_record_positives(self):
        config = deepcopy(DEFAULTS)
        config['model']['encoder'].update(width=32, depth=1, heads=4, dropout=0., attention_dropout=0.)
        config['loss']['supervised']['contrastive']['weight'] = .1
        options = training_options(config)
        scaler = {'mean': [0.] * 7, 'std': [1.] * 7}
        model = RFModel({**options['model'], 'contrastive_head': True},
            {'classification_labels': {'modulation': ['a', 'b', 'c', 'd'], 'protocol': ['x', 'y']}}, scaler, [[]] * 7)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        batch = {'iq': torch.randn(4, 512, 2), 'lengths': torch.tensor([512] * 4), 'rows': [0, 1, 2, 3],
                 'modulation': torch.tensor([0, 0, 1, 1]), 'protocol': torch.tensor([-1] * 4),
                 'targets': torch.zeros(4, 7), 'target_valid': torch.zeros(4, 7, dtype=torch.bool)}
        for labels in [torch.tensor([0, 0, 1, 1]), torch.arange(4)]:
            batch['modulation'] = labels
            before = deepcopy(model.aux['contrastive'].state_dict())
            optimizer.zero_grad(set_to_none=True)
            loss, parts = supervised_loss(model(batch['iq'], batch['lengths']), batch, scaler, [[]] * 7, options)
            loss.backward()
            active = active_heads(batch, options)
            if parts['supcon_coverage'] > 0:
                active.add('aux.contrastive.')
            clear_inactive_head_gradients(model, active, torch.device('cpu'))
            optimizer.step()
            if parts['supcon_coverage'] > 0:
                self.assertGreater(float(optimizer.state[model.aux['contrastive'][0].weight]['exp_avg'].abs().sum()), 0.)
            if parts['supcon_coverage'] == 0:
                for key, value in before.items():
                    torch.testing.assert_close(value, model.aux['contrastive'].state_dict()[key], atol=0, rtol=0)

    def test_classification_fp32_is_retained_inside_mixed_precision(self):
        c = deepcopy(DEFAULTS)
        c['model']['encoder'].update(width=32, depth=1, heads=4)
        options = training_options(c)
        model = RFModel({**options['model'], 'classification_fp32': True, 'physical_statistics': True,
                         'classification_pooling': 'mean_std'},
            {'classification_labels': {'modulation': ['a', 'b'], 'protocol': ['x', 'y']}},
            {'mean': [0.] * 7, 'std': [1.] * 7}, [[]] * 7)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            output = model(torch.randn(2, 512, 2), torch.tensor([512, 512]))
        self.assertEqual(output['modulation'].dtype, torch.float32)
        output['modulation'].sum().backward()
        self.assertTrue(torch.isfinite(model.heads['modulation'][0].weight.grad).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
