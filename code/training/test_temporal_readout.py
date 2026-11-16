"""Semantic checks for valid-length segment pooling and modulation-only changes."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from models.multiresolution import SegmentedStatistics, TemporalStatisticsReadout, temporal_readout_options
from models.rf_model import RFModel
from training.test_multiresolution import fixture
from utils.config import resolve_config, training_options
from utils.contrastive import load_with_training_aux


class TemporalReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)

    def models(self):
        template, _, config = fixture()
        schema, scaler, grids = template.schema, {'mean': [0.] * 7, 'std': [1.] * 7}, [[0., 1.]] * 7
        torch.manual_seed(31)
        baseline = RFModel(training_options(resolve_config(config))['model'], schema, scaler, grids)
        config['model']['heads']['modulation_temporal'].update(enabled=True, segments=4, segment_width=8,
                                                                dropout=0., temporal_kernel=3)
        torch.manual_seed(31)
        temporal = RFModel(training_options(resolve_config(config))['model'], schema, scaler, grids)
        return baseline, temporal, config

    def test_pool_uses_valid_ranks_handles_empty_segments_and_ignores_padding(self):
        pool = SegmentedStatistics(4, 1e-6)
        h = torch.tensor([[[1.], [99.], [3.], [5.], [99.], [7.], [9.]]])
        valid = torch.tensor([[True, False, True, True, False, True, True]])
        mean, std, present = pool(h, valid)
        torch.testing.assert_close(mean.flatten(), torch.tensor([2., 5., 7., 9.]))
        torch.testing.assert_close(std.flatten(), torch.tensor([1., .001, .001, .001]))
        self.assertTrue(present.all())
        dirty = h.clone(); dirty[~valid] = float('nan')
        for a, b in zip(pool(dirty, valid), (mean, std, present)):
            torch.testing.assert_close(a, b, atol=0., rtol=0.)
        short = torch.tensor([[[2.], [4.]]])
        a, b, mask = pool(short, torch.ones(1, 2, dtype=torch.bool))
        self.assertEqual(mask.tolist(), [[True, False, True, False]])
        self.assertTrue(torch.isfinite(a).all() and torch.isfinite(b).all())
        self.assertEqual(b[~mask].abs().sum().item(), 0.)

    def test_two_pass_variance_retains_small_variation_at_large_offset(self):
        pool = SegmentedStatistics(2, 1e-6)
        h = (torch.arange(8).float() + 1_000_000).reshape(1, 8, 1).requires_grad_()
        mean, std, _ = pool(h, torch.ones(1, 8, dtype=torch.bool))
        torch.testing.assert_close(std, torch.full((1, 2, 1), 1.25 ** .5))
        (mean.sum() + std.sum()).backward()
        self.assertTrue(torch.isfinite(h.grad).all())

    def test_local_readout_preserves_order_beyond_global_moments(self):
        options = temporal_readout_options({'enabled': True, 'segment_width': 8, 'dropout': 0.})
        head = TemporalStatisticsReadout(4, 12, 1e-6, options).eval()
        h = torch.randn(1, 4, 4).repeat_interleave(3, 1)
        reversed_h = h.flip(1)
        torch.testing.assert_close(h.mean(1), reversed_h.mean(1))
        torch.testing.assert_close(h.var(1), reversed_h.var(1))
        mask = torch.ones(1, 12, dtype=torch.bool)
        self.assertGreater((head(h, mask) - head(reversed_h, mask)).abs().max().item(), 1e-4)

    def test_common_initialization_and_other_task_outputs_are_unchanged(self):
        baseline, model, _ = self.models()
        old, new = baseline.state_dict(), model.state_dict()
        for key in old:
            torch.testing.assert_close(old[key], new[key], atol=0., rtol=0., msg=key)
        self.assertTrue(all(k.startswith('heads.modulation.temporal.') for k in new.keys() - old.keys()))
        baseline.eval(); model.eval()
        iq, lengths = torch.randn(2, 512, 2), torch.tensor([512, 512])
        with torch.no_grad():
            a, b = baseline(iq, lengths), model(iq, lengths)
        for task in ['protocol', 'parameters']:
            torch.testing.assert_close(a[task], b[task], atol=0., rtol=0.)
        for key in a['grid_logits']:
            torch.testing.assert_close(a['grid_logits'][key], b['grid_logits'][key], atol=0., rtol=0.)
        self.assertFalse(torch.equal(a['modulation'], b['modulation']))
        with self.assertRaisesRegex(ValueError, 'matching modulation_temporal'):
            load_with_training_aux(model, old, baseline.config)

    def test_full_model_padding_and_bfloat16_backward(self):
        _, model, _ = self.models(); model.eval()
        iq, lengths = torch.randn(2, 1024, 2), torch.tensor([512, 1024])
        with torch.no_grad():
            padded = model(iq, lengths)
            native = model(iq[:1, :512], lengths[:1])
            dirty = iq.clone(); dirty[0, 512:] = 1000 * torch.randn_like(dirty[0, 512:])
            changed = model(dirty, lengths)
        for task in ['modulation', 'protocol', 'parameters']:
            torch.testing.assert_close(padded[task][:1], native[task], atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(padded[task], changed[task], atol=0., rtol=0.)
        model.train(); model.zero_grad(set_to_none=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = model(iq[:1, :512], lengths[:1])
            loss = torch.nn.functional.cross_entropy(output['modulation'], torch.tensor([2]))
        loss.backward()
        for component in [*model.heads['modulation'].temporal, model.backbone.time,
                          *model.backbone.pyramid.branches]:
            gradients = [p.grad for p in component.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)

    def test_bad_configuration_is_rejected(self):
        _, _, config = self.models()
        for key, value in [('segments', 0), ('segment_width', 0), ('temporal_kernel', 2),
                           ('dropout', 1.), ('residual_scale', -1.)]:
            bad = deepcopy(config); bad['model']['heads']['modulation_temporal'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'modulation_temporal'):
                resolve_config(bad)
        bad = deepcopy(config); bad['model']['pyramid']['enabled'] = False
        with self.assertRaisesRegex(ValueError, 'requires the pyramid'):
            resolve_config(bad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
