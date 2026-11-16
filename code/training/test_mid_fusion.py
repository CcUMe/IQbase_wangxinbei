"""Causal, masking and compatibility checks for mid-encoder scale exchange."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from models.multiresolution import MidScaleFusion, mid_fusion_options
from models.rf_model import RFModel
from training.test_multiresolution import fixture
from utils.config import resolve_config, training_options
from utils.contrastive import load_with_training_aux


class MidFusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(29)

    def models(self):
        template, _, config = fixture()
        config['model']['encoder']['depth'] = 2
        schema, scaler, grids = template.schema, {'mean': [0.] * 7, 'std': [1.] * 7}, [[0., 1.]] * 7
        torch.manual_seed(43)
        baseline = RFModel(training_options(resolve_config(config))['model'], schema, scaler, grids)
        config['model']['mid_fusion'].update(enabled=True, after_block=1, tokens=4, width=24, heads=3, dropout=0.)
        torch.manual_seed(43)
        model = RFModel(training_options(resolve_config(config))['model'], schema, scaler, grids)
        return baseline, model, config

    def test_common_weights_and_zero_residual_recover_baseline(self):
        baseline, model, _ = self.models()
        a, b = baseline.state_dict(), model.state_dict()
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0., rtol=0., msg=key)
        self.assertTrue(all(k.startswith('backbone.pyramid.mid_fusion.') for k in b.keys() - a.keys()))
        baseline.eval(); model.eval()
        with torch.no_grad():
            for gain in model.backbone.pyramid.mid_fusion.gains:
                gain.zero_()
            iq, lengths = torch.randn(2, 512, 2), torch.tensor([512, 512])
            old, new = baseline(iq, lengths), model(iq, lengths)
        for task in ['modulation', 'protocol', 'parameters']:
            torch.testing.assert_close(old[task], new[task], atol=0., rtol=0.)
        with self.assertRaisesRegex(ValueError, 'matching mid_fusion'):
            load_with_training_aux(model, a, baseline.config)

    def test_fine_intervention_reaches_next_coarse_block_but_not_earlier_block(self):
        baseline, model, _ = self.models()
        iq, lengths = torch.randn(1, 512, 2), torch.tensor([512])
        for candidate, expect_change in [(baseline, False), (model, True)]:
            with self.subTest(mid_fusion=expect_change):
                candidate.eval(); early, next_input = [], []
                hooks = [candidate.backbone.blocks[0].register_forward_hook(lambda m, a, o: early.append(o.detach().clone())),
                         candidate.backbone.blocks[1].register_forward_pre_hook(lambda m, a: next_input.append(a[0].detach().clone()))]
                def perturb(module, args, result):
                    h, valid, centers = result
                    delta = torch.sin(torch.arange(h.shape[-1], device=h.device).float()) * .4
                    return h + delta * valid[..., None], valid, centers
                with torch.no_grad():
                    candidate(iq, lengths)
                    handle = candidate.backbone.pyramid.branches[1].register_forward_hook(perturb)
                    candidate(iq, lengths)
                    handle.remove()
                for handle in hooks: handle.remove()
                torch.testing.assert_close(early[0], early[1], atol=0., rtol=0.)
                difference = (next_input[0] - next_input[1]).abs().max().item()
                self.assertGreater(difference, 1e-5) if expect_change else self.assertEqual(difference, 0.)

    def test_details_are_encoded_once_and_encode_contract_is_consistent(self):
        _, model, _ = self.models(); model.eval()
        counts = [0, 0, 0]
        def count(index):
            def hook(m, a, o): counts[index] += 1
            return hook
        modules = [model.backbone.pyramid.stem, *model.backbone.pyramid.branches]
        handles = [module.register_forward_hook(count(i)) for i, module in enumerate(modules)]
        iq, lengths = torch.randn(1, 512, 2), torch.tensor([512])
        with torch.no_grad(): model(iq, lengths)
        for handle in handles: handle.remove()
        self.assertEqual(counts, [1, 1, 1])
        with torch.no_grad():
            encoded, valid, target = model.encode(iq, lengths)
            explicit, mask, _, context = model.backbone(iq, lengths, return_context=True)
        torch.testing.assert_close(encoded, explicit, atol=0., rtol=0.)
        self.assertEqual(encoded.shape, (1, 32, 32))
        self.assertEqual(target.shape, iq.shape)
        self.assertTrue(torch.equal(valid, mask))
        self.assertIn('pyramid_details', context)

    def test_scale_collection_is_invariant_to_duplicating_identical_source_tokens(self):
        options = mid_fusion_options({'enabled': True, 'tokens': 4, 'width': 12, 'heads': 3, 'dropout': 0.})
        fusion = MidScaleFusion([8, 8, 8], options).eval()
        features = [torch.randn(1, n, 8) for n in [4, 6, 10]]
        masks = [torch.ones(h.shape[:2], dtype=torch.bool) for h in features]
        centers = [torch.arange(h.shape[1]).float() for h in features]
        lengths = torch.tensor([512])
        with torch.no_grad():
            before = fusion(features, masks, centers, lengths)
            repeated = fusion([*features[:2], features[2].repeat_interleave(2, 1)],
                              [*masks[:2], masks[2].repeat_interleave(2, 1)],
                              [*centers[:2], centers[2].repeat_interleave(2)], lengths)
        for i in [0, 1]: torch.testing.assert_close(before[i], repeated[i], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(before[2].repeat_interleave(2, 1), repeated[2], atol=1e-6, rtol=1e-6)

    def test_native_padding_and_all_task_backward(self):
        _, model, _ = self.models(); model.eval()
        iq, lengths = torch.randn(2, 1024, 2), torch.tensor([512, 1024])
        with torch.no_grad():
            padded = model(iq, lengths); native = model(iq[:1, :512], lengths[:1])
            dirty = iq.clone(); dirty[0, 512:] = 1000 * torch.randn_like(dirty[0, 512:])
            changed = model(dirty, lengths)
        for task in ['modulation', 'protocol', 'parameters']:
            torch.testing.assert_close(padded[task][:1], native[task], atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(padded[task], changed[task], atol=0., rtol=0.)
        model.train()
        fusion = model.backbone.pyramid.mid_fusion
        for task in ['modulation', 'protocol', 'parameters']:
            with self.subTest(task=task):
                model.zero_grad(set_to_none=True)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    output = model(iq[:1, :512], lengths[:1])
                    loss = (output[task].square().mean() if task == 'parameters' else
                            torch.nn.functional.cross_entropy(output[task], torch.tensor([1])))
                loss.backward()
                for component in [fusion.collect, fusion.broadcast, fusion.latents, *fusion.inputs,
                                  *fusion.outputs, *model.backbone.pyramid.branches, model.backbone.blocks[1]]:
                    gradients = [p.grad for p in component.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)

    def test_only_coarse_supervision_backpropagates_to_fine_encoder(self):
        _, model, _ = self.models(); model.train()
        h, _, _ = model.encode(torch.randn(1, 512, 2), torch.tensor([512]))
        (h * torch.randn_like(h)).sum().backward()
        for branch in model.backbone.pyramid.branches:
            gradients = [p.grad for p in branch.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)

    def test_invalid_configuration_and_wrong_warmstart_layer_are_rejected(self):
        baseline, model, config = self.models()
        for key, value in [('after_block', 0), ('after_block', 2), ('tokens', 0), ('width', 25),
                           ('heads', 0), ('dropout', 1.), ('residual_scale', -1.)]:
            bad = deepcopy(config); bad['model']['mid_fusion'][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'mid_fusion'):
                resolve_config(bad)
        bad = deepcopy(config); bad['model']['pyramid']['enabled'] = False
        with self.assertRaisesRegex(ValueError, 'requires the pyramid'):
            resolve_config(bad)
        source = deepcopy(model.config); source['mid_fusion']['after_block'] = 2
        with self.assertRaisesRegex(ValueError, 'matching mid_fusion'):
            load_with_training_aux(model, model.state_dict(), source)
        self.assertIsNone(baseline.backbone.pyramid.mid_fusion)


if __name__ == '__main__':
    unittest.main(verbosity=2)
