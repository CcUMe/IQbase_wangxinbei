"""Receiver contracts: masking, coordinates, gradients and shared-weight transfer.

Small CPU checks verify execution/semantics, never C01 accuracy improvement.
"""
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from models.rf_model import RFModel
from models.tracked_receiver import TrackedReceiver, receiver_options
from utils.contrastive import load_with_training_aux
from utils.initialization import load_pretrained_backbone
from utils.config import load_config
from utils.training_display import training_overview


OPTIONS = {'enabled': True, 'filter': {'taps': 5},
           'timing': {'periods': [4., 8., 16.], 'hidden': 8},
           'readout': {'width': 16, 'dropout': 0.}}


def build(enabled=True, pretrain=False, options=None):
    return RFModel(dict(width=32, depth=2, heads=4, dropout=0., attention_dropout=0.,
        patch_size=8, phase_features=True, qk_normalization=True,
        pretrain_heads=pretrain, atomic_vocab=8 if pretrain else 0,
        emergent_vocab=16 if pretrain else 0,
        iq_geometry={'enabled': True, 'bottleneck': 8},
        receiver_frontend=(options or OPTIONS) if enabled else {'enabled': False}),
        {'classification_labels': {'modulation': ['a','b','c'], 'protocol': ['x','y']}},
        {'mean': [0.]*7, 'std': [1.]*7}, [[] for _ in range(7)])


class ReceiverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_validation(self):
        for invalid in [{'unknown': 1}, {'timing': {'periods': [0]}},
                        {'timing': {'phases': [1]}}, {'filter': {'taps': 4}},
                        {'readout': {'max_scale': float('nan')}}]:
            with self.assertRaises(ValueError):
                receiver_options(invalid)

    def test_real_stage_configs_print_training_overview(self):
        root = Path(__file__).resolve().parents[2]
        args = SimpleNamespace(init_checkpoint='',warmstart_checkpoint='',resume='',
                               config='receiver_v1',data='fixture',output='fixture')
        info = {'task_counts': {'train': {'modulation': 8,'protocol': 8,'parameter': 8}}}
        report = {'final_parameters': 1000000,'shared_ratio': .98,'auxiliary_parameters': 0}
        for stage in ['supervised','pretrain','finetune']:
            config = load_config(root/'code/configs'/stage/'shared_base_p8_receiver_v1.json5')
            output = StringIO()
            with redirect_stdout(output):
                training_overview(config,args,torch.device('cpu'),1,24,12,info,report,
                                  2,1,0,0,0,{'atomic_vocab':512,'emergent_vocab':1024} if stage=='pretrain' else None)
            self.assertIn('接收前端',output.getvalue())
            self.assertIn('原P8坐标',output.getvalue())
            self.assertIn('输出目录',output.getvalue())

    def test_identity_initialization_and_old_warmstart(self):
        torch.manual_seed(7); old = build(False).eval()
        torch.manual_seed(7); new = build().eval()
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0, msg=key)
        # Loading into a reused instance also restores the new zero residual.
        with torch.no_grad(): new.backbone.receiver_frontend.up.weight.fill_(.5)
        load_with_training_aux(new, old.state_dict(), old.config)
        for length in [512, 1024]:
            x = torch.randn(2, length, 2); lengths = torch.full((2,), length)
            with torch.no_grad(): a, b = old(x, lengths), new(x, lengths)
            for key in ['modulation','protocol','parameters']:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)
            with torch.no_grad():
                torch.testing.assert_close(old.encode(x,lengths)[0],new.encode(x,lengths)[0],rtol=0,atol=0)

    def test_trained_padding_and_mixed_lengths(self):
        torch.manual_seed(19); model = build().eval()
        with torch.no_grad():
            model.backbone.receiver_frontend.up.weight.normal_(0, .05)
            model.backbone.receiver_frontend.taps.normal_(0, .1)
            model.backbone.receiver_frontend.rate_predictor[-1].weight.normal_(0, .05)
        x = torch.randn(1, 512, 2)
        padded = torch.cat([x, torch.randn(1,512,2)*1000], 1)
        with torch.no_grad():
            native = model(x, torch.tensor([512]))
            full = model(torch.cat([padded, torch.randn_like(padded)]), torch.tensor([512,1024]))
        for key in ['modulation','protocol','parameters']:
            torch.testing.assert_close(native[key], full[key][:1], rtol=1e-4, atol=1e-4, msg=key)
        with torch.no_grad():
            native_tokens = model.encode(x,torch.tensor([512]))[0]
            padded_tokens = model.encode(padded,torch.tensor([512]))[0]
        torch.testing.assert_close(native_tokens,padded_tokens[:,:64],rtol=1e-4,atol=1e-4)

    def test_masking_no_hidden_signal_leakage(self):
        torch.manual_seed(22); model = build(pretrain=True).eval()
        with torch.no_grad(): model.backbone.receiver_frontend.up.weight.normal_(0,.05)
        x = torch.randn(2,512,2); lengths = torch.tensor([512,512])
        mask = torch.zeros(2,64,dtype=torch.bool); mask[:,12:22] = True; mask[:,40:50] = True
        hidden = mask.repeat_interleave(8,1)
        changed = x.clone(); changed[hidden] = torch.randn_like(changed[hidden])*10000
        with torch.no_grad():
            a = model(x,lengths,patch_mask=mask,return_pretrain=True)
            b = model(changed,lengths,patch_mask=mask,return_pretrain=True)
        for key in ['tokens','modulation','protocol','parameters']:
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0,msg=key)
        for key in a['pretrain']:
            torch.testing.assert_close(a['pretrain'][key],b['pretrain'][key],rtol=0,atol=0,msg=key)

    def test_all_masked_and_zero_finite_gradients(self):
        model = build(pretrain=True)
        with torch.no_grad(): model.backbone.receiver_frontend.up.weight.normal_(0,.03)
        for all_masked in [False, True]:
            model.zero_grad()
            out = model(torch.zeros(2,512,2),torch.tensor([512,512]),
                        patch_mask=torch.full((2,64),all_masked),return_pretrain=True)
            loss = out['pretrain']['atomic'].square().mean()+out['parameters'].square().mean()
            self.assertTrue(torch.isfinite(loss)); loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all(),name)
            coverage = model.backbone.receiver_frontend.last_diagnostics['token_coverage']
            self.assertEqual(float(coverage)==0., all_masked)

    def test_hypotheses_monotone_and_visible_error_only(self):
        torch.manual_seed(3); module = TrackedReceiver(32,8,OPTIONS)
        x = torch.randn(2,128,2)
        _, positions, _, interval = module.track(x,torch.ones(2,128),torch.tensor([8.,9.]),1)
        self.assertTrue((positions.diff(dim=-1)>0).all())
        self.assertTrue((interval>0).all())
        self.assertGreater(float(interval.std(-1).max().detach()),0.)
        _, no_observations, valid, constant_interval = module.track(
            x,torch.zeros(2,128),torch.tensor([8.,9.]),1)
        self.assertFalse(valid.any())
        torch.testing.assert_close(constant_interval,
            torch.tensor([8.,9.])[:,None,None].expand_as(constant_interval))
        self.assertTrue((no_observations.diff(dim=-1)>0).all())
        # A very short observation also has a well-defined control descriptor.
        self.assertTrue(torch.isfinite(module.describe(x[:,:8],torch.ones(2,8,dtype=torch.bool),torch.tensor([8,8]))).all())

    def test_gradients_reach_reception_operators_after_zero_projection(self):
        torch.manual_seed(13); model = build(); optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad()
            out = model(torch.randn(2,512,2),torch.tensor([512,512]))
            loss = F.cross_entropy(out['modulation'],torch.tensor([0,1]))+out['parameters'].square().mean()
            loss.backward()
            for name,p in model.named_parameters():
                if p.grad is not None: self.assertTrue(torch.isfinite(p.grad).all(),name)
            optimizer.step()
        parameters = dict(model.backbone.receiver_frontend.named_parameters())
        for key in ['taps','loop_gains','rate_predictor.0.weight','rate_predictor.2.weight',
                    'embedding.0.weight','hypothesis_embedding','score.weight','gate.weight','up.weight']:
            self.assertIsNotNone(parameters[key].grad,key)
            self.assertGreater(float(parameters[key].grad.norm()),0.,key)

    def test_backbone_transfer_and_config_semantics(self):
        old = build(pretrain=True)
        with torch.no_grad(): old.backbone.receiver_frontend.up.weight.normal_(0,.1)
        checkpoint = {'model_config': old.config, 'model': old.state_dict()}
        new = build(); head = new.heads['modulation'][0].weight.detach().clone()
        load_pretrained_backbone(new,checkpoint)
        for key,value in old.backbone.state_dict().items():
            torch.testing.assert_close(new.backbone.state_dict()[key],value,rtol=0,atol=0,msg=key)
        torch.testing.assert_close(new.heads['modulation'][0].weight,head,rtol=0,atol=0)
        load_with_training_aux(new,build().state_dict(),new.config)
        changed = deepcopy(checkpoint); changed['model_config']['receiver_frontend']['timing']['periods'] = [2.,8.,16.]
        with self.assertRaisesRegex(ValueError,'receiver_frontend'): load_pretrained_backbone(new,changed)
        with self.assertRaisesRegex(ValueError,'receiver_frontend'):
            load_with_training_aux(new,old.state_dict(),changed['model_config'])
        with self.assertRaisesRegex(ValueError,'receiver_frontend'): load_pretrained_backbone(build(False),checkpoint)
        legacy = build(False,True)
        load_pretrained_backbone(new,{'model_config':legacy.config,'model':legacy.state_dict()})
        self.assertEqual(float(new.backbone.receiver_frontend.up.weight.detach().abs().max()),0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
