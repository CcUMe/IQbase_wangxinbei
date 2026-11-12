"""Meaningful checks for controlled observation/scale/structure diagnostics."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models.rf_model import RFModel
from training.hypothesis_pretrain import descriptors
from training.hypothesis_probe import perturb, group_metrics
from utils.contrastive import load_with_training_aux
from utils.engine import patch_mask
from utils.metrics import classification_metrics


def fixture(**overrides):
    return RFModel(dict(width=32,depth=1,heads=4,dropout=0.,attention_dropout=0.,
        phase_features=True,qk_normalization=True,**overrides),
        {'classification_labels':{'modulation':['a','b'],'protocol':['x','y']}},
        {'mean':[0.]*7,'std':[1.]*7},[[]]*7).eval()


class HypothesisTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2026)

    def test_multiscale_preserves_common_initialization_and_learns_residual(self):
        base=fixture()
        torch.manual_seed(2026)
        changed=fixture(multiscale_frontend=True)
        for name,value in base.state_dict().items():
            torch.testing.assert_close(value,changed.state_dict()[name],rtol=0,atol=0)
        x,l=torch.randn(2,512,2),torch.tensor([512,512])
        torch.testing.assert_close(base(x,l)['modulation'],changed(x,l)['modulation'],rtol=0,atol=0)
        changed(x,l)['modulation'].sum().backward()
        self.assertGreater(float(changed.backbone.time.multiscale['mix'].weight.grad.abs().sum()),0.)
        load_with_training_aux(changed,base.state_dict(),base.config)

    def test_removing_scale_embedding_establishes_gain_invariance(self):
        m=fixture(scale_embedding=False)
        x,l=torch.randn(2,512,2),torch.tensor([512,512])
        torch.testing.assert_close(m(x,l)['modulation'],m(2*x,l)['modulation'],rtol=0,atol=0)

    def test_atomic_descriptor_is_phase_gain_invariant_and_finite(self):
        x=torch.randn(3,512,2)
        changed=torch.stack((-x[...,1],x[...,0]),-1)*2
        torch.testing.assert_close(descriptors(x),descriptors(changed),rtol=1e-3,atol=1e-3)
        self.assertTrue(torch.isfinite(descriptors(torch.zeros_like(x))).all())

    def test_new_frontend_still_cannot_see_masked_targets(self):
        m=fixture(multiscale_frontend=True,pretrain_heads=True,atomic_vocab=16)
        x,l=torch.randn(2,512,2),torch.tensor([512,512])
        mask=patch_mask(l,16,.3,2,7)
        changed=x.clone();changed[mask.repeat_interleave(16,1)]+=1000
        a=m(x,l,patch_mask=mask,return_pretrain=True)
        b=m(changed,l,patch_mask=mask,return_pretrain=True)
        torch.testing.assert_close(a['tokens'],b['tokens'],rtol=0,atol=0)
        self.assertFalse(torch.equal(a['reconstruction_target'],b['reconstruction_target']))

    def test_shift_preserves_valid_samples_and_hard_group_counts_outgroup_errors(self):
        x=torch.randn(2,1024,2);x[0,512:]=0
        lengths=torch.tensor([512,1024])
        changed=perturb(x,lengths,'shift_16')
        torch.testing.assert_close(changed[0,:512],torch.roll(x[0,:512],16,0))
        self.assertEqual(float(changed[0,512:].abs().sum()),0.)
        import numpy as np
        cm=np.zeros((23,23),dtype=int);cm[0,0]=1;cm[0,22]=1
        metric=group_metrics(classification_metrics(cm,[str(i) for i in range(23)]))
        self.assertEqual(metric['support'],2)
        self.assertEqual(metric['accuracy'],.5)


if __name__=='__main__':unittest.main()
