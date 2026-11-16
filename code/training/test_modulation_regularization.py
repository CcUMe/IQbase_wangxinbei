"""Loss semantics for controlled, modulation-only regularization experiments."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from utils.modulation_regularization import modulation_correction,symmetric_kl


class ModulationRegularizationTests(unittest.TestCase):
    def test_kl_is_symmetric_zero_for_identical_logits_and_trains_both_views(self):
        torch.manual_seed(1)
        a=torch.randn(4,23,requires_grad=True);b=torch.randn(4,23,requires_grad=True)
        torch.testing.assert_close(symmetric_kl(a,a),torch.tensor(0.))
        torch.testing.assert_close(symmetric_kl(a,b),symmetric_kl(b,a))
        loss=symmetric_kl(a,b);self.assertGreater(float(loss.detach()),0.)
        loss.backward()
        for value in [a,b]:
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(),0.)

    def test_only_applicable_modulation_labels_are_changed(self):
        a=torch.randn(3,23,requires_grad=True);labels=torch.tensor([1,-1,3])
        delta,_=modulation_correction({'modulation':a},{'modulation':labels},smoothing=.05)
        expected=F.cross_entropy(a[[0,2]],labels[[0,2]],label_smoothing=.05)-F.cross_entropy(a[[0,2]],labels[[0,2]])
        torch.testing.assert_close(delta,expected)
        delta.backward();self.assertEqual(a.grad[1].abs().sum().item(),0.)
        delta,_=modulation_correction({'modulation':a},{'modulation':torch.full((3,),-1)},smoothing=.05)
        self.assertEqual(delta.item(),0.)

    def test_double_ce_control_matches_mean_of_two_supervised_passes(self):
        a,b=torch.randn(3,23),torch.randn(3,23);labels=torch.tensor([1,2,3])
        base=F.cross_entropy(a,labels)
        delta,_=modulation_correction({'modulation':a},{'modulation':labels},second={'modulation':b})
        torch.testing.assert_close(base+delta,.5*(base+F.cross_entropy(b,labels)))


if __name__=='__main__':unittest.main(verbosity=2)
