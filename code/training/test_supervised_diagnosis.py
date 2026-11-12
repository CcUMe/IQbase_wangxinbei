"""Check the numerical definitions used to interpret observable IQ conditions."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from training.diagnose_signal_conditions import describe,FEATURES


class SupervisedDiagnosisTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(2026)

    def test_linear_autocorrelation_matches_direct_complex_product(self):
        x=torch.randn(3,512,2);lengths=torch.tensor([512]*3)
        v=describe(x,lengths);z=torch.complex(x[...,0],x[...,1]);z=z-z.mean(1,keepdim=True)
        z=z/z.abs().square().mean(1,keepdim=True).sqrt()
        c=(z[:,1:]*z[:,:-1].conj()).mean(1)
        torch.testing.assert_close(v[:,FEATURES.index('lag1_abs')],c.abs(),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(v[:,FEATURES.index('lag1_phase_cycles')],c.angle()/(2*torch.pi),atol=1e-6,rtol=1e-5)

    def test_invalid_padding_cannot_change_condition_measurement(self):
        x=torch.randn(2,1024,2);lengths=torch.tensor([512,1024]);a=describe(x,lengths)
        changed=x.clone();changed[0,512:]+=1000
        torch.testing.assert_close(a,describe(changed,lengths),rtol=0,atol=0)

    def test_complex_cosine_ignores_only_global_gain_and_phase(self):
        x=torch.complex(torch.randn(4,1024),torch.randn(4,1024))
        y=.7*x*torch.exp(torch.tensor(.37j))
        x=x/x.norm(dim=-1,keepdim=True);y=y/y.norm(dim=-1,keepdim=True)
        similarity=(x@y.conj().T).abs()
        torch.testing.assert_close(similarity.diag(),torch.ones(4),atol=1e-6,rtol=1e-6)
        self.assertLess(float(similarity[0,1]),.2)


if __name__=='__main__':unittest.main()
