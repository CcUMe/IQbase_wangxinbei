"""Mechanism/gradient/coordinate checks for the optional receiver memory."""
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.reception import ReceptionMemory,rrc_filter,fractional_samples,complex_fir,balanced_attention_mask


class ReceptionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(4);torch.manual_seed(20261114)

    def test_rrc_normalization_and_limit(self):
        for sps in [2,4,8,16,32,64]:
            h=rrc_filter(sps)
            self.assertEqual(len(h),4*sps+1);self.assertTrue(torch.isfinite(h).all())
            torch.testing.assert_close(h,h.flip(0))
            torch.testing.assert_close(h.square().sum(),torch.tensor(1.))
        self.assertTrue(torch.isfinite(rrc_filter(4,beta=.5)).all())

    def test_fractional_sampler_coordinates_and_gradient(self):
        t=torch.arange(30).float()[None,:,None].expand(2,-1,2)
        coordinate=torch.tensor([[3.25,12.75],[5.5,20.125]],requires_grad=True)
        y=fractional_samples(t,coordinate)
        torch.testing.assert_close(y[...,0],coordinate)
        y.sum().backward();torch.testing.assert_close(coordinate.grad,torch.full_like(coordinate,2.))
        x=torch.randn(2,40,2);zero=torch.zeros(2,5,2,requires_grad=True)
        torch.testing.assert_close(complex_fir(x,zero),x,rtol=0,atol=0)
        complex_fir(x,zero).square().mean().backward();self.assertTrue(torch.isfinite(zero.grad).all())

    def test_scale_balance_removes_count_prior_and_excludes_empty_scale(self):
        allowed=torch.tensor([[[True,True,True,True],[False,True,True,True]]])
        logits=balanced_attention_mask(allowed,[1,3]);p=logits.softmax(-1)
        torch.testing.assert_close(p[0,0,:1].sum(),torch.tensor(.5))
        torch.testing.assert_close(p[0,0,1:].sum(),torch.tensor(.5))
        self.assertEqual(p[0,1,0].item(),0.)
        torch.testing.assert_close(p[0,1,1:].sum(),torch.tensor(1.))

    def test_padding_and_original_sample_geometry(self):
        module=ReceptionMemory(48,16,dict(learned=True,width=48,heads=3)).eval()
        x=torch.randn(2,512,2);lengths=torch.full((2,),512);valid=torch.ones(2,512,dtype=torch.bool)
        padded=F.pad(x,(0,0,0,512));pv=torch.arange(1024)[None]<lengths[:,None]
        a=module.observations(x,valid,lengths);b=module.observations(padded,pv,lengths)
        for za,pa,ma,zb,pb,mb in zip(a[0],a[1],a[2],b[0],b[1],b[2]):
            for j in range(2):
                torch.testing.assert_close(pa[j,ma[j]],pb[j,mb[j]])
                torch.testing.assert_close(za[j,ma[j]],zb[j,mb[j]],rtol=2e-5,atol=2e-5)
        for sps,p in zip(module.options['sps'],a[1]):
            torch.testing.assert_close(p[0,:3],torch.arange(3)*sps+(sps-1)/2)

    def test_zero_residual_and_nonzero_bootstrap_gradients(self):
        module=ReceptionMemory(48,16,dict(learned=True,width=48,heads=3))
        x=torch.randn(2,512,2);lengths=torch.full((2,),512);valid=torch.ones(2,512,dtype=torch.bool)
        h=torch.randn(2,32,48);query_valid=torch.ones(2,32,dtype=torch.bool)
        delta=module(h,x,valid,lengths,query_valid)
        torch.testing.assert_close(delta,torch.zeros_like(h),rtol=0,atol=0)
        delta.sum().backward();self.assertGreater(module.projection.weight.grad.norm().item(),0.)
        with torch.no_grad():module.projection.weight.normal_(0,.02)
        module.zero_grad(set_to_none=True)
        module(h,x,valid,lengths,query_valid).square().mean().backward()
        for name,p in module.named_parameters():
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all(),name)
        self.assertGreater(module.estimate_output[-1].weight.grad.norm().item(),0.)

    def test_token_scale_units_zero_initialization_and_mask(self):
        raw=ReceptionMemory(48,16,dict(width=48,heads=3)).eval()
        scaled=ReceptionMemory(48,16,dict(width=48,heads=3,match_token_scale=True)).eval()
        scaled.load_state_dict(raw.state_dict())
        x=torch.randn(2,512,2);lengths=torch.full((2,),512)
        valid=torch.ones(2,512,dtype=torch.bool);qv=torch.ones(2,32,dtype=torch.bool);qv[:,30:]=False
        h=torch.randn(2,32,48)*5
        torch.testing.assert_close(scaled(h,x,valid,lengths,qv),torch.zeros_like(h),rtol=0,atol=0)
        with torch.no_grad():
            raw.projection.weight.normal_(0,.02);scaled.load_state_dict(raw.state_dict())
        a=raw(h,x,valid,lengths,qv);b=scaled(h,x,valid,lengths,qv)
        rms=((h.square()*qv[...,None]).sum((1,2))/(qv.sum(1)*48)).sqrt()
        torch.testing.assert_close(b,a*rms[:,None,None])
        altered=h.clone();altered[:,30:]=1e6
        torch.testing.assert_close(scaled(altered,x,valid,lengths,qv),b)

    def test_synthetic_matched_filter_symbol_evm(self):
        # Known controlled transmitter ONLY for an implementation check. Its
        # true clock/carrier are never supplied to any C01 training/validation.
        length=1024;sps=8;offset=4
        bits=torch.randint(0,2,(128,2))*2-1
        symbols=bits.float()/2**.5
        impulse=torch.zeros(1,length,2);indices=torch.arange(128)*sps+offset
        impulse[0,indices]=symbols
        kernel=rrc_filter(sps)[None,None]
        transmitted=F.conv1d(impulse.transpose(1,2),kernel.expand(2,1,-1),padding=16,groups=2).transpose(1,2)
        frequency=.037;phase=.4;t=torch.arange(length)
        z=torch.complex(transmitted[...,0],transmitted[...,1])*torch.exp(1j*(2*torch.pi*frequency*t+phase))
        iq=torch.stack([z.real,z.imag],-1)
        module=ReceptionMemory(48,16,dict(learned=True,width=48,heads=3))
        parameters=torch.zeros(1,24);parameters[0,1]=phase/torch.pi
        index=module.options['sps'].index(sps)
        parameters[0,12+6+index]=(offset-(sps-1)/2)/(sps/2)
        views,positions,masks,_=module.observations(iq,torch.ones(1,length,dtype=torch.bool),torch.tensor([length]),
            dict(centre=torch.tensor([frequency]),parameters=parameters))
        mask=masks[index][0]&(positions[index][0]>64)&(positions[index][0]<length-64)
        received=views[index][0,mask];positions=positions[index][0,mask]
        ids=((positions-offset)/sps).round().long();truth=torch.complex(symbols[ids,0],symbols[ids,1])
        measured=torch.complex(received[:,0],received[:,1])
        gain=(measured.conj()*truth).sum()/measured.abs().square().sum()
        evm=((gain*measured-truth).abs().square().mean()/truth.abs().square().mean()).sqrt().item()
        self.assertLess(evm,.15)
        print(f'Controlled QPSK/RRC known-clock matched-filter EVM={evm:.5f}',flush=True)


if __name__=='__main__':unittest.main(verbosity=2)
