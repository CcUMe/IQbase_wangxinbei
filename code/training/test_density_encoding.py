"""Padding, warmstart, gradient and training-fast-path checks for density RFModel."""
from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
import torch
import numpy as np
from torch.nn import functional as F
from models.density_encoding import DensityEncoding
from models.rf_model import RFModel
from utils.common import model_bundle
from utils.contrastive import load_with_training_aux
from training.optimize_density import density_descriptor


class DensityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_distribution_changes_with_same_mean_and_masks_padding(self):
        torch.manual_seed(42);module=DensityEncoding(8,dict(enabled=True,width=8,codewords=4,lags=[0]))
        x=torch.randn(2,8,64);valid=torch.ones(2,64,dtype=torch.bool)
        module(x,valid);a=module.descriptor.clone()
        module(x[:,:,torch.randperm(64)],valid);torch.testing.assert_close(module.descriptor,a,rtol=2e-5,atol=2e-5)
        padded=F.pad(x,(0,64));mask=torch.arange(128)[None].expand(2,-1)<64
        module(padded,mask);torch.testing.assert_close(module.descriptor,a,rtol=2e-5,atol=2e-5)
        constant=x.mean(-1,keepdim=True).expand_as(x);module(constant,valid)
        self.assertGreater(float((module.descriptor-a).detach().norm()),.01)

    def test_model_initial_equivalence_and_fast_path_gradient(self):
        source=torch.load(ROOT/'artifacts/phase_generalization/20261114_large96/mod_sam_005_ema.pt',map_location='cpu',weights_only=True,mmap=True)
        baseline=RFModel(**model_bundle(source)).eval();baseline.load_state_dict(source['model'],strict=True)
        bundle=model_bundle(source);bundle['model_config']={**bundle['model_config'],'density_encoding':{'enabled':True}}
        model=RFModel(**bundle).eval();load_with_training_aux(model,source['model'],source['model_config'])
        meta=np.load(ROOT/'artifacts/data_full/metadata.npz');iq=np.load(ROOT/'artifacts/data_full/iq.npy',mmap_mode='r')
        with torch.inference_mode():
            for length in [512,1024]:
                rows=np.flatnonzero((meta['split']==0)&(meta['length']==length)&(meta['modulation']>=0))[:2]
                x=torch.from_numpy(np.array(iq[rows,:length]));lengths=torch.full((len(x),),length)
                a=baseline(x,lengths);b=model(x,lengths)
                for task in ['modulation','protocol','parameters']:torch.testing.assert_close(a[task],b[task],rtol=0,atol=2e-5)
                descriptor=model.backbone.time.density.descriptor.clone()
                torch.testing.assert_close(density_descriptor(model,x,lengths),descriptor,rtol=0,atol=0)
        model.heads['modulation'].use_context=False
        for name,p in model.named_parameters():p.requires_grad_(name.startswith(('backbone.time.density.','heads.modulation.density_projection.')) and '.shared_projection.' not in name)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        x=torch.randn(4,512,2);lengths=torch.full((4,),512);target=torch.tensor([0,1,2,3])
        for _ in range(2):
            opt.zero_grad(set_to_none=True);d=density_descriptor(model,x,lengths)
            loss=F.cross_entropy(model.heads['modulation'](torch.zeros(4,model.heads['modulation'][0].in_features),d),target)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            opt.step()
        self.assertGreater(float(model.backbone.time.density.local.weight.grad.norm()),0.)
        self.assertGreater(float(model.heads['modulation'].density_projection.weight.grad.norm()),0.)
        model.heads['modulation'].use_context=True
        with torch.no_grad():
            model.backbone.time.density.shared_projection.weight.normal_(0,.01)
        model.eval()
        with torch.inference_mode():
            native=model(x[:2],lengths[:2]);padded=model(F.pad(x[:2],(0,0,0,512)),lengths[:2])
            for task in ['modulation','protocol','parameters']:torch.testing.assert_close(native[task],padded[task],rtol=1e-4,atol=1e-4)
        # Inferring source options from a trained state must retain the learned
        # adapter; warming the same instance from a base model must reset it.
        learned={k:v.clone() for k,v in model.state_dict().items() if k.startswith(('backbone.time.density.','heads.modulation.density_projection.'))}
        load_with_training_aux(model,model.state_dict())
        for key,value in learned.items():torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
        load_with_training_aux(model,source['model'],source['model_config'])
        self.assertEqual(float(model.backbone.time.density.shared_projection.weight.detach().abs().max()),0.)
        self.assertEqual(float(model.heads['modulation'].density_projection.weight.detach().abs().max()),0.)
        with torch.inference_mode():
            expected=baseline(x[:2],lengths[:2]);restored=model(x[:2],lengths[:2])
            for task in ['modulation','protocol','parameters']:torch.testing.assert_close(expected[task],restored[task],rtol=0,atol=2e-5)
        report=model.parameter_report();self.assertGreater(report['shared_ratio'],.8)

    def test_curriculum_has_positive_class_balanced_weights(self):
        data=np.load(ROOT/'artifacts/density_optimization/reliability_20261114/reliability.npz')
        meta=np.load(ROOT/'artifacts/data_full/metadata.npz')
        self.assertTrue((meta['split'][data['rows']]==0).all())
        w=(.35+.65*data['rank'])/.675
        self.assertGreater(float(w.min()),0.)
        for label in range(23):self.assertAlmostEqual(float(w[data['labels']==label].mean()),1.,places=5)


if __name__=='__main__':unittest.main(verbosity=2)
