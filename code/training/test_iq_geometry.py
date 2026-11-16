"""Shared-filter geometry, masked pretraining and backbone transfer contracts."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from models.rf_model import RFModel
from models.iq_geometry import IQGeometry
from utils.contrastive import load_with_training_aux
from utils.initialization import load_pretrained_backbone
from utils.engine import patch_mask


def build(enabled=True,pretrain=False,patch_size=16):
    return RFModel(dict(width=32,depth=2,heads=4,dropout=0.,attention_dropout=0.,
        patch_size=patch_size,
        phase_features=True,qk_normalization=True,pretrain_heads=pretrain,
        atomic_vocab=8 if pretrain else 0,emergent_vocab=16 if pretrain else 0,
        iq_geometry={'enabled':enabled,'bottleneck':8}),
        {'classification_labels':{'modulation':['a','b','c'],'protocol':['x','y']}},
        {'mean':[0.]*7,'std':[1.]*7},[[] for _ in range(7)])


class IQGeometryTests(unittest.TestCase):
    patch_size=16

    def model(self,*args,**kwargs):return build(*args,patch_size=self.patch_size,**kwargs)

    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_descriptor_phase_invariance_and_local_structure_sensitivity(self):
        torch.manual_seed(18);module=IQGeometry(32,self.patch_size,{'enabled':True,'bottleneck':8})
        weight=torch.randn(16,5,5)*.1;x=torch.randn(2,512,2);valid=torch.ones(2,512,dtype=torch.bool)
        angle=torch.tensor([.73,-1.32])[:,None];c,s=angle.cos(),angle.sin()
        rotated=torch.stack([x[...,0]*c-x[...,1]*s,x[...,0]*s+x[...,1]*c],-1)
        a,_=module.describe(x,valid,weight);b,_=module.describe(rotated,valid,weight)
        torch.testing.assert_close(a,b,rtol=3e-4,atol=3e-5)
        with torch.no_grad():module.up.weight.normal_(0,.03)
        patch_weight=torch.randn(32,16,self.patch_size)*.03
        first=module(x,valid,weight,torch.randn(2,512//self.patch_size,32),patch_weight)
        second=module(rotated,valid,weight,torch.randn(2,512//self.patch_size,32)*100,patch_weight)
        torch.testing.assert_close(first,second,rtol=3e-4,atol=3e-5)
        shuffled=x[:,torch.randperm(512)];d,_=module.describe(shuffled,valid,weight)
        self.assertGreater(float((a-d).norm()),1.)

    def test_zero_warmstart_and_trained_padding_and_gradients(self):
        torch.manual_seed(7);base=self.model(False).eval();model=self.model().eval()
        load_with_training_aux(model,base.state_dict(),base.config)
        for n in [512,1024]:
            x=torch.randn(2,n,2);lengths=torch.full((2,),n)
            with torch.no_grad():
                a=base(x,lengths);b=model(x,lengths)
            for key in ['modulation','protocol','parameters']:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
        x=torch.randn(2,512,2);lengths=torch.full((2,),512)
        for _ in range(2):
            opt.zero_grad();out=model(x,lengths);loss=F.cross_entropy(out['modulation'],torch.tensor([0,1]))+out['parameters'].square().mean()
            loss.backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None));opt.step()
        self.assertGreater(float(model.backbone.iq_geometry.down.weight.grad.norm()),0.)
        self.assertGreater(float(model.backbone.time[0].weight.grad[:,:2].norm()),0.)
        with torch.no_grad():
            a=model(x,lengths);b=model(F.pad(x,(0,0,0,512)),lengths)
        for key in ['modulation','protocol','parameters']:torch.testing.assert_close(a[key],b[key],rtol=1e-4,atol=1e-4)

    def test_masked_pretraining_never_reads_hidden_targets(self):
        torch.manual_seed(22);model=self.model(pretrain=True).eval()
        with torch.no_grad():model.backbone.iq_geometry.up.weight.normal_(0,.02)
        x=torch.randn(2,512,2);lengths=torch.full((2,),512);mask=patch_mask(lengths,self.patch_size,.4,8 if self.patch_size==8 else 2,19)
        changed=x.clone();changed[mask.repeat_interleave(self.patch_size,1)]+=10000
        with torch.no_grad():
            a=model(x,lengths,patch_mask=mask,return_pretrain=True);b=model(changed,lengths,patch_mask=mask,return_pretrain=True)
        for key in ['tokens','modulation','protocol','parameters']:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        for key in a['pretrain']:torch.testing.assert_close(a['pretrain'][key],b['pretrain'][key],rtol=0,atol=0)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
        for _ in range(2):
            opt.zero_grad();o=model(torch.zeros_like(x),lengths,patch_mask=mask,return_pretrain=True)
            loss=o['pretrain']['atomic'].square().mean();loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None));opt.step()

    def test_pretrain_transfer_retains_geometry_but_not_task_heads(self):
        old=self.model(pretrain=True)
        with torch.no_grad():old.backbone.iq_geometry.up.weight.normal_(0,.1)
        ck={'model_config':old.config,'model':old.state_dict()};new=self.model()
        head=new.heads['modulation'][0].weight.detach().clone()
        load_pretrained_backbone(new,ck)
        for k,v in old.backbone.state_dict().items():torch.testing.assert_close(new.backbone.state_dict()[k],v,rtol=0,atol=0)
        torch.testing.assert_close(new.heads['modulation'][0].weight,head,rtol=0,atol=0)
        changed=deepcopy(ck);changed['model_config']=deepcopy(old.config);changed['model_config']['qk_normalization']=False
        with self.assertRaisesRegex(ValueError,'qk_normalization'):load_pretrained_backbone(new,changed)
        changed['model_config']=deepcopy(old.config);changed['model_config']['iq_geometry']['lags']=[1,2]
        with self.assertRaisesRegex(ValueError,'iq_geometry'):load_pretrained_backbone(new,changed)
        legacy=self.model(False,True);load_pretrained_backbone(new,{'model_config':legacy.config,'model':legacy.state_dict()})
        self.assertEqual(float(new.backbone.iq_geometry.up.weight.detach().abs().max()),0.)

class IQGeometryP8Tests(IQGeometryTests):
    patch_size=8


if __name__=='__main__':unittest.main(verbosity=2)
