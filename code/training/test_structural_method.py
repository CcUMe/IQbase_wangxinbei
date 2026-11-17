"""Paper-equation checks and explicit C01 adaptation contracts, CPU only."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from training.build_codebook import features,assign_features,euclidean_cluster
from training.test_iq_geometry import build
from utils.config import DEFAULTS,resolve_config
from utils.engine import patch_mask,pretraining_loss
from utils.initialization import load_pretrained_backbone
from utils.normalization import observation_scale
from utils.position import position_coordinates
from utils.structure import learn_merges,StructuralTokenizer


class StructuralMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_euclidean_retains_relative_patch_energy(self):
        x=torch.zeros(1,16,2);x[:,:8,0]=1.;x[:,8:,0]=2.
        batch={'iq':x,'lengths':torch.tensor([16])}
        sphere,_=features(batch,8,None)
        euclidean,_=features(batch,8,None,'euclidean')
        torch.testing.assert_close(sphere[:,0],sphere[:,1],rtol=0,atol=0)
        torch.testing.assert_close(euclidean[:,1],2*euclidean[:,0],rtol=0,atol=0)
        centers=euclidean.squeeze(0)
        self.assertEqual(assign_features(euclidean,centers,'euclidean').tolist(),[[0,1]])

    def test_euclidean_assignment_uses_centroid_norm(self):
        centers=torch.tensor([[1.,0.],[3.,0.]])
        points=torch.tensor([[[1.1,0.],[2.8,0.]]])
        self.assertEqual(assign_features(points,centers,'euclidean').tolist(),[[0,1]])
        self.assertEqual(assign_features(points,centers,'spherical').tolist(),[[1,1]])
        cloud=torch.cat([torch.randn(40,2)*.01+centers[i] for i in range(2)])
        fitted=euclidean_cluster(cloud,2,3,17,batch_size=32,n_init=5)
        self.assertGreater(float(fitted.norm(dim=1).max()),2.)

    def test_minimum_pair_frequency_and_patch_backfill(self):
        seq=[np.array([0,0,0,0,1,1],np.int32)]
        rules,lengths=learn_merges(seq,2,4,max_span=5,min_frequency=20)
        self.assertEqual(rules,[])
        rules,lengths=learn_merges(seq*30,2,4,max_span=5,min_frequency=20)
        self.assertTrue(rules);self.assertLessEqual(max(lengths),5)
        result=StructuralTokenizer(rules).encode(seq[0])
        self.assertEqual(len(result),len(seq[0]))
        self.assertLess(int(result.max()),2+len(rules))

    def test_start_probability_is_distinct_from_final_ratio(self):
        lengths=torch.full((2048,),1024)
        actual=patch_mask(lengths,8,0.,8,123,strategy='bernoulli_starts',start_probability=.06)
        random=np.random.default_rng(123);expected=np.zeros(actual.shape,dtype=bool)
        for row in range(len(lengths)):
            for start in np.flatnonzero(random.random(121)<.06):expected[row,start:start+8]=True
        self.assertTrue(np.array_equal(actual.numpy(),expected))
        observed=float(actual.float().mean())
        self.assertGreater(observed,.25);self.assertLess(observed,.5)
        config=deepcopy(DEFAULTS);mask=config['loss']['pretraining']['mask']
        mask.update(strategy='bernoulli_starts',ratio=0.,start_probability=.06,span=8)
        resolve_config(config)
        mask['ratio']=.3
        with self.assertRaisesRegex(ValueError,'ratio=0'):resolve_config(config)

    def test_masked_token_reduction_and_empty_mask(self):
        logits=torch.tensor([[[3.,0.]]*4,[[0.,3.]]*4],requires_grad=True)
        zeros=torch.zeros(2,4,2,requires_grad=True)
        output={'modulation':torch.zeros(2,2,requires_grad=True),'protocol':torch.zeros(2,2,requires_grad=True),
            'parameters':torch.zeros(2,7,requires_grad=True),'token_valid':torch.ones(2,4,dtype=torch.bool),
            'pretrain':{'atomic':logits,'reconstruction':zeros},'reconstruction_target':torch.zeros_like(zeros)}
        mask=torch.tensor([[1,0,0,0],[1,1,1,0]],dtype=torch.bool)
        labels=torch.zeros(2,4,dtype=torch.long)
        config={'reconstruction_weight':0.,'atomic_weight':1.,'emergent_weight':0.,'pretraining_reduction':'masked_token_mean'}
        loss,_=pretraining_loss(output,{'atomic':labels},mask,config)
        ce=F.cross_entropy(logits.transpose(1,2),labels,reduction='none')
        torch.testing.assert_close(loss,ce[mask].mean())
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())
        empty,_=pretraining_loss(output,{'atomic':labels},torch.zeros_like(mask),config)
        self.assertEqual(float(empty.detach()),0.)

    def test_percentile_and_rms_normalization_contracts(self):
        torch.manual_seed(1);x=torch.randn(2,512,2);visible=torch.ones(2,512,dtype=torch.bool)
        legacy=x.square().sum((1,2)).div(visible.sum(1)).sqrt().clamp_min(1e-6)
        torch.testing.assert_close(observation_scale(x,visible),legacy,rtol=0,atol=0)
        expected=torch.quantile(torch.linalg.vector_norm(x,dim=-1),.99,dim=1)
        torch.testing.assert_close(observation_scale(x,visible,'amplitude_percentile'),expected)
        empty=observation_scale(x,torch.zeros_like(visible),'amplitude_percentile')
        self.assertTrue(torch.isfinite(empty).all())
        model=build(pretrain=True,patch_size=8).eval();model.config['normalization']='amplitude_percentile'
        with torch.no_grad():model.backbone.iq_geometry.up.weight.normal_(0,.02)
        lengths=torch.tensor([512,512]);mask=patch_mask(lengths,8,.3,8,77)
        changed=x.clone();changed[mask.repeat_interleave(8,1)]+=1000
        with torch.no_grad():
            a=model(x,lengths,patch_mask=mask,return_pretrain=True)
            b=model(changed,lengths,patch_mask=mask,return_pretrain=True)
        for key in a['pretrain']:torch.testing.assert_close(a['pretrain'][key],b['pretrain'][key],rtol=0,atol=0)
        target=build(pretrain=False,patch_size=8)
        with self.assertRaisesRegex(ValueError,'normalization'):
            load_pretrained_backbone(target,{'model_config':model.config,'model':model.state_dict()})
        target.config['normalization']='amplitude_percentile'
        load_pretrained_backbone(target,{'model_config':model.config,'model':model.state_dict()})

    def test_rope_fs_equation_and_missing_rate_rejection(self):
        options={'reference_sample_rate_hz':1e6}
        actual,_=position_coordinates(torch.tensor([512,1024]),128,8,4,'rope_fs',options,torch.tensor([1e6,2e6]))
        expected=(torch.arange(128)+.5)[None]*torch.tensor([[1.],[.5]])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'real sample_rates'):
            position_coordinates(torch.tensor([512]),64,8,4,'rope_fs',options)


if __name__=='__main__':unittest.main(verbosity=2)
