"""Coordinate semantics, legacy compatibility, and masked dual-scale RoPE."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from models.rf_model import RFModel
from utils.config import DEFAULTS, resolve_config
from utils.position import POSITION_DEFAULTS, position_coordinates


class PositionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_legacy_sample_coordinates_and_predictions_are_unchanged(self):
        def legacy(lengths,n,p,heads,*args):
            c=torch.arange(n).float()*p+(p-1)/2
            return (c/p)[None].expand(len(lengths),-1), torch.stack((c[None].expand(len(lengths),-1)/1024,
                lengths[:,None].expand(-1,n).float()/1024),-1)
        m=self.model('samples');x=torch.randn(2,512,2);lengths=torch.tensor([512,512])
        with torch.no_grad():
            actual=m(x,lengths)
            with patch('models.rf_model.position_coordinates',side_effect=legacy):expected=m(x,lengths)
        for key in ['modulation','protocol','parameters']:self.assertTrue(torch.equal(actual[key],expected[key]))

    def test_dual_preserves_sample_distance_and_adds_window_distance(self):
        positions,_=position_coordinates(torch.tensor([512,1024]),64,16,4,'dual')
        self.assertEqual(tuple(positions.shape),(2,4,64))
        self.assertEqual(float(positions[0,0,4]-positions[0,0,0]),4)
        self.assertEqual(float(positions[1,0,8]-positions[1,0,0]),8)
        self.assertEqual(float(positions[0,2,4]-positions[0,2,0]),float(positions[1,2,8]-positions[1,2,0]))
        a,c=position_coordinates(torch.tensor([512]),32,16,4,'dual')
        torch.testing.assert_close(a,positions[:1,:,:32],rtol=0,atol=0)
        torch.testing.assert_close(c,position_coordinates(torch.tensor([512]),64,16,4,'dual')[1][:,:32],rtol=0,atol=0)

    def test_physical_distances_and_absolute_duration_share_real_time_units(self):
        opts={'reference_sample_rate_hz':1e6}
        p,c=position_coordinates(torch.tensor([512,1024]),64,16,4,'physical',opts,torch.tensor([1e6,2e6]))
        self.assertEqual(float(p[0,4]-p[0,0]),float(p[1,8]-p[1,0]))
        self.assertEqual(float(c[0,0,1]),float(c[1,0,1]))
        constant={**opts,'sample_rate_hz':1e6}
        a,b=position_coordinates(torch.tensor([512]),32,16,4,'physical',constant)
        torch.testing.assert_close(a,p[:1,:32]);torch.testing.assert_close(b,c[:1,:32])

    def test_unknown_or_invalid_physical_rates_fail_without_guessing(self):
        for rates in [None,torch.tensor([0.]),torch.tensor([float('nan')]),torch.tensor([1.,2.])]:
            with self.assertRaises(ValueError):
                position_coordinates(torch.tensor([512]),32,16,4,'physical',sample_rates=rates)
        c=deepcopy(DEFAULTS);c['model']['signal']['position']='physical'
        with self.assertRaisesRegex(ValueError,'sampling-rate'):resolve_config(c)
        c['model']['signal']['position_options']['sample_rate_hz']=1e6
        resolve_config(c)

    def test_dual_masking_backprop_and_config_roundtrip(self):
        m=self.model('dual');x=torch.randn(2,512,2);lengths=torch.tensor([512,512])
        mask=torch.zeros(2,32,dtype=torch.bool);mask[:,4:10]=True
        changed=x.clone();changed[mask.repeat_interleave(16,1)]+=100
        a=m(x,lengths,patch_mask=mask);b=m(changed,lengths,patch_mask=mask)
        for key in ['modulation','protocol','parameters']:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        sum(a[k].square().mean() for k in ['modulation','protocol','parameters']).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
        restored=RFModel(m.config,m.schema,{'mean':[0.]*7,'std':[1.]*7},[[]]*7).eval()
        restored.load_state_dict(m.state_dict())
        with torch.no_grad():torch.testing.assert_close(restored(x,lengths)['modulation'],m(x,lengths)['modulation'],rtol=0,atol=0)

    def model(self,mode):
        return RFModel({'width':32,'depth':1,'heads':4,'dropout':0.,'attention_dropout':0.,
            'patch_size':16,'stft_window':64,'qk_normalization':True,'position':mode},
            {'classification_labels':{'modulation':['a','b'],'protocol':['c','d']}},
            {'mean':[0.]*7,'std':[1.]*7},[[]]*7).eval()


if __name__=='__main__':unittest.main(verbosity=2)
