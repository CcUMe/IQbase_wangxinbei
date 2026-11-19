"""Parameter candidate: masking, task isolation, migration and export contracts."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from models.rf_model import RFModel
from models.parameter_readout import parameter_readout_options, parameter_features
from utils.augmentation import augment_supervised_iq
from utils.common import save_safetensors, load_safetensors
from utils.config import load_config, training_options
from utils.engine import patch_mask, supervised_loss, pretraining_loss
from utils.head_optimizer import active_heads, clear_inactive_head_gradients
from utils.initialization import load_pretrained_backbone


def fixture(enabled=True, pretrain=True):
    config = dict(width=32, depth=1, heads=4, patch_size=16, stft_window=64,
                  dropout=0., phase_features=True, pretrain_heads=pretrain,
                  atomic_vocab=8 if pretrain else 0, emergent_vocab=12 if pretrain else 0,
                  parameter_mode='grid_residual',
                  parameter_readout=dict(enabled=enabled, hidden_width=16, dropout=0., temporal_bins=4))
    schema = {'classification_labels': {'modulation': ['a','b'], 'protocol': ['x','y']}}
    scaler = {'mean':[0.]*7,'std':[1.]*7}
    grids = [[-1.,0.,1.] if i in [0,1,2,6] else [] for i in range(7)]
    return RFModel(config,schema,scaler,grids),scaler,grids


class ParameterReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(78)

    def test_three_formal_configs_share_model_and_pretrain_has_no_augmentation(self):
        root=Path(__file__).resolve().parents[1]/'configs'
        configs=[load_config(root/stage/'receiver_parameter_v1.json5') for stage in ['supervised','pretrain','finetune']]
        self.assertTrue(all(c['model']==configs[0]['model'] for c in configs))
        pre=configs[1]
        self.assertEqual(pre['data']['augmentation']['parameter_only']['phase_rotation_probability'],0.)
        self.assertEqual(pre['data']['augmentation']['modulation_only']['phase_rotation_probability'],0.)
        self.assertTrue(Path(pre['run']['codebook'],'codebook.json').is_file())
        self.assertEqual(configs[2]['evaluation']['selection_metric'],'proxy_score')

    def test_parameter_phase_preserves_labels_power_positions_and_other_tasks(self):
        iq=torch.randn(5,512,2);iq[0,:32]=0
        valid=torch.zeros(5,7,dtype=torch.bool);valid[0,3:]=True;valid[1,:3]=True;valid[4,3]=True
        batch=dict(iq=iq,lengths=torch.full((5,),512),modulation=torch.tensor([-1,-1,0,-1,0]),
                   protocol=torch.tensor([-1,-1,-1,0,-1]),target_valid=valid,targets=torch.randn(5,7))
        before={k:v.clone() for k,v in batch.items()}
        settings=dict(phase_rotation_probability=0.,parameter_only=dict(phase_rotation_probability=1.))
        result=augment_supervised_iq(batch,settings)
        torch.testing.assert_close(result[2:],iq[2:],rtol=0,atol=0)
        torch.testing.assert_close(result.square().sum(-1),iq.square().sum(-1))
        torch.testing.assert_close(result[0,:32],iq[0,:32],rtol=0,atol=0)
        for k,v in before.items():torch.testing.assert_close(batch[k],v,rtol=0,atol=0)
        self.assertFalse(torch.equal(result[:2],iq[:2]))

    def test_feature_moments_use_valid_sample_centers_and_ignore_padding(self):
        h=torch.randn(2,4,3);valid=torch.tensor([[True,True,False,False],[True]*4]);w=valid[:,None].float().expand(-1,7,-1)
        settings=parameter_readout_options(dict(include_std=False,temporal_bins=4))
        x=parameter_features(h,valid,w,16,64,settings)
        changed=h.clone();changed[0,2:]=1000
        torch.testing.assert_close(x,parameter_features(changed,valid,w,16,64,settings),rtol=0,atol=0)
        torch.testing.assert_close(x[0,:,6],torch.full((7,),.25))
        torch.testing.assert_close(x[1,:,6],torch.full((7,),.5))
        torch.testing.assert_close(x[:,:,-4:].sum(-1),torch.ones(2,7))
        self.assertEqual(x.shape,(2,7,14))

    def test_old_backbone_transfer_is_exact_and_masked_targets_cannot_leak(self):
        old,_,_=fixture(False);new,_,_=fixture(True)
        load_pretrained_backbone(new,dict(model_config=old.config,model=old.state_dict()))
        for k,v in old.backbone.state_dict().items():torch.testing.assert_close(new.backbone.state_dict()[k],v,rtol=0,atol=0)
        new.eval();iq=torch.randn(2,512,2);lengths=torch.full((2,),512);mask=patch_mask(lengths,16,.3,3,11)
        changed=iq.clone();changed[mask.repeat_interleave(16,1)]+=1000
        a=new(iq,lengths,patch_mask=mask,return_pretrain=True);b=new(changed,lengths,patch_mask=mask,return_pretrain=True)
        for key in ['tokens','modulation','protocol','parameters']:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        for key in ['atomic','emergent']:torch.testing.assert_close(a['pretrain'][key],b['pretrain'][key],rtol=0,atol=0)
        batch=dict(atomic=torch.zeros(2,32,dtype=torch.long),emergent=torch.zeros(2,32,dtype=torch.long))
        loss,_=pretraining_loss(a,batch,mask,dict(reconstruction_weight=0.,atomic_weight=1.,emergent_weight=1.))
        loss.backward();self.assertTrue(torch.isfinite(loss))

    def test_parameter_only_update_protects_classifiers_and_inactive_fields(self):
        model,scaler,grids=fixture(pretrain=False);model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
        iq=torch.randn(2,512,2);valid=torch.zeros(2,7,dtype=torch.bool);valid[:,3]=True
        targets=torch.full((2,7),float('nan'));targets[:,3]=torch.tensor([.2,.7])
        batch=dict(iq=iq,lengths=torch.full((2,),512),modulation=torch.full((2,),-1),protocol=torch.full((2,),-1),target_valid=valid,targets=targets)
        config=dict(stage='supervised',task_weights=[1.,1.,4/3],parameter_weights=[1.]*7,model=model.config,grid_loss_weight=.1)
        before={k:v.detach().clone() for k,v in model.named_parameters()}
        output=model(iq,batch['lengths']);loss,_=supervised_loss(output,batch,scaler,grids,config);loss.backward()
        clear_inactive_head_gradients(model,active_heads(batch,config),'cpu');optimizer.step()
        for k,v in model.named_parameters():
            if k.startswith(('heads.modulation.','heads.protocol.','heads.regression.0.','heads.regression.4.')):
                torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.named_parameters() if k.startswith('heads.regression.3.')))

    def test_export_reload_produces_identical_three_task_predictions(self):
        model,scaler,grids=fixture();model.eval();config={**model.config,'pretrain_heads':False,'atomic_vocab':0,'emergent_vocab':0}
        deployed=RFModel(config,model.schema if hasattr(model,'schema') else {'classification_labels':{'modulation':['a','b'],'protocol':['x','y']}},scaler,grids).eval()
        state={k:v for k,v in model.state_dict().items() if not k.startswith('aux.')}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.safetensors';save_safetensors(path,state);deployed.load_state_dict(load_safetensors(path),strict=True)
        iq=torch.randn(2,1024,2);lengths=torch.tensor([512,1024]);iq[0,512:]=0
        with torch.no_grad():a=model(iq,lengths);b=deployed(iq,lengths)
        for key in ['modulation','protocol','parameters']:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
