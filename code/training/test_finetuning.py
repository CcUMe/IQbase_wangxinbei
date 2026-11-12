"""Fine-tuning phases, frozen updates, LR groups and pretraining anchors."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import os
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models.rf_model import RFModel
from training.train import build_optimizer
from utils.config import DEFAULTS,resolve_config
from utils.finetuning import FineTuneController
from utils.common import runtime


class FineTuningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(17)
        self.model=RFModel({'width':32,'depth':2,'heads':4,'dropout':.2,'attention_dropout':.2},
            {'classification_labels':{'modulation':['a','b'],'protocol':['c','d']}},
            {'mean':[0.]*7,'std':[1.]*7},[[]]*7)
        self.settings=deepcopy(DEFAULTS['finetuning'])
        self.settings.update(strategy='gradual',head_warmup_steps=2,upper_layers_steps=2,unfreeze_last_n_blocks=1)

    def test_frozen_parameters_stay_bitwise_equal_and_modes_follow_phase(self):
        opt=build_optimizer(self.model,DEFAULTS['optimizer'],self.settings)
        control=FineTuneController(self.model,self.settings);control.apply(0)
        before={k:v.clone() for k,v in self.model.backbone.state_dict().items()}
        x=torch.randn(2,512,2);lengths=torch.tensor([512,512])
        a=self.model.encode(x,lengths)[0];b=self.model.encode(x,lengths)[0]
        self.assertTrue(torch.equal(a,b))
        self.assertFalse(self.model.backbone.training)
        loss=self.model(x,lengths)['modulation'].square().mean();loss.backward();opt.step()
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in self.model.backbone.state_dict().items()))
        control.apply(2)
        self.assertTrue(self.model.backbone.blocks[-1].training)
        self.assertFalse(self.model.backbone.blocks[0].training)
        self.assertFalse(self.model.backbone.time.training)
        self.assertTrue(all(p.requires_grad for p in self.model.backbone.blocks[-1].parameters()))
        self.assertTrue(all(not p.requires_grad for p in self.model.backbone.blocks[0].parameters()))
        control.apply(4)
        self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
        self.assertTrue(self.model.backbone.training)
        self.assertEqual(sum(len(g['params']) for g in opt.param_groups),len(list(self.model.parameters())))

    def test_layerwise_rates_preserve_head_and_upper_layer_peaks(self):
        self.settings.update(layer_lr_decay=.8,frontend_lr_factor=.5)
        opt=build_optimizer(self.model,DEFAULTS['optimizer'],self.settings)
        groups={id(p):g for g in opt.param_groups for p in g['params']}
        base=DEFAULTS['optimizer']['learning_rate']*DEFAULTS['optimizer']['backbone_lr_factor']
        for name,p in self.model.named_parameters():
            expected=DEFAULTS['optimizer']['learning_rate']
            if name.startswith('backbone.blocks.'):
                i=int(name.split('.')[2]);expected=base*.8**(1-i)
            elif name.startswith('backbone.norm.'):expected=base
            elif name.startswith('backbone.'):expected=base*.8**2*.5
            self.assertAlmostEqual(groups[id(p)]['lr'],expected)

    def test_anchor_uses_original_pretraining_reference_after_resume(self):
        self.settings.update(strategy='full',anchor_weight=.2)
        control=FineTuneController(self.model,self.settings);control.apply(0)
        self.assertEqual(float(control.anchor_loss().detach()),0.)
        original=control.anchor_state()
        with torch.no_grad():next(self.model.backbone.parameters()).add_(.1)
        loss=control.anchor_loss();self.assertGreater(float(loss.detach()),0.)
        restored=FineTuneController(self.model,self.settings,original);restored.apply(3)
        torch.testing.assert_close(loss,restored.anchor_loss(),rtol=0,atol=0)
        restored.anchor_loss().backward()
        self.assertTrue(any(p.grad is not None and bool((p.grad!=0).any()) for p in self.model.backbone.parameters()))

    def test_invalid_phase_configuration_is_rejected(self):
        c=deepcopy(DEFAULTS);c['finetuning'].update(self.settings)
        resolve_config(c)
        c['finetuning']['unfreeze_last_n_blocks']=c['model']['encoder']['depth']+1
        with self.assertRaisesRegex(ValueError,'depth'):resolve_config(c)

    def test_full_strategy_preserves_legacy_optimizer_and_forward_updates(self):
        reference=deepcopy(self.model)
        settings=deepcopy(DEFAULTS['finetuning'])
        original=build_optimizer(reference,DEFAULTS['optimizer'])
        updated=build_optimizer(self.model,DEFAULTS['optimizer'],settings)
        control=FineTuneController(self.model,settings)
        x=torch.randn(2,512,2);lengths=torch.tensor([512,512])
        for step in range(3):
            reference.train();control.apply(step)
            for model,opt in [(reference,original),(self.model,updated)]:
                torch.manual_seed(31+step);opt.zero_grad(set_to_none=True)
                model(x,lengths)['modulation'].square().mean().backward();opt.step()
        self.assertTrue(all(torch.equal(v,reference.state_dict()[k]) for k,v in self.model.state_dict().items()))

    def test_nccl_initialization_explicitly_binds_local_device(self):
        with patch.dict(os.environ,{'RANK':'0','LOCAL_RANK':'0','WORLD_SIZE':'2','CUDA_VISIBLE_DEVICES':'2,3'}), \
             patch('torch.cuda.set_device'),patch('torch.distributed.is_initialized',return_value=False), \
             patch('torch.distributed.init_process_group') as initialize:
            device,rank,world=runtime('cuda',2)
            self.assertEqual((str(device),rank,world),('cuda:0',0,2))
            initialize.assert_called_once_with('nccl',device_id=torch.device('cuda:0'))


if __name__=='__main__':unittest.main(verbosity=2)
