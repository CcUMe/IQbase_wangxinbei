"""Mechanism, initialization, task-sharing and export checks for phase adapter."""
from copy import deepcopy
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F

from infer import Predictor
from models.phase_relations import PhaseRelations, phase_relation_options
from models.rf_model import RFModel
from training.test_multiresolution import fixture
from utils.common import TARGETS, save_safetensors, write_json
from utils.config import resolve_config, training_options
from utils.contrastive import load_with_training_aux


class PhaseRelationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(47)

    def models(self):
        template, _, config = fixture()
        schema, scaler, grids = template.schema, {'mean':[0.]*7,'std':[1.]*7}, [[0.,1.]]*7
        torch.manual_seed(43)
        old = RFModel(training_options(resolve_config(config))['model'], schema, scaler, grids)
        config['model']['phase_relations'].update(enabled=True, width=16, dropout=0.)
        torch.manual_seed(43)
        new = RFModel(training_options(resolve_config(config))['model'], schema, scaler, grids)
        return old, new, config

    def test_known_phase_increment_curvature_and_nonwrapping_history(self):
        options=phase_relation_options(dict(lags=[1,2],phase_orders=[1,2],curvature_orders=[1]))
        relation=PhaseRelations(options)
        omega=.17;theta=torch.arange(128)*omega
        iq=torch.stack([theta.cos(),theta.sin()],-1)[None]
        mask=torch.ones(1,128,dtype=torch.bool)
        actual=relation(iq,mask)
        weight=1/(1+options['magnitude_floor']**2)
        expected=torch.tensor([math.cos(omega)*weight,math.sin(omega)*weight,
            math.cos(2*omega)*weight,math.sin(2*omega)*weight,weight**2,0.,weight,weight**2])
        torch.testing.assert_close(actual[0,20,:8],expected,atol=2e-6,rtol=2e-6)
        self.assertEqual(actual[0,0].abs().sum().item(),0.)
        self.assertEqual(actual[0,1,4:6].abs().sum().item(),0.)
        dirty=iq.clone();dirty[:,-1]=100
        torch.testing.assert_close(actual[:,:100],relation(dirty,mask)[:,:100],atol=0.,rtol=0.)
        zero=torch.zeros(1,128,2,requires_grad=True)
        value=relation(zero,mask)
        self.assertEqual(value.abs().sum().item(),0.)
        value.sum().backward();self.assertTrue(torch.isfinite(zero.grad).all())

    def test_global_phase_and_normalized_gain_invariance_curvature_cancels_linear_ramp(self):
        old,new,_=self.models();relation=new.backbone.pyramid.phase_relations.relations
        x=torch.randn(2,512,2);lengths=torch.tensor([512,512]);mask=torch.ones(2,512,dtype=torch.bool)
        def normalized(a):return old.backbone.preprocess(a,lengths)[0]
        c,s=math.cos(.7),math.sin(.7)
        rotated=torch.stack([x[...,0]*c-x[...,1]*s,x[...,0]*s+x[...,1]*c],-1)
        original=relation(normalized(x),mask)
        torch.testing.assert_close(original,relation(normalized(rotated),mask),atol=5e-6,rtol=5e-5)
        torch.testing.assert_close(original,relation(normalized(x*4),mask),atol=0.,rtol=0.)
        theta=torch.arange(512)*.05
        base=torch.stack([theta.cos(),theta.sin()],-1)[None]
        theta=theta+torch.arange(512)*.03
        shifted=torch.stack([theta.cos(),theta.sin()],-1)[None]
        a=relation(base,mask[:1]);b=relation(shifted,mask[:1])
        width=12
        for index,lag in enumerate(relation.options['lags']):
            # First six coordinates are phase harmonics, next four curvature.
            torch.testing.assert_close(a[:,lag+2:,index*width+6:index*width+10],
                                       b[:,lag+2:,index*width+6:index*width+10],atol=2e-5,rtol=2e-5)
        self.assertGreater((a[:,:,0:2]-b[:,:,0:2]).abs().max().item(),.01)

    def test_same_initialization_and_exact_zero_residual_old_weight_warmstart(self):
        old,new,_=self.models()
        a,b=old.state_dict(),new.state_dict()
        for key in a:torch.testing.assert_close(a[key],b[key],atol=0.,rtol=0.,msg=key)
        self.assertTrue(all(k.startswith('backbone.pyramid.phase_relations.') for k in b.keys()-a.keys()))
        result=load_with_training_aux(new,a,old.config)
        self.assertTrue(result.missing_keys and not result.unexpected_keys)
        old.eval();new.eval();iq=torch.randn(2,1024,2);lengths=torch.tensor([512,1024])
        with torch.no_grad():before,after=old(iq,lengths),new(iq,lengths)
        for task in ['modulation','protocol','parameters']:
            torch.testing.assert_close(before[task],after[task],atol=0.,rtol=0.)
        self.assertGreater(new.parameter_report()['shared_ratio'],.8)
        changed=deepcopy(new.config);changed['phase_relations']['lags']=[1,2,4,8,16,32,128]
        with self.assertRaisesRegex(ValueError,'matching phase_relations'):
            load_with_training_aux(new,new.state_dict(),changed)
        with self.assertRaisesRegex(ValueError,'matching phase_relations'):
            load_with_training_aux(old,new.state_dict(),new.config)

    def test_nonzero_adapter_padding_grid_bounds_and_single_extraction(self):
        _,model,_=self.models();adapter=model.backbone.pyramid.phase_relations;model.eval()
        with torch.no_grad():
            for layer in adapter.projections:torch.nn.init.normal_(layer.weight,std=.1)
        iq=torch.randn(2,1024,2);lengths=torch.tensor([512,1024]);calls=[];observed=[]
        handle=adapter.relations.register_forward_hook(lambda m,a,o:calls.append(1))
        capture=adapter.register_forward_hook(lambda m,a,o:observed.append(o))
        with torch.no_grad():
            padded=model(iq,lengths)
            native=model(iq[:1,:512],lengths[:1])
            dirty=iq.clone();dirty[0,512:]=1000*torch.randn(512,2)
            changed=model(dirty,lengths)
        handle.remove();capture.remove()
        self.assertEqual(len(calls),3)
        self.assertEqual([tuple(v.shape) for v in observed[0]],[(2,64,32),(2,128,24),(2,256,24)])
        for update,stride in zip(observed[0],[16,8,4]):
            self.assertEqual(update[0,512//stride:].abs().sum().item(),0.)
            self.assertLessEqual(update.abs().max().item(),adapter.options['max_residual_scale'])
        for task in ['modulation','protocol','parameters']:
            torch.testing.assert_close(padded[task][:1],native[task],atol=3e-5,rtol=3e-5)
            torch.testing.assert_close(padded[task],changed[task],atol=0.,rtol=0.)

    def test_injection_reaches_coarse_attention_and_both_detail_local_encoders(self):
        _,model,_=self.models();model.eval();adapter=model.backbone.pyramid.phase_relations
        with torch.no_grad():
            for layer in adapter.projections:torch.nn.init.normal_(layer.weight,std=.1)
        recorded=[[],[],[]]
        modules=[model.backbone.blocks[0],*[branch.local[0] for branch in model.backbone.pyramid.branches]]
        def capture(index):
            return lambda m,a:recorded[index].append(a[0].detach().clone())
        hooks=[module.register_forward_pre_hook(capture(i)) for i,module in enumerate(modules)]
        iq=torch.randn(1,512,2);lengths=torch.tensor([512])
        with torch.no_grad():
            model(iq,lengths)
            handle=adapter.register_forward_hook(lambda m,a,o:[torch.zeros_like(v) for v in o])
            model(iq,lengths);handle.remove()
        for hook in hooks:hook.remove()
        for a,b in recorded:self.assertGreater((a-b).abs().max().item(),1e-5)

    def test_bfloat16_zero_residual_preserves_old_dtype_and_predictions(self):
        old,new,_=self.models();load_with_training_aux(new,old.state_dict(),old.config)
        old.eval();new.eval();iq=torch.randn(2,1024,2);lengths=torch.tensor([512,1024])
        with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
            a,b=old(iq,lengths),new(iq,lengths)
        for task in ['modulation','protocol','parameters']:
            self.assertEqual(a[task].dtype,b[task].dtype)
            torch.testing.assert_close(a[task],b[task],atol=0.,rtol=0.)

    def test_nonzero_adapter_portable_spectrum_matches_fft(self):
        _,model,_=self.models();model.eval()
        with torch.no_grad():
            for layer in model.backbone.pyramid.phase_relations.projections:
                torch.nn.init.normal_(layer.weight,std=.1)
            iq=torch.randn(1,512,2);lengths=torch.tensor([512])
            a=model(iq,lengths);model.config['spectral_backend']='matmul';b=model(iq,lengths)
        for task in ['modulation','protocol','parameters']:
            torch.testing.assert_close(a[task],b[task],atol=3e-4,rtol=3e-4)

    def test_each_task_trains_shared_adapter_after_zero_projection_first_update(self):
        for task in ['modulation','protocol','parameters']:
            with self.subTest(task=task):
                _,model,_=self.models();model.train();adapter=model.backbone.pyramid.phase_relations
                optimizer=torch.optim.SGD(model.parameters(),lr=.02)
                for update in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cpu',dtype=torch.bfloat16):
                        out=model(torch.randn(2,512,2),torch.tensor([512,512]))
                        loss=out[task].square().mean() if task=='parameters' else F.cross_entropy(out[task],torch.tensor([0,1]))
                    loss.backward()
                    for parameter in adapter.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    if update==1:
                        self.assertGreater(adapter.input.weight.grad.abs().sum().item(),0.)
                        for layer in adapter.projections:self.assertGreater(layer.weight.grad.abs().sum().item(),0.)
                    optimizer.step()

    def test_export_contract_and_invalid_configuration(self):
        _,model,config=self.models();model.eval()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);save_safetensors(root/'model.safetensors',model.state_dict())
            write_json(root/'config.json',dict(model_config=model.config,schema=model.schema,
                target_scaler={'mean':[0.]*7,'std':[1.]*7},parameter_grids=[[0.,1.]]*7))
            predictor=Predictor(root,device='cpu',threads=2)
            records=[dict(task=task,sample_id=str(i),iq=torch.randn(512 if i%2 else 1024,2).tolist())
                     for i,task in enumerate(['modulation','protocol','parameter'])]
            output=predictor.predict(records)
        for record,prediction in zip(records,output):
            self.assertEqual(set(prediction),{'task','sample_id','prediction'})
            self.assertEqual(prediction['sample_id'],record['sample_id'])
            if record['task']=='parameter':self.assertEqual(set(prediction['prediction']),set(TARGETS))
            else:self.assertIn(prediction['prediction'],model.schema['classification_labels'][record['task']])
        for key,value in [('lags',[1,1]),('lags',[512]),('phase_orders',[3]),('width',0),
                          ('local_kernel',2),('magnitude_floor',0.),('residual_scale',.3),('dropout',1.)]:
            bad=deepcopy(config);bad['model']['phase_relations'][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'phase_relations'):resolve_config(bad)
        bad=deepcopy(config);bad['model']['pyramid']['enabled']=False
        with self.assertRaisesRegex(ValueError,'requires the pyramid'):resolve_config(bad)


if __name__=='__main__':
    unittest.main(verbosity=2)
