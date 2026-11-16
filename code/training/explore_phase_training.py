"""Serial single-GPU diagnostics/short paired continuations of frozen snapshots.

Source jobs/checkpoints/configs are read-only. Research models have new optimizer
and schedule; this is neither strict resume nor a from-scratch architecture trial.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from models.rf_model import RFModel
from training.train import build_optimizer
from training.validate_c01_followup import predict,task_metrics,class_result
from utils.common import model_bundle,read_json,write_json,save_checkpoint,seed_all
from utils.config import resolve_config,training_options,make_scheduler
from utils.dataset import PreparedDataset,BalancedBatchSampler,collate,to_device
from utils.engine import autocast,supervised_loss
from utils.head_optimizer import active_heads,clear_inactive_head_gradients
from utils.modulation_regularization import modulation_correction
from utils.augmentation import augment_supervised_iq

ARMS={'low_lr_control':dict(smoothing=0.,double=False,kl=0.),
      'mod_smoothing_005':dict(smoothing=.05,double=False,kl=0.),
      'double_ce_control':dict(smoothing=0.,double=True,kl=0.),
      'mod_rdrop_01':dict(smoothing=0.,double=True,kl=.1),
      'mod_shift_05':dict(smoothing=0.,double=False,kl=0.,shift=.5),
      'mod_rdrop_10':dict(smoothing=0.,double=True,kl=1.)}


def checkpoint_model(path,device):
    c=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    model=RFModel(**model_bundle(c)).to(device)
    model.load_state_dict(c['model'],strict=True)
    return c,model


def evaluate_saved(model,c,rows,iq,meta,args,name,variant='original'):
    path=args.output/(name+'.npz')
    if path.exists():return dict(np.load(path,allow_pickle=False))
    model.eval()
    p=predict(model,rows,iq,meta,c['target_scaler'],args.device,args.eval_batch_size,name)
    np.savez_compressed(path,**p)
    write_json(args.output/(name+'_metrics.json'),task_metrics(p,meta,c['schema']))
    return p


@torch.inference_mode()
def profile_adapter(model,rows,iq,meta,args):
    adapter=model.backbone.pyramid.phase_relations
    stash={};captured={i:[] for i in range(3)};saturation={i:[] for i in range(3)}
    hooks=[adapter.register_forward_hook(lambda m,a,o:stash.update(updates=[v.detach().float() for v in o]))]
    def inspect_input(index):
        def hook(m,a):
            h=a[0].detach().float();update=stash['updates'][index]
            base=h-update
            ratio=update.square().mean((1,2)).sqrt()/base.square().mean((1,2)).sqrt().clamp_min(1e-9)
            captured[index].extend(ratio.cpu().tolist())
        return hook
    modules=[model.backbone.blocks[0],*[b.local[0] for b in model.backbone.pyramid.branches]]
    hooks += [m.register_forward_pre_hook(inspect_input(i)) for i,m in enumerate(modules)]
    def inspect_projection(index):
        def hook(m,a,o):saturation[index].extend((o.float().abs()>2).float().mean((1,2)).cpu().tolist())
        return hook
    hooks += [m.register_forward_hook(inspect_projection(i)) for i,m in enumerate(adapter.projections)]
    try:
        model.eval()
        for length in [512,1024]:
            selected=rows[meta['length'][rows]==length]
            for start in range(0,len(selected),64):
                r=selected[start:start+64]
                model(torch.from_numpy(np.array(iq[r,:length])).to(args.device),
                      torch.full((len(r),),length,device=args.device))
    finally:
        for hook in hooks:hook.remove()
    return {'gains':(adapter.gain_logits.sigmoid()*.25).cpu().tolist(),
        'scale_update_to_original_rms':{str(i):{'median':float(np.median(v)),'p95':float(np.quantile(v,.95))} for i,v in captured.items()},
        'scale_projection_saturation_fraction':{str(i):float(np.mean(v)) for i,v in saturation.items()}}


def diagnose(args,data,selection,iq):
    meta=data.meta
    val=data.indices;modval=val[meta['modulation'][val]>=0]
    summary={}
    for name in ['best','last']:
        c,model=checkpoint_model(args.output/'snapshots'/f'{name}.pt',args.device)
        summary[name]={'epoch':c['epoch']+1,'step':c['global_step'],'saved_metrics_validation_step':c.get('validation_global_step')}
        evaluate_saved(model,c,val,iq,meta,args,f'source_{name}_validation')
        evaluate_saved(model,c,selection['train_sample'],iq,meta,args,f'source_{name}_train_sample')
        summary[name]['adapter']=profile_adapter(model,selection['panel_validation'],iq,meta,args)
        for variant,factors in [('off',[0.,0.,0.]),('coarse_off',[0.,1.,1.]),
                                ('details_off',[1.,0.,0.]),('half',[.5,.5,.5]),('double',[2.,2.,2.])]:
            if name=='last' and variant in ['coarse_off','details_off','double']:continue
            handle=model.backbone.pyramid.phase_relations.register_forward_hook(
                lambda m,a,o,f=factors:[v*factor for v,factor in zip(o,f)])
            try:evaluate_saved(model,c,modval,iq,meta,args,f'source_{name}_{variant}')
            finally:handle.remove()
        del model,c;torch.cuda.empty_cache()
        write_json(args.output/'adapter_profile.json',summary)
    # A single weight-averaged model from this trajectory, not prediction ensembling.
    c=torch.load(args.output/'snapshots/best.pt',map_location='cpu',weights_only=True,mmap=True)
    others=[torch.load(args.output/'snapshots'/f'{name}.pt',map_location='cpu',weights_only=True,mmap=True) for name in ['best_proxy','last']]
    average={key:(sum([v.float(),*[x['model'][key].float() for x in others]])/3).to(v.dtype)
             if v.is_floating_point() else v.clone() for key,v in c['model'].items()}
    model=RFModel(**model_bundle(c)).to(args.device);model.load_state_dict(average,strict=True)
    evaluate_saved(model,c,val,iq,meta,args,'trajectory_average_validation')
    payload={k:c[k] for k in ['model_config','schema','target_scaler','parameter_grids','dataset_fingerprint','stage','training_config']}
    payload.update(model=average,global_step=max([c['global_step'],*[x['global_step'] for x in others]]),resume_allowed=False,
        research_note='Equal weight average of source best/best_proxy/last, a single model; exploratory validation-selected snapshots.')
    save_checkpoint(args.output/'trajectory_average.pt',payload)
    del model,c,others,average;torch.cuda.empty_cache()


def train_arms(args,data,selection,iq):
    train=PreparedDataset(args.data,'train');meta=data.meta
    source=torch.load(args.output/'snapshots/best.pt',map_location='cpu',weights_only=True,mmap=True)
    steps=args.steps;seed=getattr(args,'seed',20261114)
    stream_path=args.output/'training_stream.npy'
    if stream_path.exists():stream=np.load(stream_path)
    else:
        stream=np.asarray(list(BalancedBatchSampler(train,args.batch_size,steps,seed,mode='balanced')),dtype=np.int64)
        np.save(stream_path,stream)
    assert stream.shape==(steps,args.batch_size)
    stream_hash=hashlib.sha256(train.indices[stream].tobytes()).hexdigest()
    result_path=args.output/'training_results.json'
    results=read_json(result_path) if result_path.exists() else {}
    for name,regularizer in ARMS.items():
        if name in results:continue
        started=time.monotonic();seed_all(seed)
        model=RFModel(**model_bundle(source)).to(args.device);model.load_state_dict(source['model'],strict=True)
        config=resolve_config(source['training_config'])
        config['runtime'].update(gpus=os.environ['CUDA_VISIBLE_DEVICES'],processes=1)
        config['optimizer'].update(learning_rate=3e-5,backbone_lr_factor=1.)
        config['training'].update(epochs=1,max_steps=steps)
        config['data'].update(steps_per_epoch=steps,batch_size=args.batch_size)
        if regularizer.get('shift',0):
            config['data']['augmentation']['modulation_only']['circular_shift_probability']=regularizer['shift']
        if regularizer.get('phase',0):
            config['data']['augmentation']['modulation_only']['phase_rotation_probability']=regularizer['phase']
        config['scheduler'].update(warmup_steps=min(30,max(steps-1,0)),warmup_ratio=0.,warmup_start_factor=.1,min_lr_ratio=.2)
        optimizer=build_optimizer(model,config['optimizer'],config['finetuning'])
        schedule,_=make_scheduler(optimizer,config['scheduler'],steps)
        options=training_options(config);ema={k:v.detach().clone() for k,v in model.state_dict().items()}
        totals={'mod_ce':[],'mod_kl':[],'gradient_norm':[]};log=[]
        model.train()
        progress=tqdm(range(steps),desc=name,mininterval=10)
        for step in progress:
            # Same first-pass dropout across arms, independent of extra R-Drop pass.
            torch.manual_seed(seed+step);torch.cuda.manual_seed(seed+step)
            cpu_batch=collate([train[int(i)] for i in stream[step]])
            active=active_heads(cpu_batch,options)
            batch=to_device(cpu_batch,args.device)
            if any(regularizer.get(k,0) for k in ['shift','phase','noise']):
                # Augmentation must not consume the matched first-pass Dropout stream.
                with torch.random.fork_rng(devices=[torch.device(args.device).index]):
                    torch.manual_seed(seed+1_000_000+step)
                    torch.cuda.manual_seed(seed+1_000_000+step)
                    batch['iq']=augment_supervised_iq(batch,config['data']['augmentation'])
                    if regularizer.get('noise',0):
                        # Relative to observed record energy; NOT an estimate of clean SNR.
                        eligible=(batch['modulation']>=0)&(batch['protocol']<0)&~batch['target_valid'].any(-1)
                        selected=eligible&(torch.rand(len(batch['iq']),device=args.device)<.5)
                        valid=torch.arange(batch['iq'].shape[1],device=args.device)[None]<batch['lengths'][:,None]
                        original=batch['iq'];energy=original.square().sum((1,2))
                        rms=(energy/batch['lengths']).sqrt()
                        relative=.02+torch.rand(len(original),device=args.device)*(regularizer['noise']-.02)
                        noise=torch.randn_like(original)*(rms*relative/(2**.5))[:,None,None]
                        changed=(original+noise)*valid[...,None]
                        changed*= (energy/changed.square().sum((1,2)).clamp_min(1e-20)).sqrt()[:,None,None]
                        batch['iq']=torch.where(selected[:,None,None],changed,original)
            optimizer.zero_grad(set_to_none=True)
            with autocast(torch.device(args.device),'bf16'):
                output=model(batch['iq'],batch['lengths'])
                loss,parts=supervised_loss(output,batch,source['target_scaler'],source['parameter_grids'],options,distributed=False)
                second=(model(batch['iq'],batch['lengths']) if regularizer['double'] and bool((batch['modulation']>=0).any()) else None)
                correction,detail=modulation_correction(output,batch,regularizer['smoothing'],second,regularizer['kl'])
                loss=loss+options['task_weights'][0]*correction
            if not bool(torch.isfinite(loss)):raise FloatingPointError(f'{name}/{step}: nonfinite loss')
            loss.backward()
            clear_inactive_head_gradients(model,active,torch.device(args.device))
            if regularizer.get('sam',0) and bool((batch['modulation']>=0).any()):
                # SAM only on modulation steps; originals are cloned so restoration
                # is exact rather than accumulating in-place add/subtract roundoff.
                parameters=[p for p in model.parameters() if p.grad is not None]
                norm=torch.stack([p.grad.float().norm() for p in parameters]).norm().clamp_min(1e-12)
                originals=[p.detach().clone() for p in parameters]
                with torch.no_grad():
                    for p in parameters:p.add_(p.grad*(regularizer['sam']/norm))
                optimizer.zero_grad(set_to_none=True)
                try:
                    with autocast(torch.device(args.device),'bf16'):
                        perturbed=model(batch['iq'],batch['lengths'])
                        robust_loss,_=supervised_loss(perturbed,batch,source['target_scaler'],source['parameter_grids'],options,distributed=False)
                    if not bool(torch.isfinite(robust_loss)):raise FloatingPointError('Nonfinite SAM loss')
                    detail['sam_perturbed_loss']=float(robust_loss.detach())
                    detail['sam_loss_gap']=float(robust_loss.detach()-loss.detach())
                    robust_loss.backward()
                    clear_inactive_head_gradients(model,active,torch.device(args.device))
                finally:
                    with torch.no_grad():
                        for p,original in zip(parameters,originals):p.copy_(original)
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            if not math.isfinite(grad):raise FloatingPointError('Nonfinite gradients')
            optimizer.step();schedule.step()
            with torch.no_grad():
                for key,value in model.state_dict().items():
                    if value.is_floating_point():ema[key].lerp_(value.detach(),.005)
                    else:ema[key].copy_(value)
            if bool((batch['modulation']>=0).any()):
                for key in ['mod_ce','mod_kl']:totals[key].append(detail[key])
            totals['gradient_norm'].append(grad)
            if (step+1)%50==0:
                item={'step':step+1,'loss':float(loss.detach()),'gradient_norm':grad,'lr':optimizer.param_groups[0]['lr'],**detail}
                log.append(item);write_json(args.output/(name+'_progress.json'),log)
                progress.set_postfix(loss=f'{float(loss.detach()):.3f}',grad=f'{grad:.2f}')
        payload={k:source[k] for k in ['model_config','schema','target_scaler','parameter_grids','dataset_fingerprint','stage']}
        payload.update(training_config=config,global_step=steps,initialization_source_steps=source['global_step'],
                       resume_allowed=False,research_regularizer=regularizer,training_stream_sha256=stream_hash)
        scores={}
        for estimator,state in [('last',{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}),
                                ('ema',{k:v.cpu() for k,v in ema.items()})]:
            model.load_state_dict(state,strict=True)
            p=evaluate_saved(model,source,data.indices,iq,meta,args,f'{name}_{estimator}_validation')
            metric=task_metrics(p,meta,source['schema']);scores[estimator]=metric
            save_checkpoint(args.output/f'{name}_{estimator}.pt',{**payload,'model':state,'metrics':metric,'weight_estimator':estimator})
        results[name]={'metrics':scores,'stream_sha256':stream_hash,'steps':steps,'modulation_updates':len(totals['mod_ce']),
            'average_mod_ce':float(np.mean(totals['mod_ce'])),'average_mod_kl':float(np.mean(totals['mod_kl'])),
            'grad_median':float(np.median(totals['gradient_norm'])),'grad_p95':float(np.quantile(totals['gradient_norm'],.95)),
            'elapsed_seconds':time.monotonic()-started,'regularizer':regularizer}
        write_json(result_path,results)
        print('ARM COMPLETE',name,{k:{t:round(x['accuracy'],5) for t,x in m['classification'].items()} for k,m in scores.items()},flush=True)
        del model,optimizer,schedule,ema,state;torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'artifacts/data_full')
    p.add_argument('--phase',choices=['all','diagnose','train'],default='all')
    p.add_argument('--steps',type=int,default=900)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--eval-batch-size',type=int,default=128)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(visible)!=1 or not visible[0] or int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise ValueError('Run with exactly one visible GPU and one process')
    if torch.cuda.device_count()!=1:raise ValueError('Exactly one GPU is required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    data=PreparedDataset(args.data,'validation');iq=np.load(args.data/'iq.npy',mmap_mode='r')
    selection=dict(np.load(ROOT/'artifacts/c01_followup_validation/20261113_224434/selection.npz',allow_pickle=False))
    protocol=read_json(args.output/'protocol.json');protocol.update(planned_arms=list(ARMS),training_stream_seed=20261114,
        note='Double-CE control added to separate two-forward averaging from symmetric KL. All experiments serial on one physical GPU.',
        rdrop_source='https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html')
    write_json(args.output/'protocol.json',protocol)
    if args.phase in ['all','diagnose']:diagnose(args,data,selection,iq)
    if args.phase in ['all','train']:train_arms(args,data,selection,iq)
    print('FINISHED',args.output,flush=True)


if __name__=='__main__':main()
