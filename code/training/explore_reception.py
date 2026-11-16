"""Matched single-GPU joint-task reception-memory trials, preserving source files."""
import argparse
from datetime import datetime
import hashlib
import math
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from tqdm import tqdm

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from models.rf_model import RFModel
from models.reception import attach_reception,RECEPTION_DEFAULTS,reception_model
from models.carrier_input import attach_carrier_input
from training.validate_c01_followup import predict,task_metrics,class_result
from utils.common import model_bundle,read_json,write_json,save_checkpoint,seed_all
from utils.config import resolve_config,training_options,make_scheduler
from utils.dataset import PreparedDataset,BalancedBatchSampler,collate,to_device
from utils.engine import autocast,supervised_loss
from utils.head_optimizer import active_heads,clear_inactive_head_gradients
from utils.tensorboard_log import TensorBoardLog


def optimizer_for(model,settings):
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        receiver=name.startswith('backbone.reception.')
        predictor=receiver and '.estimate' in name
        lr=3e-4 if receiver and not predictor else 3e-5
        decay=settings['weight_decay']
        if p.ndim==1 and (receiver or settings['exclude_bias_and_norm_from_decay']):decay=0.
        groups.setdefault((lr,decay),[]).append(p)
    return torch.optim.AdamW([dict(params=p,lr=lr,weight_decay=wd) for (lr,wd),p in groups.items()],
        betas=settings['betas'],eps=settings['eps'])


def condition_metrics(pred,meta,schema,selection):
    rows=pred['rows'];y=meta['modulation'][rows]
    result={}
    for q in range(4):
        keep=(y>=0)&(y<14)&(selection['condition_groups'][rows,0]==q)
        result[f'first14_rms_q{q+1}']=class_result(pred['modulation'],y,schema['classification_labels']['modulation'],keep)
    return result


def evaluate(model,source,rows,iq,meta,selection,args,name):
    path=args.output/(name+'.npz')
    model.eval()
    pred=predict(model,rows,iq,meta,source['target_scaler'],'cuda:0',args.eval_batch_size,name)
    np.savez_compressed(path,**pred)
    metric=task_metrics(pred,meta,source['schema'])
    metric['conditions']=condition_metrics(pred,meta,source['schema'],selection)
    write_json(args.output/(name+'_metrics.json'),metric)
    return metric


def diagnostic(module):
    if hasattr(module,'summary'):return module.summary()
    values=module.last_diagnostics
    result={'gain':float(module.options['max_gain']*module.gain_logit.detach().sigmoid())}
    for key in ['coherence','frequency_delta','clock_log_delta','timing_delta','equalizer_delta']:
        x=values[key].float();result[key+'_abs_mean']=float(x.abs().mean())
        result[key+'_abs_max']=float(x.abs().max())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--source',type=Path,default=ROOT/'logs/finetune_v2/20261112_145913_822471/best.pt')
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--eval-batch-size',type=int,default=128)
    p.add_argument('--arms',default='control,fixed_reception,learned_reception')
    p.add_argument('--seed',type=int,default=20261114)
    p.add_argument('--predictor-init',type=Path)
    p.add_argument('--balanced-scales',action='store_true')
    p.add_argument('--freeze-predictor',action='store_true')
    p.add_argument('--match-token-scale',action='store_true')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or torch.cuda.device_count()!=1 or int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise ValueError('Exactly physical GPU 0 and a single process are required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    args.output=args.output or ROOT/'artifacts/reception_exploration'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'protocol.json').exists():raise ValueError('Use a fresh timestamped output directory')
    arms=args.arms.split(',')
    if not set(arms)<={'control','fixed_reception','learned_reception','raw_front','lag1_front','m4_front'}:raise ValueError('Unknown arm')
    if args.predictor_init and arms!=['learned_reception']:raise ValueError('Use a separate learned_reception run for calibrated initialization')
    if args.freeze_predictor and not args.predictor_init:raise ValueError('Freezing requires a calibrated predictor')
    if args.smoke:args.steps=6;args.batch_size=32;args.eval_batch_size=32
    train=PreparedDataset(ROOT/'artifacts/data_full','train');val=PreparedDataset(ROOT/'artifacts/data_full','validation')
    meta=train.meta;iq=np.load(ROOT/'artifacts/data_full/iq.npy',mmap_mode='r')
    selection=dict(np.load(ROOT/'artifacts/c01_followup_validation/20261113_224434/selection.npz',allow_pickle=False))
    source=torch.load(args.source,map_location='cpu',weights_only=True,mmap=True)
    if source['dataset_fingerprint']!=train.info['fingerprint']:raise ValueError('Data/source fingerprint mismatch')
    stream=np.asarray(list(BalancedBatchSampler(train,args.batch_size,args.steps,args.seed,mode='balanced')),dtype=np.int64)
    np.save(args.output/'training_stream.npy',stream)
    stream_hash=hashlib.sha256(train.indices[stream].tobytes()).hexdigest()
    validation_rows=val.indices
    if args.smoke:
        keep=[]
        for length in [512,1024]:
            for eligible in [meta['modulation']>=0,meta['protocol']>=0,meta['target_valid'].any(1)]:
                keep.extend(val.indices[eligible[val.indices]&(meta['length'][val.indices]==length)][:4])
        validation_rows=np.unique(keep)
    # Fixed train-side rows for source equivalence / sanity; no fitting on validation.
    verification=[]
    for length in [512,1024]:
        for eligible in [meta['modulation']>=0,meta['protocol']>=0,meta['target_valid'].any(1)]:
            verification.extend(train.indices[eligible[train.indices]&(meta['length'][train.indices]==length)][:2])
    base=RFModel(**model_bundle(source)).cuda();base.load_state_dict(source['model'],strict=True);base.eval()
    reference={}
    with torch.inference_mode():
        for length in [512,1024]:
            rows=np.asarray(verification);rows=rows[meta['length'][rows]==length]
            x=torch.from_numpy(np.array(iq[rows,:length])).cuda();lengths=torch.full((len(rows),),length,device='cuda:0')
            reference[length]={key:value.detach().cpu() for key,value in base(x,lengths).items() if key in ['modulation','protocol','parameters']}
    del base;torch.cuda.empty_cache()
    config=resolve_config(source['training_config'])
    config['optimizer'].update(learning_rate=3e-5,backbone_lr_factor=1.)
    config['runtime'].update(gpus='0',processes=1,seed=args.seed,precision='bf16',threads=4)
    config['training'].update(epochs=1,max_steps=args.steps,gradient_accumulation_steps=1)
    config['data'].update(batch_size=args.batch_size,steps_per_epoch=args.steps)
    config['scheduler'].update(name='cosine',warmup_steps=min(30,args.steps-1),warmup_ratio=0.,warmup_start_factor=.1,min_lr_ratio=.2)
    config['evaluation'].update(batch_size=args.eval_batch_size,precision='fp32',max_batches=0)
    config['run'].update(output=str(args.output),warmstart_checkpoint=str(args.source))
    options=training_options(config)
    source_hash=hashlib.sha256()
    with args.source.open('rb') as handle:
        while block:=handle.read(8*1024*1024):source_hash.update(block)
    protocol=dict(started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),physical_gpu='0',processes=1,
        source_checkpoint=str(args.source.resolve()),source_steps=source['global_step'],dataset_fingerprint=train.info['fingerprint'],
        source_sha256=source_hash.hexdigest(),
        predictor_initialization=str(args.predictor_init.resolve()) if args.predictor_init else None,
        calibration_data_source='Explicit synthetic signal generator; see predictor checkpoint protocol' if args.predictor_init else None,
        arms=arms,steps=args.steps,batch_size=args.batch_size,seed=args.seed,smoke=args.smoke,training_stream_sha256=stream_hash,
        optimizer=dict(name='AdamW',base_lr=3e-5,receiver_lr=3e-4,transform_predictor_lr=3e-5,weight_decay=config['optimizer']['weight_decay']),
        scheduler=dict(name='cosine',warmup_steps=min(30,args.steps-1),minimum_ratio=.2),
        receiver={**RECEPTION_DEFAULTS,'balance_scales':args.balanced_scales,'freeze_predictor':args.freeze_predictor,'match_token_scale':args.match_token_scale},transform_regularizer=.001,ema_decay=.995,
        input_projection_modes=dict(raw_front='matched raw-input duplicate control',lag1_front='lag1 carrier-normalized view',m4_front='lag1 plus split-half-consistent fourth-power refinement'),
        evaluation='Fixed final budget, last and EMA on complete unchanged validation; panel points are monitoring only',
        competition='One shared original foundation, original three task heads; receiver inside shared backbone, no prediction ensemble',
        engineering_gate='overall +1pp AND first14 +2pp over same source, no material other-task degradation; confirm promising increments against same-budget control',
        sources=['https://arxiv.org/abs/1605.00716','https://www.mdpi.com/2076-3417/9/5/1010',
            'https://www.bgd.ed.tum.de/pdf/2024_RFSynchronization_ICMLCN_Petry.pdf'],
        caveats=['Lag-1 centre is not true CFO; clock hypotheses are samples, not physical symbol rates',
            'RRC bank is a receiver hypothesis, not verified transmitter pulse shaping',
            'No true class/parameters/fs/SNR supplied to receiver; no signal reconstruction target',
            'Historical validation already used for research; not a blind hidden test',
            'Research checkpoint requires models.reception.reception_model(); not a bare RFModel export'])
    write_json(args.output/'protocol.json',protocol)
    write_json(args.output/'resolved_config.json',config)
    with (args.output/'train.log').open('w',buffering=1) as readable,(args.output/'events.jsonl').open('w',buffering=1) as events:
        def event(kind,**item):
            data=dict(event=kind,**item);events.write(__import__('json').dumps(data,ensure_ascii=False,allow_nan=False)+'\n')
        results={}
        for arm in arms:
            seed_all(args.seed);started=time.monotonic()
            model=RFModel(**model_bundle(source));model.load_state_dict(source['model'],strict=True)
            receiver_options=None
            if arm.endswith('_front'):
                receiver_options=dict(kind='carrier_input',mode=arm.removesuffix('_front'))
                attach_carrier_input(model,receiver_options)
            elif arm!='control':
                receiver_options={**RECEPTION_DEFAULTS,'learned':arm=='learned_reception','balance_scales':args.balanced_scales,'freeze_predictor':args.freeze_predictor,'match_token_scale':args.match_token_scale}
                attach_reception(model,receiver_options)
                if args.predictor_init:
                    calibration=torch.load(args.predictor_init,map_location='cpu',weights_only=True)
                    state=calibration['predictor'];current=model.backbone.reception.state_dict()
                    expected={key for key in current if key.startswith(('estimate1.','estimate2.','estimate_output.'))}
                    if set(state)!=expected:raise ValueError('Calibrated predictor keys do not match receiver')
                    current.update(state);model.backbone.reception.load_state_dict(current,strict=True)
            model.cuda();model.eval()
            errors={}
            with torch.inference_mode():
                for length in [512,1024]:
                    rows=np.asarray(verification);rows=rows[meta['length'][rows]==length]
                    out=model(torch.from_numpy(np.array(iq[rows,:length])).cuda(),torch.full((len(rows),),length,device='cuda:0'))
                    for key,expected in reference[length].items():
                        error=float((out[key].cpu()-expected).abs().max());errors[f'{length}/{key}']=error
                        if error>2e-5:raise ValueError(f'Nonzero initialization residual: {arm}/{length}/{key}/{error}')
            report=model.parameter_report();report['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer=optimizer_for(model,config['optimizer'])
            scheduler,_=make_scheduler(optimizer,dict(name='cosine',warmup_steps=min(30,args.steps-1),warmup_ratio=0.,
                warmup_start_factor=.1,min_lr_ratio=.2),args.steps)
            ema={key:value.detach().clone() for key,value in model.state_dict().items()}
            print('OVERVIEW',arm,'source',args.source,'steps',args.steps,'batch',args.batch_size,'params',report,'initial errors',errors,flush=True)
            readable.write(f'\n{arm}: source={args.source}; steps={args.steps}; batch={args.batch_size}; parameters={report}; initialization_max_error={max(errors.values())}\n')
            event('start',arm=arm,parameters=report,initialization_errors=errors)
            grad=[];ce=[];progress=[];panels=[];model.train()
            with TensorBoardLog(args.output,log_dir=args.output/'tensorboard'/arm,gradient_clip=1.) as tb:
                tb.metadata(protocol,report)
                bar=tqdm(range(args.steps),desc=arm+' stage1/1',mininterval=5)
                for step in bar:
                    torch.manual_seed(args.seed+step);torch.cuda.manual_seed(args.seed+step)
                    cpu=collate([train[int(i)] for i in stream[step]])
                    active=active_heads(cpu,options);batch=to_device(cpu,'cuda:0')
                    optimizer.zero_grad(set_to_none=True)
                    with autocast(torch.device('cuda:0'),'bf16'):
                        out=model(batch['iq'],batch['lengths'])
                        loss,parts=supervised_loss(out,batch,source['target_scaler'],source['parameter_grids'],options,distributed=False)
                        penalty=model.backbone.reception.regularization if receiver_options is not None else loss*0
                        if args.freeze_predictor:penalty=loss*0
                        total=loss+.001*penalty
                    if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite receiver loss')
                    total.backward();clear_inactive_head_gradients(model,active,torch.device('cuda:0'))
                    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
                    if not math.isfinite(norm):raise FloatingPointError('Nonfinite receiver gradient')
                    optimizer.step();scheduler.step();grad.append(norm)
                    if bool((cpu['modulation']>=0).any()):ce.append(float(parts['modulation']))
                    with torch.no_grad():
                        for key,value in model.state_dict().items():
                            if value.is_floating_point():ema[key].lerp_(value.detach(),.005)
                            else:ema[key].copy_(value)
                    if (step+1)%50==0 or step==0:
                        values=dict(step=step+1,loss=float(loss.detach()),transform_penalty=float(penalty.detach()),gradient_norm=norm,
                            learning_rates=[g['lr'] for g in optimizer.param_groups],parts={k:float(v) for k,v in parts.items()})
                        if receiver_options is not None:values['receiver']=diagnostic(model.backbone.reception)
                        event('step',arm=arm,**values);progress.append(values)
                        write_json(args.output/f'{arm}_progress.json',progress)
                        readable.write(f'{arm} step={step+1}/{args.steps} loss={values["loss"]:.6f} grad={norm:.4f} parts={values["parts"]}\n')
                        tb.scalar('train/total_loss',float(total.detach()),step+1);tb.scalar('train/gradient_preclip',norm,step+1)
                        tb.scalar('train/transform_penalty',float(penalty.detach()),step+1)
                        for name,value in values.get('receiver',{}).items():tb.scalar('receiver/'+name,value,step+1)
                        if bool((cpu['modulation']>=0).any()):tb.scalar('train/modulation_ce',float(parts['modulation']),step+1)
                        bar.set_postfix(loss=f'{float(loss.detach()):.3f}',grad=f'{norm:.2f}')
                    if not args.smoke and (step+1)%400==0 and step+1<args.steps:
                        panel=evaluate(model,source,selection['panel_validation'],iq,meta,selection,args,f'{arm}_panel_{step+1}')
                        panels.append(dict(step=step+1,metrics=panel));event('monitor_panel',arm=arm,step=step+1,metrics=panel)
                        print('MONITOR',arm,step+1,{k:round(v['accuracy'],4) for k,v in panel['classification'].items() if 'accuracy' in v},flush=True)
                        model.train()
                states={'last':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'ema':{k:v.cpu() for k,v in ema.items()}}
                scores={}
                for estimator,state in states.items():
                    model.load_state_dict(state,strict=True)
                    metric=evaluate(model,source,validation_rows,iq,meta,selection,args,f'{arm}_{estimator}_validation')
                    scores[estimator]=metric
                    tb.validation({**metric,'classification':{estimator+'/'+k:v for k,v in metric['classification'].items()},
                        'parameters':{estimator+'/'+k:v for k,v in metric['parameters'].items()}},args.steps)
                    payload={key:source[key] for key in ['model_config','schema','target_scaler','parameter_grids','dataset_fingerprint','stage']}
                    payload.update(model=state,training_config=config,reception_options=receiver_options,research_kind='reception_memory',
                        global_step=args.steps,initialization_source_steps=source['global_step'],resume_allowed=False,
                        metrics=metric,weight_estimator=estimator,training_stream_sha256=stream_hash)
                    save_checkpoint(args.output/f'{arm}_{estimator}.pt',payload)
                    readable.write(f'{arm}/{estimator} VALIDATION '+str({k:{f:v.get(f) for f in ["accuracy","macro_f1","weighted_f1"]} for k,v in metric['classification'].items()})+'\n')
                    event('validation',arm=arm,estimator=estimator,step=args.steps,metrics=metric)
                results[arm]=dict(metrics=scores,parameters=report,initialization_errors=errors,steps=args.steps,
                    modulation_updates=len(ce),mean_modulation_ce=float(np.mean(ce)),gradient_median=float(np.median(grad)),
                    gradient_p95=float(np.quantile(grad,.95)),clipping_fraction=float(np.mean(np.asarray(grad)>1.)),
                    elapsed_seconds=time.monotonic()-started,training_stream_sha256=stream_hash,monitor_panels=panels)
                write_json(args.output/'results.json',results)
                print('ARM COMPLETE',arm,{e:{t:round(v.get('accuracy',0),5) for t,v in m['classification'].items()} for e,m in scores.items()},flush=True)
            del model,optimizer,scheduler,ema,states,state;torch.cuda.empty_cache()
    print('FINISHED',args.output,flush=True)


if __name__=='__main__':main()
