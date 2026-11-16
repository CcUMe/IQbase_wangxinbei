"""Train and compare the shared density foundation on REAL C01, single GPU.

Control: ordinary continuation. Head control: matched extra readout exposure.
Density: prepare new parameters with old classifier fixed, then joint training.
Curriculum: same model/batches, training-only stability-based loss weights.
"""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from models.rf_model import RFModel
from utils.common import read_config_json,write_json,model_bundle,save_checkpoint,seed_all
from utils.config import resolve_config,training_options,make_scheduler
from utils.contrastive import load_with_training_aux
from utils.dataset import PreparedDataset,BalancedBatchSampler,collate,to_device
from utils.engine import autocast,supervised_loss
from utils.head_optimizer import active_heads,clear_inactive_head_gradients
from utils.tensorboard_log import TensorBoardLog
from training.explore_reception import evaluate


def new_parameter(name):return name.startswith(('backbone.time.density.','heads.modulation.density_projection.'))


def format_training_record(record):
    parts=' | '.join(f'{key}={value:.5f}' for key,value in record['parts'].items() if value!=0)
    rates=','.join(f'{rate:.3e}' for rate in record['lr'])
    return f"{record['stage']} | step={record['step']} | loss={record['loss']:.5f} | {parts} | grad(pre-clip)={record['gradient_norm']:.3f} | lr={rates}\n"


def format_validation(metric,estimator):
    lines=[f'\n全量验证 | 权重={estimator}', '任务 | 样本数 | Accuracy | Macro-F1 | Weighted-F1']
    for task,values in metric['classification'].items():
        lines.append(f"{task} | {values['support']} | {100*values['accuracy']:.3f}% | {100*values['macro_f1']:.3f}% | {100*values['weighted_f1']:.3f}%")
    lines.append('参数 | RMSE | MAE | P95绝对误差')
    for field,values in metric['parameters'].items():
        lines.append(field+' | '+' | '.join('n/a' if values[key] is None else f'{values[key]:.6g}' for key in ['rmse','mae','absolute_error_p95']))
    return '\n'.join(lines)+'\n'


def augment_modulation(iq,settings,seed):
    generator=torch.Generator(device=iq.device).manual_seed(seed)
    angle=(torch.rand(len(iq),device=iq.device,generator=generator)*2-1)*math.pi
    angle=angle*(torch.rand(len(iq),device=iq.device,generator=generator)<settings['phase_probability'])
    c,s=angle.cos()[:,None],angle.sin()[:,None]
    x=torch.stack([iq[...,0]*c-iq[...,1]*s,iq[...,0]*s+iq[...,1]*c],-1)
    shift=torch.randint(iq.shape[1],(len(iq),),device=iq.device,generator=generator)
    shift=shift*(torch.rand(len(iq),device=iq.device,generator=generator)<settings['circular_shift_probability'])
    index=(torch.arange(iq.shape[1],device=iq.device)[None]+shift[:,None])%iq.shape[1]
    augmented=x.gather(1,index[...,None].expand(-1,-1,2))
    probability=settings.get('apply_probability',1.)
    if not 0<=probability<=1:raise ValueError('augmentation.apply_probability must be in [0,1]')
    if probability<1:
        selected=torch.rand(len(iq),device=iq.device,generator=generator)<probability
        augmented=torch.where(selected[:,None,None],augmented,iq)
    return augmented


def augmentation_settings(config,arm):
    options=dict(config['augmentation'])
    if '_phase_only' in arm:options['circular_shift_probability']=0.
    if '_shift_only' in arm:options['phase_probability']=0.
    if arm.endswith('_native_mix'):options['apply_probability']=.5
    return options


def density_descriptor(model,iq,lengths):
    # Exact training fast path for the original first temporal convolution.
    # Its equivalence to the full forward descriptor is checked separately.
    with torch.no_grad():
        x,_,_,valid=model.backbone.preprocess(iq,lengths)
        amplitude=x.square().sum(-1,keepdim=True).clamp_min(1e-12).sqrt();pieces=[x,amplitude]
        if model.config['phase_features']:
            previous=F.pad(x[:,:-1],(0,0,1,0));real=(x*previous).sum(-1);imag=x[...,1]*previous[...,0]-x[...,0]*previous[...,1]
            reliability=(amplitude[...,0]*F.pad(amplitude[:,:-1,0],(1,0))).clamp_max(1)
            magnitude=(real.square()+imag.square()).sqrt().clamp_min(1e-6)
            pieces.append(torch.stack([real,imag],-1)/magnitude[...,None]*reliability[...,None])
        hidden=model.backbone.time[1](model.backbone.time[0]((torch.cat(pieces,-1)*valid[...,None]).transpose(1,2)))*valid[:,None]
    model.backbone.time.density(hidden,valid)
    return model.backbone.time.density.descriptor


def optimizer(model,c,preparation=False):
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        lr=c['optimizer']['readout_learning_rate'] if preparation else c['optimizer']['new_learning_rate' if new_parameter(name) else 'original_learning_rate']
        wd=0. if p.ndim==1 and new_parameter(name) else c['optimizer']['weight_decay']
        groups.setdefault((lr,wd),[]).append(p)
    return torch.optim.AdamW([dict(params=p,lr=lr,weight_decay=wd) for (lr,wd),p in groups.items()])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'code/configs/supervised/density_optimization.json5');p.add_argument('--smoke',action='store_true');p.add_argument('--arms')
    p.add_argument('--joint-steps',type=int,help='Override the joint budget; recorded in the resolved protocol')
    p.add_argument('--seed',type=int,help='Override the paired sampling seed for a replication')
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or torch.cuda.device_count()!=1 or int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('One process on physical GPU 0 only')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    c=read_config_json(args.config)
    if args.arms:c['training']['arms']=args.arms.split(',')
    if args.joint_steps is not None:
        if args.joint_steps<2:p.error('--joint-steps must be at least 2')
        c['training']['joint_steps']=args.joint_steps
    if args.seed is not None:c['training']['seed']=args.seed
    if args.smoke:c['training'].update(batch_size=32,readout_preparation_steps=6,joint_steps=6);c['evaluation']['batch_size']=32;c['logging']['interval']=1
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f');out=ROOT/c['run']['output']/stamp;out.mkdir(parents=True,exist_ok=False)
    source_path=ROOT/c['run']['source'];source=torch.load(source_path,map_location='cpu',weights_only=True,mmap=True)
    train=PreparedDataset(ROOT/'artifacts/data_full','train');val=PreparedDataset(ROOT/'artifacts/data_full','validation')
    if source['dataset_fingerprint']!=train.info['fingerprint']:raise ValueError('Different data split')
    meta=train.meta;iq=np.load(ROOT/'artifacts/data_full/iq.npy',mmap_mode='r');selection=dict(np.load(ROOT/'artifacts/c01_followup_validation/20261113_224434/selection.npz'))
    b=c['training']['batch_size'];prep=c['training']['readout_preparation_steps'];steps=c['training']['joint_steps'];seed=c['training']['seed']
    joint=np.asarray(list(BalancedBatchSampler(train,b,steps,seed,mode='balanced')),dtype=np.int64)
    rng=np.random.default_rng(seed+1);mod=np.flatnonzero(meta['modulation'][train.indices]>=0);buckets=[mod[meta['length'][train.indices[mod]]==length] for length in [512,1024]]
    probabilities=np.array([len(x) for x in buckets],float);probabilities/=probabilities.sum()
    warm=np.asarray([rng.choice(buckets[rng.choice(2,p=probabilities)],b,replace=True) for _ in range(prep)])
    np.save(out/'joint_stream.npy',joint);np.save(out/'readout_stream.npy',warm)
    weights=np.ones(len(meta['length']),np.float32);posterior=None;rank_all=np.ones(len(weights),np.float32)
    if any(x in c['training']['arms'] for x in ['density_curriculum','adaptive_smoothing','posterior_distill','phase_shift_posterior','phase_shift_first14_posterior']):
        rel=np.load(ROOT/c['run']['reliability']);rr=rel['rows'];assert (meta['split'][rr]==0).all() and np.array_equal(meta['modulation'][rr],rel['labels'])
        floor=c['curriculum']['minimum_relative_weight'];weights[rr]=(floor+(1-floor)*rel['rank'])/(floor+(1-floor)*.5)
        rank_all[rr]=rel['rank']
        if any(x in c['training']['arms'] for x in ['adaptive_smoothing','posterior_distill','phase_shift_posterior','phase_shift_first14_posterior']):
            posterior=np.zeros((len(weights),23),np.float32);posterior[rr]=rel['posterior']
        assert set(rr)==set(train.indices[mod])
    validation_rows=val.indices
    if args.smoke:
        validation_rows=np.unique(np.concatenate([val.indices[(meta['length'][val.indices]==n)&ok[val.indices]][:2] for n in [512,1024] for ok in [meta['modulation']>=0,meta['protocol']>=0,meta['target_valid'].any(1)]]))
    protocol=dict(config=c,source_checkpoint=str(source_path),dataset_fingerprint=train.info['fingerprint'],smoke=args.smoke,
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        joint_stream_sha256=hashlib.sha256(train.indices[joint].tobytes()).hexdigest(),readout_stream_sha256=hashlib.sha256(train.indices[warm].tobytes()).hexdigest(),
        competition='One shared time/frequency foundation, one modulation classifier; no teacher or ensemble at inference',
        control='No preparation; head_control has matched extra modulation exposure; density preparation updates only new parameters',
        curriculum='Within-class stability rank, all training samples have positive weight; no SNR assumption')
    write_json(out/'protocol.json',protocol);print('RUN',out,'ARMS',c['training']['arms'],'PREP',prep,'JOINT',steps,'BATCH',b,flush=True)
    base=RFModel(**model_bundle(source)).cuda();base.load_state_dict(source['model'],strict=True)
    evargs=argparse.Namespace(output=out,eval_batch_size=c['evaluation']['batch_size'])
    source_metric=evaluate(base,source,validation_rows,iq,meta,selection,evargs,'source_validation');del base;torch.cuda.empty_cache()
    results={}
    for arm in c['training']['arms']:
        if arm not in ['control','head_control','density','density_curriculum','adaptive_smoothing','posterior_distill','phase_shift','phase_shift_posterior','phase_shift_first14','phase_shift_first14_posterior','phase_shift_first14_phase_only','phase_shift_first14_shift_only','phase_shift_first14_native_mix','phase_shift_first14_phase_only_native_mix','phase_shift_first14_shift_only_native_mix']:raise ValueError(arm)
        augmentation=augmentation_settings(c,arm)
        has_preparation=arm in ['head_control','density','density_curriculum']
        directory=out/arm;directory.mkdir();evargs.output=directory;seed_all(seed)
        bundle=model_bundle(source);bundle['model_config']={**bundle['model_config'],'density_encoding':{**c['model']['density_encoding'],'enabled':arm.startswith('density')}}
        model=RFModel(**bundle);load_with_training_aux(model,source['model'],source['model_config']);model=model.cuda()
        report=model.parameter_report()
        if report['shared_ratio']<=.8 or report['final_parameters']>1_000_000_000:raise ValueError('Competition constraint')
        resolved=resolve_config(source['training_config']);resolved['model']['density_encoding']=model.config['density_encoding']
        resolved['run'].update(data=str(ROOT/'artifacts/data_full'),output=str(directory),warmstart_checkpoint=str(source_path),init_checkpoint='',codebook='')
        resolved['optimizer'].update(learning_rate=c['optimizer']['original_learning_rate'],weight_decay=c['optimizer']['weight_decay']);resolved['runtime'].update(gpus='0',processes=1,precision=c['training']['precision'],threads=4,seed=seed)
        resolved['data'].update(batch_size=b,steps_per_epoch=steps)
        resolved['data']['augmentation']['phase_rotation_probability']=0.
        resolved['data']['augmentation']['modulation_only'].update(phase_rotation_probability=0.,circular_shift_probability=0.,class_indices=[],apply_probability=1.)
        if arm.startswith('phase_shift'):
            resolved['data']['augmentation']['modulation_only'].update(phase_rotation_probability=augmentation['phase_probability'],circular_shift_probability=augmentation['circular_shift_probability'],apply_probability=augmentation.get('apply_probability',1.))
            if 'first14' in arm:resolved['data']['augmentation']['modulation_only']['class_indices']=list(range(14))
        resolved['data']['sampling'].update(mode='balanced',task_probabilities={'modulation':1.,'protocol':1.,'parameter':1.})
        resolved['training'].update(epochs=1,max_steps=steps,gradient_accumulation_steps=1,gradient_clip=c['training']['joint_gradient_clip'])
        resolved['scheduler'].update(name='cosine',warmup_steps=min(c['scheduler']['warmup_steps'],steps-1),warmup_ratio=0.,warmup_start_factor=.1,min_lr_ratio=c['scheduler']['minimum_ratio'])
        resolved['evaluation'].update(batch_size=c['evaluation']['batch_size'],precision='fp32',max_batches=0)
        options=training_options(resolved);write_json(directory/'resolved_config.json',resolved);write_json(directory/'parameter_report.json',report)
        print('OVERVIEW',arm,report,'readout',prep if has_preparation else 0,'joint',steps,flush=True)
        with (directory/'train.log').open('w',buffering=1) as log,(directory/'events.jsonl').open('w',buffering=1) as events,TensorBoardLog(directory,options={'enabled':c['logging']['tensorboard']},log_dir=directory/'tensorboard',gradient_clip=c['training']['joint_gradient_clip']) as tb:
            tb.metadata(resolved,report)
            log.write(f"训练总览 | 方法={arm} | 单卡GPU0 | batch={b} | 准备更新={prep if has_preparation else 0} | 联合更新={steps}\n")
            log.write(f"参数={report['final_parameters']:,} | 共享占比={100*report['shared_ratio']:.2f}% | 精度={c['training']['precision']} | 调度=warmup+cosine\n")
            for stage,count,stream in [('readout',prep if has_preparation else 0,warm),('joint',steps,joint)]:
                if not count:continue
                prepare=stage=='readout'
                for name,param in model.named_parameters():
                    param.requires_grad_(not prepare or (name.startswith('heads.modulation.') if arm=='head_control' else new_parameter(name) and '.shared_projection.' not in name))
                if arm.startswith('density'):
                    model.heads['modulation'].use_context=not prepare;model.backbone.time.density.inject_shared=not prepare
                opt=optimizer(model,c,prepare);sch,_=make_scheduler(opt,dict(name='cosine',warmup_steps=min(c['scheduler']['warmup_steps'],count-1),warmup_ratio=0.,warmup_start_factor=.1,min_lr_ratio=c['scheduler']['minimum_ratio']),count)
                if not prepare:ema={k:v.detach().clone() for k,v in model.state_dict().items()}
                bar=tqdm(range(count),desc=arm+' '+stage,mininterval=10)
                for step in bar:
                    seed_all(seed+step+(0 if prepare else 100000));cpu=collate([train[int(i)] for i in stream[step]]);active=active_heads(cpu,options);batch=to_device(cpu,'cuda:0')
                    if arm.startswith('phase_shift') and bool((cpu['modulation']>=0).any()):
                        if not bool((cpu['modulation']>=0).all()):raise ValueError('Augmentation needs a homogeneous modulation batch')
                        augmented=augment_modulation(batch['iq'],augmentation,seed+step+700000)
                        batch['iq']=torch.where((batch['modulation']<14)[:,None,None],augmented,batch['iq']) if 'first14' in arm else augmented
                    model.eval() if prepare else model.train();opt.zero_grad(set_to_none=True)
                    with autocast(torch.device('cuda:0'),c['training']['precision']):
                        if prepare and arm.startswith('density'):
                            desc=density_descriptor(model,batch['iq'],batch['lengths']);zero=torch.zeros(len(desc),model.heads['modulation'][0].in_features,device='cuda:0')
                            logits=model.heads['modulation'](zero,desc);loss=F.cross_entropy(logits.float(),batch['modulation']);parts={'modulation':float(loss.detach())}
                        else:
                            output=model(batch['iq'],batch['lengths']);logits=output['modulation']
                            loss,parts=supervised_loss(output,batch,source['target_scaler'],source['parameter_grids'],options,distributed=False)
                        if arm=='density_curriculum' and bool((cpu['modulation']>=0).any()):
                            strength=1. if prepare else 1-(1-c['curriculum']['joint_final_strength'])*step/max(1,count-1)
                            ww=torch.from_numpy(weights[np.asarray(cpu['rows'])]).cuda();ww=1+strength*(ww-1)
                            ce=F.cross_entropy(logits.float(),batch['modulation'],reduction='none',label_smoothing=options['label_smoothing'])
                            loss=loss+options['task_weights'][0]*((ce*ww).mean()-ce.mean())
                        if arm in ['adaptive_smoothing','posterior_distill','phase_shift_posterior','phase_shift_first14_posterior'] and bool((cpu['modulation']>=0).any()):
                            yy=batch['modulation'];target=torch.zeros_like(logits.float())
                            smooth_classes=14 if c['distillation']['first14_only'] else logits.shape[1]
                            if arm=='adaptive_smoothing':target[:,:smooth_classes]=1/smooth_classes
                            else:
                                target=torch.from_numpy(posterior[np.asarray(cpu['rows'])]).cuda()
                                target[:,smooth_classes:]=0;mass=target.sum(1,keepdim=True)
                                target=target/mass.clamp_min(1e-8)
                                bad=mass[:,0]<1e-8
                                target[bad,:smooth_classes]=1/smooth_classes
                            eps=torch.from_numpy(c['distillation']['maximum_smoothing']*(1-rank_all[np.asarray(cpu['rows'])])).cuda()
                            eps=eps*min(1.,(step+1)/c['distillation']['ramp_steps'])*(yy<smooth_classes)
                            hard=F.cross_entropy(logits.float(),yy,reduction='none')
                            soft=-(target*logits.float().log_softmax(-1)).sum(1)
                            loss=loss+options['task_weights'][0]*(eps*(soft-hard)).mean()
                            parts.update(soft_target_ce=float(soft.mean().detach()),smoothing_mean=float(eps.mean()))
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Loss')
                    loss.backward();clear_inactive_head_gradients(model,active,torch.device('cuda:0'))
                    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),c['training']['readout_gradient_clip' if prepare else 'joint_gradient_clip']))
                    if not math.isfinite(norm):raise FloatingPointError('Gradient')
                    opt.step();sch.step()
                    if not prepare:
                        with torch.no_grad():
                            for key,value in model.state_dict().items():
                                if value.is_floating_point():ema[key].lerp_(value.detach(),1-c['evaluation']['ema_decay'])
                                else:ema[key].copy_(value)
                    if (step+1)%c['logging']['interval']==0 or step==0:
                        record=dict(stage=stage,step=step+1,loss=float(loss.detach()),gradient_norm=norm,parts=parts,lr=[g['lr'] for g in opt.param_groups])
                        if arm.startswith('density'):record['density_projection_norm']=float(model.heads['modulation'].density_projection.weight.detach().norm())
                        events.write(json.dumps(record)+'\n');log.write(format_training_record(record));tb.scalar(stage+'/loss',record['loss'],step+1);tb.scalar(stage+'/gradient',norm,step+1)
                        for key,value in parts.items():tb.scalar(stage+'/parts/'+key,value,step+1)
                        for j,group in enumerate(opt.param_groups):tb.scalar(stage+'/learning_rate/group_'+str(j),group['lr'],step+1)
                        bar.set_postfix(loss=f'{record["loss"]:.4f}',grad=f'{norm:.2f}')
                del opt,sch
                if prepare:
                    if arm.startswith('density'):model.heads['modulation'].use_context=True;model.backbone.time.density.inject_shared=True
                    panel=evaluate(model,source,selection['panel_validation'] if not args.smoke else validation_rows,iq,meta,selection,evargs,'prepared_monitor')
                    print('PREPARED',arm,panel['classification']['modulation']['accuracy'],flush=True)
            states=dict(last={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},ema={k:v.cpu() for k,v in ema.items()});scores={}
            for estimator,state in states.items():
                model.load_state_dict(state,strict=True)
                metric=evaluate(model,source,validation_rows,iq,meta,selection,evargs,estimator+'_validation');scores[estimator]=metric
                for task,values in metric['classification'].items():
                    for key in ['accuracy','macro_f1','weighted_f1']:tb.scalar('estimators/'+estimator+'/'+task+'/'+key,values[key],steps)
                for field,values in metric['parameters'].items():
                    for key in ['rmse','mae','absolute_error_p95']:tb.scalar('estimators/'+estimator+'/parameters/'+field+'/'+key,values.get(key),steps)
                payload={**bundle,'model_config':model.config,'model':state,'training_config':resolved,'metrics':metric,'stage':'supervised',
                    'dataset_fingerprint':train.info['fingerprint'],'global_step':steps+(prep if has_preparation else 0),
                    'initialization_source_steps':source['global_step']+source.get('initialization_source_steps',0),'resume_allowed':False,
                    'optimization_protocol':protocol,'weight_estimator':estimator,'arm':arm}
                save_checkpoint(directory/(estimator+'.pt'),payload)
                log.write(format_validation(metric,estimator))
            results[arm]=dict(metrics=scores,parameters=report);write_json(out/'results.json',results)
            print('COMPLETE',arm,{k:v['classification']['modulation']['accuracy'] for k,v in scores.items()},flush=True)
        del model,states,state,ema;torch.cuda.empty_cache()
    write_json(out/'completion.json',dict(status='complete',smoke=args.smoke));print('FINISHED',out,flush=True)


if __name__=='__main__':main()
