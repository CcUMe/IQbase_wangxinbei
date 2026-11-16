"""Paired synthetic classification: raw observation vs known-condition receiver.

This is a mechanism control, NOT an estimate of C01 SNR or its achievable
accuracy. Nuisance truth never includes the modulation label. The oracle
likelihood is an optimistic known-model reference, not an information bound.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import logsumexp
import torch
from torch.nn import functional as F
from tqdm import tqdm
from models.reception import rrc_filter
from models.rf_model import RFModel
from utils.common import model_bundle,write_json,save_checkpoint
from utils.metrics import classification_metrics
from utils.tensorboard_log import TensorBoardLog

LABELS=['BPSK','QPSK','8PSK','16QAM','64QAM','256QAM']
LEVELS=[-10.,0.,10.,20.,30.]


def constellations():
    values=[np.asarray([-1.,1.],np.complex64),np.exp(1j*(np.pi/4+np.arange(4)*np.pi/2)),np.exp(2j*np.pi*np.arange(8)/8)]
    for side in [4,8,16]:
        axis=np.arange(side)*2-(side-1)
        values.append((axis[:,None]+1j*axis[None,:]).ravel()/np.sqrt(2*(side*side-1)/3))
    return [x.astype(np.complex64) for x in values]


def scores(samples,variance,valid):
    result=[]
    for alphabet in constellations():
        distance=np.abs(samples[:,:,None]-alphabet[None,None])**2
        logp=logsumexp(-distance/variance[:,None,None],axis=-1)-math.log(len(alphabet))
        result.append((logp*valid).sum(1))
    return np.stack(result,1)


def classification(logits,y):
    n=len(LABELS);pred=np.asarray(logits).argmax(1)
    return classification_metrics(np.bincount(y*n+pred,minlength=n*n).reshape(n,n),LABELS)


def prepare(output):
    if torch.cuda.is_available():raise ValueError('Preparation is CPU only: hide CUDA devices')
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    h=rrc_filter(8,span=8,beta=.35).numpy();alphabets=constellations()
    protocol=dict(seed_train=2026111411,seed_test=2026111412,classes=LABELS,
        waveform_length=1024,samples_per_symbol=8,rrc_span=8,rrc_rolloff=.35,
        nominal_waveform_snr_db=LEVELS,noise='Proper complex white Gaussian; variance=(1/8)*10**(-SNR/10), independent of class',
        frequency_cycles_per_sample=[-.06,.06],phase='uniform [-pi,pi]',gain='log-uniform [.3,3]',
        timing='integer offset uniform 0..7 samples',multipath=False,
        oracle_metadata='Generator true frequency, phase, scalar gain, period and timing; never class or symbol values',
        diagnostic='Same underlying test observations; oracle matched-filter symbols and 10% clock-error control',
        limitation='Synthetic single-pulse/AWGN setting only; not C01 conditions or performance ceiling')
    write_json(output/'protocol.json',protocol)
    receiver={}
    for split,per_level,seed in [('train',512,2026111411),('test',256,2026111412)]:
        rng=np.random.default_rng(seed);y=np.repeat(np.arange(6),len(LEVELS)*per_level)
        snr=np.tile(np.repeat(LEVELS,per_level),6).astype(np.float32);order=rng.permutation(len(y));y=y[order];snr=snr[order]
        raw=np.lib.format.open_memmap(output/(split+'_raw.npy'),mode='w+',dtype=np.float32,shape=(len(y),1024,2))
        oracle=np.lib.format.open_memmap(output/(split+'_oracle.npy'),mode='w+',dtype=np.float32,shape=(len(y),1024,2))
        oracle_logits=[];wrong_logits=[];equal_oracle=[];equal_wrong=[];evm=[]
        for start in tqdm(range(0,len(y),64),desc='Generate '+split,mininterval=10):
            stop=min(start+64,len(y));labels=y[start:stop];b=len(labels)
            symbols=np.empty((b,144),np.complex64)
            for i,label in enumerate(labels):symbols[i]=rng.choice(alphabets[label],144)
            tau=rng.integers(0,8,b);impulse=np.zeros((b,1152),np.complex64)
            rows=np.arange(b)[:,None];locations=tau[:,None]+np.arange(144)[None]*8
            impulse[rows,locations]=symbols
            clean=fftconvolve(impulse,h[None],mode='same',axes=-1)[:,64:1088].astype(np.complex64)
            variance=(.125*np.power(10.,-snr[start:stop]/10)).astype(np.float32)
            noise=(rng.standard_normal(clean.shape)+1j*rng.standard_normal(clean.shape)).astype(np.complex64)*np.sqrt(variance[:,None]/2)
            x=clean+noise;carrier=rng.uniform(-.06,.06,b);phase=rng.uniform(-np.pi,np.pi,b);gain=np.exp(rng.uniform(np.log(.3),np.log(3),b))
            rotation=np.exp(1j*(2*np.pi*carrier[:,None]*np.arange(1024)[None]+phase[:,None]))
            observed=(x*rotation*gain[:,None]).astype(np.complex64)
            # Receiver consumes the observation and class-independent nuisance
            # truth. Transmitted symbol values are used ONLY for EVM checks.
            corrected=(observed/(rotation*gain[:,None])).astype(np.complex64)
            filtered=fftconvolve(corrected,h[None],mode='same',axes=-1).astype(np.complex64)
            centres=tau[:,None]+64+8*np.arange(112)[None]
            samples=filtered[rows,centres]
            wrong=tau[:,None]+64+8*1.10*np.arange(112)[None]
            left=np.floor(wrong).astype(int);fraction=wrong-left
            valid=(wrong<960)&(wrong>=64)
            bad=(filtered[rows,np.clip(left,0,1023)]*(1-fraction)+filtered[rows,np.clip(left+1,0,1023)]*fraction).astype(np.complex64)
            repeated=np.repeat(samples,8,axis=1)
            raw[start:stop]=np.stack([observed.real,observed.imag],-1)
            oracle[start:stop]=0;oracle[start:stop,:896]=np.stack([repeated.real,repeated.imag],-1)
            if split=='test':
                oracle_logits.append(scores(samples,variance,np.ones_like(samples.real,dtype=bool)))
                wrong_logits.append(scores(bad,variance,valid))
                # Equal-count control excludes the possible confound that a
                # wrong clock has fewer valid samples near the record end.
                if not valid[:,:100].all():raise ValueError('Equal-count clock control left the valid window')
                equal_mask=np.ones((b,100),dtype=bool)
                equal_oracle.append(scores(samples[:,:100],variance,equal_mask))
                equal_wrong.append(scores(bad[:,:100],variance,equal_mask))
                evm.extend(np.mean(np.abs(samples-symbols[:,16:128])**2,axis=1).tolist())
        raw.flush();oracle.flush();np.savez(output/(split+'_metadata.npz'),labels=y,snr=snr)
        if split=='test':
            a=np.concatenate(oracle_logits);b=np.concatenate(wrong_logits)
            receiver=dict(oracle=classification(a,y),clock_error_10pct=classification(b,y),by_snr={})
            for level in LEVELS:
                keep=snr==level;receiver['by_snr'][str(level)]=dict(oracle=classification(a[keep],y[keep]),clock_error_10pct=classification(b[keep],y[keep]),
                    mean_symbol_mse=float(np.mean(np.asarray(evm)[keep])))
            np.savez_compressed(output/'oracle_reference_predictions.npz',labels=y,snr=snr,oracle_logits=a,clock_error_logits=b)
            write_json(output/'receiver_reference.json',receiver)
            eq_a=np.concatenate(equal_oracle);eq_b=np.concatenate(equal_wrong)
            equal=dict(samples_per_record=100,overall_oracle=classification(eq_a,y),overall_clock_error=classification(eq_b,y),
                by_snr={str(level):dict(oracle=classification(eq_a[snr==level],y[snr==level]),clock_error=classification(eq_b[snr==level],y[snr==level])) for level in LEVELS})
            write_json(output/'receiver_equal_count_reference.json',equal)
            np.savez_compressed(output/'equal_count_predictions.npz',labels=y,snr=snr,oracle_logits=eq_a,clock_error_logits=eq_b)
    print('PREPARED',output,{s:{k:v['accuracy'] for k,v in z.items() if isinstance(v,dict) and 'accuracy' in v} for s,z in receiver['by_snr'].items()},flush=True)


@torch.inference_mode()
def evaluate(model,data,labels,snr,length):
    model.eval();chunks=[]
    for start in range(0,len(labels),128):
        x=torch.from_numpy(np.array(data[start:start+128])).cuda()
        chunks.append(model(x,torch.full((len(x),),length,device='cuda:0'))['modulation'].float().cpu().numpy())
    z=np.concatenate(chunks)
    if not np.isfinite(z).all():raise FloatingPointError('Synthetic prediction')
    result=dict(overall=classification(z,labels),by_snr={str(level):classification(z[snr==level],labels[snr==level]) for level in LEVELS})
    return result,z


def train(output,steps,arms='raw,oracle'):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or torch.cuda.device_count()!=1:raise ValueError('GPU 0, one process only')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=torch.load(ROOT/'artifacts/reception_exploration/20261114_104034_511636/source.pt',map_location='cpu',weights_only=True,mmap=True)
    bundle=model_bundle(source);bundle['schema']=json.loads(json.dumps(bundle['schema']));bundle['schema']['classification_labels']['modulation']=LABELS
    torch.manual_seed(2026111413);initial=RFModel(**bundle).state_dict()
    digest=hashlib.sha256()
    for k,v in initial.items():digest.update(k.encode());digest.update(v.numpy().tobytes())
    rng=np.random.default_rng(2026111414);meta=np.load(output/'train_metadata.npz');testmeta=np.load(output/'test_metadata.npz')
    n=len(meta['labels']);stream=np.concatenate([rng.permutation(n) for _ in range(math.ceil(steps*128/n))])[:steps*128].reshape(steps,128)
    run=output/('training_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir()
    np.save(run/'training_stream.npy',stream)
    write_json(run/'protocol.json',dict(steps=steps,batch_size=128,initialization='random; original historical RFModel architecture, six-class modulation head',
        initial_sha256=digest.hexdigest(),data_stream_sha256=hashlib.sha256(stream.tobytes()).hexdigest(),
        original_model_config=bundle['model_config'],lr=3e-4,weight_decay=.01,warmup=60,cosine_min_ratio=.1,clip=5.,
        source_weights_loaded=False,test_selection='Fixed final step only; no test-based checkpoint selection',
        arms=arms,high_only_policy='oracle_high_only masks loss at known synthetic SNR<20; same batches/high-sample exposure, sum CE divided by full batch size'))
    results={}
    for arm in arms.split(','):
        if arm not in ['raw','oracle','oracle_high_only']:raise ValueError('Unknown synthetic arm')
        length=1024 if arm=='raw' else 896
        torch.manual_seed(2026111413);model=RFModel(**bundle);model.load_state_dict(initial,strict=True);model=model.cuda().train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda t:(.1+.9*t/60) if t<60 else .1+.9*(1+math.cos(math.pi*min(1.,(t-60)/(steps-60))))/2)
        data_arm='raw' if arm=='raw' else 'oracle'
        data=np.load(output/('train_'+data_arm+'.npy'),mmap_mode='r');history=[]
        with (run/(arm+'_train.log')).open('w',buffering=1) as log,TensorBoardLog(run,log_dir=run/'tensorboard'/arm,gradient_clip=5.) as tb:
            print('TRAIN',arm,steps,'steps',model.parameter_report(),flush=True)
            for step in tqdm(range(steps),desc='synthetic '+arm,mininterval=10):
                torch.manual_seed(2026111415+step);indices=stream[step]
                x=torch.from_numpy(np.array(data[indices])).cuda();y=torch.from_numpy(meta['labels'][indices]).cuda()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    z=model(x,torch.full((len(x),),length,device='cuda:0'))['modulation']
                    keep=torch.from_numpy(meta['snr'][indices]>=20).cuda() if arm=='oracle_high_only' else torch.ones(len(x),dtype=torch.bool,device='cuda:0')
                    if not bool(keep.any()):raise ValueError('Empty selected synthetic condition')
                    loss=F.cross_entropy(z[keep].float(),y[keep],reduction='sum')/len(x)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Synthetic train')
                loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.))
                if not math.isfinite(norm):raise FloatingPointError('Synthetic gradient')
                optimizer.step();schedule.step()
                if (step+1)%50==0:
                    row=dict(step=step+1,loss=float(loss.detach()),accuracy=float((z[keep].argmax(1)==y[keep]).float().mean()),selected_records=int(keep.sum()),gradient_norm=norm,lr=optimizer.param_groups[0]['lr'])
                    history.append(row);log.write(json.dumps(row)+'\n');tb.scalar('train/ce',row['loss'],step+1);tb.scalar('train/accuracy',row['accuracy'],step+1)
            metrics,z=evaluate(model,np.load(output/('test_'+data_arm+'.npy'),mmap_mode='r'),testmeta['labels'],testmeta['snr'],length)
            np.savez_compressed(run/(arm+'_test_predictions.npz'),logits=z,labels=testmeta['labels'],snr=testmeta['snr'])
            save_checkpoint(run/(arm+'_last.pt'),dict(model={k:v.cpu() for k,v in model.state_dict().items()},**bundle,metrics=metrics,research_kind='synthetic_causal_diagnostic',resume_allowed=False))
            write_json(run/(arm+'_history.json'),history);write_json(run/(arm+'_metrics.json'),metrics)
            results[arm]=metrics;write_json(run/'results.json',results)
            print('FINISHED',arm,'test',metrics['overall']['accuracy'],{k:v['accuracy'] for k,v in metrics['by_snr'].items()},flush=True)
        del model,optimizer,schedule;torch.cuda.empty_cache()
    write_json(output/('training_run_high_only.json' if arms=='oracle_high_only' else 'training_run.json'),dict(directory=str(run),status='complete'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['prepare','train'],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--steps',type=int,default=1800)
    parser.add_argument('--arms',default='raw,oracle')
    args=parser.parse_args();prepare(args.output) if args.phase=='prepare' else train(args.output,args.steps,args.arms)
