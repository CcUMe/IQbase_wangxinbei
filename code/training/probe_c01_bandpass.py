"""CPU-only train-fitted probe of long, carrier-centred narrowband views.

No true modulation, fs, SNR or parameter value selects a filter. The observed
lag-1 correlation phase supplies a frequency-centre proxy, not true CFO. All
three fixed sample-normalized passbands are presented to the same classifier.
"""
import argparse
import hashlib
from pathlib import Path
import os
import sys
import time

import joblib
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits
from tqdm import tqdm

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from utils.common import read_json,write_json
from utils.signal_statistics import signal_statistics
from training.probe_c01_relations import group_metrics,analyze_predictions

CUTOFFS=[.01,.025,.05]


def bandpass_view(iq,cutoff,kernel=129):
    """Demodulate, windowed-sinc FIR, remodulate; original carrier retained.

    Native-length input only. Edge denominator preserves a centred constant
    phasor. A per-record positive gain restores observed input energy; neither
    step recovers unobserved signal or claims to estimate clean SNR.
    """
    iq=iq.float();z=torch.complex(iq[...,0],iq[...,1])
    omega=torch.angle((z[:,1:]*z[:,:-1].conj()).sum(1))
    angle=torch.arange(len(z[0]),device=z.device)[None]*omega[:,None]
    phasor=torch.polar(torch.ones_like(angle),angle)
    centred=z*phasor.conj()
    n=torch.arange(kernel,device=iq.device).float()-(kernel-1)/2
    h=2*cutoff*torch.sinc(2*cutoff*n)*torch.hann_window(kernel,periodic=False,device=iq.device)
    h=h/h.sum();weight=h[None,None]
    values=torch.stack([centred.real,centred.imag],1)
    denominator=F.conv1d(torch.ones_like(values[:1,:1]),weight,padding=kernel//2).clamp_min(.1)
    smoothed=F.conv1d(values,weight.expand(2,1,-1),padding=kernel//2,groups=2)/denominator
    result=torch.complex(smoothed[:,0],smoothed[:,1])*phasor
    energy=z.abs().square().sum(1);after=result.abs().square().sum(1)
    result*= (energy/after.clamp_min(1e-20)).sqrt()[:,None]
    return torch.stack([result.real,result.imag],-1)


def self_test():
    torch.manual_seed(20261114);x=torch.randn(3,512,2)
    z=torch.complex(x[...,0],x[...,1])*torch.polar(torch.tensor(1.),torch.tensor(.713))
    rotated=torch.stack([z.real,z.imag],-1)
    for cutoff in CUTOFFS:
        y=bandpass_view(x,cutoff)
        torch.testing.assert_close(y.square().sum((1,2)),x.square().sum((1,2)),rtol=2e-6,atol=1e-5)
        torch.testing.assert_close(bandpass_view(x*4,cutoff),y*4,rtol=1e-4,atol=1e-4)
        expected=torch.complex(y[...,0],y[...,1])*torch.polar(torch.tensor(1.),torch.tensor(.713))
        actual=bandpass_view(rotated,cutoff)
        torch.testing.assert_close(torch.complex(actual[...,0],actual[...,1]),expected,rtol=2e-4,atol=2e-4)
    t=torch.arange(512);z=torch.exp(2j*torch.pi*.073*t)
    y=bandpass_view(torch.stack([z.real,z.imag],-1)[None],.025)
    torch.testing.assert_close(torch.complex(y[0,:,0],y[0,:,1]),z,rtol=2e-5,atol=2e-5)
    print('SELF TEST PASSED: native shape, gain/phase equivariance, energy, centred tone/edge response.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['all','extract','probe'],default='all')
    p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Run CPU-only with CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(4)
    if args.self_test:self_test();return
    args.output.mkdir(parents=True,exist_ok=True)
    original=ROOT/'artifacts/c01_relation_probe/20261113_230504_797513'
    args.reference=ROOT/'artifacts/c01_followup_validation/20261113_224434'
    meta=dict(np.load(ROOT/'artifacts/data_full/metadata.npz',allow_pickle=False))
    rows=np.load(original/'rows.npy');splits=dict(np.load(original/'probe_split.npz',allow_pickle=False))
    y=meta['modulation'][rows].astype(np.int64)
    old=np.load(original/'existing32.npy',mmap_mode='r')
    settings=dict(n_estimators=128,min_samples_leaf=8,max_features='sqrt',n_jobs=4,random_state=20261113)
    protocol=dict(cpu_only=True,seed=20261113,cutoffs_cycles_per_sample=CUTOFFS,kernel=129,
        feature_count_per_view=32,filter_centre='lag-1 complex correlation phase, not true CFO',
        original_feature_path=str(original),fit_rows=len(splits['fit']),internal_rows=len(splits['internal']),
        validation_rows=len(splits['validation']),classifier=settings,
        selection='highest internal first14 accuracy; full refit if >=1.5pp over matched existing32 probe',
        caveat='Exploratory reuse of internal/validation data from earlier probes; no independent blind test; standalone diagnostics, not an ensemble',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(args.output/'protocol.json',protocol)
    if args.stage in ['all','extract'] and not (args.output/'features_complete.json').exists():
        raw=np.load(ROOT/'artifacts/data_full/iq.npy',mmap_mode='r')
        features=np.lib.format.open_memmap(args.output/'filtered.npy',mode='w+',dtype='float32',shape=(len(rows),96))
        for length in [512,1024]:
            indices=np.flatnonzero(meta['length'][rows]==length)
            for start in tqdm(range(0,len(indices),256),desc=f'bandpass/native{length}',mininterval=5):
                idx=indices[start:start+256];x=torch.from_numpy(np.array(raw[rows[idx],:length]))
                # Normalize before filtering for consistent numerical conditioning.
                x=x/x.square().sum((1,2)).div(length).sqrt()[:,None,None]
                with torch.inference_mode():
                    values=[signal_statistics(bandpass_view(x,c),torch.full((len(x),),length)).numpy() for c in CUTOFFS]
                features[idx]=np.concatenate(values,axis=1)
        features.flush();write_json(args.output/'features_complete.json',{'rows':len(rows),'finite':bool(np.isfinite(features).all())})
    if args.stage=='extract':return
    filtered=np.load(args.output/'filtered.npy',mmap_mode='r')
    sets={f'cutoff_{c}':filtered[:,j*32:(j+1)*32] for j,c in enumerate(CUTOFFS)}
    sets['original_plus_filtered']=np.concatenate([old,filtered],1)
    results=read_json(args.output/'results.json') if (args.output/'results.json').exists() else {}
    baseline=read_json(original/'results.json')['existing32']['internal']['first14']['accuracy']
    with threadpool_limits(limits=4):
        for name,x in sets.items():
            if name in results:continue
            started=time.monotonic();model=ExtraTreesClassifier(**settings)
            model.fit(x[splits['fit']],y[splits['fit']]);scores={}
            for split in ['fit','internal','validation']:
                ids=splits[split];probability=model.predict_proba(x[ids]).astype(np.float32)
                scores[split]=group_metrics(probability,y[ids],splits['rms_group'][ids])
                if split=='validation':np.savez_compressed(args.output/f'{name}_validation.npz',rows=rows[ids],probability=probability)
            results[name]={'metrics':scores,'elapsed_seconds':time.monotonic()-started};write_json(args.output/'results.json',results)
            print('DONE',name,{s:round(v['first14']['accuracy'],5) for s,v in scores.items()},flush=True)
        winner=max(sets,key=lambda name:results[name]['metrics']['internal']['first14']['accuracy'])
        gain=results[winner]['metrics']['internal']['first14']['accuracy']-baseline
        write_json(args.output/'selection.json',{'winner':winner,'gain_over_original32':gain,'full_refit_gate':gain>=.015})
        if gain>=.015 and 'full_train_selected' not in results:
            x=sets[winner];model=ExtraTreesClassifier(**settings);model.fit(x[splits['all_train']],y[splits['all_train']])
            ids=splits['validation'];probability=model.predict_proba(x[ids]).astype(np.float32)
            scores=group_metrics(probability,y[ids],splits['rms_group'][ids])
            overlap=analyze_predictions(args,args.output,'full_train_selected',probability,rows[ids],meta,'validation')
            results['full_train_selected']={'source_features':winner,'validation':scores,'error_overlap':overlap}
            np.savez_compressed(args.output/'full_train_selected_validation.npz',rows=rows[ids],probability=probability)
            joblib.dump(model,args.output/'full_train_selected.joblib',compress=3)
            write_json(args.output/'results.json',results)
            print('FULL TRAIN',winner,{g:round(m['accuracy'],5) for g,m in scores.items()},flush=True)
    print('FINISHED',args.output,flush=True)


if __name__=='__main__':main()
