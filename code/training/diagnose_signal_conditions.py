"""Supervised-only condition slices from observable IQ, without inventing fs/SNR."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from utils.common import read_json,write_json
from utils.metrics import classification_metrics


FEATURES=['log_rms','dc_fraction','lag1_abs','lag4_abs','lag16_abs','lag64_abs',
    'lag1_phase_cycles','correlation_half_lag','spectral_effective_bins','spectral_entropy',
    'spectral_flatness','high_frequency_power_fraction','spectral_peak_frequency',
    'bandwidth90_bins','spectral_peak_share','centered_power_fourth_moment']


@torch.inference_mode()
def describe(iq,lengths):
    x=torch.complex(iq[...,0],iq[...,1]);n=lengths.float()
    valid=torch.arange(x.shape[1])[None]<lengths[:,None]
    x=x*valid
    energy=x.abs().square().sum(1)/n
    mean=x.sum(1)/n
    centered=(x-mean[:,None])*valid
    variance=centered.abs().square().sum(1)/n
    centered=centered/variance.sqrt().clamp_min(1e-12)[:,None]
    transformed=torch.fft.fft(centered,n=2048)
    correlation=torch.fft.ifft(transformed*transformed.conj())[:,:129]
    correlation=correlation/(n[:,None]-torch.arange(129)[None])
    magnitude=correlation.abs()
    below=magnitude[:,1:]<.5
    half=torch.where(below.any(1),below.to(torch.int64).argmax(1)+1,129)
    power=torch.fft.fft(centered,n=1024).abs().square()
    probability=power/power.sum(1,keepdim=True).clamp_min(1e-20)
    peak=probability.argmax(1)
    frequency=torch.fft.fftfreq(1024)
    shifted=probability.gather(1,(torch.arange(1024)[None]+peak[:,None]-512)%1024)
    cumulative=shifted.cumsum(1).contiguous()
    thresholds=torch.tensor([.05,.95])[None].expand(len(x),-1).contiguous()
    boundaries=torch.searchsorted(cumulative,thresholds)
    features=[energy.sqrt().clamp_min(1e-12).log(),mean.abs().square()/energy.clamp_min(1e-20),
        *[magnitude[:,lag] for lag in [1,4,16,64]],
        correlation[:,1].angle()/(2*torch.pi),half.float(),
        1/probability.square().sum(1).clamp_min(1e-20),
        -(probability*probability.clamp_min(1e-20).log()).sum(1)/np.log(1024),
        power.clamp_min(1e-20).log().mean(1).exp()/power.mean(1).clamp_min(1e-20),
        probability[:,frequency.abs()>=.25].sum(1),frequency[peak],
        (boundaries[:,1]-boundaries[:,0]+1).float(),probability.max(1).values,
        centered.abs().pow(4).sum(1)/n]
    return torch.stack(features,-1)


def ks_distance(a,b):
    combined=np.sort(np.concatenate((a,b)));a=np.sort(a);b=np.sort(b)
    return float(np.max(np.abs(np.searchsorted(a,combined,side='right')/len(a)-np.searchsorted(b,combined,side='right')/len(b))))


def measured_groups(y,prediction,groups):
    result=[]
    for group in sorted(np.unique(groups)):
        keep=groups==group
        values=[]
        for label in range(14):
            selected=keep&(y==label)
            if selected.any():values.append(float(np.mean(prediction[selected]==y[selected])))
        result.append({'group':int(group),'support':int(keep.sum()),'accuracy':float(np.mean(prediction[keep]==y[keep])),
                       'macro_recall':float(np.mean(values))})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('artifacts/data_full'))
    p.add_argument('--reference',type=Path,default=Path('artifacts/diagnostics/20261111_110649_079619/frozen_large'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    with np.load(args.data/'metadata.npz',allow_pickle=False) as handle:meta={k:handle[k] for k in handle.files}
    iq=np.load(args.data/'iq.npy',mmap_mode='r');cache={}
    for split in ['train','validation']:
        rows=np.load(args.reference/f'{split}_rows.npy');labels=meta['modulation'][rows].astype(np.int64)
        prediction=np.load(args.reference/f'reference_{split}_predictions.npy')
        assert np.array_equal(labels,np.load(args.reference/f'{split}_labels.npy'))
        parts=[]
        for begin in range(0,len(rows),1024):
            selected=rows[begin:begin+1024]
            features=describe(torch.from_numpy(np.array(iq[selected])),torch.from_numpy(meta['length'][selected].copy()))
            if not bool(torch.isfinite(features).all()):raise FloatingPointError('Nonfinite observable features')
            parts.append(features.numpy())
        values=np.concatenate(parts);np.save(args.output/f'{split}_features.npy',values)
        np.save(args.output/f'{split}_rows.npy',rows)
        cache[split]={'rows':rows,'labels':labels,'predictions':prediction,'features':values}
        print(split,'records',len(rows),'accuracy',float(np.mean(prediction==labels)),flush=True)
    shifts,slices,edges={},{},{}
    for index,name in enumerate(FEATURES):
        quantiles={}
        for label in range(14):
            train=cache['train'];val=cache['validation'];a=train['features'][train['labels']==label,index];b=val['features'][val['labels']==label,index]
            quantiles[str(label)]=np.quantile(a,[.25,.5,.75]).tolist()
            shifts.setdefault(name,[]).append({'label':label,'ks_distance':ks_distance(a,b),
                'train_quantiles':np.quantile(a,[.01,.25,.5,.75,.99]).tolist(),
                'validation_quantiles':np.quantile(b,[.01,.25,.5,.75,.99]).tolist()})
        edges[name]=quantiles;slices[name]={}
        for split in cache:
            d=cache[split];keep=d['labels']<14;y=d['labels'][keep];v=d['features'][keep,index];prediction=d['predictions'][keep]
            groups=np.array([np.searchsorted(quantiles[str(label)],value,side='right') for label,value in zip(y,v)])
            slices[name][split]=measured_groups(y,prediction,groups)
    conditions=cache['validation'];keep=conditions['labels']<14;y=conditions['labels'][keep];pred=conditions['predictions'][keep]
    groups={}
    for index,name in enumerate(FEATURES):
        groups[name]=np.array([np.searchsorted(edges[name][str(label)],value,side='right') for label,value in zip(y,conditions['features'][keep,index])])
    cross={}
    for name in ['log_rms','lag1_phase_cycles','spectral_effective_bins','spectral_peak_share']:
        cross[name]=measured_groups(y,pred,groups['lag1_abs']*4+groups[name])
    cm=np.bincount(y*23+pred,minlength=23*23).reshape(23,23)
    names=read_json(args.data/'dataset.json')['schema']['classification_labels']['modulation']
    pairs=[]
    for a in range(14):
        for b in range(a+1,14):
            pairs.append({'a':names[a],'b':names[b],'a_to_b':int(cm[a,b]),'b_to_a':int(cm[b,a]),
                'symmetric_error_rate':float((cm[a,b]+cm[b,a])/(cm[a].sum()+cm[b].sum()))})
    pairs.sort(key=lambda v:v['symmetric_error_rate'],reverse=True)
    high=groups['lag1_abs']==3;high_cm=np.bincount(y[high]*23+pred[high],minlength=529).reshape(23,23)
    protocol={'reference':str(args.reference.resolve()),'features':FEATURES,'class_conditional_thresholds':'training only',
        'uses_true_class_for_analysis_slices':True,'not_deployable_gating':True,
        'not_snr_or_true_symbol_count':True,'correlation_half_lag_censored_at':129,
        'frequency_units':'cycles/sample; lag1 phase is not verified CFO',
        'dataset_fingerprint':read_json(args.data/'dataset.json')['fingerprint']}
    write_json(args.output/'protocol.json',protocol)
    write_json(args.output/'summary.json',dict(protocol=protocol,distribution_shifts=shifts,condition_slices=slices,
        train_thresholds=edges,cross_slices=cross,top_confusion_pairs=pairs,
        hard14_metrics=classification_metrics(cm,names),high_lag1_hard14_metrics=classification_metrics(high_cm,names)))
    for name in ['lag1_abs','spectral_effective_bins','log_rms','lag1_phase_cycles','high_frequency_power_fraction']:
        print(name,slices[name]['validation'],flush=True)


if __name__=='__main__':main()
