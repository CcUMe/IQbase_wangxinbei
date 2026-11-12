"""Secondary diagnostic: fixed IQ statistics / power spectrum with shallow readouts."""
import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from training.diagnose_readout import feature_metrics
from training.hypothesis_probe import prepare_data,group_metrics
from utils.common import seed_all,read_json,write_json,save_checkpoint
from utils.config import make_scheduler
from utils.readable_log import ReadableRunLog
from utils.training_display import EpochProgress


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('artifacts/data_full'))
    p.add_argument('--statistics-cache',type=Path,default=Path('artifacts/representation_optimization/20261111_122154_915261/physical_small'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    dataset={s:prepare_data(args.data,s) for s in ['train','validation']}
    source=read_json(args.statistics_cache/'protocol.json')
    assert source['dataset_fingerprint']==dataset['train'].info['fingerprint'] and source['physical_statistics']
    labels={s:torch.from_numpy(v.meta['modulation'][v.indices].astype(np.int64)) for s,v in dataset.items()}
    names=dataset['train'].info['schema']['classification_labels']['modulation']
    inputs={'statistics':{},'spectrum':{}}
    for split,data in dataset.items():
        assert np.array_equal(np.load(args.statistics_cache/f'{split}_rows.npy'),data.indices)
        cached=np.load(args.statistics_cache/f'{split}_features.npy',mmap_mode='r')
        inputs['statistics'][split]=torch.from_numpy(np.array(cached[:,-32:]))
        iq=np.load(args.data/'iq.npy',mmap_mode='r');pieces=[]
        for begin in range(0,len(data.indices),1024):
            rows=data.indices[begin:begin+1024];x=torch.from_numpy(np.array(iq[rows]));lengths=torch.from_numpy(data.meta['length'][rows].copy())
            scale=(x.square().sum((1,2))/lengths).sqrt().clamp_min(1e-6)
            z=torch.complex(x[...,0]/scale[:,None],x[...,1]/scale[:,None])
            power=torch.fft.fft(z,n=1024).abs().square()/lengths[:,None]
            pieces.append(torch.log1p(torch.roll(power,512,-1)))
        inputs['spectrum'][split]=torch.cat(pieces)
    protocol={'device':'cpu','epochs':20,'batch_size':1024,'seeds':[2026,2027],
        'learning_rate':1e-3,'weight_decay':.01,'normalization':'train-only mean/std',
        'selection':'fixed final epoch','features':['32 fixed IQ statistics','1024 full-window log power spectrum, RMS-normalized IQ'],
        'heads':['linear','MLP hidden128 GELU'],'dataset_fingerprint':dataset['train'].info['fingerprint'],
        'statistics_source':str(args.statistics_cache.resolve()),
        'scope':'Secondary separability reference; no encoder training, not a matched architecture ablation.'}
    write_json(args.output/'protocol.json',protocol)
    results=[]
    with ReadableRunLog(args.output):
        print('固定特征参照',protocol,flush=True)
        for feature,features in inputs.items():
            x,v=features['train'],features['validation'];mean=x.mean(0);std=x.std(0,unbiased=False).clamp_min(1e-4)
            x,v=(x-mean)/std,(v-mean)/std
            for kind in ['linear','mlp']:
                for seed in [2026,2027]:
                    seed_all(seed)
                    head=(torch.nn.Linear(x.shape[1],len(names)) if kind=='linear' else torch.nn.Sequential(
                        torch.nn.Linear(x.shape[1],128),torch.nn.GELU(),torch.nn.Linear(128,len(names))))
                    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=.01)
                    scheduler,_=make_scheduler(optimizer,dict(name='cosine',warmup_steps=0,warmup_ratio=0.,
                        warmup_start_factor=0.,min_lr_ratio=.05),math.ceil(len(x)/1024)*20)
                    for epoch in range(20):
                        order=torch.randperm(len(x),generator=torch.Generator().manual_seed(seed+epoch*100003))
                        progress=EpochProgress(epoch,20,math.ceil(len(x)/1024),min_interval=1.)
                        with progress:
                            for begin in range(0,len(order),1024):
                                index=order[begin:begin+1024];optimizer.zero_grad(set_to_none=True)
                                loss=F.cross_entropy(head(x[index]),labels['train'][index]);loss.backward()
                                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
                                rate=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
                                progress.observe(float(loss.detach()),{'modulation':float(loss.detach())},{'modulation'})
                                progress.advance({'head':rate},float(norm))
                    train,_=feature_metrics(head,x,labels['train'],names)
                    validation,prediction=feature_metrics(head,v,labels['validation'],names);validation['hard14']=group_metrics(validation)
                    result={'features':feature,'head':kind,'seed':seed,'train':train,'validation':validation}
                    results.append(result);write_json(args.output/'summary.json',{'protocol':protocol,'results':results})
                    np.save(args.output/f'{feature}_{kind}_{seed}_predictions.npy',prediction)
                    save_checkpoint(args.output/f'{feature}_{kind}_{seed}.pt',dict(head=head.state_dict(),mean=mean,std=std))
                    print(feature,kind,seed,'Accuracy',validation['accuracy'],'hard14',validation['hard14']['accuracy'],flush=True)


if __name__=='__main__':main()
