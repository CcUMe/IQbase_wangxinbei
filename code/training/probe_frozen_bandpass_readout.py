"""One frozen historical best foundation, three matched lightweight readouts.

This diagnostic is run serially on GPU 0 after the existing queue finishes.
Readout budget is fixed in advance; validation is evaluated only at the end.
"""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from training.explore_phase_training import checkpoint_model
from training.probe_c01_relations import group_metrics
from utils.common import read_json,write_json,seed_all

SEED=20261114


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=10)
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or torch.cuda.device_count()!=1:raise ValueError('Exactly physical GPU 0 is required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True);device=torch.device('cuda:0')
    source=ROOT/'logs/finetune_v2/20261112_145913_822471/best.pt'
    data=ROOT/'artifacts/data_full';meta=dict(np.load(data/'metadata.npz',allow_pickle=False))
    original=ROOT/'artifacts/c01_relation_probe/20261113_230504_797513'
    rows=np.load(original/'rows.npy');labels=meta['modulation'][rows].astype(np.int64)
    train=np.flatnonzero(meta['split'][rows]==0);validation=np.flatnonzero(meta['split'][rows]==1)
    groups=np.load(ROOT/'artifacts/c01_followup_validation/20261113_224434/selection.npz',allow_pickle=False)['condition_groups'][rows,0]
    c,model=checkpoint_model(source,device);model.eval()
    if c['dataset_fingerprint']!=read_json(data/'dataset.json')['fingerprint']:raise ValueError('Dataset fingerprint mismatch')
    if model.multiresolution or c['model_config']['classification_pooling']!='mean':raise ValueError('This probe expects the fixed historical mean-pooling model')
    head_source={k:v.detach().cpu().clone() for k,v in model.heads['modulation'].state_dict().items()}
    width=head_source['0.weight'].shape[1];hidden=head_source['0.weight'].shape[0]
    protocol=dict(source_checkpoint=str(source),dataset_fingerprint=c['dataset_fingerprint'],
        encoder_frozen=True,other_task_parameters_frozen=True,epochs=args.epochs,batch_size=512,
        arms=['original_readout','raw32_readout','raw32_filtered96_readout'],
        learning_rate=1e-4,weight_decay=.01,warmup_epochs=1,cosine_min_lr_ratio=.05,
        source_width=width,hidden_width=hidden,extra_column_initialization=0,
        seed=SEED,selection='fixed final epoch for every arm, no validation-based early stopping',
        scaler_fit='all training rows only',validation_rows=len(validation),train_rows=len(train),
        caveat='Research readout bundle, not yet a standalone RFModel export; if useful, integrate transforms and verify full inference before promotion',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(args.output/'protocol.json',protocol)
    if not (args.output/'features_complete.json').exists():
        features=np.lib.format.open_memmap(args.output/'foundation.npy',mode='w+',dtype='float32',shape=(len(rows),width))
        source_logits=np.lib.format.open_memmap(args.output/'source_logits.npy',mode='w+',dtype='float32',shape=(len(rows),23))
        iq=np.load(data/'iq.npy',mmap_mode='r');captured={}
        hook=model.heads['modulation'].register_forward_pre_hook(lambda m,a:captured.update(features=a[0].detach()))
        try:
            with torch.inference_mode():
                for length in [512,1024]:
                    positions=np.flatnonzero(meta['length'][rows]==length)
                    for start in tqdm(range(0,len(positions),128),desc=f'frozen96/native{length}',mininterval=5):
                        idx=positions[start:start+128]
                        out=model(torch.from_numpy(np.array(iq[rows[idx],:length])).to(device),torch.full((len(idx),),length,device=device))
                        features[idx]=captured['features'].float().cpu().numpy();source_logits[idx]=out['modulation'].float().cpu().numpy()
        finally:hook.remove()
        features.flush();source_logits.flush()
        write_json(args.output/'features_complete.json',{'rows':len(rows),'width':width,'source':str(source)})
    del model,c;torch.cuda.empty_cache()
    base=np.load(args.output/'foundation.npy',mmap_mode='r')
    old=np.load(original/'existing32.npy',mmap_mode='r')
    filtered=np.load(ROOT/'artifacts/bandpass_probe/20261114_01/filtered.npy',mmap_mode='r')
    extra=np.concatenate([old,filtered],1)
    mean=extra[train].mean(0,dtype=np.float64).astype(np.float32)
    std=extra[train].std(0,dtype=np.float64).astype(np.float32).clip(1e-5)
    normalized=(extra-mean)/std
    xbase=torch.from_numpy(np.array(base)).to(device);xextra=torch.from_numpy(normalized).to(device)
    y=torch.from_numpy(labels).to(device);source_logits=np.load(args.output/'source_logits.npy',mmap_mode='r')
    source_metrics=group_metrics(torch.from_numpy(np.array(source_logits[validation])).softmax(1).numpy(),labels[validation],groups[validation])
    write_json(args.output/'source_metrics.json',source_metrics)
    results=read_json(args.output/'results.json') if (args.output/'results.json').exists() else {}
    for arm,dimensions in [('original_readout',0),('raw32_readout',32),('raw32_filtered96_readout',128)]:
        if arm in results:continue
        started=time.monotonic();seed_all(SEED)
        head=nn.Sequential(nn.Linear(width+dimensions,hidden),nn.GELU(),nn.Linear(hidden,23)).to(device)
        with torch.no_grad():
            head[0].weight.zero_();head[0].weight[:,:width].copy_(head_source['0.weight'])
            head[0].bias.copy_(head_source['0.bias']);head[2].weight.copy_(head_source['2.weight']);head[2].bias.copy_(head_source['2.bias'])
        x=torch.cat([xbase,xextra[:,:dimensions]],1) if dimensions else xbase
        with torch.inference_mode():
            ids=validation[:128];initial=head(x[ids]).cpu().numpy()
        initial_error=float(np.max(abs(initial-source_logits[ids])))
        if initial_error>2e-4:raise ValueError(f'Initial readout does not reproduce source: {initial_error}')
        optimizer=torch.optim.AdamW([{'params':[p for p in head.parameters() if p.ndim>1],'weight_decay':.01},
            {'params':[p for p in head.parameters() if p.ndim==1],'weight_decay':0.}],lr=1e-4)
        steps=(len(train)+511)//512;total=steps*args.epochs
        schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/steps)*(.05+.95*(1+np.cos(np.pi*min(step,total)/total))/2))
        history=[]
        for epoch in range(args.epochs):
            order=torch.tensor(np.random.default_rng(SEED+epoch).permutation(train),device=device)
            total_loss=0.;correct=0
            for start in range(0,len(order),512):
                ids=order[start:start+512];optimizer.zero_grad(set_to_none=True)
                z=head(x[ids]);loss=F.cross_entropy(z,y[ids]);loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step();schedule.step()
                total_loss+=float(loss.detach())*len(ids);correct+=int((z.argmax(1)==y[ids]).sum())
            history.append(dict(epoch=epoch+1,train_ce=total_loss/len(train),train_accuracy=correct/len(train),lr=optimizer.param_groups[0]['lr']))
            write_json(args.output/f'{arm}_history.json',history)
            print(arm,'epoch',epoch+1,history[-1],flush=True)
        with torch.inference_mode():z=torch.cat([head(x[validation[start:start+512]]) for start in range(0,len(validation),512)]).cpu().numpy()
        metrics=group_metrics(torch.from_numpy(z).softmax(1).numpy(),labels[validation],groups[validation])
        np.savez_compressed(args.output/f'{arm}_validation.npz',rows=rows[validation],modulation=z)
        results[arm]=dict(metrics=metrics,initial_logits_max_abs_error=initial_error,head_parameters=sum(p.numel() for p in head.parameters()),elapsed_seconds=time.monotonic()-started)
        write_json(args.output/'results.json',results)
        torch.save(dict(head={k:v.detach().cpu() for k,v in head.state_dict().items()},extra_mean=torch.from_numpy(mean[:dimensions].copy()),extra_std=torch.from_numpy(std[:dimensions].copy()),
            protocol=protocol,extra_dimensions=dimensions,stage='diagnostic_frozen_readout',resume_allowed=False),args.output/f'{arm}.pt')
        print('ARM COMPLETE',arm,{g:round(m['accuracy'],5) for g,m in metrics.items()},flush=True)
        del head,optimizer
    print('FINISHED',args.output,flush=True)


if __name__=='__main__':main()
