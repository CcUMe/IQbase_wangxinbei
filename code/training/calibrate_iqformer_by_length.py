"""Frozen-weight, layerwise population BN calibration using TRAIN IQ only.

Compare global running moments with native-length conditional moments. Length
is an observable input property, never a true modulation label. Each layer's
inputs are measured in eval mode after its ancestors have been calibrated.
"""
import argparse
import copy
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'compare/IQFormer'))
import numpy as np
import torch
from c01.runtime import build_model,setup_runtime
from c01.data import C01Modulation
from training.diagnose_iqformer_evaluation import loader,evaluate
from utils.common import write_json


class LengthMoments(torch.nn.Module):
    def __init__(self,model,states,device='cuda:0'):
        super().__init__();self.model=model;self.states={k:{n:{p:d[p].to(device) for p in ['mean','variance']} for n,d in s.items()} for k,s in states.items()};self.current=None
    def forward(self,iq,spectrum):
        length=iq.shape[-1]
        if self.current!=length:
            for name,module in self.model.named_modules():
                if name in self.states[length]:
                    module.running_mean.copy_(self.states[length][name]['mean'])
                    module.running_var.copy_(self.states[length][name]['variance'])
            self.current=length
        return self.model(iq,spectrum)


@torch.inference_mode()
def calibrate(model,batches,device='cuda:0'):
    model.eval();bn={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)}
    order=[];handles=[m.register_forward_pre_hook(lambda m,args,name=n:order.append(name)) for n,m in bn.items()]
    model(batches[0]['iq'].to(device),batches[0]['stft'].to(device))
    for h in handles:h.remove()
    if len(order)!=len(set(order)) or set(order)!=set(bn):raise ValueError('Need one execution per BN module')
    moments={}
    for name in order:
        total=None;square=None;count=0
        def collect(module,args):
            nonlocal total,square,count
            x=args[0].double();dims=[0,*range(2,x.ndim)]
            a=x.sum(dims);b=x.square().sum(dims)
            total=a if total is None else total+a;square=b if square is None else square+b
            count+=x.numel()//x.shape[1]
        handle=bn[name].register_forward_pre_hook(collect)
        for batch in batches:model(batch['iq'].to(device),batch['stft'].to(device))
        handle.remove();mean=total/count;variance=((square-total.square()/count)/(count-1)).clamp_min(0.)
        bn[name].running_mean.copy_(mean.float());bn[name].running_var.copy_(variance.float())
        moments[name]=dict(mean=mean.float().cpu(),variance=variance.float().cpu(),observations=count)
    print('Calibrated',len(order),'BN layers; training records',sum(len(b['iq']) for b in batches),flush=True)
    return moments


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'logs/compare_iqformer/20261113_212535_300325')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or torch.cuda.device_count()!=1:raise ValueError('GPU 0 only')
    args.output.mkdir(parents=True,exist_ok=False)
    base=torch.load(args.source/'best.pt',map_location='cpu',weights_only=True,mmap=True);config=copy.deepcopy(base['config']);setup_runtime(config,'cuda:0')
    train=C01Modulation(config['run']['data'],'train');val=C01Modulation(config['run']['data'],'validation')
    rng=np.random.default_rng(2026111405);selected=np.sort(rng.choice(len(train),4096,replace=False))
    for key in ['rows','targets','lengths','sample_ids']:setattr(train,key,getattr(train,key)[selected])
    batches=list(loader(train,config,2026111407));by_length={n:[b for b in batches if b['iq'].shape[-1]==n] for n in [512,1024]}
    validation=loader(val,config,2026111406);results={}
    write_json(args.output/'protocol.json',dict(training_rows=train.rows.tolist(),calibration_labels_used=False,
        method='Topological BN calibration: eval-mode ancestors, exact summed moments over train observations',
        length_condition='Native IQ length only; shared learned weights',config=config))
    for estimator in ['best','last']:
        ck=torch.load(args.source/(estimator+'.pt'),map_location='cpu',weights_only=True,mmap=True);states={}
        def fresh():
            model=build_model(config,ck['labels']).cuda();model.load_state_dict(ck['model'],strict=True);return model.eval()
        model=fresh();calibrate(model,batches)
        key=estimator+'_global_layerwise';results[key]=evaluate(model,validation,ck['labels'],key,args.output);del model
        for length,subset in by_length.items():
            model=fresh();states[length]=calibrate(model,subset);del model
        model=fresh();wrapped=LengthMoments(model,states).eval();key=estimator+'_length_layerwise'
        results[key]=evaluate(wrapped,validation,ck['labels'],key,args.output)
        # Independence check: same input in a full native-length batch versus
        # alone, using the same frozen population moments.
        batch=batches[0];a=batch['iq'].cuda();b=batch['stft'].cuda()
        with torch.inference_mode():
            z=wrapped(a,b);single=wrapped(a[:1],b[:1]);error=float((z[:1]-single).abs().max())
        if error>1e-4:raise ValueError('Unexpected dependence on inference batch')
        for name,p in model.named_parameters():
            if not torch.equal(p.cpu(),ck['model'][name]):raise ValueError('Learned parameter changed')
        torch.save(dict(model=ck['model'],length_moments=states,config=config,labels=ck['labels'],
            batch_independence_max_difference=error,source_checkpoint=str(args.source/(estimator+'.pt')),
            format='iqformer-length-moments-diagnostic',resume_allowed=False),args.output/(key+'_state.pt'))
        print('Batch independence max difference',error,flush=True)
        del model,wrapped,ck;torch.cuda.empty_cache();write_json(args.output/'results.json',results)
    write_json(args.output/'completion.json',dict(status='complete'))


if __name__=='__main__':main()
