"""CPU measurements of conditional BN inputs and disjoint training accuracy."""
from pathlib import Path
import sys
import json

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'compare/IQFormer'))
import numpy as np
import torch
from c01.runtime import build_model,setup_runtime,group_metrics
from c01.data import C01Modulation
from training.diagnose_iqformer_evaluation import loader
from training.calibrate_iqformer_by_length import LengthMoments
from utils.common import write_json
from utils.metrics import classification_metrics


def main():
    if torch.cuda.is_available():raise ValueError('Run with no visible CUDA devices')
    out=ROOT/'artifacts/causal_diagnosis_20261114/iqformer_bn_domains';out.mkdir(exist_ok=False)
    root=ROOT/'artifacts/causal_diagnosis_20261114/iqformer_population_bn_v2'
    ck=torch.load(root/'last_length_layerwise_state.pt',map_location='cpu',weights_only=True,mmap=True)
    original=torch.load(ck['source_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    config=ck['config'];setup_runtime(config,'cpu');data=C01Modulation(config['run']['data'],'train')
    excluded=set(json.loads((root/'protocol.json').read_text())['training_rows'])
    pool=np.flatnonzero(~np.isin(data.rows,list(excluded)));rng=np.random.default_rng(2026111416)
    selected=np.sort(rng.choice(pool,4096,replace=False))
    for key in ['rows','targets','lengths','sample_ids']:setattr(data,key,getattr(data,key)[selected])
    assert not set(data.rows)&excluded
    batches=list(loader(data,config,2026111417));model=build_model(config,ck['labels']).eval();model.load_state_dict(original['model'],strict=True)
    stats={512:{},1024:{}};current=512;handles=[]
    def hook(name):
        def collect(module,args):
            x=args[0].double();dims=[0,*range(2,x.ndim)]
            a,b=x.sum(dims),x.square().sum(dims);count=x.numel()//x.shape[1]
            old=stats[current].get(name)
            stats[current][name]=(a,b,count) if old is None else (old[0]+a,old[1]+b,old[2]+count)
        return collect
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):handles.append(module.register_forward_pre_hook(hook(name)))
    results={};predictions={};truth=[];rows=[]
    with torch.inference_mode():
        z=[]
        for batch in batches:
            current=batch['iq'].shape[-1];z.append(model(batch['iq'],batch['stft']).numpy());truth.extend(batch['target'].tolist());rows.extend(batch['rows'].tolist())
        predictions['original']=np.concatenate(z)
        for handle in handles:handle.remove()
        wrapped=LengthMoments(model,ck['length_moments'],device='cpu').eval()
        predictions['conditional']=np.concatenate([wrapped(batch['iq'],batch['stft']).numpy() for batch in batches])
    truth=np.asarray(truth);n=len(ck['labels'])
    for key,z in predictions.items():
        cm=np.bincount(truth*n+z.argmax(1),minlength=n*n).reshape(n,n)
        metric=classification_metrics(cm,ck['labels']);results[key]=dict(classification=metric,groups=group_metrics(metric))
    domains=[]
    for name in stats[512]:
        a,b,c=stats[512][name];d,e,f=stats[1024][name]
        mu0=a/c;mu1=d/f;var0=(b/c-mu0.square()).clamp_min(0);var1=(e/f-mu1.square()).clamp_min(0)
        separation=(mu0-mu1).abs()/((var0+var1)/2+1e-5).sqrt()
        ratio=(var0+1e-5)/(var1+1e-5)
        domains.append(dict(layer=name,channels=len(mu0),median_mean_separation_in_within_domain_std=float(separation.median()),
            max_mean_separation=float(separation.max()),median_variance_ratio_512_over_1024=float(ratio.median())))
    np.savez_compressed(out/'training_panel_predictions.npz',rows=rows,labels=truth,**predictions)
    write_json(out/'results.json',dict(rows=len(rows),calibration_overlap=0,training=results,bn_domains=domains,
        scope='Same original eval-mode model for all measured BN inputs; no conditional ancestors during moment measurement'))
    print({k:dict(acc=v['classification']['accuracy'],groups=v['groups']) for k,v in results.items()},flush=True)
    print(sorted(domains,key=lambda x:x['median_mean_separation_in_within_domain_std'],reverse=True)[:6],flush=True)


if __name__=='__main__':main()
