"""Training-only, class-balanced confidence/stability ranking for curriculum.

The small teacher is used offline only; it is absent from the submitted model.
Validation labels, RMS-as-SNR assumptions and class routing are not used.
"""
from pathlib import Path
import argparse,sys,os,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'compare/IQFormer'))
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import rankdata
from tqdm import tqdm
from c01.data import C01Modulation,LengthBatches,SignalViews
from c01.runtime import build_model,setup_runtime
from training.calibrate_iqformer_by_length import LengthMoments
from utils.common import write_json


class StableViews:
    def __init__(self,settings):self.views=SignalViews(settings)
    def __call__(self,records):
        a=self.views(records);rotated=[]
        for record in records:
            x=record['iq'];angle=2*np.pi*((record['row']%1009)/1009-.5)
            c,s=np.cos(angle),np.sin(angle)
            y=np.stack([x[:,0]*c-x[:,1]*s,x[:,0]*s+x[:,1]*c],-1).astype(np.float32)
            rotated.append({**record,'iq':y})
        b=self.views(rotated)
        return dict(a=a,b=b)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or torch.cuda.device_count()!=1:raise ValueError('One physical GPU 0 only')
    args.output.mkdir(parents=True,exist_ok=False)
    path=ROOT/'artifacts/causal_diagnosis_20261114/iqformer_population_bn_v2/last_length_layerwise_state.pt'
    ck=torch.load(path,map_location='cpu',weights_only=True,mmap=True);config=ck['config'];setup_runtime(config,'cuda:0')
    base=build_model(config,ck['labels']).cuda();base.load_state_dict(ck['model'],strict=True);model=LengthMoments(base,ck['length_moments']).eval()
    data=C01Modulation(ROOT/'artifacts/data_full','train')
    batches=DataLoader(data,batch_sampler=LengthBatches(data.lengths,128,True,2026111417),collate_fn=StableViews(config['input']),num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=2)
    rows=[];labels=[];confidence=[];stability=[];correct=[];posteriors=[]
    with torch.inference_mode():
        for record in tqdm(batches,desc='Training-only stability ranking',mininterval=10):
            a,b=record['a'],record['b'];y=a['target'].cuda()
            pa=model(a['iq'].cuda(),a['stft'].cuda()).float().softmax(1)
            pb=model(b['iq'].cuda(),b['stft'].cuda()).float().softmax(1)
            m=(pa+pb)/2
            js=(pa*(pa.clamp_min(1e-8).log()-m.clamp_min(1e-8).log())).sum(1)/2+(pb*(pb.clamp_min(1e-8).log()-m.clamp_min(1e-8).log())).sum(1)/2
            prob=torch.minimum(pa.gather(1,y[:,None])[:,0],pb.gather(1,y[:,None])[:,0])
            rows.append(a['rows'].numpy());labels.append(a['target'].numpy());confidence.append(prob.cpu().numpy());stability.append(js.cpu().numpy());correct.append(((pa.argmax(1)==y)&(pb.argmax(1)==y)).cpu().numpy())
            posteriors.append(((pa+pb)/2).cpu().numpy())
    rows=np.concatenate(rows);y=np.concatenate(labels);prob=np.concatenate(confidence);js=np.concatenate(stability);correct=np.concatenate(correct)
    score=prob*np.exp(-js);rank=np.empty_like(score)
    for label in range(23):
        keep=y==label;rank[keep]=(rankdata(score[keep],method='average')-.5)/keep.sum()
    order=np.argsort(rows);np.savez_compressed(args.output/'reliability.npz',rows=rows[order],labels=y[order],probability=prob[order],js=js[order],rank=rank[order],posterior=np.concatenate(posteriors)[order])
    write_json(args.output/'protocol.json',dict(teacher=str(path),training_records=len(rows),dataset_fingerprint=data.info['fingerprint'],
        formula='score=min(p(true_class|x),p(true_class|phase_rotated_x))*exp(-JS); rank=within-class empirical percentile(score)',
        rotation='2*pi*((global_row %1009)/1009-.5), same rotation for IQ and regenerated STFT',
        phase_stable_correct_fraction=float(correct.mean()),teacher_used_at_inference=False,
        labels='Training labels only for true-class confidence and within-class ranking; never validation labels',
        note='Confidence/stability proxy, not measured SNR or a guarantee of clean samples'))
    print('SAVED',args.output,'records',len(rows),'phase-stable correct',correct.mean(),flush=True)


if __name__=='__main__':main()
