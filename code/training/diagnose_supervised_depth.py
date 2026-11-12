"""Frozen supervised encoder: layer probes and exact train-only nearest neighbors."""
import argparse
import hashlib
import math
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.rf_model import RFModel
from training.hypothesis_probe import prepare_data,group_metrics
from training.diagnose_readout import feature_metrics
from utils.common import model_bundle,runtime,seed_all,write_json,read_json,save_checkpoint
from utils.config import make_scheduler
from utils.dataset import collate,to_device
from utils.metrics import classification_metrics
from utils.readable_log import ReadableRunLog
from utils.training_display import EpochProgress


LAYERS=['input','layer3','layer6','layer9','layer12_pre_norm','final']


def state_hash(model):
    value=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        value.update(name.encode());value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def extract_layers(args,device):
    out=args.output/'layers';out.mkdir(exist_ok=False)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    model=RFModel(**model_bundle(checkpoint)).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'],strict=True)
    if len(model.backbone.blocks)!=12:raise ValueError('This planned layer grid requires the current 12-layer supervised checkpoint')
    before=state_hash(model);current={};captured={}
    def capture(name,output):
        valid=torch.arange(output.shape[1],device=device)[None]*model.config['patch_size']<current['lengths'][:,None]
        captured[name]=((output.float()*valid[...,None]).sum(1)/valid.sum(1,keepdim=True)).cpu().numpy()
    handles=[model.backbone.blocks[0].register_forward_pre_hook(lambda module,inputs:capture('input',inputs[0]))]
    for index,name in [(2,'layer3'),(5,'layer6'),(8,'layer9'),(11,'layer12_pre_norm')]:
        handles.append(model.backbone.blocks[index].register_forward_hook(lambda module,inputs,output,key=name:capture(key,output)))
    handles.append(model.backbone.norm.register_forward_hook(lambda module,inputs,output:capture('final',output)))
    protocol={'checkpoint':str(args.checkpoint.resolve()),'dataset_fingerprint':checkpoint['dataset_fingerprint'],
        'layers':LAYERS,'pooling':'valid-token mean','precision':'fp32','device':str(device),
        'initial_weight_sha256':before,'frozen':True,'reference':str(args.reference.resolve()),'splits':{}}
    started=time.monotonic()
    try:
        with torch.inference_mode():
            for split in ['train','validation']:
                data=prepare_data(args.data,split)
                if data.info['fingerprint']!=checkpoint['dataset_fingerprint']:raise ValueError('Split mismatch')
                reference_rows=np.load(args.reference/f'{split}_rows.npy')
                assert np.array_equal(reference_rows,data.indices)
                np.save(out/f'{split}_rows.npy',data.indices)
                np.save(out/f'{split}_labels.npy',data.meta['modulation'][data.indices].astype(np.int64))
                arrays={name:np.lib.format.open_memmap(out/f'{split}_{name}.npy',mode='w+',dtype=np.float32,
                    shape=(len(data),model.config['width'])) for name in LAYERS}
                logits=np.lib.format.open_memmap(out/f'{split}_logits.npy',mode='w+',dtype=np.float32,
                    shape=(len(data),len(model.schema['classification_labels']['modulation'])))
                loader=DataLoader(data,batch_size=256,collate_fn=collate,num_workers=2,pin_memory=True)
                offset=0;progress=EpochProgress(0,1,len(loader),min_interval=2.)
                with progress:
                    for cpu in loader:
                        batch=to_device(cpu,device);current['lengths']=batch['lengths'];captured.clear()
                        output=model(batch['iq'],batch['lengths'])
                        count=len(cpu['rows'])
                        assert set(captured)==set(LAYERS)
                        for name,values in captured.items():arrays[name][offset:offset+count]=values
                        logits[offset:offset+count]=output['modulation'].cpu().numpy()
                        offset+=count;progress.observe(0.,{},set());progress.advance({},0.)
                for value in arrays.values():value.flush()
                logits.flush()
                baseline=np.load(args.reference/f'{split}_features.npy',mmap_mode='r')
                maximum=0.
                for begin in range(0,len(data),4096):maximum=max(maximum,float(np.max(np.abs(arrays['final'][begin:begin+4096]-baseline[begin:begin+4096]))))
                original=np.load(args.reference/f'reference_{split}_predictions.npy')
                prediction=logits.argmax(1);agreement=float(np.mean(original==prediction))
                protocol['splits'][split]={'rows':len(data),'max_final_feature_difference':maximum,
                    'cached_prediction_agreement':agreement,'accuracy':float(np.mean(prediction==data.meta['modulation'][data.indices]))}
                print(split,protocol['splits'][split],flush=True)
                assert maximum<2e-4
                del arrays,logits
    finally:
        for handle in handles:handle.remove()
    after=state_hash(model);assert before==after
    protocol.update(final_weight_sha256=after,elapsed_seconds=time.monotonic()-started)
    write_json(out/'protocol.json',protocol)


def probe_layers(args):
    cache=args.output/'layers';out=args.output/'layer_probes';out.mkdir(exist_ok=False)
    names=read_json(args.data/'dataset.json')['schema']['classification_labels']['modulation']
    train_y=torch.from_numpy(np.load(cache/'train_labels.npy'));val_y=torch.from_numpy(np.load(cache/'validation_labels.npy'))
    results=[]
    with ReadableRunLog(out):
        for name in LAYERS:
            train=torch.from_numpy(np.array(np.load(cache/f'train_{name}.npy',mmap_mode='r')))
            validation=torch.from_numpy(np.array(np.load(cache/f'validation_{name}.npy',mmap_mode='r')))
            mean,std=train.mean(0),train.std(0,unbiased=False).clamp_min(1e-4)
            x,v=(train-mean)/std,(validation-mean)/std
            for seed in [2026,2027]:
                seed_all(seed);head=torch.nn.Linear(x.shape[1],len(names));optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=.01)
                scheduler,_=make_scheduler(optimizer,dict(name='cosine',warmup_steps=0,warmup_ratio=0.,
                    warmup_start_factor=0.,min_lr_ratio=.05),math.ceil(len(x)/1024)*20)
                for epoch in range(20):
                    order=torch.randperm(len(x),generator=torch.Generator().manual_seed(seed+epoch*100003))
                    with EpochProgress(epoch,20,math.ceil(len(x)/1024),min_interval=2.) as progress:
                        for begin in range(0,len(order),1024):
                            index=order[begin:begin+1024];optimizer.zero_grad(set_to_none=True)
                            loss=F.cross_entropy(head(x[index]),train_y[index]);loss.backward()
                            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
                            rate=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
                            progress.observe(float(loss.detach()),{'modulation':float(loss.detach())},{'modulation'});progress.advance({'head':rate},float(norm))
                a,_=feature_metrics(head,x,train_y,names);b,prediction=feature_metrics(head,v,val_y,names)
                a['hard14']=group_metrics(a);b['hard14']=group_metrics(b)
                result={'layer':name,'seed':seed,'train':a,'validation':b};results.append(result)
                save_checkpoint(out/f'{name}_seed{seed}.pt',dict(head=head.state_dict(),mean=mean,std=std))
                np.save(out/f'{name}_seed{seed}_predictions.npy',prediction)
                write_json(out/'summary.json',{'results':results,'recipe':{'epochs':20,'batch_size':1024,
                    'learning_rate':1e-3,'weight_decay':.01,'seeds':[2026,2027],'normalization':'train-only mean/std',
                    'selection':'fixed final epoch','encoder_frozen':True,'device':'cpu'}})
                print(name,seed,'train',a['accuracy'],'val',b['accuracy'],'hard14',b['hard14']['accuracy'],flush=True)
            del train,validation,x,v


def evaluate_neighbors(labels,prediction,names):
    matrix=np.bincount(labels*len(names)+prediction,minlength=len(names)**2).reshape(len(names),len(names))
    return classification_metrics(matrix,names)


@torch.inference_mode()
def nearest_neighbors(args,device):
    out=args.output/'neighbors';out.mkdir(exist_ok=False)
    data=prepare_data(args.data,'train');meta=data.meta
    names=data.info['schema']['classification_labels']['modulation']
    train_rows=np.load(args.reference/'train_rows.npy');val_rows=np.load(args.reference/'validation_rows.npy')
    y=meta['modulation'][train_rows].astype(np.int64);vy=meta['modulation'][val_rows].astype(np.int64)
    assert len(np.intersect1d(meta['hash'][train_rows],meta['hash'][val_rows]))==0
    summary={}
    for kind in ['embedding','raw_iq']:
        if kind=='embedding':
            a=torch.from_numpy(np.array(np.load(args.reference/'train_features.npy',mmap_mode='r'))).to(device)
            b=torch.from_numpy(np.array(np.load(args.reference/'validation_features.npy',mmap_mode='r'))).to(device)
            mean,std=a.mean(0),a.std(0,unbiased=False).clamp_min(1e-4)
            a,b=F.normalize((a-mean)/std,dim=-1),F.normalize((b-mean)/std,dim=-1)
            labels,targets,source_rows,query_rows=y,vy,train_rows,val_rows
        else:
            iq=np.load(args.data/'iq.npy',mmap_mode='r');source_rows=train_rows[y<14];query_rows=val_rows[vy<14]
            labels,targets=y[y<14],vy[vy<14]
            arrays=[]
            for rows in [source_rows,query_rows]:
                pieces=[]
                for begin in range(0,len(rows),2048):
                    selected=rows[begin:begin+2048];x=torch.from_numpy(np.array(iq[selected])).to(device)
                    z=torch.complex(x[...,0],x[...,1]);pieces.append(z/z.abs().square().sum(1,keepdim=True).sqrt().clamp_min(1e-12))
                arrays.append(torch.cat(pieces))
            a,b=arrays
        target_tensor=torch.from_numpy(labels).to(device)
        nearest,predictions,distances,correct_max,other_max=[],[],[],[],[]
        neighbors_list=[]
        for begin in range(0,len(b),128):
            query=b[begin:begin+128]
            similarity=(query@a.conj().T).abs() if kind=='raw_iq' else query@a.T
            values,indices=similarity.topk(5,dim=-1)
            neighbor_labels=target_tensor[indices]
            votes=torch.zeros(len(query),len(names),device=device)
            votes.scatter_add_(1,neighbor_labels,torch.exp((values-values[:,:1])/.1))
            predictions.append(votes.argmax(1).cpu().numpy());nearest.append(neighbor_labels[:,0].cpu().numpy())
            distances.append(values[:,0].cpu().numpy());neighbors_list.append(indices.cpu().numpy())
            match=target_tensor[None]==torch.from_numpy(targets[begin:begin+len(query)]).to(device)[:,None]
            correct_max.append(similarity.masked_fill(~match,-torch.inf).max(1).values.cpu().numpy())
            other_max.append(similarity.masked_fill(match,-torch.inf).max(1).values.cpu().numpy())
        nearest=np.concatenate(nearest);prediction=np.concatenate(predictions);values=np.concatenate(distances)
        indices=np.concatenate(neighbors_list);same=np.concatenate(correct_max);other=np.concatenate(other_max)
        metric_names=names[:14] if kind=='raw_iq' else names
        m=evaluate_neighbors(targets,nearest,metric_names);v=evaluate_neighbors(targets,prediction,metric_names)
        m['hard14']=group_metrics(m);v['hard14']=group_metrics(v)
        summary[kind]={'one_neighbor':m,'weighted_five_neighbors':v,
            'same_label_nearest_margin_quantiles':np.quantile(same-other,[0,.1,.25,.5,.75,.9,1]).tolist(),
            'nearest_similarity_quantiles':np.quantile(values,[0,.1,.25,.5,.75,.9,.99,1]).tolist()}
        if kind=='raw_iq':
            summary[kind]['cross_label_similarity_slices']={str(t):{'support':int((other>=t).sum()),
                'same_label_also_above_threshold':int(((other>=t)&(same>=t)).sum())} for t in [.9,.95,.99,.995,.999,.9999]}
            summary[kind]['high_similarity_slices']={str(t):{'support':int((values>=t).sum()),
                'same_label_fraction':float((nearest[values>=t]==targets[values>=t]).mean()) if (values>=t).any() else None}
                for t in [.9,.95,.99,.995,.999,.9999]}
            wrong=np.flatnonzero(nearest!=targets);selected=wrong[np.argsort(values[wrong])[-50:][::-1]]
            examples=[]
            for i in selected:
                row=int(query_rows[i]);neighbor=int(source_rows[indices[i,0]])
                raw_a=np.asarray(iq[row],dtype=np.float64);raw_b=np.asarray(iq[neighbor],dtype=np.float64)
                za=raw_a[:,0]+1j*raw_a[:,1];zb=raw_b[:,0]+1j*raw_b[:,1]
                exact=abs(np.vdot(za,zb))/np.sqrt(np.vdot(za,za).real*np.vdot(zb,zb).real)
                za-=za.mean();zb-=zb.mean();centered=abs(np.vdot(za,zb))/max(1e-24,np.sqrt(np.vdot(za,za).real*np.vdot(zb,zb).real))
                examples.append({'validation_row':row,'train_row':neighbor,'true_label':names[int(targets[i])],
                    'neighbor_label':names[int(nearest[i])],'gpu_similarity':float(values[i]),'float64_similarity':float(exact),
                    'centered_float64_similarity':float(centered)})
            summary[kind]['closest_cross_label_examples']=examples
        np.savez(out/f'{kind}.npz',validation_rows=query_rows,train_rows=source_rows,labels=targets,
            prediction_1nn=nearest,prediction_5nn=prediction,nearest_similarity=values,neighbor_positions=indices,
            same_label_max_similarity=same,other_label_max_similarity=other)
        write_json(out/'summary.json',summary)
        print(kind,'1NN',m['accuracy'],'hard14',m['hard14']['accuracy'],'5NN',v['accuracy'],flush=True)
        del a,b
    write_json(out/'protocol.json',{'reference':str(args.reference.resolve()),'search_pool':'train only; no exact hash overlap',
        'embedding_metric':'train-zscore L2 normalized cosine','raw_iq_metric':'exact full-window absolute complex cosine; gain/phase aligned, no time/CFO alignment',
        'raw_iq_scope':'known first14-class subset only; no DC removal in search','precision':'fp32 / complex64, TF32 disabled',
        'five_neighbor_vote_temperature':.1,'not_label_noise_proof':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['extract','probe','neighbors'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('artifacts/data_full'))
    p.add_argument('--checkpoint',type=Path,default=Path('logs/supervised_main/20261110_233950_782228/best.pt'))
    p.add_argument('--reference',type=Path,default=Path('artifacts/diagnostics/20261111_110649_079619/frozen_large'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    device,_,world=runtime('cpu' if args.stage=='probe' else 'auto',4)
    if world!=1:raise ValueError('One independent GPU/process per diagnostic')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.stage=='extract':extract_layers(args,device)
    elif args.stage=='probe':probe_layers(args)
    else:nearest_neighbors(args,device)


if __name__=='__main__':main()
