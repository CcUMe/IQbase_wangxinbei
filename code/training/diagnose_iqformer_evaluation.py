"""Frozen-weight IQFormer evaluation-state interventions; one GPU only.

Batch-statistics modes are diagnostics, not deployable sample-independent
predictions. Validation batches are shuffled without labels to avoid the
class-sorted batches present in some caches. Recalibration sees train IQ only.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'compare/IQFormer'))
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from c01.runtime import build_model,setup_runtime,group_metrics
from c01.data import C01Modulation,SignalViews,LengthBatches
from utils.common import write_json
from utils.metrics import classification_metrics


def loader(data,config,seed):
    return DataLoader(data,batch_sampler=LengthBatches(data.lengths,128,True,seed),
        collate_fn=SignalViews(config['input']),num_workers=4,pin_memory=True,
        persistent_workers=True,prefetch_factor=2)


@torch.inference_mode()
def evaluate(model,batches,labels,name,out):
    rows=[];logits=[];truth=[]
    for batch in tqdm(batches,desc=name,mininterval=10):
        z=model(batch['iq'].cuda(),batch['stft'].cuda()).float().cpu().numpy()
        if not np.isfinite(z).all():raise FloatingPointError(name)
        rows.append(batch['rows'].numpy());truth.append(batch['target'].numpy());logits.append(z)
    rows=np.concatenate(rows);truth=np.concatenate(truth);logits=np.concatenate(logits)
    order=np.argsort(rows);rows,truth,logits=rows[order],truth[order],logits[order]
    pred=logits.argmax(1);n=len(labels)
    cm=np.bincount(truth*n+pred,minlength=n*n).reshape(n,n)
    metric=classification_metrics(cm,labels)
    result=dict(classification=metric,groups=group_metrics(metric),rows=len(rows))
    np.savez_compressed(out/(name+'.npz'),rows=rows,labels=truth,logits=logits)
    write_json(out/(name+'_metrics.json'),result)
    print(name,dict(acc=metric['accuracy'],macro_f1=metric['macro_f1'],groups=result['groups']),flush=True)
    return result


def set_bn(model,mode):
    model.eval()
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
            if mode=='all_batch' or (mode=='input_batch' and name in ['BN','BN_stft']):
                module.train();module.track_running_stats=False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=ROOT/'logs/compare_iqformer/20261113_212535_300325')
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or torch.cuda.device_count()!=1:raise ValueError('Only physical GPU 0')
    args.output.mkdir(parents=True,exist_ok=False)
    if not (args.source/'best.pt').is_file():raise ValueError('Missing source')
    best=torch.load(args.source/'best.pt',map_location='cpu',weights_only=True,mmap=True)
    config=copy.deepcopy(best['config']);setup_runtime(config,'cuda:0')
    validation=C01Modulation(config['run']['data'],'validation')
    training=C01Modulation(config['run']['data'],'train')
    rng=np.random.default_rng(2026111405);chosen=np.sort(rng.choice(len(training),4096,replace=False))
    for key in ['rows','targets','lengths','sample_ids']:setattr(training,key,getattr(training,key)[chosen])
    np.save(args.output/'calibration_training_rows.npy',training.rows)
    val_loader=loader(validation,config,2026111406);train_loader=loader(training,config,2026111407)
    protocol=dict(source=str(args.source),device='physical GPU 0, one process',
        validation_batching='label-independent shuffle within native lengths; same seed across interventions',
        train_calibration_records=len(training),train_calibration_labels_used=False,
        note='Batch-statistics validation is diagnostic only; changes normalization context and is not an inference prescription',
        original_config=config)
    write_json(args.output/'protocol.json',protocol);results={};start=time.monotonic()
    for estimator in ['best','last']:
        ck=torch.load(args.source/(estimator+'.pt'),map_location='cpu',weights_only=True,mmap=True)
        source=ck['model'];config=ck['config'];labels=ck['labels']
        def fresh():
            model=build_model(config,labels).cuda();model.load_state_dict(source,strict=True);return model.eval()
        model=fresh()
        key=estimator+'_eval';results[key]=evaluate(model,val_loader,labels,key,args.output)
        expected=ck['metrics']['classification']['modulation']['accuracy']
        if abs(results[key]['classification']['accuracy']-expected)>.001:raise ValueError('Original validation did not reproduce')
        key=estimator+'_train_panel_eval';results[key]=evaluate(model,train_loader,labels,key,args.output)
        del model
        for mode in ['input_batch','all_batch']:
            model=fresh();set_bn(model,mode);key=estimator+'_'+mode
            results[key]=evaluate(model,val_loader,labels,key,args.output)
            if any(not torch.equal(value.cpu(),source[name]) for name,value in model.state_dict().items()):
                raise ValueError('Batch-statistics probe mutated saved state')
            if mode=='all_batch':
                alternate=loader(validation,config,2026111408)
                key=estimator+'_all_batch_alternate';results[key]=evaluate(model,alternate,labels,key,args.output)
                del alternate
            del model
        model=fresh()
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats();module.momentum=None;module.train()
        with torch.inference_mode():
            for batch in tqdm(train_loader,desc=estimator+' train-only BN calibration',mininterval=10):
                model(batch['iq'].cuda(),batch['stft'].cuda())
        model.eval();key=estimator+'_train_recalibrated'
        results[key]=evaluate(model,val_loader,labels,key,args.output)
        for name,p in model.named_parameters():
            if not torch.equal(p.cpu(),source[name]):raise ValueError('Learned weights changed during calibration')
        torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()},config=config,labels=labels,
            source_checkpoint=str(args.source/(estimator+'.pt')),resume_allowed=False,
            format='iqformer-bn-diagnostic-only',metrics=results[key]),args.output/(key+'_state.pt'))
        del model,ck;torch.cuda.empty_cache();write_json(args.output/'results.json',results)
    write_json(args.output/'completion.json',dict(elapsed_seconds=time.monotonic()-start,status='complete'))


if __name__=='__main__':main()
