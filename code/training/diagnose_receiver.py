"""Paired full-validation receiver ablation on one frozen checkpoint/GPU.

The disabled condition zeros only the receiver output with a forward hook.
It preserves original weights, context/padding paths and every other feature.
This measures reliance after joint training, not an independently trained control.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.rf_model import RFModel
from utils.common import model_bundle, runtime, write_json
from utils.command_config import configured_launch
from utils.dataset import PreparedDataset, collate
from utils.engine import evaluate


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def paired(truth, enabled, disabled, selected):
    truth, enabled, disabled = (x[selected] for x in [truth, enabled, disabled])
    a, b = enabled == truth, disabled == truth
    n = len(truth)
    gain, harm = int((a & ~b).sum()), int((~a & b).sum())
    if n:
        # Descriptive record bootstrap, preserving pairing. Not a hidden-test CI.
        draws = np.random.default_rng(20261117).multinomial(n, [gain/n,harm/n,1-(gain+harm)/n],size=20000)
        interval = np.quantile((draws[:,0]-draws[:,1])/n,[.025,.975]).tolist()
    else:
        interval = [None,None]
    return {'support': n, 'enabled_accuracy': float(a.mean()) if n else None,
            'disabled_accuracy': float(b.mean()) if n else None,
            'enabled_minus_disabled': (gain-harm)/n if n else None,
            'enabled_correct_disabled_wrong': gain, 'enabled_wrong_disabled_correct': harm,
            'both_correct': int((a & b).sum()), 'both_wrong': int((~a & ~b).sum()),
            'prediction_changes': int((enabled != disabled).sum()),
            'record_bootstrap_delta_95pct': interval}


def analysis(directory, dataset, results, predictions):
    rows = dataset.indices
    truths = {task: dataset.meta[task][rows] for task in ['modulation','protocol']}
    pairs = {}
    for task,truth in truths.items():
        a,b=(predictions[mode][task+'_logits'].argmax(-1) for mode in ['enabled','disabled'])
        valid = truth >= 0
        pairs[task] = paired(truth,a,b,valid)
        pairs[task]['per_class'] = {label:paired(truth,a,b,truth==index)
            for index,label in enumerate(dataset.info['schema']['classification_labels'][task])}
        if task == 'modulation':
            for name,selected in [('first14',(truth>=0)&(truth<14)),('last9',truth>=14),
                                  ('qam',np.isin(truth,[4,5,6,7,12]))]:
                pairs[name] = paired(truth,a,b,selected)
                for mode in ['enabled','disabled']:
                    f1 = np.array(results[mode]['classification']['modulation']['f1'])
                    ids = range(14) if name=='first14' else range(14,23) if name=='last9' else [4,5,6,7,12]
                    pairs[name][mode+'_macro_f1_full23'] = float(f1[list(ids)].mean())
            fsk = dataset.info['schema']['classification_labels']['modulation'].index('2FSK')
            pairs['2fsk'] = {}
            for mode,pred in [('enabled',a),('disabled',b)]:
                count = int(((pred==fsk)&valid).sum());tp = int(((pred==fsk)&(truth==fsk)).sum())
                pairs['2fsk'][mode] = {'predicted':count,'true_positive':tp,'false_positive':count-tp,
                    'precision':tp/max(count,1),'recall':tp/max(int((truth==fsk).sum()),1),
                    'false_positive_by_true_class':{label:int(((pred==fsk)&(truth==j)).sum())
                        for j,label in enumerate(dataset.info['schema']['classification_labels']['modulation']) if j!=fsk}}
    # Offline observation slices; class labels select analysis records only.
    truth = truths['modulation']; valid = truth>=0; first = (truth>=0)&(truth<14)
    iq = np.load(dataset.directory/'iq.npy',mmap_mode='r')
    rms = np.full(len(rows),np.nan)
    indices = np.flatnonzero(valid)
    for start in range(0,len(indices),128):
        local = indices[start:start+128]; source = rows[local]
        x = np.array(iq[source],dtype=np.float64)
        lengths = dataset.meta['length'][source]
        x *= (np.arange(x.shape[1])[None]<lengths[:,None])[...,None]
        rms[local] = np.sqrt(np.square(x).sum((1,2))/lengths)
    a,b=(predictions[mode]['modulation_logits'].argmax(-1) for mode in ['enabled','disabled'])
    boundaries = np.quantile(rms[first],[.2,.4,.6,.8])
    bins = np.searchsorted(boundaries,rms,side='right')
    pairs['first14_rms_quintile_boundaries'] = boundaries.tolist()
    pairs['first14_by_rms'] = {str(i+1):paired(truth,a,b,first&(bins==i)) for i in range(5)}
    pairs['modulation_by_length'] = {str(length):paired(truth,a,b,valid&(dataset.meta['length'][rows]==length))
                                   for length in np.unique(dataset.meta['length'][rows][valid])}
    with (directory/'modulation_records.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['row','sample_id','true_class','prediction_enabled','prediction_disabled','length','raw_complex_rms'])
        for j in indices:
            writer.writerow([int(rows[j]),str(dataset.meta['sample_id'][rows[j]]),int(truth[j]),int(a[j]),int(b[j]),
                             int(dataset.meta['length'][rows[j]]),float(rms[j])])
    write_json(directory/'paired_analysis.json',pairs)
    lines = ['# 冻结第5轮模型：接收分支开/关配对验证','',
        '只将接收分支输出乘0，其余路径、所有学习权重、输入顺序和FP32精度相同。未训练新模型。',
        '这是联合训练后对分支的依赖性检查，不等价于训练时关闭分支的公平对照；关闭可能造成分布变化。','',
        '|任务/分组|开启Acc|关闭Acc|开启−关闭（百分点）|开启独有正确|关闭独有正确|',
        '|---|---:|---:|---:|---:|---:|']
    for key in ['modulation','first14','qam','last9','protocol']:
        p = pairs[key]
        lines.append(f"|{key}|{p['enabled_accuracy']:.4%}|{p['disabled_accuracy']:.4%}|{100*p['enabled_minus_disabled']:+.4f}|{p['enabled_correct_disabled_wrong']}|{p['enabled_wrong_disabled_correct']}|")
    lines += ['', '|任务|开启Macro-F1|关闭Macro-F1|','|---|---:|---:|']
    for task in ['modulation','protocol']:
        lines.append(f"|{task}|{results['enabled']['classification'][task]['macro_f1']:.4%}|{results['disabled']['classification'][task]['macro_f1']:.4%}|")
    lines += ['', '|参数RMSE（原单位）|开启|关闭|','|---|---:|---:|']
    for key,value in results['enabled']['parameters'].items():
        lines.append(f"|{key}|{value['rmse']:.8g}|{results['disabled']['parameters'][key]['rmse']:.8g}|")
    lines += ['', '2FSK及按RMS/长度的分层见`paired_analysis.json`；逐记录结果见`modulation_records.csv`。',
        '记录bootstrap区间仅为当前开发集配对差异的描述，不代表重复训练或独立隐藏测试稳定性。']
    (directory/'report.md').write_text('\n'.join(lines)+'\n')
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpus',default='0')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    if args.batch_size<1 or args.workers<0:parser.error('Invalid batch size or worker count')
    configured_launch(args.gpus,device=args.device)
    device,_,world = runtime(args.device,4)
    if world!=1:raise ValueError('This paired evaluation uses one process/GPU only')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint_hash = digest(args.checkpoint)
    checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    dataset = PreparedDataset(args.data,'validation')
    if checkpoint['dataset_fingerprint']!=dataset.info['fingerprint']:raise ValueError('Dataset fingerprint differs')
    model = RFModel(**model_bundle(checkpoint)).to(device).eval()
    model.load_state_dict(checkpoint['model'],strict=True)
    receiver = getattr(model.backbone,'receiver_frontend',None)
    if receiver is None:raise ValueError('Checkpoint has no shared receiver_frontend')
    metadata = {'checkpoint':str(args.checkpoint.resolve()),'sha256':checkpoint_hash,
                'epoch':checkpoint['epoch']+1,'global_step':checkpoint['global_step'],
                'dataset':str(args.data.resolve()),'dataset_fingerprint':dataset.info['fingerprint'],
                'rows':len(dataset),'physical_gpus':args.gpus,'world_size':world,
                'precision':'fp32','batch_size':args.batch_size,'weights_frozen':True,
                'intervention':'receiver output multiplied by zero; identical context/padding and all other paths',
                'limitation':'Post-training reliance ablation, not a separately trained receiver-disabled baseline.'}
    write_json(args.output/'metadata.json',metadata)
    print(json.dumps({'event':'start',**metadata}),flush=True)
    results,predictions = {},{}
    for mode in ['enabled','disabled']:
        chunks = {name:[] for name in ['modulation_logits','protocol_logits','parameters']}
        started,latest = time.monotonic(),time.monotonic()
        processed = 0
        def capture(module,inputs,output):
            nonlocal processed,latest
            for key in chunks:
                name = key.removesuffix('_logits')
                chunks[key].append(output[name].detach().float().cpu().numpy())
            processed += len(inputs[0])
            if time.monotonic()-latest>10 or processed==len(dataset):
                print(json.dumps({'event':'progress','mode':mode,'rows':processed,'total':len(dataset),
                                  'seconds':round(time.monotonic()-started,1)}),flush=True)
                latest = time.monotonic()
        # Output hook keeps the identical frontend and padding branches active.
        zero = receiver.register_forward_hook(lambda module,inputs,output:output*0) if mode=='disabled' else None
        handle = model.register_forward_hook(capture)
        try:
            loader = DataLoader(dataset,batch_size=args.batch_size,collate_fn=collate,
                                num_workers=args.workers,pin_memory=device.type=='cuda')
            results[mode] = evaluate(model,loader,device,checkpoint['target_scaler'],checkpoint['schema'],
                                     'fp32',0,checkpoint['metrics'].get('proxy_task_weights'))
        finally:
            handle.remove()
            if zero is not None:zero.remove()
        predictions[mode] = {key:np.concatenate(value) for key,value in chunks.items()}
        assert processed==len(dataset)
        write_json(args.output/(mode+'_metrics.json'),results[mode])
        np.savez_compressed(args.output/(mode+'_predictions.npz'),rows=dataset.indices,**predictions[mode])
        print(json.dumps({'event':'condition_complete','mode':mode,'modulation_accuracy':results[mode]['classification']['modulation']['accuracy'],
                          'protocol_accuracy':results[mode]['classification']['protocol']['accuracy']}),flush=True)
    pairs = analysis(args.output,dataset,results,predictions)
    metadata['enabled_reproduces_saved_confusion'] = {task:results['enabled']['classification'][task]['confusion_matrix']==
        checkpoint['metrics']['classification'][task]['confusion_matrix'] for task in ['modulation','protocol']}
    metadata['checkpoint_unchanged'] = digest(args.checkpoint)==checkpoint_hash
    if not metadata['checkpoint_unchanged']:raise RuntimeError('Checkpoint file changed during evaluation')
    metadata['completed'] = True
    write_json(args.output/'metadata.json',metadata)
    print(json.dumps({'event':'complete','output':str(args.output),'paired':{k:pairs[k] for k in ['modulation','first14','qam','last9','protocol']}}),flush=True)


if __name__=='__main__':main()
