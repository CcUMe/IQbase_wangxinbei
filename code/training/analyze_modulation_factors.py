"""Full C01 augmentation-factor results, including group accuracy and global F1.

Groups are offline evaluation slices of the same 23-class predictions. They
never route inference or fit class biases. EMA is the primary estimator.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
from training.validate_c01_followup import paired
from utils.common import write_json

GROUPS={'first14':list(range(14)), 'qam':[4,5,6,7,12],
        'psk_related':[1,2,3,8,13], 'fsk_msk':[9,10,11],
        'ook':[0], 'remaining9':list(range(14,23))}


def group_metrics(metric,pred,meta):
    rows=pred['rows'];y=meta['modulation'][rows]
    logits=pred['modulation'];chosen=logits.argmax(1)
    top2=np.argpartition(logits,-2,axis=1)[:,-2:]
    full=metric['classification']['modulation'];result={}
    for name,labels in GROUPS.items():
        keep=np.isin(y,labels);support=int(keep.sum())
        result[name]={'support':support,
            'accuracy':float((chosen[keep]==y[keep]).mean()),
            'top2_accuracy':float((top2[keep]==y[keep,None]).any(1).mean()),
            # Use the full confusion matrix: outside-family false positives
            # must still count against each class's precision and F1.
            'macro_f1_global':float(np.asarray(full['f1'])[labels].mean()),
            'outside_family_predictions':int((keep & ~np.isin(chosen,labels)).sum()),
            'outside_family_false_accepts':int((~keep & (y>=0) & np.isin(chosen,labels)).sum())}
    result['qam16']={key:full[key][4] for key in ['precision','recall','f1','class_support']}
    return result


def collect(runs):
    meta=np.load(ROOT/'artifacts/data_full/metadata.npz')
    ref=Path(runs[0]);rp=json.loads((ref/'protocol.json').read_text())
    source_metric=json.loads((ref/'source_validation_metrics.json').read_text())
    source=dict(np.load(ref/'source_validation.npz'))
    baseline=group_metrics(source_metric,source,meta)
    controls={e:dict(np.load(ref/'control'/(e+'_validation.npz'))) for e in ['last','ema']}
    records=[]
    for run in map(Path,runs):
        proto=json.loads((run/'protocol.json').read_text())
        for key in ['source_sha256','joint_stream_sha256','readout_stream_sha256','dataset_fingerprint']:
            if proto[key]!=rp[key]:raise ValueError('Not a paired factor comparison: '+key)
        for section in ['optimizer','scheduler','evaluation']:
            if proto['config'][section]!=rp['config'][section]:raise ValueError('Different '+section)
        for field in ['seed','joint_steps','batch_size','precision','joint_gradient_clip']:
            if proto['config']['training'][field]!=rp['config']['training'][field]:raise ValueError('Different training '+field)
        for path in sorted(run.glob('*/*_validation_metrics.json')):
            estimator=path.name.removesuffix('_validation_metrics.json')
            if estimator not in ['last','ema']:continue
            checkpoint=path.parent/(estimator+'.pt')
            if not checkpoint.exists():continue
            metric=json.loads(path.read_text());pred=dict(np.load(path.with_name(estimator+'_validation.npz')))
            if not np.array_equal(pred['rows'],source['rows']):raise ValueError('Different validation records')
            groups=group_metrics(metric,pred,meta)
            ratio={key:value['rmse']/source_metric['parameters'][key]['rmse'] for key,value in metric['parameters'].items()}
            guard=(metric['classification']['protocol']['accuracy']>=source_metric['classification']['protocol']['accuracy']-.003
                   and max(ratio.values())<=1.05
                   and groups['qam']['accuracy']>=baseline['qam']['accuracy']-.005
                   and metric['classification']['modulation']['macro_f1']>=source_metric['classification']['modulation']['macro_f1']-.003)
            comparisons={}
            y=meta['modulation'][pred['rows']]
            for name,pred_ref in [('source',source),('control_'+estimator,controls[estimator])]:
                comparisons[name]={}
                for group,labels in {'modulation':list(range(23)),**GROUPS}.items():
                    comparisons[name][group]=paired(pred_ref['modulation'],pred['modulation'],y,np.isin(y,labels))
            records.append(dict(run=str(run),arm=path.parent.name,estimator=estimator,
                checkpoint=str(checkpoint),metrics=metric,groups=groups,
                parameter_rmse_ratios=ratio,task_guard=guard,comparisons=comparisons))
    return dict(protocol=rp,source_metrics=source_metric,source_groups=baseline,records=records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    result=collect(args.runs);write_json(args.output/'factor_metrics.json',result)
    lines=['| Method | Weights | Mod Acc | Macro-F1 | Weighted-F1 | First14 Acc | QAM Acc | QAM global Macro-F1 | 16QAM P/R/F1 | Protocol Acc | Guard |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |']
    for record in result['records']:
        m=record['metrics']['classification'];g=record['groups'];q=g['qam16']
        values=[m['modulation']['accuracy'],m['modulation']['macro_f1'],m['modulation']['weighted_f1'],g['first14']['accuracy'],g['qam']['accuracy'],g['qam']['macro_f1_global']]
        lines.append('| '+record['arm']+' | '+record['estimator']+' | '+' | '.join(f'{100*v:.3f}%' for v in values)+' | '+' / '.join(f'{100*q[key]:.2f}' for key in ['precision','recall','f1'])+f" | {100*m['protocol']['accuracy']:.3f}% | {record['task_guard']} |")
    (args.output/'factor_table.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
