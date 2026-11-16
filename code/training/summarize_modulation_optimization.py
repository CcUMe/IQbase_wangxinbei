"""Summarize actual C01 model optimization runs, preserving every outcome."""
import argparse,csv,json,hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
import numpy as np
from training.validate_c01_followup import paired
from utils.common import write_json
from training.optimize_density import format_training_record,format_validation
from utils.tensorboard_log import TensorBoardLog


def csv_file(path,records):
    with path.open('w') as f:
        w=csv.DictWriter(f,list(records[0]));w.writeheader();w.writerows(records)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    ref=args.runs[0];rp=json.loads((ref/'protocol.json').read_text());meta=np.load(ROOT/'artifacts/data_full/metadata.npz');selection=np.load(ROOT/'artifacts/c01_followup_validation/20261113_224434/selection.npz')
    source=dict(np.load(ref/'source_validation.npz'));controls={e:dict(np.load(ref/'control'/(e+'_validation.npz'))) for e in ['last','ema']}
    classes=[];parameters=[];comparisons=[];perclass=[];runs=[];eligible=[]
    source_metric=json.loads((ref/'source_validation_metrics.json').read_text())
    write_json(args.output/'baseline_metrics.json',source_metric)
    for run in args.runs:
        proto=json.loads((run/'protocol.json').read_text())
        for key in ['source_sha256','joint_stream_sha256','readout_stream_sha256','dataset_fingerprint']:
            if proto[key]!=rp[key]:raise ValueError('Unmatched control: '+key)
        for section in ['optimizer','scheduler','evaluation']:
            if proto['config'][section]!=rp['config'][section]:raise ValueError('Unmatched control configuration: '+section)
        for key in ['seed','batch_size','joint_steps','precision','joint_gradient_clip','readout_preparation_steps','readout_gradient_clip']:
            if proto['config']['training'][key]!=rp['config']['training'][key]:raise ValueError('Unmatched training configuration: '+key)
        runs.append(dict(directory=str(run),protocol=proto))
        results=json.loads((run/'results.json').read_text())
        for arm,result in results.items():
            identity=run.name+'/'+arm
            directory=run/arm
            log=directory/'train.log'
            if log.exists() and log.read_text().lstrip().startswith('{'):
                legacy=directory/'train.original.log'
                if not legacy.exists():legacy.write_bytes(log.read_bytes())
                config=proto['config'];records=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines() if line.strip()]
                text=f"训练总览 | 方法={arm} | 单卡GPU0 | batch={config['training']['batch_size']} | 联合更新={config['training']['joint_steps']}\n"
                text+='以下内容由保留的events.jsonl与完整验证指标整理；原始train.log保留在train.original.log。\n'
                text+=''.join(format_training_record(record) for record in records)
                text+=''.join(format_validation(metric,estimator) for estimator,metric in result['metrics'].items())
                log.write_text(text)
            marker=directory/'tensorboard'/'estimator_metrics_backfilled.json'
            if not marker.exists() and proto['config']['logging']['tensorboard']:
                with TensorBoardLog(directory) as tb:
                    for estimator,metric in result['metrics'].items():
                        for task,values in metric['classification'].items():
                            for key in ['accuracy','macro_f1','weighted_f1']:tb.scalar('estimators/'+estimator+'/'+task+'/'+key,values[key],proto['config']['training']['joint_steps'])
                        for field,values in metric['parameters'].items():
                            for key in ['rmse','mae','absolute_error_p95']:tb.scalar('estimators/'+estimator+'/parameters/'+field+'/'+key,values[key],proto['config']['training']['joint_steps'])
                write_json(marker,dict(source='saved full validation metrics',estimators=list(result['metrics'])))
            for estimator,metric in result['metrics'].items():
                for task,values in {**metric['classification'],**metric['conditions']}.items():
                    classes.append(dict(experiment=identity,arm=arm,estimator=estimator,task=task,**{k:values[k] for k in ['support','accuracy','macro_f1','weighted_f1']}))
                for field,values in metric['parameters'].items():parameters.append(dict(experiment=identity,arm=arm,estimator=estimator,field=field,**values))
                for task in ['modulation','protocol']:
                    values=metric['classification'][task]
                    for j,label in enumerate(values['labels']):perclass.append(dict(experiment=identity,estimator=estimator,task=task,label=label,support=values['class_support'][j],precision=values['precision'][j],recall=values['recall'][j],f1=values['f1'][j]))
                pred=dict(np.load(run/arm/(estimator+'_validation.npz')));rr=pred['rows'];assert np.array_equal(rr,source['rows'])
                for reference_name,reference in [('source',source),('control_'+estimator,controls[estimator])]:
                    y=meta['modulation'][rr];masks=[('modulation',y>=0),('first14',(y>=0)&(y<14)),('remaining9',y>=14)]
                    masks.extend((f'first14_rms_q{q+1}',(y>=0)&(y<14)&(selection['condition_groups'][rr,0]==q)) for q in range(4))
                    for task,keep in masks:
                        values=paired(reference['modulation'],pred['modulation'],y,keep);lo,hi=values.pop('difference_ci95_normal')
                        comparisons.append(dict(experiment=identity,estimator=estimator,reference=reference_name,task=task,**values,ci95_lower=lo,ci95_upper=hi))
                ratio={k:v['rmse']/source_metric['parameters'][k]['rmse'] for k,v in metric['parameters'].items()}
                eligible.append(dict(experiment=identity,arm=arm,estimator=estimator,checkpoint=str(run/arm/(estimator+'.pt')),
                    modulation_accuracy=metric['classification']['modulation']['accuracy'],first14_accuracy=metric['classification']['first14']['accuracy'],
                    modulation_macro_f1=metric['classification']['modulation']['macro_f1'],
                    modulation_weighted_f1=metric['classification']['modulation']['weighted_f1'],
                    modulation_gain_pp=100*(metric['classification']['modulation']['accuracy']-source_metric['classification']['modulation']['accuracy']),
                    first14_gain_pp=100*(metric['classification']['first14']['accuracy']-source_metric['classification']['first14']['accuracy']),
                    protocol_accuracy=metric['classification']['protocol']['accuracy'],parameter_rmse_ratios=ratio,
                    task_guard=metric['classification']['protocol']['accuracy']>=source_metric['classification']['protocol']['accuracy']-.003 and max(ratio.values())<=1.05))
    csv_file(args.output/'classification.csv',classes);csv_file(args.output/'parameters.csv',parameters);csv_file(args.output/'per_class.csv',perclass);csv_file(args.output/'paired_comparisons.csv',comparisons)
    write_json(args.output/'runs.json',runs);write_json(args.output/'candidates.json',sorted(eligible,key=lambda x:x['modulation_accuracy'],reverse=True))
    names=list(dict.fromkeys(r['experiment'] for r in classes));lookup={(r['experiment'],r['estimator'],r['task']):r for r in classes}
    lines=['| 方法 | 权重 | 调制Acc | Macro-F1 | Weighted-F1 | 前14类Acc | 协议Acc |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    baseline=source_metric['classification']
    lines.append('| 原权重 | source | '+' | '.join(f'{100*v:.3f}%' for v in [baseline['modulation']['accuracy'],baseline['modulation']['macro_f1'],baseline['modulation']['weighted_f1'],baseline['first14']['accuracy'],baseline['protocol']['accuracy']])+' |')
    for identity in names:
        for estimator in ['last','ema']:
            m=lookup[identity,estimator,'modulation'];first=lookup[identity,estimator,'first14'];proto=lookup[identity,estimator,'protocol']
            lines.append('| '+identity+' | '+estimator+' | '+' | '.join(f'{100*v:.3f}%' for v in [m['accuracy'],m['macro_f1'],m['weighted_f1'],first['accuracy'],proto['accuracy']])+' |')
    (args.output/'metrics_table.md').write_text('\n'.join(lines)+'\n')
    snapshot=args.output/'source_snapshot';hashes={}
    for path in [*list((ROOT/'code/models').glob('*.py')),*list((ROOT/'code/utils').glob('*.py')),ROOT/'code/training/optimize_density.py',ROOT/'code/training/build_density_reliability.py',ROOT/'code/training/summarize_modulation_optimization.py',ROOT/'code/training/test_density_encoding.py',ROOT/'code/configs/supervised/density_optimization.json5']:
        relative=path.relative_to(ROOT);target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);data=path.read_bytes();target.write_bytes(data);hashes[str(relative)]=hashlib.sha256(data).hexdigest()
    write_json(snapshot/'sha256.json',hashes)
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
