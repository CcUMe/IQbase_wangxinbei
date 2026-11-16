"""Tables, paired comparisons and standalone figures for causal diagnostics."""
from pathlib import Path
import csv
import json
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
import numpy as np
from training.validate_c01_followup import class_result,paired
from utils.common import write_json


def csv_file(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    root=ROOT/'artifacts/causal_diagnosis_20261114';out=root/'summary';out.mkdir(exist_ok=True)
    labels=json.loads((ROOT/'artifacts/data_full/dataset.json').read_text())['schema']['classification_labels']['modulation']
    audit=root/'iqformer_eval_state';cal=root/'iqformer_population_bn_v2';rows=[];pairs=[];per_class=[]
    for directory in [audit,cal]:
        results=json.loads((directory/'results.json').read_text())
        for name,metric in results.items():
            if 'train_panel' in name:continue
            m=metric['classification'];rows.append(dict(method=name,accuracy=m['accuracy'],macro_f1=m['macro_f1'],weighted_f1=m['weighted_f1'],
                first14_accuracy=metric['groups']['first14']['accuracy'],remaining9_accuracy=metric['groups']['remaining9']['accuracy'],support=metric['rows']))
            for j,label in enumerate(labels):
                per_class.append(dict(method=name,label=label,support=m['class_support'][j],precision=m['precision'][j],recall=m['recall'][j],f1=m['f1'][j]))
    for estimator in ['best','last']:
        a=dict(np.load(audit/(estimator+'_eval.npz')));b=dict(np.load(cal/(estimator+'_length_layerwise.npz')))
        assert np.array_equal(a['rows'],b['rows']) and np.array_equal(a['labels'],b['labels'])
        for task,keep in [('all',a['labels']>=0),('first14',a['labels']<14),('remaining9',a['labels']>=14)]:
            p=paired(a['logits'],b['logits'],a['labels'],keep);lo,hi=p.pop('difference_ci95_normal')
            pairs.append(dict(estimator=estimator,task=task,**p,ci95_lower=lo,ci95_upper=hi))
    csv_file(out/'iqformer_classification.csv',rows);csv_file(out/'iqformer_per_class.csv',per_class);csv_file(out/'iqformer_paired.csv',pairs)
    main_pred=dict(np.load(ROOT/'artifacts/reception_exploration/summary_20261114/folded_validation.npz'))
    other=dict(np.load(cal/'last_length_layerwise.npz'));locations=np.searchsorted(main_pred['rows'],other['rows'])
    assert np.array_equal(main_pred['rows'][locations],other['rows'])
    y=other['labels'];a=main_pred['modulation'][locations].argmax(1);b=other['logits'].argmax(1)
    overlap={}
    for name,keep in [('all',y>=0),('first14',y<14),('remaining9',y>=14)]:
        ca=a[keep]==y[keep];cb=b[keep]==y[keep]
        overlap[name]=dict(support=int(keep.sum()),both_correct=int((ca&cb).sum()),both_wrong=int((~ca&~cb).sum()),
            only_main_correct=int((ca&~cb).sum()),only_iqformer_correct=int((~ca&cb).sum()))
    write_json(out/'cross_model_error_overlap.json',dict(comparison='Folded 95.7M candidate vs length-calibrated last IQFormer',
        groups=overlap,caveat='Observed correctness overlap only; not an attainable accuracy bound, ensemble recipe, or proof a single correct prediction is recoverable'))
    synthetic=root/'oracle_reception';run=Path(json.loads((synthetic/'training_run.json').read_text())['directory'])
    high_run=Path(json.loads((synthetic/'training_run_high_only.json').read_text())['directory'])
    pa=json.loads((run/'protocol.json').read_text());pb=json.loads((high_run/'protocol.json').read_text())
    assert pa['initial_sha256']==pb['initial_sha256'] and pa['data_stream_sha256']==pb['data_stream_sha256']
    assert np.array_equal(np.load(run/'training_stream.npy'),np.load(high_run/'training_stream.npy'))
    metrics=json.loads((run/'results.json').read_text());metrics.update(json.loads((high_run/'results.json').read_text()))
    hist=json.loads((synthetic/'histogram_probe/results.json').read_text());metrics['normalized_histogram']=hist
    ref=json.loads((synthetic/'receiver_reference.json').read_text());metrics['oracle_likelihood']=dict(overall=ref['oracle'],by_snr={k:v['oracle'] for k,v in ref['by_snr'].items()})
    metrics['clock_error_likelihood']=dict(overall=ref['clock_error_10pct'],by_snr={k:v['clock_error_10pct'] for k,v in ref['by_snr'].items()})
    replay=root/'oracle_equal_count_replay'
    for split in ['train','test']:
        for arm in ['raw','oracle']:
            a=np.load(synthetic/(split+'_'+arm+'.npy'),mmap_mode='r');b=np.load(replay/(split+'_'+arm+'.npy'),mmap_mode='r')
            assert np.array_equal(a,b),'Synthetic replay changed observation values'
    equal=json.loads((replay/'receiver_equal_count_reference.json').read_text())
    metrics['oracle_likelihood_100']=dict(overall=equal['overall_oracle'],by_snr={k:v['oracle'] for k,v in equal['by_snr'].items()})
    metrics['clock_error_likelihood_100']=dict(overall=equal['overall_clock_error'],by_snr={k:v['clock_error'] for k,v in equal['by_snr'].items()})
    write_json(out/'equal_count_clock_control.json',dict(replay_observations_identical=True,**equal))
    records=[]
    for arm,result in metrics.items():
        for condition,m in {'all':result['overall'],**result['by_snr']}.items():
            records.append(dict(method=arm,nominal_snr_db=condition,accuracy=m['accuracy'],macro_f1=m['macro_f1'],weighted_f1=m['weighted_f1'],support=m['support']))
    csv_file(out/'synthetic_classification.csv',records)
    x=dict(np.load(run/'oracle_test_predictions.npz'));z=dict(np.load(high_run/'oracle_high_only_test_predictions.npz'))
    keep=x['snr']>=20;p=paired(x['logits'],z['logits'],x['labels'],keep)
    write_json(out/'synthetic_supervision_comparison.json',dict(controls_verified=True,initial_sha256=pa['initial_sha256'],data_stream_sha256=pa['data_stream_sha256'],
        selected_loss='Known synthetic SNR >=20, sum CE divided by full batch size; high-record gradient weight unchanged',comparison=p,
        limitation='Changing selected losses also changes optimizer moments/clipping; this does not isolate gradient direction conflict'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    methods=['eval','input_batch','all_batch','global_layerwise','length_layerwise'];names=['Original','Input BN\nbatch stats','All BN\nbatch stats','Global\nlayerwise','Length\nlayerwise']
    lookup={r['method']:r for r in rows};pos=np.arange(len(methods))
    for axis,key,title in zip(axes,['accuracy','first14_accuracy'],['All 23 classes','First 14 classes']):
        for i,estimator in enumerate(['best','last']):axis.bar(pos+(i-.5)*.36,[100*lookup[estimator+'_'+m][key] for m in methods],width=.36,label=estimator)
        axis.set_xticks(pos,names);axis.set_ylim(0,100);axis.set_title(title);axis.set_ylabel('Accuracy (%)');axis.legend();axis.grid(axis='y',alpha=.2)
    fig.suptitle('Frozen IQFormer weights: train-only length-conditioned moments restore evaluation')
    fig.tight_layout();fig.savefig(out/'iqformer_bn.png',dpi=160);fig.savefig(out/'iqformer_bn.pdf');plt.close(fig)
    fig,axis=plt.subplots(figsize=(9,5))
    for arm,label,style in [('raw','RFModel: raw','-o'),('oracle','RFModel: known receiver','-o'),('oracle_high_only','RFModel: high-condition loss only','--o'),
        ('normalized_histogram','RMS-normalized symbol histogram','-s'),('oracle_likelihood_100','Known-model likelihood (100 samples)',':^'),('clock_error_likelihood_100','Same 100 samples, +10% clock error',':x')]:
        values=metrics[arm]['by_snr'];keys=sorted(values,key=float);axis.plot([float(k) for k in keys],[100*values[k]['accuracy'] for k in keys],style,label=label)
    axis.set_xlabel('Synthetic nominal waveform SNR (dB), not C01 SNR');axis.set_ylabel('Six-class test accuracy (%)');axis.set_ylim(0,105);axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'synthetic_controls.png',dpi=160);fig.savefig(out/'synthetic_controls.pdf');plt.close(fig)
    train=np.load(synthetic/'train_oracle.npy',mmap_mode='r');raw=np.load(synthetic/'train_raw.npy',mmap_mode='r');meta=np.load(synthetic/'train_metadata.npz')
    fig,axes=plt.subplots(2,3,figsize=(10,6))
    for i,label in enumerate([3,5]):
        row=np.flatnonzero((meta['labels']==label)&(meta['snr']==30))[0]
        a=raw[row];a=a/np.sqrt(np.square(a).sum(-1).mean());b=train[row,:896:8];c=b/np.sqrt(np.square(b).sum(-1).mean())
        for axis,points,title in zip(axes[i],[a,b,c],['Raw observation / RMS','Known receiver symbols','Same symbols / RMS']):
            axis.scatter(points[:,0],points[:,1],s=5,alpha=.6);axis.set_title(title);axis.set_aspect('equal');axis.set_xlim(-2,2);axis.set_ylim(-2,2)
        axes[i,0].set_ylabel(['16QAM','256QAM'][i])
    fig.suptitle('Fixed training records, synthetic nominal 30 dB; no C01 condition inference')
    fig.tight_layout();fig.savefig(out/'synthetic_examples.png',dpi=160);plt.close(fig)
    write_json(out/'summary.json',dict(iqformer=rows,synthetic=records,cross_model_overlap=overlap,supervision_controls_verified=True))
    print('Wrote causal diagnostic tables/figures',out,flush=True)


if __name__=='__main__':main()
