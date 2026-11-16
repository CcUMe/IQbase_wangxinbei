"""Train-only fitted, CPU-only probes of additional complex temporal relations.

These are independent diagnostics, not changes to the contest model or an ensemble.
Feature sets and classifier settings are fixed before inspecting held-out results.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
from utils.common import read_json, write_json
from utils.signal_statistics import signal_statistics
from utils.metrics import classification_metrics

SEED = 20261113
LAGS = [1, 2, 4, 8, 16, 32, 64, 128]


def relational_features(iq, lengths=None):
    """Global-phase/positive-gain invariant relations; no fs/label/synchronizer."""
    if lengths is not None:
        blocks, names = [], None
        for length in np.unique(lengths):
            indices = np.flatnonzero(lengths == length)
            values, current = relational_features(iq[indices, :int(length)])
            if names is None:
                names = current
                result = np.empty((len(iq), values.shape[1]), np.float64)
            assert names == current
            result[indices] = values
        return result, names
    z = iq[..., 0].astype(np.float64) + 1j*iq[..., 1].astype(np.float64)
    z -= z.mean(1, keepdims=True)
    variance = (np.abs(z)**2).mean(1, keepdims=True)
    if np.any(variance <= 0):
        raise ValueError('Degenerate centered signal')
    z /= np.sqrt(variance)
    amplitude = np.abs(z)
    power = amplitude**2
    values, names = [], []

    def add(name, value):
        names.append(name)
        values.append(value)

    phase = z / amplitude.clip(1e-12)
    first_difference = phase[:, 1:] * phase[:, :-1].conj()
    for lag in LAGS:
        a, b = z[:, lag:], z[:, :-lag]
        cross = a*b.conj()
        norm = np.sqrt((np.abs(a)**2).mean(1)*(np.abs(b)**2).mean(1)).clip(1e-12)
        correlation = cross.mean(1)/norm
        for name, value in [('real', correlation.real), ('imag', correlation.imag), ('abs', np.abs(correlation))]:
            add(f'lag{lag}_complex_correlation_{name}', value)
        unit = phase[:, lag:]*phase[:, :-lag].conj()
        reliability = np.minimum(np.abs(a), np.abs(b))
        reliability /= reliability.sum(1, keepdims=True).clip(1e-12)
        for order in [1, 2, 4, 8]:
            harmonic = unit**order
            add(f'lag{lag}_phase_harmonic{order}', np.abs(harmonic.mean(1)))
            add(f'lag{lag}_reliable_phase_harmonic{order}', np.abs((harmonic*reliability).sum(1)))
        for order in [2, 4]:
            add(f'lag{lag}_cross_moment{order}', np.log1p(np.abs((cross**order).mean(1))))
        aa, bb = power[:, lag:], power[:, :-lag]
        aa = aa-aa.mean(1, keepdims=True)
        bb = bb-bb.mean(1, keepdims=True)
        add(f'lag{lag}_power_correlation', (aa*bb).mean(1)/np.sqrt((aa**2).mean(1)*(bb**2).mean(1)).clip(1e-12))
        differences = np.abs(a-b)
        for q in [.1, .5, .9]:
            add(f'lag{lag}_increment_q{q}', np.quantile(differences, q, axis=1))
        curvature = first_difference[:, lag:]*first_difference[:, :-lag].conj()
        for order in [1, 2, 4]:
            add(f'lag{lag}_phase_curvature{order}', np.abs((curvature**order).mean(1)))

    def describe_spectrum(v, prefix):
        spectrum = np.abs(np.fft.fft(v, axis=1))**2
        total = spectrum.sum(1, keepdims=True)
        exists = total[:, 0] > 1e-24 * spectrum.shape[1]**2
        p = spectrum/total.clip(1e-30)
        p[~exists] = 0
        n = p.shape[1]
        add(prefix+'_effective_fraction', np.where(exists, 1/np.square(p).sum(1).clip(1e-30)/n, 0.))
        add(prefix+'_entropy', -(p*np.log(p.clip(1e-30))).sum(1)/np.log(n))
        add(prefix+'_peak_share', p.max(1))
        add(prefix+'_top4_share', np.partition(p, -4, axis=1)[:, -4:].sum(1))
        # Frequency unit is cycles/sample; no claim that this is true CFO.
        add(prefix+'_abs_peak_frequency', np.abs(np.fft.fftfreq(n)[p.argmax(1)]))
    for kind, v in [('complex2', z**2), ('complex4', z**4),
                    ('unitphase2', phase**2), ('unitphase4', phase**4), ('unitphase8', phase**8)]:
        describe_spectrum(v, 'nonlinear_spectrum_'+kind)
    describe_spectrum(power-power.mean(1, keepdims=True), 'envelope_spectrum')

    for segments in [4, 8]:
        local = z.reshape(len(z), segments, -1)
        local_power = (np.abs(local)**2).mean(2)
        add(f'segments{segments}_energy_std', local_power.std(1))
        add(f'segments{segments}_energy_range', np.ptp(local_power, axis=1))
        for lag in [1, 8]:
            c = (local[:, :, lag:]*local[:, :, :-lag].conj()).mean(2)
            denominator = np.sqrt((np.abs(local[:, :, lag:])**2).mean(2)*(np.abs(local[:, :, :-lag])**2).mean(2)).clip(1e-12)
            coherence = np.abs(c)/denominator
            add(f'segments{segments}_lag{lag}_coherence_mean', coherence.mean(1))
            add(f'segments{segments}_lag{lag}_coherence_std', coherence.std(1))
            add(f'segments{segments}_lag{lag}_coherence_min', coherence.min(1))
            unit = c/np.abs(c).clip(1e-12)
            add(f'segments{segments}_lag{lag}_direction_consistency', np.abs(unit.mean(1)))
    result = np.stack(values, axis=1)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite relational feature')
    return result, names


def extract(args, output, meta, info):
    if (output/'features_complete.json').exists():
        cached = read_json(output/'features_complete.json')
        if cached['dataset_fingerprint'] != info['fingerprint']:
            raise ValueError('Cached feature fingerprint mismatch')
        return
    rows = np.flatnonzero(meta['modulation'] >= 0)
    raw = np.load(args.data/'iq.npy', mmap_mode='r')
    old = np.lib.format.open_memmap(output/'existing32.npy', mode='w+', dtype='float32', shape=(len(rows), 32))
    new = None
    log_rms = np.empty(len(rows), dtype=np.float32)
    for length in np.unique(meta['length'][rows]):
        positions = np.flatnonzero(meta['length'][rows] == length)
        batches = range(0, len(positions), args.batch_size)
        for start in tqdm(batches, desc=f'extract/native{length}', mininterval=5):
            indices = positions[start:start+args.batch_size]
            x = np.array(raw[rows[indices], :length], copy=True)
            f, names = relational_features(x)
            if new is None:
                new = np.lib.format.open_memmap(output/'relations.npy', mode='w+', dtype='float32', shape=(len(rows), len(names)))
                write_json(output/'feature_names.json', {'existing': [f'existing32_{j}' for j in range(32)], 'relations': names})
            new[indices] = f.astype(np.float32)
            with torch.inference_mode():
                old[indices] = signal_statistics(torch.from_numpy(x), torch.full((len(x),), int(length))).numpy()
            log_rms[indices] = np.log(np.sqrt(np.square(x.astype(np.float64)).sum((1,2))/length))
    old.flush(); new.flush()
    np.save(output/'rows.npy', rows)
    np.save(output/'log_rms.npy', log_rms)
    write_json(output/'features_complete.json', {'dataset_fingerprint': info['fingerprint'], 'rows': len(rows),
        'existing_dimensions': 32, 'relational_dimensions': new.shape[1], 'feature_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


def selection(args, output, rows, meta):
    if (output/'probe_split.npz').exists():
        return dict(np.load(output/'probe_split.npz', allow_pickle=False))
    previous = np.load(args.reference/'selection.npz', allow_pickle=False)
    groups = previous['condition_groups'][rows, 0]
    labels = meta['modulation'][rows]
    rng = np.random.default_rng(SEED)
    fit, internal = [], []
    for label in range(23):
        for q in range(4):
            pool = np.flatnonzero((labels==label) & (groups==q) & (meta['split'][rows]==0))
            chosen = rng.choice(pool, 640, replace=False)
            fit.extend(chosen[:512]); internal.extend(chosen[512:])
    result = {'fit': np.sort(fit), 'internal': np.sort(internal),
              'validation': np.flatnonzero(meta['split'][rows]==1),
              'all_train': np.flatnonzero(meta['split'][rows]==0), 'rms_group': groups}
    assert not np.intersect1d(result['fit'], result['internal']).size
    assert not np.intersect1d(rows[result['fit']], rows[result['validation']]).size
    np.savez_compressed(output/'probe_split.npz', **result)
    return result


def measure(probability, labels, keep=None):
    if keep is None:
        keep = np.ones(len(labels), dtype=bool)
    p = probability[keep]; y = labels[keep]
    prediction = p.argmax(1)
    cm = np.bincount(y*23+prediction, minlength=23*23).reshape(23,23)
    result = classification_metrics(cm)
    supported = cm.sum(1) > 0
    result['macro_f1'] = float(np.asarray(result['f1'])[supported].mean())
    result['cross_entropy'] = float(-np.log(p[np.arange(len(y)),y].clip(1e-15)).mean())
    return result


def group_metrics(p, y, rms):
    groups = [('all23', np.ones(len(y), bool)), ('first14', y<14), ('remaining9', y>=14)]
    groups += [(f'first14_rms_q{q+1}', (y<14)&(rms==q)) for q in range(4)]
    return {name: measure(p,y,keep) for name,keep in groups}


def analyze_predictions(args, output, name, pred, rows, meta, split):
    reference = {}
    for base in ['pyramid21m','finetune96m']:
        data = np.load(args.reference/f'{base}_validation.npz', allow_pickle=False)
        indices = np.searchsorted(data['rows'],rows)
        assert np.array_equal(data['rows'][indices],rows)
        reference[base] = data['modulation'][indices].argmax(1)
    y = meta['modulation'][rows]; prediction = pred.argmax(1)
    common = (reference['pyramid21m']!=y)&(reference['finetune96m']!=y)
    result = {}
    for group, keep in [('all23',np.ones(len(y),bool)),('first14',y<14),('remaining9',y>=14)]:
        d = {'support':int(keep.sum()),'common_wrong_support':int((common&keep).sum()),
             'common_wrong_corrected':int((common&keep&(prediction==y)).sum())}
        for base, labels in reference.items():
            correct = labels==y
            d[base+'_corrected'] = int((keep&~correct&(prediction==y)).sum())
            d[base+'_damaged'] = int((keep&correct&(prediction!=y)).sum())
        result[group] = d
    record = pd.DataFrame({'row':rows,'sample_id':meta['sample_id'][rows].astype(str),'label':y,
        'prediction':prediction,'both_neural_wrong':common,'pyramid_prediction':reference['pyramid21m'],
        'finetune_prediction':reference['finetune96m']})
    record[common].to_csv(output/f'{name}_common_error_predictions.csv',index=False)
    return result


def fit_probes(args, output, meta, info):
    rows = np.load(output/'rows.npy')
    splits = selection(args, output, rows, meta)
    y = meta['modulation'][rows].astype(np.int64)
    old = np.load(output/'existing32.npy',mmap_mode='r')
    new = np.load(output/'relations.npy',mmap_mode='r')
    log_rms = np.load(output/'log_rms.npy')[:,None]
    union = np.concatenate([old,new],axis=1)
    features = {'existing32':old,'relations':new,'combined':union,
                'combined_energy':np.concatenate([union,log_rms],axis=1)}
    settings = dict(n_estimators=args.trees,min_samples_leaf=8,max_features='sqrt',n_jobs=args.threads,random_state=SEED)
    protocol = {'dataset_fingerprint':info['fingerprint'],'seed':SEED,'feature_sets':{k:v.shape[1] for k,v in features.items()},
        'fit_rows':len(splits['fit']),'internal_rows':len(splits['internal']),'validation_rows':len(splits['validation']),
        'classifier':'ExtraTreesClassifier','settings':settings,'selection':'highest internal first14 accuracy; validation never used for selection',
        'full_refit_gate':'>=1.5 percentage points internal first14 gain over existing32; then refit selected set on all train once',
        'reference_path':str(args.reference.resolve()),'cpu_only':True,
        'limits':['Independent probes, not a contest ensemble or proof of achievable end-to-end accuracy',
                  'Training split internal holdout is for the probes, not an unseen split for the pretrained neural models',
                  'Centred/normalized handcrafted views are diagnostic, not a preprocessing recommendation for all three tasks',
                  'True modulation labels and RMS quartiles are used for offline slices only; all models always predict 23 classes',
                  'Validation has been inspected in prior experiments; this is not an independent blind test']}
    write_json(output/'protocol.json',protocol)
    results = read_json(output/'results.json') if (output/'results.json').exists() else {}
    for name,x in features.items():
        if name in results:
            continue
        started=time.monotonic()
        model=ExtraTreesClassifier(**settings)
        print('FIT',name,'shape',x.shape,'train',len(splits['fit']),flush=True)
        model.fit(x[splits['fit']],y[splits['fit']])
        metrics={}
        for split in ['fit','internal','validation']:
            idx=splits[split];prob=model.predict_proba(x[idx]).astype(np.float32)
            np.savez_compressed(output/f'{name}_{split}.npz',rows=rows[idx],probability=prob)
            metrics[split]=group_metrics(prob,y[idx],splits['rms_group'][idx])
            if split=='validation':
                metrics['error_overlap']=analyze_predictions(args,output,name,prob,rows[idx],meta,split)
        metrics['elapsed_seconds']=time.monotonic()-started
        results[name]=metrics
        joblib.dump(model,output/f'{name}.joblib',compress=3)
        write_json(output/'results.json',results)
        print('DONE',name,{s:{g:round(v['accuracy'],5) for g,v in metrics[s].items()} for s in ['fit','internal','validation']},flush=True)
    # A different readout controls for dependence on the tree learner.
    if 'combined_linear' not in results:
        print('FIT combined_linear',flush=True)
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=400,solver='lbfgs',random_state=SEED))
        model.fit(union[splits['fit']],y[splits['fit']])
        iterations=int(model.named_steps['logisticregression'].n_iter_.max())
        metrics={'fit_diagnostics':{'iterations':iterations,'iteration_limit':400,'converged_before_limit':iterations<400}}
        for split in ['fit','internal','validation']:
            idx=splits[split];prob=model.predict_proba(union[idx]).astype(np.float32)
            np.savez_compressed(output/f'combined_linear_{split}.npz',rows=rows[idx],probability=prob)
            metrics[split]=group_metrics(prob,y[idx],splits['rms_group'][idx])
            if split=='validation':metrics['error_overlap']=analyze_predictions(args,output,'combined_linear',prob,rows[idx],meta,split)
        results['combined_linear']=metrics
        joblib.dump(model,output/'combined_linear.joblib',compress=3)
        write_json(output/'results.json',results)
    winner=max(features,key=lambda name:results[name]['internal']['first14']['accuracy'])
    gain=results[winner]['internal']['first14']['accuracy']-results['existing32']['internal']['first14']['accuracy']
    write_json(output/'internal_selection.json',{'winner':winner,'first14_gain':gain,'full_refit_gate_met':gain>=.015})
    if gain>=.015:
        for name, source in [('full_train_existing32','existing32'),('full_train_selected',winner)]:
            if name in results:
                continue
            print('FIT',name,source,'internal gain',gain,flush=True)
            model=ExtraTreesClassifier(**settings)
            x=features[source];model.fit(x[splits['all_train']],y[splits['all_train']])
            idx=splits['validation'];prob=model.predict_proba(x[idx]).astype(np.float32)
            metrics={'source_features':source,'validation':group_metrics(prob,y[idx],splits['rms_group'][idx]),
                'error_overlap':analyze_predictions(args,output,name,prob,rows[idx],meta,'validation')}
            np.savez_compressed(output/f'{name}_validation.npz',rows=rows[idx],probability=prob)
            joblib.dump(model,output/f'{name}.joblib',compress=3)
            results[name]=metrics
            write_json(output/'results.json',results)
    print('INTERNAL SELECTION',winner,gain,flush=True)


def self_test():
    rng=np.random.default_rng(SEED)
    iq=rng.normal(size=(3,512,2))
    baseline,names=relational_features(iq)
    c,s=np.cos(.713),np.sin(.713)
    rotated=np.stack([iq[...,0]*c-iq[...,1]*s,iq[...,0]*s+iq[...,1]*c],axis=-1)
    np.testing.assert_allclose(baseline,relational_features(rotated)[0],atol=2e-10,rtol=2e-9)
    np.testing.assert_allclose(baseline,relational_features(iq*4)[0],atol=2e-10,rtol=2e-9)
    padded=rng.normal(size=(3,1024,2));padded[:,:512]=iq
    np.testing.assert_allclose(baseline,relational_features(padded,np.full(3,512))[0],atol=1e-12)
    assert len(names)==len(set(names)) and np.isfinite(baseline).all()
    x=rng.normal(size=(512,2));sequence=np.tile(x[None],(2,1,1));sequence[1]=sequence[1,rng.permutation(512)]
    f,_=relational_features(sequence)
    assert np.max(np.abs(f[0]-f[1]))>1e-3
    print('Self-test passed:',len(names),'features; positive-gain/global-phase invariance, effective length, order response.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'artifacts/data_full')
    p.add_argument('--reference',type=Path,default=ROOT/'artifacts/c01_followup_validation/20261113_224434')
    p.add_argument('--output',type=Path)
    p.add_argument('--stage',choices=['all','extract','probe'],default='all')
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--trees',type=int,default=128)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    if args.self_test:
        self_test();return
    output=args.output or ROOT/'artifacts/c01_relation_probe'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=True)
    print('OUTPUT',output,flush=True)
    info=read_json(args.data/'dataset.json')
    if info['fingerprint']!=read_json(args.reference/'protocol.json')['dataset_fingerprint']:
        raise ValueError('Reference/data fingerprint mismatch')
    with np.load(args.data/'metadata.npz',allow_pickle=False) as f:meta={k:f[k] for k in f.files}
    with threadpool_limits(limits=args.threads):
        if args.stage in ['all','extract']:extract(args,output,meta,info)
        if args.stage in ['all','probe']:fit_probes(args,output,meta,info)


if __name__=='__main__':
    main()
