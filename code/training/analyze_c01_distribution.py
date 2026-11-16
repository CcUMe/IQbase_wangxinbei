"""Stream all original C01 waveforms and describe task/label/split distributions.

No source/cache/model mutation. Frequencies are cycles/sample, not Hz; waveform
statistics are observations, not measured SNR or a reconstruction of generation
parameters. Run extraction once, then reuse features.parquet for summaries.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ['symbol_rate', 'bandwidth', 'frequency_shift', 'pulse_width', 'pri', 'time_delay', 'number_of_pulses']
QUANTILES = [0., .01, .1, .25, .5, .75, .9, .99, 1.]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def quantiles(values):
    values = np.asarray(values)
    return {str(q): float(v) for q, v in zip(QUANTILES, np.quantile(values, QUANTILES))}


def describe_waveforms(x):
    """x[B,L,2], exact raw float32 input; Float64 accumulated diagnostics."""
    x = x.astype(np.float64)
    i, q = x[..., 0], x[..., 1]
    power = i*i + q*q
    energy = power.mean(1)
    if not np.isfinite(x).all() or np.any(energy <= 0):
        raise ValueError('Nonfinite or zero-energy original waveform encountered')
    rms = np.sqrt(energy)
    mi, mq = i.mean(1), q.mean(1)
    ci, cq = i - mi[:, None], q - mq[:, None]
    vi, vq = (ci*ci).mean(1), (cq*cq).mean(1)
    z = ci + 1j*cq
    centered = np.abs(z)**2
    normalized_power = power / energy[:, None]
    amplitude = np.sqrt(normalized_power)
    spectrum = np.abs(np.fft.fft(z, axis=1))**2
    spectrum_sum = spectrum.sum(1, keepdims=True)
    p = spectrum / spectrum_sum.clip(1e-300)
    length = x.shape[1]
    frequency = np.fft.fftfreq(length)
    result = dict(rms=rms, log_rms=np.log(rms),
                  rms_i=np.sqrt((i*i).mean(1)), rms_q=np.sqrt((q*q).mean(1)),
                  iq_power_balance=(i*i).mean(1)/energy,
                  iq_correlation=(ci*cq).mean(1)/np.sqrt(vi*vq).clip(1e-300),
                  dc_fraction=(mi*mi+mq*mq)/energy,
                  centered_energy_fraction=(vi+vq)/energy,
                  crest_factor=np.sqrt(normalized_power.max(1)),
                  normalized_power_fourth_moment=(normalized_power**2).mean(1),
                  amplitude_cv=amplitude.std(1)/amplitude.mean(1).clip(1e-300),
                  normalized_amplitude_p10=np.quantile(amplitude,.1,axis=1),
                  normalized_amplitude_p50=np.quantile(amplitude,.5,axis=1),
                  normalized_amplitude_p90=np.quantile(amplitude,.9,axis=1),
                  zero_component_fraction=(x==0).mean((1,2)),
                  spectral_flatness=np.exp(np.log((p*length).clip(1e-12)).mean(1)),
                  spectral_entropy=-(p*np.log(p.clip(1e-300))).sum(1)/np.log(length),
                  spectral_effective_bin_fraction=np.where(spectrum_sum[:,0]>0,1/(p*p).sum(1).clip(1e-300)/length,0.),
                  high_frequency_power_fraction=p[:,np.abs(frequency)>=.25].sum(1),
                  spectral_peak_frequency=frequency[p.argmax(1)],
                  spectral_centroid=(p*frequency).sum(1))
    for lag in [1,4,16,64]:
        cross=(z[:,lag:]*z[:,:-lag].conj()).mean(1)
        norm=np.sqrt(centered[:,lag:].mean(1)*centered[:,:-lag].mean(1)).clip(1e-300)
        result[f'lag{lag}_coherence']=np.abs(cross)/norm
        if lag==1:result['lag1_phase_cycles']=np.angle(cross)/(2*np.pi)
    return result


def cache_checks(cache):
    trace=pq.read_table(cache/'split_manifest.parquet',columns=['source_file','source_row'])
    with np.load(cache/'metadata.npz',allow_pickle=False) as meta:
        validity=meta['target_valid']
        groups=[meta['modulation'],meta['protocol'],np.where(validity.any(1),np.where(validity[:,0],0,1),-1)]
        split=meta['split']
        rng=np.random.default_rng(20261113);selected=[]
        for values in groups:
            for label in np.unique(values[values>=0]):
                for part in [0,1]:
                    choices=np.flatnonzero((values==label)&(split==part))
                    if len(choices):selected.extend(rng.choice(choices,min(2,len(choices)),replace=False).tolist())
    checks=defaultdict(dict)
    for row in sorted(set(selected)):
        checks[trace['source_file'][row].as_py()][trace['source_row'][row].as_py()]=row
    return checks,np.load(cache/'iq.npy',mmap_mode='r')


def extract(args):
    output=args.output
    if (output/'features.parquet').exists():
        raise FileExistsError('features.parquet exists; use --phase summarize to reuse it')
    checks,cache=cache_checks(args.cache)
    completed_checks=[]
    quality=Counter();invariance=defaultdict(dict);invariance_counts=Counter()
    schema=json.loads((args.data_root/'rf_signal_benchmark_train_v1.0/task_schema.json').read_text())
    files=sorted((args.data_root/'rf_signal_benchmark_train_v1.0').glob('*.parquet'))
    writer=None;total=0;started=time.monotonic();file_info=[]
    try:
        for file in files:
            pf=pq.ParquetFile(file);offset=0
            file_info.append(dict(file=file.name,rows=pf.metadata.num_rows,bytes=file.stat().st_size))
            for batch in pf.iter_batches(batch_size=args.batch_size):
                n=batch.num_rows
                fields={name:batch.column(batch.schema.get_field_index(name)) for name in batch.schema.names}
                iq=fields['iq'];outer=iq.offsets.to_numpy();points=iq.values
                inner=points.offsets.to_numpy();flat=points.values.to_numpy(zero_copy_only=False)
                lengths=np.diff(outer)
                quality['null_iq_rows']+=iq.null_count
                quality['null_iq_points']+=points.null_count
                quality['null_iq_values']+=points.values.null_count
                quality['malformed_points']+=int((np.diff(inner)!=2).sum())
                quality['unexpected_lengths']+=int((~np.isin(lengths,[512,1024])).sum())
                if any(quality[k] for k in ['null_iq_rows','null_iq_points','null_iq_values','malformed_points','unexpected_lengths']):
                    raise ValueError(f'Malformed IQ near {file.name}:{offset}')
                tasks=fields['task'].to_pylist()
                ml=fields['modulation_label'].to_pylist();pl=fields['protocol_label'].to_pylist()
                group=np.array([f'{t}:{m or p or ("communication" if length==1024 else "radar")}'
                                for t,m,p,length in zip(tasks,ml,pl,lengths)])
                feature_arrays={}
                for length in sorted(np.unique(lengths)):
                    selected=np.flatnonzero(lengths==length)
                    starts=inner[outer[selected]]
                    indices=starts[:,None,None]+np.arange(length)[None,:,None]*2+np.arange(2)[None,None,:]
                    x=flat[indices]
                    quality['nonfinite_rows']+=int((~np.isfinite(x).all((1,2))).sum())
                    quality['zero_energy_rows']+=int((~np.any(x!=0,axis=(1,2))).sum())
                    values=describe_waveforms(x)
                    for name,v in values.items():
                        feature_arrays.setdefault(name,np.empty(n,dtype=np.float64))[selected]=v
                    for local in selected:
                        original_row=offset+int(local)
                        if original_row in checks[file.name]:
                            cache_row=checks[file.name][original_row]
                            raw=flat[inner[outer[local]]:inner[outer[local+1]]].reshape(length,2)
                            if not np.array_equal(raw,cache[cache_row,:length]):
                                raise ValueError(f'Cache differs from original at {file.name}:{original_row}')
                            if np.any(cache[cache_row,length:]!=0):raise ValueError('Nonzero cache padding')
                            completed_checks.append(dict(source_file=file.name,source_row=original_row,cache_row=cache_row))
                    for key in np.unique(group[selected]):
                        positions=np.flatnonzero(group[selected]==key)[:max(0,8-invariance_counts[key])]
                        if not len(positions):continue
                        raw=x[positions];r=values['rms'][positions]
                        normalized=(raw/r[:,None,None]).astype(np.float32)
                        after=describe_waveforms(normalized)
                        entry=invariance[key]
                        entry['unit_rms_max_error']=max(entry.get('unit_rms_max_error',0.),float(np.max(np.abs(after['rms']-1))))
                        for name in ['crest_factor','dc_fraction','lag1_coherence','lag4_coherence','spectral_flatness','high_frequency_power_fraction','normalized_power_fourth_moment']:
                            error=float(np.max(np.abs(after[name]-values[name][positions])))
                            entry[name+'_max_abs_change']=max(entry.get(name+'_max_abs_change',0.),error)
                        invariance_counts[key]+=len(positions)
                columns={name:fields[name] for name in ['task','sample_id','modulation_label','protocol_label',*TARGETS,*[x+'_valid' for x in TARGETS]]}
                columns.update(source_file=pa.array([file.name]*n,type=pa.string()),
                               source_row=pa.array(np.arange(offset,offset+n)),length=pa.array(lengths.astype(np.int32)))
                columns.update({name:pa.array(v,type=pa.float64()) for name,v in feature_arrays.items()})
                table=pa.table(columns)
                if writer is None:writer=pq.ParquetWriter(output/'features.parquet',table.schema,compression='zstd')
                writer.write_table(table)
                offset+=n;total+=n
                if total//50000!=(total-n)//50000:
                    print(f'original rows={total:,} | {file.name} | elapsed={time.monotonic()-started:.1f}s',flush=True)
            print(f'finished {file.name} | rows={offset:,}',flush=True)
    finally:
        if writer:writer.close()
    assert len(completed_checks)==sum(map(len,checks.values()))
    write_json(output/'extraction.json',dict(rows=total,files=file_info,quality=dict(quality),
               cache_exact_checks=completed_checks,invariance_counts=dict(invariance_counts),
               rms_invariance=dict(invariance),elapsed_seconds=time.monotonic()-started,
               units='RMS uncalibrated amplitude; frequencies cycles/sample; lags samples; no measured SNR',
               source='All original Parquet IQ rows, not cached IQ for feature extraction'))
    print(f'extraction complete | rows={total:,}',flush=True)


def summarize(args):
    from scipy.stats import ks_2samp,rankdata
    from sklearn.tree import DecisionTreeClassifier
    import pandas as pd
    source=pq.read_table(args.output/'features.parquet')
    temporal=args.output/'temporal_features.parquet'
    if temporal.exists():source=source.join(pq.read_table(temporal),keys='sample_id',join_type='left outer')
    quantization=args.output/'quantization_features.parquet'
    if quantization.exists():source=source.join(pq.read_table(quantization),keys='sample_id',join_type='left outer')
    split=pq.read_table(args.cache/'split_manifest.parquet',columns=['sample_id','split'])
    split=split.append_column('cache_row',pa.array(np.arange(split.num_rows)))
    frame=source.join(split,keys='sample_id',join_type='left outer').to_pandas()
    print('joined original features and existing split',len(frame),flush=True)
    if len(frame)!=source.num_rows or frame['sample_id'].duplicated().any():raise ValueError('Join duplicated records')
    feature_names=[x for x in source.schema.names if x not in ['task','sample_id','modulation_label','protocol_label',*TARGETS,*[x+'_valid' for x in TARGETS],'source_file','source_row','length']]
    key_features=['log_rms','dc_fraction','iq_power_balance','iq_correlation','crest_factor','normalized_power_fourth_moment','lag1_coherence','lag4_coherence','lag16_coherence','lag64_coherence','spectral_flatness','spectral_entropy','high_frequency_power_fraction','spectral_centroid']
    temporal_columns=[c for c in ['relative_energy_centroid','relative_energy_spread','relative_peak_position',
                                  'zero_complex_fraction','leading_zero_points','trailing_zero_points',
                                  *[f'time_energy_bin_{i:02d}' for i in range(16)]] if c in source.schema.names]
    key_features+=temporal_columns
    description=dict(rows=len(frame),excluded_from_cache=int(frame['split'].isna().sum()),
                     task_counts={str(k):int(v) for k,v in frame['task'].value_counts().items()},
                     feature_definitions='Full native-window statistics. Per-record normalization uses complex RMS. No physical fs/SNR is inferred.',
                     by_task={},by_class={},split_shifts=[],probes={},parameter_distributions={},parameter_correlations={},parameter_condition_groups={})
    for task,df in frame.groupby('task'):
        description['by_task'][task]={'rows':len(df),'features':{f:quantiles(df[f]) for f in feature_names},
                                    'lengths':{str(k):int(v) for k,v in df['length'].value_counts().items()}}
    classes=[]
    for task,column in [('modulation','modulation_label'),('protocol','protocol_label')]:
        task_frame=frame[frame['task']==task]
        for label,df in task_frame.groupby(column):
            description['by_class'][f'{task}:{label}']={'rows':len(df),'features':{f:quantiles(df[f]) for f in feature_names}}
            row=dict(task=task,label=label,rows=len(df),rms_q01=df.rms.quantile(.01),rms_median=df.rms.median(),rms_q99=df.rms.quantile(.99))
            row.update({f+'_median':float(df[f].median()) for f in key_features if f!='log_rms'})
            classes.append(row)
        # A bin-density classifier uses only train quantile boundaries/counts.
        names=sorted(task_frame[column].unique());y=task_frame[column].map({x:i for i,x in enumerate(names)}).to_numpy()
        train=task_frame['split'].to_numpy()==0;val=task_frame['split'].to_numpy()==1
        def histogram_probe(x,y,train,val,names,bins=64):
            edges=np.unique(np.quantile(x[train],np.linspace(0,1,bins+1)))[1:-1]
            index=np.searchsorted(edges,x,side='right');counts=np.ones((len(edges)+1,len(names)))
            np.add.at(counts,(index[train],y[train]),1)
            pred=counts[index[val]].argmax(1);cm=np.bincount(y[val]*len(names)+pred,minlength=len(names)**2).reshape(len(names),len(names))
            return {'validation_accuracy':float(np.mean(pred==y[val])),'train_rows':int(train.sum()),'validation_rows':int(val.sum()),'labels':names,'confusion_matrix':cm.tolist(),'bins':len(edges)+1}
        description['probes'][task+'_energy_only']=histogram_probe(task_frame['log_rms'].to_numpy(),y,train,val,names)
        shape=['dc_fraction','iq_power_balance','iq_correlation','crest_factor','normalized_power_fourth_moment','amplitude_cv','lag1_coherence','lag4_coherence','lag16_coherence','lag64_coherence','spectral_flatness','spectral_entropy','high_frequency_power_fraction','spectral_centroid']+temporal_columns
        for name,columns in [('shape_only',shape),('shape_plus_energy',shape+['log_rms'])]:
            tree=DecisionTreeClassifier(max_depth=6,min_samples_leaf=128,random_state=2026)
            tree.fit(task_frame.loc[train,columns],y[train]);pred=tree.predict(task_frame.loc[val,columns])
            description['probes'][task+'_'+name]=dict(validation_accuracy=float(np.mean(pred==y[val])),features=columns,
                 train_rows=int(train.sum()),validation_rows=int(val.sum()),note='One fixed shallow CPU tree; not a neural baseline or a final score.')
        if task=='modulation':
            schema=json.loads((args.cache/'dataset.json').read_text())['schema']['classification_labels']['modulation']
            keep=task_frame[column].isin(schema[:14]).to_numpy();xx=task_frame['log_rms'].to_numpy()
            yy=task_frame[column].map({x:i for i,x in enumerate(schema[:14])}).fillna(-1).to_numpy().astype(int)
            description['probes']['first14_energy_only']=histogram_probe(xx[keep],yy[keep],train[keep],val[keep],schema[:14])
            description['probes']['first14_vs_remaining9_energy']=histogram_probe(xx,(~keep).astype(int),train,val,['first14','remaining9'])
    pd.DataFrame(classes).to_csv(args.output/'class_features.csv',index=False)
    shift_groups=[]
    for task,column in [('modulation','modulation_label'),('protocol','protocol_label')]:
        for label,df in frame[frame.task==task].groupby(column):shift_groups.append((task,label,df))
    for length,df in frame[frame.task=='parameter'].groupby('length'):shift_groups.append(('parameter',str(length),df))
    for task,label,df in shift_groups:
        train=df['split']==0;val=df['split']==1
        for f in key_features:
            a,b=df.loc[train,f].to_numpy(),df.loc[val,f].to_numpy()
            description['split_shifts'].append(dict(task=task,label=label,feature=f,train_rows=len(a),validation_rows=len(b),
                    ks_distance=float(ks_2samp(a,b,method='asymp').statistic),train_median=float(np.median(a)),validation_median=float(np.median(b))))
    pd.DataFrame(description['split_shifts']).to_csv(args.output/'split_shifts.csv',index=False)
    parameters=frame[frame.task=='parameter']
    for target in TARGETS:
        df=parameters[parameters[target+'_valid']];v=df[target]
        counts=v.value_counts()
        entry={'rows':len(df),'unique_values':len(counts),'quantiles':quantiles(v),
               'train_unique_values':int(df.loc[df['split']==0,target].nunique()),
               'validation_unique_values':int(df.loc[df['split']==1,target].nunique())}
        if len(counts)<=64:entry['value_counts']={str(k):int(v) for k,v in counts.sort_index().items()}
        description['parameter_distributions'][target]=entry
    for name,fields in [('communication',TARGETS[:3]),('radar',TARGETS[3:])]:
        df=parameters[parameters[fields[0]+'_valid']]
        joint=df.groupby(fields,dropna=False).size().sort_values(ascending=False)
        train_set=set(map(tuple,df.loc[df['split']==0,fields].to_numpy()))
        val_rows=df.loc[df['split']==1,fields].to_numpy()
        description.setdefault('parameter_joint',{})[name]=dict(unique_combinations=len(joint),
             most_common=[dict(values=list(index),rows=int(value)) for index,value in joint.head(12).items()],
             validation_combinations_absent_from_train=int(sum(tuple(row) not in train_set for row in val_rows)),
             note='Exact floating tuples for radar need not repeat; absence is not necessarily OOD.')
        columns=fields+key_features
        ranks=np.column_stack([rankdata(df[f].to_numpy()) for f in columns]);ranks-=ranks.mean(0);ranks/=np.linalg.norm(ranks,axis=0).clip(1e-300)
        correlations=ranks.T@ranks
        description['parameter_correlations'][name]={target:{feature:float(correlations[j,len(fields)+k]) for k,feature in enumerate(key_features)} for j,target in enumerate(fields)}
    for target in ['symbol_rate','frequency_shift','number_of_pulses']:
        df=parameters[parameters[target+'_valid']]
        description['parameter_condition_groups'][target]=[{target:float(value),'rows':len(g),
                    **{f+'_median':float(g[f].median()) for f in ['rms','lag1_coherence','spectral_flatness','spectral_centroid']}}
                    for value,g in df.groupby(target)]
    radar=parameters[parameters['pulse_width_valid']]
    description['radar_pri_above_width_fraction']=float((radar.pri>radar.pulse_width).mean())
    description['near_zero_counts']={str(eps):int((frame.rms<=eps).sum()) for eps in [1e-12,1e-6,1e-4,1e-3]}
    if 'grid_2p15_exact_waveform' in frame:
        description['quantization']={}
        for (task,label),df in frame.assign(group=np.where(frame.task=='modulation',frame.modulation_label,
                     np.where(frame.task=='protocol',frame.protocol_label,np.where(frame.length==1024,'communication','radar')))).groupby(['task','group']):
            exact=df.grid_2p15_exact_waveform==1
            description['quantization'][f'{task}:{label}']=dict(rows=len(df),exact_lattice_rows=int(exact.sum()),
                exact_lattice_fraction=float(exact.mean()),component_grid_fraction_quantiles=quantiles(df.grid_2p15_component_fraction),
                normalized_step_for_lattice_rows=quantiles(df.loc[exact,'normalized_2p15_step']) if exact.any() else None)
    description['source_file_tasks']=frame.groupby(['source_file','task']).size().reset_index(name='rows').to_dict('records')
    mod_names=json.loads((args.cache/'dataset.json').read_text())['schema']['classification_labels']['modulation']
    hard=frame[(frame.task=='modulation')&frame.modulation_label.isin(mod_names[:14])].copy()
    hard['rms_quartile']=-1
    for label in mod_names[:14]:
        keep=hard.modulation_label==label
        edges=np.quantile(hard.loc[keep&(hard['split']==0),'rms'],[.25,.5,.75])
        hard.loc[keep,'rms_quartile']=np.searchsorted(edges,hard.loc[keep,'rms'],side='right')
    description['first14_condition_slices']={}
    for split,name in [(0,'train'),(1,'validation')]:
        description['first14_condition_slices'][name]=[
            {'rms_quartile':int(q),'rows':len(g),**{f+'_median':float(g[f].median()) for f in
                ['rms','lag1_coherence','spectral_flatness','high_frequency_power_fraction','normalized_power_fourth_moment']}}
            for q,g in hard[hard['split']==split].groupby('rms_quartile')]
    if 'grid_2p15_exact_waveform' in frame:
        comm=parameters[parameters.frequency_shift_valid]
        description['frequency_shift_grid']=[{'frequency_shift':float(v),'rows':len(g),
                    'exact_grid_rows':int(g.grid_2p15_exact_waveform.sum())} for v,g in comm.groupby('frequency_shift')]
    write_json(args.output/'summary.json',description)
    print('summary complete',args.output/'summary.json',flush=True)


def extract_temporal(args):
    """Second read of raw data: time/zero distributions without FFT work."""
    path=args.output/'temporal_features.parquet'
    if path.exists():raise FileExistsError(path)
    writer=None;total=0;started=time.monotonic()
    try:
        for file in sorted((args.data_root/'rf_signal_benchmark_train_v1.0').glob('*.parquet')):
            for batch in pq.ParquetFile(file).iter_batches(batch_size=args.batch_size,columns=['sample_id','iq']):
                iq=batch.column(1);outer=iq.offsets.to_numpy();points=iq.values
                inner=points.offsets.to_numpy();flat=points.values.to_numpy(zero_copy_only=False)
                lengths=np.diff(outer);arrays={}
                for length in np.unique(lengths):
                    selected=np.flatnonzero(lengths==length);starts=inner[outer[selected]]
                    indices=starts[:,None,None]+np.arange(length)[None,:,None]*2+np.arange(2)[None,None,:]
                    x=flat[indices];power=np.square(x.astype(np.float64)).sum(-1)
                    probability=power/power.sum(1,keepdims=True)
                    time_coordinate=np.linspace(0,1,int(length))
                    center=(probability*time_coordinate).sum(1)
                    nonzero=np.any(x!=0,axis=-1)
                    values=dict(zero_i_fraction=(x[...,0]==0).mean(1),zero_q_fraction=(x[...,1]==0).mean(1),
                                zero_complex_fraction=(~nonzero).mean(1),leading_zero_points=nonzero.argmax(1),
                                trailing_zero_points=nonzero[:,::-1].argmax(1),relative_energy_centroid=center,
                                relative_energy_spread=np.sqrt((probability*(time_coordinate-center[:,None])**2).sum(1)),
                                relative_peak_position=power.argmax(1)/(int(length)-1))
                    windows=probability.reshape(len(x),16,int(length)//16).sum(-1)
                    values.update({f'time_energy_bin_{i:02d}':windows[:,i] for i in range(16)})
                    for name,v in values.items():arrays.setdefault(name,np.empty(len(lengths),dtype=np.float64))[selected]=v
                table=pa.table({'sample_id':batch.column(0),**{k:pa.array(v) for k,v in arrays.items()}})
                if writer is None:writer=pq.ParquetWriter(path,table.schema,compression='zstd')
                writer.write_table(table);total+=batch.num_rows
            print(f'temporal original rows={total:,} | {file.name} | elapsed={time.monotonic()-started:.1f}s',flush=True)
    finally:
        if writer:writer.close()


def extract_quantization(args):
    """Verify the observed 2^-15 lattice on every raw waveform, not just samples."""
    path=args.output/'quantization_features.parquet'
    if path.exists():raise FileExistsError(path)
    writer=None;total=0;started=time.monotonic();examples=defaultdict(list)
    try:
        for file in sorted((args.data_root/'rf_signal_benchmark_train_v1.0').glob('*.parquet')):
            offset=0
            columns=['sample_id','task','modulation_label','protocol_label','iq']
            for batch in pq.ParquetFile(file).iter_batches(batch_size=args.batch_size,columns=columns):
                fields={name:batch.column(batch.schema.get_field_index(name)) for name in columns}
                iq=fields['iq'];outer=iq.offsets.to_numpy();points=iq.values
                inner=points.offsets.to_numpy();flat=points.values.to_numpy(zero_copy_only=False);lengths=np.diff(outer)
                labels=[f'{t}:{m or p or ("communication" if n==1024 else "radar")}' for t,m,p,n in zip(
                    fields['task'].to_pylist(),fields['modulation_label'].to_pylist(),fields['protocol_label'].to_pylist(),lengths)]
                arrays={}
                for length in np.unique(lengths):
                    selected=np.flatnonzero(lengths==length);starts=inner[outer[selected]]
                    indices=starts[:,None,None]+np.arange(length)[None,:,None]*2+np.arange(2)[None,None,:]
                    x=flat[indices].astype(np.float64);scaled=x*32768
                    on_grid=scaled==np.rint(scaled)
                    rms=np.sqrt(np.square(x).sum(-1).mean(1))
                    values=dict(grid_2p15_component_fraction=on_grid.mean((1,2)),
                                grid_2p15_exact_waveform=on_grid.all((1,2)).astype(np.float64),
                                rms_in_2p15_steps=rms*32768,normalized_2p15_step=1/(rms*32768))
                    for name,v in values.items():arrays.setdefault(name,np.empty(len(lengths),dtype=np.float64))[selected]=v
                    for j,local in enumerate(selected):
                        key=labels[local]
                        if len(examples[key])>=8:continue
                        unique=np.unique(x[j]);delta=np.diff(unique);positive=delta[delta>0]
                        examples[key].append(dict(sample_id=fields['sample_id'][int(local)].as_py(),source_file=file.name,
                            source_row=offset+int(local),rms=float(rms[j]),unique_component_values=len(unique),
                            min_observed_step=float(positive.min()) if len(positive) else None,
                            exact_2p15_lattice=bool(on_grid[j].all()),value_excerpt=unique[:12].tolist()))
                table=pa.table({'sample_id':fields['sample_id'],**{key:pa.array(v) for key,v in arrays.items()}})
                if writer is None:writer=pq.ParquetWriter(path,table.schema,compression='zstd')
                writer.write_table(table);total+=batch.num_rows;offset+=batch.num_rows
            print(f'quantization original rows={total:,} | {file.name} | elapsed={time.monotonic()-started:.1f}s',flush=True)
    finally:
        if writer:writer.close()
    write_json(args.output/'quantization_examples.json',dict(step=1/32768,rows=total,examples=dict(examples),
               limit='A numeric lattice is observed; ADC hardware, bit depth, generation origin and quantization timing are not identified.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('/mnt/disk1/wangxinbei_contest_dataset/C01'))
    parser.add_argument('--cache',type=Path,default=ROOT/'artifacts/data_full')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--phase',choices=['extract','temporal','quantization','summarize','all'],default='all')
    args=parser.parse_args()
    if args.batch_size<1:parser.error('batch-size must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    pa.set_cpu_count(4);pa.set_io_thread_count(2)
    if args.phase in ['extract','all']:extract(args)
    if args.phase in ['temporal','all']:extract_temporal(args)
    if args.phase in ['quantization','all']:extract_quantization(args)
    if args.phase in ['summarize','all']:summarize(args)


if __name__=='__main__':main()
