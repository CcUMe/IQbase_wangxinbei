"""Publication-style plots from analyze_c01_distribution.py's numeric outputs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=Path('artifacts/data_full'))
    args=parser.parse_args();out=args.analysis
    summary=json.loads((out/'summary.json').read_text())
    schema=json.loads((args.cache/'dataset.json').read_text())['schema']
    pa.set_cpu_count(4)
    frame=pq.read_table(out/'features.parquet',columns=['task','modulation_label','protocol_label','length',
             'log_rms','lag1_coherence','spectral_flatness','high_frequency_power_fraction',
             'normalized_power_fourth_moment','symbol_rate','frequency_shift','number_of_pulses',
             'symbol_rate_valid','frequency_shift_valid','number_of_pulses_valid']).to_pandas()
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(15,9),gridspec_kw={'height_ratios':[1.4,1]})
    for ax,task,column in [(axes[0],'modulation','modulation_label'),(axes[1],'protocol','protocol_label')]:
        labels=schema['classification_labels'][task];stats=[]
        for label in labels:
            q=summary['by_class'][f'{task}:{label}']['features']['log_rms']
            stats.append(dict(label=label.replace('Radar/','R/'),whislo=q['0.01']/np.log(10),q1=q['0.25']/np.log(10),
                              med=q['0.5']/np.log(10),q3=q['0.75']/np.log(10),whishi=q['0.99']/np.log(10),fliers=[]))
        boxes=ax.bxp(stats,showfliers=False,patch_artist=True)
        for i,box in enumerate(boxes['boxes']):box.set_facecolor('#5886b7' if task=='modulation' and i<14 else '#d99b58')
        ax.set_ylabel('log10(complex RMS)');ax.tick_params(axis='x',rotation=55)
        ax.set_title(f'{task}: every original record; boxes=IQR, whiskers=1%-99%')
        ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'amplitude_by_class.png',dpi=170);fig.savefig(out/'amplitude_by_class.pdf');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(15,4.4))
    tasks=['modulation','protocol','parameter']
    axes[0].bar(tasks,[summary['task_counts'][x] for x in tasks],color=['#5886b7','#d99b58','#6aa785'])
    axes[0].set_ylabel('Original records');axes[0].set_title('Task record counts')
    axes[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    for ax,target in [(axes[1],'symbol_rate'),(axes[2],'frequency_shift')]:
        counts=summary['parameter_distributions'][target]['value_counts'];values=list(counts)
        ax.bar(range(len(values)),[counts[x] for x in values],color='#5886b7')
        ax.set_xticks(range(len(values)),[f'{float(x):g}' for x in values],rotation=60)
        ax.set_ylabel('Applicable records');ax.set_title(target+' labels (original units)')
    fig.tight_layout();fig.savefig(out/'task_and_parameter_counts.png',dpi=170);plt.close(fig)

    rng=np.random.default_rng(20261113)
    hard=frame[(frame.task=='modulation')&frame.modulation_label.isin(schema['classification_labels']['modulation'][:14])]
    selected=hard.iloc[rng.choice(len(hard),min(16000,len(hard)),replace=False)]
    fig,axes=plt.subplots(1,3,figsize=(15,4.4))
    for ax,f in zip(axes,['lag1_coherence','spectral_flatness','high_frequency_power_fraction']):
        ax.hexbin(selected.log_rms/np.log(10),selected[f],gridsize=55,mincnt=1,bins='log',cmap='viridis')
        ax.set_xlabel('log10(complex RMS)');ax.set_ylabel(f)
    fig.suptitle('First 14 modulation classes: observable conditions; these are NOT SNR measurements')
    fig.tight_layout();fig.savefig(out/'first14_observable_conditions.png',dpi=170);plt.close(fig)

    # Compare class shape medians after removing between-feature units.
    features=['lag1_coherence','lag4_coherence','spectral_flatness','high_frequency_power_fraction',
              'normalized_power_fourth_moment','dc_fraction','iq_power_balance']
    labels=schema['classification_labels']['modulation']
    values=np.array([[summary['by_class']['modulation:'+label]['features'][f]['0.5'] for f in features] for label in labels])
    normalized=(values-values.mean(0))/values.std(0).clip(1e-12)
    fig,ax=plt.subplots(figsize=(11,8));image=ax.imshow(normalized,aspect='auto',cmap='coolwarm',vmin=-2.5,vmax=2.5)
    ax.set_yticks(range(len(labels)),[x.replace('Radar/','R/') for x in labels]);ax.set_xticks(range(len(features)),features,rotation=35,ha='right')
    ax.axhline(13.5,color='black',lw=1);ax.set_title('Class medians of scale-invariant shape statistics (column standardized)')
    fig.colorbar(image,ax=ax,label='Standardized class median');fig.tight_layout();fig.savefig(out/'shape_by_class.png',dpi=170);plt.close(fig)
    shifts=pd.DataFrame(summary['split_shifts'])
    shifts.groupby(['task','feature']).ks_distance.agg(['median','max']).to_csv(out/'split_shift_summary.csv')
    if 'quantization' in summary:
        fig,axes=plt.subplots(2,1,figsize=(14,8))
        labels=schema['classification_labels']['modulation']
        q=[summary['quantization']['modulation:'+label] for label in labels]
        axes[0].bar(range(23),[v['exact_lattice_fraction']*100 for v in q],color=['#5886b7']*14+['#d99b58']*9)
        axes[0].set_xticks(range(23),[x.replace('Radar/','R/') for x in labels],rotation=55,ha='right')
        axes[0].set_ylabel('Entire waveform on grid (%)')
        axes[0].set_title('Every original I/Q component checked against the observed 2^-15 grid')
        med=[];low=[];high=[]
        for v in q[:14]:
            stat=v['normalized_step_for_lattice_rows'];med.append(stat['0.5']);low.append(stat['0.01']);high.append(stat['0.99'])
        axes[1].errorbar(range(14),med,yerr=[np.array(med)-low,np.array(high)-med],fmt='o',capsize=4,color='#5886b7')
        axes[1].set_xticks(range(14),labels[:14],rotation=45,ha='right');axes[1].set_yscale('log')
        axes[1].set_ylabel('Observed grid step / record RMS')
        axes[1].set_title('RMS normalization rescales the grid; it cannot create unobserved sub-grid samples')
        axes[1].grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'numeric_grid_distribution.png',dpi=170)
        fig.savefig(out/'numeric_grid_distribution.pdf');plt.close(fig)
    print('saved plots and split_shift_summary.csv',out,flush=True)


if __name__=='__main__':main()
