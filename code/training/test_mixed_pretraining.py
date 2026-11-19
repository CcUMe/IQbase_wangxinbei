"""Small real-IQ CPU/Gloo workflow: mixed pretraining -> pure-C01 adaptation.

Does not launch CUDA training or estimate C01 accuracy improvements.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import torch
from training.prepare_mixed_pretrain import prepare,RADIO_LABELS
from utils.common import read_json,write_json
from utils.config import load_config,model_options
from utils.dataset import PreparedDataset,collate
from utils.multisource import MultiSourceBatchSampler,validate_backbone_data_contract
from utils.run_directory import latest_run
from utils.training_workflow import stamp

ROOT=Path(__file__).resolve().parents[2]


def main():
    torch.set_num_threads(2)
    output=ROOT/'artifacts/mixed_pretrain_checks'/stamp();output.mkdir(parents=True)
    previous=read_json(ROOT/'artifacts/iterative_pipeline_checks/20261117_182940_119969/pretrain.json')
    c01=Path(previous['run']['data'])
    actual='/mnt/disk1/yangqilin/dataset/RadioML2018_01A/GOLD_XYZ_OSC.0001_1024.hdf5'
    with h5py.File(actual,'r') as source,h5py.File(output/'radio.h5','w') as target:
        rows=np.concatenate([np.arange((c*26+s)*4096,(c*26+s)*4096+16)
                             for c in [3,4] for s in [10,15]])
        for key in ['X','Y','Z']:target.create_dataset(key,data=source[key][rows])
    settings={'run':{'c01':str(c01),'radioml':str(output/'radio.h5'),'output':str(output/'mixed')},
              'selection':{'classes':['BPSK','QPSK'],'minimum_snr_db':0},
              'split':{'validation_fraction':.2,'seed':20261117}}
    mixed=prepare(settings)
    source=PreparedDataset(c01);target=PreparedDataset(mixed)
    n=len(source.meta['split'])
    for key in source.meta:assert np.array_equal(source.meta[key],target.meta[key][:n]),key
    assert target.info['schema']==source.info['schema']
    assert target.info['target_scaler']==source.info['target_scaler']
    assert (target.meta['modulation'][n:]==-1).all() and not target.meta['target_valid'][n:].any()
    assert (target.meta['protocol'][n:]==-1).all()
    external=np.flatnonzero(target.meta['source'][target.indices]==1)[0]
    item=target[external]
    with h5py.File(output/'radio.h5','r') as radio:
        assert np.array_equal(item['iq'],radio['X'][target.meta['source_row'][target.indices[external]]])
    probabilities={'probabilities':{'c01':.7,'radioml':.3}}
    samplers=[MultiSourceBatchSampler(target,4,1000,probabilities,rank=rank) for rank in [0,1]]
    draws=[list(s) for s in samplers]
    sources=[]
    for a,b in zip(*draws):
        sa=target.meta['source'][target.indices[a]];sb=target.meta['source'][target.indices[b]]
        assert np.array_equal(sa,sb) and len(np.unique(sa))==1
        assert len(np.unique(target.meta['length'][target.indices[a]]))==1
        sources.append(int(sa[0]))
    assert .25<np.mean(sources)<.35
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    def run(name,*args):
        with (output/(name+'.log')).open('w') as f:
            subprocess.run([sys.executable,'-u',*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    first=output/'codebook_v1'
    run('raw_vocabulary','code/training/build_codebook.py','--data',str(mixed),'--output',str(first),
        '--device','cpu','--atomic-vocab','8','--merges','4','--max-patches','256','--bpe-samples','32',
        '--cluster-iterations','2','--batch-size','8','--threads','2','--metric','euclidean','--min-frequency','1')
    configs={}
    for stage in ['pretrain','finetune','supervised']:
        c=load_config(ROOT/'code/configs'/stage/'shared_base_p8_receiver_v1_mixed.json5')
        c['model']['encoder'].update(width=32,depth=1,heads=4,dropout=0.,attention_dropout=0.)
        c['model']['iq_geometry']['bottleneck']=8
        c['model']['receiver_frontend']['filter']['taps']=5
        c['model']['receiver_frontend']['timing'].update(periods=[4.,8.,16.],hidden=8)
        c['model']['receiver_frontend']['readout'].update(width=16,dropout=0.)
        c['run'].update(data=str(mixed if stage=='pretrain' else c01),output=str(output/stage),
                         codebook=str(first) if stage=='pretrain' else '',init_checkpoint='',warmstart_checkpoint='')
        c['training']['epochs']=1;c['data'].update(batch_size=4,steps_per_epoch=6)
        c['data']['loader'].update(num_workers=0,pin_memory=False)
        c['runtime'].update(device='cpu',gpus='',processes=1,precision='fp32',threads=2,log_every=1)
        c['checkpoint']['save_every_steps']=0;c['evaluation'].update(batch_size=16,max_batches=0)
        configs[stage]=c;write_json(output/(stage+'.json'),c)
    from utils.common import read_config_json
    workflow=read_config_json(ROOT/'code/configs/pretrain/two_rounds_p8_receiver_v1_mixed_dual.json5')
    workflow['configs']['pretrain']=str(output/'pretrain.json');workflow['run']['output']=str(output/'pipeline')
    workflow['round1'].update(epochs=1,codebook=str(first));workflow['round2']['epochs']=1
    workflow['runtime'].update(gpu='0,1',device='cpu',processes=2);workflow['data']['batch_size_per_gpu']=4
    workflow['codebook'].update(atomic_vocab=8,merges=4,max_patches=256,cluster_iterations=2,fit_sequences=32,
                                batch_size=8,threads=2,min_frequency=1)
    write_json(output/'pipeline.json',workflow)
    run('pipeline','code/training/train.py','--config',str(output/'pipeline.json'))
    pipeline=Path(read_json(output/'pipeline/latest_pipeline.json')['pipeline_directory'])
    assert (pipeline/'completion.json').exists()
    manifest=read_json(pipeline/'manifest.json')
    assert manifest['processes']==2 and set(manifest['stages'])=={'pretrain_round1','codebook_round2','pretrain_round2'}
    for stage in ['pretrain_round1','pretrain_round2']:
        saved=torch.load(Path(manifest['stages'][stage]['run_directory'])/'last.pt',map_location='cpu',weights_only=True)
        assert saved['world_size']==2 and saved['global_step']==6
        assert set(saved['metrics']['pretraining_by_source'])=={'c01','radioml'}
        assert saved['data_provenance']['target_c01_fingerprint']==source.info['fingerprint']
    validate_backbone_data_contract(saved,source.info)
    wrong=deepcopy(source.info);wrong['fingerprint']='unrelated'
    try:validate_backbone_data_contract(saved,wrong)
    except ValueError:pass
    else:raise AssertionError('Accepted unrelated C01 split')
    configs['finetune']['run']['init_checkpoint']=str(output/'pipeline')
    write_json(output/'finetune.json',configs['finetune'])
    for stage in ['supervised','finetune']:
        run(stage,'code/training/train.py','--config',str(output/(stage+'.json')))
    tuned=latest_run(output/'finetune')
    run('export','code/training/export.py','--checkpoint',str(tuned/'best.pt'),'--output',str(output/'export'))
    run('inference','code/infer.py','--weights',str(output/'export'),
        '--input','artifacts/reception_exploration/summary_20261114/integration_input.parquet',
        '--output',str(output/'predictions.jsonl'),'--device','cpu')
    assert len((output/'predictions.jsonl').read_text().splitlines())==12
    run('resume','code/training/train.py','--config',str(output/'pipeline.json'),'--resume',str(pipeline))
    assert read_json(pipeline/'manifest.json')['stages']==manifest['stages']
    assert model_options(configs['supervised'])==model_options(configs['pretrain'])==model_options(configs['finetune'])
    report={'passed':True,'directory':str(output),'training_world_size':2,'inference_rows':12,
      'checks':['C01 split unchanged','unlabeled RadioML retained in pretraining','DDP source/length schedules agree',
                'raw IQ unchanged','two-round teacher regeneration','per-source validation',
                'pure C01 backbone adaptation with fingerprint provenance','unrelated-split rejection',
                'supervised entry','export and inference','completed-pipeline resume']}
    write_json(output/'verification.json',report);write_json(ROOT/'artifacts/mixed_pretrain_checks/latest.json',report)
    print('MIXED PRETRAINING CHECKS PASSED:',output,flush=True)


if __name__=='__main__':main()
