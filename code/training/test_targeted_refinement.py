from pathlib import Path
import json,os,subprocess,sys
import numpy as np
import torch
sys.path.insert(0,'code')
from utils.common import read_json,read_config_json,write_json
from utils.structure_sampling import TARGET_WEIGHTS,GROUP_NAMES,group_ids,token_plan,TargetGroupBatchSampler
from utils.dataset import PreparedDataset
from utils.config import model_options,load_config
from utils.training_workflow import stamp
from utils.run_directory import latest_run


def main():
    root=Path(__file__).resolve().parents[2];out=root/'artifacts/targeted_refinement_checks'/stamp();out.mkdir(parents=True)
    previous=root/'artifacts/mixed_pretrain_checks/20261117_205921_987290'
    config=read_json(previous/'pretrain.json');data=PreparedDataset(config['run']['data'])
    _,unique=np.unique(data.meta['hash'][data.indices],return_index=True)
    plan=token_plan(data,unique,1024,TARGET_WEIGHTS,np.random.default_rng(7))
    quota={name:sum(p['quota'] for p in plan if p['key'][0]==g) for g,name in enumerate(GROUP_NAMES)}
    assert sum(quota.values())==1024
    sampler=TargetGroupBatchSampler(data,4,100,TARGET_WEIGHTS,rank=0)
    second=TargetGroupBatchSampler(data,4,100,TARGET_WEIGHTS,rank=1)
    weighted=np.zeros(5)
    for key,probability in zip(sampler.keys,sampler.probabilities):weighted[key[0]]+=probability*(key[-1]//8)
    weighted/=weighted.sum();np.testing.assert_allclose(weighted,list(TARGET_WEIGHTS.values()),atol=1e-12)
    for a,b in zip(sampler,second):
     ar=data.indices[a];br=data.indices[b]
     assert np.array_equal(data.meta['length'][ar],data.meta['length'][br])
     assert np.array_equal(group_ids(data.meta,ar),group_ids(data.meta,br))
    pipeline=Path(read_json(previous/'pipeline/latest_pipeline.json')['pipeline_directory'])
    manifest=read_json(pipeline/'manifest.json');teacher=manifest['stages']['pretrain_round1']['best_checkpoint']
    config['run'].update(output=str(out/'training'),init_checkpoint=teacher,codebook='')
    config['training'].update(epochs=1,max_steps=0)
    config['data'].update(batch_size=4,steps_per_epoch=6)
    config['data']['sampling']['multi_source']={'enabled':False,'probabilities':{'c01':.85,'radioml':.15}}
    config['data']['sampling']['target_groups']={'enabled':True,'weights':TARGET_WEIGHTS}
    config['evaluation']['selection_metric']='pretraining_c01_first14_loss'
    write_json(out/'pretrain.json',config)
    workflow=read_config_json('code/configs/pretrain/refine_receiver_targeted_dual.json5')
    workflow['run'].update(output=str(out/'pipeline'),teacher_checkpoint=teacher)
    workflow['configs']['pretrain']=str(out/'pretrain.json')
    workflow['runtime'].update(gpus='0,1',processes=2,device='cpu')
    workflow['codebook'].update(atomic_vocab=8,merges=4,max_patches=512,cluster_iterations=2,
      fit_sequences=32,bpe_tokens=1024,min_frequency=1,batch_size=8,precision='fp32',label_workers=2)
    write_json(out/'workflow.json',workflow)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    def run(name,*args):
     with (out/(name+'.log')).open('w') as f:
      subprocess.run([sys.executable,'-u',*args],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    run('refinement','code/training/train.py','--config',str(out/'workflow.json'))
    directory=Path(read_json(out/'pipeline/latest_pipeline.json')['pipeline_directory'])
    m=read_json(directory/'manifest.json');assert set(m['stages'])=={'codebook_round2','pretrain_round2'}
    book=Path(m['stages']['codebook_round2']['output']);quality=read_json(book/'quality_report.json')
    assert quality['passed'] and quality['bpe_tokens_by_group_verified']==quota
    saved=torch.load(Path(m['stages']['pretrain_round2']['run_directory'])/'last.pt',map_location='cpu',weights_only=True)
    assert saved['global_step']==6 and saved['world_size']==2
    assert saved['metrics']['pretraining_c01_first14_loss']==saved['metrics']['pretraining_by_group']['c01_first14']
    assert set(saved['metrics']['pretraining_by_group'])==set(GROUP_NAMES)
    # Serial and multiprocessing assignment must produce exactly the same ESU IDs.
    from utils.structure import StructuralTokenizer
    metadata=np.load(Path(config['run']['data'])/'metadata.npz',allow_pickle=False)
    atomic=np.load(book/'atomic_labels.npy',mmap_mode='r');emergent=np.load(book/'emergent_labels.npy',mmap_mode='r')
    tokenizer=StructuralTokenizer(read_json(book/'codebook.json')['merges'])
    for row,n in enumerate(metadata['length']):
     count=int(n)//8;np.testing.assert_array_equal(emergent[row,:count],tokenizer.encode(atomic[row,:count]))
    fine=read_json(previous/'finetune.json');fine['run'].update(init_checkpoint=str(out/'pipeline'),output=str(out/'finetune'))
    # Pipeline checkpoint resolution must work for the new workflow as well.
    write_json(out/'finetune.json',fine)
    run('finetune','code/training/train.py','--config',str(out/'finetune.json'))
    tuned=latest_run(out/'finetune')
    run('export','code/training/export.py','--checkpoint',str(tuned/'best.pt'),'--output',str(out/'export'))
    run('infer','code/infer.py','--weights',str(out/'export'),'--input','artifacts/reception_exploration/summary_20261114/integration_input.parquet','--output',str(out/'predictions.jsonl'),'--device','cpu')
    assert len((out/'predictions.jsonl').read_text().splitlines())==12
    before=read_json(directory/'manifest.json')['stages']
    run('resume','code/training/train.py','--config',str(out/'workflow.json'),'--resume',str(directory))
    assert read_json(directory/'manifest.json')['stages']==before
    report={'passed':True,'directory':str(out),'checks':['exact token quotas','expected training token shares','DDP group/length schedule',
     'teacher-based second vocabulary','training-only fit rows','parallel/serial ESU equivalence','CPU two-process round2',
     'C01 first14 selection','all five validation groups','C01 fine initialization','export/inference','completed-flow resume'],
     'inference_rows':12,'quota':quota}
    write_json(out/'verification.json',report);write_json(root/'artifacts/targeted_refinement_checks/latest.json',report)
    print('TARGETED REFINEMENT CHECKS PASSED',out,flush=True)

if __name__=="__main__":main()
