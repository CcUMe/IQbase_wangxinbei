"""Run independent hypothesis diagnostics on GPUs 2/3, with durable task status."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.common import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=args.output.resolve()
    r.mkdir(parents=True,exist_ok=True)
    lock=(r/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    py=sys.executable
    template='artifacts/supervised_main/20261110_185257_762257/best.pt'
    baseline='artifacts/diagnostics/20261111_110649_079619/joint/last.pt'
    if not (r/'run_info.json').exists():
        write_json(r/'run_info.json',{'root':str(r),'gpus':[2,3],'downstream_cycles':4800,
            'batch_size':256,'pretraining_steps':1200,'fixed_final_epoch':True,'template':template,'baseline':baseline})
    if not (r/'plan.md').exists():
        (r/'plan.md').write_text('# Fixed diagnostic protocol\n\n'
            'GPUs 2/3 only. Three from-scratch arms remove scale injection, remove global-spectrum injection, '
            'or add a zero-initialized multiscale frontend. All share the historical control initialization, '
            '4800 modulation cycles, batch256 and per-cycle LR.\n\n'
            'Waveform and phase-invariant atomic pretraining both run1200 updates with identical initial state, '
            'sample/mask streams and auxiliary heads. Codebook fits training IQ only; no ESU/fs alignment. '
            'Downstream fine-tuning uses the same4800-cycle budget.\n\n'
            'Linear probes use train-only normalization,20 fixed epochs and CPU seeds2026/2027. '
            'Modulation perturbations are paired stress tests, not official evaluation. '
            'One encoder seed per arm; do not extrapolate directly to97M long training or hidden-test gains.\n')
    if not (r/'atomic_codebook/codebook.pt').exists():
        if (r/'atomic_codebook').exists():
            (r/'atomic_codebook').rename(r/f'atomic_codebook_interrupted_{int(time.time())}')
        subprocess.run([py,'code/training/hypothesis_pretrain.py','--data','artifacts/data_full',
            '--output',str(r/'atomic_codebook'),'--objective','fit','--device','cpu'],check=True)
    if not (r/'random_reference/model.pt').exists():
        import torch
        from models.rf_model import RFModel
        from utils.common import model_bundle,seed_all,save_checkpoint
        source=torch.load(template,map_location='cpu',weights_only=True,mmap=True)
        bundle=model_bundle(source)
        bundle['model_config'].update(qk_normalization=True,pretrain_heads=False,atomic_vocab=0,emergent_vocab=0)
        seed_all(2026);model=RFModel(**bundle)
        save_checkpoint(r/'random_reference/model.pt',{**bundle,'model':model.state_dict(),'model_config':model.config,
            'dataset_fingerprint':source['dataset_fingerprint'],'training_config':source['training_config'],
            'stage':'diagnostic','global_step':0})
        del model,source
    if not (r/'default_weight_hashes.json').exists():
        write_json(r/'default_weight_hashes.json',{name:hashlib.sha256((Path('weights')/name).read_bytes()).hexdigest()
            for name in ['config.json','model.safetensors']})
    if not (r/'source_snapshot').exists():
        shutil.copytree('code',r/'source_snapshot/code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    common=[py,'code/training/diagnose_tasks.py','--data','artifacts/data_full','--template-checkpoint',template,'--mode','joint']
    tasks=[]
    def task(name,command,depends=(),resumable=False):
        tasks.append(dict(name=name,command=command,depends=list(depends),resumable=resumable,status='pending'))
    for name,flag in [('no_scale','--drop-scale-embedding'),('no_global','--drop-global-spectrum'),('multiscale','--multiscale-frontend')]:
        task(name,common+['--output',str(r/name),flag],resumable=True)
    for name,objective in [('pretrain_reconstruction','reconstruction'),('pretrain_structure','structure')]:
        task(name,[py,'code/training/hypothesis_pretrain.py','--data','artifacts/data_full','--output',str(r/name),
            '--template-checkpoint',template,'--codebook',str(r/'atomic_codebook/codebook.pt'),'--objective',objective])
    for name,parent in [('finetune_reconstruction','pretrain_reconstruction'),('finetune_structure','pretrain_structure')]:
        task(name,common+['--output',str(r/name),'--pretrained-backbone',str(r/parent/'last.pt')],(parent,),True)
    task('large_interventions',[py,'code/training/hypothesis_probe.py','--data','artifacts/data_full',
        '--checkpoint','artifacts/representation_optimization/20261111_122154_915261/selected/model.pt',
        '--output',str(r/'large_interventions'),'--interventions'])
    probe_sources=[('random_probe',str(r/'random_reference/model.pt'),()),('baseline_probe',baseline,())]
    for name in ['no_scale','no_global','multiscale','pretrain_reconstruction','pretrain_structure','finetune_reconstruction','finetune_structure']:
        probe_sources.append((name+'_probe',str(r/name/'last.pt'),(name,)))
    for name,checkpoint,dependencies in probe_sources:
        task(name,[py,'code/training/hypothesis_probe.py','--data','artifacts/data_full','--checkpoint',checkpoint,
            '--output',str(r/name)],dependencies)
    state_path=r/'status.json'
    if state_path.exists():
        old=json.loads(state_path.read_text())
        previous={t['name']:t for t in old['tasks']}
        for t in tasks:
            previous_task=previous.get(t['name'],{})
            if previous_task.get('status')=='complete':
                t.update(previous_task)
    active={}
    def persist():
        write_json(state_path,{'complete':all(t['status']=='complete' for t in tasks),
            'controller_pid':os.getpid(),'tasks':tasks,'gpus':[2,3],'updated_unix':time.time()})
    persist()
    while any(t['status'] in ['pending','running'] for t in tasks):
        for gpu,(process,t,handle) in list(active.items()):
            code=process.poll()
            if code is not None:
                handle.close();t.update(status='complete' if code==0 else 'failed',returncode=code,ended_unix=time.time())
                del active[gpu];persist();print(t['name'],t['status'],flush=True)
        completed={t['name'] for t in tasks if t['status']=='complete'}
        failed={t['name'] for t in tasks if t['status'] in ['failed','blocked']}
        for t in tasks:
            if t['status']=='pending' and failed.intersection(t['depends']):t['status']='blocked'
        for gpu in [2,3]:
            if gpu in active:continue
            candidate=next((t for t in tasks if t['status']=='pending' and set(t['depends']).issubset(completed)),None)
            if candidate is None:continue
            name=candidate['name'];directory=r/name;command=list(candidate['command'])
            if directory.exists():
                checkpoint=directory/'last.pt'
                if candidate['resumable'] and checkpoint.exists():
                    command+=['--resume',str(checkpoint)]
                else:
                    directory.rename(r/f'{name}_interrupted_{int(time.time())}')
            handle=(r/f'{name}_console.log').open('a')
            process=subprocess.Popen(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=handle,stderr=subprocess.STDOUT)
            candidate.update(status='running',pid=process.pid,gpu=gpu,started_unix=time.time(),actual_command=command)
            active[gpu]=(process,candidate,handle);persist();print('started',name,'GPU',gpu,flush=True)
        if not active and not any(t['status']=='pending' for t in tasks):break
        time.sleep(1)
    persist()
    if any(t['status']!='complete' for t in tasks):raise SystemExit('Some experiments failed; inspect status.json and console logs')


if __name__=='__main__':main()
