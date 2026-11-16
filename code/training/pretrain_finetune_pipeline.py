"""One-GPU structural pretraining -> finetuning -> single-model export.

Stage configs and outputs are frozen in a fresh timestamped pipeline directory.
The finetuning source is the EXACT pretraining checkpoint from this pipeline,
never an unrelated global latest-run pointer. Resume retains stage optimizers.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
import torch
from utils.common import read_json,write_json
from utils.config import load_config,model_options
from utils.run_directory import latest_run


def root_path(value):return str((ROOT/Path(value)).resolve())


def prepare_configs(pretrain,finetune,gpu,device):
    configs=[load_config(pretrain),load_config(finetune)]
    if configs[0]['training']['stage']!='pretrain' or configs[1]['training']['stage']!='supervised':
        raise ValueError('Pipeline requires structural pretrain and supervised finetune stages')
    if model_options(configs[0])!=model_options(configs[1]):
        raise ValueError('Pretraining and finetuning must share the same backbone/model definition')
    if not configs[0]['run']['codebook']:raise ValueError('Structural pretraining requires a codebook')
    aug=configs[0]['data']['augmentation']
    if aug['phase_rotation_probability'] or any(aug['modulation_only'][k] for k in ['phase_rotation_probability','circular_shift_probability']):
        raise ValueError('Pretraining structure targets require unaugmented IQ')
    for c in configs:
        if c['training']['max_steps']:raise ValueError('Use complete epoch budgets in the automatic pipeline')
        if c['checkpoint']['resume_from']:raise ValueError('Use pipeline --resume, not a config resume_from')
        if not c['checkpoint']['save_last'] or not c['checkpoint']['save_best']:raise ValueError('Pipeline requires last/best checkpoint saving')
        if c['evaluation']['early_stopping']['patience']:raise ValueError('This full-budget pipeline requires early stopping disabled')
        c['runtime'].update(device=device,gpus=gpu if device!='cpu' else '',processes=1)
        for key in ['data','codebook','init_checkpoint','warmstart_checkpoint']:
            if c['run'][key]:c['run'][key]=root_path(c['run'][key])
    if configs[0]['run']['data']!=configs[1]['run']['data']:raise ValueError('Keep the same prepared split across stages')
    info=read_json(Path(configs[0]['run']['data'])/'dataset.json')
    codebook=Path(configs[0]['run']['codebook']);vocabulary=read_json(codebook/'codebook.json')
    if vocabulary['dataset_fingerprint']!=info['fingerprint'] or vocabulary['patch_size']!=info['patch_size']:
        raise ValueError('Codebook/data fingerprint or patch size mismatch')
    for name in ['atomic_labels.npy','emergent_labels.npy']:
        if not (codebook/name).is_file():raise FileNotFoundError(codebook/name)
    return configs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrain-config',type=Path,default=ROOT/'code/configs/pretrain/shared_base.json5')
    parser.add_argument('--finetune-config',type=Path,default=ROOT/'code/configs/finetune/shared_base.json5')
    parser.add_argument('--output',type=Path,default=ROOT/'logs/shared_base_pipeline')
    parser.add_argument('--gpu',default='0',help='Exactly one physical GPU index')
    parser.add_argument('--device',default='auto',choices=['auto','cpu'])
    parser.add_argument('--resume',type=Path,help='Resume this pipeline directory from stage last.pt')
    args=parser.parse_args()
    if not args.gpu.isdigit() or int(os.environ.get('WORLD_SIZE','1'))!=1:
        parser.error('Use one physical GPU and one pipeline process')
    if args.device!='cpu':
        active=subprocess.check_output(['nvidia-smi','--id='+args.gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        # if active:raise RuntimeError('Requested GPU is occupied; no second trainer launched: '+active)
    if args.resume:
        directory=args.resume.resolve();manifest=read_json(directory/'manifest.json')
        if manifest['device']!=args.device or manifest['gpu']!=args.gpu:raise ValueError('Resume with the original device/GPU setting')
        configs=[read_json(directory/(stage+'_config.json')) for stage in ['pretrain','finetune']]
    else:
        configs=prepare_configs(args.pretrain_config,args.finetune_config,args.gpu,args.device)
        directory=args.output.resolve()/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        directory.mkdir(parents=True,exist_ok=False)
        manifest={'device':args.device,'gpu':args.gpu,'stages':{},'configs':{
            'pretrain':str(args.pretrain_config.resolve()),'finetune':str(args.finetune_config.resolve())},
            'architecture_sha256':hashlib.sha256(json.dumps(model_options(configs[0]),sort_keys=True).encode()).hexdigest()}
        for stage,c in zip(['pretrain','finetune'],configs):
            c['run']['output']=str(directory/stage)
            write_json(directory/(stage+'_config.json'),c)
        snapshot=directory/'source_snapshot'/'code'
        shutil.copytree(ROOT/'code',snapshot,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        manifest['code_snapshot']=str(snapshot)
        hashes={str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in snapshot.rglob('*') if p.is_file()}
        write_json(directory/'source_snapshot/sha256.json',hashes)
    lock=(directory/'pipeline.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (directory/'pipeline.pid').write_text(str(os.getpid())+'\n')
    child=None
    def status(stage,**extra):
        value={'stage':stage,'pipeline_pid':os.getpid(),'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),**extra}
        temporary=directory/'status.tmp';write_json(temporary,value);temporary.replace(directory/'status.json')
        print(json.dumps(value,ensure_ascii=False),flush=True)
    def stop(signum,_):
        if child is not None and child.poll() is None:child.send_signal(signum)
        status('interrupted',signal=signum)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    write_json(directory/'manifest.json',manifest)
    pointer=directory.parent/'latest_pipeline.json'
    pointer.parent.mkdir(parents=True,exist_ok=True)
    write_json(pointer,{'pipeline_directory':str(directory),'pid':os.getpid()})
    env={**os.environ,'CUDA_VISIBLE_DEVICES':args.gpu if args.device!='cpu' else '',
         'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
    pretrain_best=None
    try:
        for stage,c in zip(['pretrain','finetune'],configs):
            completed=manifest['stages'].get(stage,{})
            if completed.get('complete'):
                if stage=='pretrain':pretrain_best=completed['best_checkpoint']
                continue
            if stage=='finetune':
                if not pretrain_best:raise ValueError('No completed pretraining checkpoint')
                c['run'].update(init_checkpoint=pretrain_best,warmstart_checkpoint='')
                write_json(directory/'finetune_config.json',c)
            config_path=directory/(stage+'_config.json')
            command=[sys.executable,'-u',str(Path(manifest['code_snapshot'])/'training/train.py'),'--config',str(config_path)]
            try:
                run=latest_run(Path(c['run']['output']));checkpoint=run/'last.pt'
            except FileNotFoundError:checkpoint=None
            if checkpoint and checkpoint.is_file():command+=['--resume',str(checkpoint)]
            console=directory/(stage+'_console.log')
            with console.open('a',buffering=1) as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
                status(stage+'_running',child_pid=child.pid,console=str(console),config=str(config_path))
                result=child.wait();child=None
            if result:raise RuntimeError(f'{stage} failed with exit code {result}; inspect {console}')
            run=latest_run(Path(c['run']['output']));last=torch.load(run/'last.pt',map_location='cpu',weights_only=True,mmap=True)
            if not last['epoch_complete'] or last['epoch']+1<c['training']['epochs']:
                raise RuntimeError(f'{stage} stopped before the full budget; resume this pipeline')
            best=run/'best.pt'
            if not best.is_file():raise FileNotFoundError(best)
            manifest['stages'][stage]={'complete':True,'run_directory':str(run),'best_checkpoint':str(best),
                                      'global_step':last['global_step']}
            write_json(directory/'manifest.json',manifest)
            if stage=='pretrain':pretrain_best=str(best)
            status(stage+'_complete',checkpoint=str(best),global_step=last['global_step'])
            del last
        export=directory/'exported_model'
        subprocess.run([sys.executable,str(Path(manifest['code_snapshot'])/'training/export.py'),'--checkpoint',manifest['stages']['finetune']['best_checkpoint'],'--output',str(export)],check=True,cwd=ROOT,env={**env,'CUDA_VISIBLE_DEVICES':''})
        write_json(directory/'completion.json',{'status':'complete','stages':manifest['stages'],'export':str(export)})
        status('complete',export=str(export))
    except Exception as exc:
        status('failed',error=repr(exc));raise


if __name__=='__main__':main()
