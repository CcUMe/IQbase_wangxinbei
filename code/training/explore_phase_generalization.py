"""New serial single-GPU generalization trials; original checkpoints untouched."""
import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from training import explore_phase_training as engine
from utils.common import read_json,write_json
from utils.dataset import PreparedDataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=900)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--arms',default='mod_phase_10,mod_phase_shift,mod_phase_noise,mod_sam_005')
    p.add_argument('--source',type=Path,default=ROOT/'artifacts/phase_exploration/20261114_035726_210852/snapshots/best.pt')
    p.add_argument('--seed',type=int,default=20261114)
    args=p.parse_args()
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or torch.cuda.device_count()!=1
            or int(os.environ.get('WORLD_SIZE','1'))!=1):
        raise ValueError('This experiment requires exactly physical GPU 0 and one process')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    definitions={
        'low_lr_control':dict(smoothing=0.,double=False,kl=0.),
        'mod_noise_only':dict(smoothing=0.,double=False,kl=0.,noise=.1),
        'mod_phase_10':dict(smoothing=0.,double=False,kl=0.,phase=1.),
        'mod_phase_shift':dict(smoothing=0.,double=False,kl=0.,phase=1.,shift=.5),
        'mod_phase_noise':dict(smoothing=0.,double=False,kl=0.,phase=1.,noise=.1),
        'mod_sam_005':dict(smoothing=0.,double=False,kl=0.,sam=.05),
    }
    engine.ARMS={name:definitions[name] for name in args.arms.split(',')}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'snapshots').mkdir(exist_ok=True)
    reference=ROOT/'artifacts/phase_exploration/20261114_035726_210852'
    source=args.source.resolve()
    if not (args.output/'snapshots/best.pt').exists():
        shutil.copy2(source,args.output/'snapshots/best.pt')
    def checksum(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            while block:=f.read(8*1024*1024):h.update(block)
        return h.hexdigest()
    source_sha=checksum(source)
    if checksum(args.output/'snapshots/best.pt')!=source_sha:
        raise ValueError('Existing output contains a different source checkpoint; use a new output directory')
    args.data=ROOT/'artifacts/data_full';args.batch_size=64 if args.smoke else 256
    args.eval_batch_size=64 if args.smoke else 128;args.device='cuda:0'
    if args.smoke:args.steps=3
    protocol=dict(planned_arms=engine.ARMS,source_checkpoint=str(source),
        steps=args.steps,batch_size=args.batch_size,smoke=args.smoke,physical_gpu='0',
        original_controls=str(reference),training_seed=args.seed,
        started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        source_sha256=source_sha,
        controls='Reuse frozen low-LR/double-CE/shift controls only with identical 900-step stream, seed, optimizer, schedule and initialization.',
        preserve_three_tasks=True,augmentation_eligibility='modulation label only; protocol<0 and no valid parameter targets',
        noise='p=.5; complex white Gaussian std / observed RMS uniform [.02,.10], original energy restored; NOT a clean-SNR estimate',
        sam='Modulation steps only, rho=.05, normalized raw gradient, exact weight restoration before optimizer update; other tasks ordinary AdamW',
        sources=['https://arxiv.org/abs/1912.03026',
            'https://www.cs.albany.edu/~mariya/lab/papers/channel_hw_augmentation_tccn24.pdf',
            'https://arxiv.org/abs/2010.01412'],
        engineering_gate='overall +1pp AND first14 +2pp over frozen source best, no substantial other-task damage; then independent stream-seed confirmation',
        caveat='Historical validation reused for research; confidence intervals conditional on these records, not independent hidden-test or initialization-seed guarantees')
    previous_path=args.output/'protocol.json'
    if previous_path.exists():
        previous=read_json(previous_path)
        for key in ['source_checkpoint','steps','batch_size','training_seed','smoke']:
            if key in previous and previous[key]!=protocol[key]:
                raise ValueError(f'Existing experiment protocol mismatch: {key}; use a new output directory')
        protocol['planned_arms']={**previous.get('planned_arms',{}),**protocol['planned_arms']}
        protocol['started_at']=previous.get('started_at',protocol['started_at'])
    protocol['updated_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    write_json(args.output/'protocol.json',protocol)
    data=PreparedDataset(args.data,'validation')
    checkpoint=torch.load(args.output/'snapshots/best.pt',map_location='cpu',mmap=True,weights_only=True)
    if checkpoint['dataset_fingerprint']!=read_json(args.data/'dataset.json')['fingerprint']:
        raise ValueError('Checkpoint/data fingerprint mismatch')
    del checkpoint
    if args.smoke:
        # Include all tasks and both native lengths; never overwrite full results.
        rows=data.indices
        take=[]
        for length in [512,1024]:
            for keep in [data.meta['modulation']>=0,data.meta['protocol']>=0,data.meta['target_valid'].any(1)]:
                take.extend(rows[(data.meta['length'][rows]==length)&keep[rows]][:8])
        data.indices=np.unique(take)
    iq=np.load(args.data/'iq.npy',mmap_mode='r')
    selection=dict(np.load(ROOT/'artifacts/c01_followup_validation/20261113_224434/selection.npz',allow_pickle=False))
    if source != (reference/'snapshots/best.pt').resolve():
        checkpoint,model=engine.checkpoint_model(args.output/'snapshots/best.pt',args.device)
        engine.evaluate_saved(model,checkpoint,data.indices,iq,data.meta,args,'source_best_validation')
        del checkpoint,model
        torch.cuda.empty_cache()
    engine.train_arms(args,data,selection,iq)
    print('FINISHED',args.output,flush=True)


if __name__=='__main__':main()
