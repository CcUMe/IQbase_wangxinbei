"""Convert a plain-BatchNorm checkpoint to per-length BatchNorm (LenBN) and re-estimate the running statistics
separately for 512- and 1024-point inputs on training batches (balanced task mix, tune subset excluded).
Weights are unchanged; only BN running statistics are recomputed.

Usage: CUDA_VISIBLE_DEVICES=1 python code/training/convert_lenbn.py --ckpt runs/base_r1/best.pt --out runs/base_r1/best_lenbn.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.tfuniem import LenBN, TFUniEM, load_legacy_state
from utils.data import Cache, gather


@torch.no_grad()
def recalibrate_per_length(model, cache, device, steps=400, seed=123):
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero((cache.meta['split'] == 0) & ~cache.meta['tune'] & (cache.meta['task'] == t))
             for t in range(3)]
    model.eval()
    bns = [m for m in model.modules() if isinstance(m, LenBN)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average
        bn.train()
    for _ in range(steps):
        b = gather(cache, np.concatenate([rng.choice(p, k) for p, k in zip(pools, (64, 64, 128))]))
        for L in (512, 1024):
            if L in b:
                model(b[L]['iq'].to(device), b[L]['task'].to(device))
    for bn in bns:
        bn.momentum = 0.1
        bn.eval()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=400)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    state = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    cache = Cache(json.loads((a.ckpt.parent / 'config.json').read_text())['cache'])
    model = TFUniEM(**state['model_cfg'])
    load_legacy_state(model, state['model'])
    model.to(device)
    recalibrate_per_length(model, cache, device, a.steps)
    state['model'] = {k: v.cpu() for k, v in model.state_dict().items()}
    state['bn'] = {'type': 'per_length', 'recalibrated_from': str(a.ckpt), 'steps': a.steps,
                   'rows': 'train split, tune subset excluded, balanced 64/64/128'}
    torch.save(state, a.out)
    print('saved', a.out)


if __name__ == '__main__':
    main()
