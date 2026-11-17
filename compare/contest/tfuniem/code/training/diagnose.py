"""Post-training diagnostics for a checkpoint (guide 10.2 + 6.3 reporting items).

1. Dump raw outputs (logits / parameter-head outputs) on the validation set and on a random training subset.
2. BatchNorm check: validation metrics with eval-mode BN (running stats) vs train-mode BN (batch stats).
3. Single-GPU inference throughput (fp32 / bf16, unfused / fused RepBlocks) per length.

Usage: CUDA_VISIBLE_DEVICES=4 python code/training/diagnose.py --ckpt runs/base_r1/best.pt --out runs/base_r1/diag
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.tfuniem import TFUniEM
from training.train import evaluate
from utils.data import Cache, validation_batches


@torch.no_grad()
def dump(model, cache, rows, device):
    out = {'row': [], 'mod': [], 'pro': [], 'par': []}
    for b in validation_batches(cache, rows, 1024):
        o = model(b['iq'].to(device), b['task'].to(device))
        out['row'].append(b['row'].numpy())
        out['mod'].append(o['modulation'].float().cpu().numpy())
        out['pro'].append(o['protocol'].float().cpu().numpy())
        out['par'].append(o['parameter'].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


@torch.no_grad()
def throughput(model, device, L, batch, dtype, iters=30):
    x = torch.randn(batch, L, 2, device=device) * 0.01
    t = torch.randint(0, 3, (batch,), device=device)
    ctx = torch.autocast('cuda', dtype=torch.bfloat16) if dtype == 'bf16' else torch.autocast('cuda', enabled=False)
    with ctx:
        for _ in range(5):
            model(x, t)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(x, t)
        torch.cuda.synchronize()
    return batch * iters / (time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--train-subset', type=int, default=60000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    state = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    cache = Cache(json.loads((a.ckpt.parent / 'config.json').read_text())['cache'])
    model = TFUniEM(**state['model_cfg'])
    model.load_state_dict(state['model'])
    model.to(device).eval()
    report = {'ckpt': str(a.ckpt), 'epoch': state['epoch']}

    # 1. raw outputs
    val_rows = cache.rows(1)
    rng = np.random.default_rng(0)
    tr_rows = np.sort(rng.choice(cache.rows(0), a.train_subset, replace=False))
    t0 = time.time()
    v = dump(model, cache, val_rows, device)
    tr = dump(model, cache, tr_rows, device)
    np.savez(a.out / 'outputs_val.npz', **v)
    np.savez(a.out / 'outputs_train_subset.npz', **tr)
    print(f'dumped outputs ({time.time() - t0:.0f}s)', flush=True)

    # 2. BN eval-mode vs train-mode
    keys = ['proxy_score', 'protocol_accuracy', 'modulation_accuracy', 'parameter_proxy']
    m_eval = evaluate(model, cache, val_rows, device)
    m_train = evaluate(model, cache, val_rows, device, train_mode_bn=True)
    report['bn_check'] = {'eval_mode_bn': {k: m_eval[k] for k in keys},
                          'train_mode_bn': {k: m_train[k] for k in keys},
                          'rmse_eval': {k: v_['rmse'] for k, v_ in m_eval['parameters'].items()},
                          'rmse_train_bn': {k: v_['rmse'] for k, v_ in m_train['parameters'].items()}}
    print('BN check', json.dumps(report['bn_check'], indent=1), flush=True)

    # 3. throughput
    fused = copy.deepcopy(model).fuse().eval()
    tp = {}
    for name, mdl in [('unfused', model), ('fused', fused)]:
        for dtype in ['fp32', 'bf16']:
            for L in (512, 1024):
                for bs in (1, 1024):
                    tp[f'{name}_{dtype}_L{L}_bs{bs}'] = throughput(mdl, device, L, bs, dtype, iters=200 if bs == 1 else 30)
    report['throughput_samples_per_s'] = tp
    report['gpu'] = torch.cuda.get_device_name()
    print('throughput', json.dumps({k: round(v_) for k, v_ in tp.items()}, indent=1), flush=True)
    (a.out / 'diag_report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
