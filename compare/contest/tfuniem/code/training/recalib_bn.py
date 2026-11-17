"""BatchNorm statistics recalibration experiment (no weight changes).

Variants evaluated on the validation set (eval-mode BN, fp32):
  a_original   running stats from training (EMA over alternating 512 / 1024 sub-batches)
  b_shared     stats re-estimated (cumulative average) on training batches of BOTH lengths, one shared set
  c_per_length stats re-estimated separately on 512-point and 1024-point training sub-batches; inference uses
               the set that matches the input length (length is known at inference time)
Recalibration batches use the same balanced task mix as training (64 mod / 64 pro / 128 param per step),
drawn from training rows excluding the tune subset.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.tfuniem import TFUniEM
from training.train import evaluate
from utils.data import Cache, gather


class LengthRouted(nn.Module):
    def __init__(self, by_len):
        super().__init__()
        self.m512, self.m1024 = by_len[512], by_len[1024]

    def forward(self, iq, task):
        return (self.m512 if iq.shape[1] == 512 else self.m1024)(iq, task)


def bn_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


@torch.no_grad()
def recalibrate(model, batches, lengths, device):
    model.eval()
    for bn in bn_layers(model):
        bn.reset_running_stats()
        bn.momentum = None  # cumulative moving average
        bn.train()
    for b in batches:
        for L in lengths:
            if L in b:
                model(b[L]['iq'].to(device), b[L]['task'].to(device))
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
    base = TFUniEM(**state['model_cfg'])
    base.load_state_dict(state['model'])
    base.to(device).eval()

    rng = np.random.default_rng(123)
    pools = [np.flatnonzero((cache.meta['split'] == 0) & ~cache.meta['tune'] & (cache.meta['task'] == t))
             for t in range(3)]
    batches = [gather(cache, np.concatenate([rng.choice(p, k) for p, k in zip(pools, (64, 64, 128))]))
               for _ in range(a.steps)]
    val_rows = cache.rows(1)
    keys = ['proxy_score', 'protocol_accuracy', 'modulation_accuracy', 'parameter_proxy']
    res = {}
    variants = {
        'a_original': base,
        'b_shared': recalibrate(copy.deepcopy(base), batches, (512, 1024), device),
        'c_per_length': LengthRouted({L: recalibrate(copy.deepcopy(base), batches, (L,), device)
                                      for L in (512, 1024)}).to(device),
    }
    for name, model in variants.items():
        m = evaluate(model, cache, val_rows, device)
        res[name] = {**{k: m[k] for k in keys}, 'rmse': {k: v['rmse'] for k, v in m['parameters'].items()},
                     'narrowband14': m['modulation']['narrowband14_accuracy'],
                     'other9': m['modulation']['other9_accuracy']}
        print(name, json.dumps({k: round(v, 4) for k, v in res[name].items() if not isinstance(v, dict)}), flush=True)
    # how different are the per-length statistics? (relative gap of running means, first stem BN)
    m512, m1024 = variants['c_per_length'].m512, variants['c_per_length'].m1024
    gaps = []
    for b5, b10 in zip(bn_layers(m512), bn_layers(m1024)):
        gaps.append(float(((b5.running_mean - b10.running_mean).abs() / (b5.running_var + b10.running_var).sqrt().clamp_min(1e-6)).mean()))
    res['per_length_mean_gap_in_std_units'] = {'median': float(np.median(gaps)), 'max': float(np.max(gaps)),
                                               'n_bn_layers': len(gaps)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'bn_recalibration.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res['per_length_mean_gap_in_std_units']))


if __name__ == '__main__':
    main()
