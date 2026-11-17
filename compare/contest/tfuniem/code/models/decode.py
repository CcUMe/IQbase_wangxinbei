"""Turn parameter-head outputs into the 7 raw-unit values (guide 5.8)."""
import numpy as np
import torch

from models.tfuniem import PARAM_SLICES


def sl(out, name):
    a, b = PARAM_SLICES[name]
    return out[:, a:b]


@torch.no_grad()
def decode_parameters(out, head_stats, combo_mode='expect', npulses_mode='expect'):
    """out: (B, 36) float tensor. Returns (B, 7) float64 numpy array, raw units; inactive group filled with 0."""
    out = out.float()
    h = head_stats
    dev = out.device
    group = sl(out, 'group').argmax(1)  # 0 pulse, 1 comm
    combos = torch.tensor(h['sr_bw_combos'], dtype=torch.float64, device=dev)  # (22, 2)
    pc = torch.softmax(sl(out, 'combo').double(), 1)
    if combo_mode == 'expect':
        srbw = pc @ combos
    else:
        srbw = combos[pc.argmax(1)]
    pf = torch.softmax(sl(out, 'fs_cls').double(), 1)
    lo, hi = h['fshift_small_range']
    small = (sl(out, 'fs_reg').double().squeeze(1) * h['fshift_small_std'] + h['fshift_small_mean']).clamp(lo, hi)
    fs = pf[:, 0] * small + pf[:, 1] * 100.0 + pf[:, 2] * 1000.0
    pn = torch.softmax(sl(out, 'npulses').double(), 1)
    vals = torch.arange(2, 7, dtype=torch.float64, device=dev)
    npl = pn @ vals if npulses_mode == 'expect' else vals[pn.argmax(1)]
    t = sl(out, 'times').double()
    mean = torch.tensor(h['time_mean'], dtype=torch.float64, device=dev)
    std = torch.tensor(h['time_std'], dtype=torch.float64, device=dev)
    rng = torch.tensor(h['time_range'], dtype=torch.float64, device=dev)
    t = torch.maximum(torch.minimum(t * std + mean, rng[:, 1]), rng[:, 0])
    comm = torch.stack([srbw[:, 0], srbw[:, 1], fs], 1)
    pulse = torch.cat([t, npl[:, None]], 1)
    is_comm = (group == 1)[:, None]
    res = torch.cat([torch.where(is_comm, comm, torch.zeros_like(comm)),
                     torch.where(is_comm, torch.zeros_like(pulse), pulse)], 1)
    return res.cpu().numpy()


DECODE_VARIANTS = {'expect': ('expect', 'expect'), 'argmax': ('argmax', 'argmax')}
