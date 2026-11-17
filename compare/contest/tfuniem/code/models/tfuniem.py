"""TF-UniEM: time-frequency dual-stream convolutional encoder + shared Transformer + task heads.

Only Conv1d / Linear / BatchNorm / LayerNorm / GELU / add / mul / matmul / softmax / mean / log are used in the
forward pass (no torch.fft, torch.stft, complex tensors, einsum, atan2, or data-dependent control flow).
Fixed spectral transforms run in float32 even under autocast.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_PROTOCOL, N_MODULATION = 8, 23
# parameter head output layout (one Linear, sliced)
PARAM_SLICES = {'group': (0, 2), 'combo': (2, 24), 'fs_cls': (24, 27), 'fs_reg': (27, 28),
                'npulses': (28, 33), 'times': (33, 36)}
N_PARAM_OUT = 36


class LenBN(nn.BatchNorm1d):
    """BatchNorm with separate running statistics per input length (512 / 1024).

    Training uses batch statistics (each length sub-batch is normalised by its own stats, as before) and updates
    only the running stats of the current length; inference uses the stats of the current length. The current
    length is set by TFUniEM before each forward (static per length bucket, not data dependent)."""
    LENGTHS = (512, 1024)

    def __init__(self, c, eps=1e-5, momentum=0.1):
        super().__init__(c, eps=eps, momentum=momentum, affine=True, track_running_stats=False)
        for L in self.LENGTHS:
            self.register_buffer(f'rm_{L}', torch.zeros(c))
            self.register_buffer(f'rv_{L}', torch.ones(c))
            self.register_buffer(f'nbt_{L}', torch.tensor(0, dtype=torch.long))
        self.cur_len = 1024

    def stats(self, L=None):
        L = L or self.cur_len
        return getattr(self, f'rm_{L}'), getattr(self, f'rv_{L}')

    def reset_running_stats(self):
        if not hasattr(self, f'nbt_{self.LENGTHS[-1]}'):  # called by the base __init__ before our buffers exist
            return
        for L in self.LENGTHS:
            getattr(self, f'rm_{L}').zero_()
            getattr(self, f'rv_{L}').fill_(1)
            getattr(self, f'nbt_{L}').zero_()

    def forward(self, x):
        rm, rv = self.stats()
        factor = 0.0
        if self.training:
            nbt = getattr(self, f'nbt_{self.cur_len}')
            nbt.add_(1)
            factor = (1.0 / float(nbt)) if self.momentum is None else self.momentum
        return F.batch_norm(x, rm, rv, self.weight, self.bias, self.training, factor, self.eps)


def conv_bn(cin, cout, k, s=1, act=True):
    layers = [nn.Conv1d(cin, cout, k, s, (k - 1) // 2, bias=False), LenBN(cout)]
    if act:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


class ConvSTFT(nn.Module):
    """Two-sided STFT of a complex signal (I,Q channels) as a fixed-weight Conv1d; returns log power (B, K, L/hop)."""

    def __init__(self, n_fft=64, hop=4, learnable=False):
        super().__init__()
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(-n_fft // 2, n_fft // 2, dtype=torch.float64)
        theta = 2 * math.pi * k[:, None] * n[None, :] / n_fft
        win = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
        cos, sin = torch.cos(theta) * win, torch.sin(theta) * win
        re = torch.stack([cos, sin], dim=1)    # I*cos + Q*sin
        im = torch.stack([-sin, cos], dim=1)   # Q*cos - I*sin
        self.conv = nn.Conv1d(2, 2 * n_fft, n_fft, stride=hop, padding=(n_fft - hop) // 2, bias=False)
        self.conv.weight.data.copy_(torch.cat([re, im], 0).float())
        self.conv.weight.requires_grad_(learnable)
        self.k = n_fft

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = self.conv(x.to(self.conv.weight.dtype))
            power = y[:, :self.k] ** 2 + y[:, self.k:] ** 2
            return torch.log(power + 1e-6)


def dft_weights(length, bins, complex_input):
    """Conv1d(kernel=L, stride=L) weights giving Re/Im of X[k] = sum x[n] exp(-j 2 pi k n / L) for k in bins."""
    n = torch.arange(length, dtype=torch.float64)
    k = torch.as_tensor(bins, dtype=torch.float64)
    theta = 2 * math.pi * torch.remainder(k[:, None] * n[None, :], length) / length
    cos, sin = torch.cos(theta), torch.sin(theta)
    if complex_input:
        re = torch.stack([cos, sin], 1)
        im = torch.stack([-sin, cos], 1)
    else:
        re, im = cos[:, None], -sin[:, None]
    return torch.cat([re, im], 0).float()  # (2K, C, L)


class GlobalSpectrum(nn.Module):
    """Full-length signal spectrum (256 centred bins) + envelope spectrum (bins 1..128) -> one D-dim token."""

    def __init__(self, d, lengths=(512, 1024), sig_bins=256, env_bins=128, ch=(32, 64, 128)):
        super().__init__()
        self.sig_bins, self.env_bins = sig_bins, env_bins
        for L in lengths:
            self.register_buffer(f'sig_{L}', dft_weights(L, range(-sig_bins // 2, sig_bins // 2), True), persistent=False)
            self.register_buffer(f'env_{L}', dft_weights(L, range(1, env_bins + 1), False), persistent=False)

        def tower():
            layers, cin = [], 1
            for c in ch:
                layers += [nn.Conv1d(cin, c, 5, 2, 2, bias=False), LenBN(c), nn.GELU()]
                cin = c
            return nn.Sequential(*layers)

        self.sig_tower, self.env_tower = tower(), tower()
        self.proj = nn.Linear(4 * ch[-1], d)
        self.type_embedding = nn.Parameter(torch.zeros(d))

    @staticmethod
    def _logpow(x, w, k, length):
        y = F.conv1d(x, w, stride=length).squeeze(-1)  # (B, 2K)
        return torch.log((y[:, :k] ** 2 + y[:, k:] ** 2) / length + 1e-6)

    def forward(self, x):  # x: (B, 2, L) normalised IQ
        length = x.shape[-1]
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            sig = self._logpow(x, getattr(self, f'sig_{length}'), self.sig_bins, length)
            env = x[:, :1] ** 2 + x[:, 1:] ** 2
            env = env - env.mean(-1, keepdim=True)
            envs = self._logpow(env, getattr(self, f'env_{length}'), self.env_bins, length)
            sig = sig - sig.mean(-1, keepdim=True)
            envs = envs - envs.mean(-1, keepdim=True)
        a = self.sig_tower(sig[:, None])
        b = self.env_tower(envs[:, None])
        feats = torch.cat([a.mean(-1), a.amax(-1), b.mean(-1), b.amax(-1)], 1)
        return self.proj(feats) + self.type_embedding  # (B, D)


class RepBlock(nn.Module):
    """GELU(BN(conv7 x) + BN(conv3 x) + BN(x)); fuse() merges into one conv7 with bias per input length."""

    def __init__(self, d):
        super().__init__()
        self.c7, self.b7 = nn.Conv1d(d, d, 7, 1, 3, bias=False), LenBN(d)
        self.c3, self.b3 = nn.Conv1d(d, d, 3, 1, 1, bias=False), LenBN(d)
        self.bid = LenBN(d)
        self.fused = None  # nn.ModuleDict {'512': conv, '1024': conv} after fuse()
        self.cur_len = 1024
        self.act = nn.GELU()

    def forward(self, x):
        if self.fused is not None:
            return self.act(self.fused[str(self.cur_len)](x))
        return self.act(self.b7(self.c7(x)) + self.b3(self.c3(x)) + self.bid(x))

    @staticmethod
    def _fold(w, bn, L):
        rm, rv = bn.stats(L)
        g = bn.weight / torch.sqrt(rv + bn.eps)
        return w * g[:, None, None], bn.bias - rm * g

    @torch.no_grad()
    def fuse(self):
        d = self.c7.in_channels
        ident = torch.zeros_like(self.c7.weight)
        ident[torch.arange(d), torch.arange(d), 3] = 1.0
        convs = {}
        for L in LenBN.LENGTHS:
            w7, b7 = self._fold(self.c7.weight, self.b7, L)
            w3, b3 = self._fold(F.pad(self.c3.weight, (2, 2)), self.b3, L)
            wi, bi = self._fold(ident, self.bid, L)
            conv = nn.Conv1d(d, d, 7, 1, 3, bias=True).to(w7.device)
            conv.weight.copy_(w7 + w3 + wi)
            conv.bias.copy_(b7 + b3 + bi)
            convs[str(L)] = conv
        self.fused = nn.ModuleDict(convs)
        for name in ['c7', 'b7', 'c3', 'b3', 'bid']:
            delattr(self, name)


class CrossDomainGate(nn.Module):
    """Bidirectional channel gating: time-mean of one stream gates the other (independent parameters per direction)."""

    def __init__(self, d):
        super().__init__()
        self.f2t = nn.Sequential(nn.Linear(d, d // 4), nn.ReLU(), nn.Linear(d // 4, d), nn.Sigmoid())
        self.t2f = nn.Sequential(nn.Linear(d, d // 4), nn.ReLU(), nn.Linear(d // 4, d), nn.Sigmoid())

    def forward(self, t, f):
        gt = self.f2t(f.mean(-1))[..., None]
        gf = self.t2f(t.mean(-1))[..., None]
        return t * gt, f * gf


class Attention(nn.Module):
    """Multi-head self-attention with per-head unit-RMS Q/K normalisation, written with matmul."""

    def __init__(self, d, heads):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.h, self.hd).permute(2, 0, 3, 1, 4).unbind(0)
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = F.normalize(q.float(), dim=-1, eps=1e-6) * math.sqrt(self.hd)
            k = F.normalize(k.float(), dim=-1, eps=1e-6) * math.sqrt(self.hd)
            att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.hd), -1)
        y = torch.matmul(att.to(v.dtype), v)
        return self.proj(y.transpose(1, 2).reshape(b, n, d))


class TransformerLayer(nn.Module):
    def __init__(self, d, heads, mlp_ratio=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, heads)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


def sinusoid_table(n, d):
    pos = torch.arange(n, dtype=torch.float64)[:, None]
    i = torch.arange(0, d, 2, dtype=torch.float64)
    ang = pos / (10000 ** (i / d))
    t = torch.zeros(n, d, dtype=torch.float64)
    t[:, 0::2], t[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return t.float()


def cls_head(d_in, d, n):
    return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d // 2), nn.GELU(), nn.Linear(d // 2, n))


class TFUniEM(nn.Module):
    TASK_HEADS = {0: 'mod_head', 1: 'pro_head', 2: 'param_head'}  # task index (modulation, protocol, parameter)

    def __init__(self, C=96, D=384, layers=6, heads=6, blocks=(2, 2, 4), n_combo=22, time_frontend='progressive'):
        """time_frontend: 'progressive' (stride-2 stem to L/4, three stages down to L/16, gate after each stage) or
        'patch16' (ablation E1, leader-style: k5 conv at full resolution, then one stride-16 patch conv straight
        to L/16; the time stream skips stages 1-2 and gates 1-2, the frequency stream is unchanged)."""
        super().__init__()
        assert time_frontend in ('progressive', 'patch16')
        self.time_frontend = time_frontend
        self.cfg = dict(C=C, D=D, layers=layers, heads=heads, blocks=list(blocks), n_combo=n_combo,
                        time_frontend=time_frontend)
        self._cur_len = None
        assert n_combo == PARAM_SLICES['combo'][1] - PARAM_SLICES['combo'][0]
        widths = [C, 2 * C, 4 * C]
        if time_frontend == 'progressive':
            self.t_stem = nn.Sequential(conv_bn(4, C // 2, 7, 2), conv_bn(C // 2, C, 3, 2))
        else:
            self.t_stem = nn.Sequential(conv_bn(4, C, 5, 1), conv_bn(C, 4 * C, 16, 16))
        self.stft = ConvSTFT(64, 4)
        self.f_stem = conv_bn(64, C, 3)
        self.t_stages, self.f_stages, self.gates = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.t_down, self.f_down = nn.ModuleList(), nn.ModuleList()
        patch = time_frontend == 'patch16'
        for i, (w, nb) in enumerate(zip(widths, blocks)):
            skip_t = patch and i < 2
            self.t_stages.append(nn.Identity() if skip_t else nn.Sequential(*[RepBlock(w) for _ in range(nb)]))
            self.f_stages.append(nn.Sequential(*[RepBlock(w) for _ in range(nb)]))
            self.gates.append(nn.Identity() if skip_t else CrossDomainGate(w))
            if i < 2:
                self.t_down.append(nn.Identity() if patch else conv_bn(w, 2 * w, 3, 2, act=False))
                self.f_down.append(conv_bn(w, 2 * w, 3, 2, act=False))
        self.merge = conv_bn(8 * C, D, 1)
        self.register_buffer('pos_table', sinusoid_table(64, D), persistent=False)
        self.global_spec = GlobalSpectrum(D)
        self.layers = nn.ModuleList([TransformerLayer(D, heads) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(D)
        self.pool_k, self.pool_v = nn.Linear(D, D), nn.Linear(D, D)
        self.len_embedding = nn.Parameter(torch.zeros(2, D))
        # task-specific
        self.queries = nn.Parameter(torch.randn(3, D) * 0.02)
        self.mod_head = cls_head(D + 1, D, N_MODULATION)
        self.pro_head = cls_head(D + 1, D, N_PROTOCOL)
        self.param_head = nn.Sequential(nn.LayerNorm(D + 1), nn.Linear(D + 1, D), nn.GELU(),
                                        nn.Linear(D, D // 2), nn.GELU(), nn.Linear(D // 2, N_PARAM_OUT))

    def task_specific_parameters(self):
        return {'queries': [self.queries], 'mod_head': list(self.mod_head.parameters()),
                'pro_head': list(self.pro_head.parameters()), 'param_head': list(self.param_head.parameters())}

    def set_length(self, L):
        for m in self.modules():
            if hasattr(m, 'cur_len'):
                m.cur_len = L

    def encode(self, iq, task):
        """iq: (B, L, 2) raw float32; task: (B,) long. Returns pooled features (B, D+1)."""
        B, L, _ = iq.shape
        if L != self._cur_len:
            self.set_length(L)
            self._cur_len = L
        with torch.autocast(device_type=iq.device.type, enabled=False):
            x = iq.float().transpose(1, 2)  # (B, 2, L)
            p = x[:, :1] ** 2 + x[:, 1:] ** 2
            rms = torch.sqrt(p.mean(-1, keepdim=True) + 1e-20)  # (B,1,1)
            x = x / rms
            s = torch.log(rms).reshape(B, 1)
            env = p / (rms ** 2)
            ramp = (torch.arange(L, device=iq.device, dtype=torch.float32) / 1024.0).expand(B, 1, L)
            spec = self.stft(x)
        t = self.t_stem(torch.cat([x, env, ramp], 1))
        f = self.f_stem(spec)
        patch = self.time_frontend == 'patch16'
        for i in range(3):
            t, f = self.t_stages[i](t), self.f_stages[i](f)
            if not (patch and i < 2):
                t, f = self.gates[i](t, f)
            if i < 2:
                t, f = self.t_down[i](t), self.f_down[i](f)
        tok = self.merge(torch.cat([t, f], 1)).transpose(1, 2)  # (B, L/16, D)
        tok = tok + self.pos_table[:tok.shape[1]].to(tok.dtype)
        g = self.global_spec(x)
        tok = torch.cat([tok, g[:, None].to(tok.dtype)], 1)
        for layer in self.layers:
            tok = layer(tok)
        tok = self.final_norm(tok)
        q = self.queries[task]  # (B, D)
        logits = torch.matmul(self.pool_k(tok), q[:, :, None]).squeeze(-1) / math.sqrt(tok.shape[-1])
        a = torch.softmax(logits.float(), -1).to(tok.dtype)
        z = torch.matmul(a[:, None], self.pool_v(tok)).squeeze(1)
        z = z + self.len_embedding[0 if L == 512 else 1].to(z.dtype)
        return torch.cat([z, s.to(z.dtype)], 1)

    def forward(self, iq, task):
        z = self.encode(iq, task)
        return {'modulation': self.mod_head(z), 'protocol': self.pro_head(z), 'parameter': self.param_head(z)}

    def fuse(self):
        for m in self.modules():
            if isinstance(m, RepBlock) and m.fused is None:
                m.fuse()
        self._cur_len = None  # force re-propagation of the current length to the new modules
        return self


def load_legacy_state(model, state_dict):
    """Load a checkpoint saved with plain BatchNorm (running_mean/var): both length slots get the old stats."""
    sd = {}
    for k, v in state_dict.items():
        if k.endswith('.running_mean'):
            for L in LenBN.LENGTHS:
                sd[k.replace('running_mean', f'rm_{L}')] = v
        elif k.endswith('.running_var'):
            for L in LenBN.LENGTHS:
                sd[k.replace('running_var', f'rv_{L}')] = v
        elif k.endswith('.num_batches_tracked'):
            for L in LenBN.LENGTHS:
                sd[k.replace('num_batches_tracked', f'nbt_{L}')] = v
        else:
            sd[k] = v
    model.load_state_dict(sd)
    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    specific = sum(p.numel() for ps in model.task_specific_parameters().values() for p in ps)
    per_task = {}
    shared = total - specific
    tp = model.task_specific_parameters()
    for task, head in [('modulation', 'mod_head'), ('protocol', 'pro_head'), ('parameter', 'param_head')]:
        own = sum(p.numel() for p in tp[head]) + model.queries.shape[1]
        per_task[task] = {'used': shared + own, 'shared_fraction_of_used': shared / (shared + own)}
    return {'total': total, 'shared': shared, 'task_specific': specific, 'shared_fraction': shared / total,
            'per_task': per_task}
