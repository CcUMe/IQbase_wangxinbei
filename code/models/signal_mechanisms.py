"""Local context and explicit signal-value routes for the unified backbone."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.multiresolution import CrossAttention, AttentiveStatistics, masked_mean


def signal_values(iq, valid, epsilon=1e-6):
    """I/Q, relative log-power/change, and reliable lag-one phase components.

    Values come from the one RMS-normalized input, without token LayerNorm.
    No symbol period, sample rate, SNR or clean reference is inferred.
    """
    with torch.autocast(device_type=iq.device.type, enabled=False):
        x = iq.float() * valid[..., None]
        previous = F.pad(x[:, :-1], (0, 0, 1, 0))
        power = x.square().sum(-1)
        log_power = power.log1p()
        pair_valid = valid & F.pad(valid[:, :-1], (1, 0), value=False)
        product = (power * previous.square().sum(-1)).clamp_min(epsilon ** 2).sqrt()
        real = x[..., 0] * previous[..., 0] + x[..., 1] * previous[..., 1]
        imag = x[..., 1] * previous[..., 0] - x[..., 0] * previous[..., 1]
        phase = torch.stack((real, imag), -1) / product[..., None] * product.clamp_max(1)[..., None]
        delta = (log_power - F.pad(log_power[:, :-1], (1, 0))) * pair_valid
        return torch.cat((x, log_power[..., None], delta[..., None], phase * pair_valid[..., None]), -1) * valid[..., None]


def fold_values(values, stride, flatten=False):
    b, n, d = values.shape
    if n % stride:
        raise ValueError('Signal value folding requires aligned lengths')
    folded = values.reshape(b, n // stride, stride, d)
    return folded.flatten(-2) if flatten else folded.mean(2)


class LocalWindowAttention(nn.Module):
    """Window self-attention with zero-padded shifts, never circular wrapping."""
    def __init__(self, width, window, shift, dropout):
        super().__init__()
        self.window, self.shift, self.dropout = window, shift, dropout
        self.heads = max(1, width // 32) if width % 32 == 0 else 1
        self.head_width = width // self.heads
        self.qkv, self.projection = nn.Linear(width, width * 3), nn.Linear(width, width)

    def forward(self, x, valid):
        b, n, d = x.shape
        right = (-(n + self.shift)) % self.window
        padded = F.pad(x * valid[..., None], (0, 0, self.shift, right))
        mask = F.pad(valid, (self.shift, right), value=False).reshape(-1, self.window)
        windows = padded.reshape(-1, self.window, d)
        q, k, v = self.qkv(windows).reshape(-1, self.window, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4).unbind(0)
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = F.normalize(q.float(), dim=-1, eps=1e-6) * math.sqrt(self.head_width)
            k = F.normalize(k.float(), dim=-1, eps=1e-6) * math.sqrt(self.head_width)
        # All-invalid windows get a dummy key and are zeroed at the output.
        safe = mask.clone()
        safe[:, 0] |= ~mask.any(-1)
        h = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v,
            attn_mask=safe[:, None, None], dropout_p=self.dropout if self.training else 0.)
        h = self.projection(h.transpose(1, 2).reshape(-1, self.window, d)) * mask[..., None]
        h = h.reshape(b, -1, d)[:, self.shift:self.shift + n]
        return h * valid[..., None]


class SignalAwareCrossAttention(CrossAttention):
    """Use the same query weights for semantic values and unnormalized signals."""
    def forward(self, query, memory, valid, temperature=1., signal=None):
        if signal is None:
            raise ValueError('Signal-aware attention requires aligned signal values')
        b, nq, d = query.shape
        if signal.shape[:2] != memory.shape[:2]:
            raise ValueError('Signal memory and semantic memory must align')
        q = self.query(self.qnorm(query)).reshape(b, nq, self.heads, self.head_width).transpose(1, 2)
        normalized = self.knorm(memory)
        k = self.key(normalized).reshape(b, -1, self.heads, self.head_width).transpose(1, 2)
        v = self.value(normalized).reshape(b, -1, self.heads, self.head_width).transpose(1, 2)
        # Only seven queries: explicit FP32 weights are small and also expose a
        # direct physical-value readout without re-normalizing those values.
        with torch.autocast(device_type=query.device.type, enabled=False):
            q = F.normalize(q.float(), dim=-1, eps=1e-6) * math.sqrt(self.head_width) / temperature
            k = F.normalize(k.float(), dim=-1, eps=1e-6) * math.sqrt(self.head_width)
            scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_width)
            weights = scores.masked_fill(~valid[:, None, None], -torch.inf).softmax(-1)
            weights = F.dropout(weights, self.dropout, self.training)
            h = weights @ v.float()
            physical = (weights @ signal.float()[:, None]).mean(1)
        semantic = self.out(h.transpose(1, 2).reshape(b, nq, d))
        return semantic, physical


class DeepAnchoredModulationReadout(nn.Module):
    """One classifier: deep descriptor plus a bounded detail correction."""
    def __init__(self, widths, classes, options, mechanism):
        super().__init__()
        p = options['projection_width']
        self.pool = nn.ModuleList([AttentiveStatistics(w, options['variance_epsilon']) for w in widths])
        self.projections = nn.ModuleList([nn.Sequential(nn.Linear(w * 3, p), nn.LayerNorm(p)) for w in widths])
        self.detail = nn.Sequential(nn.Linear(p * 2 + 12, p), nn.GELU(), nn.Linear(p, p))
        self.max_ratio = mechanism['detail_max_ratio']
        fraction = mechanism['detail_initial_ratio'] / self.max_ratio
        self.detail_logit = nn.Parameter(torch.tensor(math.log(fraction / (1 - fraction))))
        self.epsilon = options['variance_epsilon']
        self.classifier = nn.Sequential(nn.Linear(p, options['modulation_hidden']), nn.GELU(),
            nn.Dropout(options['dropout']), nn.Linear(options['modulation_hidden'], classes))

    def descriptors(self, pyramid):
        descriptors, fine_weights = [], None
        for i, (pool, projection, h, valid) in enumerate(zip(self.pool, self.projections, pyramid['features'], pyramid['valid'])):
            mean, std, weights = pool(h, valid, return_weights=True)
            # The main descriptor depends only on Transformer output. Local
            # corrections may use cross-layer summaries, but cannot replace it.
            summary = masked_mean(h.float(), valid) if i == 0 else pyramid['summaries'][i].float()
            descriptors.append(projection(torch.cat((summary, mean, std), -1)))
            if i == 2: fine_weights = weights
        values = pyramid['signal_features'].float()
        if values.shape[:2] != fine_weights.shape:
            raise ValueError('Fine-level signal values must align with modulation pooling')
        mean = (fine_weights[..., None] * values).sum(1)
        std = (fine_weights[..., None] * (values - mean[:, None]).square()).sum(1).clamp_min(self.epsilon).sqrt()
        local = torch.cat((descriptors[1], descriptors[2], mean, std), -1)
        correction = self.detail(local).tanh() * (self.max_ratio * self.detail_logit.sigmoid())
        return descriptors[0], correction

    def forward(self, pyramid):
        deep, correction = self.descriptors(pyramid)
        return self.classifier(deep + correction)
