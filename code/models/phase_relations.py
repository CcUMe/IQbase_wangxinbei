"""One small shared local phase-relation adapter for the original pyramid.

All lags are sample intervals, not symbol periods. No carrier correction or
discarding of original IQ is performed. Features are computed once, then pooled
on the exact P16/P8/P4 receptive-field grids before the existing encoders.
"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F


PHASE_RELATION_DEFAULTS = dict(
    enabled=False, width=64, lags=[1, 2, 4, 8, 16, 32, 64],
    phase_orders=[1, 2, 4], curvature_orders=[1, 2],
    magnitude_floor=0.1, phase_epsilon=1e-6, variance_epsilon=1e-6,
    local_kernel=3, dropout=0.1, residual_scale=0.1, max_residual_scale=0.25)


def phase_relation_options(options=None):
    result = deepcopy(PHASE_RELATION_DEFAULTS)
    if options is not None:
        if not isinstance(options, dict) or set(options)-set(result):
            raise ValueError('Unknown phase_relations option')
        result.update(deepcopy(options))
    if type(result['enabled']) is not bool:
        raise ValueError('phase_relations.enabled must be boolean')
    for key in ['width', 'local_kernel']:
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f'phase_relations.{key} must be a positive integer')
    if result['local_kernel'] % 2 != 1:
        raise ValueError('phase_relations.local_kernel must be odd')
    for key in ['lags', 'phase_orders', 'curvature_orders']:
        values = result[key]
        if (not isinstance(values, list) or not values or
                any(type(v) is not int or v <= 0 for v in values) or values != sorted(set(values))):
            raise ValueError(f'phase_relations.{key} must be sorted distinct positive integers')
    if result['lags'][-1] >= 512:
        raise ValueError('phase_relations.lags must be smaller than 512 samples')
    for key in ['phase_orders', 'curvature_orders']:
        if not set(result[key]) <= {1, 2, 4, 8}:
            raise ValueError(f'phase_relations.{key} supports orders 1,2,4,8')
    for key in ['magnitude_floor', 'phase_epsilon', 'variance_epsilon', 'max_residual_scale']:
        if not isinstance(result[key], (int, float)) or not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f'phase_relations.{key} must be positive and finite')
    if not math.isfinite(result['dropout']) or not 0 <= result['dropout'] < 1:
        raise ValueError('phase_relations.dropout must be in [0,1)')
    if not math.isfinite(result['residual_scale']) or not 0 < result['residual_scale'] < result['max_residual_scale']:
        raise ValueError('phase_relations requires 0 < residual_scale < max_residual_scale')
    return result


def delay(x, lag):
    """Causal shift along samples; never wraps the tail to the head."""
    return F.pad(x, (0, 0, lag, 0))[:, :x.shape[1]]


def complex_product(a, b):
    return torch.stack((a[..., 0]*b[..., 0]-a[..., 1]*b[..., 1],
                        a[..., 1]*b[..., 0]+a[..., 0]*b[..., 1]), -1)


class PhaseRelations(nn.Module):
    """Local harmonics/curvature plus explicit magnitude/history weights.

    Magnitude weights are numerical guards, not estimated SNR/confidence.
    Constant global phase cancels; constant frequency remains in lag relations
    and cancels in curvature. Original IQ/CFO/energy paths remain available.
    """
    def __init__(self, options):
        super().__init__()
        self.options = options
        self.channels = len(options['lags']) * (2*len(options['phase_orders']) + 2*len(options['curvature_orders']) + 2)

    def forward(self, normalized_iq, point_valid):
        o = self.options
        with torch.autocast(device_type=normalized_iq.device.type, enabled=False):
            x = normalized_iq.float().masked_fill(~point_valid[..., None], 0.)
            power = x.square().sum(-1, keepdim=True)
            epsilon4 = o['phase_epsilon']**4
            floor2 = o['magnitude_floor']**2

            def relation(lag):
                previous = delay(x, lag)
                conjugate = torch.stack((previous[..., 0], -previous[..., 1]), -1)
                product = complex_product(x, conjugate)
                squared = power*delay(power, lag)
                unit = product / squared.clamp_min(epsilon4).sqrt()
                history = point_valid[..., None] & delay(point_valid[..., None], lag)
                weight = squared/(squared+floor2) * history
                return unit*history, weight

            adjacent, adjacent_weight = relation(1)
            values = []
            for lag in o['lags']:
                unit, weight = relation(lag)
                previous = delay(adjacent, lag)
                conjugate = torch.stack((previous[..., 0], -previous[..., 1]), -1)
                curvature = complex_product(adjacent, conjugate)
                curvature_weight = adjacent_weight*delay(adjacent_weight, lag)
                for signal, reliability, orders in [(unit, weight, o['phase_orders']),
                                                     (curvature, curvature_weight, o['curvature_orders'])]:
                    harmonic = signal
                    for order in [1, 2, 4, 8]:
                        if order in orders:
                            values.append(harmonic*reliability)
                        if order < max(orders):
                            harmonic = complex_product(harmonic, harmonic)
                values.extend([weight, curvature_weight])
            return torch.cat(values, -1) * point_valid[..., None]


class PhaseRelationAdapter(nn.Module):
    def __init__(self, widths, patch_size, pyramid, options):
        super().__init__()
        self.options = options
        self.relations = PhaseRelations(options)
        d, kernel = options['width'], options['local_kernel']
        self.input = nn.Conv1d(self.relations.channels, d, 1, bias=False)
        self.local = nn.Conv1d(d, d, kernel, padding=kernel//2, groups=d, bias=False)
        self.output = nn.Conv1d(d, d, 1, bias=False)
        self.norm = nn.LayerNorm(d*2)
        self.projections = nn.ModuleList([nn.Linear(d*2, width, bias=False) for width in widths])
        for layer in self.projections:
            nn.init.zeros_(layer.weight)
        ratio = options['residual_scale']/options['max_residual_scale']
        self.gain_logits = nn.Parameter(torch.full((3,), math.log(ratio/(1-ratio))))
        self.dropout = nn.Dropout(options['dropout'])
        self.grids = [(patch_size, patch_size, 0)] + [(pyramid[name]['kernel'], pyramid[name]['stride'], pyramid[name]['padding'])
                                                                  for name in ['mid', 'fine']]

    def forward(self, normalized_iq, point_valid, lengths):
        raw = self.relations(normalized_iq, point_valid)
        mask = point_valid[:, None]
        h = F.silu(self.input(raw.transpose(1, 2))) * mask
        h = F.silu(self.local(h)) * mask
        h = self.output(h) * mask
        updates = []
        # Keep moments/tanh/normalization in FP32, also under AMP.
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.float()
            gains = self.gain_logits.float().sigmoid()*self.options['max_residual_scale']
            for i, (kernel, stride, padding) in enumerate(self.grids):
                count = F.avg_pool1d(mask.float(), kernel, stride, padding)
                mean = F.avg_pool1d(h, kernel, stride, padding)/count.clamp_min(1/kernel)
                second = F.avg_pool1d(h.square(), kernel, stride, padding)/count.clamp_min(1/kernel)
                std = (second-mean.square()).clamp_min(self.options['variance_epsilon']).sqrt()
                moments = torch.cat((mean, std), 1).transpose(1, 2)
                valid = torch.arange(moments.shape[1], device=h.device)[None] < lengths[:, None]//stride
                update = self.projections[i](self.dropout(self.norm(moments))).tanh()*gains[i]
                updates.append(update * valid[..., None])
        return updates
