"""Shared multi-resolution IQ/frequency features and supervised task readouts.

Engineering adaptation of MST/IQFormer/CDSiT/MCCSAN ideas, not a paper replica.
All positions are sample coordinates; no symbol period or sample rate is inferred.
"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F


PYRAMID_DEFAULTS = dict(
    enabled=False, width=192, heads=3, stem_complex_channels=48, local_blocks=2,
    transformer_depth=1, ffn_hidden=512, dropout=0.1, attention_dropout=0.1,
    residual_scale=0.1,
    mid=dict(kernel=16, stride=8, padding=4, groups=16),
    fine=dict(kernel=8, stride=4, padding=2, groups=32))
READOUT_DEFAULTS = dict(projection_width=256, modulation_hidden=256, protocol_hidden=128,
                        parameter_hidden=128, dropout=0.1, variance_epsilon=1e-6)


def pyramid_options(options=None):
    result = deepcopy(PYRAMID_DEFAULTS)
    for key, value in (options or {}).items():
        if key not in result:
            raise ValueError(f'Unknown pyramid option: {key}')
        if isinstance(result[key], dict):
            if not isinstance(value, dict) or set(value) - set(result[key]):
                raise ValueError(f'Invalid pyramid.{key}')
            result[key].update(value)
        else:
            result[key] = value
    return result


def validate_pyramid(options, readout, patch_size):
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(type(options['enabled']) is bool, 'pyramid.enabled must be boolean')
    for key in ['width', 'heads', 'stem_complex_channels', 'local_blocks', 'transformer_depth', 'ffn_hidden']:
        require(type(options[key]) is int and options[key] > 0, f'pyramid.{key} must be a positive integer')
    d, heads = options['width'], options['heads']
    require(d % 2 == 0 and d % heads == 0 and (d // heads) % 2 == 0,
            'pyramid width must be even and attention head width must be even')
    require(all(math.isfinite(options[k]) and 0 <= options[k] < 1 for k in ['dropout', 'attention_dropout']),
            'pyramid dropout must be in [0,1)')
    require(math.isfinite(options['residual_scale']) and options['residual_scale'] > 0, 'pyramid.residual_scale must be positive')
    for name in ['mid', 'fine']:
        branch = options[name]
        require(all(type(branch[k]) is int and branch[k] > 0 for k in ['kernel', 'stride', 'groups']),
                f'pyramid.{name} kernel/stride/groups must be positive integers')
        require(type(branch['padding']) is int and branch['padding'] >= 0, f'pyramid.{name}.padding must be nonnegative')
        require(d % branch['groups'] == 0, f'pyramid.{name}.groups must divide pyramid.width')
        require(branch['stride'] % 2 == 0 and branch['kernel'] == 2 * branch['stride']
                and branch['padding'] == branch['stride'] // 2 and 512 % branch['stride'] == 0,
                f'pyramid.{name} requires kernel=2*stride, padding=stride/2, even stride dividing 512')
    require(patch_size > options['mid']['stride'] > options['fine']['stride'],
            'pyramid requires coarse patch_size > mid.stride > fine.stride')
    for key in ['projection_width', 'modulation_hidden', 'protocol_hidden', 'parameter_hidden']:
        require(type(readout[key]) is int and readout[key] > 0, f'multiscale_readout.{key} must be positive integer')
    require(math.isfinite(readout['dropout']) and 0 <= readout['dropout'] < 1, 'multiscale_readout.dropout must be in [0,1)')
    require(math.isfinite(readout['variance_epsilon']) and readout['variance_epsilon'] > 0,
            'multiscale_readout.variance_epsilon must be positive')


def masked_mean(x, valid):
    return (x * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)


class ComplexConv1d(nn.Module):
    """Real/imaginary channel halves; two shared real kernels implement W*z."""
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0):
        super().__init__()
        self.real = nn.Conv1d(in_channels, out_channels, kernel, stride, padding, bias=False)
        self.imag = nn.Conv1d(in_channels, out_channels, kernel, stride, padding, bias=False)

    def forward(self, x):
        real, imag = x.chunk(2, dim=1)
        return torch.cat((self.real(real) - self.imag(imag), self.real(imag) + self.imag(real)), 1)


class LocalBlock(nn.Module):
    def __init__(self, width, groups, dropout, residual_scale):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.point_in = nn.Conv1d(width, width, 1)
        self.grouped = nn.Conv1d(width, width, 3, padding=1, groups=groups)
        self.point_out = nn.Conv1d(width, width, 1)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.full((width,), residual_scale))

    def forward(self, x, valid):
        mask = valid[:, None]
        h = self.norm(x).transpose(1, 2) * mask
        h = F.silu(self.point_in(h)) * mask
        h = F.silu(self.grouped(h)) * mask
        h = self.point_out(h).transpose(1, 2)
        return (x + self.dropout(h) * self.scale) * valid[..., None]


class FineTransformer(nn.Module):
    def __init__(self, options, rope_base):
        super().__init__()
        # Local import avoids a module cycle; RFModel is fully defined at construction.
        from models.rf_model import RotaryAttention
        d = options['width']
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attention = RotaryAttention(d, options['heads'], options['attention_dropout'],
                                         rope_base, qk_normalization=True)
        self.ffn_in = nn.Linear(d, options['ffn_hidden'] * 2)
        self.ffn_out = nn.Linear(options['ffn_hidden'], d)
        self.dropout = nn.Dropout(options['dropout'])

    def forward(self, x, valid, positions):
        x = x + self.dropout(self.attention(self.norm1(x), valid, positions))
        gate, value = self.ffn_in(self.norm2(x)).chunk(2, -1)
        x = x + self.dropout(self.ffn_out(self.dropout(F.silu(gate) * value)))
        return x * valid[..., None]


class CrossAttention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.head_width, self.dropout = heads, width // heads, dropout
        self.qnorm, self.knorm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.query, self.key, self.value, self.out = [nn.Linear(width, width) for _ in range(4)]

    def forward(self, query, memory, valid, temperature=1.):
        b, nq, d = query.shape
        q = self.query(self.qnorm(query)).reshape(b, nq, self.heads, self.head_width).transpose(1, 2)
        normalized = self.knorm(memory)
        k = self.key(normalized).reshape(b, -1, self.heads, self.head_width).transpose(1, 2)
        v = self.value(normalized).reshape(b, -1, self.heads, self.head_width).transpose(1, 2)
        with torch.autocast(device_type=query.device.type, enabled=False):
            q = F.normalize(q.float(), dim=-1, eps=1e-6) * math.sqrt(self.head_width) / temperature
            k = F.normalize(k.float(), dim=-1, eps=1e-6) * math.sqrt(self.head_width)
        h = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v,
            attn_mask=valid[:, None, None], dropout_p=self.dropout if self.training else 0.)
        return self.out(h.transpose(1, 2).reshape(b, nq, d))


class FineBranch(nn.Module):
    def __init__(self, options, name, patch_size, rope_base):
        super().__init__()
        self.spec, self.reference_stride = options[name], patch_size
        d, stem = options['width'], options['stem_complex_channels']
        self.patch = ComplexConv1d(stem, d // 2, self.spec['kernel'], self.spec['stride'], self.spec['padding'])
        self.input_norm = nn.LayerNorm(d)
        self.local = nn.ModuleList([LocalBlock(d, self.spec['groups'], options['dropout'], options['residual_scale'])
                                    for _ in range(options['local_blocks'])])
        self.blocks = nn.ModuleList([FineTransformer(options, rope_base) for _ in range(options['transformer_depth'])])
        self.norm = nn.LayerNorm(d)

    def forward(self, stem, lengths):
        x = self.patch(stem).transpose(1, 2)
        centers = torch.arange(x.shape[1], device=x.device).float() * self.spec['stride']
        centers = centers - self.spec['padding'] + (self.spec['kernel'] - 1) / 2
        valid = torch.arange(x.shape[1], device=x.device)[None] < (lengths // self.spec['stride'])[:, None]
        x = F.silu(self.input_norm(x)) * valid[..., None]
        for block in self.local:
            x = block(x, valid)
        positions = (centers / self.reference_stride)[None].expand(len(lengths), -1)
        for block in self.blocks:
            x = block(x, valid, positions)
        return self.norm(x) * valid[..., None], valid, centers


class MultiResolutionPyramid(nn.Module):
    """Registered under backbone; every task reads all three shared scales."""
    def __init__(self, coarse_width, patch_size, rope_base, options):
        super().__init__()
        self.options, self.patch_size = options, patch_size
        d, stem = options['width'], options['stem_complex_channels']
        self.stem = ComplexConv1d(1, stem, 5, padding=2)
        self.stem_norm = nn.LayerNorm(stem * 2)
        self.branches = nn.ModuleList([FineBranch(options, name, patch_size, rope_base) for name in ['mid', 'fine']])
        widths = [coarse_width, d, d]
        self.memory_projection = nn.ModuleList([nn.Linear(w, d) for w in widths])
        self.frequency_projection = nn.Linear(coarse_width, d)
        self.position = nn.Sequential(nn.Linear(2, d), nn.Tanh())
        self.scale_embedding = nn.Embedding(3, d)
        self.time_frequency = nn.ModuleList([CrossAttention(d, options['heads'], options['attention_dropout']) for _ in range(2)])
        self.frequency_scale = nn.Parameter(torch.full((2, d), options['residual_scale']))
        self.cross_scale = nn.ModuleList([CrossAttention(d, options['heads'], options['attention_dropout']) for _ in range(3)])
        self.summary_out = nn.ModuleList([nn.Linear(d, w) for w in widths])
        self.summary_scale = nn.Parameter(torch.full((3, d), options['residual_scale']))

    def coordinate_embedding(self, centers, lengths):
        coordinates = torch.stack((centers[None].expand(len(lengths), -1) / 1024,
                                   lengths[:, None].expand(-1, len(centers)).float() / 1024), -1)
        return self.position(coordinates)

    def forward(self, coarse, coarse_valid, context, lengths):
        x = context['iq']
        point_valid = context['point_valid']
        stem = self.stem(x.transpose(1, 2)).transpose(1, 2)
        stem = (F.silu(self.stem_norm(stem)) * point_valid[..., None]).transpose(1, 2)
        coarse_centers = torch.arange(coarse.shape[1], device=x.device).float() * self.patch_size + (self.patch_size - 1) / 2
        features, masks, centers = [coarse], [coarse_valid], [coarse_centers]
        frequency = self.frequency_projection(context['frequency']) + self.coordinate_embedding(coarse_centers, lengths)
        for i, branch in enumerate(self.branches):
            h, valid, center = branch(stem, lengths)
            pos = self.coordinate_embedding(center, lengths)
            h = (h + self.time_frequency[i](h + pos, frequency, coarse_valid) * self.frequency_scale[i]) * valid[..., None]
            features.append(h)
            masks.append(valid)
            centers.append(center)
        memories = [(projection(h) + self.coordinate_embedding(center, lengths) + self.scale_embedding.weight[i]) * valid[..., None]
                    for i, (projection, h, center, valid) in enumerate(zip(self.memory_projection, features, centers, masks))]
        summaries = [masked_mean(h, valid) for h, valid in zip(memories, masks)]
        fused = []
        for i in range(3):
            other = [j for j in range(3) if j != i]
            memory = torch.cat([summaries[i][:, None]] + [memories[j] for j in other], 1)
            valid = torch.cat([torch.ones_like(masks[i][:, :1])] + [masks[j] for j in other], 1)
            updated = summaries[i] + self.cross_scale[i](summaries[i][:, None], memory, valid).squeeze(1) * self.summary_scale[i]
            fused.append(self.summary_out[i](updated))
        return dict(features=features, valid=masks, summaries=fused, memory=torch.cat(memories, 1),
                    memory_valid=torch.cat(masks, 1), scale=context['scale'])


class AttentiveStatistics(nn.Module):
    def __init__(self, width, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.score = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, max(8, width // 4)),
                                   nn.Tanh(), nn.Linear(max(8, width // 4), 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, h, valid, return_weights=False):
        h = h.float()
        weights = self.score(h).squeeze(-1).masked_fill(~valid, -torch.inf).softmax(-1)
        mean = (weights[..., None] * h).sum(1)
        variance = (weights[..., None] * (h - mean[:, None]).square()).sum(1)
        std = variance.clamp_min(self.epsilon).sqrt()
        return (mean, std, weights) if return_weights else (mean, std)


class ModulationReadout(nn.Module):
    def __init__(self, widths, classes, options):
        super().__init__()
        p = options['projection_width']
        self.pool = nn.ModuleList([AttentiveStatistics(w, options['variance_epsilon']) for w in widths])
        self.projections = nn.ModuleList([nn.Sequential(nn.Linear(w * 3, p), nn.LayerNorm(p)) for w in widths])
        self.classifier = nn.Sequential(nn.Linear(p * 3, options['modulation_hidden']), nn.GELU(),
            nn.Dropout(options['dropout']), nn.Linear(options['modulation_hidden'], classes))

    def forward(self, pyramid):
        descriptors = []
        for pool, projection, h, valid, summary in zip(self.pool, self.projections, pyramid['features'], pyramid['valid'], pyramid['summaries']):
            mean, std = pool(h, valid)
            descriptors.append(projection(torch.cat((summary.float(), mean, std), -1)))
        return self.classifier(torch.cat(descriptors, -1) / math.sqrt(3))


class ProtocolReadout(nn.Module):
    def __init__(self, widths, classes, options):
        super().__init__()
        p = options['projection_width']
        self.projections = nn.ModuleList([nn.Sequential(nn.Linear(w, p), nn.LayerNorm(p)) for w in widths])
        self.classifier = nn.Sequential(nn.Linear(p * 3, options['protocol_hidden']), nn.GELU(),
            nn.Dropout(options['dropout']), nn.Linear(options['protocol_hidden'], classes))

    def forward(self, pyramid):
        return self.classifier(torch.cat([p(h.float()) for p, h in zip(self.projections, pyramid['summaries'])], -1) / math.sqrt(3))


class ParameterQueries(nn.Module):
    def __init__(self, widths, width, heads, dropout, temperature, carrier_dim=0):
        super().__init__()
        self.temperature = temperature
        self.queries = nn.Embedding(7, width)
        self.carrier_dim = carrier_dim
        if carrier_dim:
            from models.signal_mechanisms import SignalAwareCrossAttention
            self.attention = SignalAwareCrossAttention(width, heads, dropout)
        else:
            self.attention = CrossAttention(width, heads, dropout)
        self.global_projection = nn.Linear(sum(widths) + 2, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, pyramid, lengths):
        memory = pyramid['memory'].float()
        query = self.queries.weight[None].expand(memory.shape[0], -1, -1)
        physical = None
        if self.carrier_dim:
            context, physical = self.attention(query, memory, pyramid['memory_valid'], self.temperature,
                                               pyramid['signal_memory'])
        else:
            context = self.attention(query, memory, pyramid['memory_valid'], self.temperature)
        local = self.norm(query + context)
        global_input = torch.cat([h.float() for h in pyramid['summaries']] +
                                 [pyramid['scale'].float().log()[:, None], lengths.float()[:, None] / 1024], -1)
        global_features = self.global_projection(global_input)
        features = [local, global_features[:, None].expand(-1, 7, -1)]
        if physical is not None: features.append(physical)
        return torch.cat(features, -1)
