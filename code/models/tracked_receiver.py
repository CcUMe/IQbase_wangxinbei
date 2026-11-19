"""Masked-view complex filtering and iterative timing hypotheses on sample coordinates.

This is a learnable reception hypothesis bank, not an oracle symbol synchronizer.
The original waveform path and P8 grid remain unchanged. No sampling rate, class
label, clean waveform, or unmasked teacher feature is an input to this module.
"""
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F


RECEIVER_DEFAULTS = {
    'enabled': False,
    'filter': {'taps': 17, 'max_residual': .25},
    'timing': {'periods': [2., 4., 8., 16., 32., 64.], 'phases': [0., .5],
               'iterations': 2, 'hidden': 32, 'global_log_rate_limit': .35,
               'local_log_rate_limit': .15, 'max_proportional_gain': .06,
               'max_integral_gain': .002},
    'readout': {'width': 96, 'dropout': .05, 'max_scale': .2},
    'epsilon': 1e-6,
}


def receiver_options(values=None):
    result = deepcopy(RECEIVER_DEFAULTS)
    for key, value in (values or {}).items():
        if key not in result:
            raise ValueError('Unknown receiver_frontend option: ' + key)
        if isinstance(result[key], dict):
            if not isinstance(value, dict) or set(value)-set(result[key]):
                raise ValueError('Invalid receiver_frontend.' + key)
            result[key].update(value)
        else:
            result[key] = deepcopy(value)
    f, t, r = (result[k] for k in ['filter', 'timing', 'readout'])
    if type(result['enabled']) is not bool:
        raise ValueError('receiver_frontend.enabled must be boolean')
    if type(f['taps']) is not int or not 3 <= f['taps'] <= 65 or f['taps'] % 2 != 1:
        raise ValueError('receiver_frontend.filter.taps must be odd, between 3 and 65')
    for name, sequence in [('periods', t['periods']), ('phases', t['phases'])]:
        if (not isinstance(sequence, list) or not sequence or len(sequence) > 16 or
                any(type(x) not in [int, float] or not math.isfinite(x) for x in sequence)
                or len(set(sequence)) != len(sequence)):
            raise ValueError('Invalid receiver_frontend.timing.' + name)
    if any(x < 1 for x in t['periods']) or any(not 0 <= x < 1 for x in t['phases']):
        raise ValueError('Receiver periods are positive sample intervals; phases must be in [0,1)')
    if type(t['iterations']) is not int or not 1 <= t['iterations'] <= 4:
        raise ValueError('Receiver timing iterations must be between 1 and 4')
    if any(type(x) is not int or x < 1 for x in [t['hidden'], r['width']]):
        raise ValueError('Receiver hidden widths must be positive integers')
    scalars = [f['max_residual'], t['global_log_rate_limit'], t['local_log_rate_limit'],
               t['max_proportional_gain'], t['max_integral_gain'], r['dropout'],
               r['max_scale'], result['epsilon']]
    if any(type(x) not in [int, float] or not math.isfinite(x) for x in scalars):
        raise ValueError('Receiver scalar options must be finite numbers')
    if not (0 < f['max_residual'] <= 1 and 0 <= t['global_log_rate_limit'] <= .7
            and 0 <= t['local_log_rate_limit'] <= .5
            and 0 < t['max_proportional_gain'] <= .3
            and 0 < t['max_integral_gain'] <= .05
            and 0 <= r['dropout'] < 1 and 0 < r['max_scale'] <= 1
            and 0 < result['epsilon'] <= .01):
        raise ValueError('Receiver filter, timing, readout or epsilon outside supported range')
    return result


class TrackedReceiver(nn.Module):
    def __init__(self, backbone_width, patch_size, options=None):
        super().__init__()
        self.options = receiver_options(options)
        self.patch_size = patch_size
        f, t, r = (self.options[k] for k in ['filter', 'timing', 'readout'])
        count, width = len(t['periods']), r['width']
        self.register_buffer('periods', torch.tensor(t['periods']), persistent=False)
        self.register_buffer('phases', torch.tensor(t['phases']), persistent=False)
        self.taps = nn.Parameter(torch.zeros(count, 2, f['taps']))
        self.rate_predictor = nn.Sequential(nn.Linear(14, t['hidden']), nn.GELU(),
                                            nn.Linear(t['hidden'], count))
        nn.init.zeros_(self.rate_predictor[-1].weight)
        nn.init.zeros_(self.rate_predictor[-1].bias)
        # Half of each configured maximum is the initial gain. The gains learn
        # jointly, but both the integral correction and sample intervals stay bounded.
        self.loop_gains = nn.Parameter(torch.zeros(count, len(t['phases']), 2))
        self.embedding = nn.Sequential(nn.Linear(11, width), nn.GELU(),
                                       nn.Linear(width, width), nn.LayerNorm(width))
        self.hypothesis_embedding = nn.Parameter(torch.zeros(count, len(t['phases']), width))
        self.score = nn.Linear(width, 1)
        self.gate = nn.Linear(width, 1)
        nn.init.zeros_(self.score.weight); nn.init.zeros_(self.score.bias)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.dropout = nn.Dropout(r['dropout'])
        self.up = nn.Linear(width, backbone_width, bias=False)
        nn.init.zeros_(self.up.weight)
        self.last_diagnostics = {}

    def describe(self, x, visible, lengths):
        """Phase-invariant controls, with denominators independent of batch padding."""
        eps = self.options['epsilon']
        power = x.square().sum(-1)
        count = visible.sum(1).clamp_min(1)
        mean = (power * visible).sum(1) / count
        var = ((power-mean[:, None]).square()*visible).sum(1)/count
        features = [torch.log1p(mean), torch.log1p(var)]
        for lag in [1, 2, 4, 8, 16]:
            if lag >= x.shape[1]:
                features.extend([mean*0, mean*0])
                continue
            previous = F.pad(x[:, :-lag], (0, 0, lag, 0))
            pair = visible & F.pad(visible[:, :-lag], (lag, 0))
            real = (x*previous).sum(-1)
            imag = x[..., 1]*previous[..., 0]-x[..., 0]*previous[..., 1]
            a = (power*pair).sum(1)
            b = (previous.square().sum(-1)*pair).sum(1)
            denominator = (a*b).clamp_min(eps*eps).sqrt()
            features.extend([(real*pair).sum(1)/denominator, (imag*pair).sum(1)/denominator])
        features.extend([count/lengths.clamp_min(1), lengths.float().log2()/10])
        return torch.stack(features, -1)

    def filter(self, x, visible):
        f = self.options['filter']; eps = self.options['epsilon']
        taps = f['max_residual']*self.taps.float().tanh()
        impulse = torch.zeros_like(taps); impulse[:, 0, f['taps']//2] = 1
        taps = taps + impulse
        taps = taps/taps.square().sum((1, 2), keepdim=True).clamp_min(eps).sqrt()
        a, b = taps.unbind(1)
        weights = torch.cat([torch.stack([a, -b], 1), torch.stack([b, a], 1)], 0)
        real, imag = F.conv1d(x.transpose(1, 2), weights, padding=f['taps']//2).chunk(2, 1)
        support = F.conv1d(visible[:, None].float(), torch.ones(1, 1, f['taps'], device=x.device),
                           padding=f['taps']//2).squeeze(1)/f['taps']
        return torch.stack([real, imag], -1), support

    @staticmethod
    def sample(x, support, positions):
        """Linear interpolation without wrapping, extrapolation or x==0 masking."""
        # x [B,T,2], positions [B,H,K]. Clamp only for safe indexing; the
        # separate support mask removes any clamped/extrapolated observation.
        left = positions.floor().long()
        fraction = positions-left
        shape = positions.shape
        def gather(values, index):
            if values.ndim == 2:
                return values.gather(1, index.clamp(0, values.shape[1]-1).flatten(1)).reshape(shape)
            return values.gather(1, index.clamp(0, values.shape[1]-1).flatten(1)[..., None].expand(-1, -1, 2)).reshape(*shape, 2)
        a, b = gather(x, left), gather(x, left+1)
        value = a+(b-a)*fraction[..., None]
        coverage = torch.minimum(gather(support, left), gather(support, left+1))
        in_bounds = (positions >= 0) & (positions < x.shape[1]-1)
        return value*in_bounds[..., None], coverage*in_bounds

    def track(self, x, support, period, number):
        o = self.options['timing']; eps = self.options['epsilon']
        # The maximum allocation covers the smallest allowed interval. Longer
        # hypotheses have smaller arrays, instead of padding all clocks to period=2.
        bound = o['global_log_rate_limit']+o['local_log_rate_limit']+o['max_proportional_gain']
        count = max(3, math.ceil(x.shape[1]/(o['periods'][number]*math.exp(-bound)))+2)
        grid = torch.arange(count, device=x.device).float()
        origin = period[:, None]*self.phases[None]
        position = origin[..., None]+period[:, None, None]*grid
        interval = period[:, None, None].expand_as(position)
        gains = self.loop_gains[number].float().sigmoid()
        kp = o['max_proportional_gain']*gains[:, 0]
        ki = o['max_integral_gain']*gains[:, 1]
        error = torch.zeros_like(position)
        quality = torch.zeros_like(position)
        for _ in range(o['iterations']):
            middle, cm = self.sample(x, support, position)
            early, ce = self.sample(x, support, position-interval/2)
            late, cl = self.sample(x, support, position+interval/2)
            quality = torch.minimum(cm, torch.minimum(ce, cl))
            observed = quality >= 1-1e-5
            energy = early.square().sum(-1)+middle.square().sum(-1)+late.square().sum(-1)
            # A Gardner-inspired early/mid/late statistic, not a calibrated
            # physical clock-error label. Missing student observations do not update the loop.
            error = (((early-late)*middle).sum(-1)/(energy+eps)).clamp(-1, 1)*observed
            integral = o['local_log_rate_limit']*torch.tanh(torch.cumsum(error*ki[None, :, None], -1))
            interval = period[:, None, None]*torch.exp(integral+error*kp[None, :, None])
            position = origin[..., None]+F.pad(torch.cumsum(interval[..., :-1], -1), (1, 0))
        samples, coverage = self.sample(x, support, position)
        valid = coverage >= 1-1e-5
        power = samples.square().sum(-1)
        previous = F.pad(samples[..., :-1, :], (0, 0, 1, 0))
        pair = valid & F.pad(valid[..., :-1], (1, 0))
        denominator = (power*previous.square().sum(-1)).clamp_min(eps*eps).sqrt()
        real = (samples*previous).sum(-1)/denominator*pair
        imag = (samples[..., 1]*previous[..., 0]-samples[..., 0]*previous[..., 1])/denominator*pair
        # Features are formed for each hypothesis BEFORE interpolation/fusion;
        # distinct complex sampling phases are never averaged into one waveform.
        features = torch.stack([samples[..., 0], samples[..., 1], torch.log1p(power), real, imag,
                                torch.log1p(power)-torch.log1p(previous.square().sum(-1)),
                                position/1024, interval.clamp_min(eps).log2()/6,
                                coverage, error, quality], -1)*valid[..., None]
        return features, position, valid, interval

    @staticmethod
    def align(values, positions, valid, centers):
        queries = centers[None, None].expand(positions.shape[0], positions.shape[1], -1).contiguous()
        indices = torch.searchsorted(positions.detach().contiguous(), queries)
        inside = (indices > 0) & (indices < positions.shape[-1])
        right = indices.clamp(1, positions.shape[-1]-1); left = right-1
        a = positions.gather(-1, left); b = positions.gather(-1, right)
        fraction = ((queries-a)/(b-a).clamp_min(1e-6)).clamp(0, 1)
        va = values.gather(-2, left[..., None].expand(-1, -1, -1, values.shape[-1]))
        vb = values.gather(-2, right[..., None].expand(-1, -1, -1, values.shape[-1]))
        allowed = inside & valid.gather(-1, left) & valid.gather(-1, right)
        result = va+(vb-va)*fraction[..., None]
        return result*allowed[..., None], allowed

    def forward(self, x, visible, lengths, original_tokens):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()*visible[..., None]
            descriptor = self.describe(x, visible, lengths)
            rates = self.periods[None]*torch.exp(self.options['timing']['global_log_rate_limit']*
                                                self.rate_predictor(descriptor).tanh())
            filtered, support = self.filter(x, visible)
            centers = torch.arange(original_tokens.shape[1], device=x.device)*self.patch_size+(self.patch_size-1)/2
            aligned, available, rate_changes = [], [], []
            for number in range(len(self.periods)):
                features, positions, valid, interval = self.track(filtered[:, number], support, rates[:, number], number)
                embedded = self.embedding(features)+self.hypothesis_embedding[number][None, :, None]
                embedded = embedded*valid[..., None]
                value, allowed = self.align(embedded, positions, valid, centers)
                aligned.append(value); available.append(allowed)
                rate_changes.append((interval/rates[:, number, None, None]-1).abs().mean().detach())
            values = torch.cat(aligned, 1).permute(0, 2, 1, 3)
            valid = torch.cat(available, 1).transpose(1, 2)
            token_valid = centers[None] < lengths[:, None]
            valid = valid & token_valid[..., None]
            scores = self.score(values).squeeze(-1).masked_fill(~valid, -1e4)
            weights = scores.softmax(-1)*valid
            weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-6)
            pooled = (values*weights[..., None]).sum(2)
            delta = self.up(self.dropout(pooled)).tanh()*self.gate(pooled).sigmoid()
            scale = original_tokens.float().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt().detach()
            delta = delta*scale*self.options['readout']['max_scale']*valid.any(-1)[..., None]
            self.last_diagnostics = {
                'update_ratio': (delta.square().sum()/original_tokens.float().square().sum().clamp_min(1e-12)).sqrt().detach(),
                'token_coverage': (valid.any(-1).float().sum()/token_valid.sum().clamp_min(1)).detach(),
                'mean_rate_adjustment': torch.stack(rate_changes).mean(),
                'hypothesis_entropy': (-(weights*weights.clamp_min(1e-8).log()).sum(-1)*token_valid).sum().detach()/token_valid.sum().clamp_min(1),
            }
        return delta.to(original_tokens.dtype)
