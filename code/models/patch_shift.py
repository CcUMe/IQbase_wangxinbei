"""Local offset pooling with the ORIGINAL patch projection, one shared backbone.

Offsets are sample-grid offsets, not carrier phase or recovered symbol timing.
The zero-initialized residual preserves a loaded model's original predictions.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


PATCH_SHIFT_DEFAULTS = dict(enabled=False, stride=4, bottleneck=64,
                            max_scale=0.2, dropout=0.05, variance_epsilon=1e-6)


def patch_shift_options(values=None, patch_size=16):
    result = {**PATCH_SHIFT_DEFAULTS, **(values or {})}
    if set(result) != set(PATCH_SHIFT_DEFAULTS):
        raise ValueError('Unknown patch_shift setting')
    if type(result['enabled']) is not bool:
        raise ValueError('patch_shift.enabled must be boolean')
    for name in ['stride', 'bottleneck']:
        if type(result[name]) is not int or result[name] < 1:
            raise ValueError(f'patch_shift.{name} must be a positive integer')
    if result['enabled'] and (patch_size % result['stride'] or
            result['stride'] >= patch_size or (patch_size-result['stride']) % 2):
        raise ValueError('patch_shift requires an aligned finer stride dividing patch_size')
    for name in ['max_scale', 'dropout', 'variance_epsilon']:
        if not math.isfinite(result[name]):
            raise ValueError(f'patch_shift.{name} must be finite')
    if not 0 < result['max_scale'] <= 1 or not 0 <= result['dropout'] < 1 or result['variance_epsilon'] <= 0:
        raise ValueError('Invalid patch_shift scale/dropout/epsilon')
    return result


class PatchShiftAdapter(nn.Module):
    def __init__(self, width, patch_size, options):
        super().__init__()
        self.options = patch_shift_options(options, patch_size)
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(2*width, self.options['bottleneck'])
        self.dropout = nn.Dropout(self.options['dropout'])
        self.up = nn.Linear(self.options['bottleneck'], width, bias=False)
        nn.init.zeros_(self.up.weight)
        self.last_update_ratio = None

    def forward(self, hidden, original_tokens, projection, point_valid):
        p, stride = self.patch_size, self.options['stride']
        # Reuse the original learned P16 kernel at P4. No second time encoder.
        dense = F.conv1d(hidden*point_valid[:, None], projection.weight,
                         projection.bias, stride=stride, padding=(p-stride)//2)
        phases = p//stride
        b, d, n = hidden.shape[0], original_tokens.shape[-1], original_tokens.shape[1]
        if dense.shape[-1] != n*phases:
            raise ValueError('Patch-shift token geometry mismatch')
        # Dense centers average to the original patch center at every position.
        values = dense.transpose(1, 2).reshape(b, n, phases, d)
        values = self.norm(values).float()
        mean = values.mean(2)
        variance = (values-mean[:, :, None]).square().mean(2)
        description = torch.cat([mean, variance.clamp_min(self.options['variance_epsilon']).sqrt()], -1)
        delta = self.up(self.dropout(F.gelu(self.down(description.to(dense.dtype))))).tanh()
        with torch.no_grad():
            scale = original_tokens.float().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        delta = delta * scale * self.options['max_scale']
        valid = point_valid.reshape(b, n, p).any(-1)
        delta = delta * valid[..., None]
        self.last_update_ratio = (delta.detach().float().square().sum()/
            (original_tokens.detach().float().square().sum().clamp_min(1e-12))).sqrt()
        return delta.to(original_tokens.dtype)
