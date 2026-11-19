"""Parameter-specific statistics of masked shared tokens, in sample coordinates."""
import math

import torch


PARAMETER_READOUT_DEFAULTS = dict(enabled=False, hidden_width=128, dropout=0.1,
                                  include_std=True, temporal_bins=8)


def parameter_readout_options(value=None):
    value = value or {}
    if set(value) - set(PARAMETER_READOUT_DEFAULTS):
        raise ValueError('Unknown parameter_readout option')
    result = {**PARAMETER_READOUT_DEFAULTS, **value}
    if (type(result['enabled']) is not bool or type(result['include_std']) is not bool
            or type(result['hidden_width']) is not int or result['hidden_width'] < 1
            or type(result['temporal_bins']) is not int or not 0 <= result['temporal_bins'] <= 64
            or not math.isfinite(result['dropout']) or not 0 <= result['dropout'] < 1):
        raise ValueError('Invalid parameter_readout settings')
    return result


def parameter_feature_width(width, settings):
    return width * (3 if settings['include_std'] else 2) + (settings['temporal_bins'] + 4 if settings['temporal_bins'] else 0)


def parameter_features(h, valid, weights, patch_size, reference_length, settings):
    """Query statistics over sample centers; no raw IQ or target access.

    These are learned attention statistics, not measured pulse boundaries/Hz.
    """
    h = h.float() * valid[..., None]
    weights = weights.float() * valid[:, None]
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    local = torch.einsum('bjn,bnd->bjd', weights, h)
    mean = h.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
    features = [local, mean[:, None].expand(-1, weights.shape[1], -1)]
    if settings['include_std']:
        variance = torch.einsum('bjn,bnd->bjd', weights, h.square()) - local.square()
        features.append(variance.clamp_min(1e-8).sqrt())
    bins = settings['temporal_bins']
    if bins:
        position = (torch.arange(h.shape[1], device=h.device).float() + .5) * patch_size / reference_length
        center = (weights * position).sum(-1, keepdim=True)
        spread = (weights * (position - center).square()).sum(-1, keepdim=True).clamp_min(1e-8).sqrt()
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1, keepdim=True)
        entropy = entropy / valid.sum(1).float().clamp_min(2).log()[:, None, None]
        duration = torch.log1p(valid.sum(1).float() * patch_size) / math.log1p(reference_length)
        duration = duration[:, None, None].expand(-1, weights.shape[1], -1)
        index = (position * bins).long().clamp(0, bins - 1)
        masses = weights.new_zeros((*weights.shape[:2], bins))
        masses.scatter_add_(-1, index[None, None].expand_as(weights), weights)
        features.extend([center, spread, entropy, duration, masses])
    return torch.cat(features, -1)
