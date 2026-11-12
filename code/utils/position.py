"""Explicit sample, window-relative, and known-physical-time RoPE coordinates."""
import math
import torch


POSITION_DEFAULTS = dict(reference_length=1024, window_head_fraction=.5,
                         absolute_features=True, sample_rate_hz=0., reference_sample_rate_hz=1.)


def position_coordinates(lengths, count, patch, heads, mode='samples', options=None, sample_rates=None):
    options = {**POSITION_DEFAULTS, **(options or {})}
    if mode not in ['samples', 'window', 'dual', 'physical']:
        raise ValueError(f'Unknown position mode: {mode}')
    reference = options['reference_length']
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError('position_options.reference_length must be positive')
    centers = torch.arange(count, device=lengths.device).float() * patch + (patch - 1) / 2
    samples = (centers / patch)[None].expand(len(lengths), -1)
    coordinates = torch.stack((centers[None].expand(len(lengths), -1) / reference,
                               lengths[:, None].expand(-1, count).float() / reference), -1)
    if mode == 'samples':
        return samples, coordinates
    if mode == 'physical':
        if sample_rates is None:
            known = options['sample_rate_hz']
            if not math.isfinite(known) or known <= 0:
                raise ValueError('Physical RoPE requires real sample_rates; sampling rate cannot be inferred from length')
            sample_rates = torch.full((len(lengths),), known, device=lengths.device)
        rates = torch.as_tensor(sample_rates, device=lengths.device, dtype=torch.float32)
        if rates.shape != lengths.shape or not bool(torch.isfinite(rates).all() & (rates > 0).all()):
            raise ValueError('sample_rates must be one finite positive rate per record')
        reference_rate = options['reference_sample_rate_hz']
        if not math.isfinite(reference_rate) or reference_rate <= 0:
            raise ValueError('reference_sample_rate_hz must be finite and positive')
        scale = reference_rate / rates
        # Absolute coordinates must use the same physical scale as RoPE.
        return samples * scale[:, None], coordinates * scale[:, None, None]
    if not bool((lengths > 0).all()):
        raise ValueError('Window-relative RoPE requires positive lengths')
    window = samples * (reference / lengths.float())[:, None]
    if mode == 'window':
        return window, coordinates
    fraction = options['window_head_fraction']
    if heads < 2 or not 0 < fraction < 1:
        raise ValueError('Dual RoPE needs at least two heads and a fraction in (0,1)')
    window_heads = min(heads - 1, max(1, int(heads * fraction + .5)))
    positions = samples[:, None].expand(-1, heads, -1).clone()
    positions[:, -window_heads:] = window[:, None]
    return positions, coordinates
