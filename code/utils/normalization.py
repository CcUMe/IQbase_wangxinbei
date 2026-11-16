"""Signal scales shared by dictionary construction and signal encoders."""
import math
import torch


def observation_scale(iq, visible, method='rms', percentile=99., epsilon=1e-6):
    if method not in ['rms', 'amplitude_percentile']:
        raise ValueError('Unknown observation normalization: ' + method)
    if not math.isfinite(percentile) or not 0 < percentile <= 100:
        raise ValueError('Amplitude percentile must be in (0,100]')
    with torch.autocast(device_type=iq.device.type, enabled=False):
        x=iq.float()
        if method=='rms':
            # Keep historical arithmetic ordering exactly unchanged.
            masked=x*visible[...,None]
            scale=(masked.square().sum((1,2))/visible.sum(1).clamp_min(1)).sqrt()
        else:
            # Never let masked targets or padding influence the student scale.
            amplitude=torch.linalg.vector_norm(x,dim=-1).masked_fill(~visible,float('nan'))
            scale=torch.nanquantile(amplitude,percentile/100.,dim=1)
            scale=torch.nan_to_num(scale,nan=0.)
        return scale.clamp_min(epsilon)
