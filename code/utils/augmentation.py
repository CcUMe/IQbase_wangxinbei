"""Training-only transforms that preserve all seven C01 target meanings."""
import math

import torch


def rotate_phase(iq, angles):
    """Multiply each complex waveform by one unit phasor; never shift time/frequency."""
    cosine, sine = angles.cos()[:, None], angles.sin()[:, None]
    real, imag = iq[..., 0], iq[..., 1]
    return torch.stack((real * cosine - imag * sine, real * sine + imag * cosine), -1)


def augment_iq(iq, settings):
    probability = settings['phase_rotation_probability']
    if probability == 0:
        return iq
    angles = torch.rand(iq.shape[0], device=iq.device) * (2 * math.pi)
    if probability < 1:
        angles = angles * (torch.rand(iq.shape[0], device=iq.device) < probability)
    return rotate_phase(iq, angles)


def augment_supervised_iq(batch, settings):
    """Additional time-origin invariance only for records with modulation labels alone."""
    iq = augment_iq(batch['iq'], settings)
    options = settings.get('modulation_only', {})
    phase = options.get('phase_rotation_probability', 0.)
    shift = options.get('circular_shift_probability', 0.)
    if not (phase or shift):
        return iq
    eligible = (batch['modulation'] >= 0) & (batch['protocol'] < 0) & ~batch['target_valid'].any(-1)
    if phase:
        selected = eligible & (torch.rand(len(iq), device=iq.device) < phase)
        angles = torch.rand(len(iq), device=iq.device) * (2 * math.pi) * selected
        iq = rotate_phase(iq, angles)
    if shift:
        selected = eligible & (torch.rand(len(iq), device=iq.device) < shift)
        lengths = batch['lengths']
        offsets = (torch.rand(len(iq), device=iq.device) * lengths).long()
        coordinate = torch.arange(iq.shape[1], device=iq.device)[None]
        index = (coordinate + offsets[:, None]) % lengths[:, None]
        shifted = iq.gather(1, index[..., None].expand(-1, -1, 2))
        shifted = shifted * (coordinate < lengths[:, None])[..., None]
        iq = torch.where(selected[:, None, None], shifted, iq)
    return iq
