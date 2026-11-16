"""Paired original/augmented supervision, with parameter/protocol protection."""
import math
import torch
from torch.nn import functional as F


def eligible_modulation(batch, class_indices):
    return ((batch['modulation'] >= 0) & (batch['protocol'] < 0) &
            ~batch['target_valid'].any(-1) & torch.isin(batch['modulation'],
                torch.as_tensor(class_indices, device=batch['iq'].device)))


def augmented_view(iq, lengths, seed, phase_probability=1., shift_probability=1.):
    generator = torch.Generator(device=iq.device).manual_seed(seed)
    angle = (torch.rand(len(iq), device=iq.device, generator=generator)*2-1)*math.pi
    angle *= torch.rand(len(iq), device=iq.device, generator=generator) < phase_probability
    c, s = angle.cos()[:, None], angle.sin()[:, None]
    x = torch.stack([iq[..., 0]*c-iq[..., 1]*s, iq[..., 0]*s+iq[..., 1]*c], -1)
    offsets = (torch.rand(len(iq), device=iq.device, generator=generator)*lengths).long()
    offsets *= torch.rand(len(iq), device=iq.device, generator=generator) < shift_probability
    time = torch.arange(iq.shape[1], device=iq.device)[None]
    index = (time+offsets[:, None]) % lengths[:, None]
    return x.gather(1, index[..., None].expand(-1, -1, 2)) * (time < lengths[:, None])[..., None]


def paired_correction(raw_logits, augmented_logits, labels, selected, *,
                      augmented_weight=.5, consistency_weight=.05, ramp=1.):
    """Correction to mean raw CE over ALL modulation records.

    Selected records retain (1-alpha)*CE(raw) + alpha*CE(aug), plus JS.
    All other labels and tasks retain their original loss and denominator.
    """
    count = (labels >= 0).sum().clamp_min(1)
    raw = raw_logits[selected].float()
    other = augmented_logits.float()
    y = labels[selected]
    if not len(y):
        return raw_logits.float().sum()*0, {}
    hard_raw = F.cross_entropy(raw, y, reduction='none')
    hard_other = F.cross_entropy(other, y, reduction='none')
    log_a, log_b = raw.log_softmax(-1), other.log_softmax(-1)
    log_m = torch.logaddexp(log_a, log_b)-math.log(2)
    js = .5*((log_a.exp()*(log_a-log_m)).sum(-1)+(log_b.exp()*(log_b-log_m)).sum(-1))
    delta = ramp*(augmented_weight*(hard_other-hard_raw)+consistency_weight*js).sum()/count
    return delta, dict(view_raw_ce=float(hard_raw.detach().mean()),
        view_augmented_ce=float(hard_other.detach().mean()), view_js=float(js.detach().mean()),
        view_fraction=float(selected.sum()/count), view_ramp=float(ramp))
