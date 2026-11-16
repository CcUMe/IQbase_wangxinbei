"""Training-only modulation regularization; protocol/parameter loss unchanged."""
import torch
from torch.nn import functional as F


def symmetric_kl(a, b):
    """Mean symmetric KL, summed over classes, gradients through both views."""
    log_a, log_b = a.float().log_softmax(-1), b.float().log_softmax(-1)
    return .5*((log_a.exp()-log_b.exp())*(log_a-log_b)).sum(-1).mean()


def modulation_correction(first, batch, smoothing=0., second=None, kl_weight=0.):
    """Replace the first-view raw modulation CE, leaving other objectives alone.

    Second view averages two supervised losses. The caller retains a matching
    double-CE/no-KL control to distinguish consistency from gradient averaging.
    """
    valid = batch['modulation'] >= 0
    zero = first['modulation'].float().sum()*0
    if not valid.any():
        return zero, {'mod_ce':0.,'mod_kl':0.}
    labels = batch['modulation'][valid]
    a = first['modulation'][valid].float()
    original = F.cross_entropy(a, labels)
    ce = F.cross_entropy(a, labels, label_smoothing=smoothing)
    kl = zero
    if second is not None:
        b = second['modulation'][valid].float()
        ce = .5*(ce+F.cross_entropy(b,labels,label_smoothing=smoothing))
        if kl_weight:
            kl = symmetric_kl(a,b)
    elif kl_weight:
        raise ValueError('R-Drop requires a second stochastic forward')
    return ce-original+kl_weight*kl, {'mod_ce':float(ce.detach()),'mod_kl':float(kl.detach())}
