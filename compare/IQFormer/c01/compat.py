"""The two timm layer utilities used upstream, without torchvision dependencies.

DropPath uses timm's per-sample Bernoulli mask and keep-probability scaling.
trunc_normal_ is PyTorch's equivalent initializer. Upstream imports register_model
but does not use it; registering with the external timm registry is unnecessary.
"""
import torch
from torch import nn
from torch.nn.init import trunc_normal_


class DropPath(nn.Module):
    def __init__(self, drop_prob=0., scale_by_keep=True):
        super().__init__()
        self.drop_prob, self.scale_by_keep = drop_prob, scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        if keep > 0. and self.scale_by_keep:
            mask.div_(keep)
        return x * mask


def register_model(function):
    return function
