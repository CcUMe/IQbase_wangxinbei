"""Shared fine-resolution distribution encoding before P16 compression.

Learned local descriptors, soft assignments and zeroth/first/second moments.
One original time/frequency encoder is retained. No true class, SNR, carrier,
clock or sampling-rate metadata is required at inference.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F

DENSITY_DEFAULTS=dict(enabled=False,width=32,codewords=16,lags=[0,4,16],shared_scale=.1)


def density_options(values=None):
    options={**deepcopy(DENSITY_DEFAULTS),**(values or {})}
    if set(options)!=set(DENSITY_DEFAULTS):raise ValueError('Unknown density encoding option')
    if type(options['enabled'])!=bool or type(options['width'])!=int or type(options['codewords'])!=int:
        raise ValueError('Density enabled must be boolean and dimensions integers')
    if options['width']<8 or options['codewords']<4 or not options['lags'] or options['lags'][0]!=0 or any(type(x)!=int or x<0 for x in options['lags']):
        raise ValueError('Invalid density dimensions/lags')
    if not 0<=options['shared_scale']<=1:raise ValueError('Invalid shared density scale')
    return options


class DensityEncoding(nn.Module):
    def __init__(self,channels,options):
        super().__init__();self.options=density_options(options);d=self.options['width'];k=self.options['codewords']
        self.local=nn.Conv1d(channels*len(self.options['lags']),d,1)
        self.assignment=nn.Conv1d(d,k,1)
        self.centres=nn.Parameter(torch.randn(k,d)*.2)
        self.output_dim=k*(1+2*d)
        self.normalization=nn.LayerNorm(self.output_dim)
        self.shared_projection=nn.Linear(self.output_dim,channels,bias=False)
        nn.init.zeros_(self.shared_projection.weight)
        self.descriptor=None;self.inject_shared=True

    def forward(self,hidden,valid):
        original=hidden
        with torch.autocast(device_type=hidden.device.type,enabled=False):
            h=hidden.float()*valid[:,None]
            pieces=[h if lag==0 else F.pad(h,(lag,0))[:,:,:h.shape[-1]] for lag in self.options['lags']]
            # Do not insert a per-point normalization that erases amplitude.
            feature=self.local(torch.cat(pieces,1)).tanh()
            assignment=self.assignment(feature).softmax(1)*valid[:,None]
            count=valid.sum(1).clamp_min(1).float()
            mass=assignment.sum(-1)/count[:,None]
            first=torch.einsum('bkt,bdt->bkd',assignment,feature)/count[:,None,None]
            second=torch.einsum('bkt,bdt->bkd',assignment,feature.square())/count[:,None,None]
            centres=self.centres[None]
            residual=first-mass[...,None]*centres
            variance=(second-2*first*centres+mass[...,None]*centres.square()).clamp_min(0)
            descriptor=torch.cat([mass[...,None],residual,variance],-1).flatten(1)
            self.descriptor=self.normalization(descriptor)
            update=self.shared_projection(self.descriptor).tanh()
            rms=(h.square().sum((1,2))/(count*h.shape[1])).sqrt().detach()
            if not self.inject_shared:update=update*0
            update=update*rms[:,None]*self.options['shared_scale']
        return (original+update[:,:,None].to(original.dtype))*valid[:,None]


class DensityClassifier(nn.Sequential):
    """One classifier, original feature projection plus new density columns.

Original 0/1/2 state keys remain unchanged. It produces a single set of logits;
there is no prediction averaging or second independent classifier at inference.
"""
    def __init__(self,original,descriptor_dim):
        super().__init__(*list(original.children()))
        self.density_projection=nn.Linear(descriptor_dim,self[0].out_features,bias=False)
        nn.init.zeros_(self.density_projection.weight)
        self.use_context=True

    def forward(self,features,descriptor):
        context=features if self.use_context else features*0
        hidden=self[0](context)+self.density_projection(descriptor.to(features.dtype))
        return self[2](self[1](hidden))
