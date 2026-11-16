"""Local I/Q geometry using the original temporal filters, not another encoder.

For the two existing real filters A/B, form z=(A*I+B*Q)+j(A*Q-B*I).
This linear lift is phase equivariant. Its patch-local power moments and lag
correlations are invariant to a constant phase rotation; the original raw-IQ
path, scale, spectrum and positions remain available to all three tasks.
"""
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F

IQ_GEOMETRY_DEFAULTS=dict(enabled=False,bottleneck=64,lags=[1,4],max_scale=.1,
                          dropout=0.,epsilon=1e-6)


def iq_geometry_options(values=None,patch_size=16):
    options={**deepcopy(IQ_GEOMETRY_DEFAULTS),**(values or {})}
    if set(options)!=set(IQ_GEOMETRY_DEFAULTS):raise ValueError('Unknown iq_geometry option')
    if type(options['enabled']) is not bool or type(options['bottleneck']) is not int or options['bottleneck']<1:
        raise ValueError('Invalid iq_geometry enabled/bottleneck')
    lags=options['lags']
    if not isinstance(lags,list) or any(type(v) is not int or not 0<v<patch_size for v in lags) or len(set(lags))!=len(lags):
        raise ValueError('iq_geometry lags must be distinct sample offsets within one patch')
    if not all(isinstance(options[k],(float,int)) and math.isfinite(options[k]) for k in ['max_scale','dropout','epsilon']):
        raise ValueError('iq_geometry scalar options must be finite')
    if not 0<options['max_scale']<=1 or not 0<=options['dropout']<1 or options['epsilon']<=0:
        raise ValueError('Invalid iq_geometry scale/dropout/epsilon')
    return options


class IQGeometry(nn.Module):
    def __init__(self,width,patch_size,options):
        super().__init__();self.options=iq_geometry_options(options,patch_size)
        self.patch_size=patch_size
        self.descriptor_width=(width//2)*(2+2*len(self.options['lags']))
        self.norm=nn.LayerNorm(self.descriptor_width)
        self.down=nn.Linear(self.descriptor_width,self.options['bottleneck'])
        self.dropout=nn.Dropout(self.options['dropout'])
        self.up=nn.Linear(self.options['bottleneck'],width,bias=False)
        nn.init.zeros_(self.up.weight)

    def describe(self,iq,visible,original_weight):
        """Use ONLY the already-masked normalized IQ and shared Conv1d weights."""
        b,t,_=iq.shape;p=self.patch_size;epsilon=self.options['epsilon']
        if t%p:raise ValueError('IQ geometry requires a whole number of patches')
        with torch.autocast(device_type=iq.device.type,enabled=False):
            weight=original_weight[:,:2].float()
            quadrature=torch.stack((-weight[:,1],weight[:,0]),1)
            lifted=F.conv1d((iq.float()*visible[...,None]).transpose(1,2),
                           torch.cat([weight,quadrature]),padding=weight.shape[-1]//2)
            real,imag=lifted.chunk(2,1);power=real.square()+imag.square()
            def average(value,mask):
                m=mask[:,None].reshape(b,1,t//p,p)
                return (value.reshape(b,value.shape[1],t//p,p)*m).sum(-1)/m.sum(-1).clamp_min(1)
            mean=average(power,visible)
            centered=power.reshape(b,power.shape[1],t//p,p)-mean[...,None]
            variance=average(centered.square().reshape_as(power),visible)
            features=[torch.log1p(mean),torch.log1p((variance+epsilon).sqrt()-math.sqrt(epsilon))]
            for lag in self.options['lags']:
                previous_real=F.pad(real,(lag,0))[:,:,:t]
                previous_imag=F.pad(imag,(lag,0))[:,:,:t]
                pair=visible & F.pad(visible[:,:-lag],(lag,0))
                cross_real=average(real*previous_real+imag*previous_imag,pair)
                cross_imag=average(imag*previous_real-real*previous_imag,pair)
                previous_power=previous_real.square()+previous_imag.square()
                denominator=(average(power,pair)*average(previous_power,pair)+epsilon).sqrt()
                features.extend([(cross_real/denominator).clamp(-1,1),(cross_imag/denominator).clamp(-1,1)])
            descriptor=torch.cat(features,1).transpose(1,2)
            token_visible=visible.reshape(b,t//p,p).any(-1)
            return descriptor*token_visible[...,None],token_visible

    def forward(self,iq,visible,original_weight,time_tokens,patch_weight):
        with torch.autocast(device_type=iq.device.type,enabled=False):
            descriptor,token_visible=self.describe(iq,visible,original_weight)
            delta=self.up(self.dropout(F.gelu(self.down(self.norm(descriptor)))))
            # Calibrate with invariant local energy and the shared patch-filter
            # gain. Raw time-token RMS depends on absolute phase and would
            # reintroduce the very nuisance this descriptor removes.
            energy=torch.expm1(descriptor[...,:original_weight.shape[0]]).clamp_min(0).mean(-1,keepdim=True)
            gain=patch_weight.float().square().sum((1,2)).mean().sqrt()
            scale=((energy+self.options['epsilon']).sqrt()*gain).detach()
            delta=delta.tanh()*scale*self.options['max_scale']*token_visible[...,None]
        return delta.to(time_tokens.dtype)
