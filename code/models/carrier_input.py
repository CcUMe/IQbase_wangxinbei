"""One extra input view in the ORIGINAL time projection, no second encoder.

Conv(raw)+Conv(canonical) is equivalent to a single Conv on concatenated input
channels, with the original bias. Raw IQ/spectrum/energy/timing paths remain.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models.carrier_recovery import fourth_power_refinement


class CarrierInput(nn.Module):
    def __init__(self,channels,hidden,phase_features,mode):
        super().__init__()
        if mode not in ['raw','lag1','m4']:raise ValueError('Unknown carrier input mode')
        self.mode=mode;self.phase_features=phase_features
        self.projection=nn.Conv1d(channels,hidden,5,padding=2,bias=False)
        nn.init.zeros_(self.projection.weight)
        self.regularization=None;self.last_diagnostics={}

    def forward(self,features,lengths):
        time=torch.arange(features.shape[-1],device=features.device)[None]
        valid=time<lengths[:,None]
        with torch.autocast(device_type=features.device.type,enabled=False):
            x=features[:,:2].float().transpose(1,2)*valid[...,None]
            z=x-(x.sum(1)/lengths[:,None])[:,None];z=z*valid[...,None]
            a,b=z[:,1:],z[:,:-1]
            real=(a*b).sum(-1).sum(1);imag=(a[...,1]*b[...,0]-a[...,0]*b[...,1]).sum(1)
            denominator=(a.square().sum((1,2))*b.square().sum((1,2))).sqrt().clamp_min(1e-8)
            coherence=(real.square()+imag.square()).sqrt()/denominator
            frequency=torch.where(coherence>.25,torch.atan2(imag,real+1e-12)/(2*math.pi),0.).detach()
            refinement=torch.zeros_like(frequency);accepted=torch.zeros_like(frequency,dtype=torch.bool)
            if self.mode=='m4':
                refined,diagnostics=fourth_power_refinement(x,lengths,frequency)
                refinement=refined-frequency;accepted=diagnostics['accepted'];frequency=refined
            if self.mode=='raw':canonical=features.float()*valid[:,None]
            else:
                angle=2*math.pi*frequency[:,None]*time
                c,s=angle.cos(),angle.sin()
                x=torch.stack([x[...,0]*c+x[...,1]*s,x[...,1]*c-x[...,0]*s],-1)*valid[...,None]
                amplitude=x.square().sum(-1,keepdim=True).clamp_min(1e-12).sqrt()
                pieces=[x,amplitude]
                if self.phase_features:
                    previous=F.pad(x[:,:-1],(0,0,1,0))
                    real=(x*previous).sum(-1);imag=x[...,1]*previous[...,0]-x[...,0]*previous[...,1]
                    reliability=(amplitude[...,0]*F.pad(amplitude[:,:-1,0],(1,0))).clamp_max(1)
                    magnitude=(real.square()+imag.square()).sqrt().clamp_min(1e-6)
                    pieces.append(torch.stack([real,imag],-1)/magnitude[...,None]*reliability[...,None])
                canonical=torch.cat(pieces,-1).transpose(1,2)*valid[:,None]
        self.last_diagnostics=dict(coherence=coherence.detach(),frequency=frequency.detach(),refinement=refinement.detach(),accepted=accepted.detach())
        self.regularization=self.projection.weight.sum()*0
        return self.projection(canonical)*valid[:,None]

    def summary(self):
        d=self.last_diagnostics
        return dict(coherence_mean=float(d['coherence'].mean()),frequency_abs_mean=float(d['frequency'].abs().mean()),
            m4_acceptance=float(d['accepted'].float().mean()),m4_delta_abs_mean=float(d['refinement'].abs().mean()),
            projection_norm=float(self.projection.weight.detach().norm()))


def attach_carrier_input(model,options):
    if hasattr(model.backbone,'reception'):raise ValueError('Receiver already attached')
    original=model.backbone.time[0]
    with torch.random.fork_rng(devices=[]):
        model.backbone.reception=CarrierInput(original.in_channels,original.out_channels,model.config['phase_features'],options['mode'])
    context={}
    def prepare(module,args,kwargs):
        context['lengths']=args[1] if len(args)>1 else kwargs['lengths']
        if kwargs.get('patch_mask') is not None or (len(args)>2 and args[2] is not None):raise ValueError('Supervised carrier input only')
    def inject(module,args,output):
        return output+model.backbone.reception(args[0],context['lengths']).to(output.dtype)
    def clear(module,args,output):context.clear()
    model.register_forward_pre_hook(prepare,with_kwargs=True)
    original.register_forward_hook(inject)
    model.register_forward_hook(clear,always_call=True)
    return model
