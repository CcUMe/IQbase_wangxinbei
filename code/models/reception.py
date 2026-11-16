"""Optional research-only symbol reception memory for a single RF foundation.

Original IQ/time/frequency/energy paths and all task heads remain accessible.
All clock periods and positions are in samples, not physical units. The lag-1
centre is a proxy, never labelled CFO. This is RTN-inspired, not its reproduction.
"""
import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


RECEPTION_DEFAULTS=dict(width=96,heads=3,sps=[2,4,8,16,32,64],span=4,rolloff=.35,
    learned=False,max_frequency_delta=.005,max_log_rate_delta=.35,
    max_equalizer_delta=.1,attention_dropout=0.,initial_gain=.1,max_gain=.5,
    balance_scales=False,freeze_predictor=False,match_token_scale=False)


def rrc_filter(sps,span=4,beta=.35):
    """Unit-energy symmetric root-raised-cosine FIR, including both limits."""
    t=torch.arange(-span*sps//2,span*sps//2+1,dtype=torch.float64)/sps
    numerator=torch.sin(math.pi*(1-beta)*t)+4*beta*t*torch.cos(math.pi*(1+beta)*t)
    denominator=math.pi*t*(1-(4*beta*t).square())
    h=numerator/torch.where(denominator.abs()<1e-12,torch.ones_like(denominator),denominator)
    h[t.abs()<1e-12]=1+beta*(4/math.pi-1)
    singular=(t.abs()-1/(4*beta)).abs()<1e-10
    h[singular]=beta/math.sqrt(2)*((1+2/math.pi)*math.sin(math.pi/(4*beta))+(1-2/math.pi)*math.cos(math.pi/(4*beta)))
    return (h/h.square().sum().sqrt()).float()


def complex_fir(x,taps):
    """Per-record 5-tap residual equalizer, delta=0 gives exact identity."""
    # taps: [B,5,2]. This is a small complex convolution, no cyclic wraparound.
    padded=F.pad(x.transpose(1,2),(2,2)).transpose(1,2)
    windows=padded.unfold(1,5,1).permute(0,1,3,2)
    real=(windows[...,0]*taps[:,None,:,0]-windows[...,1]*taps[:,None,:,1]).sum(-1)
    imag=(windows[...,0]*taps[:,None,:,1]+windows[...,1]*taps[:,None,:,0]).sum(-1)
    return x+torch.stack([real,imag],-1)


def fractional_samples(x,positions):
    """Linear interpolator; caller masks out-of-record/edge positions."""
    left=positions.floor().long();fraction=positions-left
    a=x.gather(1,left.clamp(0,x.shape[1]-1)[...,None].expand(-1,-1,2))
    b=x.gather(1,(left+1).clamp(0,x.shape[1]-1)[...,None].expand(-1,-1,2))
    return a+(b-a)*fraction[...,None]


def balanced_attention_mask(allowed,sizes):
    """Remove the token-count prior while retaining learned content attention.

    For equal logits each nonempty scale has equal total mass, even if one
    clock has many more samples. Empty scales remain excluded.
    """
    pieces=[]
    for part in allowed.split(sizes,dim=-1):
        prior=-part.sum(-1,keepdim=True).clamp_min(1).float().log()
        pieces.append(prior.expand_as(part).masked_fill(~part,-torch.inf))
    return torch.cat(pieces,-1)


class ReceptionMemory(nn.Module):
    def __init__(self,backbone_width,patch_size,options=None):
        super().__init__();self.options={**deepcopy(RECEPTION_DEFAULTS),**(options or {})}
        if options and set(options)-set(RECEPTION_DEFAULTS):raise ValueError('Unknown reception option')
        o=self.options;d=o['width'];self.patch_size=patch_size;self.scales=len(o['sps'])
        if d%o['heads'] or o['span']%2 or min(o['sps'])<2:raise ValueError('Invalid reception dimensions')
        max_kernel=o['span']*max(o['sps'])+1
        filters=[]
        for sps in o['sps']:
            h=rrc_filter(sps,o['span'],o['rolloff']);pad=(max_kernel-len(h))//2
            filters.append(F.pad(h,(pad,pad)))
        self.register_buffer('filters',torch.stack(filters)[:,None])
        self.register_buffer('periods',torch.tensor(o['sps'],dtype=torch.float32))
        self.estimate1=nn.Conv1d(5,32,9,stride=4,padding=4)
        self.estimate2=nn.Conv1d(32,32,5,stride=4,padding=2)
        self.estimate_output=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Linear(64,2*self.scales+12))
        nn.init.zeros_(self.estimate_output[-1].weight);nn.init.zeros_(self.estimate_output[-1].bias)
        if not o['learned'] or o['freeze_predictor']:
            for name,p in self.named_parameters():
                if name.startswith(('estimate1.','estimate2.','estimate_output.')):p.requires_grad_(False)
        self.embedding=nn.Sequential(nn.Linear(8,d),nn.GELU(),nn.Linear(d,d),nn.LayerNorm(d))
        self.scale_embedding=nn.Parameter(torch.zeros(self.scales,d))
        self.query_norm=nn.LayerNorm(backbone_width)
        self.query=nn.Linear(backbone_width,d,bias=False)
        self.key_value=nn.Linear(d,2*d,bias=False)
        self.projection=nn.Linear(d,backbone_width,bias=False)
        nn.init.zeros_(self.projection.weight)
        fraction=o['initial_gain']/o['max_gain']
        self.gain_logit=nn.Parameter(torch.tensor(math.log(fraction/(1-fraction))))
        self.last_diagnostics={};self.regularization=None

    def estimate(self,x,valid,lengths):
        o=self.options
        with torch.autocast(device_type=x.device.type,enabled=False):
            x=x.float()*valid[...,None]
            mean=x.sum(1)/lengths[:,None];centered=(x-mean[:,None])*valid[...,None]
            a,b=centered[:,1:],centered[:,:-1]
            real=(a*b).sum(-1);imag=a[...,1]*b[...,0]-a[...,0]*b[...,1]
            pair=valid[:,1:]&valid[:,:-1]
            cr=(real*pair).sum(1);ci=(imag*pair).sum(1)
            denominator=(a.square().sum((1,2))*b.square().sum((1,2))).sqrt().clamp_min(1e-8)
            coherence=(cr.square()+ci.square()).sqrt()/denominator
            centre=torch.where(coherence>.25,torch.atan2(ci,cr+1e-12)/(2*math.pi),0.).detach()
            lag=F.pad(torch.stack([real,imag],1),(1,0))
            features=torch.cat([x.transpose(1,2),x.square().sum(-1)[:,None],lag],1)
        h=F.gelu(self.estimate1(features))
        mask1=(torch.arange(h.shape[-1],device=x.device)[None]*4<lengths[:,None])
        h=h*mask1[:,None]
        h=F.gelu(self.estimate2(h))
        mask2=(torch.arange(h.shape[-1],device=x.device)[None]*16<lengths[:,None])
        h=h.float()*mask2[:,None];count=mask2.sum(1).clamp_min(1)[:,None]
        mean=h.sum(-1)/count
        std=((h-mean[...,None]).square()*mask2[:,None]).sum(-1).div(count).clamp_min(1e-6).sqrt()
        parameters=self.estimate_output(torch.cat([mean,std],1)).float()
        if not o['learned']:parameters=parameters*0
        parameters=parameters.tanh()
        self.regularization=parameters.square().mean()
        return centre,coherence,parameters

    def observations(self,x,valid,lengths,override=None):
        """Return symbol views on explicit ORIGINAL sample coordinates."""
        o=self.options
        centre,coherence,parameters=self.estimate(x,valid,lengths)
        if override is not None:
            centre=override.get('centre',centre);parameters=override.get('parameters',parameters)
        with torch.autocast(device_type=x.device.type,enabled=False):
            parameters=parameters.float();x=x.float()*valid[...,None]
            frequency=centre+o['max_frequency_delta']*parameters[:,0]
            theta=math.pi*parameters[:,1]
            time=torch.arange(x.shape[1],device=x.device).float()[None]
            angle=time*frequency[:,None]*(2*math.pi)+theta[:,None]
            cosine,sine=angle.cos(),angle.sin()
            corrected=torch.stack([x[...,0]*cosine+x[...,1]*sine,x[...,1]*cosine-x[...,0]*sine],-1)
            taps=parameters[:,2:12].reshape(-1,5,2)*o['max_equalizer_delta']
            corrected=complex_fir(corrected,taps)*valid[...,None]
            weight=torch.cat([self.filters,self.filters],0).float()
            filtered=F.conv1d(corrected.transpose(1,2),weight,padding=weight.shape[-1]//2,groups=2)
            filtered=filtered.reshape(len(x),2,self.scales,x.shape[1]).permute(0,2,3,1)
            observations=[];positions=[];masks=[];periods=[]
            for index,sps in enumerate(o['sps']):
                rate=sps*torch.exp(o['max_log_rate_delta']*parameters[:,12+index])
                offset=(sps-1)/2+(sps/2)*parameters[:,12+self.scales+index]
                count=math.ceil(x.shape[1]/(sps*math.exp(-o['max_log_rate_delta'])))+1
                coordinate=offset[:,None]+rate[:,None]*torch.arange(count,device=x.device)[None]
                radius=o['span']*sps/2
                mask=(coordinate>=radius)&(coordinate<lengths[:,None]-radius-1)
                samples=fractional_samples(filtered[:,index],coordinate)*mask[...,None]
                observations.append(samples);positions.append(coordinate);masks.append(mask);periods.append(rate)
        self.last_diagnostics=dict(centre=centre.detach(),coherence=coherence.detach(),
            frequency_delta=(frequency-centre).detach(),clock_log_delta=parameters[:,12:12+self.scales].detach(),
            timing_delta=parameters[:,12+self.scales:].detach(),equalizer_delta=taps.detach())
        return observations,positions,masks,periods

    def forward(self,h,x,point_valid,lengths,query_valid):
        o=self.options;d=o['width'];heads=o['heads'];head_width=d//heads
        samples,positions,masks,periods=self.observations(x,point_valid,lengths)
        memories=[];radii=[]
        for index,(z,position,mask,period) in enumerate(zip(samples,positions,masks,periods)):
            with torch.autocast(device_type=x.device.type,enabled=False):
                count=mask.sum(1).clamp_min(1)
                scale=(z.square().sum((1,2))/count).clamp_min(1e-8).sqrt()
                normalized=z/scale[:,None,None]
                power=normalized.square().sum(-1)
                previous=F.pad(normalized[:,:-1],(0,0,1,0))
                real=(normalized*previous).sum(-1)
                imag=normalized[...,1]*previous[...,0]-normalized[...,0]*previous[...,1]
                product_power=power*previous.square().sum(-1)
                denominator=product_power.clamp_min(1e-8).sqrt()
                reliability=product_power/(product_power+.01)
                features=torch.stack([normalized[...,0],normalized[...,1],torch.log1p(power),
                    real/denominator*reliability,imag/denominator*reliability,
                    scale.log()[:,None].expand_as(power),period.log2()[:,None].expand_as(power)/6,
                    position/1024],-1)*mask[...,None]
            memories.append((self.embedding(features)+self.scale_embedding[index])*mask[...,None])
            radii.append(torch.maximum(period*2,torch.full_like(period,16))[:,None].expand_as(position))
        memory=torch.cat(memories,1);coordinate=torch.cat(positions,1);memory_valid=torch.cat(masks,1)
        radius=torch.cat(radii,1)
        query_coordinate=torch.arange(h.shape[1],device=h.device)*self.patch_size+(self.patch_size-1)/2
        allowed=((query_coordinate[None,:,None]-coordinate[:,None]).abs()<=radius[:,None])
        allowed=allowed&memory_valid[:,None]&query_valid[:,:,None]
        q=self.query(self.query_norm(h)).reshape(len(h),h.shape[1],heads,head_width).transpose(1,2)
        k,v=self.key_value(memory).reshape(len(h),memory.shape[1],2,heads,head_width).permute(2,0,3,1,4).unbind(0)
        attention_mask=(balanced_attention_mask(allowed,[len(p[0]) for p in positions]).to(q.dtype)
            if o['balance_scales'] else allowed)
        update=F.scaled_dot_product_attention(q,k,v,attn_mask=attention_mask[:,None],
            dropout_p=o['attention_dropout'] if self.training else 0.)
        update=self.projection(update.transpose(1,2).reshape(len(h),h.shape[1],d))
        delta=o['max_gain']*self.gain_logit.sigmoid()*update.tanh()*query_valid[...,None]
        if o['match_token_scale']:
            # Match units of an already trained residual stream. Detachment
            # prevents optimizing the original stream merely to amplify the
            # new view. Masked token values cannot change the scale.
            with torch.autocast(device_type=h.device.type,enabled=False):
                power=(h.float().square()*query_valid[...,None]).sum((1,2))
                count=query_valid.sum(1).clamp_min(1)*h.shape[-1]
                scale=(power/count).sqrt().detach().clamp_min(1e-6)
            delta=delta*scale[:,None,None].to(delta.dtype)
        return delta


def attach_reception(model,options):
    """Research hook; model retains original parameter names/three task heads.

    Checkpoints must be loaded with reception_model(), not bare RFModel. The
    explicit constructor installs the same hook; no hook is serialized/pickled.
    """
    if hasattr(model.backbone,'reception'):raise ValueError('Reception already attached')
    if model.multiresolution:raise ValueError('Initial study uses fixed historical mean-pooling base')
    with torch.random.fork_rng(devices=[]):
        model.backbone.reception=ReceptionMemory(model.config['width'],model.config['patch_size'],options)
    context={}
    def prepare(module,args,kwargs):
        iq=args[0] if args else kwargs['iq']
        lengths=args[1] if len(args)>1 else kwargs['lengths']
        if kwargs.get('patch_mask') is not None or (len(args)>2 and args[2] is not None):
            raise ValueError('Research reception supports supervised inference/training only')
        x,_,_,valid=module.backbone.preprocess(iq,lengths)
        context.update(x=x,valid=valid,lengths=lengths)
    def inject(module,args):
        h,valid,positions=args
        delta=model.backbone.reception(h,context['x'],context['valid'],context['lengths'],valid)
        return (h+delta.to(h.dtype))*valid[...,None],valid,positions
    def clear(module,args,output):context.clear()
    model.register_forward_pre_hook(prepare,with_kwargs=True)
    model.backbone.blocks[0].register_forward_pre_hook(inject)
    model.register_forward_hook(clear,always_call=True)
    return model


def reception_model(checkpoint):
    from models.rf_model import RFModel
    from utils.common import model_bundle
    model=RFModel(**model_bundle(checkpoint))
    options=checkpoint.get('reception_options')
    if options is not None:
        if options.get('kind')=='carrier_input':
            from models.carrier_input import attach_carrier_input
            attach_carrier_input(model,options)
        else:attach_reception(model,options)
    model.load_state_dict(checkpoint['model'],strict=True)
    return model
