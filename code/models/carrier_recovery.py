"""Conservative fourth-power line refinement of a lag-based carrier proxy.

This is a blind engineering prior, not an oracle CFO estimator. No class or
sampling-rate metadata is used. Agreement across independent record halves is
required; otherwise the original proxy is retained. Short/colored observations
can still violate the assumptions and must be evaluated on actual C01 tasks.
"""
import math
import torch


@torch.no_grad()
def fourth_power_refinement(iq,lengths,prior,search_radius=.01,fft_size=4096,min_contrast=6.):
    with torch.autocast(device_type=iq.device.type,enabled=False):
        x=iq.float();time=torch.arange(x.shape[1],device=x.device)[None]
        valid=time<lengths[:,None]
        z=torch.complex(x[...,0],x[...,1])*valid
        magnitude=z.abs().clamp_min(1e-6)
        z=z*(3/magnitude.clamp_min(3))
        z=z*torch.exp(-2j*math.pi*prior[:,None]*time)
        fourth=z.square().square()
        frequency=torch.fft.fftfreq(fft_size,device=x.device)/4
        search=frequency.abs()<=search_radius
        def estimate(start,size,maximum):
            coordinate=torch.arange(maximum,device=x.device)[None]
            indices=(coordinate+start[:,None]).clamp(0,x.shape[1]-1)
            values=fourth.gather(1,indices)
            window=(.5-.5*torch.cos(2*math.pi*coordinate/(size[:,None]-1).clamp_min(1)))*(coordinate<size[:,None])
            power=torch.fft.fft(values*window,n=fft_size,dim=-1).abs().square()
            masked=power.masked_fill(~search[None],-1.)
            peak=masked.argmax(1);row=torch.arange(len(x),device=x.device)
            left=power[row,(peak-1)%fft_size].clamp_min(1e-20).log()
            middle=power[row,peak].clamp_min(1e-20).log()
            right=power[row,(peak+1)%fft_size].clamp_min(1e-20).log()
            curvature=left-2*middle+right
            offset=torch.where(curvature.abs()>1e-8,.5*(left-right)/curvature,0.).clamp(-.5,.5)
            result=frequency[peak]+offset/(4*fft_size)
            contrast=power[row,peak]/power[:,search].mean(1).clamp_min(1e-20)
            return result,contrast
        full,contrast=estimate(torch.zeros_like(lengths),lengths,x.shape[1])
        half=lengths//2
        first,c1=estimate(torch.zeros_like(lengths),half,x.shape[1]//2)
        second,c2=estimate(half,half,x.shape[1]//2)
        tolerance=1/(4*half.float())
        consistent=((first-second).abs()<tolerance)&((full-first).abs()<tolerance)&((full-second).abs()<tolerance)
        accepted=consistent&(contrast>=min_contrast)&(c1>=min_contrast)&(c2>=min_contrast)
        delta=torch.where(accepted,full,0.)
        return prior+delta,dict(accepted=accepted,delta=delta,contrast=contrast,half_difference=(first-second).abs())
