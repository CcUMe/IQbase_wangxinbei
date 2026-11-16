"""CPU-only physical-clock calibration on explicitly synthetic controlled data.

Only the tiny receiver parameter predictor is trained, not the foundation or a
modulation classifier. Continuous sample periods/CFO are generator truth, never
invented labels for C01. Transfer to C01 must be measured independently.
"""
import argparse
from pathlib import Path
import sys
import json
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from models.reception import ReceptionMemory
from utils.common import write_json


def generate(batch,length,generator,heldout=False):
    def rand(*shape):return torch.rand(*shape,generator=generator)
    period=torch.exp(math.log(3)+rand(batch)*(math.log(64)-math.log(3)))
    beta=.15+.35*rand(batch);tau=rand(batch)*period
    carrier=(rand(batch)-.5)*.15;phase=(rand(batch)-.5)*2*math.pi
    side=torch.full((batch,),16) if heldout else torch.tensor([2,4,8])[torch.randint(0,3,(batch,),generator=generator)]
    constellation_real=(torch.floor(rand(batch,400)*side[:,None])*2-(side[:,None]-1))
    constellation_imag=(torch.floor(rand(batch,400)*side[:,None])*2-(side[:,None]-1))
    symbols=torch.complex(constellation_real,constellation_imag)/torch.sqrt(2*(side.square()-1)/3)[:,None]
    u=(torch.arange(length)[None]-tau[:,None])/period[:,None]
    integer=u.floor().long();neighbors=integer[...,None]+torch.arange(-2,4)
    distance=u[...,None]-neighbors
    b=beta[:,None,None]
    denominator=math.pi*distance*(1-(4*b*distance).square())
    numerator=torch.sin(math.pi*(1-b)*distance)+4*b*distance*torch.cos(math.pi*(1+b)*distance)
    pulse=numerator/torch.where(denominator.abs()<1e-6,torch.ones_like(denominator),denominator)
    pulse=torch.where(distance.abs()<1e-5,1+b*(4/math.pi-1),pulse)
    limit=b/math.sqrt(2)*((1+2/math.pi)*torch.sin(math.pi/(4*b))+(1-2/math.pi)*torch.cos(math.pi/(4*b)))
    pulse=torch.where((distance.abs()-1/(4*b)).abs()<1e-5,limit,pulse)
    pulse=pulse*(distance.abs()<=2)/period[:,None,None].sqrt()
    values=symbols.gather(1,(neighbors+5).clamp(0,399).reshape(batch,-1)).reshape_as(pulse)
    z=(values*pulse).sum(-1)
    # Mild unknown multipath, shared by all samples of each synthetic record.
    echo=torch.complex((rand(batch)-.5)*.2,(rand(batch)-.5)*.2)
    z=z+echo[:,None]*F.pad(z[:,:-3],(3,0))
    power=z.abs().square().mean(1)
    noise=torch.complex(torch.randn(batch,length,generator=generator),torch.randn(batch,length,generator=generator))/2**.5
    # Some band-limited noise; its distribution is a hypothesis, not C01 truth.
    taps=torch.pow(torch.tensor(.75),torch.arange(17).float());taps=taps/taps.square().sum().sqrt()
    coloured=F.conv1d(torch.stack([noise.real,noise.imag],1),taps[None,None].expand(2,1,-1),padding=8,groups=2)
    coloured=torch.complex(coloured[:,0],coloured[:,1]);choose=rand(batch)<.3
    noise=torch.where(choose[:,None],coloured,noise)
    snr=8+rand(batch)*22
    z=z+noise*(power/torch.pow(10.,snr/10)).sqrt()[:,None]
    z=z*torch.exp(1j*(2*math.pi*carrier[:,None]*torch.arange(length)[None]+phase[:,None]))
    iq=torch.stack([z.real,z.imag],-1);iq=iq/iq.square().sum((1,2)).div(length).sqrt()[:,None,None]
    return iq,period,carrier,snr


def assess(model,seed,heldout=False,count=256):
    generator=torch.Generator().manual_seed(seed);true=[];estimated=[];frequency=[];freqtrue=[];neardelta=[]
    model.eval()
    with torch.inference_mode():
        for _ in range(count//32):
            x,period,carrier,snr=generate(32,1024,generator,heldout)
            centre,coherence,p=model.estimate(x,torch.ones(32,1024,dtype=torch.bool),torch.full((32,),1024))
            deltas=p[:,12:18]*.35
            all_period=model.periods[None]*deltas.exp()
            choice=deltas.abs().argmin(1)
            estimate=all_period[torch.arange(32),choice]
            nearest=(period.log()[:,None]-model.periods.log()[None]).abs().argmin(1)
            oracle=all_period[torch.arange(32),nearest]
            true.extend(period.tolist());estimated.extend(estimate.tolist());frequency.extend((centre+.005*p[:,0]).tolist());freqtrue.extend(carrier.tolist())
            neardelta.extend(((oracle-period).abs()/period).tolist())
    relative=np.abs(np.asarray(estimated)/np.asarray(true)-1)
    return dict(records=count,median_relative_period_error=float(np.median(relative)),mean_relative_period_error=float(relative.mean()),
        nearest_template_median_relative_error=float(np.median(neardelta)),
        cfo_proxy_rmse_cycles_per_sample=float(np.sqrt(np.mean((np.asarray(frequency)-freqtrue)**2))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--steps',type=int,default=1200)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(20261114)
    model=ReceptionMemory(768,16,dict(learned=True))
    params=[p for name,p in model.named_parameters() if name.startswith(('estimate1.','estimate2.','estimate_output.'))]
    optimizer=torch.optim.AdamW(params,lr=8e-4,weight_decay=1e-3)
    generator=torch.Generator().manual_seed(2026111401)
    protocol=dict(cpu_only=True,steps=args.steps,batch_size=32,training_seed=2026111401,
        signal='Continuous-clock QPSK/16QAM/64QAM, RRC rolloff uniform [.15,.5], 4-symbol truncation',
        sample_period_range=[3,64],frequency_cycles_per_sample=[-.075,.075],synthetic_snr_db=[8,30],
        impairments='Random phase, mild 3-sample complex echo, white/filtered independent noise',
        supervision='Synthetic true period and carrier; no classification or waveform reconstruction loss',
        heldout='New random records, with 256QAM not used in calibration',
        transfer_caveat='Does not establish C01 true sampling rate, pulse shape, noise or actual symbol-clock accuracy')
    write_json(args.output/'protocol.json',protocol)
    before={'in_distribution':assess(model,9091),'heldout_256qam':assess(model,9092,True)}
    write_json(args.output/'before.json',before);print('BEFORE',before,flush=True)
    history=[];started=time.monotonic();model.train()
    for step in range(args.steps):
        length=512 if step%2==0 else 1024
        x,period,carrier,snr=generate(32,length,generator)
        centre,coherence,p=model.estimate(x,torch.ones(32,length,dtype=torch.bool),torch.full((32,),length))
        delta=(period.log()[:,None]-model.periods.log()[None])/.35
        target=delta.clamp(-.99,.99)
        nearest=delta.abs().argmin(1)
        weight=torch.full_like(target,.05);weight.scatter_(1,nearest[:,None],1.)
        clock=((p[:,12:18]-target).square()*weight).sum()/weight.sum()
        desired=(carrier-centre)/.005;keep=(coherence>.6)&(desired.abs()<.95)
        frequency=F.mse_loss(p[keep,0],desired[keep]) if keep.any() else p[:,0].sum()*0
        loss=clock+.05*frequency
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite physical calibration loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,5.);optimizer.step()
        if (step+1)%100==0:
            item=dict(step=step+1,loss=float(loss.detach()),clock_loss=float(clock.detach()),frequency_loss=float(frequency.detach()))
            history.append(item);write_json(args.output/'history.json',history);print(item,flush=True)
    after={'in_distribution':assess(model,9091),'heldout_256qam':assess(model,9092,True)}
    write_json(args.output/'after.json',after)
    state={name:value.detach().cpu() for name,value in model.state_dict().items() if name.startswith(('estimate1.','estimate2.','estimate_output.'))}
    torch.save(dict(predictor=state,protocol=protocol,metrics=after),args.output/'predictor.pt')
    print('AFTER',after,'seconds',time.monotonic()-started,flush=True)


if __name__=='__main__':main()
