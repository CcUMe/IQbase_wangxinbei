"""A single-GPU, train-internal-selected cyclic-correlation representation probe.

This is a diagnostic classifier, NOT a replacement contest foundation model and
NOT a reproduction of Cyclo-AMC's FAM/SCD estimator. No validation row is used
for normalization, fitting, early stopping or arm selection.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
from utils.common import read_json, write_json, seed_all
from training.probe_c01_relations import group_metrics

LAGS = [0, 1, 2, 4, 8, 16, 32, 64]
SEED = 20261114


def cyclic_view(iq):
    """Hann-weighted conjugate/nonconjugate CAF magnitudes, cycles/sample.

    Output: [batch, 2*8 lags, 256 cyclic-frequency bins]. FFT length 1024
    defines a common *sample-time* frequency grid for native 512/1024 records;
    padding is applied to lag products AFTER the native-length Hann window.
    Adjacent sets of four FFT bins are max-pooled to retain narrow peaks.
    We do not claim this is physical-time alignment without a known fs.
    """
    with torch.autocast(device_type=iq.device.type, enabled=False):
        z = torch.complex(iq[..., 0].float(), iq[..., 1].float())
        z = z / z.abs().square().mean(1, keepdim=True).clamp_min(1e-12).sqrt()
        channels = []
        for lag in LAGS:
            a, b = z[:, lag:], z[:, :z.shape[1]-lag]
            window = torch.hann_window(a.shape[1], periodic=False, device=iq.device)
            for product in [a*b.conj(), a*b]:
                value = torch.fft.fftshift(torch.fft.fft(product*window, n=1024, dim=-1), dim=-1)
                magnitude = value.abs()/window.sum()
                channels.append(torch.log1p(32*magnitude.reshape(len(iq), 256, 4).amax(-1)))
        return torch.stack(channels, 1)


def spectral_view(iq,window_size=64,centered_crop=False):
    """Magnitude of q=0 demodulated STFT cross-frequency covariance.

    FAM-inspired reduced frequency-pair slice: Np64, hop16, native N512/1024,
    P=N/16, right-zero-padded last windows. The paper uses N512 and a different
    network/training protocol. This remains an adaptation, not a reproduction.
    """
    with torch.autocast(device_type=iq.device.type,enabled=False):
        z=torch.complex(iq[...,0].float(),iq[...,1].float())
        z=z/z.abs().square().mean(1,keepdim=True).clamp_min(1e-12).sqrt()
        if centered_crop:
            # Summation/long-angle rotation use FP64 for numerical conditioning;
            # this does not add signal information or make the proxy true CFO.
            precise=z.to(torch.complex128)
            omega=torch.angle((precise[:,1:]*precise[:,:-1].conj()).sum(1))
            angle=torch.arange(z.shape[1],device=z.device,dtype=torch.float64)[None]*omega[:,None]
            z=(precise*torch.polar(torch.ones_like(angle),-angle)).to(torch.complex64)
        window=torch.hamming_window(window_size,periodic=False,device=z.device)
        frames=F.pad(z,(0,window_size-16)).unfold(-1,window_size,16)
        spectrum=torch.fft.fft(frames*window,dim=-1)
        p=torch.arange(spectrum.shape[1],device=z.device)
        k=torch.arange(window_size,device=z.device)
        # exp(-j*2*pi*k*p*16/64) via exact quarter turns, avoiding long-angle error.
        period=window_size//16
        table=(torch.tensor([1+0j,-1j,-1+0j,1j],device=z.device) if period==4 else
            torch.exp(-2j*torch.pi*torch.arange(period,device=z.device)/period))
        spectrum=spectrum*table[(p[:,None]*k[None])%period]
        spectrum=torch.fft.fftshift(spectrum,dim=-1)
        if centered_crop:spectrum=spectrum[...,window_size//2-32:window_size//2+32]
        covariance=spectrum.transpose(1,2)@spectrum.conj()/spectrum.shape[1]
        return torch.log1p(covariance.abs()/window.square().sum())[:,None]


class Residual2d(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(width,width,3,padding=1,bias=False),nn.BatchNorm2d(width),nn.GELU(),
            nn.Conv2d(width,width,3,padding=1,bias=False),nn.BatchNorm2d(width))
    def forward(self,x):return F.gelu(x+self.layers(x))


class SpectralProbe(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[nn.Conv2d(1,24,3,padding=1,bias=False),nn.BatchNorm2d(24),nn.GELU(),nn.MaxPool2d(3,stride=2,padding=1)]
        previous=24
        for width in [24,48,72,96]:
            if width!=previous:layers += [nn.Conv2d(previous,width,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(width),nn.GELU()]
            layers += [Residual2d(width),Residual2d(width)];previous=width
        self.encoder=nn.Sequential(*layers)
        self.readout=nn.Sequential(nn.Linear(96,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,23))
    def forward(self,x):return self.readout(self.encoder(x).mean((2,3)))


def ablate(batch,arm):
    if arm=='lag0_only':batch[:,2:]=0
    if arm=='diagonal_only':batch=batch*torch.eye(64,device=batch.device)[None,None]
    return batch


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv1d(width,width,5,padding=2,bias=False),
            nn.GroupNorm(8,width),nn.GELU(),nn.Dropout(.1),
            nn.Conv1d(width,width,5,padding=2,bias=False),nn.GroupNorm(8,width))
    def forward(self,x):return F.gelu(x+self.layers(x))


class CyclicProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(16,64,7,stride=2,padding=3,bias=False),
            nn.GroupNorm(8,64),nn.GELU(),Residual(64),
            nn.Conv1d(64,128,5,stride=2,padding=2,bias=False),
            nn.GroupNorm(8,128),nn.GELU(),Residual(128))
        self.readout = nn.Sequential(nn.Flatten(),nn.Linear(128*16,128),nn.GELU(),
            nn.Dropout(.2),nn.Linear(128,23))
    def forward(self,x):return self.readout(F.adaptive_avg_pool1d(self.encoder(x),16))


def self_test(device):
    generator = torch.Generator(device=device).manual_seed(SEED)
    x = torch.randn(4,512,2,device=device,generator=generator)
    y = cyclic_view(x)
    angle=.713;c,s=np.cos(angle),np.sin(angle)
    rotated=torch.stack((x[...,0]*c-x[...,1]*s,x[...,0]*s+x[...,1]*c),-1)
    torch.testing.assert_close(y,cyclic_view(rotated),rtol=3e-5,atol=3e-6)
    torch.testing.assert_close(y,cyclic_view(x*4),rtol=3e-5,atol=3e-6)
    assert y.shape==(4,16,256) and torch.isfinite(y).all()
    # Independent direct-DFT check at an unpooled bin, without window/padding assumptions.
    z=torch.complex(x[0,:,0],x[0,:,1]);z=z/z.abs().square().mean().sqrt()
    lag=8;v=z[lag:]*z[:-lag].conj();w=torch.hann_window(len(v),periodic=False,device=device)
    k=37;n=torch.arange(len(v),device=device)
    direct=(v*w*torch.exp(-2j*torch.pi*k*n/1024)).sum()
    torch.testing.assert_close(direct,torch.fft.fft(v*w,n=1024)[k],rtol=2e-5,atol=5e-5)
    print('SELF TEST PASSED: shape, finite, gain/phase invariance, direct DFT.',flush=True)
    covariance=spectral_view(x)
    torch.testing.assert_close(covariance,covariance.transpose(-1,-2),atol=3e-6,rtol=3e-5)
    torch.testing.assert_close(covariance,spectral_view(rotated),atol=3e-6,rtol=3e-5)
    torch.testing.assert_close(covariance,spectral_view(x*4),atol=3e-6,rtol=3e-5)
    assert covariance.shape==(4,1,64,64) and torch.isfinite(covariance).all()
    # Direct check of one off-diagonal pair, including exact down-conversion.
    z=torch.complex(x[0,:,0],x[0,:,1]);z=z/z.abs().square().mean().sqrt()
    w=torch.hamming_window(64,periodic=False,device=device);frames=F.pad(z,(0,48)).unfold(0,64,16)
    f=torch.fft.fft(frames*w,dim=-1)
    t=torch.arange(len(frames),device=device)
    a,b=5,9
    fa=f[:,a]*torch.exp(-2j*torch.pi*((t*a)%4)/4)
    fb=f[:,b]*torch.exp(-2j*torch.pi*((t*b)%4)/4)
    expected=torch.log1p((fa*fb.conj()).mean().abs()/w.square().sum())
    torch.testing.assert_close(covariance[0,0,(a+32)%64,(b+32)%64],expected,atol=3e-6,rtol=3e-5)
    print('SPECTRAL SELF TEST PASSED: Hermitian magnitude, gain/phase invariance, direct off-diagonal pair.',flush=True)


@torch.inference_mode()
def logits(model,x,indices,arm,batch_size=512):
    model.eval();output=[]
    for start in range(0,len(indices),batch_size):
        batch=x[indices[start:start+batch_size]].float()
        batch=ablate(batch,arm)
        output.append(model(batch).float().cpu().numpy())
    return np.concatenate(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=ROOT/'artifacts/data_full')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--view',choices=['caf','spectral_covariance'],default='caf')
    parser.add_argument('--spectral-window',type=int,choices=[64,256],default=64)
    parser.add_argument('--centered-crop',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))!=1 or torch.cuda.device_count()!=1:
        raise ValueError('Exactly one visible GPU required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0')
    if args.self_test:self_test(device);return
    args.output.mkdir(parents=True,exist_ok=True)
    seed_all(SEED)
    with np.load(args.data/'metadata.npz',allow_pickle=False) as f:meta={k:f[k] for k in f.files}
    rows=np.flatnonzero(meta['modulation']>=0)
    labels=meta['modulation'][rows].astype(np.int64)
    previous=np.load(ROOT/'artifacts/c01_followup_validation/20261113_224434/selection.npz',allow_pickle=False)
    groups=previous['condition_groups'][rows,0]
    rng=np.random.default_rng(SEED)
    train,internal=[],[]
    for label in range(23):
        for q in range(4):
            pool=np.flatnonzero((labels==label)&(groups==q)&(meta['split'][rows]==0))
            pool=rng.permutation(pool);cut=max(1,int(.1*len(pool)))
            internal.extend(pool[:cut]);train.extend(pool[cut:])
    train=np.sort(train);internal=np.sort(internal)
    validation=np.flatnonzero(meta['split'][rows]==1)
    np.savez_compressed(args.output/'split.npz',rows=rows,train=train,internal=internal,validation=validation,rms_group=groups)
    protocol=dict(seed=SEED,view=args.view,feature='native Hann conjugate and nonconjugate CAF magnitudes',
        lags=LAGS,fft_length=1024,cyclic_bins=256,log_scale=32,global_phase_positive_gain_invariant=True,
        fs_known=False,source='https://phwl.org/assets/papers/cycloamc_fpt25.pdf',
        reproduction=False,arms=['lag0_only','all_lags'],epochs=args.epochs,batch_size=512,
        train_rows=len(train),internal_rows=len(internal),validation_rows=len(validation),
        optimizer={'name':'AdamW','learning_rate':3e-4,'weight_decay':1e-3},
        selection='best internal first14 accuracy per arm; compare arms on internal only, then evaluate validation once',
        fingerprint=read_json(args.data/'dataset.json')['fingerprint'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caveats=['Diagnostic model, not a competition submission model',
            'The historical validation set has already been inspected in earlier experiments',
            'Magnitudes discard phase; the original IQ must remain accessible in any contest integration',
            'Cyclic frequency is NOT time and is NOT aligned with temporal token positions',
            'RMS quartiles are offline true-class slices; no label-conditioned inference'])
    write_json(args.output/'protocol.json',protocol)
    if args.view=='spectral_covariance':
        for key in ['lags','cyclic_bins','log_scale']:protocol.pop(key,None)
        protocol.update(feature='q=0 demodulated STFT cross-frequency covariance magnitude',
            fft_length=args.spectral_window,hop=16,native_samples=[512,1024],native_frame_counts=[32,64],
            window=f'symmetric Hamming, right-zero-padding{args.spectral_window-16}',arms=['diagonal_only','full_covariance'],
            centered_crop=args.centered_crop,frequency_bins_retained=64,
            centre='lag-1 correlation phase proxy' if args.centered_crop else 'original frequency origin',
            frequency_bin_spacing_cycles_per_sample=1/args.spectral_window,
            feature_transform='log1p(abs(mean_p(demodulated_fft[k]*conj(demodulated_fft[l])))/window_energy)')
        write_json(args.output/'protocol.json',protocol)
        if args.spectral_window!=64 and not args.centered_crop:raise ValueError('256-point spectrum requires an explicit centred 64-bin crop')
    shape=(16,256) if args.view=='caf' else (1,64,64)
    view_function=cyclic_view if args.view=='caf' else lambda iq:spectral_view(iq,args.spectral_window,args.centered_crop)
    feature_path=args.output/'cyclic.npy'
    if not (args.output/'features_complete.json').exists():
        view=np.lib.format.open_memmap(feature_path,mode='w+',dtype='float16',shape=(len(rows),*shape))
        raw=np.load(args.data/'iq.npy',mmap_mode='r')
        for length in [512,1024]:
            selected=np.flatnonzero(meta['length'][rows]==length)
            for start in tqdm(range(0,len(selected),512),desc=f'CAF/native{length}',mininterval=5):
                idx=selected[start:start+512]
                iq=torch.from_numpy(np.array(raw[rows[idx],:length])).to(device)
                with torch.inference_mode():view[idx]=view_function(iq).cpu().numpy().astype(np.float16)
        view.flush();del view
        write_json(args.output/'features_complete.json',{'rows':len(rows),'shape':[len(rows),*shape],'view':args.view,
            'spectral_window':args.spectral_window,'centered_crop':args.centered_crop})
    if read_json(args.output/'features_complete.json').get('view','caf')!=args.view:raise ValueError('Cached feature view mismatch')
    if args.view=='spectral_covariance':
        cached=read_json(args.output/'features_complete.json')
        if cached.get('spectral_window',64)!=args.spectral_window or cached.get('centered_crop',False)!=args.centered_crop:
            raise ValueError('Cached spectral preprocessing mismatch; use a new directory')
    x=torch.from_numpy(np.load(feature_path)).to(device)
    y=torch.from_numpy(labels).to(device)
    trainidx=torch.tensor(train,device=device)
    results={}
    for arm in protocol['arms']:
        if (args.output/f'{arm}_result.json').exists():
            results[arm]=read_json(args.output/f'{arm}_result.json');continue
        seed_all(SEED);model=(CyclicProbe() if args.view=='caf' else SpectralProbe()).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
        steps=(len(train)+511)//512
        schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step: min(1.,(step+1)/steps)*
            (.05+.95*(1+np.cos(np.pi*min(step,args.epochs*steps)/(args.epochs*steps)))/2))
        best=-1;history=[];started=time.monotonic()
        for epoch in range(args.epochs):
            model.train();order=trainidx[torch.randperm(len(trainidx),device=device)]
            losses=[];correct=0
            bar=tqdm(range(0,len(order),512),desc=f'{arm}/epoch{epoch+1:02d}',mininterval=5)
            for start in bar:
                ids=order[start:start+512];batch=x[ids].float()
                batch=ablate(batch,arm)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    z=model(batch);loss=F.cross_entropy(z,y[ids])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step();schedule.step()
                losses.append(float(loss));correct+=int((z.argmax(1)==y[ids]).sum())
                if len(losses)%50==0:bar.set_postfix(ce=f'{np.mean(losses):.3f}')
            z=logits(model,x,internal,arm)
            probability=torch.from_numpy(z).softmax(1).numpy()
            metrics=group_metrics(probability,labels[internal],groups[internal])
            item={'epoch':epoch+1,'train_ce':float(np.mean(losses)),
                'train_accuracy':correct/len(train),'internal':metrics}
            history.append(item);write_json(args.output/f'{arm}_history.json',history)
            score=metrics['first14']['accuracy']
            if score>best:
                best=score;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch+1
                torch.save({'model':state,'protocol':protocol,'epoch':best_epoch},args.output/f'{arm}_best.pt')
            print('EPOCH',arm,epoch+1,'train',round(item['train_accuracy'],4),
                'internal_all/first14',round(metrics['all23']['accuracy'],4),round(score,4),flush=True)
        model.load_state_dict(state)
        z=logits(model,x,validation,arm)
        np.savez_compressed(args.output/f'{arm}_validation.npz',rows=rows[validation],modulation=z)
        metrics=group_metrics(torch.from_numpy(z).softmax(1).numpy(),labels[validation],groups[validation])
        results[arm]={'best_internal_first14':best,'best_epoch':best_epoch,'validation':metrics,
            'parameters':sum(p.numel() for p in model.parameters()),'elapsed_seconds':time.monotonic()-started}
        write_json(args.output/f'{arm}_result.json',results[arm])
        print('ARM COMPLETE',arm,{g:round(m['accuracy'],5) for g,m in metrics.items()},flush=True)
        del model,optimizer
    winner=max(results,key=lambda k:results[k]['best_internal_first14'])
    write_json(args.output/'results.json',{'arms':results,'internal_selected_arm':winner})
    print('FINISHED',args.output,'internal selected',winner,flush=True)


if __name__=='__main__':main()
