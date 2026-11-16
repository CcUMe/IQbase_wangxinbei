"""Known-truth carrier checks; synthetic measurements are not C01 CFO accuracy."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from models.carrier_recovery import fourth_power_refinement
from models.reception import ReceptionMemory
from training.precalibrate_reception import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise ValueError('Use CUDA_VISIBLE_DEVICES empty; this check uses CPU only')
    torch.set_num_threads(4)
    torch.manual_seed(2026111407)
    result = dict(seed=2026111407, role='Controlled checks only, not C01 truth')
    for length in [512, 1024]:
        t = torch.arange(length)
        true = torch.tensor([.037125, -.02131])
        z = torch.exp(2j*torch.pi*true[:,None]*t)
        iq = torch.stack([z.real,z.imag],-1)
        lengths = torch.full((2,),length)
        estimate, _ = fourth_power_refinement(iq,lengths,true+.002)
        error = float((estimate-true).abs().max())
        assert error < 2e-6, error
        if length == 512:
            padded, _ = fourth_power_refinement(F.pad(iq,(0,0,0,512)),lengths,true+.002)
            torch.testing.assert_close(estimate,padded,rtol=0,atol=0)
        result[f'tone_{length}_maximum_error'] = error
    predictor = ReceptionMemory(768,16)
    for heldout, name in [(False,'qpsk_qam16_64'),(True,'qam256')]:
        generator = torch.Generator().manual_seed(2026111407+heldout)
        original, corrected, accepted = [], [], []
        with torch.inference_mode():
            for _ in range(16):
                x,period,true,snr = generate(32,1024,generator,heldout)
                lengths = torch.full((32,),1024)
                prior, _, _ = predictor.estimate(x,torch.ones(32,1024,dtype=torch.bool),lengths)
                estimate, details = fourth_power_refinement(x,lengths,prior)
                original.append((prior-true).abs());corrected.append((estimate-true).abs())
                accepted.append(details['accepted'])
        a,b,keep = torch.cat(original),torch.cat(corrected),torch.cat(accepted)
        result[name] = dict(records=len(a),lag1_rmse=float(a.square().mean().sqrt()),
            refined_rmse=float(b.square().mean().sqrt()),
            lag1_median_abs_error=float(a.median()),refined_median_abs_error=float(b.median()),
            acceptance=float(keep.float().mean()),error_reduced_records=int((b<a-1e-8).sum()),
            error_increased_records=int((b>a+1e-8).sum()))
    # A linearly filtered proper complex Gaussian process has no discrete
    # symbol clock. This finite control measures accidental line acceptance.
    generator = torch.Generator().manual_seed(2026111409)
    acceptance = []
    with torch.inference_mode():
        for _ in range(16):
            noise = torch.complex(torch.randn(32,1024,generator=generator),
                                  torch.randn(32,1024,generator=generator))
            width = .01+.11*torch.rand(32,generator=generator)
            frequency = torch.fft.fftfreq(1024)
            response = torch.exp(-.5*(frequency[None]/width[:,None]).square())
            z = torch.fft.ifft(torch.fft.fft(noise)*response)
            carrier = .15*(torch.rand(32,generator=generator)-.5)
            z = z*torch.exp(2j*torch.pi*carrier[:,None]*torch.arange(1024))
            z = z/z.abs().square().mean(1,keepdim=True).sqrt()
            x = torch.stack([z.real,z.imag],-1)
            lengths = torch.full((32,),1024)
            prior,_,_ = predictor.estimate(x,torch.ones(32,1024,dtype=torch.bool),lengths)
            _,details = fourth_power_refinement(x,lengths,prior)
            acceptance.append(details['accepted'])
    keep = torch.cat(acceptance)
    result['coloured_gaussian_null'] = dict(records=len(keep),acceptance=float(keep.float().mean()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
