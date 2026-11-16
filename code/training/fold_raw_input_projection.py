"""Fold the RAW duplicate-input control into the original time convolution.

Only the identical-raw-input control can be folded: carrier transformations
are nonlinear/input-dependent and must not be folded by this utility.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from models.reception import reception_model
from models.rf_model import RFModel
from utils.common import model_bundle,save_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if torch.cuda.is_available():raise ValueError('Use CUDA_VISIBLE_DEVICES empty; CPU folding/check only')
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4)
    checkpoint=torch.load(args.source,map_location='cpu',weights_only=True,mmap=True)
    if checkpoint.get('reception_options')!={'kind':'carrier_input','mode':'raw'}:
        raise ValueError('Only the raw duplicate-input checkpoint is foldable')
    extra='backbone.reception.projection.weight';original='backbone.time.0.weight'
    keys={k for k in checkpoint['model'] if k.startswith('backbone.reception.')}
    if keys!={extra}:raise ValueError('Unexpected additional receiver state')
    state=dict(checkpoint['model']);state[original]=state[original]+state.pop(extra)
    folded={**checkpoint,'model':state,'reception_options':None,'research_kind':'folded_raw_input_projection',
        'resume_allowed':False,'folded_from':str(args.source.resolve()),
        'folding_note':'Same raw input, shared linear Conv1d; weights added, original bias retained. FP32 equivalence checked; floating-point operation ordering changes.'}
    model=RFModel(**model_bundle(folded));model.load_state_dict(state,strict=True);model.eval()
    reference=reception_model(checkpoint).eval()
    meta=np.load(ROOT/'artifacts/data_full/metadata.npz');iq=np.load(ROOT/'artifacts/data_full/iq.npy',mmap_mode='r')
    checks={}
    with torch.inference_mode():
        for length in [512,1024]:
            candidates=[]
            for eligible in [meta['modulation']>=0,meta['protocol']>=0,meta['target_valid'].any(1)]:
                candidates.extend(np.flatnonzero(eligible&(meta['split']==0)&(meta['length']==length))[:2])
            rows=np.unique(candidates);x=torch.from_numpy(np.array(iq[rows,:length]));lengths=torch.full((len(rows),),length)
            a=reference(x,lengths);b=model(x,lengths)
            for task in ['modulation','protocol','parameters']:
                torch.testing.assert_close(a[task],b[task],rtol=1e-4,atol=1e-4)
                checks[f'{length}/{task}']=float((a[task]-b[task]).abs().max())
    folded['folding_validation']=dict(device='CPU FP32',max_absolute_differences=checks,
        reference_parameters=reference.parameter_report(),folded_parameters=model.parameter_report())
    args.output.parent.mkdir(parents=True,exist_ok=True);save_checkpoint(args.output,folded)
    args.output.with_suffix('.folding.json').write_text(json.dumps(folded['folding_validation'],indent=2))
    print(args.output,folded['folding_validation'],flush=True)


if __name__=='__main__':main()
