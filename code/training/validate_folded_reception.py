"""Complete single-GPU validation of an already CPU-checked raw-fold candidate."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from models.rf_model import RFModel
from training.validate_c01_followup import predict,task_metrics
from training.explore_reception import condition_metrics
from utils.common import model_bundle,save_checkpoint,write_json
from utils.dataset import PreparedDataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--reference-predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or torch.cuda.device_count()!=1:
        raise ValueError('Use only CUDA_VISIBLE_DEVICES=0 for this matched check')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if checkpoint.get('research_kind')!='folded_raw_input_projection':raise ValueError('Expected a folded research candidate')
    data=PreparedDataset(ROOT/'artifacts/data_full','validation')
    if data.info['fingerprint']!=checkpoint['dataset_fingerprint']:raise ValueError('Different dataset')
    iq=np.load(ROOT/'artifacts/data_full/iq.npy',mmap_mode='r')
    selection=dict(np.load(ROOT/'artifacts/c01_followup_validation/20261113_224434/selection.npz'))
    model=RFModel(**model_bundle(checkpoint)).cuda().eval();model.load_state_dict(checkpoint['model'],strict=True)
    pred=predict(model,data.indices,iq,data.meta,checkpoint['target_scaler'],'cuda:0',128,'folded_raw_full_validation')
    metric=task_metrics(pred,data.meta,checkpoint['schema'])
    metric['conditions']=condition_metrics(pred,data.meta,checkpoint['schema'],selection)
    ref=dict(np.load(args.reference_predictions));assert np.array_equal(pred['rows'],ref['rows'])
    differences={}
    for task in ['modulation','protocol']:
        keep=data.meta[task][pred['rows']]>=0
        differences[task]=dict(support=int(keep.sum()),changed_labeled_predictions=int((pred[task][keep].argmax(1)!=ref[task][keep].argmax(1)).sum()),
            maximum_absolute_logit_difference=float(np.abs(pred[task][keep]-ref[task][keep]).max()))
    differences['maximum_absolute_parameter_differences_raw_units']=np.abs(pred['parameters']-ref['parameters']).max(0).tolist()
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'folded_validation.npz',**pred)
    write_json(args.output/'folded_metrics.json',metric);write_json(args.output/'folded_equivalence.json',differences)
    # Atomic replacement updates only this turn's new candidate, never its
    # original trained source or the default model. Metrics reflect actual GPU
    # floating-point evaluation rather than copying the pre-fold score.
    save_checkpoint(args.checkpoint,{**checkpoint,'metrics':metric,'folding_full_validation':differences})
    print(json.dumps(dict(classification={k:{f:v[f] for f in ['accuracy','macro_f1','weighted_f1']} for k,v in metric['classification'].items()},differences=differences)),flush=True)


if __name__=='__main__':main()
