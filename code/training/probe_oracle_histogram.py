"""Train-only geometric readout of the SAME normalized oracle symbol samples.

No constellation templates, class-conditioned transform, true SNR, noise
variance or generator symbols are inputs. This does not synchronize C01.
"""
from pathlib import Path
import sys
import warnings

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from training.diagnose_oracle_reception import classification,LEVELS
from utils.common import write_json


def histogram(path):
    data=np.load(path,mmap_mode='r');features=np.zeros((len(data),32*32),np.float32)
    for start in range(0,len(data),128):
        x=np.array(data[start:start+128,:896:8])
        scale=np.sqrt(np.square(x).sum(-1).mean(1)).clip(1e-6)
        x=x/scale[:,None,None]
        coordinate=((x+2)*31/4).clip(0,31);left=np.floor(coordinate).astype(np.int64);fraction=coordinate-left
        b=len(x);row=np.arange(b)[:,None];block=features[start:start+b]
        for di in [0,1]:
            for dq in [0,1]:
                i=(left[...,0]+di).clip(0,31);q=(left[...,1]+dq).clip(0,31)
                weight=(fraction[...,0] if di else 1-fraction[...,0])*(fraction[...,1] if dq else 1-fraction[...,1])
                np.add.at(block,(row,i*32+q),weight/112)
    np.testing.assert_allclose(features.sum(1),1,atol=1e-5)
    return features


def main():
    if torch.cuda.is_available():raise ValueError('CPU only: hide CUDA devices')
    root=ROOT/'artifacts/causal_diagnosis_20261114/oracle_reception';out=root/'histogram_probe';out.mkdir(exist_ok=False)
    train=histogram(root/'train_oracle.npy');test=histogram(root/'test_oracle.npy')
    a=np.load(root/'train_metadata.npz');b=np.load(root/'test_metadata.npz')
    scaler=StandardScaler();train=scaler.fit_transform(train);test=scaler.transform(test)
    model=LogisticRegression(C=1.,max_iter=500,tol=1e-4,solver='lbfgs')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning);model.fit(train,a['labels'])
    z=model.decision_function(test);zt=model.decision_function(train)
    result=dict(overall=classification(z,b['labels']),training=classification(zt,a['labels']),
        by_snr={str(level):classification(z[b['snr']==level],b['labels'][b['snr']==level]) for level in LEVELS},
        protocol=dict(input='112 known-receiver samples, per-record RMS normalization exactly as repeated-symbol RF input',
            features='32x32 bilinear histogram, coordinates [-2,2], all observations counted',
            classifier='Train-standardized logistic regression, C=1, max_iter=500, no test selection',
            used_class_templates=False,used_true_snr_or_noise_variance=False,train_records=len(train),test_records=len(test),
            convergence_warnings=[str(w.message) for w in caught]))
    np.savez_compressed(out/'predictions.npz',logits=z,labels=b['labels'],snr=b['snr'])
    np.savez(out/'weights.npz',coefficient=model.coef_,intercept=model.intercept_,mean=scaler.mean_,scale=scaler.scale_)
    write_json(out/'results.json',result);print('HISTOGRAM',result['overall']['accuracy'],{k:v['accuracy'] for k,v in result['by_snr'].items()},'warnings',result['protocol']['convergence_warnings'],flush=True)


if __name__=='__main__':main()
