"""Check diagnostic controls, state isolation and batch-independent inference."""
from pathlib import Path
import sys
import json

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'compare/IQFormer'))
import numpy as np
import torch
from models.rf_model import RFModel
from training.diagnose_oracle_reception import constellations
from training.calibrate_iqformer_by_length import LengthMoments
from c01.runtime import build_model,setup_runtime
from c01.data import C01Modulation,SignalViews
from utils.common import model_bundle,write_json


def main():
    if torch.cuda.is_available():raise ValueError('CPU checks only')
    torch.set_num_threads(4);root=ROOT/'artifacts/causal_diagnosis_20261114';out=root/'summary'
    meta=np.load(ROOT/'artifacts/data_full/metadata.npz')
    calibration=json.loads((root/'iqformer_population_bn_v2/protocol.json').read_text())['training_rows']
    assert (meta['split'][calibration]==0).all()
    for alphabet in constellations():np.testing.assert_allclose(np.abs(alphabet).mean() if len(alphabet)==2 else np.mean(np.abs(alphabet)**2),1.,atol=1e-6)
    source=torch.load(ROOT/'artifacts/reception_exploration/20261114_104034_511636/source.pt',map_location='cpu',weights_only=True,mmap=True)
    main_model=RFModel(**model_bundle(source))
    inventory=dict(batchnorm=sum(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in main_model.modules()),
        layernorm=sum(isinstance(m,torch.nn.LayerNorm) for m in main_model.modules()))
    assert inventory['batchnorm']==0;del main_model,source
    checks={}
    for estimator in ['best','last']:
        ck=torch.load(root/'iqformer_population_bn_v2'/(estimator+'_length_layerwise_state.pt'),map_location='cpu',weights_only=True,mmap=True)
        original=torch.load(ck['source_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
        setup_runtime(ck['config'],'cpu');model=build_model(ck['config'],ck['labels']);model.load_state_dict(ck['model'],strict=True)
        for name,p in model.named_parameters():assert torch.equal(p,original['model'][name]),name
        wrapped=LengthMoments(model,ck['length_moments'],device='cpu').eval();data=C01Modulation(ck['config']['run']['data'],'train');views=SignalViews(ck['config']['input'])
        examples={}
        for length in [512,1024]:
            indices=np.flatnonzero(data.lengths==length)[:2];examples[length]=views([data[int(i)] for i in indices])
        with torch.inference_mode():
            for length in [512,1024]:
                batch=examples[length];a=wrapped(batch['iq'],batch['stft']);b=wrapped(batch['iq'][:1],batch['stft'][:1])
                alternate=examples[1536-length];wrapped(alternate['iq'],alternate['stft'])
                c=wrapped(batch['iq'],batch['stft'])
                torch.testing.assert_close(a[:1],b,rtol=1e-4,atol=1e-4);torch.testing.assert_close(a,c,rtol=0,atol=0)
                checks[f'{estimator}/{length}']=dict(single_vs_batch_max=float((a[:1]-b).abs().max()),length_switch_difference=float((a-c).abs().max()))
    result=dict(status='passed',calibration_only_training_split=True,learned_iqformer_weights_unchanged=True,
        original_main_normalization_inventory=inventory,inference_independence=checks,
        constellation_definitions_unit_power=True,control_hash_verification='summarize_causal_diagnosis checks initial tensors and exact training streams')
    write_json(out/'checks.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
