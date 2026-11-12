"""Fold a frozen-base MLP readout and train-only standardization into one RF model."""
import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from models.rf_model import RFModel
from utils.common import model_bundle, read_json, save_checkpoint
from utils.config import resolve_config
from utils.contrastive import load_with_training_aux


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--readout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    readout = torch.load(args.readout, map_location='cpu', weights_only=True)
    if readout['head_type'] != 'mlp':
        raise ValueError('This promotion preserves the original MLP architecture; choose an MLP readout')
    parent = args.readout.parent.parent
    protocol = read_json(parent / 'protocol.json')
    source = torch.load(readout['source_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    if source['stage'] != 'supervised':
        raise ValueError('Promote a jointly supervised base so that all competition task heads are trained')
    if source['dataset_fingerprint'] != readout['dataset_fingerprint']:
        raise ValueError('Readout/base dataset mismatch')
    bundle = model_bundle(source)
    pooling = protocol.get('pooling', 'mean')
    bundle['model_config'] = {**bundle['model_config'], 'classification_pooling': pooling,
                              'physical_statistics': protocol.get('physical_statistics', False),
                              'classification_fp32': True,
                              'contrastive_head': False, 'pretrain_heads': False, 'atomic_vocab': 0, 'emergent_vocab': 0}
    model = RFModel(**bundle).eval()
    load_with_training_aux(model, source['model'], source['model_config'])
    state = dict(readout['head'])
    mean, std = readout['feature_mean'], readout['feature_std']
    raw_weight = state['0.weight'] / std[None]
    raw_bias = state['0.bias'] - raw_weight @ mean
    state['0.weight'], state['0.bias'] = raw_weight, raw_bias
    model.heads['modulation'].load_state_dict(state, strict=True)
    # Numerical equivalence of standardization folding on actual cached features.
    import numpy as np
    cached = np.load(parent / 'validation_features.npy', mmap_mode='r')
    positions = np.linspace(0, len(cached) - 1, min(256, len(cached)), dtype=int)
    x = torch.from_numpy(np.array(cached[positions]))
    reference = torch.nn.Sequential(torch.nn.Linear(len(mean), model.heads['modulation'][0].out_features),
        torch.nn.GELU(), torch.nn.Linear(model.heads['modulation'][0].out_features,
                                       model.heads['modulation'][2].out_features))
    reference.load_state_dict(readout['head'])
    with torch.no_grad():
        expected = reference((x - mean) / std)
        actual = model.heads['modulation'](x)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    config = resolve_config(source['training_config'])
    config['model']['heads']['classification_pooling'] = pooling
    config['model']['heads']['physical_statistics'] = protocol.get('physical_statistics', False)
    config['model']['heads']['classification_fp32'] = True
    config['loss']['supervised']['contrastive']['weight'] = 0.
    count = len(np.load(parent / 'train_labels.npy', mmap_mode='r'))
    head_updates = math.ceil(count / protocol['batch_size']) * readout['epochs']
    payload = dict(model=model.state_dict(), model_config=model.config, schema=bundle['schema'],
        target_scaler=bundle['target_scaler'], parameter_grids=bundle['parameter_grids'],
        dataset_fingerprint=source['dataset_fingerprint'], stage='supervised', training_config=config,
        codebook_signature=None, global_step=head_updates, epoch=readout['epochs'] - 1,
        initialization_source_steps=source['global_step'] + source.get('repair_source_steps', source.get('initialization_source_steps', 0)),
        metrics={}, source_base_checkpoint=readout['source_checkpoint'], source_readout=str(args.readout.resolve()),
        readout_protocol=protocol, resume_allowed=False,
        folding_maximum_logit_error=float((actual - expected).abs().max()))
    save_checkpoint(args.output, payload)
    print({'checkpoint': str(args.output), 'parameters': model.parameter_report(),
           'folding_maximum_logit_error': payload['folding_maximum_logit_error']})


if __name__ == '__main__':
    main()
