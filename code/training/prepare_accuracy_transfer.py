"""Relocate the three matched experiments to another checkout. Never starts training."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from utils.common import read_json, write_json
from utils.config import load_config, validate_config
from utils.training_workflow import ROOT, validate_devices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initializations', type=Path, required=True,
                        help='Directory holding the three *_random_init.pt files')
    parser.add_argument('--data', type=Path, default=ROOT / 'artifacts/data_full_p8')
    parser.add_argument('--gpus', default='2,3', help='Physical GPU pair; experiments run first/second/first serially')
    parser.add_argument('--output', type=Path, default=ROOT / 'code/configs/experiments/accuracy_fusion_transfer.json')
    args = parser.parse_args()
    cards = validate_devices(args.gpus, 2, 'auto')
    data, initializations, output = args.data.resolve(), args.initializations.resolve(), args.output.resolve()
    if output.exists():
        parser.error('Output already exists; choose a new --output to preserve the previous experiment')
    for name in ['dataset.json', 'iq.npy', 'metadata.npz']:
        if not (data / name).is_file():
            parser.error(f'Missing data file or broken symlink: {data / name}; copy the P8 cache with rsync -L')
    info = read_json(data / 'dataset.json')
    if info['patch_size'] != 8:
        parser.error('These experiments require the unchanged P8 prepared cache')
    iq = np.load(data / 'iq.npy', mmap_mode='r')
    if iq.shape != (info['rows'], 1024, 2):
        parser.error('IQ cache dimensions do not match dataset.json')
    stages = []
    planned = []
    for name, gpu in zip(['control', 'gate_norm', 'window256'], [cards[0], cards[1], cards[0]]):
        initial = initializations / (name + '_random_init.pt')
        if not initial.is_file():
            parser.error('Missing matched random initialization: ' + str(initial))
        state = torch.load(initial, map_location='cpu', weights_only=True, mmap=True)
        if state['global_step'] != 0 or state.get('initialization_kind') != 'matched_random_initialization':
            parser.error('Expected the original matched RANDOM initialization, not trained weights')
        if state['dataset_fingerprint'] != info['fingerprint']:
            parser.error('Initialization/data fingerprint mismatch; keep the original prepared split')
        for key in ['schema', 'target_scaler', 'parameter_grids']:
            if state[key] != info[key]:
                parser.error('Initialization/data contract mismatch: ' + key)
        config = load_config(ROOT / f'code/configs/supervised/accuracy_fusion_20261116/{name}.json')
        config['run'].update(data=str(data), warmstart_checkpoint=str(initial), init_checkpoint='',
                             output=str(ROOT / 'logs' / output.stem / name))
        config['runtime'].update(gpus=gpu, processes=1)
        config['checkpoint']['resume_from'] = ''
        validate_config(config)
        path = output.parent / (output.stem + '_configs') / (name + '.json')
        if path.exists():
            parser.error('Generated stage config exists: ' + str(path))
        planned.append((path, config))
        stages.append({'name': name, 'config': str(path)})
        del state
    for path, config in planned:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {'workflow': 'experiment_suite',
                       'run': {'output': str(ROOT / 'logs' / output.stem / 'suite')},
                       'runtime': {'gpus': ','.join(cards)}, 'experiments': stages})
    print('Prepared fresh matched supervised experiments; no training started.')
    print('Start from repository root:')
    print(f'python code/training/train.py --config {output} --background')


if __name__ == '__main__':
    main()
