"""Measure retention on fixed SYNTHETIC clock tests, never infer C01 truth."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from models.reception import ReceptionMemory
from training.precalibrate_reception import assess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise ValueError('Use CUDA_VISIBLE_DEVICES empty; CPU check only')
    torch.set_num_threads(4)
    reference = torch.load(args.calibration, map_location='cpu', weights_only=True)
    result = dict(reference=str(args.calibration), original_metrics=reference['metrics'],
        protocol='Same synthetic generation and seeds 9091 / 9092 as calibration; 256 records each',
        caveat='These are not C01 period measurements or modulation accuracy', checkpoints={})
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        options = checkpoint['reception_options']
        if not options or options.get('kind') == 'carrier_input':
            raise ValueError('Needs a reception-memory checkpoint')
        config = checkpoint['model_config']
        receiver = ReceptionMemory(config['width'], config['patch_size'], options)
        prefix = 'backbone.reception.'
        state = {k.removeprefix(prefix):v for k,v in checkpoint['model'].items() if k.startswith(prefix)}
        receiver.load_state_dict(state, strict=True)
        identical = all(torch.equal(receiver.state_dict()[k], v) for k,v in reference['predictor'].items())
        result['checkpoints'][str(path)] = dict(predictor_identical_to_calibration=identical,
            in_distribution=assess(receiver,9091),heldout_256qam=assess(receiver,9092,True))
        del checkpoint, state, receiver
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
