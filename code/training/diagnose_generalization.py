"""Compare clean and constant-phase-rotated train/validation signals without fitting."""
import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from models.rf_model import RFModel
from utils.augmentation import rotate_phase
from utils.common import model_bundle, runtime, write_json
from utils.dataset import PreparedDataset, collate, to_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-class', type=int, default=128)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    device, _, _ = runtime(args.device, 4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
    model = RFModel(**model_bundle(checkpoint)).to(device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    report = {'checkpoint': str(args.checkpoint.resolve()), 'per_class': args.per_class,
              'dataset_fingerprint': checkpoint['dataset_fingerprint'], 'splits': {}}
    for split in ['train', 'validation']:
        dataset = PreparedDataset(args.data, split)
        if dataset.info['fingerprint'] != checkpoint['dataset_fingerprint']:
            raise ValueError('Dataset fingerprint differs')
        rows = dataset.indices
        # Exact same-waveform labels are used only to describe evaluation slices.
        # These labels are never passed to the model or its preprocessing.
        _, inverse, counts = np.unique(dataset.meta['hash'], return_inverse=True, return_counts=True)
        paired_rows = np.flatnonzero((dataset.meta['task'] == 2) & (counts[inverse] > 1))
        paired = {bytes(dataset.meta['hash'][r]): int(r) for r in paired_rows}
        rng = np.random.default_rng(3981)
        results = {}
        for task in ['modulation', 'protocol']:
            labels = dataset.meta[task][rows]
            positions = []
            for label in np.unique(labels[labels >= 0]):
                eligible = np.flatnonzero(labels == label)
                positions.extend(rng.choice(eligible, min(args.per_class, len(eligible)), replace=False).tolist())
            loader = DataLoader(Subset(dataset, positions), batch_size=128, collate_fn=collate, num_workers=2)
            all_logits, all_rotated, targets = [], [], []
            with torch.inference_mode():
                for cpu in loader:
                    batch = to_device(cpu, device)
                    iq = batch['iq']
                    angles = torch.full((len(iq),), math.pi / 2, device=device)
                    all_logits.append(model(iq, batch['lengths'])[task].cpu())
                    all_rotated.append(model(rotate_phase(iq, angles), batch['lengths'])[task].cpu())
                    targets.append(batch[task].cpu())
            logits, rotated, target = torch.cat(all_logits), torch.cat(all_rotated), torch.cat(targets)
            prediction, rotated_prediction = logits.argmax(-1), rotated.argmax(-1)
            results[task] = {
                'rows': len(target), 'accuracy': float((prediction == target).float().mean()),
                'cross_entropy': float(F.cross_entropy(logits, target)),
                'mean_confidence': float(logits.softmax(-1).max(-1).values.mean()),
                'rotated_accuracy': float((rotated_prediction == target).float().mean()),
                'prediction_flip_fraction': float((prediction != rotated_prediction).float().mean()),
                'per_class_accuracy': [float((prediction[target == j] == j).float().mean()) for j in range(logits.shape[-1])],
            }
            correct = (prediction == target).numpy()
            selected_rows = rows[positions]
            slices = {}
            for field in [0, 2]:
                groups = {}
                for i, row in enumerate(selected_rows):
                    parameter_row = paired.get(bytes(dataset.meta['hash'][row]))
                    if parameter_row is None or not dataset.meta['target_valid'][parameter_row, field]:
                        continue
                    value = float(dataset.meta['targets'][parameter_row, field])
                    groups.setdefault(value, []).append(bool(correct[i]))
                slices[dataset.info['schema']['parameter_targets'][field]] = {
                    str(value): {'support': len(values), 'accuracy': float(np.mean(values))}
                    for value, values in sorted(groups.items())}
            results[task]['exact_paired_parameter_slices'] = slices
            results[task]['by_length'] = {
                str(int(length)): {'support': int((dataset.meta['length'][selected_rows] == length).sum()),
                                   'accuracy': float(correct[dataset.meta['length'][selected_rows] == length].mean())}
                for length in np.unique(dataset.meta['length'][selected_rows])}
            print(split, task, results[task], flush=True)
        report['splits'][split] = results
    write_json(args.output, report)


if __name__ == '__main__':
    main()
