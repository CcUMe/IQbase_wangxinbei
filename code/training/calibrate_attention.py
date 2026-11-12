"""Validate a bounded global QK scale without retraining or ensembling."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader, Subset
from models.rf_model import RFModel
from training.repair_experiment import validation_rows
from utils.common import model_bundle, runtime, save_checkpoint, write_json
from utils.dataset import PreparedDataset, collate
from utils.engine import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scales', nargs='+', type=float, default=[.75, 1., 1.25, 1.5, 2.])
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    if any(scale <= 0 or scale > 4 for scale in args.scales):
        p.error('Calibration scales must be in (0,4]')
    args.output.mkdir(parents=True, exist_ok=False)
    device, _, world = runtime(args.device)
    if world != 1:
        raise ValueError('Single-process calibration')
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    data = PreparedDataset(args.data, 'validation')
    if data.info['fingerprint'] != source['dataset_fingerprint']:
        raise ValueError('Different data split')
    model = RFModel(**model_bundle(source)).to(device).eval()
    model.load_state_dict(source['model'])
    if not model.config['qk_normalization']:
        raise ValueError('Calibrate an already repaired QK-normalized model')
    indices, rows = validation_rows(data)
    loader = DataLoader(Subset(data, indices), batch_size=128, collate_fn=collate, num_workers=2)
    results, metrics_by_scale = [], {}
    for scale in args.scales:
        for block in model.backbone.blocks:
            block.attention.qk_scale = scale
        metrics = evaluate(model, loader, device, source['target_scaler'], source['schema'], 'fp32')
        row = {'qk_scale': scale, 'proxy_score': metrics['proxy_score'],
               'modulation': metrics['classification']['modulation']['accuracy'],
               'protocol': metrics['classification']['protocol']['accuracy'], 'parameter_proxy': metrics['parameter_proxy']}
        results.append(row)
        metrics_by_scale[scale] = metrics
        print(row, flush=True)
    chosen = max(results, key=lambda row: row['proxy_score'])
    source['model_config'] = {**source['model_config'], 'qk_scale': chosen['qk_scale']}
    source['training_config']['model']['encoder']['qk_scale'] = chosen['qk_scale']
    source['validation_subset'] = True
    source['metrics'] = metrics_by_scale[chosen['qk_scale']]
    source['best_score'] = chosen['proxy_score']
    source['attention_calibration'] = {'results': results, 'selected': chosen, 'rows': rows}
    save_checkpoint(args.output / 'model.pt', source)
    write_json(args.output / 'calibration.json', {'results': results, 'selected': chosen, 'validation_rows': len(rows)})


if __name__ == '__main__':
    main()
