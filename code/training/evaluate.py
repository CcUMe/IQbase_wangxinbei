"""Evaluate one checkpoint with valid masks and original target units."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch.utils.data import DataLoader
from models.rf_model import RFModel
from utils.common import load_checkpoint, model_bundle, runtime, write_json
from utils.dataset import PreparedDataset, collate
from utils.engine import evaluate
from utils.training_display import classification_summary
from utils.command_config import command_defaults, checkpoint_reference, configured_launch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--gpus', default='')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--precision', choices=['fp32', 'bf16', 'fp16'])
    p.add_argument('--max-batches', type=int, default=0, help='0 always evaluates the full validation set')
    command_defaults(p, {'run': {'data': 'data', 'checkpoint': 'checkpoint', 'output': 'output'},
                         'runtime': {'device': 'device', 'gpus': 'gpus', 'batch_size': 'batch_size',
                                     'precision': 'precision', 'max_batches': 'max_batches'}})
    args = p.parse_args()
    configured_launch(args.gpus)
    args.checkpoint = checkpoint_reference(args.checkpoint)
    checkpoint = load_checkpoint(args.checkpoint)
    saved = checkpoint['training_config']
    settings = saved['evaluation'] if saved.get('schema_version') == 2 else {}
    threads = saved['runtime']['threads'] if saved.get('schema_version') == 2 else saved.get('threads', 4)
    batch_size = args.batch_size if args.batch_size is not None else settings.get('batch_size', 128)
    precision = args.precision or settings.get('precision', 'fp32')
    if batch_size < 1 or args.max_batches < 0:
        p.error('batch-size must be positive and max-batches nonnegative')
    device, _, world = runtime(args.device, threads)
    if world > 1:
        raise ValueError('Evaluation is a single-process command')
    dataset = PreparedDataset(args.data, 'validation')
    if checkpoint['dataset_fingerprint'] != dataset.info['fingerprint']:
        raise ValueError('Checkpoint and validation dataset fingerprints differ')
    model = RFModel(**model_bundle(checkpoint)).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    metrics = evaluate(model, loader, device, checkpoint['target_scaler'], checkpoint['schema'], precision,
                       args.max_batches, settings.get('proxy_task_weights'))
    write_json(args.output, metrics)
    classification_summary(metrics)
    print(f"Validation rows={metrics['rows']}; local proxy={metrics['proxy_score']:.4f} (not official score)")


if __name__ == '__main__':
    main()
