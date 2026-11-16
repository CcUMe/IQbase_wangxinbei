"""Evaluate a C01 IQFormer checkpoint, with optional per-row logits export."""
import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[1] / 'code'))

import numpy as np

from c01.runtime import C01Modulation, evaluate, load_trained, make_loader, print_metrics, save_metrics
from utils.readable_log import ReadableRunLog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, help='Default: original checkpoint data directory')
    parser.add_argument('--split', choices=['train', 'validation'], default='validation')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--max-batches', type=int, default=0)
    parser.add_argument('--save-predictions', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.max_batches < 0 or (args.batch_size is not None and args.batch_size < 1) or (args.num_workers is not None and args.num_workers < 0):
        parser.error('Invalid batch/worker limits')
    model, config, labels, device, checkpoint = load_trained(args.checkpoint, args.device)
    if args.batch_size:
        config['data']['eval_batch_size'] = args.batch_size
    if args.num_workers is not None:
        config['data']['num_workers'] = args.num_workers
    dataset = C01Modulation(args.data or config['run']['data'], args.split)
    if dataset.info['fingerprint'] != checkpoint['dataset_fingerprint'] or dataset.labels != labels:
        raise ValueError('Checkpoint and dataset split/schema differ')
    loader, _ = make_loader(dataset, config, False)
    args.output.mkdir(parents=True, exist_ok=False)
    with ReadableRunLog(args.output):
        metrics, predictions = evaluate(model, loader, config, device, args.max_batches, args.save_predictions)
        metrics.update(checkpoint=str(args.checkpoint.resolve()), split=args.split,
                       dataset_fingerprint=dataset.info['fingerprint'], checkpoint_step=checkpoint['global_step'])
        print_metrics(metrics)
        save_metrics(args.output, metrics)
        if predictions:
            np.savez_compressed(args.output / 'predictions.npz', **predictions)
        print(f'结果 | {args.output} | complete={metrics["complete"]}', flush=True)


if __name__ == '__main__':
    main()
