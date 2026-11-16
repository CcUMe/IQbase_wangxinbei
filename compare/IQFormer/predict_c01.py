"""Read official C01 Parquet and emit modulation-only prediction JSONL."""
import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[1] / 'code'))

import numpy as np
import pyarrow.parquet as pq
import torch

from c01.data import SignalViews
from c01.runtime import autocast, load_trained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='Official Parquet; only task=modulation is predicted')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    model, config, labels, device, _ = load_trained(args.checkpoint, args.device)
    view = SignalViews(config['input'], training=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predicted, skipped = 0, 0
    with args.output.open('x', encoding='utf-8') as handle, torch.inference_mode():
        for chunk in pq.ParquetFile(args.input).iter_batches(batch_size=args.batch_size, columns=['task', 'sample_id', 'iq']):
            records = chunk.to_pylist()
            outputs, groups = {}, {}
            for i, record in enumerate(records):
                if record['task'] != 'modulation':
                    skipped += 1
                    continue
                iq = np.asarray(record['iq'], dtype=np.float32)
                if iq.ndim != 2 or iq.shape[1] != 2 or len(iq) not in [512, 1024] or not np.isfinite(iq).all():
                    raise ValueError(f'Invalid IQ shape/values: {record["sample_id"]}')
                groups.setdefault(len(iq), []).append((i, {'iq': iq, 'sample_id': record['sample_id']}))
            for items in groups.values():
                batch = view([row for _, row in items])
                with autocast(config, device):
                    logits = model(batch['iq'].to(device), batch['stft'].to(device)).float()
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError('Nonfinite prediction logits')
                for (i, row), label in zip(items, logits.argmax(-1).tolist()):
                    outputs[i] = {'task': 'modulation', 'sample_id': row['sample_id'], 'prediction': labels[label]}
            for i in sorted(outputs):
                handle.write(json.dumps(outputs[i], ensure_ascii=False) + '\n')
                predicted += 1
    print(f'预测完成 | modulation={predicted} | 跳过其他任务={skipped} | {args.output}')


if __name__ == '__main__':
    main()
