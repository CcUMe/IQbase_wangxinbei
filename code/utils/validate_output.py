"""Check output schema/order/finite values; never substitutes format for accuracy."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyarrow.parquet as pq
from utils.common import TARGETS, read_json


def validate(input_path, output_path, schema):
    expected = pq.ParquetFile(input_path).iter_batches(batch_size=4096, columns=['task', 'sample_id'])
    count = 0
    with Path(output_path).open() as file:
        for batch in expected:
            for row in batch.to_pylist():
                line = file.readline()
                if not line:
                    raise ValueError('Missing output rows')
                result = json.loads(line)
                if set(result) != {'task', 'sample_id', 'prediction'} or result['task'] != row['task'] or result['sample_id'] != row['sample_id']:
                    raise ValueError('Output keys, IDs or input order mismatch')
                pred = result['prediction']
                if row['task'] == 'parameter':
                    if not isinstance(pred, dict) or set(pred) != set(TARGETS):
                        raise ValueError('Parameter output must have exactly seven fields')
                    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in pred.values()):
                        raise ValueError('Parameters must be finite numbers')
                elif pred not in schema['classification_labels'][row['task']]:
                    raise ValueError('Unknown output class label')
                count += 1
        if file.read().strip():
            raise ValueError('Unexpected extra output rows')
    return count


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--schema', type=Path, required=True)
    args = p.parse_args()
    print(f'Validated {validate(args.input, args.output, read_json(args.schema))} output records')
