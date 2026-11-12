"""Stream rank outputs back into input order, replacing public output atomically."""
from contextlib import ExitStack
import heapq
import json
from pathlib import Path


def merge_prediction_shards(shards, output):
    output = Path(output)
    temporary = output.with_suffix(output.suffix + '.tmp')
    count = 0
    with ExitStack() as stack:
        streams = [stack.enter_context(Path(path).open(encoding='utf-8')) for path in shards]
        rows = heapq.merge(*[(json.loads(line) for line in stream) for stream in streams], key=lambda x:x['row'])
        with temporary.open('w',encoding='utf-8') as target:
            for row in rows:
                if row['row'] != count:
                    raise ValueError(f'Inference shards contain a missing/duplicate row at {count}')
                target.write(json.dumps(row['result'],ensure_ascii=False,allow_nan=False)+'\n')
                count += 1
    temporary.replace(output)
    return count
