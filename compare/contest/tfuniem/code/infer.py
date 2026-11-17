"""Unified inference entry: Parquet (task, sample_id, iq) -> JSONL (task, sample_id, prediction).

CLI:  python code/infer.py INPUT.parquet OUTPUT.jsonl [--weights PATH]
      python code/infer.py --input INPUT.parquet --output OUTPUT.jsonl [--weights PATH]
API:  model = load_model(weights_path); predict(input_parquet, output_jsonl, model)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.decode import decode_parameters
from models.tfuniem import TFUniEM
from utils.common import TARGETS, TASKS, iq_column_to_flat

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / 'weights' / 'best.pt'


def load_model(weights=DEFAULT_WEIGHTS, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(weights, map_location='cpu', weights_only=False)
    model = TFUniEM(**state['model_cfg'])
    model.load_state_dict(state['model'])
    model.eval().fuse().to(device)
    model.bundle = {k: state[k] for k in ['head_stats', 'schema']}
    model.device = device
    return model


@torch.no_grad()
def predict(input_parquet, output_jsonl, model, batch_size=1024):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    table = pq.read_table(input_parquet, columns=['task', 'sample_id', 'iq'])
    tasks = table['task'].to_pylist()
    ids = table['sample_id'].to_pylist()
    lengths, flat, off = iq_column_to_flat(table['iq'])
    names = model.bundle['schema']['classification_labels']
    hs = model.bundle['head_stats']
    preds = [None] * len(tasks)
    task_idx = np.array([TASKS.index(t) for t in tasks])
    for L in sorted(set(lengths.tolist())):
        rows = np.flatnonzero(lengths == L)
        for i in range(0, len(rows), batch_size):
            r = rows[i:i + batch_size]
            x = np.stack([flat[off[j]:off[j + 1]] for j in r]).reshape(len(r), L, 2)
            x = torch.from_numpy(x).to(model.device)
            t = torch.from_numpy(task_idx[r]).to(model.device)
            out = model(x, t)
            mod = out['modulation'].float().argmax(1).cpu().numpy()
            pro = out['protocol'].float().argmax(1).cpu().numpy()
            par = decode_parameters(out['parameter'], hs, 'expect', 'expect')
            for k, j in enumerate(r):
                if tasks[j] == 'modulation':
                    preds[j] = names['modulation'][mod[k]]
                elif tasks[j] == 'protocol':
                    preds[j] = names['protocol'][pro[k]]
                else:
                    preds[j] = {name: float(par[k, c]) for c, name in enumerate(TARGETS)}
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for task, sid, p in zip(tasks, ids, preds):
            f.write(json.dumps({'task': task, 'sample_id': sid, 'prediction': p}, ensure_ascii=False) + '\n')
    return len(preds)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pos', nargs='*')
    ap.add_argument('--input')
    ap.add_argument('--output')
    ap.add_argument('--weights', default=str(DEFAULT_WEIGHTS))
    a = ap.parse_args()
    inp = a.input or (a.pos[0] if a.pos else None)
    out = a.output or (a.pos[1] if len(a.pos) > 1 else None)
    if not inp or not out:
        ap.error('need input and output paths')
    t0 = time.time()
    model = load_model(a.weights)
    n = predict(inp, out, model)
    print(f'wrote {n} predictions to {out} in {time.time() - t0:.2f}s', flush=True)


if __name__ == '__main__':
    main()
