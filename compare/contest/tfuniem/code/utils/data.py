"""Cache loading, per-row training labels, balanced batch sampling and validation iteration."""
import json
from pathlib import Path

import numpy as np
import torch

LENGTH_INDEX = {512: 0, 1024: 1}


class Cache:
    def __init__(self, root):
        root = Path(root)
        self.root = root
        self.info = json.loads((root / 'dataset.json').read_text(encoding='utf-8'))
        m = np.load(root / 'meta.npz', allow_pickle=False)
        self.meta = {k: m[k] for k in m.files}
        self.iq = {int(L): np.memmap(root / f'iq_{L}.f32', dtype=np.float32, mode='r', shape=(int(n), int(L), 2))
                   for L, n in self.info['cache_counts'].items()}
        self.labels = build_labels(self.meta, self.info)

    def rows(self, split, task=None):
        sel = self.meta['split'] == split
        if task is not None:
            sel &= self.meta['task'] == task
        return np.flatnonzero(sel)


def build_labels(meta, info):
    """Vectorised targets for every row. -1 / NaN where not applicable."""
    h = info['head_stats']
    n = len(meta['task'])
    T, V = meta['targets'], meta['valid']
    comm, pulse = V[:, 0], V[:, 3]
    group = np.full(n, -1, np.int64)
    group[pulse] = 0
    group[comm] = 1
    combo_index = {tuple(c): i for i, c in enumerate(h['sr_bw_combos'])}
    combo = np.full(n, -1, np.int64)
    for i in np.flatnonzero(comm):
        combo[i] = combo_index.get((T[i, 0], T[i, 1]), -1)
    fs = T[:, 2]
    fs_cls = np.full(n, -1, np.int64)
    fs_cls[comm & (np.abs(fs) <= 50)] = 0
    fs_cls[comm & (fs == 100)] = 1
    fs_cls[comm & (fs == 1000)] = 2
    if (fs_cls[comm] < 0).any():
        raise ValueError('frequency_shift value outside {|f|<=50, 100, 1000}')
    fs_reg = np.full(n, np.nan, np.float32)
    small = fs_cls == 0
    fs_reg[small] = (fs[small] - h['fshift_small_mean']) / h['fshift_small_std']
    npl = np.full(n, -1, np.int64)
    npl[pulse] = np.rint(T[pulse, 6]).astype(np.int64) - 2
    times = np.full((n, 3), np.nan, np.float32)
    for j, col in enumerate((3, 4, 5)):
        times[pulse, j] = (T[pulse, col] - h['time_mean'][j]) / h['time_std'][j]
    return {'task': meta['task'].astype(np.int64), 'mod': meta['mod'].astype(np.int64),
            'pro': meta['pro'].astype(np.int64), 'group': group, 'combo': combo, 'fs_cls': fs_cls,
            'fs_reg': fs_reg, 'npulses': npl, 'times': times}


def gather(cache, rows):
    """Returns {L: dict of tensors} with one sub-batch per length present in rows (rows sorted per length)."""
    out = {}
    lengths = cache.meta['length'][rows]
    for L in (512, 1024):
        r = np.sort(rows[lengths == L])
        if not len(r):
            continue
        pos = cache.meta['pos'][r]
        x = torch.from_numpy(np.ascontiguousarray(cache.iq[L][pos]))
        lab = {k: torch.from_numpy(np.ascontiguousarray(v[r])) for k, v in cache.labels.items()}
        lab['row'] = torch.from_numpy(r)
        out[L] = {'iq': x, **lab}
    return out


class BalancedBatches(torch.utils.data.IterableDataset):
    """Infinite stream of steps: per step sample counts per task (uniform, with replacement across steps)."""

    def __init__(self, cache_root, per_task, seed, rank=0, rows_override=None):
        self.cache_root, self.per_task, self.seed, self.rank = cache_root, per_task, seed, rank
        self.rows_override = rows_override

    def __iter__(self):
        cache = Cache(self.cache_root)
        wi = torch.utils.data.get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = np.random.default_rng([self.seed, self.rank, wid])
        pools = []
        for t, k in enumerate(self.per_task):
            pool = cache.rows(0, t) if self.rows_override is None else \
                self.rows_override[cache.meta['task'][self.rows_override] == t]
            pools.append((pool, k))
        while True:
            rows = np.concatenate([rng.choice(pool, k, replace=True) for pool, k in pools if k])
            yield gather(cache, rows)


def validation_batches(cache, rows, batch_size=1024):
    lengths = cache.meta['length'][rows]
    for L in (512, 1024):
        r = np.sort(rows[lengths == L])
        for i in range(0, len(r), batch_size):
            yield gather(cache, r[i:i + batch_size])[L]
