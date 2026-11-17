"""One-pass Parquet -> float32 memmap cache + hash-grouped split (reproduces the leader's split rule).

Outputs in --output:
  iq_512.f32 / iq_1024.f32   raw float32, shape (n_len, len, 2), C order
  meta.npz                   per-row arrays in original file order
  dataset.json               counts, split check, training-side statistics
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.common import (LENGTHS, TARGETS, TASKS, envelope_snr_db, hash_in_fraction, iq_column_to_flat,
                          read_json, split_salt, waveform_hash, write_json)

EXPECTED = {'train': 1060812, 'validation': 118182}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path, default=Path('/mnt/disk1/wangxinbei_contest_dataset/C01'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--batch-rows', type=int, default=8192)
    ap.add_argument('--cross-check-manifest', type=Path, default=None,
                    help='Optional existing sample_manifest.parquet (sample_id, iq_blake2b_128) to compare hashes')
    args = ap.parse_args()
    out = args.output
    if out.exists() and any(out.iterdir()):
        ap.error(f'{out} is not empty')
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log('pyarrow', pa.__version__, 'numpy', np.__version__)

    train_dir = args.data_root / 'rf_signal_benchmark_train_v1.0'
    io_dir = args.data_root / 'rf_signal_sample_io_v1.0'
    schema = read_json(train_dir / 'task_schema.json')
    mod_map = {n: i for i, n in enumerate(schema['classification_labels']['modulation'])}
    pro_map = {n: i for i, n in enumerate(schema['classification_labels']['protocol'])}

    # ---- official IO samples: hashes to exclude
    st = pq.read_table(io_dir / 'rf_signal_sample_io_input_v1.0.parquet')
    s_len, s_flat, s_off = iq_column_to_flat(st['iq'])
    sample_hashes = {waveform_hash(int(s_len[i]), s_flat[s_off[i]:s_off[i + 1]]) for i in range(st.num_rows)}
    sample_ids = set(st['sample_id'].to_pylist())
    log(f'IO samples: {st.num_rows} rows, {len(sample_hashes)} distinct hashes')

    salt_val, salt_tune = split_salt(b'2026'), split_salt(b'2027')
    fh = {L: open(out / f'iq_{L}.f32', 'wb') for L in LENGTHS}
    count = {L: 0 for L in LENGTHS}
    cols = {k: [] for k in ['sample_id', 'task', 'length', 'pos', 'hash', 'mod', 'pro', 'rms', 'env_snr',
                            'source_file', 'source_row']}
    targets, valids = [], []
    files = sorted(train_dir.glob('*.parquet'))
    for fi, path in enumerate(files):
        pf = pq.ParquetFile(path)
        row0 = 0
        for batch in pf.iter_batches(batch_size=args.batch_rows):
            n = batch.num_rows
            lengths, flat, off = iq_column_to_flat(batch.column('iq'))
            if not np.isfinite(flat).all():
                raise ValueError(f'non-finite IQ in {path.name} near row {row0}')
            bad = set(lengths.tolist()) - set(LENGTHS)
            if bad:
                raise ValueError(f'unexpected lengths {bad}')
            cols['hash'].extend(waveform_hash(int(lengths[i]), flat[off[i]:off[i + 1]]) for i in range(n))
            task = batch.column('task').to_pylist()
            cols['task'].append(np.array([TASKS.index(t) for t in task], np.int8))
            cols['sample_id'].extend(batch.column('sample_id').to_pylist())
            cols['length'].append(lengths.astype(np.int16))
            cols['mod'].append(np.array([mod_map.get(x, -1) if x is not None else -1
                                         for x in batch.column('modulation_label').to_pylist()], np.int16))
            cols['pro'].append(np.array([pro_map.get(x, -1) if x is not None else -1
                                         for x in batch.column('protocol_label').to_pylist()], np.int16))
            t = np.stack([batch.column(k).fill_null(0).to_numpy(zero_copy_only=False).astype(np.float64)
                          for k in TARGETS], 1)
            v = np.stack([batch.column(k + '_valid').fill_null(False).to_numpy(zero_copy_only=False).astype(bool)
                          for k in TARGETS], 1)
            targets.append(np.where(v, t, 0.0))
            valids.append(v)
            pos = np.empty(n, np.int64)
            rms = np.empty(n, np.float32)
            env = np.full(n, np.nan, np.float32)
            for L in LENGTHS:
                idx = np.flatnonzero(lengths == L)
                if not len(idx):
                    continue
                # rows of one length are contiguous in flat only per-row; gather explicitly
                x = np.stack([flat[off[i]:off[i + 1]] for i in idx]).reshape(len(idx), L, 2)
                fh[L].write(np.ascontiguousarray(x, dtype='<f4').tobytes())
                pos[idx] = count[L] + np.arange(len(idx))
                count[L] += len(idx)
                rms[idx] = np.sqrt((x.astype(np.float64) ** 2).sum(-1).mean(-1))
                pulse = v[idx, 3]
                if pulse.any():
                    env[idx[pulse]] = envelope_snr_db(x[pulse])
            cols['pos'].append(pos)
            cols['rms'].append(rms)
            cols['env_snr'].append(env)
            cols['source_file'].append(np.full(n, fi, np.int8))
            cols['source_row'].append(np.arange(row0, row0 + n, dtype=np.int64))
            row0 += n
        log(f'{path.name}: {row0} rows; cache counts {count}')
    for f in fh.values():
        f.close()

    meta = {k: (np.concatenate(v) if k not in ('sample_id', 'hash') else v) for k, v in cols.items()}
    meta['sample_id'] = np.asarray(meta['sample_id'])
    hashes = meta['hash']
    meta['hash'] = np.frombuffer(b''.join(hashes), dtype='V16')
    meta['targets'] = np.concatenate(targets)
    meta['valid'] = np.concatenate(valids)
    n = len(hashes)
    log(f'total rows {n}')

    # ---- split (leader rule): hash XOR salt(b'2026') < 0.1*2^64 -> validation; IO-sample hashes excluded
    split = np.fromiter((1 if hash_in_fraction(h, salt_val, 0.1) else 0 for h in hashes), np.int8, n)
    excluded = np.fromiter((h in sample_hashes for h in hashes), bool, n)
    split[excluded] = -1
    tune = np.fromiter((hash_in_fraction(h, salt_tune, 0.05) for h in hashes), bool, n) & (split == 0)
    meta['split'], meta['tune'] = split, tune
    id_overlap = int(np.isin(meta['sample_id'], list(sample_ids)).sum())

    # leakage check
    tr = set(h for h, s in zip(hashes, split) if s == 0)
    va = set(h for h, s in zip(hashes, split) if s == 1)
    leak = len(tr & va)
    rows = {'train': int((split == 0).sum()), 'validation': int((split == 1).sum()), 'excluded': int(excluded.sum())}
    log('split rows', rows, 'cross-split hash overlap', leak, 'IO sample_id found in train data', id_overlap)

    cross = None
    if args.cross_check_manifest:
        m = pq.read_table(args.cross_check_manifest, columns=['sample_id', 'iq_blake2b_128'])
        mh = dict(zip(m['sample_id'].to_pylist(), m['iq_blake2b_128'].to_pylist()))
        mism = sum(1 for sid, h in zip(meta['sample_id'].tolist(), hashes) if mh.get(sid) != h)
        cross = {'manifest': str(args.cross_check_manifest), 'rows': m.num_rows, 'hash_mismatches': mism}
        log('cross-check vs manifest:', cross)

    # ---- training-side statistics
    trn = split == 0
    T, V = meta['targets'], meta['valid']
    scaler = {'mean': [], 'std': []}
    for j, name in enumerate(TARGETS):
        vals = T[trn & V[:, j], j]
        scaler['mean'].append(float(vals.mean()))
        scaler['std'].append(float(vals.std()))
    comm = trn & V[:, 0]
    pulse = trn & V[:, 3]
    combos = sorted(set(map(tuple, T[comm][:, :2].tolist())))
    fs = T[comm, 2]
    fs_values = sorted(set(fs.tolist()))
    small = np.abs(fs) <= 50
    head = {
        'sr_bw_combos': [list(c) for c in combos],
        'fshift_values': fs_values,
        'fshift_classes': ['small', '100', '1000'],
        'fshift_small_mean': float(fs[small].mean()), 'fshift_small_std': float(fs[small].std()),
        'fshift_small_range': [float(fs[small].min()), float(fs[small].max())],
        'npulses_values': sorted(set(T[pulse, 6].tolist())),
        'time_mean': [float(T[pulse, j].mean()) for j in (3, 4, 5)],
        'time_std': [float(T[pulse, j].std()) for j in (3, 4, 5)],
        'time_range': [[float(T[pulse, j].min()), float(T[pulse, j].max())] for j in (3, 4, 5)],
    }
    env_q = np.nanpercentile(meta['env_snr'][pulse], [25, 50, 75]).tolist()
    validity_patterns = Counter(map(tuple, V[meta['task'] == 2].astype(int).tolist()))
    task_counts = {sname: {TASKS[t]: int(((split == s) & (meta['task'] == t)).sum()) for t in range(3)}
                   for sname, s in [('train', 0), ('validation', 1), ('excluded', -1)]}
    length_counts = {sname: {str(L): int(((split == s) & (meta['length'] == L)).sum()) for L in LENGTHS}
                     for sname, s in [('train', 0), ('validation', 1)]}
    info = {
        'format_version': 1, 'data_root': str(args.data_root), 'files': [p.name for p in files],
        'rows': n, 'split_rows': rows, 'expected_rows': EXPECTED,
        'split_matches_leader': rows['train'] == EXPECTED['train'] and rows['validation'] == EXPECTED['validation'],
        'cross_split_hash_overlap': leak, 'tune_rows': int(tune.sum()),
        'io_sample_hashes': len(sample_hashes), 'io_sample_ids_in_train_files': id_overlap,
        'cache_counts': {str(L): count[L] for L in LENGTHS}, 'task_counts': task_counts,
        'length_counts': length_counts,
        'validity_patterns_parameter': {''.join(map(str, k)): v for k, v in validity_patterns.items()},
        'schema': schema, 'target_scaler': scaler, 'head_stats': head,
        'env_snr_train_quartiles_db': env_q, 'cross_check': cross,
        'pyarrow': pa.__version__, 'numpy': np.__version__, 'elapsed_s': time.time() - t0,
    }
    np.savez(out / 'meta.npz', **meta)
    write_json(out / 'dataset.json', info)
    log('done', {k: info[k] for k in ['split_rows', 'split_matches_leader', 'cross_split_hash_overlap',
                                       'cache_counts', 'tune_rows']})
    if not info['split_matches_leader'] or leak:
        log('ERROR: split does not match the leader counts or leaks; do not train on this cache')
        sys.exit(2)


if __name__ == '__main__':
    main()
