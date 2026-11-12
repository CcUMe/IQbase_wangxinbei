"""Create a hash-grouped split and mmap cache, without changing source data."""
import argparse
from collections import Counter
import hashlib
import struct
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from utils.common import TARGETS, TASKS, read_json, write_json
from utils.command_config import command_defaults


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Optional grouped preparation settings; CLI values override')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('artifacts/c01_data_audit/sample_manifest.parquet'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--limit-per-task', type=int, default=0, help='Per task per split, with group closure; 0 = full data')
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--exclude-io-samples', action=argparse.BooleanOptionalAction, default=True)
    command_defaults(parser, {'run': {'data_root': 'data_root', 'manifest': 'manifest', 'output': 'output'},
                              'split': {'seed': 'seed', 'validation_fraction': 'val_fraction', 'exclude_io_samples': 'exclude_io_samples'},
                              'cache': {'patch_size': 'patch_size', 'limit_per_task': 'limit_per_task'}})
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1 or args.limit_per_task < 0 or args.patch_size < 1 or 512 % args.patch_size:
        parser.error('Invalid split fraction or sample limit')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output is not empty; choose a new directory to preserve previous caches')
    if args.output.resolve() == args.data_root.resolve() or args.data_root.resolve() in args.output.resolve().parents:
        parser.error('Cache must be outside the source dataset')
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.manifest.exists():
        parser.error('Run code/utils/audit_c01_dataset.py first to create the full content-hash manifest')
    schema = read_json(args.data_root / 'rf_signal_benchmark_train_v1.0/task_schema.json')
    table = pq.read_table(args.manifest)
    ids = table['sample_id'].to_pylist()
    hashes = table['iq_blake2b_128'].to_pylist()
    sample = pq.read_table(args.data_root / 'rf_signal_sample_io_v1.0/rf_signal_sample_io_input_v1.0.parquet', columns=['sample_id'])
    sample_ids = set(sample['sample_id'].to_pylist())
    excluded_hashes = {h for sid, h in zip(ids, hashes) if sid in sample_ids} if args.exclude_io_samples else set()
    # XOR with a fixed seed-derived salt permutes the 64-bit hash space, preserving
    # uniform group probabilities without using IDs, filenames or target values.
    salt = int.from_bytes(hashlib.blake2b(str(args.seed).encode(), digest_size=8).digest(), 'little')
    threshold = int(args.val_fraction * 2**64)
    split = np.fromiter((1 if (int.from_bytes(h[:8], 'little') ^ salt) < threshold else 0 for h in hashes), np.int8)
    excluded = np.fromiter((h in excluded_hashes for h in hashes), bool)
    split[excluded] = -1
    if args.limit_per_task:
        rng = np.random.default_rng(args.seed)
        tasks = np.asarray(table['task'].to_pylist())
        selected_hashes = set()
        for part in [0, 1]:
            for task in TASKS:
                candidates = np.flatnonzero((split == part) & (tasks == task))
                # Round-robin class/validity strata keep tiny smoke subsets useful.
                label_col = table[f'{task}_label'].to_pylist() if task != 'parameter' else None
                if label_col is not None:
                    strata = np.asarray([str(label_col[i]) for i in candidates])
                else:
                    strata = np.asarray([str(table['iq_length'][int(i)].as_py()) for i in candidates])
                buckets = [list(rng.permutation(candidates[strata == label])) for label in np.unique(strata)]
                picked = []
                while len(picked) < args.limit_per_task and any(buckets):
                    for bucket in buckets:
                        if bucket and len(picked) < args.limit_per_task:
                            picked.append(bucket.pop())
                selected_hashes.update(hashes[i] for i in picked)
        chosen = np.fromiter((h in selected_hashes for h in hashes), bool)
        table = table.filter(pa.array(chosen))
        split = split[chosen]
        hashes = [h for h, yes in zip(hashes, chosen) if yes]
    else:
        chosen = split >= 0
        table = table.filter(pa.array(chosen))
        split = split[chosen]
        hashes = [h for h, yes in zip(hashes, chosen) if yes]
    n = table.num_rows
    if not n or not (split == 0).any() or not (split == 1).any():
        raise ValueError('Both train and validation need samples')
    train_hashes = {h for h, part in zip(hashes, split) if part == 0}
    val_hashes = {h for h, part in zip(hashes, split) if part == 1}
    if train_hashes & val_hashes:
        raise AssertionError('Group leakage')
    ids = table['sample_id'].to_pylist()
    metadata = {'sample_id': np.asarray(ids), 'hash': np.asarray(hashes, dtype='V16'), 'split': split,
                'length': table['iq_length'].to_numpy().astype(np.int32),
                'task': np.asarray([TASKS.index(t) for t in table['task'].to_pylist()], dtype=np.int8)}
    for task in ['modulation', 'protocol']:
        mapping = {name: idx for idx, name in enumerate(schema['classification_labels'][task])}
        metadata[task] = np.asarray([mapping.get(x, -1) for x in table[f'{task}_label'].to_pylist()], dtype=np.int64)
    targets, validity, means, stds, grids = [], [], [], [], []
    for name in TARGETS:
        valid = table[name + '_valid'].to_numpy().astype(bool)
        raw = table[name].to_numpy().astype(np.float32)
        if not np.isfinite(raw[valid]).all():
            raise ValueError(f'Nonfinite valid target: {name}')
        values = raw[(split == 0) & valid]
        if len(values) < 2:
            raise ValueError(f'Need at least two training labels for {name}; increase sample limit')
        mean, std = float(values.mean(dtype=np.float64)), float(values.std(dtype=np.float64))
        means.append(mean)
        stds.append(max(std, abs(mean) * 1e-6, 1e-12))
        unique = np.unique(values)
        grids.append(unique.tolist() if name in ['symbol_rate', 'bandwidth', 'frequency_shift', 'number_of_pulses'] and len(unique) <= 64 else [])
        targets.append(np.where(valid, raw, 0))
        validity.append(valid)
    metadata['targets'] = np.stack(targets, 1)
    metadata['target_valid'] = np.stack(validity, 1)
    if (metadata['length'] % args.patch_size).any():
        raise ValueError('Patch size must divide all sequence lengths')
    cache = np.lib.format.open_memmap(args.output / 'iq.npy', mode='w+', dtype=np.float32, shape=(n, 1024, 2))
    cache[:] = 0
    files = table['source_file'].to_pylist()
    source_rows = table['source_row'].to_numpy()
    for filename in sorted(set(files)):
        requested = np.flatnonzero(np.asarray(files) == filename)
        pf = pq.ParquetFile(args.data_root / 'rf_signal_benchmark_train_v1.0' / filename)
        boundaries = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
        by_group = np.searchsorted(boundaries[1:], source_rows[requested], side='right')
        for group in np.unique(by_group):
            positions = requested[by_group == group]
            column = pf.read_row_group(int(group), columns=['iq'])['iq'].combine_chunks()
            offsets = column.offsets.to_numpy()
            inner = column.values
            inner_offsets = inner.offsets.to_numpy()
            flat = inner.values.to_numpy()
            for dest in positions:
                row = int(source_rows[dest] - boundaries[group])
                lo, hi = int(inner_offsets[offsets[row]]), int(inner_offsets[offsets[row + 1]])
                length = int(metadata['length'][dest])
                array = flat[lo:hi].reshape(-1, 2)
                if array.shape != (length, 2) or not np.isfinite(array).all():
                    raise ValueError(f'Bad IQ row: {filename}:{source_rows[dest]}')
                digest = hashlib.blake2b(struct.pack('<I', length) + array.astype('<f4', copy=False).tobytes(), digest_size=16).digest()
                if digest != hashes[dest]:
                    raise ValueError('Source IQ differs from audit manifest; rerun the audit for this data version')
                cache[dest, :length] = array
        print(f'Cached {filename}: {len(requested)} rows', flush=True)
    cache.flush()
    np.savez_compressed(args.output / 'metadata.npz', **metadata)
    fingerprint = hashlib.sha256(b''.join(hashes) + split.tobytes() + str(args.patch_size).encode()).hexdigest()
    info = {'format_version': 1, 'fingerprint': fingerprint, 'data_root': str(args.data_root.resolve()),
            'manifest': str(args.manifest.resolve()), 'rows': n, 'train_rows': int((split == 0).sum()),
            'validation_rows': int((split == 1).sum()), 'seed': args.seed, 'val_fraction': args.val_fraction,
            'patch_size': args.patch_size, 'excluded_sample_hashes': len(excluded_hashes),
            'exclude_io_samples': args.exclude_io_samples,
            'cross_split_hash_overlap': 0, 'schema': schema,
            'target_scaler': {'mean': means, 'std': stds}, 'parameter_grids': grids,
            'task_counts': {name: dict(Counter(table['task'].filter(pa.array(split == part)).to_pylist()))
                            for name, part in [('train', 0), ('validation', 1)]}}
    write_json(args.output / 'dataset.json', info)
    pq.write_table(table.append_column('split', pa.array(split)), args.output / 'split_manifest.parquet', compression='zstd')
    print({key: info[key] for key in ['rows', 'train_rows', 'validation_rows', 'task_counts', 'cross_split_hash_overlap']}, flush=True)


if __name__ == '__main__':
    main()
