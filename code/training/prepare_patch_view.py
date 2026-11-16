"""Create another patch coordinate contract over an unchanged prepared split.

IQ, labels and split files are read-only references to the existing cache.
Only dataset.json gets a new patch size/fingerprint; no waveform is resampled.
"""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils.common import read_json, write_json


def prepare_view(source, output, patch_size):
    source, output = Path(source).resolve(), Path(output).resolve()
    if type(patch_size) is not int or patch_size < 1 or 512 % patch_size:
        raise ValueError('Patch size must be a positive divisor of 512')
    info = read_json(source / 'dataset.json')
    with np.load(source / 'metadata.npz', allow_pickle=False) as metadata:
        lengths, split, hashes = metadata['length'], metadata['split'], metadata['hash']
        if len(lengths) != info['rows'] or np.any(lengths % patch_size):
            raise ValueError('Patch size/metadata row mismatch')
        if int((split == 0).sum()) != info['train_rows'] or int((split == 1).sum()) != info['validation_rows']:
            raise ValueError('Source split does not match dataset.json')
        fingerprint = hashlib.sha256(hashes.tobytes() + split.tobytes() + str(patch_size).encode()).hexdigest()
    revised = deepcopy(info)
    revised.update(patch_size=patch_size, fingerprint=fingerprint, source_cache=str(source),
                   source_cache_fingerprint=info['fingerprint'],
                   patch_view_note='Same IQ, row ordering, labels and split; only patch coordinates change')
    if output.exists():
        if read_json(output / 'dataset.json') != revised:
            raise ValueError('Existing output is different; choose a new view directory')
        for name in ['iq.npy', 'metadata.npz']:
            if (output / name).resolve() != (source / name).resolve():
                raise ValueError('Existing view does not reference the expected source: ' + name)
        return revised
    if output == source:
        raise ValueError('Keep the source cache unchanged')
    output.mkdir(parents=True, exist_ok=False)
    for name in ['iq.npy', 'metadata.npz', 'split_manifest.parquet']:
        original = source / name
        if original.exists():
            (output / name).symlink_to(original)
    write_json(output / 'dataset.json', revised)
    return revised


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--patch-size', type=int, required=True)
    args = parser.parse_args()
    info = prepare_view(args.source, args.output, args.patch_size)
    print({key: info[key] for key in ['patch_size', 'fingerprint', 'rows', 'train_rows', 'validation_rows']}, flush=True)


if __name__ == '__main__':
    main()
