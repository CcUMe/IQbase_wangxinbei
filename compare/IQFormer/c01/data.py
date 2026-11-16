"""Existing C01 hash split, modulation rows only, unpadded length buckets."""
import math
from pathlib import Path

import numpy as np
import torch
from scipy.signal import stft
from torch.utils.data import Dataset, Sampler

from utils.common import read_json


class C01Modulation(Dataset):
    def __init__(self, directory, split):
        self.directory = Path(directory)
        self.info = read_json(self.directory / 'dataset.json')
        self.labels = self.info['schema']['classification_labels']['modulation']
        with np.load(self.directory / 'metadata.npz', allow_pickle=False) as meta:
            target = meta['modulation']
            self.rows = np.flatnonzero((meta['split'] == {'train': 0, 'validation': 1}[split]) & (target >= 0))
            self.targets = target[self.rows].astype(np.int64)
            self.lengths = meta['length'][self.rows].astype(np.int64)
            self.sample_ids = meta['sample_id'][self.rows]
        if not len(self.rows) or np.any((self.targets < 0) | (self.targets >= len(self.labels))):
            raise ValueError(f'Invalid/empty modulation {split} split')
        if not np.isin(self.lengths, [512, 1024]).all():
            raise ValueError('Expected native C01 lengths 512 or 1024')
        self.iq = None

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        return {**self.__dict__, 'iq': None}

    def __getitem__(self, index):
        if self.iq is None:
            self.iq = np.load(self.directory / 'iq.npy', mmap_mode='r')
        row, length = int(self.rows[index]), int(self.lengths[index])
        return {'iq': np.array(self.iq[row, :length], dtype=np.float32, copy=True),
                'target': int(self.targets[index]), 'row': row,
                'sample_id': str(self.sample_ids[index])}


class LengthBatches(Sampler):
    """Every row once per repeat, shuffled within length then across batches.

    No replacement sampling, discarded tail, cross-length padding, or hidden
    task/length information passed to the classifier.
    """
    def __init__(self, lengths, batch_size, shuffle, seed, repeats=1):
        self.buckets = [np.flatnonzero(lengths == n) for n in sorted(np.unique(lengths))]
        self.batch_size, self.shuffle, self.seed = batch_size, shuffle, seed
        self.repeats, self.epoch = repeats, 0

    def __len__(self):
        return self.repeats * sum(math.ceil(len(b) / self.batch_size) for b in self.buckets)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for _ in range(self.repeats):
            for bucket in self.buckets:
                indices = rng.permutation(bucket) if self.shuffle else bucket
                batches.extend(indices[i:i + self.batch_size].tolist()
                               for i in range(0, len(indices), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


class SignalViews:
    def __init__(self, settings, training=False):
        self.settings, self.training = settings, training

    def __call__(self, batch):
        iq = np.stack([row['iq'] for row in batch])
        if not np.isfinite(iq).all():
            raise ValueError('Nonfinite I/Q input')
        if self.settings['normalization'] == 'rms':
            scale = np.sqrt(np.square(iq).sum(-1).mean(-1)).clip(self.settings['rms_epsilon'])
            iq = iq / scale[:, None, None]
        # Phase augmentation, when explicitly requested, affects BOTH modalities.
        probability = self.settings['phase_rotation_probability'] if self.training else 0.
        if probability:
            angles = np.random.uniform(-np.pi, np.pi, len(iq))
            angles *= np.random.random(len(iq)) < probability
            z = (iq[..., 0] + 1j * iq[..., 1]) * np.exp(1j * angles[:, None])
            iq = np.stack((z.real, z.imag), -1).astype(np.float32)
        # Exact upstream dataset.py view: stft(I, 1, 'blackman', 31, 30, 128).
        # torch.Tensor(complex_array) discards imaginary values in upstream.
        # Explicit .real reproduces that conversion; do NOT use abs/log-power.
        _, _, spectrum = stft(iq[..., 0], fs=1., window='blackman',
                              nperseg=self.settings['stft_window'],
                              noverlap=self.settings['stft_window'] - 1,
                              nfft=128, detrend=False, return_onesided=True,
                              boundary='zeros', padded=True, scaling='spectrum', axis=-1)
        if spectrum.shape[-1] != iq.shape[1]:
            raise ValueError('STFT time length does not match I/Q')
        result = {'iq': torch.from_numpy(np.ascontiguousarray(iq.transpose(0, 2, 1))),
                  'stft': torch.from_numpy(np.ascontiguousarray(spectrum[:, None, :32].real, dtype=np.float32)),
                  'rows': torch.tensor([row.get('row', -1) for row in batch]),
                  'sample_ids': [row['sample_id'] for row in batch]}
        if all('target' in row for row in batch):
            result['target'] = torch.tensor([row['target'] for row in batch], dtype=torch.long)
        return result
