"""Memory-mapped data, homogeneous-length task-balanced training batches."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from utils.common import read_json


class PreparedDataset(Dataset):
    def __init__(self, directory, split='train', codebook_dir=None, merge_paired=False):
        self.directory = Path(directory)
        self.info = read_json(self.directory / 'dataset.json')
        with np.load(self.directory / 'metadata.npz', allow_pickle=False) as f:
            self.meta = {key: f[key] for key in f.files}
        self.indices = np.flatnonzero(self.meta['split'] == (0 if split == 'train' else 1))
        self.iq = None
        self.codebook_dir = Path(codebook_dir) if codebook_dir else None
        self.atomic = self.emergent = None
        if self.codebook_dir:
            config = read_json(self.codebook_dir / 'codebook.json')
            if config['dataset_fingerprint'] != self.info['fingerprint']:
                raise ValueError('Codebook labels belong to a different prepared dataset')
            if config['patch_size'] != self.info['patch_size']:
                raise ValueError('Codebook/data patch size mismatch')
        if merge_paired:
            if split != 'train':
                raise ValueError('Pair-label merging is only allowed on the training split')
            self._merge_paired()

    def _merge_paired(self):
        first, keep = {}, []
        for idx in self.indices:
            key = bytes(self.meta['hash'][idx])
            if key not in first:
                first[key] = idx
                keep.append(idx)
                continue
            previous = first[key]
            for task in ['modulation', 'protocol']:
                label = self.meta[task][idx]
                if label >= 0:
                    old = self.meta[task][previous]
                    if old >= 0 and old != label:
                        raise ValueError('Conflicting paired classification labels')
                    self.meta[task][previous] = label
            valid = self.meta['target_valid'][idx]
            overlap = valid & self.meta['target_valid'][previous]
            if not np.array_equal(self.meta['targets'][idx, overlap], self.meta['targets'][previous, overlap]):
                raise ValueError('Conflicting paired parameter labels')
            self.meta['targets'][previous, valid] = self.meta['targets'][idx, valid]
            self.meta['target_valid'][previous] |= valid
        self.indices = np.asarray(keep, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['iq'] = state['atomic'] = state['emergent'] = None
        return state

    def __getitem__(self, index):
        row = self.indices[index]
        if self.iq is None:
            self.iq = np.load(self.directory / 'iq.npy', mmap_mode='r')
        length = int(self.meta['length'][row])
        result = {'iq': np.array(self.iq[row, :length], copy=True), 'length': length,
                  'modulation': self.meta['modulation'][row], 'protocol': self.meta['protocol'][row],
                  'targets': self.meta['targets'][row], 'target_valid': self.meta['target_valid'][row],
                  'sample_id': str(self.meta['sample_id'][row]), 'row': int(row)}
        if self.codebook_dir:
            if self.atomic is None:
                self.atomic = np.load(self.codebook_dir / 'atomic_labels.npy', mmap_mode='r')
                self.emergent = np.load(self.codebook_dir / 'emergent_labels.npy', mmap_mode='r')
            tokens = length // self.info['patch_size']
            result['atomic'] = np.array(self.atomic[row, :tokens], dtype=np.int64)
            result['emergent'] = np.array(self.emergent[row, :tokens], dtype=np.int64)
        return result


def collate(batch):
    maximum = max(row['length'] for row in batch)
    iq = np.zeros((len(batch), maximum, 2), dtype=np.float32)
    for i, row in enumerate(batch):
        iq[i, :row['length']] = row['iq']
    output = {'iq': torch.from_numpy(iq), 'lengths': torch.tensor([r['length'] for r in batch]),
              'sample_ids': [r['sample_id'] for r in batch], 'rows': [r['row'] for r in batch]}
    for name in ['modulation', 'protocol', 'targets', 'target_valid']:
        output[name] = torch.from_numpy(np.stack([r[name] for r in batch]))
    if 'atomic' in batch[0]:
        for name in ['atomic', 'emergent']:
            size = max(len(r[name]) for r in batch)
            arr = np.full((len(batch), size), -100, dtype=np.int64)
            for i, row in enumerate(batch):
                arr[i, :len(row[name])] = row[name]
            output[name] = torch.from_numpy(arr)
    return output


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, steps, seed=2026, rank=0, mode='balanced', task_probabilities=None):
        self.batch_size, self.steps, self.seed, self.rank = batch_size, steps, seed, rank
        self.epoch = 0
        self.mode = mode
        self.task_probabilities = task_probabilities or {'modulation': 1., 'protocol': 1., 'parameter': 1.}
        if mode not in ['balanced', 'weighted', 'proportional']:
            raise ValueError('Unknown task sampling mode')
        rows = dataset.indices
        self.groups = {}
        membership = [dataset.meta['modulation'][rows] >= 0,
                      dataset.meta['protocol'][rows] >= 0,
                      dataset.meta['target_valid'][rows].any(1)]
        for task, eligible in enumerate(membership):
            by_length = {}
            for length in np.unique(dataset.meta['length'][rows]):
                choices = np.flatnonzero(eligible & (dataset.meta['length'][rows] == length))
                if len(choices):
                    by_length[int(length)] = choices
            if by_length:
                self.groups[task] = by_length

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.steps

    def __iter__(self):
        schedule = np.random.default_rng(self.seed + self.epoch)
        draws = np.random.default_rng(self.seed + self.epoch * 100003 + self.rank * 1009)
        tasks = list(self.groups)
        names = ['modulation', 'protocol', 'parameter']
        weights = np.asarray([sum(len(v) for v in self.groups[t].values()) if self.mode == 'proportional'
                              else self.task_probabilities[names[t]] for t in tasks], dtype=float)
        weights /= weights.sum()
        for step in range(self.steps):
            # All ranks select the same task/length but draw different samples.
            task = tasks[(self.epoch * self.steps + step) % len(tasks)] if self.mode == 'balanced' else schedule.choice(tasks, p=weights)
            buckets = self.groups[task]
            lengths = list(buckets)
            counts = np.asarray([len(buckets[n]) for n in lengths], dtype=float)
            length = schedule.choice(lengths, p=counts / counts.sum())
            yield draws.choice(buckets[length], self.batch_size, replace=True).tolist()


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}
