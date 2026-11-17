"""Shared constants and small helpers for TF-UniEM."""
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pyarrow.compute as pc

TASKS = ['modulation', 'protocol', 'parameter']
TARGETS = ['symbol_rate', 'bandwidth', 'frequency_shift', 'pulse_width', 'pri', 'time_delay', 'number_of_pulses']
COMM_TARGETS = TARGETS[:3]
PULSE_TARGETS = TARGETS[3:]
LENGTHS = [512, 1024]

# 14 narrowband real-capture modulation classes (guide 2.4); the other 9 are simulated.
NARROWBAND_MODS = ['OOK', 'BPSK', 'QPSK', '8PSK', 'OQPSK', 'AQPSK', '16QAM', '32QAM', '64QAM', '128QAM',
                   '256QAM', '2FSK', '4FSK', 'MSK']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def waveform_hash(length, flat_f32):
    """Content hash identical to the leader's: blake2b(<I length> + iq[L,2] little-endian f4 bytes)."""
    return hashlib.blake2b(struct.pack('<I', length) + flat_f32.astype('<f4', copy=False).tobytes(),
                           digest_size=16).digest()


def split_salt(seed_bytes):
    return int.from_bytes(hashlib.blake2b(seed_bytes, digest_size=8).digest(), 'little')


def hash_in_fraction(digest, salt, fraction):
    return (int.from_bytes(digest[:8], 'little') ^ salt) < int(fraction * 2 ** 64)


def iq_column_to_flat(col):
    """Accept list<list<float>> and list<fixed_size_list<float>[2]>. Returns (lengths int64, flat float32, offsets)."""
    if hasattr(col, 'combine_chunks'):
        col = col.combine_chunks()
    lengths = pc.list_value_length(col).to_numpy(zero_copy_only=False).astype(np.int64)
    flat = col.flatten().flatten().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    if flat.size != 2 * lengths.sum():
        raise ValueError(f'IQ flatten size mismatch: {flat.size} vs 2*{lengths.sum()}')
    offsets = np.concatenate([[0], np.cumsum(2 * lengths)])
    return lengths, flat, offsets


def envelope_snr_db(x):
    """x: (N, L, 2) float. 10*log10(p90/p10) of the 3-point smoothed power envelope (same as eda2_c01.py)."""
    p = x[..., 0].astype(np.float64) ** 2 + x[..., 1].astype(np.float64) ** 2
    pad = np.pad(p, ((0, 0), (1, 1)))
    sm = (pad[:, :-2] + pad[:, 1:-1] + pad[:, 2:]) / 3.0  # np.convolve(..., 'same') with zero padding
    lo, hi = np.percentile(sm, [10, 90], axis=1)
    return 10 * np.log10(hi / np.maximum(lo, 1e-30))
