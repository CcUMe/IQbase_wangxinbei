"""Shared configuration, deterministic runtime and tensor-only checkpoint IO."""
import json
import os
import random
import struct
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

TARGETS = ['symbol_rate', 'bandwidth', 'frequency_shift', 'pulse_width', 'pri', 'time_delay', 'number_of_pulses']
TASKS = ['modulation', 'protocol', 'parameter']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_config_json(path):
    """JSON configuration with // and /* */ comments; preserve strings and error lines."""
    text = Path(path).read_text(encoding='utf-8-sig')
    chars = list(text)
    quoted, escaped, i = False, False, 0
    while i < len(text):
        if quoted:
            if escaped:
                escaped = False
            elif text[i] == '\\':
                escaped = True
            elif text[i] == '"':
                quoted = False
            i += 1
        elif text[i] == '"':
            quoted = True
            i += 1
        elif text.startswith('//', i):
            end = text.find('\n', i)
            end = len(text) if end < 0 else end
            chars[i:end] = [' ' if c not in '\r\n' else c for c in text[i:end]]
            i = end
        elif text.startswith('/*', i):
            end = text.find('*/', i + 2)
            if end < 0:
                raise ValueError(f'Unterminated block comment in {path}')
            end += 2
            chars[i:end] = [' ' if c not in '\r\n' else c for c in text[i:end]]
            i = end
        else:
            i += 1
    return json.loads(''.join(chars))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runtime(device='auto', threads=4):
    torch.set_num_threads(threads)
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda'):
        if world > 2:
            raise ValueError('This host is configured for two-GPU communication groups only')
        if world == 2:
            visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1').split(',')[:2]
            if len(visible) != 2 or frozenset(visible) not in [frozenset(('0', '1')), frozenset(('2', '3')), frozenset(('4', '5'))]:
                raise ValueError('This host supports GPU pairs 0,1 or 2,3 or 4,5. Set CUDA_VISIBLE_DEVICES to one supported pair.')
            # Verified on this host: NCCL cuMem host allocation crashes even for
            # a valid pair; its standard /dev/shm fallback works. Preserve P2P.
            # Scope is this process; an explicit user environment value wins.
            os.environ.setdefault('NCCL_CUMEM_HOST_ENABLE', '0')
        if device == 'cuda':
            device = 'cuda:0'
        if world > 1:
            device = f'cuda:{local_rank}'
        torch.cuda.set_device(torch.device(device))
    if world > 1 and not dist.is_initialized():
        if device.startswith('cuda'):
            dist.init_process_group('nccl', device_id=torch.device(device))
        else:
            dist.init_process_group('gloo')
    return torch.device(device), rank, world


def load_checkpoint(path, device='cpu'):
    # Checkpoints produced by this project contain tensors and primitive values only.
    return torch.load(path, map_location=device, weights_only=True)


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def save_safetensors(path, state):
    """Write the standard safetensors format; no third-party dependency required.

    Export uses float32 model weights. Non-floating buffers retain integer/bool types.
    """
    mapping = {np.dtype('float32'): 'F32', np.dtype('float64'): 'F64',
               np.dtype('int64'): 'I64', np.dtype('int32'): 'I32', np.dtype('bool'): 'BOOL'}
    header, arrays, offset = {}, [], 0
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        if value.is_floating_point():
            value = value.float()
        arr = value.numpy()
        arr = arr.astype(arr.dtype.newbyteorder('<'), copy=False)
        if arr.dtype not in mapping:
            raise ValueError(f'Unsupported export dtype {arr.dtype}: {name}')
        header[name] = {'dtype': mapping[arr.dtype], 'shape': list(arr.shape),
                        'data_offsets': [offset, offset + arr.nbytes]}
        arrays.append(arr)
        offset += arr.nbytes
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * ((-len(raw)) % 8)
    path = Path(path)
    with path.open('wb') as file:
        file.write(struct.pack('<Q', len(raw)))
        file.write(raw)
        for arr in arrays:
            file.write(arr.tobytes())


def load_safetensors(path):
    mapping = {'F32': '<f4', 'F64': '<f8', 'I64': '<i8', 'I32': '<i4', 'BOOL': '?'}
    raw = Path(path).read_bytes()
    if len(raw) < 8:
        raise ValueError('Invalid safetensors file')
    size = struct.unpack('<Q', raw[:8])[0]
    if size > 100_000_000 or size + 8 > len(raw):
        raise ValueError('Invalid safetensors header size')
    header = json.loads(raw[8:8 + size])
    payload = memoryview(raw)[8 + size:]
    state, expected_offset = {}, 0
    for name, spec in header.items():
        if name == '__metadata__':
            continue
        dtype = np.dtype(mapping[spec['dtype']])
        lo, hi = spec['data_offsets']
        shape = spec['shape']
        if any(not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError('Invalid tensor shape')
        elements = 1
        for n in shape:
            elements *= n
        if lo != expected_offset or hi > len(payload) or hi - lo != elements * dtype.itemsize:
            raise ValueError(f'Invalid tensor offsets: {name}')
        state[name] = torch.from_numpy(np.frombuffer(payload[lo:hi], dtype=dtype).copy().reshape(shape))
        expected_offset = hi
    if expected_offset != len(payload):
        raise ValueError('Unexpected bytes after tensor payload')
    return state


def model_bundle(checkpoint):
    return {key: checkpoint[key] for key in ['model_config', 'schema', 'target_scaler', 'parameter_grids']}


def log_json(path, record, console=True):
    line = json.dumps(record, ensure_ascii=False, allow_nan=False)
    if console:
        print(line, flush=True)
    with Path(path).open('a', encoding='utf-8') as file:
        file.write(line + '\n')
