"""Shared configuration, checkpoint loading and full classification evaluation."""
import copy
import csv
import hashlib
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.common import read_config_json, read_json, write_json
from utils.metrics import classification_metrics
from .data import C01Modulation, LengthBatches, SignalViews
from .official_model import IQFormer

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]


def path_from_root(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path):
    config = read_config_json(path)
    defaults = read_config_json(BASE / 'c01.json5')
    if set(config) != set(defaults):
        raise ValueError('Config sections must match c01.json5')
    for section in defaults:
        if set(config[section]) != set(defaults[section]):
            raise ValueError(f'Unknown/missing {section} option; use c01.json5 as template')
    for section, keys, minimum in [
        ('data', ['batch_size', 'eval_batch_size', 'repeats_per_epoch', 'prefetch_factor'], 1),
        ('data', ['num_workers'], 0),
        ('training', ['epochs'], 1),
        ('training', ['early_stopping_patience', 'max_train_batches', 'max_val_batches'], 0),
        ('logging', ['log_every_steps'], 1),
        ('runtime', ['threads'], 1),
        ('scheduler', ['patience', 'warmup_epochs'], 0),
    ]:
        for key in keys:
            if type(config[section][key]) is not int or config[section][key] < minimum:
                raise ValueError(f'Invalid {section}.{key}')
    for section, keys in [('optimizer', ['learning_rate', 'eps']), ('input', ['rms_epsilon'])]:
        for key in keys:
            v = config[section][key]
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f'Invalid {section}.{key}')
    for section, keys in [('training', ['gradient_clip', 'early_stopping_min_delta']),
                          ('optimizer', ['weight_decay']), ('scheduler', ['min_lr'])]:
        for key in keys:
            if not math.isfinite(config[section][key]) or config[section][key] < 0:
                raise ValueError(f'Invalid {section}.{key}')
    for section, key in [('input', 'phase_rotation_probability'), ('training', 'label_smoothing'),
                          ('model', 'drop_rate'), ('model', 'drop_path_rate')]:
        if not 0 <= config[section][key] < 1:
            raise ValueError(f'Invalid {section}.{key}')
    model = config['model']
    if (len(model['layers']) != len(model['embed_dims']) or not model['layers']
            or any(type(x) is not int or x < 1 for x in model['layers'])
            or sum(model['layers']) <= 1
            or any(type(x) is not int or x < 16 or x % 16 for x in model['embed_dims'])):
        raise ValueError('Invalid model stage dimensions/depths')
    if config['input']['normalization'] not in ['none', 'rms']:
        raise ValueError('input.normalization must be none or rms')
    if (config['input']['phase_rotation_probability'] and config['data']['num_workers']
            and config['data']['persistent_workers']):
        raise ValueError('Phase augmentation requires persistent_workers=false for reproducible epoch resumes')
    if config['input']['stft_window'] not in [31, 61]:
        raise ValueError('Supported odd Blackman windows are 31 (code) and 61 (paper)')
    if config['runtime']['precision'] not in ['fp32', 'bf16']:
        raise ValueError('Supported precision: fp32 or bf16')
    scheduler = config['scheduler']
    if scheduler['name'] not in ['plateau', 'cosine'] or not 0 < scheduler['factor'] < 1:
        raise ValueError('Invalid scheduler')
    if not 0 <= scheduler['min_lr'] <= config['optimizer']['learning_rate']:
        raise ValueError('min_lr must be between zero and peak LR')
    if scheduler['warmup_epochs'] >= config['training']['epochs']:
        raise ValueError('warmup_epochs must be smaller than total epochs')
    if scheduler['name'] == 'plateau' and scheduler['warmup_epochs']:
        raise ValueError('Official plateau recipe has no warmup')
    if len(config['optimizer']['betas']) != 2 or not all(0 <= b < 1 for b in config['optimizer']['betas']):
        raise ValueError('Invalid AdamW betas')
    config['run'] = {key: str(path_from_root(value)) for key, value in config['run'].items()}
    return config


def setup_runtime(config, device_override=None):
    if int(__import__('os').environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This comparison runner is single-device; do not use torchrun')
    settings = config['runtime']
    torch.set_num_threads(settings['threads'])
    random.seed(settings['seed'])
    np.random.seed(settings['seed'])
    torch.manual_seed(settings['seed'])
    name = device_override or settings['device']
    name = ('cuda:0' if torch.cuda.is_available() else 'cpu') if name == 'auto' else name
    device = torch.device(name)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(settings['seed'])
        if settings['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('This GPU does not support BF16')
    elif settings['precision'] != 'fp32':
        raise ValueError('Use fp32 for CPU checks')
    torch.backends.cuda.matmul.allow_tf32 = settings['allow_tf32']
    torch.backends.cudnn.allow_tf32 = settings['allow_tf32']
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = settings['deterministic']
    torch.use_deterministic_algorithms(settings['deterministic'])
    return device


def autocast(config, device):
    return (torch.autocast(device.type, dtype=torch.bfloat16)
            if config['runtime']['precision'] == 'bf16' else nullcontext())


def source_identity():
    result = read_json(BASE / 'source_info.json')
    result['runtime_model_sha256'] = hashlib.sha256((BASE / 'c01/official_model.py').read_bytes()).hexdigest()
    result['runtime_compat_sha256'] = hashlib.sha256((BASE / 'c01/compat.py').read_bytes()).hexdigest()
    return result


def build_model(config, labels):
    return IQFormer(**config['model'], num_classes=len(labels), fork_feat=False)


def make_loader(dataset, config, training):
    settings = config['data']
    sampler = LengthBatches(dataset.lengths,
                            settings['batch_size' if training else 'eval_batch_size'],
                            training, config['runtime']['seed'],
                            settings['repeats_per_epoch'] if training else 1)
    options = dict(batch_sampler=sampler, collate_fn=SignalViews(config['input'], training),
                   num_workers=settings['num_workers'], pin_memory=settings['pin_memory'],
                   generator=torch.Generator().manual_seed(config['runtime']['seed']))
    if settings['num_workers']:
        options.update(persistent_workers=settings['persistent_workers'],
                       prefetch_factor=settings['prefetch_factor'])
    return DataLoader(dataset, **options), sampler


def group_metrics(metrics):
    cm = np.asarray(metrics['confusion_matrix'])
    result = {}
    for name, indices in [('first14', range(min(14, len(cm)))), ('remaining9', range(14, len(cm)))]:
        indices = list(indices)
        support = int(cm[indices].sum())
        if support:
            result[name] = {'support': support, 'accuracy': float(cm.diagonal()[indices].sum() / support),
                            'macro_f1': float(np.asarray(metrics['f1'])[indices].mean())}
    return result


@torch.inference_mode()
def evaluate(model, loader, config, device, max_batches=0, save_predictions=False):
    model.eval()
    size = len(loader.dataset.labels)
    cm = np.zeros((size, size), dtype=np.int64)
    loss_sum, confidence_sum, count = 0., 0., 0
    collected = {key: [] for key in ['rows', 'labels', 'logits']}
    progress = tqdm(loader, total=min(len(loader), max_batches or len(loader)),
                    desc='Validation', disable=not config['logging']['progress_bar'], dynamic_ncols=True)
    for index, batch in enumerate(progress):
        if max_batches and index >= max_batches:
            break
        iq, spectrum, y = [batch[k].to(device, non_blocking=True) for k in ['iq', 'stft', 'target']]
        with autocast(config, device):
            logits = model(iq, spectrum).float()
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('Nonfinite validation logits')
        loss_sum += float(F.cross_entropy(logits, y, reduction='sum'))
        confidence_sum += float(logits.softmax(-1).max(-1).values.sum())
        prediction = logits.argmax(-1)
        cm += torch.bincount(y * size + prediction, minlength=size * size).reshape(size, size).cpu().numpy()
        count += len(y)
        progress.set_postfix(ce=f'{loss_sum/count:.4f}', acc=f'{cm.trace()/count:.2%}')
        if save_predictions:
            collected['rows'].append(batch['rows'].numpy())
            collected['labels'].append(y.cpu().numpy())
            collected['logits'].append(logits.cpu().numpy())
    if not count:
        raise ValueError('No evaluation samples')
    metrics = classification_metrics(cm, loader.dataset.labels)
    metrics.update(cross_entropy=loss_sum / count, mean_confidence=confidence_sum / count)
    result = {'classification': {'modulation': metrics}, 'groups': group_metrics(metrics),
              'rows': count, 'complete': count == len(loader.dataset)}
    return result, {key: np.concatenate(value) for key, value in collected.items()} if save_predictions else None


def print_metrics(metrics):
    m = metrics['classification']['modulation']
    print(f"验证调制 | Acc={m['accuracy']:.4%} | Micro-F1={m['micro_f1']:.4%} | "
          f"Macro-F1={m['macro_f1']:.4%} | Weighted-F1={m['weighted_f1']:.4%} | "
          f"CE={m['cross_entropy']:.4f} | n={m['support']:,}", flush=True)
    print('条件分组 | ' + ' | '.join(f"{k}: Acc={v['accuracy']:.4%}, F1={v['macro_f1']:.4%}, n={v['support']}"
                                           for k, v in metrics['groups'].items()), flush=True)
    for i, label in enumerate(m['labels']):
        print(f"  {i:02d} {label:30s} | P={m['precision'][i]:.2%} | Recall={m['recall'][i]:.2%} | "
              f"F1={m['f1'][i]:.2%} | n={m['class_support'][i]}", flush=True)


def save_metrics(directory, metrics):
    directory = Path(directory)
    write_json(directory / 'metrics.json', metrics)
    m = metrics['classification']['modulation']
    with (directory / 'per_class.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['class', 'precision', 'recall', 'f1', 'support'])
        for i, label in enumerate(m['labels']):
            writer.writerow([label, m['precision'][i], m['recall'][i], m['f1'][i], m['class_support'][i]])
    np.save(directory / 'confusion_matrix.npy', np.asarray(m['confusion_matrix'], dtype=np.int64))


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def resume_config(config):
    config = copy.deepcopy(config)
    config.pop('logging')
    config['run'].pop('output')
    config['runtime'].pop('device')
    return config


def load_trained(path, device='auto'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if checkpoint.get('format') != 'iqformer-c01-v1':
        raise ValueError('Expected a checkpoint from train_c01.py')
    config, labels = checkpoint['config'], checkpoint['labels']
    target = setup_runtime(config, device)
    model = build_model(config, labels).to(target)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.eval(), config, labels, target, checkpoint
