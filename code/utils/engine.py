"""Masked supervised/self-supervised losses and original-unit evaluation."""
import time

import numpy as np
import torch
from torch.nn import functional as F

from utils.common import TARGETS
from utils.dataset import to_device
from utils.metrics import classification_metrics
from utils.contrastive import supervised_contrastive


def autocast(device, precision):
    enabled = precision != 'fp32'
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    if device.type == 'cpu' and precision == 'fp16':
        enabled = False
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def zero_anchor(output):
    tensors = [output['modulation'], output['protocol'], output['parameters']]
    tensors += list(output.get('grid_logits', {}).values())
    tensors += list(output.get('pretrain', {}).values())
    if 'contrastive_embedding' in output:
        tensors.append(output['contrastive_embedding'])
    return sum(x.float().sum() * 0 for x in tensors)


def supervised_loss(output, batch, scaler, grids, config, distributed=True):
    loss = zero_anchor(output)
    parts = {}
    weights = config.get('task_weights', [1.0, 1.0, 1.0])
    # Reduce counts once per microbatch. Local means cannot simply be averaged
    # by DDP when paired/missing labels differ between ranks.
    import torch.distributed as dist
    world = dist.get_world_size() if distributed and dist.is_initialized() else 1
    grid_matches = {}
    for key, logits in output['grid_logits'].items():
        j = int(key)
        values = torch.as_tensor(grids[j], device=logits.device)
        minimum, index = (batch['targets'][:, j, None] - values).abs().min(-1)
        match = batch['target_valid'][:, j] & (minimum <= 1e-6 * values.abs().max().clamp_min(1))
        grid_matches[key] = match, index
    counts = torch.stack([(batch[t] >= 0).sum() for t in ['modulation','protocol']] +
        [batch['target_valid'][:, j].sum() for j in range(7)] +
        [grid_matches[k][0].sum() for k in grid_matches]).float()
    global_counts = counts.clone()
    if world > 1:
        dist.all_reduce(global_counts)
    def normalize(value, index):
        return value * (world * counts[index] / global_counts[index].clamp_min(1)) if world > 1 else value
    for j, task in enumerate(['modulation', 'protocol']):
        valid = batch[task] >= 0
        value = F.cross_entropy(output[task][valid].float(), batch[task][valid],
                                label_smoothing=config.get('label_smoothing', 0.0)) if valid.any() else loss * 0
        value = normalize(value, j)
        loss = loss + weights[j] * value
        parts[task] = float(value.detach())
    if config.get('contrastive_weight', 0) and weights[0]:
        if 'contrastive_embedding' not in output:
            raise ValueError('Supervised contrastive objective requires its training-only projection head')
        rows = torch.as_tensor(batch['rows'], dtype=torch.long, device=batch['iq'].device)
        value, stats = supervised_contrastive(output['contrastive_embedding'], batch['modulation'], rows,
                                              config['contrastive_temperature'], config.get('contrastive_gather', True))
        loss = loss + weights[0] * config['contrastive_weight'] * value
        parts.update(supcon=float(value.detach()), supcon_coverage=stats['coverage'],
                     supcon_positives=stats['positives_per_anchor'])
    mean = torch.as_tensor(scaler['mean'], device=batch['iq'].device)
    std = torch.as_tensor(scaler['std'], device=batch['iq'].device)
    targets = (batch['targets'].float() - mean) / std
    errors = output['parameters'].float() - targets
    valid = batch['target_valid']
    fields, field_weights = [], []
    parameter_weights = config.get('parameter_weights', [1.] * 7)
    for j in range(7):
        if global_counts[j + 2] > 0 and parameter_weights[j] > 0:
            value = loss * 0
            if valid[:, j].any():
                residual = errors[valid[:, j], j]
                value = (F.huber_loss(residual, torch.zeros_like(residual), delta=config.get('huber_delta', 1.))
                         if config.get('regression_loss') == 'huber' else residual.square().mean())
            fields.append(normalize(value, j + 2))
            field_weights.append(parameter_weights[j])
    value = (torch.stack(fields) * torch.as_tensor(field_weights, device=errors.device)).sum() / sum(field_weights) if fields else loss * 0
    loss = loss + weights[2] * value
    parts['parameters'] = float(value.detach())
    grid_terms = []
    for slot, (key, logits) in enumerate(output['grid_logits'].items()):
        match, index = grid_matches[key]
        if global_counts[9 + slot] > 0:
            value = F.cross_entropy(logits[match].float(), index[match]) if match.any() else loss * 0
            grid_terms.append(normalize(value, 9 + slot))
    if grid_terms:
        value = torch.stack(grid_terms).mean()
        loss = loss + config.get('grid_loss_weight', 0.1) * value
        parts['grid'] = float(value.detach())
    return loss, parts


def patch_mask(lengths, patch_size, ratio, span, seed, iq=None, strategy='random', energy_fraction=0.5, energy_floor=0.05):
    if strategy not in ['random', 'energy_mixed']:
        raise ValueError('Unknown mask strategy')
    rng = np.random.default_rng(seed)
    counts = (lengths.detach().cpu().numpy() // patch_size).astype(int)
    result = np.zeros((len(counts), int(counts.max())), dtype=bool)
    for i, n in enumerate(counts):
        target = max(1, min(n - 1, int(round(n * ratio))))
        selected = set()
        if strategy == 'energy_mixed' and energy_fraction > 0:
            if iq is None:
                raise ValueError('Energy masking requires IQ')
            energy = iq[i, :n * patch_size].detach().float().square().sum(-1).reshape(n, patch_size).mean(-1).cpu().numpy()
            probability = energy + max(float(energy.mean()), 1e-12) * energy_floor
            probability = probability / probability.sum()
            selected.update(rng.choice(n, min(target, max(1, int(target * energy_fraction))), replace=False, p=probability).tolist())
        if len(selected) == target:
            result[i, list(selected)] = True
            continue
        for start in rng.permutation(n):
            for pos in range(int(start), min(int(start) + span, n)):
                selected.add(pos)
                if len(selected) == target:
                    break
            if len(selected) == target:
                break
        result[i, list(selected)] = True
    return torch.from_numpy(result).to(lengths.device)


def pretraining_loss(output, batch, mask, config, codebook=None):
    selected = mask & output['token_valid']
    denominator = selected.sum(1).clamp_min(1)
    loss, parts = zero_anchor(output), {}
    prediction = output['pretrain']['reconstruction'].float()
    target = output['reconstruction_target'].reshape(prediction.shape)
    per_token = (prediction - target).square().mean(-1)
    reconstruction = ((per_token * selected).sum(1) / denominator).mean()
    continuous_weight = config.get('reconstruction_weight', 1.0)
    loss = loss + continuous_weight * reconstruction
    parts['reconstruction'] = float(reconstruction.detach())
    for kind in ['atomic', 'emergent']:
        weight = config.get(kind + '_weight', 1.0)
        if kind in output['pretrain'] and weight:
            logits = output['pretrain'][kind].float()
            labels = batch[kind]
            ce = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction='none')
            if config.get('equalize_emergent_spans', False) and kind == 'emergent':
                spans = torch.as_tensor(codebook['symbol_lengths'], device=logits.device)
                ce = ce / spans[labels.clamp_min(0)]
            value = ((ce * selected).sum(1) / denominator).mean()
            loss = loss + weight * value
            parts[kind] = float(value.detach())
    if config.get('spectral_loss_weight', 0):
        p = prediction.shape[-1] // 2
        a, b = prediction.reshape(*prediction.shape[:2], p, 2), target.reshape(*target.shape[:2], p, 2)
        fa = torch.fft.fft(torch.complex(a[..., 0], a[..., 1]))
        fb = torch.fft.fft(torch.complex(b[..., 0], b[..., 1]))
        per_token = (torch.log1p(fa.abs().square()) - torch.log1p(fb.abs().square())).square().mean(-1)
        value = ((per_token * selected).sum(1) / denominator).mean()
        loss = loss + config['spectral_loss_weight'] * value
        parts['spectrum'] = float(value.detach())
    return loss, parts


@torch.inference_mode()
def evaluate(model, loader, device, scaler, schema, precision='fp32', max_batches=0, proxy_weights=None):
    model.eval()
    confusion = {task: np.zeros((len(labels), len(labels)), np.int64) for task, labels in schema['classification_labels'].items()}
    classification_sums = {task: [0., 0.] for task in confusion}
    count = np.zeros(7, np.int64)
    sse, sae = np.zeros(7), np.zeros(7)
    error_chunks = [[] for _ in range(7)]
    mean, std = np.asarray(scaler['mean']), np.asarray(scaler['std'])
    started, rows = time.monotonic(), 0
    for step, batch in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        batch = to_device(batch, device)
        with autocast(device, precision):
            output = model(batch['iq'], batch['lengths'])
        rows += len(batch['lengths'])
        for task in confusion:
            y = batch[task].cpu().numpy()
            prediction = output[task].argmax(-1).cpu().numpy()
            valid = y >= 0
            np.add.at(confusion[task], (y[valid], prediction[valid]), 1)
            selected = batch[task] >= 0
            if selected.any():
                logits = output[task][selected].float()
                classification_sums[task][0] += float(F.cross_entropy(logits, batch[task][selected], reduction='sum'))
                classification_sums[task][1] += float(logits.softmax(-1).max(-1).values.sum())
        prediction = output['parameters'].float().cpu().numpy() * std + mean
        truth = batch['targets'].cpu().numpy()
        valid = batch['target_valid'].cpu().numpy()
        if not np.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite parameter predictions')
        for j in range(7):
            errors = prediction[valid[:, j], j] - truth[valid[:, j], j]
            count[j] += len(errors)
            sse[j] += np.square(errors).sum()
            sae[j] += np.abs(errors).sum()
            error_chunks[j].append(np.abs(errors))
    metrics = {'classification': {}, 'parameters': {}, 'rows': rows, 'elapsed_seconds': time.monotonic() - started}
    for task, matrix in confusion.items():
        metrics['classification'][task] = {**classification_metrics(matrix, schema['classification_labels'][task]),
            'cross_entropy': classification_sums[task][0] / max(int(matrix.sum()), 1),
            'mean_confidence': classification_sums[task][1] / max(int(matrix.sum()), 1)}
    rmse = np.sqrt(sse / np.maximum(count, 1))
    for j, name in enumerate(TARGETS):
        errors = np.concatenate(error_chunks[j]) if error_chunks[j] else np.empty(0)
        metrics['parameters'][name] = {'count': int(count[j]), 'rmse': float(rmse[j]) if count[j] else None,
            'mae': float(sae[j] / count[j]) if count[j] else None,
            'absolute_error_p95': float(np.quantile(errors, .95)) if len(errors) else None}
    r = float((rmse[count > 0] / std[count > 0]).mean()) if (count > 0).any() else None
    metrics['parameter_proxy'] = 1 / (1 + r) if r is not None else 0.0
    weights = proxy_weights or {'protocol': 30., 'modulation': 30., 'parameter': 40.}
    metrics['proxy_score'] = (weights['protocol'] * metrics['classification']['protocol']['accuracy'] +
                              weights['modulation'] * metrics['classification']['modulation']['accuracy'] + weights['parameter'] * metrics['parameter_proxy'])
    metrics['proxy_task_weights'] = weights
    metrics['score_note'] = 'Local normalized-RMSE proxy, NOT the official contest score.'
    return metrics


@torch.inference_mode()
def evaluate_pretraining(model, loader, device, config, codebook, maximum=20):
    model.eval()
    values = []
    for step, batch in enumerate(loader):
        if maximum and step >= maximum:
            break
        batch = to_device(batch, device)
        mask = patch_mask(batch['lengths'], model.config['patch_size'], config['mask_ratio'], config['mask_span'], config.get('mask_seed', 12345) + step,
                          batch['iq'], config.get('mask_strategy', 'random'), config.get('energy_fraction', .5), config.get('energy_floor', .05))
        with autocast(device, config.get('validation_precision', config['precision'])):
            output = model(batch['iq'], batch['lengths'], patch_mask=mask, return_pretrain=True)
            loss, _ = pretraining_loss(output, batch, mask, config, codebook)
        values.extend([float(loss)] * len(batch['lengths']))
    return float(np.mean(values)) if values else None
