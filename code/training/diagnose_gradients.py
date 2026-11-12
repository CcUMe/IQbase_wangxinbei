"""Reproduce task gradients and attention statistics on fixed training batches."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F

from models.rf_model import RFModel
from utils.common import model_bundle, runtime, seed_all, write_json
from utils.config import resolve_config, training_options
from utils.dataset import PreparedDataset, collate, to_device
from utils.engine import autocast, supervised_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--precision', default='bf16', choices=['bf16', 'fp32'])
    args = p.parse_args()
    device, _, world = runtime(args.device)
    if world != 1:
        raise ValueError('Diagnostic is single-process')
    seed_all(2026)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    dataset = PreparedDataset(args.data)
    if dataset.info['fingerprint'] != checkpoint['dataset_fingerprint']:
        raise ValueError('Different dataset split')
    rows = dataset.indices.copy()
    dataset.indices = np.arange(dataset.info['rows'])
    model = RFModel(**model_bundle(checkpoint)).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    options = training_options(resolve_config(checkpoint['training_config']))
    observations, handles = {}, []
    current = {}

    def activation_hook(name):
        def hook(module, inputs, output):
            x = output.detach().float()
            current[name] = {'rms': float(x.square().mean().sqrt()), 'max_abs': float(x.abs().max())}
            if x.ndim == 3 and (name.startswith('block_') or name == 'norm'):
                temporal_rms = (x - x.mean(1, keepdim=True)).square().mean().sqrt()
                current[name]['temporal_variation_ratio'] = float(temporal_rms / x.square().mean().sqrt().clamp_min(1e-12))
        return hook

    def attention_hook(name):
        def hook(module, inputs):
            with torch.no_grad():
                x, valid, positions = inputs
                b, n, _ = x.shape
                q, k, v = module.qkv(x).reshape(b, n, 3, module.heads, module.head_width).permute(2, 0, 3, 1, 4).unbind(0)
                q, k = module.rotate(q, positions), module.rotate(k, positions)
                if module.qk_normalization:
                    q = F.normalize(q.float(), dim=-1, eps=1e-6) * module.head_width ** .5 * module.qk_scale
                    k = F.normalize(k.float(), dim=-1, eps=1e-6) * module.head_width ** .5
                with torch.autocast(device_type=device.type, enabled=False):
                    logits = q.float() @ k.float().transpose(-2, -1) / module.head_width ** .5
                    probabilities = logits.masked_fill(~valid[:, None, None], -float('inf')).softmax(-1)
                    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
                current[name] = {'q_rms': float(q.float().square().mean().sqrt()),
                                 'k_rms': float(k.float().square().mean().sqrt()),
                                 'logit_max_abs': float(logits.abs().max()), 'logit_std': float(logits.std()),
                                 'normalized_entropy': float((entropy / valid.sum(-1)[:, None, None].float().log()).mean()),
                                 'near_one_hot_fraction': float((probabilities.max(-1).values > .99).float().mean())}
        return hook

    for name in ['time', 'spectrum', 'global_spectrum', 'norm']:
        module = getattr(model.backbone, name)
        if module is not None:
            handles.append(module.register_forward_hook(activation_hook(name)))
    for i, block in enumerate(model.backbone.blocks):
        handles.append(block.register_forward_hook(activation_hook(f'block_{i}')))
        handles.append(block.attention.register_forward_pre_hook(attention_hook(f'attention_{i}')))
    rng = np.random.default_rng(2026)
    vectors, names = [], []
    for task, label in enumerate(['modulation', 'protocol', 'parameter']):
        for length in [512, 1024]:
            eligible = rows[(dataset.meta['task'][rows] == task) & (dataset.meta['length'][rows] == length)]
            if not len(eligible):
                continue
            chosen = rng.choice(eligible, args.batch_size, replace=False)
            batch = to_device(collate([dataset[int(row)] for row in chosen]), device)
            model.zero_grad(set_to_none=True)
            current.clear()
            with autocast(device, args.precision):
                output = model(batch['iq'], batch['lengths'])
                loss, parts = supervised_loss(output, batch, checkpoint['target_scaler'], checkpoint['parameter_grids'], options)
            loss.backward()
            groups, top = defaultdict(float), []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    norm = float(param.grad.float().norm())
                    group = '.'.join(name.split('.')[:3] if name.startswith('backbone.blocks.') else name.split('.')[:2])
                    groups[group] += norm ** 2
                    top.append((norm, name))
            vector = torch.cat([param.grad.detach().flatten() for param in model.backbone.parameters()])
            vector = F.normalize(vector.float(), dim=0)
            vectors.append(vector)
            key = f'{label}_{length}'
            names.append(key)
            observations[key] = {'rows': chosen.tolist(), 'loss': float(loss.detach()), 'loss_parts': parts,
                'gradient_total': sum(groups.values()) ** .5,
                'gradient_modules': {k: v ** .5 for k, v in sorted(groups.items(), key=lambda item: -item[1])},
                'largest_parameter_gradients': sorted(top, reverse=True)[:15], 'activations': dict(current)}
            print(key, 'loss', round(float(loss.detach()), 4), 'grad', round(observations[key]['gradient_total'], 4), flush=True)
    matrix = torch.stack(vectors) @ torch.stack(vectors).T
    write_json(args.output, {'checkpoint': str(args.checkpoint), 'epoch': checkpoint['epoch'] + 1,
                            'precision': args.precision, 'observations': observations,
                            'gradient_cosine_names': names, 'backbone_gradient_cosines': matrix.cpu().tolist()})
    for handle in handles:
        handle.remove()


if __name__ == '__main__':
    main()
