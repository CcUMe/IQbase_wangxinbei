"""Prevent zero-gradient DDP anchors from updating inactive AdamW task heads."""
import torch
import torch.distributed as dist


PREFIXES = ['heads.modulation.', 'heads.protocol.', 'heads.queries.', 'heads.regression.',
            'grids.0.', 'grids.1.', 'grids.2.', 'grids.6.',
            'aux.reconstruction.', 'aux.atomic.', 'aux.emergent.']
PREFIXES.append('aux.contrastive.')
PREFIXES.extend(f'heads.regression.{j}.' for j in range(7))


def active_heads(batch, config):
    active = set()
    if config['stage'] == 'pretrain':
        if config['reconstruction_weight'] or config['spectral_loss_weight']:
            active.add('aux.reconstruction.')
        for kind in ['atomic', 'emergent']:
            if config[kind + '_weight']:
                active.add(f'aux.{kind}.')
        return active
    for j, name in enumerate(['modulation', 'protocol']):
        if config['task_weights'][j] and bool((batch[name] >= 0).any()):
            active.add(f'heads.{name}.')
            # Contrastive projection activity is added after the loss confirms
            # at least one valid cross-record positive anchor on this rank.
    valid = batch['target_valid']
    weights = torch.as_tensor(config['parameter_weights'])
    if config['task_weights'][2] and bool((valid & (weights > 0)).any()):
        active.add('heads.queries.')
        if (config['model'].get('pyramid', {}).get('enabled', False)
                or config['model'].get('hierarchy', {}).get('enabled', False)):
            active.update(f'heads.regression.{j}.' for j in range(7) if weights[j] > 0 and bool(valid[:, j].any()))
        else:
            active.add('heads.regression.')
    for j in ([0, 1, 2, 6] if config['model']['parameter_mode'] == 'grid_residual' else []):
        if bool(valid[:, j].any()) and (config['grid_loss_weight'] or config['task_weights'][2] * weights[j]):
            active.update(['heads.queries.', f'grids.{j}.'])
    return active


def clear_inactive_head_gradients(model, active, device, world=1):
    if world > 1:
        flags = torch.tensor([prefix in active for prefix in PREFIXES], dtype=torch.int32, device=device)
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        active = {prefix for prefix, flag in zip(PREFIXES, flags.cpu().tolist()) if flag}
    for name, parameter in model.named_parameters():
        if not name.startswith('backbone.') and not any(name.startswith(prefix) for prefix in active):
            parameter.grad = None
