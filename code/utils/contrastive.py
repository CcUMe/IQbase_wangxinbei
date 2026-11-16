"""Supervised contrast across different records, with differentiable DDP keys."""
import torch
import torch.distributed as dist
from torch.nn import functional as F


def supervised_contrastive(embedding, labels, rows, temperature=.1, gather=True):
    """Local-anchor SupCon; exclude repeated occurrences of the same row from all pairs."""
    with torch.autocast(device_type=embedding.device.type, enabled=False):
        features = F.normalize(embedding.float(), dim=-1, eps=1e-8)
        candidates, all_labels, all_rows = features, labels, rows
        world = 1
        if gather and dist.is_initialized() and dist.get_world_size() > 1:
            from torch.distributed.nn.functional import all_gather
            candidates = torch.cat(all_gather(features), dim=0)
            world = dist.get_world_size()
            metadata = torch.stack((labels, rows), -1)
            parts = [torch.empty_like(metadata) for _ in range(dist.get_world_size())]
            dist.all_gather(parts, metadata)
            combined = torch.cat(parts)
            all_labels, all_rows = combined.unbind(-1)
        # Keep the gathered graph even if this rank has no positive anchors.
        # Other ranks may need its keys during collective backward.
        zero = features.sum() * 0 + candidates.sum() * 0
        different_record = rows[:, None] != all_rows[None]
        eligible = different_record & (labels[:, None] >= 0) & (all_labels[None] >= 0)
        positives = eligible & (labels[:, None] == all_labels[None])
        counts = positives.sum(-1)
        anchors = counts > 0
        if not bool(anchors.any()):
            return zero, {'coverage': 0., 'positives_per_anchor': 0.}
        logits = (features[anchors] @ candidates.T) / temperature
        mask, positive = eligible[anchors], positives[anchors]
        logits = logits - torch.logsumexp(logits.masked_fill(~mask, -torch.inf), dim=-1, keepdim=True)
        per_anchor = -(logits.masked_fill(~positive, 0).sum(-1) / counts[anchors])
        if world > 1:
            global_positive = ((all_rows[:, None] != all_rows[None]) &
                               (all_labels[:, None] == all_labels[None]) & (all_labels[:, None] >= 0))
            denominator = global_positive.any(-1).sum().clamp_min(1)
            loss = per_anchor.sum() * world / denominator
        else:
            loss = per_anchor.mean()
        fraction = float(anchors.sum() / (labels >= 0).sum().clamp_min(1))
        return loss + zero, {'coverage': fraction, 'positives_per_anchor': float(counts[anchors].float().mean())}


def load_with_training_aux(model, state, source_config=None):
    """Warmstart auxiliary changes and add zero-weight std columns to old classifiers."""
    source_pyramid = (source_config or {}).get('pyramid', {}).get('enabled', False)
    source_hierarchy = (source_config or {}).get('hierarchy', {}).get('enabled', False)
    source_hybrid = (source_config or {}).get('hybrid', {}).get('enabled', False)
    if (source_config is not None and
            source_config.get('hierarchy', {}).get('mechanism', {}).get('enabled', False) != model.preserve_signal):
        raise ValueError('Warmstart requires matching signal-preserving mechanism')
    if (source_config is not None and source_hybrid and model.hybrid and
            source_config['hybrid'].get('topology', 'parallel') != model.config['hybrid']['topology']):
        raise ValueError('Warmstart requires matching shared-frontend topology')
    if model.multiresolution or source_pyramid or source_hierarchy:
        adding_phase = False
        if source_config is not None:
            from models.multiresolution import mid_fusion_options
            from models.phase_relations import phase_relation_options
            old_mid = mid_fusion_options(source_config.get('mid_fusion'))
            new_mid = model.config['mid_fusion']
            if (old_mid['enabled'] or new_mid['enabled']) and old_mid != new_mid:
                raise ValueError('Warmstart requires matching mid_fusion architecture and insertion layer')
            old_phase = phase_relation_options(source_config.get('phase_relations'))
            new_phase = model.config['phase_relations']
            adding_phase = new_phase['enabled'] and not old_phase['enabled']
            if old_phase['enabled'] and old_phase != new_phase:
                raise ValueError('Warmstart requires matching phase_relations; cannot discard or reinterpret trained phase features')
        if (source_config is not None and
                source_config.get('modulation_temporal', {}).get('enabled', False) !=
                model.config['modulation_temporal']['enabled']):
            raise ValueError('Warmstart requires matching modulation_temporal readout; use a checkpoint from the same architecture')
        source_multi = source_pyramid or source_hierarchy
        if (not model.multiresolution or (source_config is not None and not source_multi)
                or (source_config is not None and source_hybrid != model.hybrid)
                or (source_config is not None and (source_hierarchy and not source_hybrid) != model.hierarchical)):
            raise ValueError('Warmstart requires matching multi-resolution architecture')
        if adding_phase:
            prefix = 'backbone.pyramid.phase_relations.'
            current = model.state_dict()
            missing = set(current)-set(state)
            unexpected = set(state)-set(current)
            if unexpected or any(not key.startswith(prefix) for key in missing):
                raise ValueError('Adding phase_relations may only introduce its own new parameters')
            # The added residual must start at zero even if loading into a model
            # instance which had previously trained this adapter.
            with torch.no_grad():
                for layer in model.backbone.pyramid.phase_relations.projections:
                    layer.weight.zero_()
            return model.load_state_dict(state, strict=False)
        return model.load_state_dict(state, strict=True)
    current = model.state_dict()
    state = dict(state)
    from models.iq_geometry import iq_geometry_options
    geometry_prefix='backbone.iq_geometry.'
    source_geometry=(source_config.get('iq_geometry',{}).get('enabled',False) if source_config is not None
                     else any(key.startswith(geometry_prefix) for key in state))
    target_geometry=model.config.get('iq_geometry',{}).get('enabled',False)
    if source_geometry and (not target_geometry or (source_config is not None and
            iq_geometry_options(source_config.get('iq_geometry'),model.config['patch_size'])!=model.config['iq_geometry'])):
        raise ValueError('Warmstart cannot discard or reinterpret trained iq_geometry')
    adding_geometry=target_geometry and not source_geometry
    if adding_geometry:
        with torch.no_grad():model.backbone.iq_geometry.up.weight.zero_()
    from models.patch_shift import patch_shift_options
    shift_prefix = 'backbone.patch_shift.'
    source_shift = (source_config.get('patch_shift', {}).get('enabled', False)
                    if source_config is not None else any(k.startswith(shift_prefix) for k in state))
    target_shift = model.config.get('patch_shift', {}).get('enabled', False)
    if source_shift and source_config is not None and patch_shift_options(
            source_config.get('patch_shift'), model.config['patch_size']) != model.config['patch_shift']:
        raise ValueError('Warmstart cannot discard or reinterpret a trained patch_shift adapter')
    adding_shift = target_shift and not source_shift
    if adding_shift:
        with torch.no_grad():
            model.backbone.patch_shift.up.weight.zero_()
    density_prefixes = ('backbone.time.density.', 'heads.modulation.density_projection.')
    source_density = (source_config.get('density_encoding', {}).get('enabled', False)
                      if source_config is not None else any(key.startswith(density_prefixes) for key in state))
    adding_density = model.config.get('density_encoding', {}).get('enabled', False) and not source_density
    if adding_density:
        # A base-model warmstart must restore exact initial behavior even when
        # reusing an instance whose new residuals have previously been trained.
        with torch.no_grad():
            model.backbone.time.density.shared_projection.weight.zero_()
            model.heads.modulation.density_projection.weight.zero_()
        for key,value in current.items():
            if key.startswith(density_prefixes):
                state[key]=value
    d = model.config['width']
    def layout(config):
        features = [('mean', d)]
        if config.get('classification_pooling', 'mean') == 'mean_std':
            features.append(('std', d))
        if config.get('physical_statistics', False):
            features.append(('physical', 32))
        result, offset = {}, 0
        for name, width in features:
            result[name] = slice(offset, offset + width)
            offset += width
        return result
    old_config = source_config or {'classification_pooling': 'mean_std' if state.get('heads.modulation.0.weight', current['heads.modulation.0.weight']).shape[1] == 2 * d else 'mean'}
    old_layout, new_layout = layout(old_config), layout(model.config)
    for task in ['modulation', 'protocol']:
        key = f'heads.{task}.0.weight'
        if key in state and (state[key].shape != current[key].shape or old_layout != new_layout):
            old = state[key]
            if not set(old_layout).issubset(new_layout):
                raise ValueError('Warmstart cannot silently discard classification features')
            if old.shape[0] == current[key].shape[0] and old.shape[1] == max(v.stop for v in old_layout.values()):
                expanded = old.new_zeros(current[key].shape)
                for name, source_slice in old_layout.items():
                    expanded[:, new_layout[name]] = old[:, source_slice]
                state[key] = expanded
    reset_aux = any(key.startswith('aux.contrastive.') and
                    (key not in current or value.shape != current[key].shape) for key, value in state.items())
    compatible = {key: value for key, value in state.items()
                  if not key.startswith('aux.contrastive.') or (not reset_aux and key in current)}
    result = model.load_state_dict(compatible, strict=False)
    source_refined = source_config.get('local_refinement', False) if source_config is not None else any(
        key.startswith('backbone.time.refinement.') for key in state)
    adding_refinement = model.config.get('local_refinement', False) and not source_refined
    allowed_missing = ('aux.contrastive.',)
    if adding_geometry:
        allowed_missing += (geometry_prefix,)
    if adding_shift:
        allowed_missing += (shift_prefix,)
    if adding_refinement:
        allowed_missing += ('backbone.time.refinement.',)
    source_multiscale = source_config.get('multiscale_frontend', False) if source_config is not None else any(
        key.startswith('backbone.time.multiscale.') for key in state)
    if model.config.get('multiscale_frontend', False) and not source_multiscale:
        allowed_missing += ('backbone.time.multiscale.',)
    invalid = ([key for key in result.missing_keys if not key.startswith(allowed_missing)] +
               [key for key in result.unexpected_keys if not key.startswith('aux.contrastive.')])
    if invalid:
        raise ValueError(f'Incompatible model weights: {invalid}')
    return result
