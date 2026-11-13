"""Readable versioned experiment settings; compile to internal runtime options."""
from copy import deepcopy
import math

from utils.common import TARGETS, read_config_json
from utils.position import POSITION_DEFAULTS
from models.multiresolution import PYRAMID_DEFAULTS, READOUT_DEFAULTS, validate_pyramid
from models.hierarchical import HIERARCHY_DEFAULTS, HYBRID_DEFAULTS, validate_hierarchy


DEFAULTS = {
    'schema_version': 2,
    'experiment': {'name': 'c01', 'description': ''},
    'run': {'data': '', 'output': '', 'codebook': '', 'init_checkpoint': '', 'warmstart_checkpoint': ''},
    'model': {
        'encoder': {'width': 384, 'depth': 8, 'heads': 6, 'mlp_ratio': 4.0,
                    'dropout': 0.1, 'attention_dropout': 0.1, 'rope_base': 10000.0, 'qk_normalization': False, 'qk_scale': 1.0},
        'signal': {'patch_size': 16, 'stft_window': 64, 'phase_features': False,
                   'global_spectrum': True, 'spectral_backend': 'fft',
                   'position': 'samples', 'rms_epsilon': 1e-6, 'normalize_fusion': False, 'local_refinement': False,
                   'position_options': deepcopy(POSITION_DEFAULTS),
                   'scale_embedding': True, 'global_spectrum_input': True, 'multiscale_frontend': False},
        'heads': {'parameter_mode': 'regression', 'classification_hidden_ratio': 0.5,
                  'grid_residual_scale': 0.1, 'parameter_query_temperature': 1.0, 'contrastive_projection_dim': 128,
                  'classification_pooling': 'mean', 'physical_statistics': False, 'classification_fp32': False,
                  'multiscale_readout': deepcopy(READOUT_DEFAULTS)},
        'pyramid': deepcopy(PYRAMID_DEFAULTS),
        'hierarchy': deepcopy(HIERARCHY_DEFAULTS),
        'hybrid': deepcopy(HYBRID_DEFAULTS),
    },
    'data': {'batch_size': 128, 'steps_per_epoch': 0, 'merge_paired': False,
             'augmentation': {'phase_rotation_probability': 0.0,
                              'modulation_only': {'phase_rotation_probability': 0.0, 'circular_shift_probability': 0.0}},
             'sampling': {'mode': 'balanced', 'task_probabilities': {'modulation': 1.0, 'protocol': 1.0, 'parameter': 1.0}},
             'loader': {'num_workers': 2, 'pin_memory': True, 'persistent_workers': False, 'prefetch_factor': 2}},
    'training': {'stage': 'supervised', 'epochs': 30, 'max_steps': 0,
                 'gradient_accumulation_steps': 1, 'gradient_clip': 1.0},
    'finetuning': {'strategy': 'full', 'head_warmup_steps': 500, 'upper_layers_steps': 1500,
                   'unfreeze_last_n_blocks': 4, 'layer_lr_decay': 1.0, 'frontend_lr_factor': 1.0,
                   'anchor_weight': 0.0, 'validation_every_steps': 0, 'validate_stage_boundaries': True},
    'optimizer': {'name': 'adamw', 'learning_rate': 3e-4, 'backbone_lr_factor': 1.0,
                  'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-8,
                  'exclude_bias_and_norm_from_decay': False, 'skip_inactive_heads': True},
    'scheduler': {'name': 'cosine', 'warmup_steps': 500, 'warmup_ratio': 0.0,
                  'warmup_start_factor': 0.0, 'min_lr_ratio': 0.05},
    'loss': {
        'supervised': {'task_weights': {'modulation': 1.0, 'protocol': 1.0, 'parameter': 1.0},
                       'contrastive': {'weight': 0.0, 'temperature': 0.1, 'gather_distributed': True},
                       'label_smoothing': 0.0, 'regression': 'mse', 'huber_delta': 1.0,
                       'parameter_weights': {name: 1.0 for name in TARGETS}, 'grid_weight': 0.1},
        'pretraining': {'mask': {'ratio': 0.3, 'span': 2, 'strategy': 'random',
                                  'energy_fraction': 0.5, 'energy_floor': 0.05},
                        'weights': {'reconstruction': 0.0, 'atomic': 1.0, 'emergent': 1.0, 'spectrum': 0.0},
                        'equalize_emergent_spans': False},
    },
    'evaluation': {'every_epochs': 1, 'batch_size': 128, 'max_batches': 0, 'precision': 'fp32',
                   'mask_seed': 12345, 'selection_metric': 'auto',
                   'early_stopping': {'patience': 0, 'min_delta': 0.001},
                   'proxy_task_weights': {'modulation': 30.0, 'protocol': 30.0, 'parameter': 40.0}},
    'checkpoint': {'resume_from': '', 'save_last': True, 'save_best': True,
                   'save_best_modulation': False, 'save_best_proxy': False,
                   'save_every_steps': 0, 'save_every_epochs': 0},
    'runtime': {'device': 'auto', 'gpus': '', 'processes': 1, 'precision': 'bf16', 'seed': 2026, 'threads': 4,
                'log_every': 50, 'deterministic': False, 'allow_tf32': False,
                'progress_bar': True, 'progress_min_interval': 0.5},
}


def merge_known(base, updates, path=''):
    """Reject misspellings instead of silently accepting ineffective settings."""
    if not isinstance(updates, dict):
        raise ValueError(f'{path or "config"} must be an object')
    result = deepcopy(base)
    for key, value in updates.items():
        location = f'{path}.{key}' if path else key
        if key not in base:
            raise ValueError(f'Unknown configuration field: {location}')
        if isinstance(base[key], dict):
            result[key] = merge_known(base[key], value, location)
        else:
            if isinstance(base[key], bool) and type(value) is not bool:
                raise ValueError(f'{location} must be a JSON boolean')
            result[key] = deepcopy(value)
    return result


def legacy_config(raw):
    """Read the original flat experiment format, without changing checkpoints."""
    result = deepcopy(DEFAULTS)
    mapping = {
        'seed': ('runtime', 'seed'), 'precision': ('runtime', 'precision'), 'threads': ('runtime', 'threads'),
        'log_every': ('runtime', 'log_every'), 'num_workers': ('data', 'loader', 'num_workers'),
        'epochs': ('training', 'epochs'), 'batch_size': ('data', 'batch_size'), 'steps_per_epoch': ('data', 'steps_per_epoch'),
        'merge_paired': ('data', 'merge_paired'), 'learning_rate': ('optimizer', 'learning_rate'),
        'weight_decay': ('optimizer', 'weight_decay'), 'backbone_lr_factor': ('optimizer', 'backbone_lr_factor'),
        'warmup_steps': ('scheduler', 'warmup_steps'), 'gradient_clip': ('training', 'gradient_clip'),
        'validation_batches': ('evaluation', 'max_batches'), 'label_smoothing': ('loss', 'supervised', 'label_smoothing'),
        'regression_loss': ('loss', 'supervised', 'regression'), 'grid_loss_weight': ('loss', 'supervised', 'grid_weight'),
        'mask_ratio': ('loss', 'pretraining', 'mask', 'ratio'), 'mask_span': ('loss', 'pretraining', 'mask', 'span'),
        'mask_strategy': ('loss', 'pretraining', 'mask', 'strategy'),
        'equalize_emergent_spans': ('loss', 'pretraining', 'equalize_emergent_spans'),
        **{key + '_weight': ('loss', 'pretraining', 'weights', key) for key in ['reconstruction', 'atomic', 'emergent']},
        'spectral_loss_weight': ('loss', 'pretraining', 'weights', 'spectrum'),
    }
    result['loss']['pretraining']['weights']['reconstruction'] = 1.0
    for key, value in raw.items():
        if key == 'model':
            for name, option in value.items():
                if name in ['pyramid', 'hierarchy', 'hybrid']:
                    template = {'pyramid': PYRAMID_DEFAULTS, 'hierarchy': HIERARCHY_DEFAULTS, 'hybrid': HYBRID_DEFAULTS}[name]
                    result['model'][name] = merge_known(template, option, f'model.{name}')
                    continue
                groups = [group for group in ['encoder', 'signal', 'heads'] if name in result['model'][group]]
                if len(groups) != 1:
                    raise ValueError(f'Unknown legacy model setting: {name}')
                result['model'][groups[0]][name] = option
            if 'attention_dropout' not in value:
                result['model']['encoder']['attention_dropout'] = value.get('dropout', 0.1)
        elif key == 'task_weights':
            result['loss']['supervised']['task_weights'] = dict(zip(['modulation', 'protocol', 'parameter'], value, strict=True))
        elif key in mapping:
            node = result
            for part in mapping[key][:-1]:
                node = node[part]
            node[mapping[key][-1]] = value
        else:
            raise ValueError(f'Unknown legacy configuration field: {key}')
    return result


def resolve_config(raw):
    config = merge_known(DEFAULTS, raw) if 'schema_version' in raw else legacy_config(raw)
    validate_config(config)
    return config


def load_config(path):
    return resolve_config(read_config_json(path))


def resume_comparison_config(config):
    """Exclude loading/saving and display controls from strict resume matching."""
    result = deepcopy(config)
    result.pop('checkpoint', None)
    result.pop('run', None)
    result['training'].pop('max_steps', None)  # Stop cap never changes the planned LR schedule.
    for key in ['progress_bar', 'progress_min_interval', 'gpus', 'processes']:
        result['runtime'].pop(key, None)
    return result


def model_options(c):
    result = {**{k: v for group in ['encoder', 'signal', 'heads'] for k, v in c['model'][group].items()},
              'pyramid': deepcopy(c['model']['pyramid']), 'hierarchy': deepcopy(c['model']['hierarchy']),
              'hybrid': deepcopy(c['model']['hybrid'])}
    if result['hierarchy']['enabled']:
        validate_hierarchy(result['hierarchy'], result['multiscale_readout'])
    if result['hierarchy']['enabled'] and not result['hybrid']['enabled']:
        result['width'] = result['hierarchy']['stage_widths'][-1]
        result['depth'] = sum(result['hierarchy']['stage_depths'])
    return result


def validate_config(c):
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(c['schema_version'] == 2, 'Only schema_version=2 is supported')
    require(all(isinstance(v, str) for v in c['run'].values()), 'run paths must be strings; use "" for unused paths')
    require(not (c['run']['init_checkpoint'] and c['run']['warmstart_checkpoint']), 'Choose run.init_checkpoint OR run.warmstart_checkpoint')
    require(isinstance(c['runtime']['gpus'], str), 'runtime.gpus must be a string')
    require(type(c['runtime']['processes']) is int and c['runtime']['processes'] in [1, 2], 'runtime.processes must be 1 or 2')
    require(isinstance(c['checkpoint']['resume_from'], str), 'checkpoint.resume_from must be a string; use an empty string to start a new run')
    require('\x00' not in c['checkpoint']['resume_from'], 'checkpoint.resume_from contains a null byte')
    def finite_numbers(node, path='config'):
        if isinstance(node, dict):
            for key, value in node.items():
                finite_numbers(value, f'{path}.{key}')
        elif isinstance(node, list):
            for value in node:
                finite_numbers(value, path)
        elif isinstance(node, (float, int)) and not math.isfinite(node):
            raise ValueError(f'{path} must be finite')
    finite_numbers(c)
    m = model_options(c)
    d, t, o, s, e, r = (c[key] for key in ['data', 'training', 'optimizer', 'scheduler', 'evaluation', 'runtime'])
    require(type(r['seed']) is int and r['seed'] >= 0 and type(e['mask_seed']) is int and e['mask_seed'] >= 0, 'Seeds must be nonnegative integers')
    require(r['progress_min_interval'] >= 0, 'progress_min_interval must be nonnegative')
    for key, value in {'width': m['width'], 'depth': m['depth'], 'heads': m['heads'], 'patch_size': m['patch_size'],
                       'stft_window': m['stft_window'], 'batch_size': d['batch_size'], 'epochs': t['epochs'],
                       'gradient_accumulation_steps': t['gradient_accumulation_steps'], 'every_epochs': e['every_epochs'],
                       'evaluation.batch_size': e['batch_size'], 'threads': r['threads'], 'log_every': r['log_every']}.items():
        require(type(value) is int and value > 0, f'{key} must be a positive integer')
    for key, value in {'steps_per_epoch': d['steps_per_epoch'], 'max_steps': t['max_steps'], 'num_workers': d['loader']['num_workers'],
                       'max_batches': e['max_batches'], 'save_every_steps': c['checkpoint']['save_every_steps'],
                       'save_every_epochs': c['checkpoint']['save_every_epochs']}.items():
        require(type(value) is int and value >= 0, f'{key} must be a nonnegative integer')
    if not m['hierarchy']['enabled'] or m['hybrid']['enabled']:
        require(m['width'] % m['heads'] == 0 and (m['width'] // m['heads']) % 2 == 0, 'Attention head width must be even')
    require(512 % m['patch_size'] == 0 and m['stft_window'] >= m['patch_size'] and (m['stft_window'] - m['patch_size']) % 2 == 0,
            'Patch must divide 512; STFT and patch centers must align')
    require(m['position'] in ['samples', 'window', 'dual', 'physical'], 'Unknown position mode')
    position = m['position_options']
    require(position['reference_length'] > 0, 'position_options.reference_length must be positive')
    require(0 < position['window_head_fraction'] < 1, 'window_head_fraction must be in (0,1)')
    require(m['position'] != 'dual' or m['heads'] >= 2, 'dual position needs at least two heads')
    require(position['sample_rate_hz'] >= 0 and position['reference_sample_rate_hz'] > 0, 'Invalid sampling-rate configuration')
    require(m['position'] != 'physical' or position['sample_rate_hz'] > 0,
            'Current dataset has no sampling-rate metadata; physical mode requires a real constant sample_rate_hz at train and inference')
    require(m['spectral_backend'] in ['fft', 'matmul'], 'spectral_backend must be fft or matmul')
    require(m['parameter_mode'] in ['regression', 'grid_residual'], 'Unknown parameter_mode')
    require(type(m['contrastive_projection_dim']) is int and m['contrastive_projection_dim'] > 0, 'contrastive_projection_dim must be a positive integer')
    require(m['classification_pooling'] in ['mean', 'mean_std', 'multiscale_attention_stats'], 'Unknown classification_pooling')
    pyramid = c['model']['pyramid']
    hierarchy = c['model']['hierarchy']
    hybrid = c['model']['hybrid']
    require(all(type(v['enabled']) is bool for v in [pyramid, hierarchy, hybrid]), 'Architecture enabled flags must be boolean')
    require(not hybrid['enabled'] or hierarchy['enabled'], 'hybrid.enabled requires hierarchy.enabled')
    require(hybrid['topology'] in ['parallel', 'serial'], 'hybrid.topology must be parallel or serial')
    if hierarchy['mechanism']['enabled']:
        require(hybrid['enabled'] and hybrid['topology'] == 'serial',
                'Signal-preserving mechanism requires the serial unified backbone')
    if hybrid['enabled'] and hybrid['topology'] == 'serial':
        require(m['patch_size'] % math.prod(hierarchy['downsample_strides']) == 0,
                'Unified stage reduction must divide Transformer patch_size')
        require(hierarchy['spectrum']['centered_windows'],
                'Unified frontend requires spectrum.centered_windows=true for aligned sample centers')
        require(m['position_options']['reference_length'] == 1024,
                'Unified frontend and cross-layer readout use reference_length=1024')
    require(not (pyramid['enabled'] and hierarchy['enabled']), 'Choose hierarchy OR pyramid, not both')
    if pyramid['enabled']:
        validate_pyramid(pyramid, m['multiscale_readout'], m['patch_size'])
    if hierarchy['enabled']:
        validate_hierarchy(hierarchy, m['multiscale_readout'])
        require(c['finetuning']['layer_lr_decay'] == 1 and c['finetuning']['frontend_lr_factor'] == 1,
                'Hierarchical supervised training requires uniform backbone LR (layer/frontend factors=1)')
    require((pyramid['enabled'] or hierarchy['enabled']) == (m['classification_pooling'] == 'multiscale_attention_stats'),
            'Multi-scale architecture and classification_pooling=multiscale_attention_stats must be enabled together')
    if pyramid['enabled'] or hierarchy['enabled']:
        require(t['stage'] == 'supervised', 'Multi-resolution architecture currently supports supervised stage only')
        require(m['position'] == 'samples', 'Multi-resolution architecture currently requires position=samples')
        require(m['classification_fp32'], 'Multi-resolution task readouts require classification_fp32=true')
        require(not any(m[k] for k in ['physical_statistics', 'local_refinement', 'multiscale_frontend']),
                'Multi-resolution model uses its own readout/frontends; disable legacy statistics/refinement')
        require(c['loss']['supervised']['contrastive']['weight'] == 0, 'Multi-resolution model currently requires contrastive.weight=0')
        require(not c['run']['init_checkpoint'], 'Multi-resolution pretraining adaptation is deferred; use random supervised initialization')
        require(c['finetuning']['strategy'] == 'full' and not c['finetuning']['anchor_weight'],
                'Multi-resolution supervised verification requires full training without a pretraining anchor')
    require(all(math.isfinite(m[key]) and m[key] > 0 for key in ['mlp_ratio', 'rope_base', 'rms_epsilon', 'classification_hidden_ratio', 'parameter_query_temperature', 'qk_scale']), 'Model scales must be finite and positive')
    require(0 <= m['grid_residual_scale'] and math.isfinite(m['grid_residual_scale']), 'grid_residual_scale must be nonnegative')
    require(all(0 <= m[key] < 1 for key in ['dropout', 'attention_dropout']), 'Dropout must be in [0,1)')
    require(o['name'] == 'adamw' and o['learning_rate'] > 0 and o['backbone_lr_factor'] > 0 and o['weight_decay'] >= 0 and o['eps'] > 0, 'Invalid AdamW settings')
    require(len(o['betas']) == 2 and all(0 <= x < 1 for x in o['betas']), 'AdamW betas must be in [0,1)')
    require(s['name'] in ['cosine', 'constant'], 'scheduler.name must be cosine or constant')
    require(type(s['warmup_steps']) is int and s['warmup_steps'] >= 0 and 0 <= s['warmup_ratio'] < 1, 'Invalid warmup settings')
    require(not (s['warmup_steps'] and s['warmup_ratio']), 'Choose warmup_steps OR warmup_ratio; set the other to 0')
    require(0 <= s['min_lr_ratio'] <= 1 and 0 <= s['warmup_start_factor'] <= 1, 'LR ratios must be in [0,1]')
    require(r['precision'] in ['fp32', 'fp16', 'bf16'] and e['precision'] in ['fp32', 'fp16', 'bf16'], 'Unknown precision')
    require(t['stage'] in ['supervised', 'pretrain'] and t['gradient_clip'] >= 0, 'Invalid training stage/gradient clip')
    f = c['finetuning']
    require(f['strategy'] in ['full', 'gradual'], 'finetuning.strategy must be full or gradual')
    require(all(type(f[k]) is int and f[k] >= 0 for k in ['head_warmup_steps', 'upper_layers_steps', 'validation_every_steps']),
            'Fine-tuning step budgets must be nonnegative integers')
    require(type(f['unfreeze_last_n_blocks']) is int and f['unfreeze_last_n_blocks'] > 0,
            'unfreeze_last_n_blocks must be a positive integer')
    require(0 < f['layer_lr_decay'] <= 1 and f['frontend_lr_factor'] > 0 and f['anchor_weight'] >= 0,
            'Invalid fine-tuning LR factors or anchor_weight')
    require(t['stage'] == 'supervised' or (f['strategy'] == 'full' and not f['anchor_weight'] and not f['validation_every_steps']),
            'Fine-tuning phases/anchor/step validation require supervised stage')
    if f['strategy'] == 'gradual':
        require(f['unfreeze_last_n_blocks'] <= m['depth'], 'unfreeze_last_n_blocks exceeds encoder depth')
        require(f['head_warmup_steps'] + f['upper_layers_steps'] > 0, 'gradual strategy requires a nonempty frozen phase')
    require(d['sampling']['mode'] in ['balanced', 'weighted', 'proportional'], 'Unknown sampling mode')
    require(0 <= d['augmentation']['phase_rotation_probability'] <= 1, 'phase_rotation_probability must be in [0,1]')
    require(all(0 <= x <= 1 for x in d['augmentation']['modulation_only'].values()), 'modulation_only augmentation probabilities must be in [0,1]')
    require(all(math.isfinite(x) and x > 0 for x in d['sampling']['task_probabilities'].values()), 'Task sampling weights must be finite and positive')
    require(type(d['loader']['prefetch_factor']) is int and d['loader']['prefetch_factor'] > 0, 'prefetch_factor must be positive')
    u, p = c['loss']['supervised'], c['loss']['pretraining']
    require(u['contrastive']['weight'] >= 0 and u['contrastive']['temperature'] > 0, 'Invalid supervised contrastive settings')
    require(u['regression'] in ['mse', 'huber'] and u['huber_delta'] > 0 and 0 <= u['label_smoothing'] < 1, 'Invalid supervised loss settings')
    for name, weights in [('task_weights', u['task_weights']), ('parameter_weights', u['parameter_weights']),
                          ('pretraining.weights', p['weights']), ('proxy_task_weights', e['proxy_task_weights'])]:
        require(all(math.isfinite(x) and x >= 0 for x in weights.values()) and sum(weights.values()) > 0, f'Invalid {name}')
    require(u['grid_weight'] >= 0 and 0 < p['mask']['ratio'] < 1 and type(p['mask']['span']) is int and p['mask']['span'] > 0, 'Invalid grid weight/mask')
    require(p['mask']['strategy'] in ['random', 'energy_mixed'] and 0 <= p['mask']['energy_fraction'] <= 1 and p['mask']['energy_floor'] > 0, 'Invalid energy masking')
    require(e['selection_metric'] in ['auto', 'proxy_score', 'protocol_accuracy', 'modulation_accuracy', 'parameter_proxy', 'pretraining_loss'], 'Unknown selection_metric')
    require(e['selection_metric'] == 'auto' or (t['stage'] == 'pretrain') == (e['selection_metric'] == 'pretraining_loss'), 'Selection metric is incompatible with training stage')
    early = e['early_stopping']
    require(type(early['patience']) is int and early['patience'] >= 0 and early['min_delta'] >= 0,
            'Invalid early_stopping patience/min_delta')
    require(not early['patience'] or not e['max_batches'], 'Early stopping requires full validation (max_batches=0)')


def training_options(c):
    """Internal flat view; public files and saved resolved configs remain grouped."""
    u, p = c['loss']['supervised'], c['loss']['pretraining']
    return {**c['runtime'], **c['training'], **c['optimizer'],
            'model': model_options(c),
            'batch_size': c['data']['batch_size'], 'steps_per_epoch': c['data']['steps_per_epoch'],
            'merge_paired': c['data']['merge_paired'], 'num_workers': c['data']['loader']['num_workers'],
            'augmentation': c['data']['augmentation'],
            'task_weights': [u['task_weights'][name] for name in ['modulation', 'protocol', 'parameter']],
            'parameter_weights': [u['parameter_weights'][name] for name in TARGETS],
            'regression_loss': u['regression'], 'huber_delta': u['huber_delta'],
            'label_smoothing': u['label_smoothing'], 'grid_loss_weight': u['grid_weight'],
            'contrastive_weight': u['contrastive']['weight'], 'contrastive_temperature': u['contrastive']['temperature'],
            'contrastive_gather': u['contrastive']['gather_distributed'],
            'mask_ratio': p['mask']['ratio'], 'mask_span': p['mask']['span'], 'mask_strategy': p['mask']['strategy'],
            'energy_fraction': p['mask']['energy_fraction'], 'energy_floor': p['mask']['energy_floor'],
            **{name + '_weight': p['weights'][name] for name in ['reconstruction', 'atomic', 'emergent']},
            'spectral_loss_weight': p['weights']['spectrum'], 'equalize_emergent_spans': p['equalize_emergent_spans'],
            'mask_seed': c['evaluation']['mask_seed'], 'validation_precision': c['evaluation']['precision']}


def make_scheduler(optimizer, settings, total_steps):
    """LR for updates 0..T-1: linear warmup, then cosine reaching the floor."""
    import torch
    warmup = settings['warmup_steps'] or int(total_steps * settings['warmup_ratio'])
    if warmup >= total_steps:
        raise ValueError(f'warmup_steps={warmup} must be smaller than planned optimizer updates={total_steps}')
    floor, start = settings['min_lr_ratio'], settings['warmup_start_factor']

    def factor(step):
        if step < warmup:
            return start + (1 - start) * (step + 1) / warmup
        if settings['name'] == 'constant':
            return 1.0
        if total_steps > 1 and total_steps - warmup == 1:
            return floor
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup - 1)))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor), warmup
