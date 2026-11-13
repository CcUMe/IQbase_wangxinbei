"""One time/frequency backbone shared by all C01 tasks."""
import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from utils.signal_statistics import signal_statistics, STATISTICS_DIM
from utils.position import POSITION_DEFAULTS, position_coordinates
from models.multiresolution import (READOUT_DEFAULTS, pyramid_options, validate_pyramid,
    MultiResolutionPyramid, ModulationReadout, ProtocolReadout, ParameterQueries)
from models.hierarchical import hierarchy_options, validate_hierarchy, HierarchicalBackbone, SharedScaleFusion, HYBRID_DEFAULTS
from models.signal_mechanisms import DeepAnchoredModulationReadout, fold_values


class RotaryAttention(nn.Module):
    def __init__(self, width, heads, dropout, rope_base=10000.0, qk_normalization=False, qk_scale=1.0):
        super().__init__()
        if width % heads or (width // heads) % 2:
            raise ValueError('Attention head width must be even')
        self.heads, self.head_width, self.dropout = heads, width // heads, dropout
        self.qk_normalization = qk_normalization
        self.qk_scale = qk_scale
        self.qkv = nn.Linear(width, width * 3)
        self.projection = nn.Linear(width, width)
        self.register_buffer('inv_freq', rope_base ** (-torch.arange(0, self.head_width, 2).float() / self.head_width), persistent=False)

    def rotate(self, x, positions):
        angle = positions.float()[..., None] * self.inv_freq
        cos, sin = angle.cos(), angle.sin()
        if positions.ndim == 2:
            cos, sin = cos[:, None], sin[:, None]
        a, b = x[..., 0::2].float(), x[..., 1::2].float()
        return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2).to(x.dtype)

    def forward(self, x, valid, positions):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.rotate(q, positions), self.rotate(k, positions)
        if self.qk_normalization:
            # Pre-LN does not bound the learned Q/K projections. Normalize each
            # head to unit RMS so logit temperature cannot grow with Q/K norms.
            with torch.autocast(device_type=q.device.type, enabled=False):
                q = F.normalize(q.float(), dim=-1, eps=1e-6) * math.sqrt(self.head_width) * self.qk_scale
                k = F.normalize(k.float(), dim=-1, eps=1e-6) * math.sqrt(self.head_width)
            q, k = q.to(v.dtype), k.to(v.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :],
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.projection(y.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, width, heads, dropout, mlp_ratio=4.0, attention_dropout=None, rope_base=10000.0, qk_normalization=False, qk_scale=1.0):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = RotaryAttention(width, heads, dropout if attention_dropout is None else attention_dropout, rope_base, qk_normalization, qk_scale)
        hidden = max(1, int(width * mlp_ratio))
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid, positions):
        x = x + self.dropout(self.attention(self.norm1(x), valid, positions))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x * valid[..., None]


class TimeFrontend(nn.Sequential):
    """Optional zero-initialized local residual processing before patch compression."""
    def __init__(self, channels, width, patch_size, refinement=False, multiscale=False):
        super().__init__(nn.Conv1d(channels, width // 2, 5, padding=2), nn.GELU(),
                         nn.Conv1d(width // 2, width, patch_size, stride=patch_size))
        if refinement:
            # Preserve the RNG stream of every existing backbone/head parameter.
            with torch.random.fork_rng(devices=[]):
                self.refinement = nn.Sequential(nn.Conv1d(width // 2, width // 2, 5, padding=2, groups=width // 2),
                    nn.GELU(), nn.Conv1d(width // 2, width // 2, 1))
                nn.init.zeros_(self.refinement[2].weight)
                nn.init.zeros_(self.refinement[2].bias)
        if multiscale:
            with torch.random.fork_rng(devices=[]):
                hidden = width // 2
                self.multiscale = nn.ModuleDict({
                    'filters': nn.ModuleList([nn.Conv1d(hidden, hidden, 3, padding=rate,
                        dilation=rate, groups=hidden) for rate in [1, 4, 16]]),
                    'mix': nn.Conv1d(hidden * 3, hidden, 1)})
                nn.init.zeros_(self.multiscale['mix'].weight)
                nn.init.zeros_(self.multiscale['mix'].bias)

    def forward(self, x):
        hidden = self[1](self[0](x))
        if hasattr(self, 'refinement'):
            hidden = hidden + self.refinement(hidden)
        if hasattr(self, 'multiscale'):
            branches = [F.gelu(layer(hidden)) for layer in self.multiscale['filters']]
            hidden = hidden + self.multiscale['mix'](torch.cat(branches, 1))
        return self[2](hidden)


class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d, p, w = config['width'], config['patch_size'], config['stft_window']
        if w < p or (w - p) % 2 or 1024 % p:
            raise ValueError('STFT window/patch must have aligned centers; patch must divide 1024')
        self.patch_size, self.stft_window = p, w
        channels = 5 if config.get('phase_features', False) else 3
        self.time = TimeFrontend(channels, d, p, config.get('local_refinement', False),
                                 config.get('multiscale_frontend', False))
        self.spectrum = nn.Sequential(nn.Conv1d(w, d // 2, 1), nn.GELU(), nn.Conv1d(d // 2, d, 3, padding=1))
        self.gate = nn.Linear(d * 2, d)
        self.freq_projection = nn.Linear(d, d)
        self.scale_projection = nn.Sequential(nn.Linear(1, d), nn.Tanh())
        self.position_projection = nn.Sequential(nn.Linear(2, d), nn.Tanh())
        self.mask_embedding = nn.Parameter(torch.zeros(d))
        self.global_spectrum = nn.Sequential(nn.Linear(1024, d), nn.GELU(), nn.Linear(d, d)) if config.get('global_spectrum', True) else None
        n, k = torch.arange(w).float(), torch.arange(w).float()
        angle = 2 * math.pi * k[:, None] * n[None] / w
        window = torch.hann_window(w)
        real = torch.stack((angle.cos() * window, angle.sin() * window), 1)
        imag = torch.stack((-angle.sin() * window, angle.cos() * window), 1)
        self.register_buffer('stft_kernels', torch.cat((real, imag)), persistent=False)
        self.register_buffer('window_power', window.square().sum(), persistent=False)
        self.blocks = nn.ModuleList([Block(d, config['heads'], config['dropout'], config['mlp_ratio'],
                                          config['attention_dropout'], config['rope_base'], config['qk_normalization'], config['qk_scale']) for _ in range(config['depth'])])
        self.norm = nn.LayerNorm(d)

    def preprocess(self, iq, lengths, patch_mask=None):
        # All derived input features use the same masked view. Target values never
        # contribute to student RMS normalization or spectral/phase features.
        iq = iq.float()
        b, t, _ = iq.shape
        valid = torch.arange(t, device=iq.device)[None] < lengths[:, None]
        visible = valid.clone()
        if patch_mask is not None:
            visible = visible & ~patch_mask.repeat_interleave(self.patch_size, 1)[:, :t]
        masked = iq * visible[..., None]
        scale = (masked.square().sum((1, 2)) / visible.sum(1).clamp_min(1)).sqrt().clamp_min(self.config['rms_epsilon'])
        target = iq / scale[:, None, None]
        x = target * visible[..., None]
        return x, target, scale, valid

    def forward(self, iq, lengths, patch_mask=None, sample_rates=None, return_context=False):
        x, target, scale, point_valid = self.preprocess(iq, lengths, patch_mask)
        b, t, _ = x.shape
        p = self.patch_size
        n = t // p
        valid = torch.arange(n, device=x.device)[None] * p < lengths[:, None]
        amplitude = x.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
        features = [x, amplitude]
        if self.config.get('phase_features', False):
            previous = F.pad(x[:, :-1], (0, 0, 1, 0))
            real = x[..., 0] * previous[..., 0] + x[..., 1] * previous[..., 1]
            imag = x[..., 1] * previous[..., 0] - x[..., 0] * previous[..., 1]
            reliability = (amplitude.squeeze(-1) * F.pad(amplitude[:, :-1, 0], (1, 0))).clamp_max(1)
            magnitude = (real.square() + imag.square()).sqrt().clamp_min(1e-6)
            features.append(torch.stack((real, imag), -1) / magnitude[..., None] * reliability[..., None])
        if return_context:
            # Clear padding between convolutions. Otherwise a padded 512-point
            # record can differ from the same record evaluated at native length.
            inputs = torch.cat(features, -1) * point_valid[..., None]
            hidden = self.time[1](self.time[0](inputs.transpose(1, 2))) * point_valid[:, None]
            time = self.time[2](hidden).transpose(1, 2) * valid[..., None]
        else:
            time = self.time(torch.cat(features, -1).transpose(1, 2)).transpose(1, 2)
        # Real-valued fixed DFT convolution, centers exactly equal to time patches.
        with torch.autocast(device_type=x.device.type, enabled=False):
            transformed = F.conv1d(x.float().transpose(1, 2), self.stft_kernels.float(),
                                   stride=p, padding=(self.stft_window - p) // 2)
            re, im = transformed.chunk(2, 1)
            power = (re.square() + im.square()) / self.window_power
            local = torch.log1p(power)
            global_power = None
            if self.global_spectrum is not None:
                if self.config.get('spectral_backend', 'fft') == 'fft':
                    spectrum = torch.fft.fft(torch.complex(x[..., 0], x[..., 1]), n=1024)
                    global_power = spectrum.abs().square()
                else:
                    # Portable alternative; computed only when explicitly selected.
                    coordinate = torch.arange(1024, device=x.device).float()
                    phase = 2 * math.pi * coordinate[:, None] * coordinate[None] / 1024
                    padded = F.pad(x, (0, 0, 0, 1024 - t))
                    re = padded[..., 0] @ phase.cos().T + padded[..., 1] @ phase.sin().T
                    im = padded[..., 1] @ phase.cos().T - padded[..., 0] @ phase.sin().T
                    global_power = re.square() + im.square()
                global_power = torch.roll(global_power, 512, -1) / lengths[:, None]
                global_power = torch.log1p(global_power)
        if return_context:
            hidden = self.spectrum[1](self.spectrum[0](local * valid[:, None])) * valid[:, None]
            frequency = self.spectrum[2](hidden).transpose(1, 2) * valid[..., None]
        else:
            frequency = self.spectrum(local).transpose(1, 2)
        if self.config['normalize_fusion']:
            time = F.layer_norm(time, (time.shape[-1],))
            frequency = F.layer_norm(frequency, (frequency.shape[-1],))
        gate = torch.sigmoid(self.gate(torch.cat((time, frequency), -1)))
        h = time + gate * self.freq_projection(frequency)
        h = h + self.scale_projection(scale.log()[:, None])[:, None] * float(self.config['scale_embedding'])
        if global_power is not None:
            global_features = self.global_spectrum(global_power)
            if self.config['normalize_fusion']:
                global_features = F.layer_norm(global_features, (global_features.shape[-1],))
            h = h + global_features[:, None] * float(self.config['global_spectrum_input'])
        if patch_mask is not None:
            h = h + patch_mask[..., None] * self.mask_embedding
        else:
            h = h + self.mask_embedding * 0
        positions, coordinates = position_coordinates(lengths, n, p, self.config['heads'],
            self.config['position'], self.config['position_options'], sample_rates)
        # Preserve the origin/length path for timing tasks. A zero graph anchor
        # keeps DDP safe when this optional feature is disabled for an ablation.
        h = h + self.position_projection(coordinates) * float(self.config['position_options']['absolute_features'])
        if self.config['normalize_fusion']:
            h = F.layer_norm(h, (h.shape[-1],)) * valid[..., None]
        for block in self.blocks:
            h = block(h, valid, positions)
        encoded = self.norm(h) * valid[..., None]
        if return_context:
            return encoded, valid, target, dict(iq=x, scale=scale, point_valid=point_valid, frequency=frequency)
        return encoded, valid, target


class HybridBackbone(Backbone):
    """Keep original Transformer keys and add one jointly trained detail path."""
    def __init__(self, config):
        super().__init__(config)
        self.detail = HierarchicalBackbone(config, with_readout=False)
        widths = [config['width'], config['hierarchy']['stage_widths'][-1], config['hierarchy']['stage_widths'][0]]
        self.hybrid_fusion = SharedScaleFusion(widths, config['hierarchy'])

    def forward(self, iq, lengths, patch_mask=None, sample_rates=None, return_pyramid=False):
        if patch_mask is not None:
            raise ValueError('Hybrid backbone currently supports supervised training/inference only')
        coarse, valid, target, context = super().forward(iq, lengths, sample_rates=sample_rates, return_context=True)
        if not return_pyramid:
            return coarse, valid, target
        detail = self.detail.forward_features(iq, lengths)
        # Three resolutions, one shared readout: the original base, contextual
        # detail after staged encoding, and the high-resolution first stage.
        features = [coarse, detail['features'][-1], detail['features'][0]]
        masks = [valid, detail['valid'][-1], detail['valid'][0]]
        center = torch.arange(coarse.shape[1], device=iq.device).float() * self.patch_size + (self.patch_size - 1) / 2
        centers = [center, detail['centers'][-1], detail['centers'][0]]
        return self.hybrid_fusion(features, masks, centers, lengths, context['scale'])


class UnifiedBackbone(nn.Module):
    """One shared frontend feeds recurrent/staged encoding and Transformer core."""
    def __init__(self, config):
        super().__init__()
        self.config, self.patch_size = config, config['patch_size']
        self.stem = HierarchicalBackbone(config, with_readout=False, with_frontend_global=True)
        reduction = math.prod(config['hierarchy']['downsample_strides'])
        if self.patch_size % reduction:
            raise ValueError('Unified frontend stage reduction must divide Transformer patch_size')
        self.adapter_stride = self.patch_size // reduction
        d = config['width']
        self.token_projection = nn.Conv1d(config['hierarchy']['stage_widths'][-1], d,
                                          self.adapter_stride, stride=self.adapter_stride)
        self.token_norm = nn.LayerNorm(d)
        self.position_projection = nn.Sequential(nn.Linear(2, d), nn.Tanh())
        # Same core module definitions and state keys as the original backbone.
        self.blocks = nn.ModuleList([Block(d, config['heads'], config['dropout'], config['mlp_ratio'],
            config['attention_dropout'], config['rope_base'], config['qk_normalization'], config['qk_scale'])
            for _ in range(config['depth'])])
        self.norm = nn.LayerNorm(d)
        widths = [d, config['hierarchy']['stage_widths'][-1], config['hierarchy']['stage_widths'][0]]
        self.hybrid_fusion = SharedScaleFusion(widths, config['hierarchy'])

    def forward(self, iq, lengths, patch_mask=None, sample_rates=None, return_pyramid=False):
        if patch_mask is not None:
            raise ValueError('Unified backbone currently supports supervised training/inference only')
        detail = self.stem.forward_features(iq, lengths)
        h = self.token_projection(detail['features'][-1].transpose(1, 2)).transpose(1, 2)
        valid = torch.arange(h.shape[1], device=iq.device)[None] < (lengths // self.patch_size)[:, None]
        h = self.token_norm(h) * valid[..., None]
        positions, coordinates = position_coordinates(lengths, h.shape[1], self.patch_size, self.config['heads'],
            self.config['position'], self.config['position_options'], sample_rates)
        h = h + self.position_projection(coordinates) * float(self.config['position_options']['absolute_features'])
        for block in self.blocks:
            h = block(h, valid, positions)
        h = self.norm(h) * valid[..., None]
        if not return_pyramid:
            return h, valid, detail['target']
        # Adapter centers remain (p-1)/2 + j*p for the configured even-stride
        # stage reductions and non-overlapping final projection.
        center = torch.arange(h.shape[1], device=iq.device).float() * self.patch_size + (self.patch_size - 1) / 2
        pyramid = self.hybrid_fusion([h, detail['features'][-1], detail['features'][0]],
            [valid, detail['valid'][-1], detail['valid'][0]],
            [center, detail['centers'][-1], detail['centers'][0]], lengths, detail['scale'], detail['late_global'])
        if detail['signal_features'] is not None:
            values = detail['signal_features']
            pyramid['signal_features'] = values
            pyramid['signal_memory'] = torch.cat([fold_values(values, stride) for stride in
                [self.patch_size, math.prod(self.config['hierarchy']['downsample_strides']), 1]], 1)
        return pyramid


class RFModel(nn.Module):
    def __init__(self, model_config, schema, target_scaler, parameter_grids):
        super().__init__()
        defaults = dict(width=384, depth=8, heads=6, dropout=0.1, patch_size=16, stft_window=64,
                        phase_features=False, global_spectrum=True, spectral_backend='fft', position='samples',
                        parameter_mode='regression', pretrain_heads=False, atomic_vocab=0, emergent_vocab=0,
                        mlp_ratio=4.0, attention_dropout=model_config.get('dropout', 0.1), rope_base=10000.0,
                        rms_epsilon=1e-6, classification_hidden_ratio=0.5, grid_residual_scale=0.1,
                        parameter_query_temperature=1.0, qk_normalization=False, normalize_fusion=False, qk_scale=1.0,
                        contrastive_head=False, contrastive_projection_dim=128, classification_pooling='mean',
                        physical_statistics=False, local_refinement=False, classification_fp32=False,
                        scale_embedding=True, global_spectrum_input=True, multiscale_frontend=False)
        self.config = {**defaults, **model_config}
        self.config['pyramid'] = pyramid_options(model_config.get('pyramid'))
        self.config['hierarchy'] = hierarchy_options(model_config.get('hierarchy'))
        self.config['hybrid'] = {**HYBRID_DEFAULTS, **model_config.get('hybrid', {})}
        self.config['multiscale_readout'] = {**READOUT_DEFAULTS, **model_config.get('multiscale_readout', {})}
        self.hybrid = self.config['hybrid']['enabled']
        if self.config['hybrid']['topology'] not in ['parallel', 'serial']:
            raise ValueError('hybrid.topology must be parallel or serial')
        self.preserve_signal = self.config['hierarchy']['mechanism']['enabled']
        if self.preserve_signal and not (self.hybrid and self.config['hybrid']['topology'] == 'serial'):
            raise ValueError('Signal-preserving mechanism requires the serial unified backbone')
        self.hierarchical = self.config['hierarchy']['enabled'] and not self.hybrid
        if self.hybrid and not self.config['hierarchy']['enabled']:
            raise ValueError('hybrid.enabled requires hierarchy.enabled for the detail encoder')
        if (self.hierarchical or self.hybrid) and self.config['pyramid']['enabled']:
            raise ValueError('Choose hierarchy OR legacy pyramid, not both')
        if self.hierarchical or self.hybrid:
            validate_hierarchy(self.config['hierarchy'], self.config['multiscale_readout'])
        if self.hierarchical:
            self.config['width'] = self.config['hierarchy']['stage_widths'][-1]
            self.config['depth'] = sum(self.config['hierarchy']['stage_depths'])
        self.multiresolution = self.config['pyramid']['enabled'] or self.hierarchical or self.hybrid
        if self.multiresolution:
            if not self.hierarchical and not self.hybrid:
                validate_pyramid(self.config['pyramid'], self.config['multiscale_readout'], self.config['patch_size'])
            if self.config['position'] != 'samples':
                raise ValueError('Multi-resolution supervised model currently requires position=samples')
            if any(self.config.get(k, False) for k in ['pretrain_heads', 'contrastive_head', 'local_refinement', 'multiscale_frontend', 'physical_statistics']):
                raise ValueError('Multi-resolution supervised model does not support legacy refinement/statistics or pretraining/contrastive heads')
            if self.config['classification_pooling'] != 'multiscale_attention_stats':
                raise ValueError('Multi-resolution model requires classification_pooling=multiscale_attention_stats')
            if not self.config['classification_fp32']:
                raise ValueError('Multi-resolution task readouts require classification_fp32=true')
        supplied = model_config.get('position_options', {})
        self.config['position_options'] = {**POSITION_DEFAULTS, **supplied}
        if 'reference_sample_rate' in model_config and 'reference_sample_rate_hz' not in supplied:
            self.config['position_options']['reference_sample_rate_hz'] = model_config['reference_sample_rate']
        self.backbone = (UnifiedBackbone(self.config) if self.hybrid and self.config['hybrid']['topology'] == 'serial' else
                         HybridBackbone(self.config) if self.hybrid else
                         HierarchicalBackbone(self.config) if self.hierarchical else Backbone(self.config))
        self.schema = schema
        d = self.config['width']
        if self.multiresolution:
            readout = self.config['multiscale_readout']
            if self.hybrid:
                branch = self.config['hierarchy']
                widths = [d, branch['stage_widths'][-1], branch['stage_widths'][0]]
                memory_width, memory_heads, attention_dropout = branch['fusion_width'], branch['fusion_heads'], branch['dropout']
            elif self.hierarchical:
                branch = self.config['hierarchy']
                widths, memory_width, memory_heads = branch['stage_widths'], branch['fusion_width'], branch['fusion_heads']
                attention_dropout = branch['dropout']
            else:
                branch = self.config['pyramid']
                self.backbone.pyramid = MultiResolutionPyramid(d, self.config['patch_size'], self.config['rope_base'], branch)
                widths, memory_width, memory_heads = [d, branch['width'], branch['width']], branch['width'], branch['heads']
                attention_dropout = branch['attention_dropout']
            self.heads = nn.ModuleDict({
                'modulation': (DeepAnchoredModulationReadout(widths, len(schema['classification_labels']['modulation']), readout,
                                  self.config['hierarchy']['mechanism']) if self.preserve_signal else
                               ModulationReadout(widths, len(schema['classification_labels']['modulation']), readout)),
                'protocol': ProtocolReadout(widths, len(schema['classification_labels']['protocol']), readout),
                'queries': ParameterQueries(widths, memory_width, memory_heads, attention_dropout,
                    self.config['parameter_query_temperature'], carrier_dim=6 if self.preserve_signal else 0),
                'regression': nn.ModuleList([nn.Sequential(nn.Linear(memory_width * 2 + (6 if self.preserve_signal else 0), readout['parameter_hidden']),
                    nn.GELU(), nn.Linear(readout['parameter_hidden'], 1)) for _ in range(7)])})
            grid_width = memory_width * 2 + (6 if self.preserve_signal else 0)
        else:
            self._legacy_heads(d, schema)
            grid_width = d * 2
        # Timing fields stay continuous even when a smoke subset has few values.
        self.grid_indices = [j for j, values in enumerate(parameter_grids) if j in [0, 1, 2, 6] and 1 < len(values) <= 64]
        self.grids = nn.ModuleDict()
        means, stds = target_scaler['mean'], target_scaler['std']
        for j in self.grid_indices:
            self.register_buffer(f'grid_values_{j}', (torch.tensor(parameter_grids[j]) - means[j]) / stds[j], persistent=False)
            if self.config['parameter_mode'] == 'grid_residual':
                self.grids[str(j)] = nn.Linear(grid_width, len(parameter_grids[j]))
        self.aux = nn.ModuleDict()
        classification_width = d * (2 if self.config['classification_pooling'] == 'mean_std' else 1)
        classification_width += STATISTICS_DIM if self.config['physical_statistics'] else 0
        if self.config['contrastive_head']:
            self.aux['contrastive'] = nn.Sequential(nn.Linear(classification_width, d), nn.GELU(),
                                                  nn.Linear(d, self.config['contrastive_projection_dim']))
        if self.config['pretrain_heads']:
            self.aux['reconstruction'] = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, self.config['patch_size'] * 2))
            for kind in ['atomic', 'emergent']:
                if self.config[f'{kind}_vocab']:
                    self.aux[kind] = nn.Linear(d, self.config[f'{kind}_vocab'])

    def _legacy_heads(self, d, schema):
        # Preserve historical parameter names and construction order.
        classification_width = d * (2 if self.config['classification_pooling'] == 'mean_std' else 1)
        classification_width += STATISTICS_DIM if self.config['physical_statistics'] else 0
        hidden = max(1, int(d * self.config['classification_hidden_ratio']))
        self.heads = nn.ModuleDict({task: nn.Sequential(nn.Linear(classification_width, hidden), nn.GELU(),
                     nn.Linear(hidden, len(schema['classification_labels'][task]))) for task in ['modulation', 'protocol']})
        self.heads['queries'] = nn.Embedding(7, d)
        self.heads['regression'] = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, 1))

    def encode(self, iq, lengths, patch_mask=None, sample_rates=None):
        return self.backbone(iq, lengths, patch_mask, sample_rates)

    def forward(self, iq, lengths, patch_mask=None, return_pretrain=False, sample_rates=None):
        if self.multiresolution:
            if patch_mask is not None or return_pretrain:
                raise ValueError('Multi-resolution model currently supports supervised training/inference only')
            if self.hierarchical or self.hybrid:
                pyramid = self.backbone(iq, lengths, sample_rates=sample_rates, return_pyramid=True)
            else:
                h, valid, _, context = self.backbone(iq, lengths, sample_rates=sample_rates, return_context=True)
                pyramid = self.backbone.pyramid(h, valid, context, lengths)
            # Heads, variance, attention probabilities and scalar losses use FP32.
            with torch.autocast(device_type=iq.device.type, enabled=False):
                output = {task: self.heads[task](pyramid) for task in ['modulation', 'protocol']}
                features = self.heads['queries'](pyramid, lengths)
                reg = torch.cat([head(features[:, j]) for j, head in enumerate(self.heads['regression'])], -1)
                return self._decode_parameters(output, features, reg)
        h, valid, target = self.encode(iq, lengths, patch_mask, sample_rates)
        pooled = h.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        classification_h = h.float() if self.config['classification_fp32'] else h
        classification_mean = classification_h.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        classification_features = classification_mean
        if self.config['classification_pooling'] == 'mean_std':
            variance = ((classification_h - classification_mean[:, None]).square() * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
            classification_features = torch.cat((classification_mean, variance.clamp_min(1e-8).sqrt()), -1)
        if self.config['physical_statistics']:
            student_iq, _, _, _ = self.backbone.preprocess(iq, lengths, patch_mask)
            classification_features = torch.cat((classification_features, signal_statistics(student_iq, lengths)), -1)
        context = torch.autocast(device_type=iq.device.type, enabled=False) if self.config['classification_fp32'] else nullcontext()
        with context:
            output = {task: self.heads[task](classification_features) for task in ['modulation', 'protocol']}
        if 'contrastive' in self.aux:
            output['contrastive_embedding'] = self.aux['contrastive'](classification_features)
        score = torch.einsum('bnd,jd->bjn', h, self.heads['queries'].weight) / (math.sqrt(h.shape[-1]) * self.config['parameter_query_temperature'])
        weights = score.masked_fill(~valid[:, None], -1e4).softmax(-1)
        local = torch.einsum('bjn,bnd->bjd', weights, h)
        features = torch.cat((local, pooled[:, None].expand(-1, 7, -1)), -1)
        reg = self.heads['regression'](features).squeeze(-1)
        output = self._decode_parameters(output, features, reg)
        if return_pretrain:
            output['tokens'], output['token_valid'], output['reconstruction_target'] = h, valid, target
            output['pretrain'] = {name: head(h) for name, head in self.aux.items()}
        return output

    def _decode_parameters(self, output, features, reg):
        output['grid_logits'] = {}
        if self.config['parameter_mode'] == 'grid_residual':
            decoded = []
            for j in range(7):
                if str(j) in self.grids:
                    logits = self.grids[str(j)](features[:, j])
                    output['grid_logits'][str(j)] = logits
                    decoded.append(logits.softmax(-1) @ getattr(self, f'grid_values_{j}') + self.config['grid_residual_scale'] * reg[:, j].tanh())
                else:
                    decoded.append(reg[:, j])
            reg = torch.stack(decoded, -1)
        output['parameters'] = reg
        return output

    def parameter_report(self):
        total = sum(p.numel() for p in self.parameters())
        shared = sum(p.numel() for p in self.backbone.parameters())
        auxiliary = sum(p.numel() for p in self.aux.parameters())
        final_total = total - auxiliary
        return dict(total_training_parameters=total, auxiliary_parameters=auxiliary,
                    final_parameters=final_total, shared_parameters=shared, shared_ratio=shared / final_total)
