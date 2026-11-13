"""High-resolution time/frequency fusion, BiLSTM and shared staged encoding.

Inspired by IQFormer; explicit I-STFT real/imaginary and complex-IQ power
views, masked LayerNorm and staged downsampling are C01 adaptations.
"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.multiresolution import ComplexConv1d, CrossAttention, masked_mean
from models.signal_mechanisms import LocalWindowAttention, signal_values, fold_values


HIERARCHY_DEFAULTS = dict(
    enabled=False, stage_widths=[96, 192, 384], stage_depths=[2, 3, 3],
    downsample_strides=[2, 2], stem_complex_channels=32, conv_kernel=5,
    lstm_hidden=48, lstm_layers=2, lstm_precision='fp32', mlp_ratio=4.0,
    fusion_width=192, fusion_heads=3, dropout=0.1, residual_scale=0.1,
    spectrum=dict(i_window=31, i_nfft=128, i_bins=32, iq_window=64, centered_windows=False),
    mechanism=dict(enabled=False, recurrent_window=128, attention_window=16, signal_gain=0.1,
                   detail_max_ratio=0.25, detail_initial_ratio=0.1))
HYBRID_DEFAULTS = dict(enabled=False, topology='parallel')


def global_log_power(iq, lengths, backend):
    """One full-window spectrum, shared by the frontend when using serial mode."""
    with torch.autocast(device_type=iq.device.type, enabled=False):
        x = iq.float()
        if backend == 'fft':
            spectrum = torch.fft.fft(torch.complex(x[..., 0], x[..., 1]), n=1024)
            power = spectrum.abs().square()
        else:
            coordinate = torch.arange(1024, device=x.device).float()
            phase = 2 * math.pi * coordinate[:, None] * coordinate[None] / 1024
            source = F.pad(x, (0, 0, 0, 1024 - x.shape[1]))
            real = source[..., 0] @ phase.cos().T + source[..., 1] @ phase.sin().T
            imag = source[..., 1] @ phase.cos().T - source[..., 0] @ phase.sin().T
            power = real.square() + imag.square()
        return torch.log1p(torch.roll(power, 512, -1) / lengths[:, None])


def hierarchy_options(options=None):
    result = deepcopy(HIERARCHY_DEFAULTS)
    for key, value in (options or {}).items():
        if key not in result:
            raise ValueError(f'Unknown hierarchy option: {key}')
        if isinstance(result[key], dict):
            if not isinstance(value, dict) or set(value) - set(result[key]):
                raise ValueError(f'Invalid hierarchy.{key}')
            result[key].update(value)
        else:
            result[key] = value
    return result


def validate_hierarchy(options, readout=None):
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(type(options['enabled']) is bool, 'hierarchy.enabled must be boolean')
    widths, depths, strides = (options[k] for k in ['stage_widths', 'stage_depths', 'downsample_strides'])
    require(isinstance(widths, list) and len(widths) == 3 and all(type(w) is int and w > 0 for w in widths),
            'hierarchy.stage_widths must contain three positive integers')
    require(isinstance(depths, list) and len(depths) == 3 and all(type(d) is int and d > 0 for d in depths),
            'hierarchy.stage_depths must contain three positive integers')
    require(isinstance(strides, list) and len(strides) == 2 and all(type(s) is int and s > 1 and s % 2 == 0 for s in strides),
            'hierarchy.downsample_strides must contain two positive even strides >=2')
    require(512 % math.prod(strides) == 0, 'hierarchy cumulative downsampling must divide 512')
    for key in ['stem_complex_channels', 'conv_kernel', 'lstm_hidden', 'lstm_layers', 'fusion_width', 'fusion_heads']:
        require(type(options[key]) is int and options[key] > 0, f'hierarchy.{key} must be a positive integer')
    require(options['conv_kernel'] % 2 == 1, 'hierarchy.conv_kernel must be odd')
    require(options['lstm_precision'] in ['fp32', 'autocast'], 'hierarchy.lstm_precision must be fp32 or autocast')
    require(options['fusion_width'] % options['fusion_heads'] == 0 and
            (options['fusion_width'] // options['fusion_heads']) % 2 == 0, 'hierarchy fusion head width must be even')
    require(math.isfinite(options['mlp_ratio']) and options['mlp_ratio'] > 0, 'hierarchy.mlp_ratio must be positive')
    require(math.isfinite(options['residual_scale']) and options['residual_scale'] > 0, 'hierarchy.residual_scale must be positive')
    require(math.isfinite(options['dropout']) and 0 <= options['dropout'] < 1, 'hierarchy.dropout must be in [0,1)')
    spectrum = options['spectrum']
    require(all(type(spectrum[k]) is int and spectrum[k] > 0 for k in ['i_window', 'i_nfft', 'i_bins', 'iq_window']),
            'hierarchy spectrum sizes must be positive integers')
    require(type(spectrum['centered_windows']) is bool, 'spectrum.centered_windows must be boolean')
    require(not spectrum['centered_windows'] or spectrum['iq_window'] % 2 == 1,
            'Centered shared spectra require an odd IQ window')
    require(spectrum['i_window'] % 2 == 1 and spectrum['i_window'] <= spectrum['i_nfft'],
            'hierarchy I-STFT needs an odd window <= i_nfft')
    require(spectrum['i_bins'] <= spectrum['i_nfft'] // 2 + 1, 'hierarchy.i_bins exceeds one-sided frequency bins')
    mechanism = options['mechanism']
    require(type(mechanism['enabled']) is bool, 'mechanism.enabled must be boolean')
    require(type(mechanism['recurrent_window']) is int and mechanism['recurrent_window'] > 0
            and 512 % mechanism['recurrent_window'] == 0, 'recurrent_window must be a positive divisor of 512')
    require(type(mechanism['attention_window']) is int and mechanism['attention_window'] > 1,
            'attention_window must be an integer >1')
    require(0 < mechanism['signal_gain'] <= 1, 'mechanism.signal_gain must be in (0,1]')
    require(0 < mechanism['detail_initial_ratio'] < mechanism['detail_max_ratio'] <= 1,
            'Detail correction requires 0 < initial_ratio < max_ratio <= 1')
    if readout is not None:
        for key in ['projection_width', 'modulation_hidden', 'protocol_hidden', 'parameter_hidden']:
            require(type(readout[key]) is int and readout[key] > 0, f'multiscale_readout.{key} must be positive integer')
        require(math.isfinite(readout['dropout']) and 0 <= readout['dropout'] < 1, 'multiscale_readout.dropout must be in [0,1)')
        require(math.isfinite(readout['variance_epsilon']) and readout['variance_epsilon'] > 0,
                'multiscale_readout.variance_epsilon must be positive')


class HighResolutionSpectrum(nn.Module):
    """Hop-one I-STFT real/imaginary plus signed complex-IQ log power."""
    def __init__(self, options, width):
        super().__init__()
        self.options = options
        iw, bins, nfft = options['i_window'], options['i_bins'], options['i_nfft']
        angle = 2 * math.pi * torch.arange(bins).float()[:, None] * torch.arange(iw).float()[None] / nfft
        window = torch.blackman_window(iw, periodic=not options['centered_windows'])
        self.register_buffer('i_kernels', torch.cat((angle.cos(), -angle.sin()))[:, None] * window, persistent=False)
        self.register_buffer('i_scale', window.square().sum().sqrt(), persistent=False)
        w = options['iq_window']
        angle = 2 * math.pi * torch.arange(w).float()[:, None] * torch.arange(w).float()[None] / w
        window = torch.hann_window(w, periodic=not options['centered_windows'])
        real = torch.stack((angle.cos() * window, angle.sin() * window), 1)
        imag = torch.stack((-angle.sin() * window, angle.cos() * window), 1)
        self.register_buffer('iq_kernels', torch.cat((real, imag)), persistent=False)
        self.register_buffer('iq_power', window.square().sum(), persistent=False)
        # Both views have independent embeddings, including for odd model widths.
        left = max(1, width // 2)
        right = max(1, width - left)
        self.i_embedding = nn.Sequential(nn.Linear(bins * 2, left), nn.LayerNorm(left), nn.GELU())
        self.iq_embedding = nn.Sequential(nn.Linear(w, right), nn.LayerNorm(right), nn.GELU())
        self.mix = nn.Sequential(nn.Linear(left + right, width), nn.LayerNorm(width))

    def forward(self, iq, valid):
        with torch.autocast(device_type=iq.device.type, enabled=False):
            source = iq.float().transpose(1, 2)
            i_view = F.conv1d(source[:, :1], self.i_kernels.float(), padding=self.options['i_window'] // 2)
            i_view = (i_view / self.i_scale).transpose(1, 2)
            # A fixed zero-padding convention keeps one frame per original
            # sample for both odd/even windows, independent of batch padding.
            w = self.options['iq_window']
            padded = F.pad(source, ((w - 1) // 2, w // 2))
            real, imag = F.conv1d(padded, self.iq_kernels.float()).chunk(2, 1)
            power = torch.log1p((real.square() + imag.square()) / self.iq_power).transpose(1, 2)
        return self.mix(torch.cat((self.i_embedding(i_view), self.iq_embedding(power)), -1)) * valid[..., None]


class LengthAwareBiLSTM(nn.Module):
    """Native-length buckets avoid reverse recurrence through padded samples."""
    def __init__(self, width, hidden, layers, dropout, precision, window_size=0, residual_scale=0.1):
        super().__init__()
        self.precision = precision
        self.window_size = window_size
        self.rnn = nn.LSTM(width, hidden, num_layers=layers, bidirectional=True,
                           batch_first=True, dropout=dropout if layers > 1 else 0.)
        self.projection = nn.Linear(hidden * 2, width)
        self.norm = nn.Identity() if window_size else nn.LayerNorm(width)
        self.input_norm = nn.LayerNorm(width) if window_size else nn.Identity()
        self.register_parameter('residual_gain', nn.Parameter(torch.full((width,), residual_scale)) if window_size else None)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths, valid):
        groups, indices = [], []
        # Training is already homogeneous by length; validation can mix lengths.
        # No packed recurrent kernels or CUDA-only extension is required.
        with torch.autocast(device_type=x.device.type,
                            enabled=self.precision != 'fp32' and torch.is_autocast_enabled(x.device.type),
                            dtype=torch.get_autocast_dtype(x.device.type)):
            source = x.float() if self.precision == 'fp32' else x
            source = self.input_norm(source)
            for length in lengths.unique(sorted=True).tolist():
                index = torch.nonzero(lengths == length, as_tuple=True)[0]
                sequence = source.index_select(0, index)[:, :length].contiguous()
                if self.window_size:
                    if length % self.window_size:
                        raise ValueError('Local recurrent windows must divide valid length')
                    sequence = sequence.reshape(-1, self.window_size, sequence.shape[-1])
                result, _ = self.rnn(sequence)
                if self.window_size:
                    result = result.reshape(len(index), length, -1)
                groups.append(F.pad(result, (0, 0, 0, x.shape[1] - length)))
                indices.append(index)
            order = torch.cat(indices).argsort()
            result = torch.cat(groups).index_select(0, order)
        update = self.dropout(self.projection(result))
        if self.residual_gain is not None: update = update * self.residual_gain
        return self.norm(x + update) * valid[..., None]


class EfficientAdditiveAttention(nn.Module):
    """IQFormer-style global aggregation, linear in the number of samples."""
    def __init__(self, width):
        super().__init__()
        self.query, self.key = nn.Linear(width, width), nn.Linear(width, width)
        self.global_weight = nn.Parameter(torch.randn(width, 1) / math.sqrt(width))
        self.projection, self.out = nn.Linear(width, width), nn.Linear(width, width)

    def forward(self, x, valid):
        query, key = self.query(x), self.key(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = F.normalize(query.float(), dim=-1, eps=1e-6)
            k = F.normalize(key.float(), dim=-1, eps=1e-6)
            weight = (q @ self.global_weight.float()).squeeze(-1) / math.sqrt(x.shape[-1])
            weight = F.normalize(weight * valid, dim=1, eps=1e-6)
            global_query = (weight[..., None] * q).sum(1, keepdim=True)
            interaction = global_query * k
        return self.out(self.projection(interaction.to(query.dtype)) + q.to(query.dtype)) * valid[..., None]


class HierarchicalBlock(nn.Module):
    def __init__(self, width, options, shift=0):
        super().__init__()
        self.local_norm, self.attention_norm, self.ffn_norm = [nn.LayerNorm(width) for _ in range(3)]
        self.local_conv = nn.Conv1d(width, width, options['conv_kernel'], padding=options['conv_kernel'] // 2, groups=width)
        self.local_in, self.local_out = nn.Linear(width, width * 2), nn.Linear(width * 2, width)
        mechanism = options['mechanism']
        self.attention = (LocalWindowAttention(width, mechanism['attention_window'], shift, options['dropout'])
                          if mechanism['enabled'] else EfficientAdditiveAttention(width))
        hidden = max(8, int(width * options['mlp_ratio'] * 2 / 3 / 8) * 8)
        self.ffn_in, self.ffn_out = nn.Linear(width, hidden * 2), nn.Linear(hidden, width)
        self.scales = nn.Parameter(torch.full((3, width), options['residual_scale']))
        self.dropout = nn.Dropout(options['dropout'])

    def forward(self, x, valid):
        h = self.local_norm(x) * valid[..., None]
        h = self.local_conv(h.transpose(1, 2)).transpose(1, 2) * valid[..., None]
        x = x + self.dropout(self.local_out(F.gelu(self.local_in(h)))) * self.scales[0]
        x = x + self.dropout(self.attention(self.attention_norm(x), valid)) * self.scales[1]
        gate, value = self.ffn_in(self.ffn_norm(x)).chunk(2, -1)
        x = x + self.dropout(self.ffn_out(self.dropout(F.silu(gate) * value))) * self.scales[2]
        return x * valid[..., None]


class SharedScaleFusion(nn.Module):
    def __init__(self, widths, options):
        super().__init__()
        d = options['fusion_width']
        self.memory_projection = nn.ModuleList([nn.Linear(w, d) for w in widths])
        self.position = nn.Sequential(nn.Linear(2, d), nn.Tanh())
        self.scale_embedding = nn.Embedding(3, d)
        self.cross_scale = nn.ModuleList([CrossAttention(d, options['fusion_heads'], options['dropout']) for _ in widths])
        self.summary_out = nn.ModuleList([nn.Linear(d, w) for w in widths])
        self.scales = nn.Parameter(torch.full((3, d), options['residual_scale']))

    def forward(self, features, masks, centers, lengths, scale, global_features=None):
        memories = []
        for i, (projection, h, valid, center) in enumerate(zip(self.memory_projection, features, masks, centers)):
            coordinate = torch.stack((center[None].expand(len(lengths), -1) / 1024,
                                      lengths[:, None].expand(-1, len(center)).float() / 1024), -1)
            memory = projection(h) + self.position(coordinate) + self.scale_embedding.weight[i]
            if global_features is not None:
                memory = memory + global_features[:, None]
            memories.append(memory * valid[..., None])
        summaries = [masked_mean(memory, valid) for memory, valid in zip(memories, masks)]
        fused = []
        for i in range(3):
            other = [j for j in range(3) if j != i]
            memory = torch.cat([summaries[i][:, None]] + [memories[j] for j in other], 1)
            valid = torch.cat([torch.ones_like(masks[i][:, :1])] + [masks[j] for j in other], 1)
            summary = summaries[i] + self.cross_scale[i](summaries[i][:, None], memory, valid).squeeze(1) * self.scales[i]
            fused.append(self.summary_out[i](summary))
        return dict(features=features, valid=masks, summaries=fused, memory=torch.cat(memories, 1),
                    memory_valid=torch.cat(masks, 1), scale=scale)


class HierarchicalBackbone(nn.Module):
    def __init__(self, config, with_readout=True, with_frontend_global=False):
        super().__init__()
        self.config, self.options = config, config['hierarchy']
        options, widths = self.options, self.options['stage_widths']
        mechanism = options['mechanism']
        self.preserve_signal = mechanism['enabled']
        first = widths[0]
        self.patch_size = config['patch_size']  # Cache contract only; does not compress the input.
        self.time = ComplexConv1d(1, options['stem_complex_channels'], options['conv_kernel'], padding=options['conv_kernel'] // 2)
        self.time_projection = (nn.Sequential(nn.Linear(options['stem_complex_channels'] * 2, first), nn.GELU())
                                if self.preserve_signal else nn.Sequential(nn.Linear(options['stem_complex_channels'] * 2, first), nn.LayerNorm(first), nn.GELU()))
        self.spectrum = HighResolutionSpectrum(options['spectrum'], first)
        self.fusion = nn.Sequential(nn.Linear(first * 2, first * 2), nn.GELU(), nn.Linear(first * 2, first))
        self.gate = nn.Linear(first * 2, first)
        self.fusion_norm = nn.Identity() if self.preserve_signal else nn.LayerNorm(first)
        self.fusion_input_norm = nn.LayerNorm(first * 2) if self.preserve_signal else nn.Identity()
        self.register_parameter('fusion_gain', nn.Parameter(torch.full((first,), options['residual_scale'])) if self.preserve_signal else None)
        self.scale_projection = None if self.preserve_signal else nn.Sequential(nn.Linear(1, first), nn.Tanh())
        self.recurrent = LengthAwareBiLSTM(first, options['lstm_hidden'], options['lstm_layers'], options['dropout'], options['lstm_precision'],
            mechanism['recurrent_window'] if self.preserve_signal else 0, options['residual_scale'])
        self.stages = nn.ModuleList([nn.ModuleList([HierarchicalBlock(w, options,
            mechanism['attention_window'] // 2 if j % 2 and self.preserve_signal else 0) for j in range(depth)])
                                    for w, depth in zip(widths, options['stage_depths'])])
        self.stage_norms = nn.ModuleList([nn.Identity() if self.preserve_signal else nn.LayerNorm(w) for w in widths])
        if self.preserve_signal:
            factors = [1, options['downsample_strides'][0], math.prod(options['downsample_strides'])]
            self.signal_projections = nn.ModuleList([nn.Linear(6 * factor, w, bias=False) for factor, w in zip(factors, widths)])
            self.signal_gains = nn.ParameterList([nn.Parameter(torch.full((w,), mechanism['signal_gain'])) for w in widths])
        self.position = nn.ModuleList([nn.Sequential(nn.Linear(2, w), nn.Tanh()) for w in widths])
        self.downsample = nn.ModuleList([nn.Conv1d(widths[i], widths[i + 1], 2 * stride, stride=stride, padding=stride // 2)
                                         for i, stride in enumerate(options['downsample_strides'])])
        self.readout_fusion = SharedScaleFusion(widths, options) if with_readout else None
        self.global_spectrum = (nn.Sequential(nn.Linear(1024, options['fusion_width']), nn.GELU(),
            nn.Linear(options['fusion_width'], options['fusion_width'])) if with_readout and config['global_spectrum'] else None)
        global_width = options['fusion_width'] if self.preserve_signal else first
        self.frontend_global = (nn.Sequential(nn.Linear(1024, global_width), nn.GELU(), nn.Linear(global_width, global_width), nn.LayerNorm(global_width))
                                if with_frontend_global and config['global_spectrum'] else None)
        self.register_parameter('global_context_scale', nn.Parameter(torch.full((global_width,), options['residual_scale']))
                                if self.frontend_global is not None else None)

    @property
    def blocks(self):
        # FineTuneController's full-mode accounting; registration stays in stages.
        return tuple(block for stage in self.stages for block in stage)

    @property
    def norm(self):
        return self.stage_norms[-1]

    def preprocess(self, iq, lengths, patch_mask=None):
        if patch_mask is not None:
            raise ValueError('Hierarchical backbone currently supports supervised training only')
        valid = torch.arange(iq.shape[1], device=iq.device)[None] < lengths[:, None]
        masked = iq.float() * valid[..., None]
        scale = (masked.square().sum((1, 2)) / lengths.clamp_min(1)).sqrt().clamp_min(self.config['rms_epsilon'])
        x = masked / scale[:, None, None]
        return x, x, scale, valid

    def forward_features(self, iq, lengths, patch_mask=None):
        x, target, scale, valid = self.preprocess(iq, lengths, patch_mask)
        protected = signal_values(x, valid, self.config['rms_epsilon']) if self.preserve_signal else None
        time = self.time_projection(self.time(x.transpose(1, 2)).transpose(1, 2)) * valid[..., None]
        frequency = self.spectrum(x, valid)
        late_global = None
        if self.frontend_global is not None:
            descriptor = self.frontend_global(global_log_power(x, lengths, self.config['spectral_backend']))
            if self.preserve_signal:
                late_global = descriptor * self.global_context_scale
            else:
                frequency = (frequency + descriptor[:, None] * self.global_context_scale) * valid[..., None]
        joint = self.fusion_input_norm(torch.cat((time, frequency), -1))
        update = self.gate(joint).sigmoid() * self.fusion(joint)
        if self.fusion_gain is not None: update = update * self.fusion_gain
        h = self.fusion_norm(time + update)
        if self.scale_projection is not None: h = h + self.scale_projection(scale.log()[:, None])[:, None]
        h = h * valid[..., None]
        h = self.recurrent(h, lengths, valid)
        features, masks, centers = [], [], []
        center = torch.arange(h.shape[1], device=x.device).float()
        current_lengths, sample_step = lengths, 1
        for i, (stage, norm) in enumerate(zip(self.stages, self.stage_norms)):
            coordinate = torch.stack((center[None].expand(len(lengths), -1) / 1024,
                                      lengths[:, None].expand(-1, len(center)).float() / 1024), -1)
            h = (h + self.position[i](coordinate)) * valid[..., None]
            for block in stage:
                h = block(h, valid)
            h = norm(h) * valid[..., None]
            if protected is not None:
                h = (h + self.signal_projections[i](fold_values(protected, sample_step, flatten=True)) * self.signal_gains[i]) * valid[..., None]
            features.append(h)
            masks.append(valid)
            centers.append(center)
            if i < len(self.downsample):
                stride = self.options['downsample_strides'][i]
                h = self.downsample[i](h.transpose(1, 2)).transpose(1, 2)
                current_lengths = current_lengths // stride
                valid = torch.arange(h.shape[1], device=x.device)[None] < current_lengths[:, None]
                offset = center[0] + (stride - 1) / 2 * sample_step
                sample_step *= stride
                center = torch.arange(h.shape[1], device=x.device).float() * sample_step + offset
                h = h * valid[..., None]
        return dict(features=features, valid=masks, centers=centers, scale=scale, target=target, iq=x,
                    signal_features=protected, late_global=late_global)

    def forward(self, iq, lengths, patch_mask=None, sample_rates=None, return_pyramid=False):
        detail = self.forward_features(iq, lengths, patch_mask)
        if not return_pyramid:
            return detail['features'][-1], detail['valid'][-1], detail['target']
        if self.readout_fusion is None:
            raise ValueError('Headless detail encoder must be read by the shared hybrid fusion')
        x, scale = detail['iq'], detail['scale']
        global_features = None
        if self.global_spectrum is not None:
            global_features = self.global_spectrum(global_log_power(x, lengths, self.config['spectral_backend']))
        return self.readout_fusion(detail['features'], detail['valid'], detail['centers'], lengths, scale, global_features)
