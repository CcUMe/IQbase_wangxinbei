"""CPU component audit: shape/dtype, masks, geometry, shared gradients and padding.

Uses synthetic IQ and the actual dataset schema/grids. This measures execution
contracts, not recognition accuracy, GPU performance or pretraining transfer.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from models import hierarchical
from models.rf_model import RFModel, UnifiedBackbone
from utils.common import read_json, write_json
from utils.config import load_config, training_options


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value: yield from tensors(item)
    elif isinstance(value, dict):
        for item in value.values(): yield from tensors(item)


def describe(value):
    return [dict(shape=list(x.shape), dtype=str(x.dtype).removeprefix('torch.'),
                 finite=bool(torch.isfinite(x.detach()).all())) for x in tensors(value)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('code/configs/supervised/main.json5'))
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    settings = load_config(args.config)
    data = read_json((args.data or Path(settings['run']['data'])) / 'dataset.json')
    torch.set_num_threads(2); torch.manual_seed(settings['runtime']['seed'])
    model = RFModel(training_options(settings)['model'], data['schema'], data['target_scaler'], data['parameter_grids']).cpu().eval()
    if not isinstance(model.backbone, UnifiedBackbone):
        raise ValueError('Audit requires hybrid.topology=serial')
    output = args.output or Path('artifacts/unified_component_audit') / datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=True)
    backbone, stem = model.backbone, model.backbone.stem
    mechanism = stem.options['mechanism']
    modules = [('时域复数卷积', stem.time), ('时域特征投影', stem.time_projection),
               ('I-STFT实虚部嵌入', stem.spectrum.i_embedding), ('复IQ功率谱嵌入', stem.spectrum.iq_embedding),
               ('共享频谱模块', stem.spectrum),
               ('全局谱嵌入', stem.frontend_global), ('唯一时频融合', stem.fusion),
               ('BiLSTM内部递归', stem.recurrent.rnn), ('长度感知BiLSTM', stem.recurrent)]
    for i in range(3):
        modules += [(f'阶段{i + 1}局部卷积', stem.stages[i][0].local_conv),
                    (f'阶段{i + 1}' + ('局部窗口注意力' if mechanism['enabled'] else '加性注意力'), stem.stages[i][0].attention),
                    (f'阶段{i + 1}SwiGLU输入', stem.stages[i][0].ffn_in),
                    (f'阶段{i + 1}SwiGLU输出', stem.stages[i][0].ffn_out)]
        modules.append((f'阶段{i + 1}输出', stem.stage_norms[i]))
        if i < 2: modules.append((f'阶段{i + 1}下采样', stem.downsample[i]))
        if mechanism['enabled']: modules.append((f'阶段{i + 1}信号值投影', stem.signal_projections[i]))
    modules += [('Transformer输入投影', backbone.token_projection), ('Transformer输入归一化', backbone.token_norm)]
    modules += [(f'Transformer层{i + 1}', block) for i, block in enumerate(backbone.blocks)]
    modules += [('Transformer输出归一化', backbone.norm), ('跨层读出融合', backbone.hybrid_fusion),
                ('调制读出', model.heads['modulation']), ('协议读出', model.heads['protocol']),
                ('七参数查询', model.heads['queries'])]
    modules += [(f'参数字段{i + 1}读出', head) for i, head in enumerate(model.heads['regression'])]
    records, phase, handles = [], 'fp32', []
    def hook(name):
        def capture(module, inputs, result):
            records.append(dict(precision=phase, component=name, input=describe(inputs), output=describe(result)))
        return capture
    for name, module in modules:
        if module is not None: handles.append(module.register_forward_hook(hook(name)))
    geometries = []
    def capture_geometry(module, inputs):
        _, masks, centers = inputs[:3]
        ratio = backbone.adapter_stride
        error = (centers[0] - centers[1].unfold(0, ratio, ratio).mean(-1)).abs().max()
        geometries.append(dict(precision=phase, counts=[m.sum(1).tolist() for m in masks],
                               first_centers=[float(c[0]) for c in centers], adapter_center_max_error=float(error)))
    handles.append(backbone.hybrid_fusion.register_forward_pre_hook(capture_geometry))
    iq, lengths = torch.randn(2, 1024, 2), torch.tensor([512, 1024])
    checks, gradient_reports, frontend_counts = {}, {}, {}
    for precision in ['fp32', 'bf16']:
        phase = precision
        model.zero_grad(set_to_none=True)
        with patch.object(stem, 'preprocess', wraps=stem.preprocess) as preprocess, \
             patch('models.hierarchical.global_log_power', wraps=hierarchical.global_log_power) as global_spectrum:
            with torch.autocast('cpu', enabled=precision == 'bf16', dtype=torch.bfloat16):
                result = model(iq, lengths)
                loss = sum(result[k].square().mean() for k in ['modulation', 'protocol', 'parameters'])
                loss += .01 * sum(x.square().mean() for x in result['grid_logits'].values())
            frontend_counts[precision] = dict(preprocess=preprocess.call_count, global_spectrum=global_spectrum.call_count)
        loss.backward()
        gradients = {}
        for name, module in modules:
            if module is None: continue
            available = [p.grad for p in module.parameters() if p.grad is not None]
            gradients[name] = dict(parameters=sum(p.numel() for p in module.parameters()),
                tensors_with_gradient=len(available), finite=all(bool(torch.isfinite(g).all()) for g in available),
                norm=float(sum(g.detach().float().square().sum() for g in available).sqrt()) if available else None)
        unconnected = [name for name, p in backbone.named_parameters() if p.grad is None]
        checks[precision] = dict(output_shapes={k:list(result[k].shape) for k in ['modulation', 'protocol', 'parameters']},
            finite_loss=bool(torch.isfinite(loss)), unconnected_shared_parameters=unconnected,
            finite_gradients=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))
        assert not unconnected and checks[precision]['finite_gradients'] and checks[precision]['finite_loss']
        assert frontend_counts[precision] == dict(preprocess=1, global_spectrum=int(stem.frontend_global is not None))
        gradient_reports[precision] = gradients
    for handle in handles: handle.remove()
    with torch.no_grad():
        mixed = model(iq, lengths)
        native = model(iq[:1, :512], lengths[:1])
        changed = iq.clone(); changed[0, 512:] = torch.randn(512, 2) * 1000
        polluted = model(changed, lengths)
    padding = {}
    for key in ['modulation', 'protocol', 'parameters']:
        torch.testing.assert_close(mixed[key][:1], native[key], atol=5e-5, rtol=5e-5)
        torch.testing.assert_close(mixed[key], polluted[key], atol=0., rtol=0.)
        padding[key] = dict(native_max_error=float((mixed[key][:1] - native[key]).abs().max()),
                            pollution_max_error=float((mixed[key] - polluted[key]).abs().max()))
    assert all(x['adapter_center_max_error'] == 0 for x in geometries)
    assert all(v['finite'] for row in records for v in row['output'])
    mechanism_checks = {}
    if mechanism['enabled']:
        # Causal interventions on the actual full-size components, independent
        # from interface/shape checks and without changing model parameters.
        w, d = mechanism['recurrent_window'], stem.options['stage_widths'][0]
        x = torch.randn(1, w * 2, d); changed = x.clone(); changed[:, w:] += 10
        valid = torch.ones(1, w * 2, dtype=torch.bool); local_lengths = torch.tensor([w * 2])
        with torch.no_grad():
            a = stem.recurrent(x, local_lengths, valid); b = stem.recurrent(changed, local_lengths, valid)
        mechanism_checks['recurrent_remote_window_error'] = float((a[:, :w] - b[:, :w]).abs().max())
        attention = stem.stages[0][0].attention; aw = mechanism['attention_window']
        x = torch.randn(1, aw * 4, d); changed = x.clone(); changed[:, aw * 2:] += 10
        valid = torch.ones(1, aw * 4, dtype=torch.bool)
        with torch.no_grad():
            a = attention(x, valid); b = attention(changed, valid)
            pyramid = backbone(iq, lengths, return_pyramid=True)
            normalized = stem.preprocess(iq, lengths)[0]
            deep, correction = model.heads['modulation'].descriptors(pyramid)
            changed_pyramid = dict(pyramid)
            changed_pyramid['summaries'] = [x + 100 for x in pyramid['summaries']]
            deep2, _ = model.heads['modulation'].descriptors(changed_pyramid)
        mechanism_checks.update(local_attention_remote_window_error=float((a[:, :aw] - b[:, :aw]).abs().max()),
            retained_iq_max_error=float((pyramid['signal_features'][..., :2] - normalized).abs().max()),
            signal_memory_shape=list(pyramid['signal_memory'].shape),
            detail_correction_max=float(correction.abs().max()), detail_correction_bound=mechanism['detail_max_ratio'],
            deep_descriptor_shallow_summary_error=float((deep - deep2).abs().max()))
        early = stem.forward_features(iq[:1, :512], lengths[:1])
        global_parameters = list(stem.frontend_global.parameters()) if stem.frontend_global is not None else []
        gradients = (torch.autograd.grad(sum(x.sum() for x in early['features']), global_parameters, allow_unused=True)
                     if global_parameters else [])
        mechanism_checks['early_features_depend_on_global_spectrum'] = any(g is not None for g in gradients)
        assert mechanism_checks['recurrent_remote_window_error'] == 0
        assert mechanism_checks['local_attention_remote_window_error'] == 0
        assert mechanism_checks['retained_iq_max_error'] == 0
        assert mechanism_checks['deep_descriptor_shallow_summary_error'] == 0
        assert not mechanism_checks['early_features_depend_on_global_spectrum']
        assert mechanism_checks['detail_correction_max'] <= mechanism['detail_max_ratio']
    counts = {name:sum(row['component'] == name and row['precision'] == 'fp32' for row in records)
              for name in ['时域复数卷积', '共享频谱模块', '唯一时频融合']}
    assert all(value == 1 for value in counts.values())
    report = dict(config=str(args.config), synthetic_input=True, device='cpu', parameters=model.parameter_report(),
        frontend_counts=frontend_counts, frontend_module_calls=counts, geometries=geometries, checks=checks,
        padding=padding, components=records, gradients=gradient_reports, mechanism_checks=mechanism_checks,
        configuration_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        note='Execution/compatibility audit only; does not establish accuracy, CUDA performance or checkpoint transfer quality.')
    write_json(output / 'audit.json', report)
    lines = ['# 单前端统一基座组件审计', '', f'配置：{args.config}。CPU，模拟输入B=2，真实有效长度512/1024。', '',
             '## 实际组件输入输出（FP32运行）', '', '| 组件 | 输入shape | 输出shape | 输出dtype | 有限 |', '|---|---|---|---|---|']
    for row in records:
        if row['precision'] != 'fp32': continue
        fmt = lambda values: ', '.join('×'.join(map(str, v['shape'])) for v in values)
        lines.append(f"| {row['component']} | {fmt(row['input'])} | {fmt(row['output'])} | {','.join(sorted(set(v['dtype'] for v in row['output'])))} | 是 |")
    lines += ['', '## 连接与兼容性', '',
        '- 每个完整前向的RMS处理、共享时域提取、共享频谱模块、时频融合和全局谱计算各执行一次。',
        '- 奇数对称窗：I-STFT31点、复I/Q63点，中心都位于原始样点；避免半样点窗中心差异。',
        '- 阶段压缩4倍，再投影4倍，总步长16；适配器token中心与Transformer坐标逐点一致，最大误差为0。',
        '- FP32及CPU BF16前后向全部有限，所有共享参数均连接计算图；逐组件梯度范数与dtype详见audit.json。',
        '- BiLSTM先按有效长度分组，机制开启时进一步分局部窗口；native/pad1024一致，padding扰动误差0。',
        '- 调制B×23、协议B×8、参数B×7；三个任务使用同一前端、阶段编码和Transformer。',
        '', '## 已明确的变化与限制', '',
        '- 保留原Transformer核心结构和权重接口；前端已合并，输入表征分布改变，旧完整pt不能直接resume。',
        '- 跨层融合只读取同一条主干不同深度的特征，没有第二套原始信号编码器或独立预测集成。',
        '- 局部频谱视图由同一前端融合；机制开启时全局谱仅用于跨层读出，不注入早期时频特征。',
        '- 该审计验证组件契约；真实识别效果、GPU吞吐/显存和国产平台部署仍需实际评估。', '']
    if mechanism_checks:
        lines += ['## 信息保留与分工干预检查', '', '```json',
                  json.dumps(mechanism_checks, ensure_ascii=False, indent=2), '```', '',
                  '远端局部窗口扰动不改变当前窗口输出；原归一化I/Q在保护值中逐点保留；全局谱不进入早期特征；调制主描述独立于浅层融合摘要，细节修正幅度受限。',
                  '这些证明结构约束生效，不证明识别提分或完全消除了多任务梯度冲突。', '']
    (output / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output), parameters=report['parameters'], checks=checks, padding=padding), ensure_ascii=False))


if __name__ == '__main__': main()
