"""Rank-zero training overview and per-epoch optimizer-step progress."""
from collections import defaultdict
import os
import sys
import time
import numpy as np

import torch
from tqdm import tqdm
from utils.metrics import classification_metrics


def classification_summary(metrics, details=True):
    for task, values in metrics['classification'].items():
        m = classification_metrics(values['confusion_matrix'], values.get('labels'))
        name = {'modulation': '调制识别', 'protocol': '协议识别'}.get(task, task)
        print(f"  {name} | Accuracy={m['accuracy']:.2%} | Macro-F1={m['macro_f1']:.2%}"
              f" | Weighted-F1={m['weighted_f1']:.2%} | 样本数={m['support']:,}", flush=True)
        if details:
            print(f"    {'类别':<34} {'Recall/类准确率':>16} {'Precision':>11} {'F1':>11} {'Support':>9}", flush=True)
            for label, recall, precision, f1, support in zip(m['labels'], m['recall'], m['precision'], m['f1'], m['class_support']):
                print(f'    {label:<34} {recall:>15.2%} {precision:>11.2%} {f1:>11.2%} {support:>9,}', flush=True)


def training_overview(settings, args, device, world, train_rows, val_rows, info, report,
                      steps, warmup, cap, global_step, epoch_start, codebook):
    model, optimizer = settings['model'], settings['optimizer']
    training, data, runtime = settings['training'], settings['data'], settings['runtime']
    stage = '多任务监督训练'
    if training['stage'] == 'pretrain':
        weights = settings['loss']['pretraining']['weights']
        structural = bool(codebook and (weights['atomic'] > 0 or weights['emergent'] > 0))
        reconstruction = weights['reconstruction'] > 0 or weights['spectrum'] > 0
        stage = '结构 + 重建预训练' if structural and reconstruction else ('纯结构预训练' if structural else '重建预训练')
    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    total = training['epochs'] * steps
    effective = data['batch_size'] * world * training['gradient_accumulation_steps']
    lines = [
        '=' * 88,
        f"训练总览 | {settings['experiment']['name']} | {stage}",
        f"说明       : {settings['experiment']['description'] or '-'}",
        f"设备       : {device_name} | {world} 个进程 | 精度 {runtime['precision']} | seed {runtime['seed']}",
    ]
    if device.type == 'cuda':
        lines.append(f"可见 GPU   : {os.environ.get('CUDA_VISIBLE_DEVICES', '(未显式设置)')}")
    hierarchy = model.get('hierarchy', {})
    hybrid = model.get('hybrid', {}).get('enabled', False)
    model_line = (f"模型       : 分阶段共享编码 | widths={hierarchy['stage_widths']}"
                  f" depths={hierarchy['stage_depths']} | 高效加性注意力 | 参数头={model['heads']['parameter_mode']}"
                  if hierarchy.get('enabled') and not hybrid else
                  f"模型       : width={model['encoder']['width']} depth={model['encoder']['depth']} heads={model['encoder']['heads']}"
                  f" | patch={model['signal']['patch_size']} | 参数头={model['heads']['parameter_mode']}"
                  f" | QK norm={model['encoder']['qk_normalization']} | fusion norm={model['signal']['normalize_fusion']}")
    lines.extend([
        f"数据       : 训练 {train_rows:,} 条 | 验证 {val_rows:,} 条 | 采样 {data['sampling']['mode']}",
        '原始任务数 : ' + ' | '.join(f'{name}={count:,}' for name, count in info['task_counts']['train'].items()),
        model_line,
        f"模型参数   : 最终 {report['final_parameters'] / 1e6:.3f}M | 共享 {report['shared_ratio']:.2%}"
        f" | 训练辅助参数 {report['auxiliary_parameters'] / 1e6:.3f}M",
        f"表征结构   : pooling={model['heads']['classification_pooling']}"
        f" | physical_statistics={model['heads']['physical_statistics']}"
        f" | local_refinement={model['signal']['local_refinement']}",
        f"位置编码   : {model['signal']['position']}"
        f" | options={model['signal']['position_options']}",
        f"Batch      : 每卡微批次 {data['batch_size']} × {world} 卡 × 累积 {training['gradient_accumulation_steps']}"
        f" = 有效全局 {effective}",
        f"训练预算   : {training['epochs']} epochs × {steps:,} steps/epoch = {total:,} 次参数更新",
        f"当前起点   : epoch {epoch_start + 1} | 已完成 {global_step:,} 次更新"
        f" | 本次停止上限 {min(cap, total) if cap else total:,}",
        f"优化器     : AdamW | 头部峰值 LR={optimizer['learning_rate']:.2e}"
        f" | 基座峰值 LR={optimizer['learning_rate'] * optimizer['backbone_lr_factor']:.2e}"
        f" | weight_decay={optimizer['weight_decay']:g}",
        f"学习率     : {settings['scheduler']['name']} | 预热 {warmup:,} steps"
        f" | 最低比例 {settings['scheduler']['min_lr_ratio']:g}",
    ])
    pyramid = model.get('pyramid', {})
    if hybrid and model['hybrid']['topology'] == 'serial':
        mechanism = hierarchy['mechanism']
        a, b = hierarchy['downsample_strides']
        strides = [model['signal']['patch_size'], a * b, 1]
        lines.append('统一基座   : 一次时频提取/融合 → BiLSTM → 阶段编码 → token投影 → 原Transformer核心')
        lines.append(f"前半段     : widths={hierarchy['stage_widths']} depths={hierarchy['stage_depths']}"
                     f" | BiLSTM {hierarchy['lstm_layers']}层 hidden={hierarchy['lstm_hidden']} / {hierarchy['lstm_precision']}")
        lines.append(f"投影衔接   : 阶段累计stride={a * b} → adapter_stride={model['signal']['patch_size'] // (a * b)}"
                     f" → Transformer累计stride={model['signal']['patch_size']}")
        lines.append('跨层Token  : ' + ' | '.join(f'L={length}: {"/".join(str(length // stride) for stride in strides)}'
                                                  for length in [512, 1024]) + '（Transformer末层/阶段末层/阶段首层）')
        lines.append(f"频谱对齐   : I窗{hierarchy['spectrum']['i_window']} / 复IQ窗{hierarchy['spectrum']['iq_window']}"
                     + ('，对称居中；全局谱后移至跨层读出' if mechanism['enabled'] else '，对称居中；全局谱并入同一前端')
                     + ' | 三任务共享完整主干')
        if mechanism['enabled']:
            lines.append(f"机制分工   : BiLSTM局部窗={mechanism['recurrent_window']}原样点"
                         f" | 阶段注意力窗={mechanism['attention_window']} tokens | 全局关系由Transformer建模")
            lines.append('信号保留   : 原归一化I/Q、log功率/变化、差分相位实虚分量；绕过逐token归一化进入任务读出')
            lines.append(f"调制读出   : 深层描述＋有界细节修正 | 初始={mechanism['detail_initial_ratio']:g}"
                         f" 上限={mechanism['detail_max_ratio']:g}（逐维幅度，非logit比例）")
    elif hybrid:
        a, b = hierarchy['downsample_strides']
        strides = [model['signal']['patch_size'], a * b, 1]
        lines.append(f"组合基座   : 保留原P{model['signal']['patch_size']} Transformer主干；共享细节支路"
                     f" widths={hierarchy['stage_widths']} depths={hierarchy['stage_depths']}")
        lines.append(f"连续时序   : 支路融合后 {hierarchy['lstm_layers']} 层BiLSTM"
                     f" | 每方向hidden={hierarchy['lstm_hidden']} | {hierarchy['lstm_precision']}")
        lines.append('融合Token  : ' + ' | '.join(f'L={length}: {"/".join(str(length // stride) for stride in strides)}'
                                                  for length in [512, 1024]) + '（原主干/支路上下文/支路细节）')
        lines.append('任务读出   : 同一共享基座统一融合 | 单个调制分类器 / 协议头 / 七参数查询头')
    elif hierarchy.get('enabled'):
        a, b = hierarchy['downsample_strides']
        strides = [1, a, a * b]
        lines.append(f"连续时序   : 融合后 {hierarchy['lstm_layers']} 层BiLSTM | 每方向hidden={hierarchy['lstm_hidden']}"
                     f" | recurrent_precision={hierarchy['lstm_precision']} | 首次下采样前保留完整有效长度")
        lines.append('有效Token  : ' + ' | '.join(f'L={length}: {"/".join(str(length // stride) for stride in strides)}'
                                                  for length in [512, 1024]) + '（高/中/低分辨率）')
        lines.append('频谱视图   : I-STFT实部＋虚部 / 完整复I/Q局部log功率谱 | hop=1 | 全局谱='
                     + str(model['signal']['global_spectrum']))
        lines.append('任务读出   : 三阶段特征＋跨尺度摘要 | 一个23类调制头 / 8类协议头 / 七查询＋七独立参数MLP')
    elif pyramid.get('enabled'):
        strides = [model['signal']['patch_size'], pyramid['mid']['stride'], pyramid['fine']['stride']]
        lines.append(f"多尺度基座 : 复数stem={pyramid['stem_complex_channels']}复通道"
                     f" | 分支width={pyramid['width']} | local={pyramid['local_blocks']} / Transformer={pyramid['transformer_depth']}")
        lines.append('有效Token  : ' + ' | '.join(f'L={length}: {"/".join(str(length // stride) for stride in strides)}'
                                                  for length in [512, 1024]) + '（主干/中/细）')
        lines.append('任务读出   : 调制=三尺度注意力统计 | 协议=融合摘要 | 参数=七查询＋七独立MLP | 单模型共享权重')
    if training['stage'] == 'pretrain':
        pretrain = settings['loss']['pretraining']
        weights = pretrain['weights']
        active = {k: v for k, v in weights.items() if v > 0 and (codebook or k not in ['atomic', 'emergent'])}
        lines.append(f"预训练目标 : {active} | mask={pretrain['mask']['ratio']:.0%}"
                     f" span={pretrain['mask']['span']} strategy={pretrain['mask']['strategy']}")
        if codebook:
            lines.append(f"结构词表   : Atomic={codebook['atomic_vocab']} | Emergent={codebook['emergent_vocab']}")
    else:
        loss = settings['loss']['supervised']
        fine = settings['finetuning']
        if not (args.init_checkpoint or args.warmstart_checkpoint or args.resume) and fine['strategy'] == 'full':
            lines.append('监督策略   : 随机初始化，全参从第一步更新，无冻结或任务头单独训练阶段')
        else:
            lines.append(f"微调策略   : {fine['strategy']} | 任务头预热 {fine['head_warmup_steps']} steps"
                         f" | 上层适配 {fine['upper_layers_steps']} steps | 最后 {fine['unfreeze_last_n_blocks']} blocks"
                         f" | 层间LR倍率 {fine['layer_lr_decay']:g} | anchor_weight={fine['anchor_weight']:g}")
        lines.append(f"监督目标   : task_weights={loss['task_weights']} | 回归={loss['regression']} | grid_weight={loss['grid_weight']:g}")
        lines.append(f"训练增强   : 随机整窗相位旋转概率={data['augmentation']['phase_rotation_probability']:g}"
                     f" | label_smoothing={loss['label_smoothing']:g} | 验证/推理不增强")
        lines.append(f"表征目标   : SupCon weight={loss['contrastive']['weight']:g}"
                     f" | temperature={loss['contrastive']['temperature']:g} | 同类不同记录为正样本")
    mode = (f"续训 {args.resume}" if args.resume else
            f"加载完整模型 {args.warmstart_checkpoint}" if args.warmstart_checkpoint else
            f"加载基座 {args.init_checkpoint}" if args.init_checkpoint else '随机初始化')
    evaluation = settings['evaluation']
    early = evaluation.get('early_stopping', {'patience': 0, 'min_delta': 0.})
    lines.extend([
        f"初始化     : {mode}",
        f"检查点     : save_last={settings['checkpoint']['save_last']}"
        f" | 每 {settings['checkpoint']['save_every_steps']} steps 定期保存（0=关闭）"
        f" | 按 save_last 在 epoch 末保存；best 按验证选择",
        f"验证       : 每 {evaluation['every_epochs']} epochs | batch={evaluation['batch_size']}"
        f" | {'全量' if not evaluation['max_batches'] else str(evaluation['max_batches']) + ' batches'}"
        f" | selection={evaluation['selection_metric']}",
        f"提前停止   : patience={early['patience']} 次完整验证（0=关闭） | 最小改善={early['min_delta']:g}",
        f"配置       : {args.config}",
        f"数据目录   : {args.data}",
        f"输出目录   : {args.output}",
        '显示口径   : step=一次参数更新；loss=本 epoch 平均；分项为有效目标的未加权平均；grad=裁剪前范数。',
    ])
    if world > 1:
        lines.append('双卡显示   : 仅 rank 0 输出；进度条损失为 rank 0 本地统计。')
    lines.extend(['=' * 88, ''])
    print('\n'.join(lines), flush=True)


def active_loss_parts(batch, config):
    """Inspect CPU metadata before transfer; zero placeholders are not observations."""
    if config['stage'] == 'pretrain':
        return {name for name, weight in [('atomic', config['atomic_weight']), ('emergent', config['emergent_weight']),
                    ('reconstruction', config['reconstruction_weight']), ('spectrum', config['spectral_loss_weight'])] if weight > 0}
    active = {task for index, task in enumerate(['modulation', 'protocol'])
              if config['task_weights'][index] > 0 and bool((batch[task] >= 0).any())}
    if config.get('contrastive_weight', 0) and 'modulation' in active:
        active.add('supcon')
    field_weights = torch.as_tensor(config['parameter_weights']) > 0
    if config['task_weights'][2] > 0 and bool((batch['target_valid'] & field_weights).any()):
        active.add('parameters')
    if config['grid_loss_weight'] > 0 and bool(batch['target_valid'].any()):
        active.add('grid')
    return active


class EpochProgress:
    """One bar per epoch. Loss counters remain useful when bars are disabled."""
    def __init__(self, epoch, epochs, steps, initial=0, enabled=True, min_interval=0.5, file=None, gradient_clip=1.):
        self.epoch, self.epochs, self.steps, self.initial = epoch, epochs, steps, initial
        self.enabled, self.min_interval = enabled, min_interval
        self.file = file or sys.stdout
        self.loss_sum, self.microbatches, self.updates = 0., 0, 0
        self.part_sums, self.part_counts = defaultdict(float), defaultdict(int)
        self.started, self.bar = time.monotonic(), None
        self.elapsed = None
        self.last_rates = {}
        self.gradient_norms, self.gradient_clip = [], gradient_clip

    def __enter__(self):
        if self.enabled:
            self.bar = tqdm(total=self.steps, initial=self.initial, desc=f'Epoch {self.epoch + 1:02d}/{self.epochs:02d}',
                            unit='step', dynamic_ncols=True, mininterval=self.min_interval, miniters=1,
                            leave=True, file=self.file, ascii=True)
        return self

    def observe(self, loss, parts, active):
        self.loss_sum += loss
        self.microbatches += 1
        for name in active & parts.keys():
            self.part_sums[name] += parts[name]
            self.part_counts[name] += 1

    def advance(self, rates, gradient_norm=None):
        self.updates += 1
        self.last_rates = rates
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)
        snapshot = getattr(self.file, 'log_snapshot', None)
        if snapshot and (self.updates == 1 or self.updates % getattr(self, 'log_every', 50) == 0 or self.initial + self.updates == self.steps):
            parts = ' | '.join(f'{name}={self.part_sums[name] / n:.4f}' for name, n in self.part_counts.items() if n)
            learning_rates = ' | '.join(f'{name}={value:.2e}' for name, value in rates.items())
            snapshot(f'Epoch {self.epoch + 1:02d}/{self.epochs:02d} | step {self.initial + self.updates}/{self.steps}'
                     f' | loss={self.loss_sum / max(self.microbatches, 1):.4f} | {parts} | {learning_rates}'
                     + (f' | grad={gradient_norm:.3f}' if gradient_norm is not None else ''))
        if self.bar is not None:
            aliases = {'modulation': 'mod', 'protocol': 'proto', 'parameters': 'param',
                       'supcon': 'supcon',
                       'anchor': 'anchor',
                       'grid': 'grid', 'atomic': 'atom', 'emergent': 'emerg',
                       'reconstruction': 'recon', 'spectrum': 'spec'}
            postfix = {'loss': f'{self.loss_sum / max(self.microbatches, 1):.4f}'}
            for name, alias in aliases.items():
                if self.part_counts[name]:
                    postfix[alias] = f'{self.part_sums[name] / self.part_counts[name]:.4f}'
            for prefix, label in [('backbone', 'lr_b'), ('heads', 'lr_h')]:
                value = next((v for k, v in rates.items() if k.startswith(prefix)), None)
                if value is not None:
                    postfix[label] = f'{value:.2e}'
            if gradient_norm is not None:
                postfix['grad'] = f'{gradient_norm:.2f}'
            self.bar.set_postfix(postfix, refresh=False)
            self.bar.update(1)

    def summary(self):
        result = {'loss': self.loss_sum / self.microbatches if self.microbatches else None,
                'parts': {key: self.part_sums[key] / n for key, n in self.part_counts.items() if n},
                'optimizer_updates': self.updates, 'initial_epoch_step': self.initial,
                'epoch_step': self.initial + self.updates, 'epoch_steps': self.steps,
                'elapsed_seconds': self.elapsed if self.elapsed is not None else time.monotonic() - self.started,
                'learning_rates': self.last_rates}
        if self.gradient_norms:
            values = np.asarray(self.gradient_norms)
            result['gradient_norm'] = {'median': float(np.median(values)), 'p95': float(np.quantile(values, .95)),
                                      'max': float(values.max()),
                                      'clipped_fraction': float((values > self.gradient_clip).mean()) if self.gradient_clip else 0.}
        return result

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = time.monotonic() - self.started
        if self.bar is not None:
            if exc_type:
                self.bar.set_description(f'Epoch {self.epoch + 1:02d}/{self.epochs:02d} interrupted', refresh=False)
            self.bar.close()


def epoch_summary(epoch, epochs, training, metrics, evaluated, improved, saved, complete):
    status = '完成' if complete else '提前停止'
    loss = f"{training['loss']:.4f}" if training['loss'] is not None else '-'
    print(f"Epoch {epoch + 1:02d}/{epochs:02d} {status} | step {training['epoch_step']}/{training['epoch_steps']}"
          f" | train_loss={loss} | {training['elapsed_seconds']:.1f}s", flush=True)
    if training.get('gradient_norm'):
        grad = training['gradient_norm']
        print(f"  梯度（裁剪前）| median={grad['median']:.3f} | p95={grad['p95']:.3f}"
              f" | max={grad['max']:.3f} | 裁剪比例={grad['clipped_fraction']:.1%}", flush=True)
    if evaluated:
        if 'pretraining_validation_loss' in metrics:
            print(f"  验证 | pretrain_loss={metrics['pretraining_validation_loss']:.4f}", flush=True)
        else:
            classification = metrics['classification']
            print(f"  验证 | mod_acc={classification['modulation']['accuracy']:.2%}"
                  f" | proto_acc={classification['protocol']['accuracy']:.2%}"
                  f" | param_proxy={metrics['parameter_proxy']:.4f}"
                  f" | proxy_score={metrics['proxy_score']:.3f}（本地代理分）", flush=True)
            if all('cross_entropy' in classification[task] for task in ['modulation', 'protocol']):
                print(f"  验证损失 | mod_CE={classification['modulation']['cross_entropy']:.4f}"
                      f" | proto_CE={classification['protocol']['cross_entropy']:.4f}", flush=True)
            classification_summary(metrics)
            values = ' | '.join(f"{name}={value['rmse']:.3e}" for name, value in metrics['parameters'].items() if value['rmse'] is not None)
            print(f'  参数 RMSE（原单位）| {values}', flush=True)
        print(f"  选择结果 | {'验证指标改善' if improved else '验证指标未改善'}", flush=True)
    else:
        print('  验证 | 本轮按配置跳过', flush=True)
    if saved:
        print('  已保存 | ' + ', '.join(saved), flush=True)
    print('', flush=True)
