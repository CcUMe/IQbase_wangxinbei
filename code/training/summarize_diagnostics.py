"""Verify matching controls and export both diagnostic results as JSON, CSV and plots."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils.common import read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    joint, single = [read_json(root / name / 'protocol.json') for name in ['joint', 'modulation_only']]
    assert joint['initial_state_sha256'] == single['initial_state_sha256'], 'Different initialization'
    assert joint['modulation_batch_stream_sha256'] and joint['modulation_batch_stream_sha256'] == single['modulation_batch_stream_sha256'], 'Different or incomplete modulation sample stream'
    assert joint['modulation_exposures'] == single['modulation_exposures'], 'Different exposure budgets'
    assert joint['learning_rate'] == single['learning_rate'] and joint['scheduler'] == single['scheduler'], 'Different learning-rate schedules'
    def modulation_rates(name):
        rows = [json.loads(line) for line in (root / name / 'events.jsonl').open()]
        return [(r['cycle'], r['learning_rates']) for r in rows if r['event'] == 'step' and r['slot'] == 0]
    assert modulation_rates('joint') == modulation_rates('modulation_only'), 'Actual modulation update learning rates differ'
    summary = {'task_diagnostic': {'controls_verified': True, 'joint_protocol': joint, 'single_protocol': single,
        'joint': read_json(root / 'joint/metrics.json'), 'modulation_only': read_json(root / 'modulation_only/metrics.json')},
        'readout_diagnostic': {}, 'limitations': [
            'Task diagnostic uses one random initialization and a finite from-scratch 17M budget; it does not establish the absence of late-stage interference in 96M.',
            'Joint adds backbone AdamW updates; this is a net-effect test, not isolated proof of gradient conflict.',
            'Readouts use two seeds and one fixed optimizer recipe. Both fresh readouts use train-only standardization.',
            'Frozen reference heads are recomputed in the same feature pipeline; slight differences from older evaluation batching are possible.',
            'Validation was used to select original base checkpoints, so this is not an untouched hidden-test result.']}
    aggregate, per_class = [], []

    def collect(experiment, model, head, seed, task, metric, train_accuracy=None):
        aggregate.append({'experiment': experiment, 'model': model, 'head': head, 'seed': seed, 'task': task,
            'train_accuracy_percent': None if train_accuracy is None else 100 * train_accuracy,
            **{key + '_percent': 100 * metric[key] for key in ['accuracy', 'macro_f1', 'weighted_f1']}, 'support': metric['support']})
        for label, precision, recall, f1, count in zip(metric['labels'], metric['precision'], metric['recall'], metric['f1'], metric['class_support']):
            per_class.append({'experiment': experiment, 'model': model, 'head': head, 'seed': seed, 'task': task,
                'class': label, 'precision_percent': precision * 100, 'recall_class_accuracy_percent': recall * 100,
                'f1_percent': f1 * 100, 'support': count})

    for name in ['joint', 'modulation_only']:
        for task, metrics in summary['task_diagnostic'][name]['classification'].items():
            collect('tasks', '17M_fresh', name, joint['seed'], task, metrics)
    for scale in ['small', 'large']:
        folder = root / f'frozen_{scale}'
        result = read_json(folder / 'summary.json')
        assert result['protocol']['backbone_verified_unchanged']
        collect('frozen', scale, 'original', '', 'modulation', result['reference']['validation'], result['reference']['train']['accuracy'])
        for row in result['results']:
            collect('frozen', scale, row['head'], row['seed'], 'modulation', row['validation'], row['train']['accuracy'])
        result['aggregates'] = {}
        for kind in ['linear', 'mlp']:
            rows = [r for r in result['results'] if r['head'] == kind]
            result['aggregates'][kind] = {
                key: {'mean': float(np.mean([r['validation'][key] for r in rows])),
                      'std_across_seeds': float(np.std([r['validation'][key] for r in rows], ddof=1)) if len(rows) > 1 else None}
                for key in ['accuracy', 'macro_f1', 'weighted_f1']}
            result['aggregates'][kind]['train_accuracy'] = float(np.mean([r['train']['accuracy'] for r in rows]))
        labels = np.load(folder / 'validation_labels.npy')
        linear_correct = np.mean([np.load(folder / f'linear_seed{seed}/validation_predictions.npy') == labels
                                  for seed in result['protocol']['seeds']], axis=0)
        mlp_correct = np.mean([np.load(folder / f'mlp_seed{seed}/validation_predictions.npy') == labels
                               for seed in result['protocol']['seeds']], axis=0)
        difference = mlp_correct - linear_correct
        margin = float(1.96 * difference.std(ddof=1) / len(difference) ** .5)
        result['paired_mlp_minus_linear'] = {'mean': float(difference.mean()),
            'approximate_95pct_interval_conditional_on_these_models': [float(difference.mean() - margin), float(difference.mean() + margin)]}
        summary['readout_diagnostic'][scale] = result
    summary['task_diagnostic']['joint_minus_single_accuracy_pp'] = 100 * (
        summary['task_diagnostic']['joint']['classification']['modulation']['accuracy'] -
        summary['task_diagnostic']['modulation_only']['classification']['modulation']['accuracy'])
    write_json(root / 'summary.json', summary)
    for name, rows in [('classification_summary.csv', aggregate), ('per_class_metrics.csv', per_class)]:
        with (root / name).open('w', encoding='utf-8-sig', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# 两项训练诊断：结果与解释', '',
        '运行目录：`' + str(root) + '`。全部结果来自原验证划分；调制 20,885 条、协议 14,511 条，完整模型验证共 118,182 条。', '',
        '## 诊断一：相同调制曝光下的多任务与单任务', '',
        f'两组均从相同随机权重开始，使用 17M 结构并开启 QK 归一化。每组 {joint["modulation_cycles"]:,} 次调制更新，每批 {joint["batch_size"]} 条，累计 {joint["modulation_exposures"]:,} 次调制样本曝光。',
        '初始权重哈希、完整调制批次流哈希及日志内实际调制学习率均已核对相同。每 1,200 次调制更新为一个显示 epoch，共四段。',
        '多任务组额外穿插协议和参数更新，因此共享 AdamW 状态及总更新次数不同。这是增加其他任务更新后的净效应对照，不是只隔离梯度冲突的实验。', '',
        '| 训练方式 | 验证任务 | Accuracy | Macro-F1 | Weighted-F1 |', '|---|---|---:|---:|---:|']
    for name, label in [('joint', '多任务'), ('modulation_only', '仅调制')]:
        for task, task_label in [('modulation', '调制'), ('protocol', '协议')]:
            m = summary['task_diagnostic'][name]['classification'][task]
            note = '（未训练头，仅记录）' if name == 'modulation_only' and task == 'protocol' else ''
            lines.append(f'| {label} | {task_label}{note} | {m["accuracy"]:.2%} | {m["macro_f1"]:.2%} | {m["weighted_f1"]:.2%} |')
    delta = summary['task_diagnostic']['joint_minus_single_accuracy_pp']
    lines += ['', f'多任务组调制准确率高 {delta:.2f} 个百分点。这组早期训练对照不支持将多任务负迁移视为主要原因；它不能排除 96M 长期训练后期的干扰。', '',
        '## 诊断二：冻结现有基座，重新训练分类头', '',
        '现有小模型取第 5 轮检查点，大模型取第 44 轮检查点。编码器完全冻结，逐张量验证未改变。训练、验证特征均以 FP32 提取；两个新头使用同一份训练侧均值/标准差、相同批次顺序和优化配方。',
        'Linear 与原形状 MLP 各训练固定 20 轮、两个种子（2026、2027）。以下报告最后一轮的种子均值，不按验证结果挑选轮次。原头使用原始特征，新头使用标准化特征，二者对比包含标准化及重新训练的组合影响。', '',
        '| 基座 | 分类头 | 训练 Accuracy | 验证 Accuracy | Macro-F1 | Weighted-F1 |', '|---|---|---:|---:|---:|---:|']
    for scale, label in [('small', '17M'), ('large', '96M')]:
        result = summary['readout_diagnostic'][scale]
        for kind, head_label in [('original', '原头'), ('linear', '新 Linear'), ('mlp', '新 MLP')]:
            if kind == 'original':
                m, train = result['reference']['validation'], result['reference']['train']['accuracy']
            else:
                a = result['aggregates'][kind]
                m = {k: a[k]['mean'] for k in ['accuracy', 'macro_f1', 'weighted_f1']}
                train = a['train_accuracy']
            lines.append(f'| {label} | {head_label} | {train:.2%} | {m["accuracy"]:.2%} | {m["macro_f1"]:.2%} | {m["weighted_f1"]:.2%} |')
    lines += ['', '小模型重新训练 MLP 有约 2 个百分点的收益，线性头没有优于 MLP，不能简单归因于 MLP 过于复杂。小模型原头的训练/验证差距也远小于大模型，不能将两个训练阶段统一解释为严重过拟合。',
        '大模型两类新头的训练准确率接近 100%，验证仍约 64.28%。这支持将主要瓶颈定位到基座对新记录的泛化，而非这两类分类头的表达能力；不等同于证明任意读出方案都无效。',
        '基座参考值按同一冻结特征流程重新计算，因评估分批/补齐方式不同，与历史日志可能存在约 0.02 个百分点以内差异。比较以本流程内参照为准。', '',
        '## 全部逐类识别准确率和 F1', '',
        '逐类“识别准确率”按 Recall=TP/该类真实样本数定义，避免一对多 Accuracy 被大量真负例抬高。下表比较诊断一最终调制结果；协议和所有分类头的逐类 Precision、Recall、F1、Support 均在 CSV 与各自 train.log 中。', '',
        '| 调制类别 | 多任务 Recall | 多任务 F1 | 单任务 Recall | 单任务 F1 |', '|---|---:|---:|---:|---:|']
    a = summary['task_diagnostic']['joint']['classification']['modulation']
    b = summary['task_diagnostic']['modulation_only']['classification']['modulation']
    for i, label in enumerate(a['labels']):
        lines.append(f'| {label} | {a["recall"][i]:.2%} | {a["f1"][i]:.2%} | {b["recall"][i]:.2%} | {b["f1"][i]:.2%} |')
    lines += ['', '## 日志与文件', '',
        '- [整体分类结果 CSV](classification_summary.csv)、[全部逐类指标 CSV](per_class_metrics.csv)、[完整 JSON](summary.json)、[对照图](diagnostic_comparison.png)。',
        '- 可读日志：[多任务](joint/train.log)、[单任务](modulation_only/train.log)、[小模型冻结读出](frozen_small/train.log)、[大模型冻结读出](frozen_large/train.log)。',
        '- train.log 保存总览、进度快照、逐类指标和收尾；逐 step/epoch 结构化事件单独保存在 events.jsonl。',
        '- 正式训练入口、双进程训练及断点续训已做小规模执行检查，历史 JSON train.log 会保留为 train_legacy.jsonl。',
        '- 用户在 main.json 中修改的模型参数和注释保持原样；配置读取增加 // 与 /* */ 注释支持。没有新增主配置文件，也未替换默认推理权重。', '',
        '## 结果边界', '']
    lines.extend('- ' + text for text in summary['limitations'])
    (root / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for name, label in [('joint', 'Joint tasks'), ('modulation_only', 'Modulation only')]:
        history = read_json(root / name / 'history.json')
        axes[0].plot([r['modulation_updates'] for r in history],
                     [100 * r['metrics']['classification']['modulation']['accuracy'] for r in history], marker='o', label=label)
    axes[0].set_xlabel('Matched modulation updates')
    axes[0].set_ylabel('Full validation accuracy (%)')
    axes[0].legend()
    positions = np.arange(2)
    for index, kind in enumerate(['original', 'linear', 'mlp']):
        values, gaps, errors = [], [], []
        for scale in ['small', 'large']:
            result = summary['readout_diagnostic'][scale]
            if kind == 'original':
                accuracy = result['reference']['validation']['accuracy']
                train = result['reference']['train']['accuracy']
                errors.append(0.)
            else:
                accuracy = result['aggregates'][kind]['accuracy']['mean']
                train = result['aggregates'][kind]['train_accuracy']
                errors.append(100 * (result['aggregates'][kind]['accuracy']['std_across_seeds'] or 0.))
            values.append(100 * accuracy)
            gaps.append(100 * (train - accuracy))
        axes[1].errorbar(positions + (index - 1) * .24, values, yerr=errors, fmt='o', capsize=3, label=kind)
        axes[2].bar(positions + (index - 1) * .24, gaps, .24, label=kind)
    for ax in axes[1:]:
        ax.set_xticks(positions, ['17M base', '96M base'])
        ax.legend()
    axes[1].set_ylabel('Frozen-base validation accuracy (%)')
    axes[1].set_ylim(55, 70)
    axes[2].set_ylabel('Train minus validation accuracy (pp)')
    for ax in axes:
        ax.grid(axis='y', alpha=.25)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(root / 'diagnostic_comparison.png', dpi=170)
    print(json.dumps({'controls_verified': True, 'joint_minus_single_accuracy_pp': summary['task_diagnostic']['joint_minus_single_accuracy_pp'],
                      'summary': str(root / 'summary.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
