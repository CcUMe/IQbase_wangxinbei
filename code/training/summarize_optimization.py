"""Summarize completed representation experiments without rerunning or selecting epochs."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True, help='Existing completed optimization artifact root')
    p.add_argument('--diagnostics', type=Path, required=True)
    args = p.parse_args()
    r, d = args.output.resolve(), args.diagnostics.resolve()
    baseline = read(r / 'baseline_full_validation.json')
    selected = read(r / 'selected/metrics.json')
    assert baseline['rows'] == selected['rows'] == 118182
    assert baseline['classification']['protocol'] == selected['classification']['protocol']
    assert baseline['parameters'] == selected['parameters']
    warm = {name: read(r / name / 'history.json') for name in ['ce_control', 'supcon']}
    wp = {name: read(r / name / 'protocol.json') for name in warm}
    assert wp['ce_control']['initial_state_sha256'] == wp['supcon']['initial_state_sha256']
    assert wp['ce_control']['sample_stream_sha256'] == wp['supcon']['sample_stream_sha256']
    scratch = {'joint_control': read(d / 'joint/history.json'),
               **{name: read(r / name / 'history.json') for name in ['scratch_supcon', 'scratch_local']}}
    initial = read(d / 'joint/protocol.json')
    for name in ['scratch_supcon', 'scratch_local']:
        protocol = read(r / name / 'protocol.json')
        assert protocol['reference_initial_state_sha256'] == initial['initial_state_sha256']
        assert protocol['modulation_batch_stream_sha256'] == initial['modulation_batch_stream_sha256']
    rows = []
    for name, records in {**warm, **scratch}.items():
        metrics = records[-1]['metrics']['classification']['modulation']
        rows.append({'experiment': name, 'accuracy': metrics['accuracy'],
                     'macro_f1': metrics['macro_f1'], 'weighted_f1': metrics['weighted_f1']})
    readouts = []
    for scale in ['small', 'large']:
        variants = [('mean', d / f'frozen_{scale}/summary.json')]
        variants += [(name, r / f'{name}_{scale}/summary.json') for name in ['statistics', 'physical', 'combined']]
        for feature, path in variants:
            data = read(path)
            assert data['protocol']['encoder_frozen']
            for head in ['linear', 'mlp']:
                results = [v for v in data['results'] if v['head'].lower() == head]
                assert len(results) == 2
                readouts.append({'scale': scale, 'features': feature, 'head': head,
                    'accuracy_mean': statistics.mean(v['validation']['accuracy'] for v in results),
                    'macro_f1_mean': statistics.mean(v['validation']['macro_f1'] for v in results),
                    'accuracy_seeds': [v['validation']['accuracy'] for v in results],
                    'device': data['protocol'].get('device', 'cuda'), 'source': str(path)})
    with (r / 'classification_per_class.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task', 'label', 'support', 'baseline_precision', 'baseline_recall', 'baseline_f1',
                         'selected_precision', 'selected_recall', 'selected_f1'])
        for task, metric in selected['classification'].items():
            old = baseline['classification'][task]
            for i, label in enumerate(metric['labels']):
                writer.writerow([task, label, metric['class_support'][i],
                    *[old[key][i] for key in ['precision', 'recall', 'f1']],
                    *[metric[key][i] for key in ['precision', 'recall', 'f1']]])
    summary = {'full_validation_rows': selected['rows'], 'baseline': baseline, 'selected': selected,
               'controlled_training': rows, 'frozen_readouts': readouts,
               'source_checkpoint': str(r / 'selected/model.pt'),
               'interpretation': 'Modest local-validation improvement; structural representation learning and scaling remain unresolved.',
               'validation_reused_for_exploration': True,
               'independent_test_improvement_established': False}
    (r / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 诊断后的模型优化与实测结果', '',
        '本轮得到可直接推理的单模型，但调制泛化提升有限。全部比较沿用原哈希分组划分，原始数据未改动，训练和验证不共享完全相同的 I/Q 哈希；官方样例及其重复组已隔离。', '',
        '## 最终单模型：全量验证', '',
        '| 任务 | 指标 | 原 95M 模型 | 优化后 96.32M 模型 |', '|---|---|---:|---:|']
    for task, label in [('modulation', '调制'), ('protocol', '协议')]:
        for key, display in [('accuracy', 'Accuracy / Micro-F1'), ('macro_f1', 'Macro-F1'), ('weighted_f1', 'Weighted-F1')]:
            lines.append(f"| {label} | {display} | {baseline['classification'][task][key]:.4%} | {selected['classification'][task][key]:.4%} |")
    delta = (selected['classification']['modulation']['accuracy'] - baseline['classification']['modulation']['accuracy']) * 100
    lines += ['', f'验证共 {selected["rows"]:,} 条，其中调制 {selected["classification"]["modulation"]["support"]:,} 条、协议 {selected["classification"]["protocol"]["support"]:,} 条。调制净提升 **{delta:.4f} 个百分点**。协议所有分类统计、七项参数所有误差与原模型完全一致。', '',
        f'本地代理分 {baseline["proxy_score"]:.4f} → {selected["proxy_score"]:.4f}，不是官方成绩。', '',
        '| 参数 | RMSE（原始单位，两模型一致） | 有效验证数 |', '|---|---:|---:|']
    for name, metric in selected['parameters'].items():
        lines.append(f'| {name} | {metric["rmse"]:.8g} | {metric["count"]:,} |')
    lines += ['', '## 实际保留的改动', '',
        '保留原共享时频编码器、协议头和七项参数头。调制头输入由隐层均值扩展为均值 + 标准差 + 32 项固定 I/Q 统计量，冻结基座，仅用训练侧调制标签训练 MLP 读出层。统计量包括幅度分布、高阶矩/累积量、复相关和相位谐波；无外部数据库、真实采样率、SNR 或答案元数据。', '',
        '读出层训练固定 20 epochs、batch 1024、AdamW LR=1e-3、weight_decay=0.01，cosine 最低比例 0.05；固定末轮结果，不按验证选轮。两个种子 2026/2027；导出预定的 2026 单模型，没有平均或集成两个模型。原 95M 基座来自 44 轮联合监督模型，不是本轮新完成的大规模结构预训练。', '',
        '训练侧特征均值/标准差已经代数折入首层权重。正式模型无特征缓存依赖；协议头新增输入列权重为零以保持原预测。分类头及统计量使用 FP32，基座仍支持 BF16；归一化折入后避免分类头 BF16 消减误差。最终 96,321,869 参数，共享 93,849,216，占 97.4329%。', '',
        '默认 `weights/` 已更新；原默认权重备份到本目录 `previous_default_weights/`。完整训练权重在 `selected/model.pt`；这是合并后的权重检查点，没有可恢复的读出训练优化器，继续训练应使用 `--warmstart-checkpoint`。', '',
        '## 没有启用的候选：控制实验', '',
        '| 实验 | 调制 Accuracy | Macro-F1 | Weighted-F1 |', '|---|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["experiment"]} | {row["accuracy"]:.4%} | {row["macro_f1"]:.4%} | {row["weighted_f1"]:.4%} |')
    lines += ['',
        'ce_control / supcon：同一 17M 联合检查点、相同初始化与全任务采样流、6000 次更新、全局相位增强概率 0.1；仅 SupCon 权重 0 vs 0.1 不同。两边均有相同随机初始化的训练辅助投影，以消除初始化差异。SupCon 没有胜过 CE。', '',
        'joint_control / scratch_supcon / scratch_local：相同随机共享权重与调制批次流，从头训练，4800 次调制更新，联合任务共 14400 次 AdamW 更新，无增强；逐 cycle 学习率一致。SupCon 权重 0.3、局部残差前端单独对照都没有胜过原联合基线。该结果适用于本次单种子和预算，不能推断所有对比学习或局部前端均无效。', '',
        '因此 `main.json` 中 SupCon 权重为 0、local_refinement=false。实现保留供研究，未把负结果写成默认改进。SupCon 正样本来自同类不同记录；重复采到同一行会排除，自身两种视图不是本次实验正样本定义。', '',
        '## 冻结基座读出：两个种子平均', '',
        '| 基座 | 特征 | 读出 | Accuracy 均值 | 两个种子 Accuracy |', '|---|---|---|---:|---|']
    for row in readouts:
        seeds = ' / '.join(f'{v:.4%}' for v in row['accuracy_seeds'])
        lines.append(f'| {row["scale"]} | {row["features"]} | {row["head"]} | {row["accuracy_mean"]:.4%} | {seeds} |')
    lines += ['',
        'mean=均值；statistics=均值+标准差；physical=均值+32项统计；combined=均值+标准差+32项统计。各头采用相同训练侧标准化与预算；combined 在 CPU 训练，其他主要在 GPU，设备上的随机排列/浮点运算不保证逐位相同，跨组属于探索性比较。最终完整模型 FP32 全量验证比缓存读出分数更接近部署，因此以开头的 64.5008% 为准。', '',
        '## 对瓶颈的判断', '',
        '诊断未支持“多任务负迁移是主要原因”，移除辅助任务在匹配调制曝光的预算中反而下降。大模型训练调制准确率近 100%，验证约 64%，重训相同均值读出几乎不变；单纯加参数或再换大分类头缺乏证据。统计读出的小幅增益说明池化损失了一部分信息，但尚不能证明编码器学到了更深的生成结构。', '',
        '按训练侧每类一阶相关幅度四分位阈值进行事后分层，原模型验证准确率约 46.14%、50.94%、71.78%、87.86%。这是相关性诊断，分层使用真实类别，仅用于离线分析；它不是 SNR 测量，不是噪声上限证明，不进入模型推理。', '',
        '下一阶段应优先对难混淆类别建立同步/频偏/符号尺度的控制实验，再检验共享编码器是否能在观测变化下保持类别结构；先以线性读出和固定预算迁移收益验证，再投入大规模预训练与参数扩展。当前数据缺乏采样率元信息，不能虚构物理时间坐标来实现 RoPE-fs。', '',
        '本轮多次使用同一验证集比较候选，因此这 0.27 个百分点属于探索性本地收益；没有独立测试集确认，不能承诺线上提升或泛化瓶颈已经解决。', '',
        '## 配置、运行与复现', '',
        '`code/configs/supervised/main.json` 是唯一主配置，已对齐导出模型的 768×12、8 heads、QK 归一化、统计读出与 FP32 分类头。旧配置和已启动进程不会自动更新。主配置描述后续联合训练的结构和配方，不是此次“原 44 轮基座 + 冻结读出 20 轮”的一键等价复现。', '',
        '直接推理使用现有 `weights/`，无需重新训练。若开展后续联合微调，可运行以下命令；它会重置优化器并产生新的时间戳目录，后续精度必须重新验证：', '', '```bash',
        'CUDA_VISIBLE_DEVICES=2,3 /home/yangqilin/anaconda3/envs/py313/bin/python -m torch.distributed.run \\',
        '  --standalone --nproc_per_node=2 code/training/train.py \\',
        '  --data artifacts/data_full --config code/configs/supervised/main.json \\',
        f'  --warmstart-checkpoint {args.output}/selected/model.pt \\',
        '  --output artifacts/next_finetune', '```', '',
        '精确复现本次候选先运行 `diagnose_readout.py --checkpoint <原95M检查点> --pooling mean_std --physical-statistics --device cpu --seeds 2026 2027 --epochs 20`，再用 `promote_readout.py --readout <输出>/mlp_seed2026/head.pt --output <检查点>` 合并。原始命令选项及设备见各目录 protocol.json，实验配置见 resolved_config.json；CE 对照复现应使用其保存的配置快照，避免主配置后续改动改变增强。', '',
        '## 检查与完整指标', '',
        '35 项回归检查通过；CPU 双进程对比梯度和全 batch 参考对齐，含无正样本 rank 与不均匀有效锚点；合法 GPU 2/3 的 NCCL 双卡训练、梯度累积、续训通过；正式模型 BF16 热启动通过。safetensors 导出逐张量往返检查通过；官方六条样例 CPU FP32、GPU FP32/BF16 推理及输出 Schema 检查通过。', '',
        '- [逐类 Precision / Recall / F1 CSV](classification_per_class.csv)',
        '- [原模型全量验证](baseline_full_validation.json)',
        '- [最终模型全量验证](selected/metrics.json)',
        '- [所有候选与协议汇总](summary.json)',
        '- [最终接口检查状态](validation_status.json)',
        '- [回归检查](tests.log)', '',
        'train.log 保留人可读总览、逐 epoch/step 进度快照、完整分类表与参数误差；events.jsonl 存放机器可读事件。每次主训练自动加时间戳。', '']
    (r / 'report.md').write_text('\n'.join(lines))
    print(f'Report: {r / "report.md"}')


if __name__ == '__main__':
    main()
