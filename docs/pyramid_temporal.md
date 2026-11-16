# 金字塔调制头：全局与分段统计聚合

TensorBoard默认随训练/续训记录到本次运行目录；旧进程可增量补录或跟随JSONL。[可视化使用说明](tensorboard.md)。

> 状态更新：该配置是先前分段头候选，保留供既有运行续训。当前主干优化候选为pyramid_midfusion，使用原版任务头；两者未合并。[主干说明](pyramid_midfusion.md)。

2026-11-13，Asia/Shanghai。本次只优化调制读出，保留P16/P8/P4金字塔共享编码器、时频交互、协议头和七参数头。没有重新引入串行BiLSTM结构，也未修改损失、任务采样、学习率或预训练目标。本版尚无正式准确率增益结论。

## 启动

在项目根目录、py313环境运行：

```bash
python code/training/train.py --config code/configs/supervised/pyramid_temporal.json5
```

GPU1单卡，batch256，梯度累积1；数据仍为artifacts/data_full。新训练保存到 `logs/supervised_pyramid_temporal/<上海时间戳>/`。原AdamW峰值3e-4、500步预热＋cosine最低比例0.05、40轮上限与8次验证早停全部保留。终端总览会显示实际分段聚合参数。

原版配置 `code/configs/supervised/pyramid_v1_171823.json5` 内容保持不变，用于基线和原17:18断点续训。**新调制头不能直接resume或warmstart原版完整pt；本配置默认从零联合监督训练。**恢复新版运行时用本配置和该新版的last.pt：

```bash
python code/training/train.py --config code/configs/supervised/pyramid_temporal.json5 \
  --resume logs/supervised_pyramid_temporal/<时间戳>/last.pt
```

## 调制读出

对粗／中／细每个尺度分别执行：

```text
尺度特征H与有效mask
 ├ 原跨尺度摘要＋全局注意力均值/标准差 → 原256维全局描述g
 └ 按有效token顺序分成4段
   → 每段均值/标准差 → 64维段描述
   → 沿4段的depthwise卷积(kernel3)＋逐点混合残差
   → 按顺序展开4×64 → 投影到256维描述l
 → g + 可学习增益 * l
三个尺度的256维描述拼接 → 原768→256→23分类器
```

每个尺度都保留原全局路径，新增支路没有细节幅度硬上限，也没有预设粗尺度必须主导。增益为每通道参数，初始0.1，之后可自由学习；它不是贡献百分比或logit权重上限。

1024点输入的三个尺度仍为64×384、128×192、256×192。划分4段后，各自每段16／32／64个有效token；512点的数量减半。分段由每个样本有效token的相对顺序确定，不依赖批内padding长度，也不把段数解释为符号数或真实物理时间。

分段均值、方差使用FP32，两次归约先算均值再算中心化平方，避免大偏置下直接用E[x²]−E[x]²造成消减误差。实现通过scatter/gather归约，不构造B×段数×token×通道的大张量。空段置零并遮蔽；真实512/1024输入下四段均有效。展平聚合保留段的先后位置，不把段序列再次平均掉。

这是一项基于现有统计聚合方向的工程改造，不是可学习高阶矩池化论文的复现，也不是已验证的困难调制类别解决方案。

## 配置

新增控制集中在 `model.heads.modulation_temporal`：

| 参数 | 本配置 | 含义 |
|---|---:|---|
| enabled | true | 开启新增调制聚合；默认false，旧配置保持原行为 |
| segments | 4 | 每尺度有效token分段数 |
| segment_width | 64 | 每段统计描述投影宽度 |
| temporal_kernel | 3 | 段序列上的正奇数卷积核 |
| dropout | 0.1 | 新增支路的dropout |
| residual_scale | 0.1 | 无硬上限的可学习增益初值，须为正 |

新增模块使用隔离的初始化随机数状态，固定同一随机种子时，不改变任何已有编码器、全局调制头、协议头和参数头张量的初始化。评估模式下，协议/参数/网格输出与同初始化原版逐元素一致。训练时共享编码器会接收新的调制梯度，因此这些任务的后续精度仍可能变化，需要一起观察。

## 参数与竞赛接口

| 项目 | 原版 | 新版 |
|---|---:|---:|
| 总参数 | 20,966,006 | 21,277,814 |
| 共享参数 | 19,153,568 | 19,153,568 |
| 共享比例 | 91.36% | 90.02% |

新增311,808个参数，全部属于调制头。保持一个共享基座，模型总量≤1B、共享比例>80%；调制23类、协议8类、参数七字段接口不变。推理继续只读取task/sample_id/iq，新增分段来自有效长度和模型特征，不读取标签或sample_id数值。

## 验收范围

- 新增6项语义检查通过：有效token分段、空段及padding遮蔽、稳定方差、顺序敏感性、旧模块初始化/其他任务输出一致、FP32和CPU BF16梯度及配置拒错。
- 8项配置检查通过，另53项旧多尺度／阶段架构、warmstart、日志和恢复相关检查通过，共67项。
- data_full实际schema下统计参数量，原17:18的best.pt严格加载成功，原版断点配方检查一致；基线配置文件SHA256保持不变。
- 完整384×8模型在真实小缓存上CPU BF16更新2步，再从last.pt严格恢复到第3步；AdamW状态与cosine恢复流程通过。
- 独立评估入口、单模型导出、CPU推理及官方6行输入输出格式校验通过。只在独立artifacts目录保存测试权重，没有替换正式weights或原日志。

这些结果证明代码路径可执行，不证明准确率提升；未启动正式GPU训练，也未测量A100吞吐或完整data_full验证收益。下一步应从正式训练日志比较调制整体/前14类、Macro-F1、协议指标和七参数RMSE。

记录：`artifacts/pyramid_temporal_implementation/20261113/parameter_report.json`、`pipeline_status.json`、`smoke_config.json`及阶段日志。smoke配置降低了batch/预算并缩短预热，只用于功能验收，不是正式训练建议。

实现位于 `code/models/multiresolution.py` 的SegmentedStatistics、TemporalStatisticsReadout及ModulationReadout；RFModel负责可选开关和旧配置兼容。可运行：

```bash
python code/training/test_temporal_readout.py
python code/training/test_multiresolution.py
```
