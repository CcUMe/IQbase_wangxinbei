# 95M 模型退化：定位、修复与真实数据验证

日期：2026-11-10。全部分数为本地代理分，不是官方比赛成绩。最终检查点、指标和导出路径见 [repair_summary.json](../artifacts/attention_repair/repair_summary.json)。

后续更新：用户重训 95M 后的最新诊断和默认权重选择见 [2026-11-11 泛化报告](generalization_20261111.md)。本文的 17M/74.68 分权重选择是当时记录，当前默认已更新为新训练的 95M。

## 1. 已复现的主要原因：注意力温度失控

原模型在 Transformer 输入使用 Pre-LayerNorm，但线性投影后的 Q/K 没有尺度约束。训练中投影权重增长会提高 Q/K 范数，使 `QKᵀ / sqrt(head_dim)` 的温度不断变尖。输入归一化并不限制投影输出幅度。

在相同训练批次上，95M 原模型退化检查点的末层注意力分数达到约 7 万（通信）至 17 万（雷达），归一化熵约 0.006–0.007，约 95% 注意力接近单点硬选择。17M 对照的相应分数约几十，注意力更分散。95M 的最佳检查点在精度明显退化前已经出现严重饱和。

修复为逐头 Q/K 单位 RMS 归一化，在 FP32 中计算，再交给注意力算子。每头维度 96 时，理想实数计算下注意力分数绝对值上界约 sqrt(96)=9.8，避免用投影范数隐式提高 softmax 温度。归一化不增加模型参数。

修复后检查点的相同批次末层 Q/K RMS 为 1，分数约 8.5，归一化熵约 0.53–0.62，检查到的近单点注意力比例为 0。

原始定位：[large_best_gradients.json](../artifacts/model_scale_diagnosis_20261110/large_best_gradients.json)、[large_last_gradients.json](../artifacts/model_scale_diagnosis_20261110/large_last_gradients.json)、[small_best_gradients.json](../artifacts/model_scale_diagnosis_20261110/small_best_gradients.json)。修复后：[gradient_diagnosis.json](../artifacts/attention_repair/repaired_2400/gradient_diagnosis.json)。这些是固定批次诊断，不是全验证集的注意力分布统计。

## 2. 已修复的实现错误：无标签任务头被 AdamW 更新

原实现为了满足多任务 DDP 的梯度同步，用零梯度连接保留全部输出头。这能避免 unused-parameter 错误，但零梯度不等同于 grad=None：AdamW 仍可能根据历史动量和权重衰减修改当前没有监督的任务头。

回归测试先让调制头产生动量，再进行只有体制标签的更新：原实现会改变调制头；修复后该头权重逐项不变。

修复在反向传播、DDP 同步之后，将本次更新没有有效目标的头的 grad 设为 None，再执行裁剪与优化器更新。梯度累积取微批次并集，DDP 跨 rank 取活动并集，避免各卡优化器状态不一致。共享基座正常更新。

代码：[head_optimizer.py](../code/utils/head_optimizer.py)、[train.py](../code/training/train.py)。

## 3. 等预算因果对照

两组均从同一个 95M 最佳检查点开始，重置优化器，使用相同数据划分、采样顺序、每次 256 条样本、学习率 3e-4 和 2400 次更新。两组都重置优化器，因此不能把优化器重置的收益单独归因于代码修复。

固定验证子集用于过程监测，最终两组都评估全部 118,182 条验证记录：

| 指标 | 原实现继续训练 | QK 归一化 + 任务头保护 |
|---|---:|---:|
| 调制准确率 | 52.26% | 57.30% |
| 体制准确率 | 89.70% | 93.59% |
| 参数代理精度 | 0.6862 | 0.7078 |
| 总代理分 | 70.03 | 73.58 |
| 最后 400 次更新的裁剪前梯度范数 P95 | 12.51 | 1.29 |

七项原始单位 RMSE 均下降：码元率约 10.5%、带宽 9.8%、频移 29.2%、脉宽 7.7%、PRI 7.0%、时延 10.4%、脉冲数 11.3%。

证据：[原实现全量验证](../artifacts/attention_repair/control_2400/full_validation.json)、[修复全量验证](../artifacts/attention_repair/repaired_2400/full_validation.json)。过程曲线：[repair_comparison.png](../artifacts/attention_repair/repair_comparison.png)。

## 4. 进一步精调与未采用的改动

同一起点额外 1200 次更新，固定验证子集上的代理分：继续使用 3e-4 为 73.63，改用 1e-4 为 74.47。低学习率精调模型在全量验证集上为 74.58。最终候选再以不同采样种子补充训练，并做完整验证；具体选择记录在 summary 中，不能把额外训练的全部收益归因于第一阶段代码修复。

最终选中的 95M 模型，全量代理分 74.6765，调制准确率 59.83%，体制准确率 94.08%。额外验证 QK scale=0.75、1.0、1.25、1.5、2.0 后，没有找到优于 1.0 的设置，因此保持 1.0。完整 95M 修复检查点和 safetensors 均已保存。

17M 基线在当前代码下重新完整验证，代理分 75.1197，仍略高且参数量更小。因此默认 weights/ 采用该 17M；95M 修复权重在 artifacts/attention_repair/weights/ 单独保留。模型按真实验证结果选择，不采用多模型集成。

融合分支幅度也有增长，但直接给已有权重开启融合归一化会改变其特征分布，短测精度明显下降。最终修复保持 normalize_fusion=false；没有把所有“稳定化开关”一起强行打开。

原 200 轮 cosine 计划使学习率长时间接近峰值，是风险放大因素。等预算实验不改变两组学习率，却仍观察到修复收益，因此根因不能只归结为 epoch 数。当前已在主配置启用 QK 归一化与非活动头保护，并提供较低学习率的修复配置。

## 5. 使用与复现

已提供 [main_large_repaired.json](../code/configs/supervised/comparisons/large_repaired.json)。完整热启动保留任务头，重置优化器与调度器：

```bash
C01_PY=/home/yangqilin/anaconda3/envs/py313/bin/python
CUDA_VISIBLE_DEVICES=2,3 "$C01_PY" -m torch.distributed.run \
  --standalone --nproc_per_node=2 code/training/train.py \
  --data artifacts/data_full --config code/configs/supervised/comparisons/large_repaired.json \
  --warmstart-checkpoint artifacts/attention_repair/final/model.pt \
  --batch-size 128 --output logs/repaired_main
```

两卡每卡 128，使有效 batch 仍为 256。新训练仍自动创建时间戳目录。旧进程已经加载的模型不会因修改源文件而自动改变；恢复原配方使用 resume，切换修复配方使用 warmstart-checkpoint。原 init-checkpoint 只加载基座，会重新初始化任务头，不能代替这里的完整热启动。

梯度定位可用 diagnose_gradients.py；受控实验可用 repair_experiment.py。新增 test_stability.py 覆盖注意力分数有界、融合尺度有界、无标签头动量保护及遮挡目标泄漏。CPU 两进程和 GPU 2,3 双卡验收均通过；原核心与配置检查保持通过。

## 6. 结果边界

本次已复现并修复注意力饱和与非活动头错误更新，并获得真实数据的稳定性和精度改善。17M 基线为 75.12，最终 95M 为 74.68；保留小模型作为当前默认推理方案。扩参收益尚未得到证明，当前没有完成修复版 95M 从零完整重训，也没有证明结构预训练的额外收益。

验证集参与模型选择，结果不代表隐藏测试成绩。CUDA12、国产平台和最终 Docker 验收仍属于后续提交工作。
