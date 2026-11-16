# 金字塔共享基座：监督训练

2026-11-13：当前骨干优化候选为 `code/configs/supervised/pyramid_midfusion.json5`，保留原金字塔，在P16第4层后加入单次共享瓶颈交互，后4层继续编码。固定原版任务头；分段头关闭。原版和分段版配置保留作基线／既有运行续训。[主干实现](pyramid_midfusion.md)、[分段头历史候选](pyramid_temporal.md)。

## 启动与恢复

在项目根目录、py313环境执行：

```bash
python code/training/train.py --config code/configs/supervised/pyramid_midfusion.json5
```

GPU1单卡、batch256、梯度累积1，数据为 `artifacts/data_full`。新训练保存到 `logs/supervised_pyramid_midfusion/<上海时间戳>/`；无需重建缓存或词表。换卡可追加 `--gpus 2`。配置默认从零监督，不会自动加载预训练或替换部署权重。

恢复原版金字塔运行的完整状态（继续使用原版配置）：

```bash
python code/training/train.py --config code/configs/supervised/pyramid_v1_171823.json5 \
  --resume logs/supervised_main/20261112_171823_031143/last.pt
```

resume继续写原时间戳目录，并恢复优化器、cosine、混合精度及进度。恢复中期交互运行时使用pyramid_midfusion配置及该运行的精确last.pt；分段头运行继续用pyramid_temporal配置。原版、分段版和中期交互版不能交叉resume。严格resume要求训练配方、数据和架构一致；已撤下串行架构的pt不能直接用于此模型。

## 当前结构

```text
原始I/Q＋有效长度
 → RMS归一化
 ├ 原时频前端＋幅度/相位差特征＋log RMS侧信息
 │ → P16 token → Transformer前4层 → 中期交互 → 后4层 → 粗尺度
 └ 共享复数stem
   ├ 步长8的中尺度分支：局部卷积块＋一层Transformer
   └ 步长4的细尺度分支：局部卷积块＋一层Transformer
 → 中/细分支读取原频谱，三尺度中期交互后保留各自token
 → 原末端跨尺度摘要融合
 → 原三尺度调制统计读出／协议摘要／七字段独立参数查询
```

1024点输入的三尺度尺寸为64×384、128×192、256×192；512点的token长度减半。全部任务读取同一共享基座的多尺度输出。

- `model.encoder`：原Transformer核心，当前384×8、6头。
- `model.signal`：原时频前端，P16、局部谱窗64、相位差特征、全局谱和log RMS尺度嵌入。
- `model.pyramid`：中／细尺度复数分支，宽192，各两个局部块和一层Transformer；开启状态。
- `model.heads.multiscale_readout`：三尺度投影、分类器和七字段MLP尺寸。
- `model.mid_fusion`：第4层后单次交互，8个192维token、3头、增益初值0.1；分尺度读取、等权汇总、同时写回。
- `model.heads.modulation_temporal.enabled=false`：当前固定原版任务头，不叠加分段聚合。
- `model.hierarchy.enabled=false`、`model.hybrid.enabled=false`：不创建串行BiLSTM阶段主干。默认配置解析出来的未启用参数不参与该模型。

当前候选总参数21,565,238，共享19,752,800，占91.60%；比原版增加599,232个共享参数，原任务头规模不变。训练与导出继续检查总量≤1B、共享>80%。调制输出23类、协议8类，参数输出七字段及原单位逆变换不变。

## 训练配方

从零监督；AdamW峰值学习率3e-4、weight decay 0.01；500步线性预热后cosine衰减至峰值的5%；40轮上限。单卡batch256、梯度累积1，三任务均衡采样。损失为分类CE、有效参数MSE和权重0.1的网格辅助项。未引入新预训练目标、对比损失或额外模型。

每轮全量FP32验证，按调制准确率保存best，同时保留协议和七参数指标、best_modulation.pt及best_proxy.pt。8次全量验证未达到0.1个百分点的显著改善时早停。终端和train.log保留训练总览、逐轮进度及完整分类表，机器记录单独保存在events.jsonl。

## 评估、导出与推理

把下方精确checkpoint替换为自己选定的新运行权重；示例评估原金字塔best.pt：

```bash
python code/training/evaluate.py --data artifacts/data_full \
  --checkpoint logs/supervised_main/20261112_171823_031143/best.pt \
  --output artifacts/pyramid_evaluation.json --device cuda --gpus 1

python code/training/export.py \
  --checkpoint logs/supervised_main/20261112_171823_031143/best.pt \
  --output artifacts/pyramid_export

python code/infer.py --input <输入.parquet> --output <输出.jsonl> \
  --weights artifacts/pyramid_export --device cuda --gpus 1 --processes 1
```

推理只读取task/sample_id/iq，不读取标签或训练manifest。导出不需要GPU。原最佳权重的独立导出已保存在 `artifacts/model_versions/pyramid_v1_171823/weights`。

## 已有结果与实现

原运行第10轮调制准确率64.34%、Macro-F1 64.14%；同一checkpoint协议准确率94.41%、Macro-F1 94.38%。这些是原版本地验证指标，不是官方成绩，也不代表新版收益；新版只有CPU短流程验收，尚未正式GPU训练。

源码在 `code/models/rf_model.py` 和 `code/models/multiresolution.py`。可用既有CPU检查验证多尺度接口：

```bash
python code/training/test_multiresolution.py
python code/training/test_mid_fusion.py
```

历史配置、严格续训与源码快照详见 [保留说明](pyramid_v1_171823.md)。分类退步的分析另保留为 [诊断记录](architecture_regression_20261113.md)，不作为当前架构推荐。
