# 单前端统一基座：监督训练

当前入口仍为`code/configs/supervised/main.json5`，`hybrid.topology=serial`。当前进一步启用hierarchy.mechanism：局部时序建模、六值幅相保护、全局谱后移和深层主导调制读出。具体机理与实测见[本轮优化说明](signal_mechanism_optimization.md)。当前模型共用一套时频前端，随后串接BiLSTM、阶段编码与原Transformer核心。前一版双前端并行结构已不再是默认入口。

## 启动

```bash
python code/training/train.py --config code/configs/supervised/main.json5
```

保留用户选择的GPU1，单卡batch256，梯度累计1。换卡追加`--gpus 2`。训练数据仍为artifacts/data_full，无需重建缓存或词表。

输出自动保存到`logs/supervised_main/<上海时间戳>/`，包括pt、resolved_config.json、train.log、events.jsonl和metrics.json。从零监督初始化，未加载旧pt；不会自动替换现有weights。

同一单前端架构恢复：

```bash
python code/training/train.py --config code/configs/supervised/main.json5 \
  --resume logs/supervised_main/<时间戳>/last.pt
```

原单主干、阶段替换版和并行组合版的完整pt都不能直接resume为当前模型；它们的前向结构不同。保留的是原Transformer核心结构和参数接口，不保证旧完整主干的表征数值或权重迁移收益。

## 当前结构与配置

```text
原始I/Q＋有效长度
 → 一次RMS处理
 → 共享时域流／共享频域流（I-STFT实虚、复I/Q局部功率谱、全局谱）
 → 一次门控时频融合
 → 两层局部BiLSTM，128原样点窗口
 → 阶段1：1024×64，1块
 → 阶段2：512×128，2块
 → 阶段3：256×192，2块
 → token投影：64×384
 → 原Transformer核心：8层、6头
 → 深层主导多层读出＋六值幅相memory → 23类调制／8类协议／七参数
```

上图以1024点输入为例，512点的时间长度均减半。统一读取Transformer末层、阶段3、阶段1，三个来源分别为64×384、256×192、1024×64；共同投影192维。没有第二套原始时频提取，没有独立支路预测器。

- `model.encoder`控制后半段Transformer的width/depth/heads；扩到768/12时前端无需同倍率扩宽。
- `model.hierarchy`控制统一主干前半段：时频视图、BiLSTM、阶段宽度/深度、下采样、跨层读出尺寸。
- `model.hybrid.topology=serial`选单前端串行结构；parallel仅保留给历史组合模型兼容，不作为当前方案。
- `signal.patch_size=16`为Transformer token相对原样点的累计步长；阶段先压缩4倍，输入投影再压缩4倍。
- 当前局部谱窗为31/63点且centered_windows=true，使两种局部谱居中到同一原样点。全局谱计算一次但后移至跨层读出；阶段注意力改为16-token局部窗。
- 任务头规模位于`model.heads.multiscale_readout`。

默认AdamW峰值3e-4、500步warmup＋cosine最低比例0.05、40epoch上限；8次全量验证未达到0.1个百分点的显著改善则早停。任务平衡采样，CE、有效参数MSE、0.1网格辅助损失；未加入FR、SupCon或波形重建。若GPU显存不足可降低batch，累计仍为1；新配方不能严格resume原配方。

## 组件兼容性审计

详细尺寸、窗口/位置、精度、梯度及参数头分析见[逐组件审计](unified_architecture_audit.md)。

按data_full完整类别/网格：总参数19,171,989，共享17,448,944，共享91.0127%。训练和导出继续核验参数≤1B、共享>80%。所有任务使用同一前端与编码主干。

可自行复查：

```bash
python code/training/test_signal_mechanisms.py
python code/training/audit_components.py --config code/configs/supervised/main.json5
```

审计在CPU用合成I/Q及实际schema/网格检查组件契约，不测量准确率或GPU性能。报告在artifacts/unified_component_audit中，含逐组件FP32/BF16输入输出和梯度。

## 导出与推理

训练结束后选一个精确checkpoint：

```bash
python code/training/export.py \
  --checkpoint logs/supervised_main/<时间戳>/best_modulation.pt --output weights
python code/infer.py --input <输入.parquet> --output <输出.jsonl> \
  --weights weights --device cuda --gpus 1 --processes 1
```

上述export会替换weights中的文件，应在选定模型后执行。推理仍只读取task/sample_id/iq，参数回原始单位、七字段全输出，不读取有效性标签。代码功能验收不代表已完成国产平台实测、Docker或全部比赛交付。

源码：rf_model.py中的UnifiedBackbone；共享前端与阶段编码在hierarchical.py；任务读出在multiresolution.py。
