# 17:18 历史监督版本保留说明

对应运行：`logs/supervised_main/20261112_171823_031143/`。

## 版本和配置

入口为 `code/configs/supervised/pyramid_v1_171823.json5`。这是原 384×8 Transformer 与中／细尺度分支版本：`pyramid.enabled=true`，`hierarchy.enabled=false`、`hybrid.enabled=false`、`hierarchy.mechanism.enabled=false`。不会创建后续 IQFormer 阶段主干、BiLSTM、统一串行主干或幅相保留改造模块。

参数总数 20,966,006；共享部分 19,153,568，占 91.36%。调制识别 23 类、协议识别 8 类、参数估计 7 个字段共用一个模型。

配置直接复制该运行的 `resolved_config.json`，保留原模型、损失、采样、优化器和调度配方。仅将新运行输出分开，并显式关闭后来新增的结构。`experiment` 保留原标识以通过严格续训检查。单卡 GPU 1、batch 256、梯度累积 1；AdamW 初始学习率 3e-4，500 步预热后 cosine，最低学习率比例 0.05，完整预算 40 轮。

当前 `supervised/main.json5` 继续表示最新架构，未被回退。训练入口也支持原配置，但不要用当前 main 配置加载这一版的完整权重。

## 从零重新训练

在项目根目录、py313 环境执行：

```bash
python code/training/train.py --config code/configs/supervised/pyramid_v1_171823.json5
```

新日志自动保存到 `logs/supervised_pyramid_v1/<时间戳>/`。如需更换单卡，可加 `--gpus 0` 等。

## 从原断点继续

```bash
python code/training/train.py --config code/configs/supervised/pyramid_v1_171823.json5 --resume logs/supervised_main/20261112_171823_031143/last.pt
```

这会恢复模型、优化器、学习率调度和训练进度，并继续写入原时间戳目录；不会新建独立实验。保留原训练配方才能严格续训。已检查的 `last.pt` 保存于 step 42,000；测试没有写回该文件。

如果只继承权重，重新建立优化器／学习率计划并保存到新目录：

```bash
python code/training/train.py --config code/configs/supervised/pyramid_v1_171823.json5 --warmstart-checkpoint logs/supervised_main/20261112_171823_031143/best.pt
```

## 独立源码和推理权重

归档位置：`artifacts/model_versions/pyramid_v1_171823/`。

- `source/`：经旧权重验证的兼容源码、历史配置和环境文件。
- `source_snapshot.tar.gz`：上述源码快照压缩包。
- `weights/`：由原 `best.pt` 导出的独立推理模型；没有覆盖项目共用 `weights/`。
- `original_resolved_config.json` 等：原运行元数据的副本。
- `compatibility_report.json`：严格加载、配方匹配、优化器状态恢复及 CPU 内存更新记录。
- `partial_evaluation.json`：仅验证集 32 行的流程检查，不能视为完整验证精度。
- `archive_manifest.json`：源码、配置、导出权重校验值及原始 checkpoint 引用。

原运行没有逐次源码快照。此处保存的是**当前经实际旧 checkpoint 验证的兼容实现**，不声称逐字恢复了 17:18 的原始源码。模型和训练配方来自该次运行；原训练 checkpoint 仍保留在原日志目录，未重复拷贝到归档。

以后若主代码继续改动，可在项目根目录使用归档源码入口：

```bash
python artifacts/model_versions/pyramid_v1_171823/source/code/training/train.py --config artifacts/model_versions/pyramid_v1_171823/source/code/configs/supervised/pyramid_v1_171823.json5
```

源码快照不含数据集、Python 环境和完整训练 checkpoint，仍需当前项目中的数据缓存及 py313 环境。所有配置相对路径以启动工作目录为基准。

推理时显式选择归档权重：

```bash
python code/infer.py --weights artifacts/model_versions/pyramid_v1_171823/weights --input /path/to/input.parquet --output /path/to/predictions.jsonl --device auto --gpus 1
```

## 已完成的检查

CPU 上严格加载原 `best.pt` 和 `last.pt`，检查原模型配置和训练配方匹配、参数量一致、512／1024 点输出形状及有限数值；恢复 `last.pt` 的 AdamW／cosine／GradScaler 状态，并在内存中用真实三类任务的 6 行数据完成一次反向传播和优化器更新。另检查导出张量往返一致、实际验证入口、官方 6 行示例推理及输出格式。没有启动正式 GPU 训练，也没有重新测定全量验证精度。
