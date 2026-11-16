# 官方IQFormer在C01上的独立调制识别基线

从项目根目录运行以下命令。默认使用现有 `artifacts/data_full`，无需重新制作数据；路径也可在配置中修改。

```bash
conda activate py313
CUDA_VISIBLE_DEVICES=1 python compare/IQFormer/train_c01.py --config compare/IQFormer/c01.json5
```

这里只训练23类调制识别，是用于比较的独立基线；不输出协议和七个参数，不作为完整三任务竞赛提交程序。
已有主模型、预训练权重和配置不受影响。

## 下载来源与模型保真

- 上游：[WestdoorSad/IQFormer](https://github.com/WestdoorSad/IQFormer)。
- 固定提交：`7ee6ac949551b24d45f218762cab919e0cb6b4f9`。
- `upstream/`：完整官方GitHub下载，文件保持原样。
- `c01/official_model.py`：同一网络源码，仅重定向两行timm工具导入，并将无维度的`squeeze`改为`squeeze(2)`，使batch=1也可运行。
- `c01/compat.py`：提供DropPath、截断正态初始化及未被使用的注册器占位，无需安装torchvision/timm；不改变网络层、参数或计算拓扑。
- `source_info.json`：上游提交、原文件SHA256、适配文件SHA256及变更说明。
- `LICENSE`及`upstream/LICENSE`：保留原作者MIT许可证。

保留官方I/Q分支、STFT分支、融合、双层BiLSTM、编码器和原分类头，仅将类别数设为23。
兼容修复前后的模型在batch≥2、eval模式下输出逐值相同，见`test_c01.py`。

**使用作者实际发布的代码配方，而非声称完全复现论文表格。**
下载的`main.py`长输入构造为layers=[3,3,3]、widths=[64,64,64]、dropout=0.2、drop_path=0.2；论文表格与之有差异。
下载的`dataset/dataset.py`对包括长输入在内的信号均使用31点窗，论文HisarMod表格写61点。
当前默认按代码选31；配置允许61，但这是明确的输入设置变化。

## 输入与数据组织

- 读取已有缓存的modulation标签行：训练186,113条，验证20,885条，沿用主模型已有的hash划分与23类顺序。
- 512/1024点按原长度分桶，既不补零混批，也不截短信号；所有记录都参与。
- 一个batch只含调制样本，一个epoch默认无放回遍历全部调制训练记录。
- 主模型原先的task-balanced epoch包含三任务更新，调制曝光量不同；比较收敛速度应看调制样本曝光和更新次数。
- 默认不做每条RMS归一化，不加相位增强，与官方数据接口一致。输入BatchNorm仍保留。
- 默认STFT为`scipy.signal.stft(I, fs=1, window='blackman', nperseg=31, noverlap=30, nfft=128)`的前32个频率bin。
- **STFT使用实部**：官方`torch.Tensor(complex)`会丢弃虚部，因此显式取`.real`复现该行为，不能替换成abs或log功率谱。
- `fs=1`是与官方代码相同的归一化分析坐标，不代表已知C01真实采样率为1 Hz。
- 默认不使用协议或参数标签，也不编造SNR分组。

## 配置

只需维护`c01.json5`一个配置，支持注释，按run/data/input/model/training/optimizer/scheduler/runtime/logging分组。

默认训练配方取下载代码的AdamW LR=1e-3、weight_decay=0.01、无标签平滑、无梯度裁剪、FP32。
调度为验证CE驱动的ReduceLROnPlateau，factor=0.5、patience=3、min_lr=5e-5；训练60轮，验证Acc连续10轮不提升时停止。
梯度范数仍会记录；若显式启用裁剪，改变`training.gradient_clip`。

`scheduler.name="cosine"`提供可选cosine及按epoch预热，此时依赖`training.epochs`、`scheduler.warmup_epochs`和`scheduler.min_lr`。
启用cosine、RMS归一化或其他可选项应视为单独配方变化，默认不启用。
BF16仅在支持的CUDA设备上可选；当前执行检查为CPU FP32，正式CUDA吞吐/精度尚未测量。

## 终端、日志与权重

训练先输出总览，再显示每epoch的step进度，右侧显示CE、训练Acc、LR和裁剪前梯度。
每轮打印全量调制Accuracy、Micro/Macro/Weighted-F1、前14类/后9类结果，以及23类Precision/Recall/F1/Support。

输出：`logs/compare_iqformer/<上海时间戳>/`，包含：

- `train.log`：可读日志；`events.jsonl`：结构化step/epoch日志。
- `resolved_config.json`、`source_info.json`、`parameter_report.json`。
- `last.pt`：最后一个已完成epoch的模型、AdamW、调度、RNG等状态。
- `best.pt`、`best_metrics.json`：验证调制准确率最高的权重和对应指标。
- `metrics.json`、`per_class.csv`、`confusion_matrix.npy`：最近一次验证结果。
- `tensorboard/`：损失、LR、梯度、整体/逐类/前14类与后9类指标。

```bash
python -m tensorboard.main --logdir logs/compare_iqformer --host 127.0.0.1 --port 6006
```

## 续训

```bash
CUDA_VISIBLE_DEVICES=1 python compare/IQFormer/train_c01.py \
  --config compare/IQFormer/c01.json5 \
  --resume logs/compare_iqformer/实际时间戳/last.pt
```

从已完成的epoch恢复；为保留历史，新建时间戳目录，start事件记录源checkpoint。
数据指纹、类别、模型源码和训练配方必须匹配。默认配置的CPU短程检查验证了中断后续训与连续训练逐张量一致。
不能通过改总epochs后resume改变cosine计划。已完成全部计划轮次的权重可直接评估，不接受假装续训。
若启用相位增强并使用多worker，请设persistent_workers=false，以便按epoch重建可恢复的worker随机状态。
恢复时会复制仍可核验的历史best.pt；若源目录已缺少对应best，历史最优仍以源记录为准，新目录只保存后续真正超过历史最优的best。

## 独立验证与逐记录结果

```bash
CUDA_VISIBLE_DEVICES=1 python compare/IQFormer/evaluate_c01.py \
  --checkpoint logs/compare_iqformer/实际时间戳/best.pt \
  --save-predictions \
  --output artifacts/iqformer_evaluation_01
```

输出目录必须不存在。默认全量验证；可用`--split train`评估训练记录。
`predictions.npz`含缓存全局rows、真实labels、23维logits，可与其他模型按row逐样本配对。
不重新划分数据，不用验证标签选择接收处理。

## 官方Parquet推理

```bash
CUDA_VISIBLE_DEVICES=1 python compare/IQFormer/predict_c01.py \
  --checkpoint logs/compare_iqformer/实际时间戳/best.pt \
  --input /mnt/disk1/wangxinbei_contest_dataset/C01/rf_signal_sample_io_v1.0/rf_signal_sample_io_input_v1.0.parquet \
  --output artifacts/iqformer_modulation_predictions.jsonl
```

仅读取task/sample_id/iq，输出`{"task":"modulation","sample_id":"...","prediction":"OOK"}`。
输入若含协议/参数记录会跳过并报告数量；官方6条样例应输出2条调制预测，不伪造其他任务结果。
输出文件必须不存在，以保护已有结果。

## 执行检查

```bash
CUDA_VISIBLE_DEVICES='' python compare/IQFormer/test_c01.py -v

CUDA_VISIBLE_DEVICES='' python compare/IQFormer/train_c01.py \
  --config compare/IQFormer/c01.json5 --device cpu --epochs 1 --num-workers 0 \
  --max-train-batches 2 --max-val-batches 2 --output artifacts/iqformer_cpu_smoke
```

有限批次检查会明确标注非全量验证，不能用其准确率判断模型效果。
默认配置不限制批次数。正式训练需使用最上面的正常命令。
如需补依赖：`python -m pip install -r compare/IQFormer/requirements.txt`；当前py313环境已补齐einops，其余所需依赖已存在。
