**偏移Patch适配与原始/增强成对训练**

这一版沿前14类定向相位/循环时移增强的有效方向，新增一个共享局部Patch偏移适配器，并同时保留原始视图监督。目标是减少Patch切分敏感性和只看增强视图的类别代价。尚未执行正式GPU训练，不能保证超过67.57%。

一条命令后台启动（默认物理GPU0）：

```bash
bash code/training/run_shift_consistent.sh
```

脚本已指定本项目py313解释器，使用nohup后台运行；会打印PID、启动日志和tail命令，关闭终端后继续。若要使用另一张已分配GPU，例如GPU1：

```bash
C01_GPU=1 bash code/training/run_shift_consistent.sh
```

前台运行：

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
/home/yangqilin/anaconda3/envs/py313/bin/python \
code/training/train_shift_consistent.py \
--config code/configs/supervised/shift_consistent.json5
```

所有相对数据/权重/输出路径均按仓库根目录解析。配置：[shift_consistent.json5](../code/configs/supervised/shift_consistent.json5)。这是专用训练入口，配置不能交给旧train.py。

**结构与起点**

起点固定为上一轮67.570%候选：`logs/supervised_density/20261114_164340_348077/phase_shift_first14/ema.pt`。完整加载三任务权重，新增残差输出零初始化。

原时域第一层卷积后的高分辨率特征，复用原P16卷积核以stride4读取4个局部偏移。每个原Patch位置汇总这些偏移的均值/标准差，经低秩投影产生小残差，注入原时域token；再与原局部频谱、全局频谱、尺度和位置通路融合，进入原768×12 Transformer。4个局部偏移的平均中心保持原Patch中心，序列进入Transformer时仍是原来的64/32 token，不增加另一套完整编码器或推理预测集成。

这里的偏移是样点网格位置，不是载波相位、真实码元周期或物理同步。它只是提高局部切分稳健性的候选，不宣称严格时间平移不变。原有通路保留，以供参数任务读取绝对时间与原始观测；适配器本身也由三任务共同训练。

总参数95,856,525，新增149,056，约为原模型的0.156%；共享参数93,998,272，占98.061%。虽然新增参数很少，密集局部投影仍有额外计算，不能据此称推理零开销。

在实际67.57%权重上验证：512/1024长度，FP32/CPU BF16，零残差模型三任务输出与原模型逐项差为0。这个检查证明初始化兼容，不证明训练后精度。

**原始/增强监督**

默认只对前14类、没有协议或参数标签的纯调制记录构造增强视图。原始视图始终参与，稳定阶段每条目标记录采用：

`0.5 * CE(raw) + 0.5 * CE(augmented) + 0.05 * JS(p_raw, p_augmented)`。

前2000步线性引入增强项和一致性项。其余调制记录、协议与参数继续采用原始监督。损失按原batch的调制记录数归一，避免目标类子集被额外放大；训练时的真实类只决定监督增强资格，推理不读取类别、教师或可靠性文件。

一致性项使用两个视图的23类预测分布，不使用弱教师，也不把概率裁成某个真实家族。两次前向都会训练同一套参数；Dropout仍开启，因此该项也会约束一部分训练随机性。原始/增强分支仅在训练存在，推理一次前向。

**默认长训练预算**

- 单卡batch256，24,000次优化器更新，调制/协议/参数按三任务轮转，各8000次。是有放回采样，不等于完整遍历若干轮。
- 原权重峰值LR2e-5，新适配器2e-4；500步warmup后cosine下降至20%，BF16，梯度裁剪1，EMA0.995。偏置/归一化参数不做weight decay。
- 每2400步，对last与EMA分别做全量118182条原始输入FP32验证；不会平均两者预测。
- 每500步保存含模型、优化器、调度、GradScaler、EMA的last.pt。新文件先原子保存再替换。
- 每个新运行建立上海时区时间戳目录；保存实际配置、源码快照、可读train.log、events.jsonl、TensorBoard及各次验证预测。
- 不因中期无改善提前截断cosine；固定预算完成后导出历史最佳。

成对视图使目标调制批次有额外前向，密集Patch投影也有计算成本；实际耗时和显存需在你的GPU上测量。若需要改batch或总步数，应在首次启动前调整配置；它们影响训练配方，不能在严格resume时更改。

**选模与无人值守保护**

先全量评估新模型的零残差起点，并保存为初始best.pt。2026-11-18起，best.pt统一按三任务本地综合分选取；下列相对起点保护项仅继续报告诊断，不再代替综合分决定主权重：

- 协议Accuracy下降不超过0.3个百分点；
- QAM五类合计Accuracy下降不超过0.5个百分点；
- 调制Macro-F1下降不超过0.3个百分点；
- 任一参数RMSE增长不超过5%。

这些是工程选模条件，不是官方评分公式；也不强制每个类别的召回都不下降。last/EMA的未通过候选仍保留最新权重和逐次评估文件。最终从last/EMA候选中按三任务本地综合分取最高；口径见[综合选模说明](composite_checkpoint_selection.md)。这仍会受到反复使用本地开发集的选择偏差影响。

若始终没有提高，就保留零残差起点，不把最终退步权重当最佳。最后自动输出`completion.json`，其`improved`字段明确是否超过起点。这个保护保住本地指标和历史权重，不能保证隐藏测试收益。

**查看、恢复与推理**

运行目录为`logs/supervised_shift_consistent/<时间戳>/`，父目录`latest_run.json`记录最新创建路径。多次启动会产生多个独立进程，启动脚本不替你排队。

恢复指定运行（替换为实际目录）：

```bash
bash code/training/run_shift_consistent.sh \
  --resume logs/supervised_shift_consistent/<时间戳>/last.pt
```

使用同一配置与训练精度恢复，继续原目录、样本流、优化器、cosine、EMA和步数。只支持该专用入口恢复last.pt；best.pt、last_model.pt、ema_model.pt没有完整优化器断点。旧train.py会拒绝按自己的断点格式恢复这些文件，避免混用。修改架构或训练配方应开新运行，不要伪装成严格resume。

临时限制更新数可加`--max-steps 3000`；它只截断运行，不重定义24,000步调度，后续可恢复。SIGINT/SIGTERM可结束进程，从最近完成的定期断点恢复；不保存反向/更新到一半的状态。

TensorBoard：

```bash
/home/yangqilin/anaconda3/envs/py313/bin/tensorboard \
  --logdir logs/supervised_shift_consistent --port 6006
```

训练结束后单模型权重自动位于`<运行目录>/exported_best/`，原weights目录不覆盖。推理：

```bash
/home/yangqilin/anaconda3/envs/py313/bin/python code/infer.py \
  --weights logs/supervised_shift_consistent/<时间戳>/exported_best \
  --input <待测parquet> --output <输出jsonl> --device cpu --precision fp32
```

保存best.pt也可交给原export.py重新导出；配置包含新适配器，统一RFModel和Predictor均支持。`evaluations/step_<步数>/`分别保存last/EMA的全量预测和指标，best.pt同时保存其对应指标与权重类型。

**验证记录**

专项行为测试覆盖原始/增强损失的精确权重、非目标记录分母、协议/参数保护、能量/有效长度、偏移适配器padding、零残差热启动与拒绝重解释、选模条件。真实140条小缓存和64宽2层模型完成六步三任务训练、EMA、全量小缓存评估和导出；分三步中断再恢复，与连续六步的模型、EMA、优化器和调度逐张量一致。另验证12条512/1024、三任务统一Predictor推理。

证据目录：`artifacts/shift_consistent_implementation/20261114_200545_747483/`。这些是CPU实现与兼容性检查，测试子集成绩不用于方法收益结论。本次没有启动正式GPU训练。
