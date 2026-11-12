# 结构预训练与任务适配执行指南

配置已经按用途分目录，同一实验使用独立完整JSON，不再反复改同一份pretrain_structure.json切换轮次。总目录、旧文件迁移表和参数说明见[配置索引](../code/configs/README.md)。当前没有全量实验能够证明第二轮一定优于第一轮，或继承初始化一定优于随机初始化。

## 实现流程

| 阶段 | 配置 | 实际作用 |
|---|---|---|
| 第一轮词表 | codebook/round1.json | 训练侧原始I/Q Patch特征聚类，学习BPE，给训练与验证赋固定标签 |
| 第一轮预训练 | pretrain/round1.json | 随机初始化，预测被遮挡位置的Atomic/Emergent结构标签 |
| 第二轮词表 | codebook/round2.json | 第一轮冻结编码器提取Token特征，重新聚类、学习BPE并生成标签 |
| 第二轮继承 | pretrain/round2_continue.json | 加载第一轮共享基座，新建结构头、优化器和cosine周期 |
| 第二轮随机初始化 | pretrain/round2_scratch.json | 使用同一份第二轮标签，全模型随机初始化 |
| 下游微调 | finetune/同名实验.json | 加载相应预训练共享基座，以真实任务标签训练下游头及基座 |

第一轮局部特征先按整条信号RMS归一化，再按Patch向量单位范数归一化。默认512个原子中心、最多512次相邻单元合并，组合最大展开5个原子位置；实际合并数以词表codebook.json为准。组合编号回填原Token位置，不把输入改成可变长度文本Token。

聚类与BPE拟合只用训练侧；验证侧只按固定规则赋标签。第二轮教师在eval与inference_mode下离线提取特征，不参与学生训练或最终推理。本项目依据成果介绍适配，未核实作者原文的第二轮初始化细节。

学生的I/Q、包络、相位、STFT、全窗谱及RMS都来自同一个受遮挡输入，完整信号仅用于离线标签与目标，防止双流泄漏。真实任务标签不进入结构预训练损失，任务元信息仍参与均衡抽样。当前没有真实采样率，使用样点RoPE，不称为物理时间RoPE-fs。

## 第一轮

在项目根目录激活py313环境后运行：

```bash
python code/training/build_codebook.py --config code/configs/codebook/round1.json
python code/training/train.py --config code/configs/pretrain/round1.json
```

词表输出为artifacts/structure_research_01/codebook_v1，预训练输出父目录为logs/pretrain_v1。已有完整第一轮词表可跳过构词步骤；不要覆盖已有词表。新训练会创建新的时间戳子目录。

第一轮即可做下游适配：

```bash
python code/training/train.py --config code/configs/finetune/round1.json
```

它读取logs/pretrain_v1的latest_run.json并加载对应best.pt。要固定某一次实验，将run.init_checkpoint改成其精确.pt路径。

## 第二轮词表

codebook/round2.json已经与pretrain/round2_continue.json引用同一份第一轮检查点：logs/pretrain_v1/20261111_234233_793915/best.pt。换第一轮来源时，两处配套修改。如果第一轮仍在训练，等结束或复制一份固定检查点，避免best.pt在构词和训练之间更新。

```bash
python code/training/build_codebook.py --config code/configs/codebook/round2.json
```

输出为artifacts/structure_research_01/codebook_v2，完成时显示feature_kind: encoder；第一轮为raw_iq。已构好的第二轮词表可直接复用。构词入口会拒绝非空输出目录，需要重新构建时使用新版本目录，并同步修改两个第二轮预训练配置的run.codebook。

## 第二轮的两种初始化

继承第一轮基座：

```bash
python code/training/train.py --config code/configs/pretrain/round2_continue.json
python code/training/train.py --config code/configs/finetune/round2_continue.json
```

预训练输出logs/pretrain_v2，微调输出logs/finetune_v2。run.init_checkpoint非空，代码仅加载backbone；即使新旧词表都为512/1024，编号含义也已改变，因此重置结构头、优化器和cosine周期，不使用第一轮resume。

随机初始化对照：

```bash
python code/training/train.py --config code/configs/pretrain/round2_scratch.json
python code/training/train.py --config code/configs/finetune/round2_scratch.json
```

预训练输出logs/pretrain_v2_scratch，微调输出logs/finetune_v2_scratch。两个初始化路径及resume_from均为空，使用与继承方案相同的codebook_v2。第二轮两份配置起始模型、损失、预算、学习率计划与GPU设置一致，只有初始化、名称和输出不同；后续可以分别研究适合的学习率。

保留第一轮基线及两种第二轮结果，按相同下游流程比较调制Accuracy/Macro-F1、协议识别与七参数误差。新旧词表的结构CE不能直接比较绝对大小判断改进。

## 评估、导出与推理

三份共用工具配置保留原先第一轮微调来源。评估第二轮时，先将deployment/evaluation.json和deployment/export.json的run.checkpoint改为logs/finetune_v2或logs/finetune_v2_scratch，输出路径按实验区分。再把deployment/inference.json的run.weights指向对应导出目录。

```bash
python code/training/evaluate.py --config code/configs/deployment/evaluation.json
python code/training/export.py --config code/configs/deployment/export.json
python code/infer.py --config code/configs/deployment/inference.json
```

最终推理只读取一个导出的共享模型及其配置，不读取训练缓存、词表或教师。训练运行目录的best.pt按配置指定指标选取，预训练best默认来自固定掩码验证损失，不保证是下游最佳模型。

## 预算、参数与恢复

- 三份结构预训练配置沿用整理时用户的当前配方：768宽、12层、6头、Patch16、STFT64；80 epochs、每卡batch256、Atomic/Emergent各1，重建与谱权重0。第一轮新配置不是历史运行预算的精确复现。
- 微调沿用分阶段配方：任务头500更新、上层1500更新、其余全参，总10000更新；头部峰值LR1e-4、基座1e-5、全程cosine。详见[微调说明](finetuning.md)。
- reconstruction.json保留原重建对照预算200 epochs；和结构路线做公平研究时，需要主动匹配训练预算及架构，不能直接按现有轮数比较。
- 掩码默认比例0.3、跨度2、random策略；跨度为初始连续Token段长度。损失在遮挡且有效的位置计算，先按样本归一，再求均值。
- 扩参时保持预训练与相应微调的encoder/signal和位置编码配置一致。当前supervised/main.json为384宽，与768宽预训练基座不能直接互载。
- 训练/推理runtime.gpus="2"、processes=1为单卡；"2,3"和2为双卡，也支持0/1、4/5。构词表、独立评估单进程，导出CPU。有效训练batch=每卡batch×进程数×梯度累积数。见[兼容性说明](device_compatibility.md)。
- 同阶段断点恢复在对应实验JSON中填写checkpoint.resume_from为原last.pt，继续写原时间戳目录；进程数、词表、batch及训练配方应一致。换词表或开展下游适配使用init_checkpoint；同词表全模型热启动使用warmstart_checkpoint。
- 所有相对路径从项目根目录启动时解析，不相对于JSON所在子目录。配置修改不会影响已启动的进程。
