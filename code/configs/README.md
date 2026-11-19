# 配置目录与参数说明

当前统一入口为 `python code/training/train.py --config <配置>`：

2026-11-19参数读出候选：[`finetune/receiver_parameter_v1.json5`](finetune/receiver_parameter_v1.json5)、[`supervised/receiver_parameter_v1.json5`](supervised/receiver_parameter_v1.json5)、[`pretrain/receiver_parameter_v1.json5`](pretrain/receiver_parameter_v1.json5)共用原receiver_v1骨干，新增七项独立参数回归头、query统计及样点位置读出。监督/微调提高参数损失权重，并仅对纯参数记录加入50%整窗相位增强；best仍按三任务综合分选择。见[配置、启动和验证说明](../../docs/parameter_readout_v1.md)。尚无正式精度提升结论，旧运行与配置保留。

2026-11-18定向第二轮：[`pretrain/refine_receiver_targeted_dual.json5`](pretrain/refine_receiver_targeted_dual.json5)使用固定第一轮教师重构词表，经审计后在GPU2、3执行第二轮50轮；构词及训练均提高C01前14类的Token份额。独立微调入口为[`finetune/receiver_targeted_round2.json5`](finetune/receiver_targeted_round2.json5)，原监督入口和共享骨干保留。详见[配额、启动及恢复](../../docs/receiver_targeted_round2.md)。

2026-11-18第一轮迁移筛查：[`finetune/r1_comparison_gpu0.json5`](finetune/r1_comparison_gpu0.json5)在GPU0依次微调当前联合第一轮、旧C01第一轮固定权重；同架构、同任务头seed、单卡batch512，保留20轮调度并先限步到5轮。记录前14类Acc/F1、协议及七项参数指标，完成后自动写对照报告。详见[运行与恢复](../../docs/r1_finetune_comparison_gpu0.md)。

2026-11-17联合数据候选：[`pretrain/two_rounds_p8_receiver_v1_mixed_dual.json5`](pretrain/two_rounds_p8_receiver_v1_mixed_dual.json5)使用C01＋RadioML联合词表，GPU2、3双卡50＋50轮后停止；独立C01微调用[`finetune/shared_base_p8_receiver_v1_mixed.json5`](finetune/shared_base_p8_receiver_v1_mixed.json5)，从头监督用[`supervised/shared_base_p8_receiver_v1_mixed.json5`](supervised/shared_base_p8_receiver_v1_mixed.json5)。三入口共享同一receiver_v1定义。数据划分和恢复见[联合预训练说明](../../docs/mixed_c01_radioml_pretraining.md)。

2026-11-17接收前端v1候选：先使用[`supervised/shared_base_p8_receiver_v1.json5`](supervised/shared_base_p8_receiver_v1.json5)从头监督测试。新增复FIR与可迭代候选采样轨迹，以零初始化残差接入原P8共享基座；默认GPU0、batch256、20轮。三阶段配置已同时提供：[`pretrain/shared_base_p8_receiver_v1.json5`](pretrain/shared_base_p8_receiver_v1.json5)、[`pretrain/two_rounds_p8_receiver_v1.json5`](pretrain/two_rounds_p8_receiver_v1.json5)、[`pretrain/shared_base_p8_receiver_v1_round2.json5`](pretrain/shared_base_p8_receiver_v1_round2.json5)、[`finetune/shared_base_p8_receiver_v1.json5`](finetune/shared_base_p8_receiver_v1.json5)。本版尚无C01精度增益结论，旧配置保留。见[实现、参数及检查说明](../../docs/receiver_frontend_v1.md)。

| 用途 | 正式配置 |
| --- | --- |
| P8从头监督 | [supervised/shared_base_p8.json5](supervised/shared_base_p8.json5) |
| P8独立第一轮 | [pretrain/shared_base_p8.json5](pretrain/shared_base_p8.json5) |
| P8独立第二轮 | [pretrain/shared_base_p8_round2.json5](pretrain/shared_base_p8_round2.json5) |
| P8两轮预训练 | [pretrain/two_rounds_p8.json5](pretrain/two_rounds_p8.json5) |
| P8独立微调 | [finetune/shared_base_p8.json5](finetune/shared_base_p8.json5) |

两轮LR-v2只执行50轮→新教师构词→50轮，然后停止；微调独立20轮。两轮保留当前GPU2,3、每卡256；第一轮6e-4/6e-4，第二轮骨干3e-4/结构头6e-4，均10%warmup+cosine。微调默认单卡GPU0、batch256，骨干1e-4/任务头3e-4，3%warmup+cosine。三阶段共享骨干；P16的`shared_base.json5`及`pipeline/`历史实验配置保留。旧流程配置中的`configs.finetune`只为读取历史配置保留，在统一入口下不参与预训练，也不会触发微调。详细参数、初始化、恢复及双卡设置见[统一训练说明](../../docs/unified_training.md)。

独立第二轮只需配置`run.init_checkpoint`（第一轮固定权重）及`run.codebook`（该教师生成的新词表），再执行统一`train.py`。未构词时先用[codebook/shared_base_p8_round2.json5](codebook/shared_base_p8_round2.json5)离线生成；已有完整词表则跳过。更换词表使用骨干初始化，恢复同一轮才使用`resume`。

以下为保留的历史实验。

2026-11-14新增[`supervised/shift_consistent.json5`](supervised/shift_consistent.json5)：从67.57%候选进行单次24,000步长训练，使用共享Patch偏移适配器和原始/增强成对监督。此配置属于专用`train_shift_consistent.py`，不交给旧`train.py`解析。后台启动：`bash code/training/run_shift_consistent.sh`。[详细说明](../../docs/shift_consistent_training.md)。

2026-11-13更新：当前监督候选为[`supervised/pyramid_phase.json5`](supervised/pyramid_phase.json5)，在原金字塔内共享提取多间隔相位谐波与曲率，按P16/P8/P4窗口对齐后以零初始化有界残差参与编码。21.074M参数、共享91.40%，尚无正式精度收益结论。原版及旧候选配置保留，用各自配置续训。[结构与启动说明](../../docs/pyramid_phase.md)。

配置按用途分目录，每个实验是一份完整JSON，可以直接运行；不使用配置继承、隐式合并或软链接。同一实验的模型、数据、损失、训练预算和GPU设置都在自己的文件中。

各JSON已就地添加中文`//`注释，说明参数作用、可选值、0或空字符串的含义、阶段限制及配套修改关系。它们采用项目支持的带注释JSON语法（JSONC），应通过`load_config`或`read_config_json`读取；Python原生`json.load`不能直接读取注释。训练实际保存的resolved_config.json、工具生成的词表元数据及推理结果仍为标准JSON。

## 目录和实验选择

```text
configs/
  data/prepare.json                 # 数据缓存与划分
  codebook/
    round1.json                     # 原始I/Q特征词表
    round2.json                     # 第一轮教师特征词表
  pretrain/
    round1.json                     # 第一轮，从头训练
    round2_continue.json            # 第二轮，继承第一轮基座
    round2_scratch.json             # 第二轮，随机初始化基座
    reconstruction.json             # 波形重建对照
  finetune/
    round1.json                     # 适配第一轮基座
    round2_continue.json            # 适配第二轮继承方案
    round2_scratch.json             # 适配第二轮随机初始化方案
  supervised/
    pyramid_phase.json5             # 当前候选：共享局部相位关系适配，原金字塔与任务头
    pyramid_midfusion.json5         # 先前中期交互候选，保留供历史运行续训
    pyramid_temporal.json5          # 先前分段调制头候选，保留用于已有运行续训
    pyramid_v1_171823.json5          # 原版基线与17:18断点续训
    small.json                      # 小模型连续回归对照
    comparisons/
      small_reference.json          # 384×8参考配置
      large_30e.json                # 768×12、30轮对照
      large_stable.json             # 稳定性候选
      large_repaired.json           # 注意力与任务头修复候选
  deployment/
    evaluation.json                 # 独立全量验证
    export.json                     # 单模型导出
    inference.json                  # Parquet到JSONL推理
  smoke/
    supervised.json                 # 微型监督流程检查
    pretrain.json                   # 微型混合预训练流程检查
```

预训练/微调配置保持原架构；当前候选为原金字塔＋共享相位适配，从零监督，尚未接入结构预训练。严格resume必须使用对应架构的配置；相位版支持从原21M金字塔完整热启动并新增零残差，但不复用旧优化器／调度。旧运行仍用各自配置，已撤下串行方案不兼容。

## 短命令启动

在项目根目录、已激活py313环境中执行。第一次做结构预训练：

```bash
python code/training/build_codebook.py --config code/configs/codebook/round1.json
python code/training/train.py --config code/configs/pretrain/round1.json
python code/training/train.py --config code/configs/finetune/round1.json5
```

已有第一轮模型，做第二轮继承方案：

```bash
python code/training/build_codebook.py --config code/configs/codebook/round2.json
python code/training/train.py --config code/configs/pretrain/round2_continue.json
python code/training/train.py --config code/configs/finetune/round2_continue.json5
```

第二轮随机初始化对照复用同一份第二轮词表，不必再次构词：

```bash
python code/training/train.py --config code/configs/pretrain/round2_scratch.json
python code/training/train.py --config code/configs/finetune/round2_scratch.json5
```

已有完整词表可跳过构词步骤；构词入口要求输出目录为空，防止覆盖旧标签。第二轮教师和第二轮继承配置均固定到当前第一轮运行的同一份best.pt；若换第一轮运行，这两个来源应配套修改。若第一轮仍在训练，等待其结束或复制固定检查点，避免best.pt被更新。

三组实验的路径已分开：

| 实验 | 词表 | 预训练输出 / 微调输出 | 初始化 |
|---|---|---|---|
| 第一轮 | artifacts/structure_research_01/codebook_v1 | logs/pretrain_v1 / logs/finetune_v1 | 基座随机初始化 |
| 第二轮继承 | artifacts/structure_research_01/codebook_v2 | logs/pretrain_v2 / logs/finetune_v2 | 第一轮基座，结构头重置 |
| 第二轮随机初始化 | 同一份codebook_v2 | logs/pretrain_v2_scratch / logs/finetune_v2_scratch | 全模型随机初始化 |

三个预训练配置的数值配方由整理时的当前配置复制；不声称还原历史第一轮预算。第二轮两种初始化方案保持相同模型、损失、学习率计划、预算和GPU安排，便于起步比较，最终仍需考虑各自适合的学习率。微调三份配置共用full全参策略：头部3e-4、基座3e-5、3%预热后cosine至5%、20轮。微调入口统一单卡、累积1次，有效batch256。best.pt按三任务本地综合分选取（体制30／调制30／参数40，非官方分）。

日常监督训练：

```bash
python code/training/train.py --config code/configs/supervised/pyramid_phase.json5
```

评估、导出、推理：

```bash
python code/training/evaluate.py --config code/configs/deployment/evaluation.json
python code/training/export.py --config code/configs/deployment/export.json
python code/infer.py --config code/configs/deployment/inference.json
```

deployment是共用工具配置，保留原来的第一轮微调来源。评估其他实验时，将evaluation/export的run.checkpoint指向对应微调目录；导出到新的run.output，再把inference的run.weights指向该导出目录。评估输出、导出目录、推理输出也应按实验分开。工具不会自动根据最近运行的预训练切换来源。

数据、输出、词表、教师和初始化检查点均在run分组。所有相对路径都以启动工作目录为基准，移动配置文件不改变这个规则。检查点可填精确.pt路径，或通过latest_run.json解析的实验父目录；并行实验建议使用精确路径。已保存的历史记录和已启动进程不受配置移动影响。

## 单卡与双卡

训练和推理的runtime.gpus与runtime.processes配套修改：单卡为"2"和1，双卡为"2,3"和2；双卡还支持0/1、4/5。入口自动启动torchrun。构词表与独立评估保持单进程，在runtime.gpus选择一张卡，导出在CPU执行。整理没有更改现有GPU选择。

训练有效batch为data.batch_size × processes × gradient_accumulation_steps；推理batching.batch_size为读取后再分片的记录数。更多说明见[单双卡检查说明](../../docs/device_compatibility.md)。

手动停止按一次Ctrl+C后等待整组退出。中断不会临时保存额外检查点，应从最近完整的last.pt恢复。训练/推理入口已增加异常清理、启动器死亡保护及退出超时兜底，详见[停止说明](../../docs/device_compatibility.md#手动停止与异常退出)。

## TensorBoard：logging.tensorboard

所有训练配置默认启用，当前phase配置显式列出enabled、log_every_steps、flush_secs、max_queue和classwise_metrics。0记录间隔跟随runtime.log_every，事件放在本次运行的tensorboard子目录，仅rank0写入。切换可视化开关不影响严格resume匹配。[完整使用说明](../../docs/tensorboard.md)。

## 断点续训：checkpoint

```json
"checkpoint": {
  "resume_from": "logs/某实验/时间戳/last.pt",
  "save_last": true,
  "save_best": true,
  "save_every_steps": 500,
  "save_every_epochs": 0
}
```

- resume_from为空表示新运行；填确切last.pt恢复模型、AdamW、cosine、混合精度、epoch/step及最佳指标，并继续写原时间戳目录。
- save_every_steps按完整优化器更新定期保存，0关闭轮内保存；save_every_epochs控制额外的完整epoch快照。定期保存需要save_last=true。
- 严格resume要求词表、架构、训练配方、batch和进程数一致。换词表或下游适配使用run.init_checkpoint，仅继承基座；同阶段全模型热启动使用run.warmstart_checkpoint并重建优化器。
- run.init_checkpoint与warmstart_checkpoint互斥；开始新实验时清空resume_from。配置中的resume_from优先于初始化设置；显式CLI初始化参数可以覆盖。
- training.max_steps仅为停止上限，不重算cosine周期；完整epochs和steps_per_epoch决定学习率计划。

每份训练配置独立保存全部参数，读取支持//与/* */注释；未知字段和明显不一致的设置会报错。实际运行会在时间戳目录保存resolved_config.json。若复制配置开展新实验，应同步修改experiment.name和run.output。

以下保留参数详解；当前候选使用encoder控制原Transformer，pyramid控制原中／细尺度分支，heads控制原读出。新增参数集中在model.phase_relations，包含lag、谐波／曲率阶数、编码宽度、数值保护、局部核、dropout和有界增益，配置内逐项注释。mid_fusion、modulation_temporal、hierarchy、hybrid均关闭。

## 模型：`model`

下表保留历史研究接口，不代表当前phase配置取值；当前配置中的physical_statistics=false、classification_pooling=multiscale_attention_stats：

| 参数 | 当前值 | 含义与使用条件 |
|---|---|---|
| `signal.local_refinement` | `false` | Patch 前的局部残差卷积；匹配预算对照未提升，保留研究接口，不默认启用 |
| `heads.classification_pooling` | `"mean_std"` | `mean` 只用 Token 均值；`mean_std` 拼接有效 Token 标准差，保留波动信息；改变分类头输入维度 |
| `heads.physical_statistics` | `true` | 分类头额外接入 32 项 I/Q 幅度、相位、相关及高阶统计量；无采样率/SNR 等额外元数据 |
| `heads.classification_fp32` | `true` | 分类统计及分类头使用 FP32，编码器仍遵从运行精度；适用于标准化折入权重后的精度保护 |
| `heads.contrastive_projection_dim` | `128` | SupCon 训练辅助投影维度；只有对比权重大于 0 才建立，导出时剔除 |
| `loss.supervised.contrastive.weight` | `0.0` | 额外调制监督对比损失权重；本轮 0.1 热启动与 0.3 从头训练均未超过 CE，保持关闭 |
| `loss.supervised.contrastive.temperature` | `0.1` | 对比 softmax 温度，必须大于 0；减小会强化困难负样本 |
| `loss.supervised.contrastive.gather_distributed` | `true` | 使用可反传的跨卡正/负样本；排除重复采到同一记录，按全局有效锚点数归一化 |

缺省设置保持旧模型语义；当前`supervised/pyramid_phase.json5`显式保留多分辨率FP32读出，并新增共享相位适配器。只有原金字塔到新增零残差适配的完整热启动受到支持，不泛化为任意多分辨率结构之间的迁移，详见[相位版说明](../../docs/pyramid_phase.md)。

三项假设诊断新增了同属 `model.signal` 的研究开关，当前主配置无需改动：`scale_embedding=true` 保留原始 RMS 尺度回注；`global_spectrum_input=true` 保留全局频谱回注；`multiscale_frontend=false` 关闭 Patch 前 dilation=1/4/16 并行残差。前两个开关置 false 时保留模块及共同初始化，只将该路径贡献置零；`global_spectrum=false` 则直接不建立全局模块，两者含义不同。本轮匹配预算下，去尺度、去全局回注和多尺度前端均未提高调制识别，不推荐直接启用这些改动。详细指标见 [假设验证报告](../../artifacts/hypothesis_validation/20261111_155810_763553/report.md)。

| 分组 | 参数 | 含义与影响 |
|---|---|---|
| `encoder` | `width / depth / heads` | 隐藏宽度、层数、注意力头数；增加通常提高容量和开销，不保证提升精度；每头维度必须为偶数 |
| `encoder` | `mlp_ratio` | Transformer 前馈隐藏维度相对 width 的倍率；降低可节省计算 |
| `encoder` | `dropout / attention_dropout` | 残差及 MLP Dropout、注意力 Dropout；增大正则也可能欠拟合 |
| `encoder` | `rope_base` | RoPE 频率基数；不是采样率，不赋予未知物理单位 |
| `encoder` | `qk_normalization` | 对每个注意力头的 Q/K 做 FP32 单位 RMS 归一化，抑制投影幅度增长造成的注意力饱和；不增加训练参数 |
| `encoder` | `qk_scale` | QK 归一化后的固定 logits 倍率，仅归一化开启时生效；本次验证 0.75–2.0，保留 1.0 |
| `signal` | `patch_size` | 每个 token 对应采样点数；小 patch 提高时间分辨率并增加 token；改变后须重建数据缓存和词表 |
| `signal` | `stft_window` | 局部频谱窗口；与 patch 必须中心对齐；增大改变时频分辨率折中 |
| `signal` | `phase_features` | 是否加入可靠性加权差分相位特征；当前主模型保留原设置 true，应与 false 做独立对照 |
| `signal` | `global_spectrum` | 是否加入整窗频谱旁路 |
| `signal` | `spectral_backend` | `fft` 或可移植的实数 `matmul`；后者通常计算更多 |
| `signal` | `position` | `samples`保持旧样点模式；`window`归一化窗口进度；`dual`按head同时使用两种尺度；`physical`必须提供真实采样率 |
| `signal` | `position_options` | 参考长度、窗口头比例、绝对位置开关和真实采样率设置；详见[位置编码说明](../../docs/position_encoding.md)；无fs的窗口/双尺度模式不是RoPE-fs |
| `signal` | `rms_epsilon` | I/Q 共同 RMS 的数值下限，不对 I、Q 独立归一化 |
| `signal` | `normalize_fusion` | 可选的时/频分支与融合输入归一化；会改变已有基座的特征分布，不能直接视作可无损开启的修复 |
| `heads` | `parameter_mode` | `regression` 连续回归；`grid_residual` 仅对训练侧离散字段使用概率期望加残差，时间参数始终连续 |
| `heads` | `classification_hidden_ratio` | 分类头隐藏维度相对 width 的倍率 |
| `heads` | `grid_residual_scale` | 网格残差的最大幅度，以每字段训练标准差为单位；太小可能限制偏离网格的预测 |
| `heads` | `parameter_query_temperature` | 参数查询注意力温度；小于 1 更集中，大于 1 更平滑 |

改变 width、depth、MLP、输入通道等结构设置后，不能直接载入形状不同的旧基座。预训练与微调应保持相同的基座设置。模型启动时仍检查最终参数量不超过 1B、共享比例超过 80%。

## 数据与更新预算：`data`、`training`

- `data.batch_size`：每卡、每个微批次样本数。
- `training.gradient_accumulation_steps`：累积多少微批次才更新一次参数；有效全局 batch = 每卡 batch × 卡数 × 累积次数。DDP 在非更新边界使用 `no_sync()`。
- `data.steps_per_epoch`：每个 epoch 的**优化器更新次数**；0 表示按训练记录数和有效全局 batch 自动计算。
- `training.epochs`：训练轮数。均衡采样是有放回的更新预算，不保证每轮覆盖全部记录。
- `training.max_steps`：总更新次数上限；0 不限制。短跑不会压缩完整 cosine 曲线，适合续训和调试。
- `training.stage`：`supervised` 或 `pretrain`；命令行 `--stage` 可覆盖。
- `training.gradient_clip`：全局梯度范数裁剪阈值；0 关闭裁剪，非有限梯度仍报错。
- `data.merge_paired`：只在训练侧合并完全相同波形的已知跨任务标签；默认关闭，作为独立对照。
- `data.augmentation.phase_rotation_probability`：仅监督训练时，对每条 I/Q 波形以该概率乘一个随机整窗单位复数，角度在 0 到 2π 均匀分布。保持幅度、频谱功率、相对相位、时间位置和七项参数标签；不移动时间、不引入频移、不共轭。0 关闭，1 每条都增强。旧配置缺省为 0，验证、推理和结构词表预训练不增强。
- `data.augmentation.modulation_only`：`phase_rotation_probability`、`circular_shift_probability` 仅用于只有调制标签的记录；协议记录、参数记录及合并后带其他任务标签的记录均跳过。循环平移保留有效长度并在长度内操作，不改变参数样本的时间原点。本轮热启动对照未提高调制准确率，主配置两项均保持 0。
- `data.sampling.mode`：`balanced` 按三个任务轮转；`weighted` 按 `task_probabilities` 抽任务；`proportional` 按各任务可用记录数抽取。只有 weighted 使用自定义概率权重，它们不必加起来等于 1。
- `data.loader`：`num_workers`、`pin_memory`、`persistent_workers`、`prefetch_factor` 控制读取。后两项仅在 worker 数大于 0 时生效，避免 PyTorch 参数冲突。

采样概率决定任务出现次数，`loss.supervised.task_weights` 决定每次出现时的梯度权重，两者含义不同。默认均衡采样配等权损失，不把竞赛 30/30/40 机械地当作最佳训练权重。

## 优化与学习率：`optimizer`、`scheduler`

```json
"optimizer": {
  "name": "adamw",
  "learning_rate": 0.0003,
  "backbone_lr_factor": 0.1,
  "weight_decay": 0.01,
  "betas": [0.9, 0.999],
  "eps": 1e-8,
  "exclude_bias_and_norm_from_decay": false
},
"scheduler": {
  "name": "cosine",
  "warmup_steps": 500,
  "warmup_ratio": 0.0,
  "warmup_start_factor": 0.0,
  "min_lr_ratio": 0.05
}
```

`learning_rate`是任务头峰值学习率；基座峰值乘`backbone_lr_factor`。是否对偏置和所有一维参数免除权重衰减可独立设置；新监督main启用免衰减，通用默认保留原行为false以兼容历史配方。

`optimizer.skip_inactive_heads` 保护本次参数更新中没有有效训练目标的任务头，随附配置均开启，生产训练入口要求为true。为完成DDP的零梯度连接会使AdamW误认为该头仍参与训练，继续施加历史动量与衰减；保护开启后，在同步反向传播后将这些参数的grad恢复为None。梯度累积取所有微批次的活动并集，DDP再跨rank取并集，避免各卡参数分歧。共享基座继续参与优化。

cosine 按**优化器更新**调度：先线性预热，再余弦下降。在完整正常长度的计划中，最后一次更新使用组内峰值乘 `min_lr_ratio`。例如头部峰值 3e-4、比例 0.05，最低为 1.5e-5；基座倍率 0.1 时最低为 1.5e-6。各组的学习率倍率始终保持。

`warmup_steps` 与 `warmup_ratio` 二选一，另一项设为 0；都为 0 则无预热。比例按完整计划总更新数计算。预热首步因子为 `start + (1-start)/warmup_steps`，末步达到 1。预热步数达到或超过总更新数会报错，不再自动截成总步数的 10%。`name: constant` 可做预热后恒定学习率对照。

日志 `learning_rates` 记录**刚完成更新实际使用的**各组学习率。`--max-steps` 提前停止时未必降到最低点，这是按完整计划调度的预期行为。若要完整验证短周期，应同时设置合适的 epochs、steps_per_epoch 和预热量。

## 损失：`loss`

`supervised`：

- `task_weights`：分别设置 modulation、protocol、parameter 权重。
- `label_smoothing`：分类标签平滑；过大可能损害置信度与精度。
- `regression` / `huber_delta`：MSE 或 Huber，delta 在标准化目标单位下定义。
- `parameter_weights`：七个字段分别设置权重；只在当前批次有效的字段间加权归一，0 关闭该字段的回归损失。
- `grid_weight`：网格分类辅助损失权重；独立于回归损失，仅 grid_residual 有对应网格头时生效。将某字段回归权重设为 0 不会自动关闭网格辅助损失。

`pretraining`：

- `mask.ratio / span`：遮挡比例、连续遮挡跨度，span 单位为 token。
- `mask.strategy`：`random` 或 `energy_mixed`；后者根据能量抽一部分，再随机补足固定遮挡预算。
- `mask.energy_fraction / energy_floor`：能量选点份额及相对平均能量的平滑底值；仅 energy_mixed 生效。
- `weights.reconstruction / atomic / emergent / spectrum`：I/Q 重建、原子标签、组合标签、局部功率谱重建损失权重。谱重建同样属于连续重建类辅助目标。
- `equalize_emergent_spans`：按组合展开长度缩放回填损失；默认 false，需单独比较。

正式纯结构配置保持 reconstruction=0、spectrum=0。设置 emergent=0 可做 Atomic-only 对照。没有词表时不能靠 Atomic/Emergent 权重训练。

## 验证、检查点与运行

`evaluation.every_epochs` 控制验证频率；最终轮及 max_steps 提前停止时强制验证。`batch_size / precision` 与训练独立；`max_batches=0` 表示全量，预训练验证批数以各实验配置为准，0表示全量。限制验证批数可能缺少部分任务，监督正式比较应保持 0。

`selection_metric=auto` 时，监督训练按本地 proxy_score，预训练按最小预训练验证损失选 best。监督和微调统一按综合分；旧单任务选模配置读入后升级为proxy_score。**预训练损失选 best 仍不能证明下游最好，需用真实标签微调后比较。** `mask_seed` 固定预训练验证遮挡，`proxy_task_weights` 只是本地代理指标权重，不补足官方尚未披露的参考尺度。

`checkpoint.save_last / save_best / save_every_epochs` 分别控制最新、最佳和每 N 轮快照；`save_every_steps` 控制轮内覆盖最新检查点，间隔 0 关闭。关闭 save_last 也会关闭轮内定期保存。`resume_from` 控制恢复来源；保存设置可在严格续训时调整。改变训练配方应开始新实验；schema_version=2 的历史检查点可使用新增字段的缺省值恢复，更早的旧调度格式请用 init-checkpoint 或 warmstart-checkpoint 开始新实验。

训练 `--output` 是实验父目录，每次新训练自动创建 `YYYYMMDD_HHMMSS_微秒` 子目录，时间固定为 Asia/Shanghai。rank 0 创建后广播给其他 rank，避免双卡各写各的目录。运行目录保存 run_info.json，父目录保存原子更新的 latest_run.json。用 `python code/utils/run_directory.py --latest 实验父目录` 获取最近创建的运行目录，再读取其中的 best.pt / last.pt。该目录不一定包含当前最优模型。

使用 `--resume /完整运行目录/last.pt` 时复用原目录，不添加新时间戳，此时可以省略 --output；若显式传入，只允许原目录或其父目录。`--init-checkpoint` 则属于新实验，仍创建新时间戳。数据缓存、词表和权重导出的 --output 用途保持各自的明确路径，不额外增加时间戳。

`runtime` 包含 device、precision、seed、threads、log_every、deterministic、allow_tf32。deterministic 开启后遇到无确定性实现的算子会报错，CUDA 还可能需要 CUBLAS_WORKSPACE_CONFIG。TF32 是 FP32 矩阵计算精度/吞吐选项，不替代混合精度。双卡通信组仍由 CUDA_VISIBLE_DEVICES 选择，只支持 0,1 / 2,3 / 4,5。

训练终端先显示实验、设备、数据规模、模型参数、有效 batch、训练预算、学习率和目标权重总览，然后每个 epoch 显示一条按优化器 step 递进的进度条。右侧 `loss` 是本 epoch 已处理微批次的平均总损失；`mod/proto/param/grid` 或 `atom/emerg/recon/spec` 是有对应有效目标且权重非零时的未加权损失均值，不把其他任务的占位零纳入分项平均。因此分项均值不能直接相加得到总损失。`lr_b/lr_h` 是最近一次更新实际使用的基座/任务头学习率。

- `runtime.progress_bar=true`：显示进度条；false 仍显示总览、epoch 验证摘要和结束信息，适合重定向终端输出。
- `runtime.progress_min_interval=0.5`：进度条最小刷新秒数；每个更新 step 都统计，终端按此间隔刷新，减少输出开销；0 表示每个更新 step 刷新。
- `runtime.log_every`：控制详细 step JSON 写入 `events.jsonl`、以及可读进度快照写入 `train.log` 的步数间隔，与终端刷新独立。

双卡只由 rank 0 显示，进度损失是其本卡统计，不额外加入跨卡同步。续训时条形从已完成的 epoch step 继续，均值只统计本次恢复后处理的部分；max_steps 中途停止保留实际百分比，不伪装成整轮完成。显示设置可以在新版检查点续训时调整。训练结束前的验证状态、分类准确率、七项原单位 RMSE 和检查点保存信息单独显示。进度条依赖 tqdm，已加入环境清单。

验证同时列出调制与协议的 Accuracy、Macro-F1、Weighted-F1，并列出每类 Precision、Recall（该类识别准确率）、F1、Support。完整混淆矩阵及上述字段保存到 `metrics.json`。单标签多分类的 Micro-F1 等于整体 Accuracy，JSON 中保留该字段，终端不重复一列。没有真实 SNR 标签时不显示虚构的“按 SNR 准确率”。

`train.log` 是终端总览、定期进度、逐类验证表和收尾信息的可读文本，去掉进度条的回车重绘；事件保存在 `events.jsonl`。恢复历史运行时，如果原 `train.log` 是 JSON 行，将原样保留到 `train_legacy.jsonl`，重名时递增后缀，再建立新的可读日志。历史指标和记录不会删除。

`evaluation.early_stopping`按selection_metric判断，patience=0关闭；开启时要求全量验证。min_delta使用指标的绝对单位，综合分下0.001表示0.001分。早停计数随checkpoint恢复，不影响任何微小新高的best保存；当前金字塔入口patience=8。

## 数据准备、词表与推理参数

```bash
python code/training/prepare_data.py --config code/configs/data/prepare.json
python code/training/build_codebook.py --config code/configs/codebook/round1.json
python code/infer.py --config code/configs/deployment/inference.json
```

数据配置 `split` 控制随机种子、验证比例与接口样例隔离，`cache` 控制 patch 和测试限额。`limit_per_task=0` 保留全量；非零时对每任务、每个 split 分别限额且保持重复组完整。

词表配置 `atomic` 控制聚类容量、采样上限、迭代及计算分块；`emergent` 控制合并次数、最长跨度、拟合序列上限；`runtime` 控制批量、设备、种子和线程。参数增大通常增加离线开销，不保证下游收益。

推理配置runtime控制设备、精度、线程及processes。单卡配置gpus="2"、processes=1，双卡配置gpus="2,3"、processes=2；也支持0/1、4/5。推理自动分片并保持原始输出顺序。batching.batch_size为读入后再分片的记录数；训练data.batch_size则是每卡微批次大小。spectrum.backend=null表示沿用导出模型设置。训练缓存、manifest和结构词表都不参与正式推理。检查范围见[单双卡兼容性说明](../../docs/device_compatibility.md)。

命令行显式参数优先于文件；训练命令原有 `--batch-size / --epochs / --stage / --merge-paired` 仍可用，新增 `--no-merge-paired` 可关闭配置中的合并。原扁平配置可读取并补全为新结构；新实验建议复制对应的完整文件修改。

`--warmstart-checkpoint` 用于以完整模型权重开始新实验，保留已有任务头，重置优化器和调度器。它与只加载共享基座的 `--init-checkpoint`、完整恢复原配方的 `--resume` 不同，三者互斥。完整热启动要求相同数据划分、标签定义、目标尺度和结构词表。新归一化开关不改变权重形状，但改变前向行为，需要重新适配，不能直接修改导出配置后宣称模型已修复。

## 旧配置路径迁移

旧路径已移除，请使用下表的新路径。历史报告、日志及检查点内的旧路径保持原样，可用此表定位；旧版完整配置备份在[artifacts/config_reorganization/20261112_125518_857116](../../artifacts/config_reorganization/20261112_125518_857116/migration.json)。历史实验数值应以原运行的resolved_config.json为准。

| 原configs下路径 | 新configs下路径 |
|---|---|
| data_prepare.json | [data/prepare.json](data/prepare.json) |
| codebook.json | [codebook/round2.json](codebook/round2.json) |
| pretrain_structure.json | [pretrain/round2_continue.json](pretrain/round2_continue.json) |
| pretrain_reconstruction.json | [pretrain/reconstruction.json](pretrain/reconstruction.json) |
| finetune.json | [finetune/round1.json5](finetune/round1.json5) |
| main.json / main.json5（已撤下） | [supervised/pyramid_v1_171823.json5](supervised/pyramid_v1_171823.json5)；架构不同，旧串行pt不能直接续训 |
| small.json | [supervised/small.json](supervised/small.json) |
| main_small_reference.json | [supervised/comparisons/small_reference.json](supervised/comparisons/small_reference.json) |
| main_large_30e.json | [supervised/comparisons/large_30e.json](supervised/comparisons/large_30e.json) |
| main_large_repaired.json | [supervised/comparisons/large_repaired.json](supervised/comparisons/large_repaired.json) |
| main_large_stable.json | [supervised/comparisons/large_stable.json](supervised/comparisons/large_stable.json) |
| smoke.json | [smoke/supervised.json](smoke/supervised.json) |
| pretrain_smoke.json | [smoke/pretrain.json](smoke/pretrain.json) |
| evaluation.json | [deployment/evaluation.json](deployment/evaluation.json) |
| export.json | [deployment/export.json](deployment/export.json) |
| inference.json | [deployment/inference.json](deployment/inference.json) |

## 真实C01调制优化实验

[supervised/density_optimization.json5](supervised/density_optimization.json5)是配对实验配置，由`python code/training/optimize_density.py --config <该文件>`读取，不直接交给常规`train.py`。`--arms`选择普通续训、分布编码或定向增强等方法；每种方法独立训练和导出，不进行预测集成。完整说明和实际结果见[调制优化入口](../../docs/modulation_optimization.md)与[真实实验报告](../../docs/research/C01_共享基座调制优化实验_20261114.md)。

常规训练配置也新增`data.augmentation.modulation_only.class_indices`：空列表表示全部调制类别，`[0,1,2,3,4,5,6,7,8,9,10,11,12,13]`限定前14类。它与同级相位/时移概率配合，只影响监督训练中的纯调制记录。原配置默认空列表，行为保持一致；推理不读取类别标签。

同级`apply_probability`控制整体增强资格，默认1保持原行为。例如0.5表示约一半符合条件的记录保留原始视图，其余再按相位/时移概率变换；它不等价于分别把两个概率设为0.5。本轮50%原始＋50%相位增强未超过全量相位方案，保留为可复现实验开关。结果见[相位/时移拆分与种子复核](../../docs/research/C01_相位时移拆分与原始视图实验_20261114.md)。
