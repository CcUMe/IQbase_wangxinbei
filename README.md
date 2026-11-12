# C01 电磁感知模型：训练与推理代码

本项目实现一个共享时频双流基座，完成 23 类调制/波形识别、8 类体制识别和七项参数估计。包含分组数据划分、连续重建预训练、Atomic/Emergent 结构预训练、第二轮词表更新、多任务微调、评估、导出和统一推理。当前用于初步测试，未制作 Docker 镜像。

2026-11-11 诊断后优化已完成。全部 118,182 条本地验证记录上，调制 Accuracy **64.23% → 64.50%**，Macro-F1 **64.15% → 64.45%**；协议 Accuracy 保持 **94.42%**，七项参数误差完全一致。默认 `weights/` 已换为验证后的 96.32M 单模型，共享参数占 97.43%；原默认权重已备份。有效改动是冻结原 95M 共享基座，增加均值/标准差与 I/Q 统计读出并重训调制头。监督对比损失、局部残差前端未胜过控制组，默认关闭。提升有限，深层表征和扩参泛化瓶颈尚未解决，所有代理分均不代表官方成绩。完整实测及失败对照见 [本轮优化报告](artifacts/representation_optimization/20261111_122154_915261/report.md)。

原始基座来源为 `logs/supervised_main/20261110_233950_782228/best.pt`；优化后的完整权重为 `artifacts/representation_optimization/20261111_122154_915261/selected/model.pt`。前期诊断及六组增强对照见 [泛化报告](docs/generalization_20261111.md)，注意力/优化器修复见 [历史报告](docs/attention_repair_report.md)。

对外讨论可使用 [模型方案与验证总报告](docs/C01_model_and_validation_review_20261111.md)、[可编辑 Word](docs/C01_model_and_validation_review_20261111.docx) 或 [讨论资料包](docs/review_20261111/C01_discussion_package_20261111.zip)。总报告汇总截至 2026-11-11 的方案、历次验证、负结果、证据边界和下一步问题；资料包包含图表、指标 CSV 与原始证据，不包含原始波形或模型大权重。

训练配置已整理为模型、数据、训练预算、优化器、cosine 调度、损失、验证、检查点和运行设置。参数含义、影响与完整示例见 [配置说明](code/configs/README.md)。本轮配置及调度验证见 [配置更新测试报告](docs/config_v2_test_report.md)。

两项定位实验现已完成：[同调制曝光的单/多任务对照与冻结基座读出报告](artifacts/diagnostics/20261111_110649_079619/report.md)。复现入口为 `code/training/diagnose_tasks.py`、`diagnose_readout.py`，汇总入口为 `summarize_diagnostics.py`。所有验证均提供 Accuracy、Macro/Weighted-F1 和逐类指标；日志已拆分为可读的 `train.log` 与结构化的 `events.jsonl`。

三项假设验证也已完成：[观测依赖、细粒度前端、预训练迁移报告](artifacts/hypothesis_validation/20261111_155810_763553/report.md)。17 个队列任务覆盖从头去尺度/去全局频谱、多尺度前端、同预算重建/结构预训练及冻结读出/扰动评估。所有候选的末轮调制识别均未超过匹配的 17M 控制组，默认 `weights/` 和 `main.json` 未替换。大模型的90°相位敏感性主要集中在雷达，不能直接解释前14类数字调制的低准确率。本轮结构目标为相位不变局部统计原子预测，没有 ESU 或物理时间对齐，不能代表参考论文的完整方法。

进一步的 [监督模型深入诊断](artifacts/supervised_deep_diagnosis/20261111_171538_148177/report.md) 已检查全部调制记录的观测分层、六个深度位置的冻结读出及精确近邻。前14类最低RMS组训练/验证约99.88%/14.31%，最高组约99.96%/79.33%；低RMS组的最终特征能被线性头完全拟合，验证仍约13.53%。证据将过拟合定位到特定观测条件，尚未证明真实低SNR、信息论不可识别或某个单一架构根因。该轮冻结了原95M监督模型，默认权重未修改。

## 配置驱动启动

当前预训练后微调已加入分阶段解冻、可选分层LR/预训练权重anchor，以及调制/综合最佳检查点，按第一轮、第二轮继承、第二轮随机初始化分别使用独立JSON。候选设置与恢复规则见[微调说明](docs/finetuning.md)，尚未声称带来精度提升。

已激活py313环境后，核心入口只需传配置，数据/输出/词表/检查点和GPU均在配置中：

```bash
python code/training/build_codebook.py --config code/configs/codebook/round1.json
python code/training/train.py --config code/configs/pretrain/round1.json
python code/training/train.py --config code/configs/finetune/round1.json
python code/training/evaluate.py --config code/configs/deployment/evaluation.json
python code/training/export.py --config code/configs/deployment/export.json
python code/infer.py --config code/configs/deployment/inference.json
```

训练会按runtime.gpus和processes自动启动双卡。下方保留的长命令是等价CLI用法；日常优先使用上述方式。详细配置与第二轮词表操作见[执行指南](docs/pretraining_and_adaptation.md)。

## 环境与目录

当前可直接使用的解释器：

```bash
C01_PY=/home/yangqilin/anaconda3/envs/py313/bin/python
C01_DATA_ROOT=/mnt/disk1/wangxinbei_contest_dataset/C01
```

已使用 Python 3.13、PyTorch 2.13.0+cu130、NumPy 2.5.1、PyArrow 25.0.0 测试。原样例在 PyArrow 19.0.0 中存在读取问题，使用经验证的 25.0.0 环境。当前 CUDA 运行时是 13.0；赛题基线为 CUDA 12.x，正式提交前需在目标 CUDA/国产平台重新验证。项目没有改动任何现有环境或原始数据。

```text
code/
  infer.py                      # Parquet -> JSONL 统一入口；亦提供 Predictor API
  models/rf_model.py            # 时频对齐、动态门控、共享 Transformer、多任务头
  training/prepare_data.py      # 内容哈希分组、训练侧统计、mmap 缓存
  training/build_codebook.py    # Atomic 聚类、Emergent BPE、第二轮词表
  training/train.py             # 监督/自监督、混合精度、断点续训、DDP
  training/evaluate.py          # 两项分类、七项原单位误差
  training/export.py            # 单模型 safetensors 导出
  training/smoke_test.py        # 一键初步测试
  training/test_core.py         # 信息泄漏/有效位/BPE/数值/导出检查
  training/test_stability.py    # 注意力饱和、任务头动量保护与遮挡回归
  training/diagnose_gradients.py # 模块梯度、注意力尺度与熵定位
  training/repair_experiment.py # 固定批次和验证子集的真实数据修复对照
  training/diagnose_readout.py # 冻结编码器，比较均值/标准差/IQ统计读出
  training/promote_readout.py  # 将训练侧标准化与读出层合并为一个完整模型
  training/optimize_representation.py # 同初始化/采样流的 CE/SupCon 控制实验
  training/summarize_optimization.py # 汇总本轮实际优化与逐类指标
  training/run_hypotheses.py   # GPU2/3 三项假设验证队列与状态记录
  training/hypothesis_pretrain.py # 同初始化/输入/掩码的两种预训练诊断
  training/hypothesis_probe.py # 固定冻结线性读出与成对观测扰动
  training/summarize_hypotheses.py # 全部结果、逐类F1、成对区间及对照图
  configs/                     # data/codebook/pretrain/finetune/supervised/deployment/smoke
  utils/                       # 数据、损失、结构词表、接口校验与运行环境
weights/                       # 正式训练后导出的 config.json/model.safetensors
artifacts/                     # 数据审计、缓存、实验检查点及测试产物
environment/                   # 当前环境记录与依赖声明
docs/                          # 实现说明和初测报告
logs/                          # 推理日志；各实验另有自己的 train.log
```

本机双卡只允许 `0,1`、`2,3`、`4,5`。启动脚本和训练运行时均检查这一限制。以下双卡例子使用 `0,1`，请只选择已分配给你的同组设备。

本地双卡测试发现当前环境的 NCCL 主机 cuMem 分配路径会导致崩溃；关闭该路径后最小通信和完整 DDP 训练均通过。训练进程在双卡时默认设置 `NCCL_CUMEM_HOST_ENABLE=0`，不禁用 GPU P2P，不修改系统配置；显式环境值可覆盖。该回退方式也见 [NVIDIA NCCL 运行时说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/runtime_and_mpi_issues.html)。

## 一键初步测试

在项目根目录执行：

```bash
"$C01_PY" code/training/smoke_test.py \
  --data-root "$C01_DATA_ROOT" \
  --output artifacts/my_first_smoke \
  --gpu 0
```

需要额外检查双卡时，增加 `--ddp-gpus 0,1`。每次选择新的输出目录，避免覆盖实验结果。默认构建每个任务、每个 split 约 96 条记录的测试缓存，并保持重复组完整，因此实际数量略大。底层分组依据 90/10 哈希划分；小测试集对两侧分别限额，所以选出后的样本数比例可以接近 1:1。

三项假设诊断的具体预算、比较口径和限制见其 [预定方案](artifacts/hypothesis_validation/20261111_155810_763553/plan.md)。复现整个队列可运行 `code/training/run_hypotheses.py --output <新的实验目录>`，它依赖本项目已有全量缓存、17M模板检查点及历史匹配控制组，只使用 GPU 2、3。中断后对同一目录再次运行会跳过已完成任务；下游诊断恢复优化器/调度器，未完成的短预训练/读出阶段会保留旧目录并重跑。汇总使用 `code/training/summarize_hypotheses.py --output <实验目录>`。

脚本依次测试：关键回归检查、数据缓存、监督训练与续训、波形重建、首轮词表、纯结构预训练、教师特征重建词表、第二轮混合目标预训练、联合微调、评估、导出、GPU/CPU 推理及 JSONL 校验。测试模型为 64 宽、2 层、32/64 结构词表，只用于检验代码路径，不代表比赛模型精度。

结果在输出目录的 `smoke_status.json`，每一步的完整日志在 `logs/`。通过后的权重在 `weights/`，官方样例预测在 `prediction_device.jsonl` 与 `prediction_cpu.jsonl`。GPU 与 CPU 的浮点数不要求逐位一致，分类/参数分别按同一模型计算。

中断后可对同一输出目录增加 `--resume`，保留已经通过的阶段，继续失败或未执行的阶段；关键回归检查会重新运行。

## 准备全量训练数据

已有全量审计索引时：

```bash
"$C01_PY" code/training/prepare_data.py \
  --data-root "$C01_DATA_ROOT" \
  --manifest artifacts/c01_data_audit/sample_manifest.parquet \
  --output artifacts/data_full
```

如需重新审计数据版本：

```bash
"$C01_PY" code/utils/audit_c01_dataset.py \
  --data-root "$C01_DATA_ROOT" \
  --output-dir artifacts/c01_data_audit
```

缓存输出 `iq.npy`、`metadata.npz`、`dataset.json` 和 `split_manifest.parquet`。原始 I/Q 保留 Float32，按最长 1024 点缓存；全量约需 10 GB 级别空间。读取时按长度裁切，并由训练采样器按长度组批。源码中的每条波形会再次核对与审计哈希一致。

数据处理保证：同内容哈希不跨 split、六个接口样例及其重复组隔离、目标中心/尺度和参数码表只由训练侧计算。元数据不含可用于推理的答案捷径；ID 仅负责记录与回填。

## 建立联合监督基线

小模型监督对照：

```bash
python code/training/train.py --config code/configs/supervised/small.json
```

主模型双卡训练：

```bash
python code/training/train.py --config code/configs/supervised/main.json

C01_MAIN_RUN=$("$C01_PY" code/utils/run_directory.py --latest artifacts/supervised_main)
```

主监督训练使用`supervised/main.json`；模型规模、前端、任务头及优化配方均在其中调整，运行时以保存的`resolved_config.json`为准。配置文件支持`//`、`/* */`注释。`data.batch_size`是每卡微批次；每卡256、累积1次时，双卡有效全局batch为512。cosine、预热和最低学习率比例位于`scheduler`，步数按优化器更新计算。

`--max-steps 30` 可做短跑；`--merge-paired` 只在训练侧合并同波形的已知跨任务标签，与不合并分别比较。训练按三个任务等配额轮转，任务内按真实长度比例抽样；这是有放回的均衡更新预算，不等同于每轮恰好遍历每条记录一次。

主配置已启用 `model.encoder.qk_normalization` 和 `optimizer.skip_inactive_heads`，峰值学习率为 1e-4。继续相同配方使用 `--resume`；更换优化配方并保留全部任务头时使用 `--warmstart-checkpoint`，它会重置优化器；`--init-checkpoint` 只载入基座。已启动的进程不会自动加载源文件改动。

也可以在 `code/configs/supervised/main.json` 的 `checkpoint.resume_from` 填入原时间戳目录的 `last.pt`，然后按原命令启动，无需再传 `--resume`；留空表示新训练。`checkpoint.save_every_steps=500` 会在完整优化器更新后定期原子覆盖 `last.pt`，epoch 结束也保存，减少意外中断时损失的进度。续训恢复模型、优化器、cosine、混合精度及 epoch/step，仍写原目录。命令行 `--resume` 优先，`--no-resume` 可忽略配置并新建运行；详见 [配置续训说明](code/configs/README.md#断点续训checkpoint)。

`supervised/main.json`保留用户当前设置：384宽、8层、6头，`classification_pooling="mean_std"`、`physical_statistics=true`、`classification_fp32=true`。它与既有95M基座尺寸不同，不能直接加载该大模型检查点。统计量只由有效I/Q计算，直接测试既有模型精度可使用原`weights/`。

如需热启动既有优化权重`artifacts/representation_optimization/20261111_122154_915261/selected/model.pt`，先使用与其匹配的模型结构，并在run.warmstart_checkpoint填写来源。该合并权重没有原读出训练优化器状态，不能resume；init_checkpoint只保留基座，会重置任务头。

`--output` 指定实验父目录，新训练自动创建上海时区的时间戳子目录，例如 `artifacts/supervised_main/20261110_153000_123456/`。同一命令重复启动会生成不同目录，双卡共享一个目录。终端总览会显示实际路径；父目录的 `latest_run.json` 记录最近创建的运行，上述辅助命令读取该路径。续训应选择确定的检查点，最新目录不等同于最佳模型。

每次运行目录包含 `best.pt`、`last.pt`、`metrics.json`、`parameter_report.json`、`resolved_config.json`、`train.log`、`events.jsonl` 和 `run_info.json`。`best.pt` 按 `evaluation.selection_metric` 选择，当前主配置为调制准确率，同时始终报告协议与参数误差；`auto` 在监督阶段按本地代理分选择，代理分不是官方成绩。断点续训要求保持配置、数据划分、世界大小、词表及配对合并方式一致，并继续写入检查点所在的原目录：

终端先输出训练总览，再逐 epoch 显示 step 进度、平均损失、主要分项和学习率；验证输出调制/协议的 Accuracy、Macro-F1、Weighted-F1，以及逐类 Recall（该类识别准确率）、Precision、F1、Support 和七项参数 RMSE。双卡仅主进程显示。

`train.log` 保存同样的可读总览、定期进度快照、完整验证表及收尾信息，不再混入 JSON 或终端重绘控制符。机器可读的逐 step/epoch 记录单独写入 `events.jsonl`，完整指标保存在 `metrics.json`。恢复历史 JSON 版日志的运行时，原 `train.log` 会先保留为 `train_legacy.jsonl`（重名时递增后缀），再开始可读日志；旧记录不会丢失。`runtime.progress_bar`、`progress_min_interval` 控制终端进度条，`runtime.log_every` 控制事件及文件内进度快照的步数间隔。

```bash
python code/training/train.py --config code/configs/supervised/main.json
```

## 预训练候选一：I/Q 掩码重建

完整的阶段解释、两轮结构词表与任务适配命令见[预训练与适配执行指南](docs/pretraining_and_adaptation.md)。预训练与微调的encoder/signal配置需要保持一致，实际尺寸和预算以当前配置及运行目录内resolved_config.json为准。已启用QK归一化与非活动头保护。第一轮即可适配，第二轮为可选实验。训练和推理的单/双卡设置与已执行验证见[兼容性说明](docs/device_compatibility.md)。

```bash
python code/training/train.py --config code/configs/pretrain/reconstruction.json
```

时域、局部时频谱、全窗频谱和相位特征均从同一受遮挡输入生成。RMS 只由可见点计算，完整 I/Q 仅进入目标分支。配置 `loss.pretraining.mask.ratio`、`span` 控制遮挡量与连续跨度。

可设置 `loss.pretraining.mask.strategy: "energy_mixed"`，使一部分遮挡位置依能量加权选择、其余随机选取，总遮挡数量保持相同。这是关键区域掩码的项目实现候选，不是对讲座未披露阈值算法的复现。`loss.pretraining.weights.spectrum` 可增加局部 Patch 功率谱辅助目标，先保持为 0 做独立对照。

## 预训练候选二：Atomic/Emergent 结构学习

第一轮词表与预训练：

```bash
python code/training/build_codebook.py --config code/configs/codebook/round1.json
python code/training/train.py --config code/configs/pretrain/round1.json
```

第二轮词表与继承基座方案：

```bash
python code/training/build_codebook.py --config code/configs/codebook/round2.json
python code/training/train.py --config code/configs/pretrain/round2_continue.json
```

第二轮随机初始化对照复用同一词表：

```bash
python code/training/train.py --config code/configs/pretrain/round2_scratch.json
```

三种实验有独立配置和输出目录。已有完整词表可跳过构词；构词入口拒绝非空目录。第二轮教师与继承初始化需引用同一份固定第一轮检查点。构词聚类及BPE只用训练侧，验证侧使用固定规则赋标签。词表默认512个原子单元、最多512次合并、最大展开5个原子位置。详细来源、预算、参数及比较规则见[配置索引](code/configs/README.md)与[执行指南](docs/pretraining_and_adaptation.md)。

## 共同微调、评估、导出

按预训练实验选择对应的微调配置，下面三条是不同实验，按需执行：

```bash
python code/training/train.py --config code/configs/finetune/round1.json
python code/training/train.py --config code/configs/finetune/round2_continue.json
python code/training/train.py --config code/configs/finetune/round2_scratch.json
```

deployment目录下的evaluation.json、export.json和inference.json是共用工具配置，默认来源仍为第一轮微调。评估其他实验时，修改相应检查点、输出及推理权重路径后执行：

```bash
python code/training/evaluate.py --config code/configs/deployment/evaluation.json
python code/training/export.py --config code/configs/deployment/export.json
python code/infer.py --config code/configs/deployment/inference.json
```

评估输出两个混淆矩阵、准确率和宏平均召回、七项有效样本数/原单位 RMSE/MAE/P95。训练损失使用训练侧标准化量，评估和导出预测恢复原始数值尺度。频移有效真零参与损失，无效 NaN/零不参与。

导出移除全部训练辅助头，保存 `weights/config.json` 与 `weights/model.safetensors`，逐张量校验读写一致，再检查参数量和共享比例。自带标准 safetensors 读写器，无需额外下载依赖。最终推理不加载词表或教师，也不加载多个任务专属基座。

## 统一推理与结果校验

```bash
CUDA_VISIBLE_DEVICES=0 "$C01_PY" code/infer.py \
  --input "$C01_DATA_ROOT/rf_signal_sample_io_v1.0/rf_signal_sample_io_input_v1.0.parquet" \
  --output artifacts/predictions.jsonl --weights weights \
  --log logs/inference.json

"$C01_PY" code/utils/validate_output.py \
  --input "$C01_DATA_ROOT/rf_signal_sample_io_v1.0/rf_signal_sample_io_input_v1.0.parquet" \
  --output artifacts/predictions.jsonl \
  --schema "$C01_DATA_ROOT/rf_signal_benchmark_train_v1.0/task_schema.json"
```

输入仅需 `task/sample_id/iq`，输出保持原顺序。两种分类返回 Schema 原字符串；参数返回全部七项有限数字，不输出有效位。`--device cpu` 支持 CPU；`--spectral-backend matmul` 用实数矩阵乘替代全窗 FFT，局部 STFT 本来就使用实数固定卷积，便于后续平台适配。实际国产平台尚未验证。

当前官方接口只明确文件格式，未披露最终命令签名；本项目提供上述 CLI 和 `Predictor.predict(records)` Python API，收到正式调用约定后可做轻量适配。当前没有采样率字段，使用样点坐标；模型 API 支持已知采样率时的物理 RoPE，但现有数据 CLI 不推测采样率。

新增可选 `position="dual"`，按注意力头同时使用样点间距与窗口相对进度，属于无采样率的研究候选，不称为RoPE-fs；当前配置仍保持samples。真实physical模式只接受明确的真实采样率，并同步绝对时间坐标。配置、迁移限制与验证见[位置编码说明](docs/position_encoding.md)。

模型和输入无关的关键检查可独立运行：

```bash
"$C01_PY" code/training/test_core.py
```

更多说明见 [实现说明](docs/implementation.md) 和 [初步测试报告](docs/self_test_report.md)。全量长期训练、比赛精度和目标平台迁移仍需后续实验确认。
