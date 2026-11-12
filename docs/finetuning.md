# 预训练后的分阶段微调

启动命令保持简短：

```bash
python code/training/train.py --config code/configs/finetune/round1.json
```

本次没有启动正式全量微调，也没有修改已有默认模型权重。用户设置的模型、采样卡组、输入/输出与预训练来源保持不变；默认微调配方改为以下候选，质量收益尚未验证。

## 当前默认策略

使用同一个共享基座、相同三任务监督损失，按优化器更新次数推进：

| 更新编号（从0开始） | 阶段 | 可训练模块 | 冻结模块模式 |
|---|---|---|---|
| 0–499 | heads | 分类/参数等下游任务头 | 整个backbone eval，关闭其Dropout |
| 500–1999 | upper | 最后4个Transformer Block、backbone最终LayerNorm、任务头 | 前端/融合/较低Block保持eval |
| 2000–9999 | full | 全部基座及任务头 | 正常训练模式 |

每显示epoch为1000次更新，总10个显示epoch，即10000次更新。它是任务均衡、有放回采样预算，不代表每轮遍历所有训练记录。全局batch为每卡batch×进程数×梯度累积数，当前每卡256、双卡、累积1即512。

AdamW：任务头峰值LR=1e-4、基座峰值1e-5、weight_decay=0.01；全程使用一次cosine计划，预热300更新、最低比例0.05。切阶段不重置优化器或cosine，也不对每个新解冻层另行预热。预热期间冻结参数grad为None，既不更新权重也不建立AdamW动量。

优化器在冻结前包含全部参数，分组和参数顺序固定；解冻后保留已有头部动量。DDP仅注册当前可训练参数，切阶段时清理旧reducer钩子并同步重建包装，避免只改requires_grad导致梯度同步失效。此实现已在本机PyTorch2.13执行验证，其他平台/版本仍须做同样验收。

## 配置字段

```json
"finetuning": {
  "strategy": "gradual",
  "head_warmup_steps": 500,
  "upper_layers_steps": 1500,
  "unfreeze_last_n_blocks": 4,
  "layer_lr_decay": 1.0,
  "frontend_lr_factor": 1.0,
  "anchor_weight": 0.0,
  "validation_every_steps": 500,
  "validate_stage_boundaries": true
}
```

- strategy=full：直接全参微调；两个冻结阶段预算忽略。默认库配置仍为full，因此旧监督/预训练配置保持原行为；finetune目录中的三份任务适配配置显式启用gradual。
- strategy=gradual：使用上表阶段。upper_layers_steps是上层适配持续步数，结束位置是head_warmup_steps+upper_layers_steps，不是独立绝对截止值。计划总更新数必须为全参阶段留出预算。
- unfreeze_last_n_blocks：不能超过模型深度。upper阶段只更新所选Block和最终归一化，不更新时频前端与位置投影。
- layer_lr_decay：1表示关闭分层学习率。小于1时，第i层（0起）峰值LR=任务头LR×backbone_lr_factor×decay^(depth-1-i)。最终LayerNorm使用基座峰值LR。
- frontend_lr_factor：时频前端/融合/位置等其他基座参数的额外倍率；启用分层时其LR=基座峰值×decay^depth×该倍率。当前两项均为1，便于单独比较阶段策略。
- anchor_weight：0关闭L2-SP；正值时约束基座靠近开始适配时加载的权重，只约束基座、不约束随机任务头。实现使用所有基座参数元素数归一化平方差；冻结参数不产生梯度。系数需要按实际梯度量级调节，没有已经验证的最优值。
- validation_every_steps：0关闭轮内验证；默认500。阶段边界也验证，同步发生时只验证一次。epoch末已有同一步验证则复用，避免重复。

anchor参考权重单独保存在训练检查点，恢复时继续使用原预训练参考，不能重新以当前微调权重为参考。开启会增加训练显存和检查点大小；导出不携带anchor或其他辅助模型。该功能与普通weight decay不同，且当前默认关闭。

阶段策略和anchor要求提供初始化/恢复检查点。共享基座的架构、position及position_options必须与预训练匹配。适配不继续计算Atomic/Emergent目标、不传结构词表，不自动启用增强或标签平滑。

## 可进一步研究的微调策略对照

finetune/round1.json、round2_continue.json和round2_scratch.json区分预训练来源，当前使用相同微调配方。若研究下表的微调策略，应复制对应JSON为单独实验，并设置独立experiment.name和run.output；固定同一精确预训练检查点、seed、模型和其他设置。

| 候选 | strategy | 头部LR / 基座LR | 分层 / anchor |
|---|---|---|---|
| A 原直接微调 | full | 3e-4 / 3e-5 | 关闭 |
| B 低LR直接微调 | full | 1e-4 / 1e-5 | 关闭 |
| C 低LR分阶段 | gradual | 1e-4 / 1e-5 | 关闭 |

先固定相同10000总更新、任务流、验证节奏与两个种子。C少做的基座更新属于策略本身，不把它描述成同基座更新次数比较。后续分层LR、anchor独立加入，避免将多个候选混为单因素收益。阶段策略不保证优于全参微调。

## 验证、存盘与恢复

- best.pt：按evaluation.selection_metric，当前auto即本地综合代理分选取。
- best_modulation.pt：按调制Accuracy单独选取。
- best_proxy.pt：按本地综合代理分单独选取。
- last.pt：当前更新进度，按保存间隔与验证写入。

checkpoint.save_best_modulation / save_best_proxy控制两个额外最佳文件，微调配置默认开启。评估仍输出两分类Accuracy、Macro/Weighted-F1及逐类指标、七参数原尺度误差。额外最佳文件是本地比较候选，最终仍导出一个模型，不集成。

检查点保存global_step、原始anchor、最佳指标、阶段状态、优化器和调度器。恢复以global_step确定下一次更新阶段，即使恰好在解冻边界中断也会正确推进；必须保持同一阶段预算、分层LR、anchor配置和实际卡数。max_steps只控制停止上限，可更改，不会改变总cosine计划。旧检查点若没有额外最佳指标记录，无法凭空还原此前所有候选权重，额外指标历史从恢复后的记录开始积累。

本次没有加入自动early stopping；可用预算限制、较密验证及最佳检查点做初筛。已有进程不会自动加载文件变更。旧微调检查点继续原配方应读取原运行的resolved_config.json并resume；新策略应开启新实验，不能直接续原全参优化器而称为同配方恢复。

## 本次执行验证

新增检查包括冻结参数逐项不变与Dropout模式、各层LR、预训练anchor恢复、full策略与原更新一致、显式NCCL设备绑定。实际小模型检查包含CPU与GPU2/3（NCCL、BF16、累积2）三个阶段训练，连续/中断恢复的模型、AdamW、调度器、anchor及最佳指标逐项一致；额外检查恰好在阶段边界恢复、原预训练路径、导出及CPU推理。

用户报告的barrier设备提示已通过在CUDA进程组初始化时指定device_id处理；实际NCCL短测未出现原提示。这不代表所有NCCL提示或其他训练问题都被解决，也不需要仅为该提示停止正在训练的进程。
