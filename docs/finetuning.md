# 预训练后的全参微调

按预训练来源选择一条命令：

```bash
python code/training/train.py --config code/configs/finetune/round1.json5
python code/training/train.py --config code/configs/finetune/round2_continue.json5
python code/training/train.py --config code/configs/finetune/round2_scratch.json5
```

run.init_checkpoint只加载对应预训练backbone，任务头随机初始化，同时新建AdamW和调度器。第二轮微调加载第二轮预训练模型。为严谨比较应固定同一份检查点快照，而非持续变化的best.pt路径。

微调默认单卡GPU2，可将runtime.gpus改为需要的单个GPU编号。

## 默认策略

从第一次更新起训练全部共享基座及任务头，预热期间也更新基座。关闭gradual、分层LR和预训练权重anchor。

| 项目 | 设置 |
|---|---|
| finetuning.strategy | full |
| 任务头峰值LR | 3e-4 |
| 基座峰值LR | 3e-5，所有基座层同倍率 |
| AdamW weight_decay / betas / eps | 0.01 / [0.9, 0.999] / 1e-8 |
| 每卡batch / 有效batch | 256 / 256 |
| 设备与累积 | 单卡、单进程、累积1次；每批更新参数 |
| 预算 | 20轮，steps_per_epoch=0按数据量计算 |
| 完整缓存对应预算 | 每轮4144更新，共82880更新 |
| 梯度裁剪 | 全局L2范数1.0，日志统计裁剪前范数 |
| 训练 / 验证精度 | BF16 / FP32 |

当前完整缓存训练记录为1,060,812条：

```text
有效batch = 每卡batch × 进程数 × 梯度累积次数
每轮更新 = ceil(训练记录数 / 有效batch)
总更新T = epochs × 每轮更新
```

保留三任务均衡、有放回采样；显示epoch定义更新预算，不保证逐条遍历。当前统一单卡、累积1次。严格比较模型时应固定batch、seed、采样及预算。

## 线性预热与cosine

```json
"scheduler": {
  "name": "cosine",
  "warmup_steps": 0,
  "warmup_ratio": 0.03,
  "warmup_start_factor": 0.0,
  "min_lr_ratio": 0.05
}
```

以82880次更新为例，前2486次线性升至峰值，第一步实际LR非零。之后连续余弦下降，最后一次更新达到峰值5%：任务头1.5e-5、基座1.5e-6。两组使用相同倍率曲线，始终保持10:1，不在epoch或任务切换时重启。

warmup是学习率预热，不是冻结参数。峰值学习率采用此前有效的全参微调量级；3%预热让随机任务头与基座共同平稳起步。这是新的待验证配方，不承诺精度一定提高。

改变数据量、有效batch或epochs会改变总周期和预热长度。max_steps只截断运行，不压缩原cosine周期。不能通过缩短max_steps把几步检查当作完整学习率计划。

## 损失、验证与选模

保持原模型、输入、三任务监督损失和采样。此次不额外加入相位增强、标签平滑、监督对比或结构辅助损失。微调无需提供词表。

每轮全量验证，关闭额外轮内验证。日志输出调制/协议Accuracy、Macro-F1、Weighted-F1、逐类指标及七参数原单位误差。

- best.pt：调制Accuracy最佳。
- best_modulation.pt：保留现有独立保存开关，对应同一调制最优候选。
- best_proxy.pt：本地综合代理分最佳；代理分不是比赛正式成绩。
- last.pt：每500次更新及验证时保存，用于同配方恢复。

最终选择一个共享模型导出，不集成。默认优先调制识别；竞赛综合取舍可另行评估best_proxy候选。

## 新训练与恢复

三份入口的checkpoint.resume_from均为空，从相应预训练基座新建时间戳目录。已有运行不会自动读取配置变更。

旧gradual运行应使用它自己的resolved_config.json与last.pt恢复；使用当前新配方需要新建运行，不继承旧优化器和cosine状态。当前新配方中断后，才填写它自己的last.pt，并保持模型、卡数、累积、预算和调度一致。

## 保留的研究选项

代码仍支持gradual、layer_lr_decay、frontend_lr_factor、anchor_weight，当前入口全部关闭。head_warmup_steps=0、upper_layers_steps=0表示没有冻结阶段；unfreeze_last_n_blocks只对gradual生效。

未来验证gradual时，应固定精确预训练检查点、LR、预算和采样，只改策略；不要再次把冻结、降LR、缩短预算混成一个因素。

## 检查

配置回归覆盖.json与.json5训练入口。机制测试检查cosine端点、参数组比例、恢复一致性、全参更新、冻结模式和DDP设备绑定；短程检查不替代正式质量实验。
