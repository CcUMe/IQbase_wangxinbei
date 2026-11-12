# 结构化配置与 cosine 更新验证

本文记录配置更新时的历史初测。后续注意力/优化器修复及当前默认权重见 [修复报告](attention_repair_report.md)。

日期：2026-11-10。环境沿用已有 Python 3.13 / PyTorch 2.13+cu130，不修改原始数据和全局环境。

## 改动

- 七份实验配置统一为 schema_version=2：experiment、model、data、training、optimizer、scheduler、loss、evaluation、checkpoint、runtime。模型内分 encoder、signal、heads。
- 新增独立的数据准备、结构词表、推理运行配置。命令行显式参数覆盖文件，未知字段报错。
- 显式配置 AdamW、基座学习率倍率、权重衰减排除、梯度累积、任务采样、字段损失权重、验证与存盘频率。
- cosine 改为明确的预热和最低比例，按优化器更新调度。移除原有“预热自动截成总步数 10%”和固定 5% 下限。
- 模型中的 MLP 倍率、注意力 Dropout、RoPE 基数、RMS 数值下限、分类头容量、网格残差幅度及参数查询温度接入配置。
- 原主模型的 phase_features=true 保留；正式结构配置重建与谱损失仍为 0。没有在本次配置整理中宣称这些设置最优。

## 验证结果

1. 原有 9 项核心检查与 7 项配置/调度检查通过，包括遮挡信息泄漏、无效标签、BPE、导出、配置合法性、损失权重和 cosine 端点。
2. [全流程状态](../artifacts/config_v2_pipeline/smoke_status.json)：复用已核验的 592 条缓存，共 17 个阶段通过，包含监督训练、续训、重建、两轮词表、纯结构与混合结构短测、微调、评估、导出、GPU/CPU 推理、输出校验及 0,1 双卡。
3. [配置集成检查](../artifacts/config_v2_checks/result.json)：每次更新累积 3 个微批次，完整 6 次更新与“更新 2 次中断后续训”的最终模型权重逐项一致，调度状态相同。
4. 上述累积实验头部峰值 3e-4、基座倍率 0.1、最低比例 0.1，最后一次实际更新学习率分别为 3e-5、3e-6；验证间隔和每轮快照生效。
5. 0,1 双卡累积短测通过，同时启用 weighted 任务采样、worker=1、persistent_workers 和自定义预取。
6. 数据准备、词表、推理均通过 --config 启动并验证命令行覆盖，最终六条样例输出格式通过校验。

新增配置测试入口：

```bash
/home/yangqilin/anaconda3/envs/py313/bin/python code/training/test_config.py
```

## 边界

全部为真实小样本的执行与数值语义检查，不代表正式比赛精度。未启动全量训练，未更新根目录初测权重，未完成 Docker/CUDA12/国产平台验收。旧格式检查点仍能推理，也能用于 init-checkpoint；由于调度语义变化，不直接用新版 resume 恢复旧版训练。

## 训练终端显示更新

新增启动总览、每 epoch 优化器 step 进度条、有效分项损失均值、基座/头部实际学习率，以及 epoch 验证和保存摘要。双卡仅 rank 0 输出；JSON 日志继续写文件。增加 tqdm 依赖和 runtime.progress_bar / progress_min_interval 设置。

[显示集成结果](../artifacts/training_display_smoke/result.json)验证：监督训练、累积微批次、epoch 中途停止与续训、关闭进度条续训、纯结构损失显示和 0,1 双卡均正常；恢复后最终权重与连续训练逐项一致；终端不再输出 JSON 事件，train.log 每行仍为有效 JSON。原有 9 项核心检查和 7 项配置检查通过。

## 自动时间戳保存

训练 --output 现在指定实验父目录，新训练自动创建上海时区的 YYYYMMDD_HHMMSS_微秒 子目录。rank 0 创建并广播同一路径；续训写入检查点原目录，可省略 --output。run_info.json 记录创建及续训信息，latest_run.json 提供最近创建运行的路径。流程脚本与文档已同步调整。

[时间戳全流程](../artifacts/timestamp_pipeline/smoke_status.json)的 17 个阶段通过，包含续训、两轮词表与训练、导出、推理及双卡；[重复启动检查](../artifacts/timestamp_repeat_check/result.json)确认同一父目录启动两次产生两个独立目录，省略 --output 的实际续训没有新建目录，双卡只产生一个运行目录。
