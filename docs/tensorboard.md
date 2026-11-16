# TensorBoard训练可视化

2026-11-13，Asia/Shanghai。监督、预训练和微调共用train.py，均已接入TensorBoard。py313环境已安装tensorboard 2.21.0，requirements与environment.yml同步更新。只在rank0创建事件写入器；未改变模型、采样、损失、优化器或学习率计划。

## 查看曲线

在项目根目录、已激活py313环境执行：

```bash
python -m tensorboard.main --logdir logs --host 127.0.0.1 --port 6006
```

浏览器打开 `http://127.0.0.1:6006`。如果训练在远程工作机，在本地建立SSH转发后访问同一地址：

```bash
ssh -L 6006:127.0.0.1:6006 yangqilin@<服务器地址>
```

在Runs中选择要比较的时间戳实验，可用tag筛选 `validation/modulation/accuracy`、`validation/modulation/macro_f1`、`validation/protocol/accuracy` 等。准确率与F1记录为0～1；参数误差为原始标签单位。所有曲线的横轴step统一为**优化器更新数**，续训不重新从0编号。

TensorBoard本身按日志目录读取事件文件；使用PyTorch的SummaryWriter写入，不需要安装TensorFlow。[TensorBoard官方使用说明](https://www.tensorflow.org/tensorboard/get_started)、[项目写入实现](../code/utils/tensorboard_log.py)。

## 新训练与续训

原训练命令不变，例如当前主干候选：

```bash
python code/training/train.py --config code/configs/supervised/pyramid_midfusion.json5
```

自动保存到：

```text
logs/supervised_pyramid_midfusion/<时间戳>/
  train.log                    # 原可读日志
  events.jsonl                 # 原机器日志；新增step活跃分项信息
  tensorboard/
    events.out.tfevents.*       # 原生TensorBoard事件
  last.pt / best.pt / ...
```

resume继续写原运行的tensorboard目录。若恢复到step T，保留≤T的记录，利用purge_step=T+1隐藏原分支中T之后的旧尾部，再写新进度。可视化配置不参与严格训练配方比较，因此开启、关闭或调整记录频率不会阻止恢复原本兼容的checkpoint。

验证及epoch结束立即flush；正常结束、Python异常和Ctrl+C均通过资源上下文关闭写入器。旧的文本日志和JSONL记录仍保留。`kill -9`等强制终止无法保证尚在队列中的事件已落盘。

## 配置

当前midfusion配置已经显式列出以下分组，其他原有训练配置自动采用相同默认值，无需修改原配置或旧checkpoint：

```json
"logging": {
  "tensorboard": {
    "enabled": true,
    "log_every_steps": 0,
    "flush_secs": 30,
    "max_queue": 10,
    "classwise_metrics": true
  }
}
```

- enabled：关闭后不导入SummaryWriter、不创建事件目录；原日志仍保存。
- log_every_steps：0跟随runtime.log_every；正整数使用独立更新间隔。全局step1始终记录。TensorBoard频率可以高于原JSONL频率。
- flush_secs：后台刷新周期，正整数秒；验证/epoch结束和退出另行刷新。
- max_queue：异步事件队列大小，正整数。
- classwise_metrics：是否记录逐类Recall、Precision、F1及Support；关闭仍有总体指标。

若在新环境中使用，先安装依赖：

```bash
python -m pip install -r environment/requirements.txt
```

## 记录内容与口径

| Tag组 | 内容 |
|---|---|
| train/step_loss | 每次记录的优化器更新总损失、实际活跃的分类/参数/网格/预训练等分项 |
| train/epoch_loss | 与终端epoch汇总一致的总损失及活跃批次分项均值 |
| train/learning_rate | 各参数组本次更新实际使用的学习率 |
| train/gradient | 裁剪前范数及该次更新是否超过阈值 |
| train/epoch_gradient | 整个当前epoch统计窗口的median、p95、max、裁剪比例 |
| validation/modulation、validation/protocol | Accuracy、Micro/Macro/Weighted-F1、Precision/Recall、CE及平均置信度等 |
| validation/*_class_* | 各类别的Recall、Precision、F1、Support |
| validation/parameters | 七字段各自的RMSE、MAE、绝对误差p95、有效计数 |
| validation/pretraining/loss | 预训练验证总损失 |
| validation/local_proxy | 本地代理分，不是官方竞赛成绩 |
| validation/selection | 实际选模分、历史最优和早停计数 |
| run | 配置文本、参数量、数据量、阶段与有效batch等 |

任务均衡采样时，一个更新可能只训练某项任务；其他分项的0通常是占位值。新记录使用明确的active_parts，只写活跃分项，避免曲线被占位0污染。梯度累积时step分项沿用原events的按累计微批次归一化口径；epoch分项是原终端的活跃批次均值，二者不应直接当作同一平均值。

多卡时训练分项沿用原rank0日志口径，没有为可视化增加跨卡归约；当前单卡设置不涉及该差异。未引入逐步权重直方图、计算图追踪或额外模型前向。部分epoch或恢复后的剩余epoch统计窗口会随原日志记录complete标识，不把局部窗口冒充完整历史epoch平均值。

## 已运行实验：补录与实时跟随

已经在本次接入中补录：

- `logs/supervised_main/20261112_171823_031143/tensorboard_history/`
- `logs/supervised_pyramid_temporal/20261113_115059_514367/tensorboard_history/`

旧进程不会热加载新日志功能。如果它还在训练，可以在另一个终端执行：

```bash
python code/training/backfill_tensorboard.py \
  --run logs/supervised_pyramid_temporal/20261113_115059_514367 --follow
```

默认每5秒读取新增JSONL；不加--follow只补录当前已有记录。运行目录要指向精确时间戳，而非实验父目录。补录有字节偏移状态，正常重复执行只读取新内容；不完整的末行会等待写完。按Ctrl+C退出即可，训练进程不受影响。

历史输出独立放在tensorboard_history，避免与原生writer同时写同一路径。它只读取现有配置、JSONL和可匹配的metrics.json；不读checkpoint、不重跑验证、不修改原日志或模型。历史日志缺失的step、过去各轮未保存的完整逐类数据都不会被补造：一般每轮可恢复Acc/Macro/Weighted-F1和RMSE，只有与当前metrics.json匹配的轮次可补CE、逐类表等完整字段。旧记录无法区分真实活跃零损失与占位零时，会跳过零分项。

历史曲线请使用Step轴比较；事件文件的wall time是导入时间。能够确定恢复step的旧start记录会清理弃用尾部；极早的预训练恢复记录若缺step，工具会提示无法精确清理该分支，不声称重建出不存在的历史信息。

## 实际验收

9项事件文件专项检查、31项配置/恢复/日志/进程/微调回归检查通过，共40项。完整midfusion模型在真实小缓存上CPU BF16训练2步、恢复到3步；同时关闭TensorBoard从零训练到3步，最终模型权重逐张量相同。读取真实事件文件确认：优化器step为1/2/3、验证step为2/3、注入的旧尾部999被清理、活跃任务过滤和checkpoint指标一致。未启动正式GPU训练或停止现有训练。

验收记录：`artifacts/tensorboard_implementation/20261113/training_status.json`及阶段日志。测试命令：

```bash
python code/training/test_tensorboard.py
```
