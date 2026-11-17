# 当前共享基座：三阶段入口与自动流程

每阶段只保留一个本方案的正式配置，均交给`code/training/train.py`读取：

| 阶段 | 配置 | 初始化与目标 |
| --- | --- | --- |
| 从头有监督 | `code/configs/supervised/shared_base.json5` | 随机基座/任务头，三任务监督，前14类相位增强 |
| 结构预训练 | `code/configs/pretrain/shared_base.json5` | 随机基座，原始I/Q第一轮词表的原子/组合预测，30轮；不作输入增强 |
| 下游微调 | `code/configs/finetune/shared_base.json5` | 加载对应预训练backbone，随机任务头，全参微调，前14类相位增强 |

旧round配置和历史专用实验入口保留，不自动替换其权重或含义。本方案未启用density、Patch偏移、金字塔或双视图一致性损失。

## 本次架构改进

保持768宽、12层、6头、P16时频Transformer和调制/协议/参数任务头。新增`model.iq_geometry`模块作为共享时域路径的局部残差，进入原时频融合与同一套Transformer；不是第二个基座或分类器。

将原时域第一层卷积的I/Q权重记作A/B，复用它们计算：

`z = (A*I + B*Q) + j(A*Q - B*I)`。

两部分共用原卷积参数，不新增一套学习滤波器。这一线性表示对全局相位旋转等变。随后在每个原P16位置计算：

- 滤波功率的均值和标准差；
- 滞后1、4个采样点的复相关实部/虚部，经能量归一化得到局部相干关系。

这些描述在全局相位旋转下保持不变，但局部幅度分布、相位演变和随时间变化的频率仍可不同。它们按原时间顺序保留，不像旧density方法先全局汇总。

描述经过LayerNorm和64维瓶颈映射到768维，零初始化输出投影以残差加入原时域token。残差尺度由旋转不变的局部能量及原Patch卷积核增益估计，不使用会随相位变化的原始token RMS。max_scale约束相对这一估计尺度的输出，不能理解为严格限制在原token范数的10%。

原I/Q、幅度、已有相位差、log RMS、局部/全局频谱、位置及参数读出路径继续存在；没有把整个基座硬变成对所有变化都不变的编码器。新增201280参数，最终95908749参数，共享比例约98.06%。

这里的1/4是采样点间隔，不解释为真实符号周期；未宣称恢复了同步符号星座。借鉴的是[I/Q共享变换研究](https://arxiv.org/abs/2606.31352)的结构动机，并非完整复现Dualformer。模块收益尚待正式训练判断。

## 三阶段兼容约束

三份配置的`model`定义一致。只有损失、初始化、优化器分组和监督阶段增强不同。

- 结构预训练只读取已掩码的归一化IQ；新增特征不读取被遮蔽的原值。保留原P16坐标与512/1024长度契约，不改变结构标签位置。
- 当前第一轮使用`artifacts/structure_research_01/codebook_v1`：原始I/Q特征构词、没有旧教师；原子512、组合1024，重建权重0。30轮后用本次新模型重新构词，第二轮不能复用旧模型生成的历史v2。词表/数据fingerprint必须一致。
- `init_checkpoint`仅迁移backbone，不迁移旧任务头或结构预测头。宽度、深度、注意力头数、QK/位置/归一化语义需要匹配；已学习的geometry不能被悄悄丢弃或重解释。
- 原无geometry的权重允许以零残差方式增加模块；原生512/1024长度的三任务初始输出逐值相同。新增路径还检查了padding和非零残差下的掩码泄漏。
- 预训练不做相位增强；有监督/微调对前14类纯调制记录作全局相位旋转，不作循环时移。验证和推理使用原始观测，无真实类别分流。

这些约束也已写入仓库`AGENTS.md`，后续架构改动须同步维护三阶段。

## 单阶段手动运行

从仓库根目录、py313环境启动，以下命令每次只执行一项：

```bash
CUDA_VISIBLE_DEVICES=0 python code/training/train.py --config code/configs/supervised/shared_base.json5
CUDA_VISIBLE_DEVICES=0 python code/training/train.py --config code/configs/pretrain/shared_base.json5
CUDA_VISIBLE_DEVICES=0 python code/training/train.py --config code/configs/finetune/shared_base.json5
```

手动微调默认读取独立预训练输出`logs/pretrain_shared_base`的对应运行；若要微调迭代流程的第二轮，请用`--init-checkpoint <本流程/pretrain_round2/时间戳/best.pt>`明确指定。指定流程目录时会解析该流程已完成第二轮的精确best.pt。

## 配置驱动的两轮预训练与独立微调

现统一使用 `python code/training/train.py --config ...`，两轮预训练完成后停止，微调单独启动。专用Python及bash启动器已移除。P16历史流程配置可交给统一入口执行预训练：

```text
python code/training/train.py --config code/configs/pipeline/shared_base_iterative.json5
python code/training/train.py --config code/configs/finetune/shared_base.json5
```

微调前在其配置的 `run.init_checkpoint` 指定所需P16第二轮流程或精确`.pt`，避免使用其他单轮结果。旧流程配置中的`configs.finetune`不参与两轮预训练。

两轮流程仍冻结代码及实际配置，第一轮30轮→用本轮最佳骨干重建词表→继承骨干第二轮50轮；结构头、优化器及cosine重新初始化。恢复保留当前轮状态，构词只占第一张卡。详细参数、日志和恢复规则见[统一训练说明](unified_training.md)。历史权重、日志及运行快照保留。
