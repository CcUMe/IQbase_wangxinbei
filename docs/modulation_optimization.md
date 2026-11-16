# 真实C01调制识别优化入口

代码入口：`code/training/optimize_density.py`；唯一实验配置：`code/configs/supervised/density_optimization.json5`。

当前默认方法为仅前14类相位/循环时移增强，4800步联合续训；从原SAM源权重开始。最终EMA调制Acc 67.570%、Macro-F1 67.332%、Weighted-F1 67.266%，前14类Acc 52.407%，协议Acc 97.202%。导出模型在`artifacts/density_optimization/selected_model_20261114`，完整比较见[真实实验报告](research/C01_共享基座调制优化实验_20261114.md)。分布结构与软目标方法未作为默认方案。

后续拆分与第二种子复核已完成：仅相位EMA为67.618%/67.680%，组合为67.570%/67.522%；50%原始视图混合为67.278%（仅主种子）。相位单项保留了主要增益，但相对组合的微小领先未满足预设严格替换条件，因此默认方法与原参考权重保留。较高分的相位单项研究候选已导出至`artifacts/modulation_factors/20261114_205946/phase_only_candidate`；完整结论见[拆分实验报告](research/C01_相位时移拆分与原始视图实验_20261114.md)。

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
 /home/yangqilin/anaconda3/envs/py313/bin/python -u code/training/optimize_density.py \
 --config code/configs/supervised/density_optimization.json5
```

默认数据是`artifacts/data_full`，沿用现有划分与原始IQ。以现有约95.7M、768×12共享时频基座的SAM候选为起点。没有修改原训练数据、标签、划分或历史权重。单卡训练，不使用梯度累积。

常规`code/training/train.py`也支持在`data.augmentation.modulation_only.class_indices`中指定训练侧增强类别。例如前14类为`[0,1,2,3,4,5,6,7,8,9,10,11,12,13]`，空列表表示全部调制类别。此处类别索引来自当前schema，不能用于推理分流。常规入口对带协议或参数标签的记录保持保护。常规入口和配对实验入口的随机数流不同，应分别保存实际配置，不能宣称两者逐步完全复现。

## 可选方法

`--arms`指定逗号分隔的方法，分别建立模型训练、保存与验证，不能把多个方法的预测平均作为结果。

| 方法 | 实际修改 |
| --- | --- |
| control | 原模型普通续训，提供同预算控制 |
| head_control | 额外训练原调制头，再联合训练，控制额外样本曝光 |
| density | 共享前端增加可学习分布编码，调制头增加相应输入列 |
| density_curriculum | 相同结构，加训练侧稳定性课程权重 |
| phase_shift | 仅调制批次做相位／循环时移增强 |
| phase_shift_first14 | 仅目标14类做上述增强，其余类别保持原始输入 |
| phase_shift_first14_phase_only | 相同目标14类，仅相位增强，关闭时移 |
| phase_shift_first14_shift_only | 相同目标14类，仅循环时移，关闭相位增强 |
| phase_shift_first14_phase_only_native_mix | 记录级选择50%原始视图、50%完整相位增强视图；当前结果未超过全量相位增强 |
| adaptive_smoothing | 依据训练侧稳定性决定强度，目标14类内均匀平滑 |
| posterior_distill | 相同强度，使用教师的类内软分布替代均匀分布 |
| phase_shift_posterior | 全调制增强，加软分布目标 |
| phase_shift_first14_posterior | 目标14类增强，加软分布目标 |

增强和目标分布中的类别信息只用于有监督训练，推理时统一预测全部23类。协议与参数训练不作相位／时移增强，仍使用原始IQ与原标签。增强保存每条记录的长度与能量。

## 分布编码结构

原输入仍由一套时域前端、局部复频谱和全局频谱进入同一Transformer。新增模块读取原时域第一层卷积的细尺度特征，在P16压缩之前计算可学习软分配以及零／一／二阶条件矩，采样滞后为0、4、16点。这些是样点间隔，不解释为真实符号周期。

分布描述一方面受限残差注入共享时域路径，供三个任务访问；另一方面作为调制分类器的额外输入列。分类公式为：

`logits = W2(GELU(W_context * context + W_density * descriptor + bias))`

它只有一个调制分类输出，没有多个基座或预测集成。原时频主干、尺度与绝对时间通路保持可访问。新列和共享注入初始为零，加载原权重后的三个任务初始输出逐值检查通过。

当前实现对应历史最好单尺度基座，未对金字塔／串行层级骨干启用该模块；旧配置默认关闭，仍可加载原权重。分布版本总参数96,546,205，共享比例97.66%。

该结构借鉴[NetVLAD](https://arxiv.org/abs/1511.07247)和[可学习字典编码](https://arxiv.org/abs/1804.00385)的分配／统计聚合思想，是独立适配，不是论文完整复现。

## 训练阶段和公平比较

- 分布编码先准备新增参数：固定原基座及原分类器参数，只训练新增描述与输入列。原上下文在这阶段置零，让新增输入实际获得分类梯度。
- 随后恢复完整上下文，所有参数进行三任务联合训练。
- 普通control不包含准备阶段；head_control补上相同的额外调制曝光。分布方法应同时对比这两项和源权重。
- 增强与软目标方法直接联合训练，不包含新增参数准备。
- 默认联合4800次更新（基础比较用`--joint-steps 2400`；`--seed`用于配对复核，均写入实际配置）、batch256，三任务各约1600次更新。样本有放回抽取，不等于完整遍历一轮。
- 默认准备1200次调制更新；有准备的方法额外使用307200次调制样本曝光，应明确记录这项预算差异。
- 50步warmup后cosine至峰值20%。原权重LR3e-5，新分布参数LR3e-4；准备阶段LR5e-4。BF16训练、FP32验证。
- 各方法使用相同源权重、联合训练流及随机种子，分别保存last与EMA并作全量三任务验证。

## 可靠性与软目标

运行`build_density_reliability.py --output <新目录>`可以重新制作训练侧文件。它只读取调制训练记录，用已修正BN统计的小模型比较原始与全局相位旋转视图：

`score = min(p_y(original), p_y(rotated)) * exp(-JS)`

随后在每个真实训练类别内部作分位数排序。课程权重始终为正，各类别期望权重相同。该值是预测稳定性代理，不是SNR，验证数据不参与排序。

软目标方法中，默认只作用于前14类：

`epsilon = 0.4 * (1 - within_class_rank) * ramp`

`target = (1-epsilon) * one_hot(y) + epsilon * teacher_distribution`

教师分布在训练时归一化到目标类别组；均匀平滑控制使用相同强度和组范围。默认正确类别保留至少60%的目标质量。训练标签本身没有修改；推理不需要教师、可靠性文件、真实类别或其他额外元信息。

这是受[知识蒸馏](https://arxiv.org/abs/1503.02531)启发的记录自适应平滑，教师本身弱于主模型，因此必须以真实C01对照结果判断，不能预先认定有效。

## 日志、权重与推理

每次运行自动创建`logs/supervised_density/<上海时间戳>/`，其下按方法保存：

- 实际配置、参数报告、可读train.log、events.jsonl及实时TensorBoard；早期实验的原始文本另存train.original.log；
- last／EMA完整权重、全量原始预测、三任务指标与前14类条件分组；TensorBoard的estimators/last与estimators/ema分别显示最终指标；
- 固定训练流及SHA256，便于核对控制是否一致。

完整验证包含118182条记录，其中调制20885条；没有删除困难记录或更换验证划分。指标由`summarize_modulation_optimization.py`导出Acc、Macro／Weighted-F1、逐类指标、七参数RMSE／MAE／P95及逐记录配对比较。多个方法共享的验证集已经用于开发，不是独立隐藏测试。

checkpoint可由更新后的`RFModel(**model_bundle(checkpoint))`严格加载。`resume_allowed=False`表示当前实验未保存完整优化器断点，应作warmstart，不能声称严格恢复。

现有`code/training/export.py`和`code/infer.py`支持这些模型。分布版本已通过逐张量导出往返及12条、两种长度、三个任务的统一推理检查。最终候选应以全量验证结果选择后再导出；源权重和默认weights不自动覆盖。

```bash
/home/yangqilin/anaconda3/envs/py313/bin/tensorboard \
 --logdir logs/supervised_density --port 6006
```

目前已完成的第一轮结构对照，目录`20261114_135908_266843`：源权重66.919%，普通续训EMA66.881%，额外头训练控制66.694%，分布EMA66.785%，分布＋课程EMA66.814%。这版分布结构未带来收益，尚未作为默认模型推荐。后续真实数据优化结果以汇总报告为准。

2400步定向增强初步候选：`logs/supervised_density/20261114_162246_625207/phase_shift_first14/ema.pt`，调制Acc 67.211%、Macro-F1 67.021%、Weighted-F1 66.955%，前14类Acc 51.857%，协议Acc 97.209%。相对源权重的Acc提高0.292个百分点，逐记录配对的未作多重比较校正的95%近似区间为[-0.037,+0.621]个百分点，仍包含零。因此是小幅正向候选，不能称为已确认突破。只覆盖这一初始化和开发划分。

该候选的单模型导出在`artifacts/density_optimization/candidate_6721`，已经通过逐张量导出往返与CPU FP32的12条三任务推理检查。默认`weights`未覆盖。4800步配对复核已经完成，最终EMA为67.570%，结果单列在summary_4800_20261114，不与2400步混作同预算对照。
