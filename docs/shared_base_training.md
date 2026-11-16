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

手动微调默认读取独立预训练输出`logs/pretrain_shared_base`的对应运行；若要微调迭代流程的第二轮，请用`--init-checkpoint <本流程/pretrain_round2/时间戳/best.pt>`明确指定。自动流程不使用全局指针，而是绑定本流程第二轮的精确best.pt。

## 当前自动迭代流程：30轮→新模型构词→50轮→微调

```bash
bash code/training/run_iterative_pretrain.sh
```

默认物理GPU0、单进程，所有阶段串行。需要另一张已分配单卡时可设`C01_GPU=1`；新流程会检查该卡是否已有计算进程，非空则拒绝新启动，不会自行结束其他任务。Python默认本项目py313，可用`C01_PYTHON`指定解释器。

脚本后台执行，关闭终端不影响运行。集中配置为`code/configs/pipeline/shared_base_iterative.json5`，包含预算、构词和运行参数，不复制多套完整模型配置。

| 顺序 | 输入/初始化 | 训练或操作 | 输出 |
| --- | --- | --- | --- |
| 第一轮 | 当前新架构随机初始化＋原始I/Q的v1词表 | 预训练30轮；峰值LR3e-4 | 本轮结构验证损失最佳的best.pt |
| 重建词表 | 冻结本流程第一轮best.pt提取完整输入特征 | 仅训练集拟合512原子聚类与最多512次BPE合并；全数据固定赋标签 | 新建本流程独立词表，不覆盖v1/v2 |
| 第二轮 | 第一轮完整共享backbone＋新词表 | 预训练50轮；新结构头峰值3e-4，骨干1e-4 | 本轮结构验证损失最佳的best.pt |
| 微调 | 第二轮共享backbone＋随机三任务头 | 20轮；任务头3e-4，骨干3e-5 | 按调制准确率选择的best.pt |
| 导出 | 微调best.pt | 删除辅助训练头、校验权重往返 | exported_model单模型 |

第一轮严格使用原始I/Q构建的词表，不加载旧模型；第二轮继承第一轮骨干，包括已学习的iq_geometry。新词表即便仍为512/1024类，类别编号语义已改变，所以结构预测头必须随机重建；不能使用全模型warmstart。切换阶段时优化器和3%warmup+cosine调度器也重新创建；只有恢复同一中断阶段时才保留优化器与调度进度。预训练重建损失权重始终为0。两轮合计80轮是当前默认预算，不是已验证最优分配。

构词默认抽取约200000个训练patch用于聚类，20次聚类迭代；最多20000条去重训练序列拟合BPE，组合跨度最多5。第一轮教师只在离线构词时使用，第二轮与微调不保留在线教师，也不引入推理依赖。验证集不参与聚类中心和BPE规则拟合。训练单卡batch256、无梯度累积；构词batch128。

每次建立`logs/shared_base_iterative/<时间戳>/`，保存：

- 冻结源码、pipeline_config.json、pretrain_round1_config.json、codebook_round2_config.json、pretrain_round2_config.json、finetune_config.json；各阶段使用同一源码快照，后续修改仓库不影响运行；
- `pipeline.pid`、`status.json`、`manifest.json`，后者记录精确阶段运行目录及checkpoint；
- `pretrain_round1_console.log`、`codebook_round2_console.log`、`pretrain_round2_console.log`、`finetune_console.log`；训练阶段目录内另有可读train.log、JSONL、TensorBoard和每1000步last.pt；
- `codebooks/round2_<时间戳>/`保存本次新词表；manifest记录第一轮教师精确路径及SHA256，防止混入历史教师；
- 前一阶段完整成功后才进入后一阶段；微调结束后导出单模型到`exported_model/`。

若阶段失败，会记录failed及错误，不会加载无关旧权重继续下一阶段。最后写`completion.json`。原weights目录不覆盖。

## 查看与恢复

`logs/shared_base_iterative/latest_pipeline.json`给出最近创建的流程目录。直接查看其中`status.json`和当前阶段console.log即可。

```bash
bash code/training/run_iterative_pretrain.sh --resume logs/shared_base_iterative/<流程时间戳>
```

恢复检查冻结源码SHA256，使用冻结配置及当前训练阶段last.pt；已完成阶段跳过。保留原优化器和cosine进度。若离线构词中断，已写出完整codebook.json的词表会校验后复用，否则在新attempt目录重新构建，保留半成品。停止时可向pipeline.pid发送SIGTERM，协调程序只停止自己的子进程并等待其释放GPU；训练从最近已保存断点恢复，不额外承诺保存中断瞬间状态。

旧`run_pretrain_finetune.sh`两阶段脚本及旧流水线快照保留，原来已经启动的80轮/v2训练不会被配置文件修改自动改变；新实验应使用本节迭代入口。

## 单卡与双卡切换

`train.py`本身支持双卡DDP，三阶段正式训练配置均可只调整`runtime.gpus`和`runtime.processes`：单卡为`"gpus":"0", "processes":1`；双卡为`"gpus":"0,1", "processes":2`。只支持通信组0,1或2,3或4,5，不支持跨组组合。也可在单次命令覆盖：

```bash
# 双卡第一轮预训练；微调时换成finetune配置并明确指定对应预训练检查点。
python code/training/train.py --config code/configs/pretrain/shared_base.json5 --gpus 0,1 --processes 2
# 单卡。
python code/training/train.py --config code/configs/pretrain/shared_base.json5 --gpus 0 --processes 1
```

batch_size是每进程/每卡批量，梯度累计为1时，单卡256对应全局256、双卡256对应全局512；如需维持全局256，双卡将data.batch_size改为128。改变卡数/批量会改变每轮更新数和cosine总步数；不能把单卡中断checkpoint按双卡resume，切换需新训练流程。

以上命令适用于直接运行train.py。迭代pipeline会按自身流程配置覆盖训练模板的runtime，所以切换完整一键流程应使用下方双卡入口及对应pipeline配置。旧两阶段pipeline仍限制单卡；构词保持单进程。

## 双卡一键迭代入口

```bash
# 默认0,1，确认两张卡都空闲后启动。
bash code/training/run_iterative_pretrain_dual.sh

# 选择另一组已分配的空闲通信卡组。
C01_GPUS=2,3 bash code/training/run_iterative_pretrain_dual.sh
# 等价CLI覆盖：
bash code/training/run_iterative_pretrain_dual.sh --gpus 4,5
```

独立配置为`code/configs/pipeline/shared_base_iterative_dual.json5`。第一轮30轮→构词→第二轮50轮→微调20轮→导出保持不变。两轮预训练和微调的`runtime.processes=2`，通过现有train.py启动torchrun/DDP；构词使用所选组的第一张物理卡，例如2,3组只在GPU2构词，另一张卡此阶段不承载本流程训练。导出使用CPU。每阶段完成并退出后才启动下一阶段，不同时启动多项训练。

`data.batch_size_per_gpu=128`覆盖三项训练阶段的每卡批量，全局batch=128×2×累计1=256，与当前单卡保持一致；学习率不自动放大，3%warmup和cosine依据实际总更新数计算。构词batch独立为128。若希望全局512，将这个配置项改成256后新建流程；不能改正在运行或恢复中的冻结配置。

启动会检查所选两张卡的计算进程，任一卡忙则报错，不结束其他任务。默认双卡组0,1当前若已被其他训练占用，应等待其结束或选择自己已分配且空闲的2,3或4,5组。启动只创建新实验，不会自动停止或转换已有单卡流程。

输出为`logs/shared_base_iterative_dual/<时间戳>/`，目录结构、TensorBoard、best/last、词表教师绑定、源码快照与单卡一致。`latest_pipeline.json`给出最新流程，`status.json`记录当前阶段、子进程PID和阶段进程数。后台脚本的启动日志位于该根目录的launch/。停止时向pipeline.pid发送SIGTERM，会向当前阶段的整个进程组转发停止信号，涵盖torchrun和两个worker。

```bash
bash code/training/run_iterative_pretrain_dual.sh \
  --resume logs/shared_base_iterative_dual/<流程时间戳>
```

恢复自动使用冻结卡组，不需要再次传C01_GPUS；若显式指定则必须与原组一致。已完成阶段跳过，训练阶段恢复last.pt及原优化器/调度；不能将当前单卡流程直接交给双卡入口resume。单卡入口`run_iterative_pretrain.sh`及其旧配置继续可用。

双进程执行检查可在CPU运行`python code/training/test_iterative_pipeline.py --processes 2`：使用真实小缓存、Gloo双进程完成两轮训练、单进程构词、微调、导出和统一推理；该检查不代表NCCL性能、正式准确率或双卡加速比。

验证证据位于`artifacts/shared_base_pipeline/`：模块相位性质、有限梯度、预训练掩码泄漏、迁移和原生长度零残差一致性检查；真实小缓存完成有监督、预训练、微调、源码冻结自动串联、导出及12条三任务推理。小缓存分数仅用于执行检查，不是准确率收益证据。
