# P8共享基座：双卡迭代预训练与微调

2026-11-15新增候选。它把整个共享时频骨干的patch步长从16改为8，不增加P8旁路或第二个编码器。768维、12层、6注意力头、64点局部功率谱窗、I/Q几何残差和三任务头保留。512/1024点输入分别产生64/128个有效token。三阶段均使用同一骨干和输入语义。

这次是一个成套候选：除P8外，预训练随机连续遮蔽跨度从2改为8，最终遮蔽比例仍为30%；所以将来与P16比较不能把差值单独归因为patch大小。没有新增基于ESU边界的遮蔽或损失筛选，没有引入双视图一致性、EMA或新分类头。原子/组合损失各权重1，连续重建权重0。

## 三阶段入口

| 阶段 | 正式配置 |
|---|---|
| 从头有监督 | code/configs/supervised/shared_base_p8.json5 |
| 第一轮结构预训练 | code/configs/pretrain/shared_base_p8.json5 |
| 预训练基座微调 | code/configs/finetune/shared_base_p8.json5 |
| 完整双卡迭代流程 | code/configs/pipeline/shared_base_iterative_p8_dual.json5 |

第一轮原始I/Q词表预训练30轮→本轮最佳P8模型重建词表→继承P8骨干预训练50轮、重新初始化结构头→全参微调20轮→导出。双卡默认物理GPU2,3，每卡128，全局batch256，梯度累计1。重建词表时只用GPU2；导出用CPU。学习率和cosine沿用P16迭代方案。

## 一次性准备P8缓存与首轮词表

```bash
python code/training/prepare_patch_view.py \
  --source artifacts/data_full --output artifacts/data_full_p8 --patch-size 8
python code/training/build_codebook.py \
  --config code/configs/codebook/shared_base_p8_round1.json5
```

P8数据视图通过链接读取原iq.npy、metadata.npz和split_manifest.parquet，波形不重采样，行顺序、分类/参数标签、有效长度、训练验证划分和参数标准化完全保留。只有独立dataset.json中的patch_size和指纹改变，并记录原缓存路径/指纹。构词与训练均以只读方式打开共享数据文件。

首轮新词表保存到artifacts/structure_research_p8/codebook_v1，原子512类、最多512次组合合并。它基于P8原始特征构建，不读取P16教师或历史P16标签。每条最长信号的结构标签位置由64变成128，故不能复用P16的第一轮/第二轮词表。首轮构词配置默认CPU，避免准备期间占用训练卡；第二轮用本流程P8教师在GPU2离线构词。

## 启动与恢复

```bash
bash code/training/run_iterative_pretrain_p8_dual.sh

# 已完成一次准备后，也可直接交给通用双卡入口：
bash code/training/run_iterative_pretrain_dual.sh \
  --config code/configs/pipeline/shared_base_iterative_p8_dual.json5

# 恢复本P8流程，不能填写P16流程目录：
bash code/training/run_iterative_pretrain_p8_dual.sh \
  --resume logs/shared_base_p8_dual/<时间戳>
```

输出为logs/shared_base_p8_dual/<时间戳>/，含源码快照、各阶段实际配置、console日志、TensorBoard、last/best、第二轮词表和导出模型。启动日志由通用双卡launcher写入logs/shared_base_iterative_dual/launch/；实际P8运行目录由logs/shared_base_p8_dual/latest_pipeline.json指向。所选卡组忙时拒绝启动，不自动终止其他任务。

三个正式配置也能直接交给train.py单阶段运行；微调需用--init-checkpoint明确指定对应P8检查点。P16与P8前端权重尺寸和位置契约不同，当前不允许直接完整迁移或resume；这次第一轮从随机初始化开始。P16配置、词表、权重与日志保留。

## 检查范围

test_iq_geometry.py覆盖P8/P16的非零几何残差掩码目标不泄漏、全局相位性质、有限梯度、512/1024有效长度与padding一致性、预训练骨干迁移。test_iterative_pipeline.py --processes 2 --patch-size 8用真实小缓存完成CPU/Gloo双进程两轮预训练、单进程构词、微调、导出和12条统一推理，并执行P8从头有监督入口。检查通过代表执行契约成立，不能宣称准确率提升或双卡加速比。
