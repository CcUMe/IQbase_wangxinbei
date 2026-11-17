# P8共享基座：配置驱动预训练与独立微调

2026-11-15新增候选。它把整个共享时频骨干的patch步长从16改为8，不增加P8旁路或第二个编码器。768维、12层、6注意力头、64点局部功率谱窗、I/Q几何残差和三任务头保留。512/1024点输入分别产生64/128个有效token。三阶段均使用同一骨干和输入语义。

这次是一个成套候选：除P8外，预训练随机连续遮蔽跨度从2改为8，最终遮蔽比例仍为30%；所以将来与P16比较不能把差值单独归因为patch大小。没有新增基于ESU边界的遮蔽或损失筛选，没有引入双视图一致性、EMA或新分类头。原子/组合损失各权重1，连续重建权重0。

## 三阶段入口

| 阶段 | 正式配置 |
|---|---|
| 从头有监督 | code/configs/supervised/shared_base_p8.json5 |
| 第一轮结构预训练 | code/configs/pretrain/shared_base_p8.json5 |
| 预训练基座微调 | code/configs/finetune/shared_base_p8.json5 |
| 两轮预训练 | code/configs/pretrain/two_rounds_p8.json5 |

第一轮原始I/Q欧氏词表预训练50轮→本轮最佳P8模型重建词表→继承P8骨干预训练50轮后停止，结构头重新初始化。微调由独立配置加载第二轮骨干，全参20轮。LR-v2两轮入口保留用户当前GPU2,3、每卡256；微调默认单卡GPU0、batch256。历史运行保持各自冻结配置。

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

```text
python code/training/train.py --config code/configs/pretrain/two_rounds_p8.json5
python code/training/train.py --config code/configs/finetune/shared_base_p8.json5
```

两项训练独立启动，预训练不自动微调。配置参数、精确权重选择、日志及恢复方式见[统一训练说明](unified_training.md)。重复专用入口已移除；历史实验配置和训练快照保留。

## 检查范围

test_iq_geometry.py覆盖P8/P16的非零几何残差掩码目标不泄漏、全局相位性质、有限梯度、512/1024有效长度与padding一致性、预训练骨干迁移。test_iterative_pipeline.py --processes 2 --patch-size 8用真实小缓存完成CPU/Gloo双进程两轮预训练、单进程构词、微调、导出和12条统一推理，并执行P8从头有监督入口。检查通过代表执行契约成立，不能宣称准确率提升或双卡加速比。
