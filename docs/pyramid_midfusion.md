# 金字塔骨干：第4层单次跨尺度瓶颈交互

TensorBoard默认随训练/续训记录到本次运行目录；旧进程可增量补录或跟随JSONL。[可视化使用说明](tensorboard.md)。

2026-11-13，Asia/Shanghai。当前主干优化候选为 `code/configs/supervised/pyramid_midfusion.json5`。保留原P16/P8/P4并行架构和原版三任务头，**显式关闭分段调制头**。只研究让中、细尺度信息在编码中期进入P16后半段的作用，未同时加入串行BiLSTM、粗尺度局部卷积、多窗频谱、新损失或预训练目标。尚无正式准确率增益结论。

## 启动

在项目根目录、py313环境执行：

```bash
python code/training/train.py --config code/configs/supervised/pyramid_midfusion.json5
```

GPU1单卡、batch256、梯度累积1；数据仍为artifacts/data_full。输出为 `logs/supervised_pyramid_midfusion/<上海时间戳>/`。AdamW峰值3e-4、weight decay0.01、500步预热后cosine至5%、40轮上限、8次全量验证早停保持原版。默认从零联合监督，不自动加载权重或替换正式weights。

原版 `pyramid_v1_171823.json5` 与分段版 `pyramid_temporal.json5` 内容均保留。三种架构不能交叉严格resume或完整warmstart；继续既有分段运行仍使用pyramid_temporal。新主干的断点恢复示例：

```bash
python code/training/train.py --config code/configs/supervised/pyramid_midfusion.json5 \
  --resume logs/supervised_pyramid_midfusion/<时间戳>/last.pt
```

## 前向结构

```text
原始I/Q → 原有效长度/RMS/时频前端（保留log RMS和全局谱）
 ├ P16嵌入 → 原Transformer第1～4层 ──┐
 └ 共享复数stem                       │
   ├ P8局部块＋原分支Transformer ─────┤
   └ P4局部块＋原分支Transformer ─────┘
                    ↓
        一次共享瓶颈读取／同时残差写回
                    ↓
 ├ P16 → 原Transformer第5～8层 → 原末层归一化
 ├ P8  → 保留更新后的中尺度token
 └ P4  → 保留更新后的细尺度token
                    ↓
         原末端跨尺度摘要交互与三任务读出
```

这是粗分支中期融合：仅P16在交互后继续通过后4层，P8/P4没有新增交互后的编码层。中、细分支和各自频谱交互只计算一次；结果通过当前前向的context传到末端，不重做特征提取，也不跨样本缓存状态。

1024点输入仍为P16的64×384、P8的128×192、P4的256×192；512点长度减半。任务头仍输出23类调制、8类协议及七个原单位参数。三个任务共享所有编码和中期交互权重，没有按任务建立独立基座。

## 瓶颈细节

1. 三尺度各自经过更新支路的PreNorm和线性投影至192维，加入样点中心/有效长度坐标与尺度标识；不额外归一化整个原始残差路径。
2. 8个192维可学习交互token，分别以独立的有效key softmax读取三个尺度；读取注意力的权重在尺度之间共享。
3. 三份读取结果等权平均，加到初始交互token上。细尺度不会仅因token数量更多，就在统一softmax中天然获得更多源质量；等权汇总不代表三尺度对决策贡献相等。
4. 三个尺度分别读取同一组更新后的交互token，投影回384/192/192维，并以初始0.1、可自由学习的逐通道残差增益写回。
5. 先计算全部更新量，再写回各尺度，避免顺序更新。完整尺度序列保留，8个token只传递交换信息，不替代原token。

只有更新支路使用PreNorm，未施加相位/幅度不变性约束。位置按原样点中心和长度除以1024编码，与末端交互一致；共享交互token本身不假装对应8个真实时间位置。不推测真实采样率、符号周期或SNR。无效query在写回后置零，无效key在读取时遮蔽。

新增注意力矩阵的规模约随8×三尺度token总数增长；线性投影、特征保留及后4层仍有开销，未据此宣称GPU吞吐已经改善。

## 配置

新增参数集中在 `model.mid_fusion`，原版与分段版缺省为关闭：

| 参数 | 本配置 | 含义 |
|---|---:|---|
| enabled | true | 开启单次中期交换 |
| after_block | 4 | 第几层完成后执行，从1计数；必须小于encoder.depth |
| tokens | 8 | 可学习交互token数量 |
| width | 192 | 交换公共维度，能被heads整除 |
| heads | 3 | 交互cross-attention头数，与P16的6头分别配置 |
| dropout | 0.1 | 新增注意力和残差支路dropout |
| residual_scale | 0.1 | 逐通道增益初值，须为正；不设硬上限 |

原 `model.heads.modulation_temporal.enabled=false`，任务头结构和参数量保持原版。插入层数会改变前向语义，即使张量形状相同也不能当作同架构warmstart；加载检查会拒绝。

## 规模与兼容性

| 项目 | 原版 | 中期交互版 |
|---|---:|---:|
| 总参数 | 20,966,006 | 21,565,238 |
| 共享参数 | 19,153,568 | 19,752,800 |
| 共享比例 | 91.36% | 91.60% |
| 任务头参数 | 1,812,438 | 1,812,438 |

新增599,232个参数全部注册在backbone.pyramid.mid_fusion，计入共享基座。满足当前实现检查的模型≤1B、共享>80%约束，目录与JSONL接口不变。

新增模块使用隔离的初始化随机状态。相同随机种子时，所有已有编码器与任务头张量均与原版一致。旧配置不创建新增参数；原版及分段版best.pt均已严格加载。零写回增益时，完整默认尺寸模型的三任务输出与同初始化原版逐元素一致。

## 已完成验收

- 7项新增检查：旧权重/零残差、细分支干预只影响后续粗层、细分支只计算一次、分尺度读取对相同token重复的不变性、512/1024及padding、三任务CPU BF16梯度、粗层监督反传到细分支、配置/加载拒错等分组检查。
- 67项现有配置、原金字塔、分段头、历史主干、日志、warmstart和恢复检查通过。合计74项。
- 完整384×8模型实测：仅干预细分支，P16第4层输出最大差0；第5层输入最大差约0.005247。证明信息确实进入后半段编码，不是仅改变了末端摘要。
- 原版生产checkpoint在当前实现与冻结旧源码上的调制/协议/参数预测逐元素差均为0。原版及分段版配置SHA256不变，原实验权重未写回。
- 完整模型在真实小缓存上CPU BF16训练2步，严格恢复到第3步；独立评估、导出、官方6行输入推理及JSONL格式检查通过。

这些是功能和信息通路验收，不能证明调制准确率提高。未启动正式GPU训练，未替换weights，未干预既有训练。后续应与原版同轮次的整体Accuracy、Macro/Weighted-F1、前14类、协议和七参数RMSE比较；不能把分段头结果和本主干结果混为同一实验。

记录：`artifacts/pyramid_midfusion_implementation/20261113/backbone_audit.json`、`pipeline_status.json`和各阶段日志。smoke_config只用于CPU短流程，缩小batch/预算并改变预热，不是正式配方。

实现：`code/models/multiresolution.py`的MidScaleFusion、MultiResolutionPyramid.encode_details，以及`code/models/rf_model.py`的Backbone.forward第4层交互点。测试：

```bash
python code/training/test_mid_fusion.py
python code/training/test_multiresolution.py
```
