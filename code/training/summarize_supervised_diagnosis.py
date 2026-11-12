"""Explain supervised generalization using condition slices, depth and exact neighbors."""
import argparse
import csv
import hashlib
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from training.diagnose_signal_conditions import FEATURES
from training.diagnose_supervised_depth import LAYERS
from utils.common import read_json,write_json


def bins_for(condition,feature,labels,name):
    index=FEATURES.index(name);edges=condition['train_thresholds'][name]
    return np.array([np.searchsorted(edges[str(int(label))],value,side='right') if label<14 else -1
        for label,value in zip(labels,feature[:,index])])


def slice_predictions(labels,prediction,groups):
    return [{'group':g,'support':int((groups==g).sum()),'accuracy':float(np.mean(prediction[groups==g]==labels[groups==g]))} for g in range(4)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=args.output.resolve();torch.set_num_threads(4)
    run=read_json(r/'run_info.json');condition=read_json(r/'conditions/summary.json')
    layer=read_json(r/'layer_probes/summary.json');neighbors=read_json(r/'neighbors/summary.json')
    protocol=read_json(r/'layers/protocol.json');coverage=read_json(r/'parameter_pair_coverage.json')
    assert protocol['initial_weight_sha256']==protocol['final_weight_sha256']
    assert all(v['cached_prediction_agreement']==1. for v in protocol['splits'].values())
    assert len(layer['results'])==12
    assert all(hashlib.sha256((Path('weights')/name).read_bytes()).hexdigest()==sha for name,sha in run['weights_sha256'].items())
    data={};calibration={};risk_names=['log_rms','high_frequency_power_fraction']
    for split in ['train','validation']:
        labels=np.load(r/'layers'/f'{split}_labels.npy');features=np.load(r/'conditions'/f'{split}_features.npy')
        logits=torch.from_numpy(np.array(np.load(r/'layers'/f'{split}_logits.npy',mmap_mode='r')))
        prediction=logits.argmax(1).numpy();confidence=logits.softmax(-1).max(1).values.numpy()
        loss=F.cross_entropy(logits,torch.from_numpy(labels),reduction='none').numpy()
        data[split]={'labels':labels,'features':features,'prediction':prediction,'confidence':confidence,'loss':loss,'groups':{}}
        calibration[split]={}
        for name in risk_names:
            groups=bins_for(condition,features,labels,name);data[split]['groups'][name]=groups
            values=[]
            for group in range(4):
                keep=groups==group;correct=prediction[keep]==labels[keep]
                values.append({'group':group,'support':int(keep.sum()),'accuracy':float(correct.mean()),
                    'mean_confidence':float(confidence[keep].mean()),'cross_entropy':float(loss[keep].mean()),
                    'wrong_prediction_confidence':float(confidence[keep][~correct].mean()) if (~correct).any() else None,
                    'errors':int((~correct).sum())})
            calibration[split][name]=values
    depth_slices=[]
    with (r/'layer_probe_condition_metrics.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['layer','seed','split','condition','quartile','support','accuracy'])
        for name in LAYERS:
            x=np.load(r/'layers'/f'train_{name}.npy',mmap_mode='r')
            for seed in [2026,2027]:
                saved=torch.load(r/'layer_probes'/f'{name}_seed{seed}.pt',map_location='cpu',weights_only=True)
                head=torch.nn.Linear(x.shape[1],23);head.load_state_dict(saved['head']);head.eval()
                prediction=[]
                with torch.no_grad():
                    for begin in range(0,len(x),8192):
                        batch=torch.from_numpy(np.array(x[begin:begin+8192]))
                        prediction.append(head((batch-saved['mean'])/saved['std']).argmax(1).numpy())
                train_prediction=np.concatenate(prediction)
                val_prediction=np.load(r/'layer_probes'/f'{name}_seed{seed}_predictions.npy')
                row={'layer':name,'seed':seed,'slices':{}}
                for split,pred in [('train',train_prediction),('validation',val_prediction)]:
                    row['slices'][split]={}
                    for feature in risk_names:
                        values=slice_predictions(data[split]['labels'],pred,data[split]['groups'][feature])
                        row['slices'][split][feature]=values
                        for v in values:writer.writerow([name,seed,split,feature,v['group']+1,v['support'],v['accuracy']])
                depth_slices.append(row)
            print('Depth risk slices completed:',name,flush=True)
    small_root=Path('artifacts/diagnostics/20261111_110649_079619/frozen_small')
    assert np.array_equal(np.load(small_root/'validation_rows.npy'),np.load(r/'layers/validation_rows.npy'))
    small=np.load(small_root/'reference_validation_predictions.npy')
    historical={name:slice_predictions(data['validation']['labels'],small,data['validation']['groups'][name]) for name in risk_names}
    neighbor_slices={}
    for kind in ['embedding','raw_iq']:
        saved=np.load(r/'neighbors'/f'{kind}.npz');rows=saved['validation_rows']
        allrows=np.load(r/'layers/validation_rows.npy');lookup={int(row):i for i,row in enumerate(allrows)}
        positions=np.asarray([lookup[int(row)] for row in rows])
        neighbor_slices[kind]={}
        for name in risk_names:
            groups=data['validation']['groups'][name][positions]
            neighbor_slices[kind][name]={key:slice_predictions(saved['labels'],saved[key],groups) for key in ['prediction_1nn','prediction_5nn']}
    cross={}
    a=data['validation']['groups']['log_rms'];b=data['validation']['groups']['high_frequency_power_fraction']
    for x in range(4):
        for y in range(4):
            keep=(a==x)&(b==y);count=int(keep.sum())
            cross[f'{x+1},{y+1}']={'support':count,'accuracy':float(np.mean(data['validation']['prediction'][keep]==data['validation']['labels'][keep])) if count else None}
    conclusion={'calibration':calibration,'layer_condition_slices':depth_slices,'historical_small_validation':historical,
        'historical_comparison_not_matched_training_budget':True,'neighbor_condition_slices':neighbor_slices,
        'rms_hf_cross_slices':cross,'conditions':condition,'layer_probes':layer,'neighbors':neighbors,
        'parameter_pair_coverage':coverage,'frozen_weight_hash_unchanged':True,'default_weights_unchanged':True}
    write_json(r/'summary.json',conclusion)
    with (r/'condition_metrics.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['split','condition','quartile','support','accuracy','mean_confidence','cross_entropy','errors'])
        for split,items in calibration.items():
            for name,values in items.items():
                for v in values:writer.writerow([split,name,v['group']+1,*[v[k] for k in ['support','accuracy','mean_confidence','cross_entropy','errors']]])
    lines=['# 监督模型深入诊断：错误条件、编码器深度与表征几何','',
        '本轮没有训练或比较预训练目标。研究对象是原95.7M、44轮联合监督模型，使用其完整调制训练/验证记录。它是严重过拟合的原始基座；当前96.3M统计读出权重保持原样。','',
        '## 1. 过拟合集中在哪些观测条件','',
        '每个类别分别用训练记录确定四分位阈值，再把同一阈值应用于验证。以下都仅统计前14类，避免不同类别比例干扰。分层用到真实类别，只作离线分析，不是可部署的质量门控，也不是真实SNR。','',
        '| 条件 | 四分位 | 训练Acc | 验证Acc | 验证平均置信度 | 验证CE | 支持数 |','|---|---|---:|---:|---:|---:|---:|']
    for name in risk_names:
        for a,b in zip(calibration['train'][name],calibration['validation'][name]):
            lines.append(f'| {name} | Q{b["group"]+1} | {a["accuracy"]:.4%} | {b["accuracy"]:.4%} | {b["mean_confidence"]:.4%} | {b["cross_entropy"]:.5f} | {b["support"]:,} |')
    low=calibration['validation']['log_rms'];high=calibration['validation']['high_frequency_power_fraction']
    share_low=sum(v['errors'] for v in low[:2])/sum(v['errors'] for v in low)
    share_high=sum(v['errors'] for v in high[2:])/sum(v['errors'] for v in high)
    lines += ['',f'较低RMS的一半记录贡献前14类错误的 {share_low:.2%}；较高高频能量占比的一半贡献 {share_high:.2%}。两种分组重叠，不能相加。困难条件下训练准确率仍近100%，验证却很低，过拟合具有强烈的条件集中性。','',
        '高频能量按 |f|≥0.25 cycles/sample 统计；也可能包含合法宽带信号成分，不能直接当作噪声功率。RMS是信号与噪声共同作用的观测量，人工乘整体增益不会改变SNR，先前增益压力测试的小幅降分也说明“幅度小本身导致错误”不是充分解释。','',
        '## 2. 是否存在明显训练/验证观测分布变化','',
        '| 观测量 | 14类中KS距离中位数 | 最大KS距离 |','|---|---:|---:|']
    for name in ['log_rms','lag1_abs','high_frequency_power_fraction','spectral_flatness','spectral_effective_bins','correlation_half_lag']:
        v=[x['ks_distance'] for x in condition['distribution_shifts'][name]]
        lines.append(f'| {name} | {np.median(v):.5f} | {max(v):.5f} |')
    lines += ['',
        '这些一维统计的分布差异幅度较小，未发现足以直接解释巨大训练—验证差距的明显单变量偏移；这不是对高维域偏移或隐藏生成条件一致性的证明。谱有效bin数不是独立符号数，相关半宽不是符号周期。','',
        '## 3. 深度是否破坏了已有判别信息','',
        '| 位置 | 线性读出训练Acc均值 | 验证Acc均值 | 前14类验证Acc | 验证Macro-F1 |','|---|---:|---:|---:|---:|']
    for name in LAYERS:
        values=[v for v in layer['results'] if v['layer']==name]
        lines.append(f'| {name} | {np.mean([v["train"]["accuracy"] for v in values]):.4%} | {np.mean([v["validation"]["accuracy"] for v in values]):.4%} | {np.mean([v["validation"]["hard14"]["accuracy"] for v in values]):.4%} | {np.mean([v["validation"]["macro_f1"] for v in values]):.4%} |')
    lines += ['',
        '全部探针均冻结同一监督编码器、有效Token均值池化、训练侧标准化、线性头、20轮、CPU种子2026/2027、AdamW LR1e-3/WD0.01、cosine floor0.05；固定末轮。共同最终层特征与历史缓存逐元素差为0，训练/验证原头预测一致率100%；编码器全部权重哈希前后相同。','',
        '在此探针口径下，验证表现随深度增加，没有发现优于最终层的中间层，最终LayerNorm前后也没有明显降分。后几层大幅提高训练可分性，同时只有较小验证增益；这支持“新增容量主要增加训练拟合能力”，不支持“最后几层把原本更好的全局线性判别信息毁掉了”。不等于证明所有非线性信息都被完整保留。','',
        '| 位置 | RMS最低组训练Acc | RMS最低组验证Acc | RMS最高组验证Acc | 高频能量最高组验证Acc |','|---|---:|---:|---:|---:|']
    for name in LAYERS:
        values=[v for v in depth_slices if v['layer']==name]
        f=lambda split,key,g:float(np.mean([v['slices'][split][key][g]['accuracy'] for v in values]))
        lines.append(f'| {name} | {f("train","log_rms",0):.4%} | {f("validation","log_rms",0):.4%} | {f("validation","log_rms",3):.4%} | {f("validation","high_frequency_power_fraction",3):.4%} |')
    lines += ['', 'RMS最低组尤其具有区分力：最终层把训练记录变为近乎完全线性可分，验证却仍停在约13%–15%；相反，RMS最高组验证能力随深度明显提高。当前模型在较好观测上确实学到了可迁移信息，对最困难条件增加的拟合能力主要未转化为泛化。这里的“困难”由本模型的实测表现定义，不等于已证明输入在信息论上不可识别。']
    lines += ['', '## 4. 换分类规则和近重复检查能否解释问题','',
        '| 特征/度量 | 1NN Acc | 5NN Acc | 前14类1NN Acc |','|---|---:|---:|---:|']
    for name,v in neighbors.items():lines.append(f'| {name} | {v["one_neighbor"]["accuracy"]:.4%} | {v["weighted_five_neighbors"]["accuracy"]:.4%} | {v["one_neighbor"]["hard14"]["accuracy"]:.4%} |')
    maximum=neighbors['raw_iq']['nearest_similarity_quantiles'][-1]
    lines += ['',
        'embedding使用训练侧标准化和L2归一化的最终监督特征；raw_iq只在已知前14类子集中进行全窗复数余弦搜索，消除整体增益和相位差，不对齐时间或频偏。候选池严格只有训练集，训练/验证没有完全相同I/Q哈希。',
        f'该原始波形度量下，验证集最近训练波形的最大相似度约 {maximum:.6f}，没有达到0.99的近邻；最近跨标签实例已用Float64复核。没有发现大量“几乎完全相同波形配不同标签”的直接证据，但没有覆盖时间/频偏变换下的全部近重复。近邻低分不能证明信息论上不可识别。','',
        '## 5. 历史小模型是否失败在同样条件','',
        '| 条件 | 四分位 | 历史17M验证Acc | 原95M验证Acc |','|---|---|---:|---:|']
    for name in risk_names:
        for a,b in zip(historical[name],calibration['validation'][name]):lines.append(f'| {name} | Q{a["group"]+1} | {a["accuracy"]:.4%} | {b["accuracy"]:.4%} |')
    lines += ['',
        '这是原17M第5轮与95M第44轮两个历史实例，训练预算和配方不等，不能把差异单独归因于参数量。它描述的是：当前大模型对较好观测的提升更明显，最差条件下没有同步改善。','',
        '## 6. 目前更有证据的解释与未解决的问题','',
        '已确认：前14类错误高度集中于特定观测条件；同类条件的训练记录却几乎全被拟合正确。后几层继续提高训练可分性，泛化增益很小。现有证据优先支持“观测条件相关的困难样本发生严重训练拟合，稳定细粒度结构提取不足”这一定位。','',
        '合理但未完成因果验证的机制是：固定训练记录中稳定类别线索较弱或难提取，硬标签交叉熵仍要求逐条拟合；较大容量及重复曝光可以拟合记录特有细节，而新记录中这些细节不复现。现有结果还不能区分真实噪声/有限窗口造成的类别重叠，与接收机缺少同步、符号尺度建模或信道补偿造成的可提取信息未被利用。','',
        'RMS归一化按比例同时缩放信号和噪声，不会凭空降低SNR；缓存Float32及有效长度检查未发现错误。分类头重训和近邻规则都未带来根本改善，注意力稳定性修复也不等于获得了条件泛化。当前不应继续把原因笼统写成“只学了浅层特征”或“分类头太复杂”。','',
        f'真实参数覆盖仍有限：前14类只有 {coverage["0"]["communication_targets_first14"]} 条训练、{coverage["1"]["communication_targets_first14"]} 条验证记录能配到真实通信参数。正式PDF和数据没有可直接使用的fs/SNR字段，本轮不虚构物理单位，也不把这些代理量直接叫低SNR或真实符号数。','',
        '下一步最有针对性的验证是：固定初始化/样本流，对困难观测的监督权重或训练课程做单因素干预，分别监测整体、困难组和较好组；同时用有明确条件控制的接收处理验证信息是否仍可提取。不能靠筛掉验证难例提高报分，所有竞赛样本仍须预测。该建议尚未被本轮训练实验验证。','',
        '本地验证已用于模型选择和多次探索，不属于独立隐藏测试；后续修复收益仍需独立验证与编码器多种子重复。','',
        '## 产物','',
        '- [条件分层、置信度、交叉熵CSV](condition_metrics.csv)',
        '- [逐层逐条件CSV](layer_probe_condition_metrics.csv)',
        '- [完整指标](summary.json)',
        '- [条件阈值和分布统计](conditions/summary.json)',
        '- [冻结权重与特征一致性](layers/protocol.json)',
        '- [全部层级探针](layer_probes/summary.json)',
        '- [原始I/Q近邻与Float64复核实例](neighbors/summary.json)',
        '- [回归检查](tests.log)','']
    (r/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    for split in ['train','validation']:
        axes[0].plot(range(1,5),[v['accuracy']*100 for v in calibration[split]['log_rms']],marker='o',label=split)
    axes[0].set(xlabel='Within-class training RMS quartile',ylabel='Hard14 accuracy (%)',title='Condition-concentrated overfitting');axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].plot(range(1,5),[v['accuracy']*100 for v in historical['high_frequency_power_fraction']],marker='o',label='Historical17M / 5epochs')
    axes[1].plot(range(1,5),[v['accuracy']*100 for v in calibration['validation']['high_frequency_power_fraction']],marker='o',label='95M / 44epochs')
    axes[1].set(xlabel='Within-class high-frequency energy quartile',ylabel='Hard14 validation accuracy (%)',title='Historical runs (unmatched budgets)');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    for split in ['train','validation']:
        axes[2].plot(range(len(LAYERS)),[np.mean([v[split]['accuracy'] for v in layer['results'] if v['layer']==name])*100 for name in LAYERS],marker='o',label=split)
    axes[2].set_xticks(range(len(LAYERS)),['Input','L3','L6','L9','L12 pre-LN','Final'],rotation=30,ha='right')
    axes[2].set(ylabel='Frozen linear accuracy (%)',title='One fixed supervised encoder; two readout seeds');axes[2].legend();axes[2].grid(alpha=.2)
    fig.savefig(r/'diagnosis.png',dpi=200);fig.savefig(r/'diagnosis.pdf');plt.close(fig)
    print(r/'report.md',flush=True)


if __name__=='__main__':main()
