"""Audit completed hypothesis experiments and report all outcomes, including negatives."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from training.hypothesis_probe import group_metrics, PERTURBATIONS
from utils.common import write_json


def read(path):return json.loads(path.read_text())


def average_probe(results, condition='clean', key='accuracy'):
    return float(np.mean([(v['clean'] if condition=='clean' else v['perturbations'][condition])[key] for v in results]))


def paired_interval(root, candidate, control, repeats=2000):
    a,b=root/candidate,root/control
    labels=np.load(a/'validation_labels.npy')
    assert np.array_equal(labels,np.load(b/'validation_labels.npy'))
    assert np.array_equal(np.load(a/'validation_rows.npy'),np.load(b/'validation_rows.npy'))
    differences=np.mean([((np.load(a/f'clean_seed{seed}_predictions.npy')==labels).astype(float)-
        (np.load(b/f'clean_seed{seed}_predictions.npy')==labels)) for seed in [2026,2027]],axis=0)
    generator=np.random.default_rng(20261111)
    samples=[]
    for _ in range(repeats//50):
        index=generator.integers(0,len(labels),size=(50,len(labels)))
        samples.extend(differences[index].mean(1).tolist())
    return {'candidate':candidate,'control':control,'accuracy_difference':float(differences.mean()),
        'paired_validation_bootstrap_95_interval':np.quantile(samples,[.025,.975]).tolist(),
        'repeats':repeats,'note':'Conditional on these fitted encoders/readout seeds; not encoder-training uncertainty or correction for repeated validation exploration.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=args.output.resolve()
    status=read(r/'status.json')
    if not status['complete']:raise ValueError('Experiments incomplete; do not issue a final report')
    baseline=Path('artifacts/diagnostics/20261111_110649_079619/joint')
    reference=read(baseline/'protocol.json')
    models={'control':read(baseline/'metrics.json')}
    arms=['no_scale','no_global','multiscale','finetune_reconstruction','finetune_structure']
    for name in arms:
        protocol=read(r/name/'protocol.json')
        assert protocol['reference_initial_state_sha256']==reference['initial_state_sha256'],name
        assert protocol['modulation_batch_stream_sha256']==reference['modulation_batch_stream_sha256'],name
        models[name]=read(r/name/'metrics.json')
        assert models[name]['rows']==118182,name
    raw=read(r/'pretrain_reconstruction/protocol.json');structure=read(r/'pretrain_structure/protocol.json')
    for key in ['initial_state_sha256','initial_backbone_sha256','sample_stream_sha256','mask_stream_sha256','steps','batch_size']:
        assert raw[key]==structure[key],key
    vocab=read(r/'atomic_codebook/protocol.json')
    meta=np.load('artifacts/data_full/metadata.npz',allow_pickle=False)
    fitted=np.load(r/'atomic_codebook/fitted_rows.npy')
    assert bool((meta['split'][fitted]==0).all())
    probes={t['name']:read(r/t['name']/'results.json') for t in status['tasks'] if t['name'].endswith('_probe')}
    for name in probes:
        assert read(r/name/'protocol.json')['backbone_and_heads_bitwise_unchanged'],name
    interventions=read(r/'large_interventions/summary.json')
    fixed=read(r/'fixed_features/summary.json') if (r/'fixed_features/summary.json').exists() else None
    rms_only=read(r/'rms_only_diagnostic.json') if (r/'rms_only_diagnostic.json').exists() else None
    target_stability=read(r/'target_invariance_diagnostic.json') if (r/'target_invariance_diagnostic.json').exists() else None
    assert read(r/'large_interventions/protocol.json')['backbone_and_heads_bitwise_unchanged']
    original_hashes=read(r/'default_weight_hashes.json')
    assert all(hashlib.sha256((Path('weights')/name).read_bytes()).hexdigest()==digest for name,digest in original_hashes.items())
    comparisons=[paired_interval(r,name,'baseline_probe') for name in ['no_scale_probe','no_global_probe','multiscale_probe','finetune_reconstruction_probe','finetune_structure_probe']]
    comparisons += [paired_interval(r,'pretrain_structure_probe','pretrain_reconstruction_probe'),
                    paired_interval(r,'pretrain_structure_probe','random_probe'),
                    paired_interval(r,'finetune_structure_probe','finetune_reconstruction_probe')]
    write_json(r/'paired_probe_comparisons.json',comparisons)
    with (r/'per_class_metrics.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['experiment','condition','task','seed','label','support','precision','recall','f1'])
        def emit(experiment,condition,task,seed,metric):
            for i,label in enumerate(metric['labels']):writer.writerow([experiment,condition,task,seed,label,
                metric['class_support'][i],metric['precision'][i],metric['recall'][i],metric['f1'][i]])
        for name,metrics in models.items():
            for task,metric in metrics['classification'].items():emit(name,'clean',task,'encoder_seed2026',metric)
        for name,metric in interventions.items():emit('large_interventions',name,'modulation','trained',metric)
        for name,results in probes.items():
            for result in results:
                emit(name,'clean','modulation',result['seed'],result['clean'])
                for condition,metric in result['perturbations'].items():emit(name,condition,'modulation',result['seed'],metric)
        if fixed:
            for result in fixed['results']:emit('fixed_'+result['features']+'_'+result['head'],'clean','modulation',result['seed'],result['validation'])
    summary={'training_results':models,'frozen_probes':probes,'large_interventions':interventions,
        'fixed_feature_references':fixed,'rms_only_diagnostic':rms_only,'target_invariance_diagnostic':target_stability,
        'paired_probe_comparisons':comparisons,'checks':{'common_initialization_matches':True,'downstream_sample_stream_matches':True,
            'pretraining_inputs_masks_initialization_match':True,'codebook_train_only':True,'default_weights_unchanged':True},
        'scope':'17M one encoder seed, two CPU linear readout seeds; 96M interventions are inference-only; fixed budgets, exploratory validation.'}
    write_json(r/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(18,5),layout='constrained')
    for name,path in [('control',baseline),('no_scale',r/'no_scale'),('no_global',r/'no_global'),('multiscale',r/'multiscale')]:
        history=read(path/'history.json')
        axes[0].plot([v['modulation_updates'] for v in history],
            [v['metrics']['classification']['modulation']['accuracy']*100 for v in history],marker='o',label=name)
    axes[0].set(xlabel='Modulation updates (batch 256)',ylabel='Validation accuracy (%)',title='Matched from-scratch ablations')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    cases=['random_probe','pretrain_reconstruction_probe','pretrain_structure_probe',
           'baseline_probe','finetune_reconstruction_probe','finetune_structure_probe']
    labels=['Random','Waveform','Structure','Scratch FT','Waveform FT','Structure FT']
    positions=np.array([0,1,2,4,5,6])
    axes[1].bar(positions,[average_probe(probes[n])*100 for n in cases],color=['gray','#5577aa','#cc7733']*2)
    for x,name in zip(positions,cases):
        axes[1].scatter([x-.08,x+.08],[v['clean']['accuracy']*100 for v in probes[name]],c='black',s=12,zorder=3)
    axes[1].set_xticks(positions,labels,rotation=35,ha='right')
    axes[1].set(ylabel='Frozen linear accuracy (%)',title='Fixed 20-epoch linear probes')
    axes[1].grid(axis='y',alpha=.2)
    for name,label in [('baseline_probe','Scratch FT'),('finetune_reconstruction_probe','Waveform FT'),('finetune_structure_probe','Structure FT')]:
        conditions=['clean','gain_double','phase_90','cfo_plus','cfo_minus','shift_16']
        axes[2].plot(range(len(conditions)),[average_probe(probes[name],condition)*100 for condition in conditions],marker='o',label=label)
    axes[2].set_xticks(range(6),['Clean','Gain x2','Phase 90','CFO +','CFO -','Shift 16'],rotation=35,ha='right')
    axes[2].set(ylabel='Frozen linear accuracy (%)',title='Paired modulation stress tests')
    axes[2].legend(fontsize=8);axes[2].grid(alpha=.2)
    figure.suptitle('17M diagnostics: one encoder seed; two fixed CPU linear-readout seeds',fontsize=12)
    figure.savefig(r/'comparison.png',dpi=200)
    figure.savefig(r/'comparison.pdf')
    plt.close(figure)
    lines=['# 三个问题的验证结果','','所有任务均完成；主配置及默认推理权重保持原样。[预定方案](plan.md)记录假设、预算与边界。',
        '中断前的部分训练保留在前一时间戳目录，本报告只使用完整重跑及已核对哈希的历史控制组。','',
        '## 一、观测信息依赖：当前大模型的直接干预','',
        '| 干预 | 调制 Accuracy | 前14类 Accuracy | Macro-F1 | 预测翻转比例 |','|---|---:|---:|---:|---:|']
    for name,metric in interventions.items():lines.append(f'| {name} | {metric["accuracy"]:.4%} | {metric["hard14"]["accuracy"]:.4%} | {metric["macro_f1"]:.4%} | {metric["prediction_flip_fraction"]:.4%} |')
    lines += ['', '| 大模型干预 | 前14类 | CSS/GFSK/pi4DQPSK/FM | 五类雷达 |', '|---|---:|---:|---:|']
    for name,metric in interventions.items():
        cm=np.asarray(metric['confusion_matrix'])
        values=[np.diag(cm)[a:b].sum()/cm[a:b].sum() for a,b in [(0,14),(14,18),(18,23)]]
        lines.append('| '+name+' | '+' | '.join(f'{value:.4%}' for value in values)+' |')
    clean_cm=np.asarray(interventions['clean']['confusion_matrix'])
    rotated_cm=np.asarray(interventions['phase_90']['confusion_matrix'])
    clean_radar=np.diag(clean_cm)[18:23].sum()/clean_cm[18:23].sum()
    rotated_radar=np.diag(rotated_cm)[18:23].sum()/rotated_cm[18:23].sum()
    lines += ['', f'相位90度扰动的主要损失来自雷达：{clean_radar:.4%} → {rotated_radar:.4%}；前14类则为 {interventions["clean"]["hard14"]["accuracy"]:.4%} → {interventions["phase_90"]["hard14"]["accuracy"]:.4%}。因此，整体相位敏感性不能直接当作前14类原分布识别率偏低的主因。雷达相位鲁棒性与数字调制细粒度泛化应分开定位。']
    lines += ['',
        'drop_scale仅关闭显式原始RMS尺度回注；drop_global仅关闭全局频谱回注，局部频谱分支仍在。直接干预使已有编码器接收不同特征分布，精度下降说明依赖，不足以证明该依赖造成过拟合。gain/phase是成对观测扰动；cfo单位为cycles/sample，没有声称物理Hz或真实SNR。干预使用相同排序和组批，组内比较优先于与历史日志末位差异。','',
        '## 二、从头消融与多尺度前端：完整下游评估','',
        '| 组 | 调制 Accuracy | 前14类 Accuracy | 调制 Macro-F1 | 调制 Weighted-F1 | 协议 Accuracy | 协议 Macro-F1 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,metrics in models.items():
        m=metrics['classification']['modulation'];q=metrics['classification']['protocol'];g=group_metrics(m)
        lines.append(f'| {name} | {m["accuracy"]:.4%} | {g["accuracy"]:.4%} | {m["macro_f1"]:.4%} | {m["weighted_f1"]:.4%} | {q["accuracy"]:.4%} | {q["macro_f1"]:.4%} |')
    lines += ['',
        '每组17M、同一共同参数初始化、调制批次哈希一致，4800次调制更新/14400次联合更新，batch256，逐cycle LR匹配。no_scale/no_global从头关闭各自回注；multiscale保留两个回注并加入零初始化多尺度残差，不改变Token数。三组可分别回答指定路径及指定前端候选是否有收益，不能代表所有架构。预训练微调两组额外经历1200次无标签更新。','',
        '| 组 | ' + ' | '.join(models['control']['parameters']) + ' |',
        '|---|'+'---:|'*7]
    for name,metrics in models.items():lines.append('| '+name+' | '+' | '.join(f'{v["rmse"]:.7g}' for v in metrics['parameters'].values())+' |')
    lines += ['', '上表是七项参数原单位RMSE。正式比赛需同时考虑协议和参数任务，不能只按调制增益替换模型。','',
        '## 三、预训练目标是否转化为可迁移表征','',
        f'词表只由训练侧 {vocab["fitted_records"]} 条不重复记录的 {vocab["fitted_patches"]} 个Patch拟合，使用 {vocab["used_codes"]}/{vocab["vocab_size"]} 个结构码。没有使用类别或参数数值。两个预训练组的全部初始化、采样流、掩码流已核对一致；同样建立两个辅助头、计算量尽量匹配，只改变激活的目标。','',
        f'固定掩码验证目标损失：重建 {raw["final_fixed_mask_losses"]["validation"]:.6f}，结构CE {structure["final_fixed_mask_losses"]["validation"]:.6f}。这两个数的单位不同，不能直接比较大小判断谁学得更好。','',
        '| 编码器 | 冻结Linear训练 Acc均值 | 冻结Linear验证 Acc均值 | 前14类 Acc均值 | Macro-F1均值 | 两个读出种子验证Acc |',
        '|---|---:|---:|---:|---:|---|']
    for name,results in probes.items():
        train=np.mean([v['training']['accuracy'] for v in results]);hard=np.mean([v['clean']['hard14']['accuracy'] for v in results])
        seeds=' / '.join(f'{v["clean"]["accuracy"]:.4%}' for v in results)
        lines.append(f'| {name} | {train:.4%} | {average_probe(results):.4%} | {hard:.4%} | {average_probe(results,key="macro_f1"):.4%} | {seeds} |')
    lines += ['',
        '所有线性读出固定CPU、相同训练侧特征标准化、20epochs、batch1024、AdamW1e-3/WD0.01、cosine floor0.05；末轮结果，不按验证选轮。random_probe、pretrain_*_probe检验微调前的表征；finetune_*_probe检验同预算下游训练后的表征。读出种子重复不等于重复训练编码器。','',
        '| 编码器 | clean | gain×0.5 | gain×2 | phase90° | CFO+0.001 | CFO−0.001 | shift16 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,results in probes.items():lines.append('| '+name+' | '+' | '.join(f'{average_probe(results,condition):.4%}' for condition in PERTURBATIONS)+' |')
    if fixed:
        lines += ['', '## 补充参照：固定特征包含多少识别信息','',
            '| 特征 | 分类头 | 验证Accuracy均值 | 前14类Accuracy均值 |', '|---|---|---:|---:|']
        for feature in ['statistics','spectrum']:
            for head in ['linear','mlp']:
                selected=[v for v in fixed['results'] if v['features']==feature and v['head']==head]
                assert len(selected)==2
                overall=np.mean([v['validation']['accuracy'] for v in selected])
                hard=np.mean([v['validation']['hard14']['accuracy'] for v in selected])
                lines.append(f'| {feature} | {head} | {overall:.4%} | {hard:.4%} |')
        lines += ['', 'statistics是32项全窗I/Q统计；spectrum是与原全局分支一致的1024点log功率谱。固定特征无需训练编码器，读出固定20轮、两个种子；MLP隐藏128。它们是信息可分性的参照，不是参数匹配的架构消融，低分不能证明原始I/Q的可达上限。']
    if rms_only:
        a=rms_only['results']['23'];b=rms_only['results']['14']
        lines += ['', f'仅log-RMS的固定256分位直方图分类器，23类验证Accuracy={a["validation"]["accuracy"]:.4%}；仅在前14类拟合/评估时={b["validation"]["accuracy"]:.4%}。它说明尺度有部分标签相关信息，不能说明深模型实际使用了多少，更不能证明存在答案泄漏。']
    if target_stability:
        train_agreement=target_stability['splits']['train']['agreement']
        val_agreement=target_stability['splits']['validation']['agreement']
        lines += ['', f'在训练/验证各1024条真实波形上额外检查目标稳定性：幅度×0.7、相位+0.3712rad后，离散结构标签一致率分别为 {train_agreement:.6%} / {val_agreement:.6%}。该检查验证目标定义及量化数值稳定性，不代表编码器自动获得相同不变性。']
    lines += ['',
        '扰动只用于调制评估；循环平移后没有拿原时延标签评分。保持较高原始余弦相似度本身也不能证明类别结构稳定，因此同时报告固定线性头的准确率和预测翻转比例，完整值见summary.json。','',
        '## 成对验证不确定性','',
        '| 冻结读出比较 | Accuracy差值（百分点） | 成对bootstrap 95%区间（百分点） |',
        '|---|---:|---|']
    for v in comparisons:
        low,high=v['paired_validation_bootstrap_95_interval']
        lines.append(f'| {v["candidate"]} − {v["control"]} | {v["accuracy_difference"]*100:+.4f} | [{low*100:+.4f}, {high*100:+.4f}] |')
    lines += ['',
        '区间只描述这些固定模型在当前验证记录上的差值不确定性，未包含重新训练编码器的随机性，也没有修正多次使用验证集探索的影响；不能当作隐藏测试显著性保证。','',
        '## 可得结论与不能外推的部分','']
    control=models['control']['classification']['modulation']['accuracy']
    for name,label in [('no_scale','去掉尺度回注'),('no_global','去掉全局频谱'),('multiscale','多尺度前端'),('finetune_reconstruction','波形预训练后微调'),('finetune_structure','结构预训练后微调')]:
        delta=(models[name]['classification']['modulation']['accuracy']-control)*100
        lines.append(f'- {label}：本次完整下游预算相对控制组变化 {delta:+.4f} 个百分点。')
    delta=(average_probe(probes['pretrain_structure_probe'])-average_probe(probes['pretrain_reconstruction_probe']))*100
    final_delta=(models['finetune_structure']['classification']['modulation']['accuracy']-models['finetune_reconstruction']['classification']['modulation']['accuracy'])*100
    lines += [f'- 结构相对波形目标：微调前冻结线性读出变化 {delta:+.4f} 个百分点；相同下游微调后直接调制Accuracy变化 {final_delta:+.4f} 个百分点。应联合看这些结果和扰动保持能力，不能仅凭预训练loss下结论。','',
        '这是17M、单编码器种子、有限预算的筛选，加上96M已有权重的推理干预；尚未完成96M从头多种子消融。结构目标是局部统计原子预测，没有ESU、迭代词表或真实物理时间对齐，因此正负结果均不能直接判定参考论文的完整方法。','',
        '本轮没有自动更新main.json或weights/。如候选获得明显、稳定的细粒度识别收益，再做97M规模验证，比较协议和参数代价后决定迁移。','',
        '## 文件与执行核验','',
        '- [任务状态、全部命令及退出码](status.json)',
        '- [完整指标与逐条件混淆矩阵](summary.json)',
        '- [对照图 PNG](comparison.png)、[PDF](comparison.pdf)',
        '- [全部逐类Precision/Recall/F1](per_class_metrics.csv)',
        '- [成对比较](paired_probe_comparisons.json)',
        '- [恢复一致性执行测试](smoke/status.json)',
        '- [计划与适用边界](plan.md)','',
        '各组train.log保留可读训练总览、逐epoch进度快照和分类表；events.jsonl保存结构化事件，下游对照检查点包含可恢复的优化器和调度器状态。预训练检查点保存权重，预训练中断需重跑该短阶段。','']
    (r/'report.md').write_text('\n'.join(lines))
    print(r/'report.md')


if __name__=='__main__':main()
