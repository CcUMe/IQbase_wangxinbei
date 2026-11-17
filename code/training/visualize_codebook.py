"""Read-only plots of a completed raw-IQ vocabulary and its cached labels.

Full-data counts; training-only independent samples for PCA/quantization.
Does not refit a vocabulary, change training, or require a GPU.
"""
import argparse
import base64
import csv
from datetime import datetime
import html
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

from training.build_codebook import features,assign_features
from utils.dataset import PreparedDataset,collate
from utils.common import read_json,write_json


def distribution(counts):
    counts=np.asarray(counts,dtype=np.float64)
    p=counts/max(1.,counts.sum());positive=p[p>0]
    entropy=float(-(positive*np.log(positive)).sum())
    ordered=np.sort(p)[::-1]
    return {'tokens':int(counts.sum()),'used':int((counts>0).sum()),'empty':int((counts==0).sum()),
            'effective_vocabulary':float(np.exp(entropy)),'top1_share':float(ordered[0]),
            'top10_share':float(ordered[:10].sum())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codebook',type=Path,default=ROOT/'artifacts/structure_research_p8_euclidean/codebook_v1')
    parser.add_argument('--data',type=Path,default=ROOT/'artifacts/data_full_p8')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--seed',type=int,default=20261116)
    parser.add_argument('--rows-per-bucket',type=int,default=96)
    parser.add_argument('--patches-per-row',type=int,default=24)
    args=parser.parse_args()
    torch.set_num_threads(2)
    book=args.codebook.resolve();data=args.data.resolve()
    config=read_json(book/'codebook.json');info=read_json(data/'dataset.json')
    if config['feature_kind']!='raw_iq':parser.error('This geometry plot requires a raw-IQ dictionary, not encoder features')
    if config['dataset_fingerprint']!=info['fingerprint']:parser.error('Vocabulary and data fingerprint mismatch')
    if min(args.rows_per_bucket,args.patches_per_row)<1:parser.error('Sampling sizes must be positive')
    patch=config['patch_size'];k=config['atomic_vocab'];esu=config['emergent_vocab']
    checkpoint=torch.load(book/'codebook.pt',map_location='cpu',weights_only=True)
    if checkpoint['config']!=config:raise ValueError('Vocabulary tensor/JSON metadata mismatch')
    centers=checkpoint['centers'].numpy()
    assert centers.shape==(k,2*patch)
    atomic=np.load(book/'atomic_labels.npy',mmap_mode='r')
    emergent=np.load(book/'emergent_labels.npy',mmap_mode='r')
    train=PreparedDataset(data,'train');meta=train.meta
    assert atomic.shape==emergent.shape==(info['rows'],1024//patch)
    output=args.output or ROOT/'artifacts/codebook_visualizations'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('euclidean_p8_%Y%m%d_%H%M%S')
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'figure.facecolor':'white','axes.spines.top':False,
                         'axes.spines.right':False,'font.size':10,'savefig.dpi':160})
    names=info['schema']['classification_labels']['modulation']
    counts=np.zeros(k,np.int64);split_counts=np.zeros((2,k),np.int64)
    task_counts=np.zeros((3,k),np.int64);class_counts=np.zeros((len(names),k),np.int64)
    ecounts=np.zeros(esu,np.int64)
    pos=np.arange(atomic.shape[1])[None]
    print('1/4 全量原子/组合标签统计（排除padding）',flush=True)
    for start in range(0,len(atomic),8192):
        end=min(start+8192,len(atomic));a=np.asarray(atomic[start:end]);e=np.asarray(emergent[start:end])
        valid=pos<meta['length'][start:end,None]//patch
        if np.any(a[valid]>=k) or np.any(e[valid]>=esu) or np.any(a[~valid]!=65535) or np.any(e[~valid]!=65535):
            raise ValueError('Invalid/incomplete vocabulary label cache')
        counts+=np.bincount(a[valid],minlength=k)
        ecounts+=np.bincount(e[valid],minlength=esu)
        for split in range(2):
            selected=valid&(meta['split'][start:end,None]==split)
            split_counts[split]+=np.bincount(a[selected],minlength=k)
        for task in range(3):
            selected=valid&(meta['task'][start:end,None]==task)&(meta['split'][start:end,None]==0)
            task_counts[task]+=np.bincount(a[selected],minlength=k)
        labels=meta['modulation'][start:end]
        selected=valid&(labels[:,None]>=0)&(meta['split'][start:end,None]==0)
        joint=np.broadcast_to(labels[:,None],a.shape)[selected].astype(np.int64)*k+a[selected]
        class_counts+=np.bincount(joint,minlength=len(names)*k).reshape(len(names),k)
    assert counts.sum()==int((meta['length']//patch).sum())
    print('2/4 训练集独立抽样、标签回查及PCA',flush=True)
    rng=np.random.default_rng(args.seed);chosen=[]
    for task in np.unique(meta['task']):
        for length in np.unique(meta['length']):
            available=np.flatnonzero((meta['task'][train.indices]==task)&(meta['length'][train.indices]==length))
            if len(available):chosen.extend(rng.choice(available,min(len(available),args.rows_per_bucket),replace=False).tolist())
    vectors=[];sample_labels=[]
    for batch in DataLoader(Subset(train,chosen),batch_size=64,collate_fn=collate):
        h,valid=features(batch,patch,None,config['metric'],config['normalization'],config['amplitude_percentile'])
        predicted=assign_features(h,checkpoint['centers'],config['metric']).numpy()
        for i,row in enumerate(batch['rows']):
            n=int(batch['lengths'][i])//patch
            if not np.array_equal(predicted[i,:n],atomic[row,:n]):raise ValueError('Cached labels differ from recomputed nearest centers')
            keep=rng.choice(n,min(n,args.patches_per_row),replace=False)
            vectors.append(h[i,keep].numpy());sample_labels.append(np.array(atomic[row,keep],dtype=np.int64))
    x=np.concatenate(vectors);labels=np.concatenate(sample_labels)
    mean=x.mean(0);_,singular,vt=np.linalg.svd(x-mean,full_matrices=False)
    explained=singular[:2]**2/(singular**2).sum()
    projected=(x-mean)@vt[:2].T;projected_centers=(centers-mean)@vt[:2].T
    center_iq=centers.reshape(k,patch,2)
    center_rms=np.sqrt((center_iq**2).sum(-1).mean(-1))
    local_rms=np.sqrt((x.reshape(-1,patch,2)**2).sum(-1).mean(-1))
    errors=((x-centers[labels])**2).sum(-1)
    probability=counts/counts.sum();order=np.argsort(-counts)
    stats={'codebook':str(book),'data':str(data),'metric':config['metric'],'normalization':config['normalization'],
           'patch_size':patch,'fit_patch_count':config['feature_pool_rows'],'full_data':distribution(counts),
           'train':distribution(split_counts[0]),'validation':distribution(split_counts[1]),
           'emergent_position_weighted':distribution(ecounts),'center_rms_min':float(center_rms.min()),
           'center_rms_max':float(center_rms.max()),'most_used_cluster':int(order[0]),
           'sampled_training_records':len(chosen),'sampled_training_patches':len(x),
           'sample_label_mismatches':0,'pca_explained_variance_ratio':explained.tolist(),
           'sample_squared_quantization_error_mean':float(errors.mean()),
           'sample_squared_quantization_error_median':float(np.median(errors)),
           'sample_squared_quantization_error_p95':float(np.quantile(errors,.95)),
           'sampling_note':'PCA is fit on independent training-only task/length-bucket samples; not the original fitting pool.',
           'interpretation':'Two-dimensional overlap/unsupervised label association is not a downstream accuracy bound.'}
    write_json(output/'statistics.json',stats)
    with (output/'atomic_clusters.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['cluster_id','all_patch_count','share','train_count','validation_count','centroid_rms'])
        for i in range(k):writer.writerow([i,int(counts[i]),float(probability[i]),int(split_counts[0,i]),int(split_counts[1,i]),float(center_rms[i])])
    np.savez_compressed(output/'analysis_arrays.npz',centers=centers,atomic_counts=counts,split_counts=split_counts,
                        task_counts=task_counts,class_counts=class_counts,emergent_position_counts=ecounts,
                        sampled_features=x,sampled_clusters=labels,pca_components=vt[:2],pca_mean=mean)

    def save(fig,name):
        fig.savefig(output/(name+'.png'),bbox_inches='tight')
        fig.savefig(output/(name+'.pdf'),bbox_inches='tight')
        plt.close(fig)

    print('3/4 绘制全量占用、几何投影、I/Q中心及类别关联',flush=True)
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    fig.suptitle('P8 欧氏原子词表：全量使用情况（有效patch，排除padding）',fontsize=17)
    axes[0,0].bar(np.arange(k),counts,color='#257b99',width=1)
    axes[0,0].set(xlabel='原子词项 / cluster ID',ylabel='patch数量（对数轴）',yscale='log',title=f'已使用 {np.count_nonzero(counts)}/{k} 个簇')
    axes[0,1].plot(np.arange(1,k+1),np.cumsum(probability[order])*100,lw=2,color='#c76530')
    axes[0,1].plot([1,k],[100/k,100],ls='--',color='gray',label='均匀分布参考')
    axes[0,1].legend();axes[0,1].set(xlabel='按频次排序的前N个簇',ylabel='累计patch占比（%）',ylim=(0,101),title=f"Top1={100*probability[order[0]]:.2f}% · 有效词表={stats['full_data']['effective_vocabulary']:.1f}")
    for split,label,color in [(0,'训练集','#257b99'),(1,'验证集','#d98239')]:
        axes[1,0].plot(np.arange(k),split_counts[split]/split_counts[split].sum(),label=label,color=color,alpha=.8,lw=1)
    axes[1,0].set(xlabel='cluster ID',ylabel='各自集合内patch占比',title='训练/验证的词项频率');axes[1,0].legend()
    axes[1,1].scatter(center_rms,probability*100,s=18,c='#257b99',alpha=.7)
    axes[1,1].annotate(f'ID {order[0]}',(center_rms[order[0]],probability[order[0]]*100),xytext=(8,6),textcoords='offset points')
    axes[1,1].set(xlabel='聚类中心RMS（相对于信号级RMS）',ylabel='全量patch占比（%）',title='低能量中心是否占用过多patch？')
    save(fig,'01_usage')

    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    fig.suptitle(f'原始16维patch特征的PCA投影 · 前两维解释 {100*explained.sum():.1f}% 方差',fontsize=16)
    points=axes[0].scatter(projected[:,0],projected[:,1],c=labels,cmap='turbo',s=3,alpha=.45,rasterized=True,vmin=0,vmax=k-1)
    fig.colorbar(points,ax=axes[0],label='cluster ID（编号不代表距离）')
    axes[0].set(title=f'{len(x):,}个独立抽样训练patch（按簇编号着色）')
    points=axes[1].scatter(projected_centers[:,0],projected_centers[:,1],c=center_rms,cmap='viridis',s=15+600*np.sqrt(probability),alpha=.9,edgecolors='white',linewidths=.3)
    fig.colorbar(points,ax=axes[1],label='中心RMS')
    for idx in order[:5]:axes[1].annotate(str(idx),(projected_centers[idx,0],projected_centers[idx,1]),fontsize=8)
    axes[1].set(title='512个聚类中心（点大小随使用率增加）')
    for ax in axes:ax.set(xlabel=f'PC1（{100*explained[0]:.1f}%）',ylabel=f'PC2（{100*explained[1]:.1f}%）');ax.set_aspect('equal',adjustable='datalim')
    save(fig,'02_geometry')

    examples=[]
    for group in [order[:8],order[k//2:k//2+4],order[-4:]]:
        examples.extend(map(int,group))
    fig,axes=plt.subplots(4,4,figsize=(14,10),layout='constrained')
    fig.suptitle('原子聚类中心的I/Q形状：高频8个 / 中位4个 / 低频4个',fontsize=16)
    for ax,idx in zip(axes.flat,examples):
        ax.plot(np.arange(patch),center_iq[idx,:,0],'-o',ms=3,label='I',color='#247ba0')
        ax.plot(np.arange(patch),center_iq[idx,:,1],'-o',ms=3,label='Q',color='#d98239')
        ax.axhline(0,lw=.6,color='gray',alpha=.6)
        ax.set_title(f'ID {idx} · 占比{100*probability[idx]:.3f}% · RMS {center_rms[idx]:.3f}',fontsize=9)
        ax.set_xticks(range(patch));ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8);fig.supxlabel('patch内采样点（无采样率，不解释为物理时间）')
    fig.supylabel('归一化I/Q值；各子图纵轴独立')
    save(fig,'03_iq_centers')

    selected=np.argsort(-class_counts.sum(0))[:64]
    selected=selected[np.argsort(projected_centers[selected,0])]
    smooth=.5
    conditional=(class_counts+smooth)/(class_counts.sum(1,keepdims=True)+smooth*k)
    baseline=(class_counts.sum(0)+smooth)/(class_counts.sum()+smooth*k)
    lift=np.log2(conditional[:,selected]/baseline[selected])
    fig,ax=plt.subplots(figsize=(17,9),layout='constrained')
    heat=ax.imshow(lift,aspect='auto',cmap='RdBu_r',vmin=-3,vmax=3)
    ax.set_yticks(range(len(names)),names);ax.set_xticks(range(len(selected)),selected,rotation=90,fontsize=8)
    ax.set(xlabel='调制标注训练记录中最常用的64个簇（按PC1排列）',ylabel='调制/波形类别',
           title='类别与局部结构词项的关联：红色=相对更常用，蓝色=相对更少用\nlog2[类别内词频 / 全部调制标注训练记录词频]；仅用于诊断，不参与构词')
    fig.colorbar(heat,ax=ax,label='log2倍数（显示范围截到±3）')
    save(fig,'04_class_association')

    spans=np.asarray(config['symbol_lengths'])
    span_counts=np.bincount(spans,weights=ecounts,minlength=config['max_span']+1)
    stats['emergent_span_position_shares']={str(i):float(span_counts[i]/ecounts.sum()) for i in range(1,len(span_counts))}
    write_json(output/'statistics.json',stats)
    captions=[('01_usage','全量簇占用与训练/验证分布'),('02_geometry','独立抽样patch及聚类中心的PCA投影'),
              ('03_iq_centers','代表性聚类中心的I/Q形状'),('04_class_association','调制类别与局部结构的关联')]
    description=(f"全量{counts.sum():,}个有效patch；使用{np.count_nonzero(counts)}/{k}个原子簇；"
                 f"Top1占比{100*probability[order[0]]:.2f}%；有效词表{stats['full_data']['effective_vocabulary']:.1f}。")
    document=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>P8欧氏词表可视化</title>',
              '<style>body{font:16px sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;background:#f5f7fa;color:#223}article{background:white;padding:22px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}p{line-height:1.7}</style>',
              '<h1>P8欧氏词表可视化</h1><p>'+html.escape(description)+'</p>',
              '<p>原始I/Q → 信号级RMS → P8局部特征 → 欧氏聚类。PCA只保留部分方差；簇与类别不要求一一对应，不能由图推断识别准确率上限。</p>']
    for filename,title in captions:
        encoded=base64.b64encode((output/(filename+'.png')).read_bytes()).decode()
        document.append(f'<article><h2>{html.escape(title)}</h2><img src="data:image/png;base64,{encoded}"></article>')
    document.append('</html>');(output/'index.html').write_text('\n'.join(document))
    (output/'report.md').write_text('# P8欧氏词表可视化\n\n'+description+'\n\n'+
        '\n\n'.join(f'![{title}]({filename}.png)' for filename,title in captions)+
        '\n\nPCA使用训练集独立抽样，不是原聚类拟合样本的完整复现；所有占用统计排除padding。类别关联只使用训练标签作诊断，不参与聚类拟合，也不代表下游准确率。\n')
    print('4/4 完成：'+str(output),flush=True)
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
