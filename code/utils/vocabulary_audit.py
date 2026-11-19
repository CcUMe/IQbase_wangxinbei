"""Read-only quota, split, label-usage and compositionality checks."""
from pathlib import Path
import numpy as np
from utils.common import read_json,write_json
from utils.dataset import PreparedDataset
from utils.structure_sampling import GROUP_NAMES,group_ids


def audit_vocabulary(directory,data_directory):
    directory=Path(directory);book=read_json(directory/'codebook.json');data=PreparedDataset(data_directory)
    meta=data.meta;groups=data.pretrain_groups;counts=meta['length']//book['patch_size']
    spans=np.asarray(book['symbol_lengths']);hist={};columns=np.arange(1024//book['patch_size'])
    for group in range(len(GROUP_NAMES)):
        for split in [0,1]:hist[(group,split)]={kind:np.zeros(book[kind+'_vocab'],np.int64) for kind in ['atomic','emergent']}
    for kind in ['atomic','emergent']:
        labels=np.load(directory/(kind+'_labels.npy'),mmap_mode='r')
        for start in range(0,len(counts),8192):
            end=min(len(counts),start+8192);valid=columns[None]<counts[start:end,None]
            for group,split in hist:
                chosen=(groups[start:end]==group)&(meta['split'][start:end]==split)
                if not chosen.any():continue
                token=np.asarray(labels[start:end][chosen])[valid[chosen]]
                if token.max(initial=0)>=book[kind+'_vocab']:raise ValueError('Invalid structure labels')
                hist[(group,split)][kind]+=np.bincount(token,minlength=book[kind+'_vocab'])
    usage={}
    for (group,split),values in hist.items():
        result={}
        for kind,count in values.items():
            q=count/max(1,count.sum());present=q>0
            stats={'used_ids':int(present.sum()),'largest_id_fraction':float(q.max()),
                   'perplexity':float(np.exp(-(q[present]*np.log(q[present])).sum())),'positions':int(count.sum())}
            if kind=='emergent':
                units=(count/spans).sum()
                stats.update(merged_position_coverage=float(q[spans>=2].sum()),length3plus_position_coverage=float(q[spans>=3].sum()),
                             average_expanded_unit_length=float(count.sum()/max(1,units)))
            result[kind]=stats
        usage[GROUP_NAMES[group]+'_'+('train' if split==0 else 'validation')]=result
    if usage['c01_first14_train']['atomic']['used_ids']<=1:raise ValueError('C01 first14 atomic vocabulary collapsed to at most one identity')
    quota={}
    if book.get('sampling_policy')=='target_token_quota':
        with np.load(directory/'fit_rows.npz',allow_pickle=False) as fitted:
            assert (meta['split'][fitted['atomic_rows']]==0).all()
            rows=fitted['bpe_rows'];assert (meta['split'][rows]==0).all()
            g=group_ids(meta,rows);lengths=fitted['bpe_crop_lengths']
            quota={name:int(lengths[g==i].sum()) for i,name in enumerate(GROUP_NAMES)}
        expected=read_json(directory/'sampling_report.json')['bpe_tokens_by_group']
        if quota!=expected:raise ValueError('BPE token quota differs from recorded fit rows')
    report={'passed':True,'teacher':book.get('teacher_checkpoint'),'sampling_policy':book.get('sampling_policy'),
            'bpe_tokens_by_group_verified':quota,'usage':usage,
            'limits':['No downstream accuracy guarantee.','Usage and long-unit coverage alone do not prove transferable structure.']}
    write_json(directory/'quality_report.json',report)
    return report
