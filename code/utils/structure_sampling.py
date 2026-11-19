"""Auditable token quotas for structural vocabulary discovery and pretraining."""
import numpy as np
from torch.utils.data import Sampler

GROUP_NAMES=['c01_first14','c01_other_modulation','c01_protocol','c01_parameter','radioml']
TARGET_WEIGHTS=dict(zip(GROUP_NAMES,[.4,.1,.15,.2,.15]))


def validate_weights(weights):
    if not isinstance(weights,dict) or set(weights)!=set(GROUP_NAMES):raise ValueError('Expected five target-group weights')
    values=np.asarray([weights[name] for name in GROUP_NAMES],float)
    if not np.isfinite(values).all() or (values<0).any() or values.sum()<=0:raise ValueError('Invalid target-group weights')
    return values/values.sum()


def group_ids(meta,rows):
    source=meta['source'][rows];mod=meta['modulation'][rows];protocol=meta['protocol'][rows]
    result=np.full(len(rows),-1,np.int8)
    result[(source==0)&(mod>=0)&(mod<14)]=0
    result[(source==0)&(mod>=14)]=1
    result[(source==0)&(result<0)&(protocol>=0)]=2
    result[(source==0)&(result<0)&meta['target_valid'][rows].any(1)]=3
    result[source==1]=4
    if (result<0).any():raise ValueError('Unassigned pretraining target group')
    return result


def strata(dataset,positions):
    rows=dataset.indices[positions];meta=dataset.meta;groups=group_ids(meta,rows)
    label=meta['modulation'][rows].copy();label[groups==2]=meta['protocol'][rows[groups==2]]
    label[groups==3]=(meta['target_valid'][rows[groups==3]]*(1<<np.arange(7))).sum(1)
    label[groups==4]=meta['external_modulation'][rows[groups==4]]
    snr=np.full(len(rows),-1000,np.int32);snr[groups==4]=meta['snr_db'][rows[groups==4]].astype(np.int32)
    keys,inverse=np.unique(np.stack([groups,label,snr,meta['length'][rows]],1),axis=0,return_inverse=True)
    return {tuple(int(v) for v in key):positions[inverse==i] for i,key in enumerate(keys)}


def allocate(total,weights):
    values=np.asarray(weights,float);values=values/values.sum();raw=total*values
    quota=np.floor(raw).astype(int)
    for i in np.argsort(-(raw-quota),kind='stable')[:total-int(quota.sum())]:quota[i]+=1
    return quota


def token_plan(dataset,positions,total,weights,rng):
    """Exact integer token quotas, never oversample records within a stratum."""
    probabilities=validate_weights(weights);buckets=strata(dataset,positions)
    group_quota=allocate(total,probabilities);result=[]
    for group,amount in enumerate(group_quota):
        if not amount:continue
        keys=[k for k in buckets if k[0]==group]
        if not keys:raise ValueError('Missing fit group: '+GROUP_NAMES[group])
        for key,quota in zip(keys,allocate(int(amount),np.ones(len(keys)))):
            if not quota:continue
            tokens=key[-1]//dataset.info['patch_size'];needed=(int(quota)+tokens-1)//tokens
            if needed>len(buckets[key]):raise ValueError('Not enough unique training rows for token quota: '+str(key))
            chosen=rng.choice(buckets[key],needed,replace=False)
            result.append({'key':key,'positions':chosen,'tokens':tokens,'quota':int(quota)})
    assert sum(p['quota'] for p in result)==total
    return result


class TargetGroupBatchSampler(Sampler):
    """Expected token shares match weights despite different record lengths.

    The schedule is rank-independent, while within-stratum draws differ by rank.
    Class/SNR metadata only controls sampling; it is not an encoder input.
    """
    def __init__(self,dataset,batch_size,steps,weights,seed=2026,rank=0):
        self.dataset=dataset;self.batch_size=batch_size;self.steps=steps;self.seed=seed;self.rank=rank;self.epoch=0
        w=validate_weights(weights);self.buckets=strata(dataset,np.arange(len(dataset)));self.keys=[];probabilities=[]
        for key in self.buckets:
            group=key[0]
            if not w[group]:continue
            count=sum(k[0]==group for k in self.buckets)
            self.keys.append(key);probabilities.append(w[group]/count/(key[-1]//dataset.info['patch_size']))
        for group in np.flatnonzero(w):
            if not any(key[0]==group for key in self.keys):raise ValueError('Missing sampling group: '+GROUP_NAMES[group])
        self.probabilities=np.asarray(probabilities);self.probabilities/=self.probabilities.sum()

    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return self.steps
    def __iter__(self):
        schedule=np.random.default_rng(self.seed+self.epoch)
        draws=np.random.default_rng(self.seed+self.epoch*100003+self.rank*1009)
        for index in schedule.choice(len(self.keys),size=self.steps,p=self.probabilities):
            yield draws.choice(self.buckets[self.keys[index]],self.batch_size,replace=True).tolist()
