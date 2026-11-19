"""Source-aware pretraining sampling and explicit C01 backbone-transfer contracts."""
from types import SimpleNamespace
import numpy as np
from torch.utils.data import Sampler
from utils.dataset import BalancedBatchSampler


class MultiSourceBatchSampler(Sampler):
    """One source/length per step; all DDP ranks share the schedule, not draws.

    RadioML records do not require fabricated C01 classification/parameter labels.
    C01 retains the existing balanced task/length sampler inside its quota.
    """
    def __init__(self,dataset,batch_size,steps,settings,seed=2026,rank=0):
        self.dataset=dataset;self.batch_size=batch_size;self.steps=steps
        self.settings=settings;self.seed=seed;self.rank=rank;self.epoch=0
        if dataset.info.get('storage')!='multi_source':raise ValueError('Source sampling requires a prepared mixed dataset')
        if [s['name'] for s in dataset.info['sources']]!=['c01','radioml']:raise ValueError('Unexpected source definitions')
        source=dataset.meta['source'][dataset.indices]
        self.c01_positions=np.flatnonzero(source==0)
        external=np.flatnonzero(source==1)
        if not len(self.c01_positions) or not len(external):raise ValueError('Both sources need training records')
        c01=SimpleNamespace(indices=dataset.indices[self.c01_positions],meta=dataset.meta)
        self.c01=BalancedBatchSampler(c01,batch_size,steps,seed,rank,'balanced')
        classes=dataset.meta['external_modulation'][dataset.indices[external]]
        snrs=dataset.meta['snr_db'][dataset.indices[external]]
        lengths=dataset.meta['length'][dataset.indices[external]]
        self.buckets={}
        for c,s,n in sorted(set(zip(classes.tolist(),snrs.tolist(),lengths.tolist()))):
            self.buckets[(c,s,n)]=external[(classes==c)&(snrs==s)&(lengths==n)]

    def __len__(self):return self.steps

    def set_epoch(self,epoch):
        self.epoch=epoch;self.c01.set_epoch(epoch)

    def __iter__(self):
        schedule=np.random.default_rng(self.seed+self.epoch)
        draws=np.random.default_rng(self.seed+self.epoch*100003+self.rank*1009)
        probabilities=np.asarray([self.settings['probabilities'][k] for k in ['c01','radioml']],dtype=float)
        probabilities/=probabilities.sum()
        sources=schedule.choice(2,size=self.steps,p=probabilities)
        c01=iter(self.c01);keys=list(self.buckets)
        for source in sources:
            if source==0:
                yield self.c01_positions[np.asarray(next(c01))].tolist()
            else:
                key=keys[int(schedule.integers(len(keys)))]
                yield draws.choice(self.buckets[key],self.batch_size,replace=True).tolist()


def balanced_source_positions(dataset,positions,count,rng):
    """Equal source quotas for BPE fitting; no repeated records or held-out fit."""
    if 'source' not in dataset.meta:
        return rng.choice(positions,min(count,len(positions)),replace=False)
    source=dataset.meta['source'][dataset.indices[positions]]
    buckets=[positions[source==s] for s in np.unique(source)]
    count=min(count,len(positions));quotas=np.zeros(len(buckets),dtype=int)
    remaining=count
    while remaining:
        eligible=[i for i,b in enumerate(buckets) if quotas[i]<len(b)]
        increment=max(1,remaining//len(eligible))
        for i in eligible:
            amount=min(increment,len(buckets[i])-quotas[i],remaining)
            quotas[i]+=amount;remaining-=amount
    selected=np.concatenate([rng.choice(b,int(q),replace=False) for b,q in zip(buckets,quotas)])
    return rng.permutation(selected)


def validate_backbone_data_contract(checkpoint,info):
    """Allow mixed-pretrained -> original C01 only with preserved split provenance."""
    same=checkpoint['dataset_fingerprint']==info['fingerprint']
    provenance=checkpoint.get('data_provenance',{})
    target=(provenance.get('target_c01_fingerprint')==info['fingerprint'] and
            provenance.get('c01_split_preserved') is True and checkpoint.get('stage')=='pretrain')
    if not (same or target):raise ValueError('Backbone initialization data contract mismatch: dataset_fingerprint')
    for key in ['schema','target_scaler','parameter_grids']:
        if checkpoint[key]!=info[key]:raise ValueError('Backbone initialization data contract mismatch: '+key)
