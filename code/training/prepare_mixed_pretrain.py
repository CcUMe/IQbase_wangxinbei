"""Preserve the C01 split; add hash-grouped RadioML records as unlabeled IQ.

Stores metadata and read-only source references, without copying either IQ array.
This tool does not use a GPU and does not change the original datasets.
"""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
from utils.common import read_config_json,read_json,write_json

RADIO_LABELS=['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK','16APSK','32APSK',
 '64APSK','128APSK','16QAM','32QAM','64QAM','128QAM','256QAM','AM-SSB-WC','AM-SSB-SC',
 'AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK']


def prepare(config):
    c01=Path(config['run']['c01']).resolve();radio=Path(config['run']['radioml']).resolve()
    output=Path(config['run']['output']).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Choose a new empty output directory')
    info=read_json(c01/'dataset.json')
    with np.load(c01/'metadata.npz',allow_pickle=False) as loaded:
        original={k:loaded[k] for k in loaded.files}
    if info['patch_size']!=8:raise ValueError('This preparation requires the P8 C01 view')
    wanted=config['selection']['classes'];min_snr=config['selection']['minimum_snr_db']
    if not wanted or len(set(wanted))!=len(wanted) or not set(wanted)<=set(RADIO_LABELS):raise ValueError('Invalid RadioML classes')
    fraction=config['split']['validation_fraction'];seed=config['split']['seed']
    if not 0<fraction<1:raise ValueError('Invalid validation fraction')
    salt=int.from_bytes(hashlib.blake2b(str(seed).encode(),digest_size=8).digest(),'little')
    cut=int(fraction*2**64)
    source_rows=[];hashes=[];class_ids=[];snrs=[]
    # Metadata is scanned in chunks: no assumption about the file's class ordering.
    with h5py.File(radio,'r') as h:
        total=len(h['X'])
        if h['X'].shape[1:]!=(1024,2) or h['Y'].shape!=(total,24) or h['Z'].shape!=(total,1):raise ValueError('Bad RadioML shape')
        chosen_classes=np.asarray([RADIO_LABELS.index(x) for x in wanted])
        for start in range(0,total,16384):
            y=h['Y'][start:start+16384];z=h['Z'][start:start+16384,0]
            labels=y.argmax(1)
            if not (((y==0)|(y==1)).all() and (y.sum(1)==1).all()):raise ValueError('Malformed RadioML one-hot metadata')
            chosen=np.flatnonzero(np.isin(labels,chosen_classes)&(z>=min_snr))
            for offset in range(0,len(chosen),256):
                local=chosen[offset:offset+256];indices=start+local
                iq=h['X'][indices].astype('<f4',copy=False)
                if not np.isfinite(iq).all():raise ValueError('Nonfinite RadioML IQ')
                hashes.extend(hashlib.blake2b(struct.pack('<I',1024)+a.tobytes(),digest_size=16).digest() for a in iq)
                source_rows.extend(indices.tolist());class_ids.extend(labels[local].tolist());snrs.extend(z[local].tolist())
            if start%262144==0:print(f'RadioML scanned {min(start+16384,total):,}/{total:,}; selected {len(hashes):,}',flush=True)
    if not hashes:raise ValueError('No external records selected')
    external_hash=np.asarray(hashes,dtype='V16')
    external_split=np.fromiter(((int.from_bytes(h[:8],'little')^salt)<cut for h in hashes),np.int8)
    # An exact waveform shared with C01 must inherit its existing split.
    c01_hash,first=np.unique(original['hash'],return_index=True)
    positions=np.searchsorted(c01_hash,external_hash)
    match=(positions<len(c01_hash))
    match[match] &= c01_hash[positions[match]]==external_hash[match]
    external_split[match]=original['split'][first[positions[match]]]
    n=len(original['split']);m=len(external_split)
    external={
      'sample_id':np.asarray([f'rml2018:{r:07d}' for r in source_rows]),
      'hash':external_hash,'split':external_split,'length':np.full(m,1024,dtype=np.int32),
      'task':np.full(m,3,dtype=np.int8),'modulation':np.full(m,-1,dtype=np.int64),
      'protocol':np.full(m,-1,dtype=np.int64),'targets':np.zeros((m,7),dtype=np.float32),
      'target_valid':np.zeros((m,7),dtype=bool)}
    if set(original)!=set(external):raise ValueError('Unsupported C01 metadata fields: '+str(set(original)^set(external)))
    metadata={k:np.concatenate([original[k],external[k]]) for k in original}
    metadata.update(source=np.concatenate([np.zeros(n,np.int8),np.ones(m,np.int8)]),
      source_row=np.concatenate([np.arange(n,dtype=np.int64),np.asarray(source_rows,dtype=np.int64)]),
      external_modulation=np.concatenate([np.full(n,-1,np.int16),np.asarray(class_ids,np.int16)]),
      snr_db=np.concatenate([np.full(n,np.nan,np.float32),np.asarray(snrs,np.float32)]))
    train_hash=np.unique(metadata['hash'][metadata['split']==0]);valid_hash=np.unique(metadata['hash'][metadata['split']==1])
    if np.intersect1d(train_hash,valid_hash).size:raise ValueError('Cross-source exact-waveform split leakage')
    digest=hashlib.sha256(metadata['hash'].tobytes()+metadata['split'].tobytes()+str(info['patch_size']).encode()).hexdigest()
    result=deepcopy(info)
    result.update(format_version=2,storage='multi_source',fingerprint=digest,rows=n+m,
      train_rows=int((metadata['split']==0).sum()),validation_rows=int((metadata['split']==1).sum()),
      sources=[{'name':'c01','kind':'npy','path':str((c01/'iq.npy').resolve()),'fingerprint':info['fingerprint']},
               {'name':'radioml','kind':'hdf5','path':str(radio),'file_bytes':radio.stat().st_size}],
      data_provenance={'target_c01_fingerprint':info['fingerprint'],'preparation':config,
                       'external_iq_transformed':False,'c01_split_preserved':True},
      external_class_names=RADIO_LABELS,cross_split_hash_overlap=0)
    result['task_counts']=deepcopy(info['task_counts'])
    for part,name in [(0,'train'),(1,'validation')]:
        result['task_counts'][name]['unlabeled_radioml']=int((external_split==part).sum())
    counts={label:{str(s):{'train':int(((np.asarray(class_ids)==RADIO_LABELS.index(label))&(np.asarray(snrs)==s)&(external_split==0)).sum()),
                         'validation':int(((np.asarray(class_ids)==RADIO_LABELS.index(label))&(np.asarray(snrs)==s)&(external_split==1)).sum())}
                  for s in np.unique(snrs).tolist()} for label in wanted}
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'metadata.npz',**metadata)
    write_json(output/'split_report.json',{'c01_train':info['train_rows'],'c01_validation':info['validation_rows'],
       'radioml_train':int((external_split==0).sum()),'radioml_validation':int((external_split==1).sum()),
       'selected_external':m,'cross_source_exact_matches':int(match.sum()),'cross_split_hash_overlap':0,
       'grouping':'IQ-content hash; no capture/session IDs available for RadioML', 'by_external_class_snr':counts})
    write_json(output/'dataset.json',result)
    print(f'Prepared mixed dataset: {output}; C01={n:,}, RadioML={m:,}; raw IQ not copied',flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    prepare(read_config_json(parser.parse_args().config))
