"""Training-only Atomic clustering, bounded Emergent BPE, streaming label cache.

--teacher-checkpoint enables the second vocabulary iteration. The teacher is used
only offline and is not needed by the final inference model.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from models.rf_model import RFModel
from utils.common import load_checkpoint, model_bundle, runtime, save_checkpoint, seed_all, write_json
from utils.dataset import PreparedDataset, collate, to_device
from utils.command_config import command_defaults, checkpoint_reference, configured_launch
from utils.structure import StructuralTokenizer, learn_merges
from utils.structure import encode_emergent_chunk
from utils.normalization import observation_scale
from utils.multisource import balanced_source_positions
from utils.structure_sampling import GROUP_NAMES,TARGET_WEIGHTS,validate_weights,token_plan


@torch.inference_mode()
def features(batch, patch_size, teacher, metric='spherical', normalization='rms', percentile=99.):
    if teacher is not None:
        h, valid, _ = teacher.encode(batch['iq'], batch['lengths'])
    else:
        iq = batch['iq'].float()
        valid_points = torch.arange(iq.shape[1], device=iq.device)[None] < batch['lengths'][:, None]
        scale = observation_scale(iq,valid_points,normalization,percentile)
        h = (iq / scale[:, None, None]).reshape(len(iq), -1, patch_size * 2)
        valid = torch.arange(h.shape[1], device=iq.device)[None] * patch_size < batch['lengths'][:, None]
    # Euclidean clustering retains patch energy relative to the signal scale.
    # The legacy spherical path removes each feature vector's length.
    return (F.normalize(h.float(), dim=-1) if metric=='spherical' else h.float()), valid


@torch.inference_mode()
def assign_features(h,centers,metric='spherical'):
    flat=h.reshape(-1,h.shape[-1])
    score=flat@centers.T
    if metric=='euclidean':score=2*score-centers.square().sum(-1)
    return score.argmax(-1).reshape(h.shape[:-1])


def euclidean_cluster(pool,k,iterations,seed,batch_size=16384,n_init=5):
    from sklearn.cluster import MiniBatchKMeans
    if len(pool)<k:raise ValueError('Need at least one feature per cluster')
    estimator=MiniBatchKMeans(n_clusters=k,max_iter=iterations,batch_size=batch_size,
                             n_init=n_init,random_state=seed)
    estimator.fit(pool.numpy())
    print(f'Euclidean MiniBatchKMeans: inertia={estimator.inertia_:.6g}, iterations={estimator.n_iter_}, n_init={n_init}',flush=True)
    return torch.from_numpy(estimator.cluster_centers_.copy()).float()


@torch.inference_mode()
def cluster(pool, k, iterations, device, seed, chunk_size=4096):
    if len(pool) < k:
        raise ValueError(f'Only {len(pool)} feature vectors for {k} clusters; increase pool or decrease atomic-vocab')
    generator = torch.Generator().manual_seed(seed)
    centers = pool[torch.randperm(len(pool), generator=generator)[:k]].to(device)
    for step in range(iterations):
        sums, counts = torch.zeros_like(centers), torch.zeros(k, device=device)
        objective = 0.0
        for chunk in pool.split(chunk_size):
            chunk = chunk.to(device)
            similarity = chunk @ centers.T
            maximum, assignment = similarity.max(1)
            sums.index_add_(0, assignment, chunk)
            counts.index_add_(0, assignment, torch.ones(len(chunk), device=device))
            objective += float((1 - maximum).sum())
        update = F.normalize(sums / counts[:, None].clamp_min(1), dim=-1)
        centers = torch.where((counts > 0)[:, None], update, centers)
        print(f'Clustering iteration {step + 1}/{iterations}: mean cosine distance {objective / len(pool):.6f}', flush=True)
    return centers.cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, help='Grouped codebook settings; CLI values override')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--teacher-checkpoint', type=Path)
    p.add_argument('--atomic-vocab', type=int, default=512)
    p.add_argument('--merges', type=int, default=512)
    p.add_argument('--max-span', type=int, default=5)
    p.add_argument('--max-patches', type=int, default=200000)
    p.add_argument('--bpe-samples', type=int, default=20000)
    p.add_argument('--cluster-iterations', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--gpus', default='')
    p.add_argument('--cluster-chunk-size', type=int, default=4096)
    p.add_argument('--metric',choices=['spherical','euclidean'],default='spherical')
    p.add_argument('--normalization',choices=['rms','amplitude_percentile'],default='rms')
    p.add_argument('--amplitude-percentile',type=float,default=99.)
    p.add_argument('--cluster-batch-size',type=int,default=16384)
    p.add_argument('--cluster-initializations',type=int,default=5)
    p.add_argument('--bpe-tokens',type=int,default=0,help='Token budget overrides bpe-samples when positive')
    p.add_argument('--min-frequency',type=int,default=1,help='Stop BPE if best eligible pair is below this count')
    p.add_argument('--sampling-policy',choices=['legacy','target_token_quota'],default='legacy')
    p.add_argument('--group-weights',type=json.loads,default=TARGET_WEIGHTS)
    p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--label-workers',type=int,default=0)
    command_defaults(p, {'run': {'data': 'data', 'output': 'output', 'teacher_checkpoint': 'teacher_checkpoint'},
                         'atomic': {'vocab_size': 'atomic_vocab', 'max_patches': 'max_patches', 'iterations': 'cluster_iterations', 'chunk_size': 'cluster_chunk_size',
                                    'metric':'metric','normalization':'normalization','amplitude_percentile':'amplitude_percentile',
                                    'batch_size':'cluster_batch_size','initializations':'cluster_initializations'},
                         'emergent': {'merges': 'merges', 'max_span': 'max_span', 'fit_sequences': 'bpe_samples','fit_tokens':'bpe_tokens','min_frequency':'min_frequency'},
                         'sampling':{'policy':'sampling_policy','group_weights':'group_weights'},
                         'runtime': {'batch_size': 'batch_size', 'seed': 'seed', 'device': 'device', 'threads': 'threads', 'gpus': 'gpus','precision':'precision','label_workers':'label_workers'}})
    args = p.parse_args()
    if args.metric not in ['spherical','euclidean'] or args.normalization not in ['rms','amplitude_percentile']:
        p.error('Unknown dictionary distance or normalization')
    configured_launch(args.gpus)
    if args.sampling_policy=='target_token_quota':
        validate_weights(args.group_weights)
        if args.bpe_tokens<=0:p.error('Target-token policy requires an explicit positive emergent.fit_tokens budget')
    if not 0<=args.label_workers<=16:p.error('label_workers must be between 0 and 16')
    if args.teacher_checkpoint:
        args.teacher_checkpoint = checkpoint_reference(args.teacher_checkpoint)
    if args.atomic_vocab < 2 or args.merges < 0 or args.atomic_vocab + args.merges >= 65535 or args.max_span < 1 or args.cluster_iterations < 1 or args.bpe_samples < 1:
        p.error('Invalid vocabulary/merge/span/iteration size')
    if min(args.batch_size, args.max_patches, args.threads, args.cluster_chunk_size) < 1:
        p.error('Batch size, feature budget, threads and chunk size must be positive')
    if not 0<args.amplitude_percentile<=100 or min(args.cluster_batch_size,args.cluster_initializations,args.min_frequency)<1 or args.bpe_tokens<0:
        p.error('Invalid percentile, clustering restarts/batch or BPE token/frequency budget')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Choose a new output directory; preserve earlier vocabulary versions')
    args.output.mkdir(parents=True, exist_ok=True)
    device, _, world = runtime(args.device, args.threads)
    if world != 1:
        raise ValueError('Build a codebook in one process')
    seed_all(args.seed)
    def extract(batch):
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=args.precision=='bf16'):
            return features(to_device(batch,device),patch,teacher,args.metric,args.normalization,args.amplitude_percentile)
    train = PreparedDataset(args.data, 'train')
    info = train.info
    patch = info['patch_size']
    teacher = None
    if args.teacher_checkpoint:
        checkpoint = load_checkpoint(args.teacher_checkpoint)
        if checkpoint['dataset_fingerprint'] != info['fingerprint']:
            raise ValueError('Teacher must have been trained with the same grouped dataset split')
        teacher = RFModel(**model_bundle(checkpoint)).to(device).eval()
        teacher.load_state_dict(checkpoint['model'])
        if (teacher.config.get('normalization','rms')!=args.normalization or
                (args.normalization=='amplitude_percentile' and teacher.config.get('amplitude_percentile',99.)!=args.amplitude_percentile)):
            raise ValueError('Teacher and dictionary normalization must match')
    # Distinct waveforms, with equal feature-pool quotas across task/length buckets.
    _, first = np.unique(train.meta['hash'][train.indices], return_index=True)
    unique_positions = first
    rng = np.random.default_rng(args.seed)
    sampling_report={};fit_rows={}
    buckets = {}
    for position in unique_positions:
        row = train.indices[position]
        key = (int(train.meta['source'][row]) if 'source' in train.meta else 0,
               int(train.meta['task'][row]), int(train.meta['length'][row]))
        buckets.setdefault(key, []).append(int(position))
    if args.sampling_policy=='target_token_quota':
        plan=token_plan(train,unique_positions,args.max_patches,args.group_weights,rng)
        fit_rows['atomic_rows']=np.concatenate([train.indices[item['positions']] for item in plan])
        sampling_report['atomic_tokens_by_group']={name:sum(item['quota'] for item in plan if item['key'][0]==i) for i,name in enumerate(GROUP_NAMES)}
    else:
        plan=[]
        for key,positions in buckets.items():
            source_buckets=sum(other[0]==key[0] for other in buckets)
            quota=max(1,args.max_patches//len({k[0] for k in buckets})//source_buckets)
            needed_rows=min(len(positions),(quota+key[2]//patch-1)//(key[2]//patch))
            plan.append({'positions':rng.choice(positions,needed_rows,replace=False),'quota':quota,'key':key})
            # Legacy RNG order included feature-position draws between buckets.
            plan[-1]['feature_indices']=rng.choice(needed_rows*(key[2]//patch),min(quota,needed_rows*(key[2]//patch)),replace=False)
    pools = []
    for item in plan:
        chosen=item['positions'];quota=item['quota']
        loader = DataLoader(Subset(train, chosen.tolist()), batch_size=args.batch_size, collate_fn=collate)
        vectors = []
        for batch in loader:
            h, valid = extract(batch)
            vectors.append(h[valid].cpu())
        vectors = torch.cat(vectors)
        indices = item.get('feature_indices')
        if indices is None:indices=rng.choice(len(vectors),min(quota,len(vectors)),replace=False)
        pools.append(vectors[indices])
    pool = torch.cat(pools)
    pool_size = len(pool)
    centers = (euclidean_cluster(pool,args.atomic_vocab,args.cluster_iterations,args.seed,args.cluster_batch_size,args.cluster_initializations)
               if args.metric=='euclidean' else cluster(pool, args.atomic_vocab, args.cluster_iterations, device, args.seed, args.cluster_chunk_size))
    del pool, pools
    centers_device = centers.to(device)
    maximum = 1024 // patch
    atomic = np.lib.format.open_memmap(args.output / 'atomic_labels.npy', mode='w+', dtype=np.uint16, shape=(info['rows'], maximum))
    atomic[:] = 65535
    all_data = PreparedDataset(args.data, 'train')
    all_data.indices = np.arange(info['rows'])
    loader = DataLoader(all_data, batch_size=args.batch_size, collate_fn=collate)
    for step,batch in enumerate(loader):
        h, valid = extract(batch)
        assignment = assign_features(h,centers_device,args.metric).cpu().numpy()
        for i, row in enumerate(batch['rows']):
            length = int(batch['lengths'][i]) // patch
            atomic[row, :length] = assignment[i, :length]
        if step%500==0:print(f'Atomic assignment batch {step+1:,}/{len(loader):,}',flush=True)
    atomic.flush()
    # Only TRAIN sequences fit BPE rules. Validation receives fixed-vocabulary labels.
    if args.sampling_policy=='target_token_quota':
        plan=token_plan(train,unique_positions,args.bpe_tokens,args.group_weights,rng)
        sequences=[];fitting_rows=[];crop_starts=[];crop_lengths=[]
        for item in plan:
            remaining=item['quota']
            for position in item['positions']:
                row=int(train.indices[position]);count=min(item['tokens'],remaining)
                start=int(rng.integers(item['tokens']-count+1)) if count<item['tokens'] else 0
                sequences.append(np.asarray(atomic[row,start:start+count],dtype=np.int32))
                fitting_rows.append(row);crop_starts.append(start);crop_lengths.append(count);remaining-=count
            assert remaining==0
        fitting_rows=np.asarray(fitting_rows,dtype=np.int64)
        fit_rows.update(bpe_rows=fitting_rows,bpe_crop_starts=np.asarray(crop_starts),bpe_crop_lengths=np.asarray(crop_lengths))
        sampling_report['bpe_tokens_by_group']={name:sum(item['quota'] for item in plan if item['key'][0]==i) for i,name in enumerate(GROUP_NAMES)}
        sampling_report['weights']=args.group_weights
        assert (train.meta['split'][fit_rows['atomic_rows']]==0).all() and (train.meta['split'][fitting_rows]==0).all()
        np.savez_compressed(args.output/'fit_rows.npz',**fit_rows)
        write_json(args.output/'sampling_report.json',sampling_report)
    elif args.bpe_tokens:
        shuffled=rng.permutation(unique_positions)
        cumulative=np.cumsum(train.meta['length'][train.indices[shuffled]]//patch)
        fitting_positions=shuffled[:min(len(shuffled),int(np.searchsorted(cumulative,args.bpe_tokens))+1)]
    else:
        fitting_positions = balanced_source_positions(train,unique_positions,args.bpe_samples,rng)
    if args.sampling_policy!='target_token_quota':
        fitting_rows = train.indices[fitting_positions]
        sequences = [np.asarray(atomic[row, :int(train.meta['length'][row]) // patch], dtype=np.int32) for row in fitting_rows]
        if args.bpe_tokens and sum(map(len,sequences))>args.bpe_tokens:
            sequences[-1]=sequences[-1][:(args.bpe_tokens-sum(map(len,sequences[:-1])))]
    print(f'Fitting BPE on {len(sequences):,} training sequences',flush=True)
    merges, symbol_lengths = learn_merges(sequences, args.atomic_vocab, args.merges, args.max_span,args.min_frequency)
    tokenizer = StructuralTokenizer(merges)
    emergent = np.lib.format.open_memmap(args.output / 'emergent_labels.npy', mode='w+', dtype=np.uint16, shape=atomic.shape)
    emergent[:] = 65535
    if args.label_workers:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        np.save(args.output/'lengths.npy',train.meta['length'])
        emergent.flush()
        jobs=[(str(args.output/'atomic_labels.npy'),str(args.output/'emergent_labels.npy'),str(args.output/'lengths.npy'),
               patch,merges,start,min(info['rows'],start+50000)) for start in range(0,info['rows'],50000)]
        with ProcessPoolExecutor(max_workers=args.label_workers,mp_context=multiprocessing.get_context('spawn')) as workers:
            for end in workers.map(encode_emergent_chunk,jobs):print(f'Emergent assignment {end:,}/{info["rows"]:,}',flush=True)
    else:
        for row, length in enumerate(train.meta['length']):
            count = int(length) // patch
            emergent[row, :count] = tokenizer.encode(atomic[row, :count])
            if row%100000==0:print(f'Emergent assignment {row+1:,}/{info["rows"]:,}',flush=True)
    emergent.flush()
    config = dict(dataset_fingerprint=info['fingerprint'], patch_size=patch, atomic_vocab=args.atomic_vocab,
        emergent_vocab=args.atomic_vocab + len(merges), requested_merges=args.merges, merges=merges,
        max_span=args.max_span, symbol_lengths=symbol_lengths, feature_kind='encoder' if teacher else 'raw_iq',
        teacher_checkpoint=str(args.teacher_checkpoint.resolve()) if teacher else None,
        seed=args.seed, cluster_iterations=args.cluster_iterations, fitted_training_sequences=len(sequences),
        feature_pool_rows=pool_size, requested_max_patches=args.max_patches, requested_bpe_samples=args.bpe_samples,
        cluster_chunk_size=args.cluster_chunk_size, batch_size=args.batch_size,
        metric=args.metric,normalization=args.normalization,amplitude_percentile=args.amplitude_percentile,
        cluster_batch_size=args.cluster_batch_size,cluster_initializations=args.cluster_initializations,
        requested_bpe_tokens=args.bpe_tokens,fitted_training_tokens=sum(map(len,sequences)),min_frequency=args.min_frequency,
        sampling_policy=args.sampling_policy,group_weights=args.group_weights if args.sampling_policy!='legacy' else {},
        teacher_feature_precision=args.precision,sampling_report=sampling_report,
        fitting_sequences_by_source={str(source):int((train.meta['source'][fitting_rows]==source).sum())
           for source in np.unique(train.meta['source'])} if 'source' in train.meta else {},
        note='C01 adaptation with explicit sampling/normalization; no claim of bit-exact author-code reproduction.')
    # The artifact contains no live teacher dependency for training or inference.
    save_checkpoint(args.output / 'codebook.pt', {'centers': centers, 'config': config})
    write_json(args.output / 'codebook.json', config)
    print({key: config[key] for key in ['feature_kind', 'atomic_vocab', 'emergent_vocab', 'fitted_training_sequences']})


if __name__ == '__main__':
    main()
