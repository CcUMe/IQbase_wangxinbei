"""Training-only Atomic clustering, bounded Emergent BPE, streaming label cache.

--teacher-checkpoint enables the second vocabulary iteration. The teacher is used
only offline and is not needed by the final inference model.
"""
import argparse
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


@torch.inference_mode()
def features(batch, patch_size, teacher):
    if teacher is not None:
        h, valid, _ = teacher.encode(batch['iq'], batch['lengths'])
    else:
        iq = batch['iq'].float()
        valid_points = torch.arange(iq.shape[1], device=iq.device)[None] < batch['lengths'][:, None]
        scale = ((iq.square().sum(-1) * valid_points).sum(1) / batch['lengths']).sqrt().clamp_min(1e-6)
        h = (iq / scale[:, None, None]).reshape(len(iq), -1, patch_size * 2)
        valid = torch.arange(h.shape[1], device=iq.device)[None] * patch_size < batch['lengths'][:, None]
    return F.normalize(h.float(), dim=-1), valid


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
    command_defaults(p, {'run': {'data': 'data', 'output': 'output', 'teacher_checkpoint': 'teacher_checkpoint'},
                         'atomic': {'vocab_size': 'atomic_vocab', 'max_patches': 'max_patches', 'iterations': 'cluster_iterations', 'chunk_size': 'cluster_chunk_size'},
                         'emergent': {'merges': 'merges', 'max_span': 'max_span', 'fit_sequences': 'bpe_samples'},
                         'runtime': {'batch_size': 'batch_size', 'seed': 'seed', 'device': 'device', 'threads': 'threads', 'gpus': 'gpus'}})
    args = p.parse_args()
    configured_launch(args.gpus)
    if args.teacher_checkpoint:
        args.teacher_checkpoint = checkpoint_reference(args.teacher_checkpoint)
    if args.atomic_vocab < 2 or args.merges < 0 or args.atomic_vocab + args.merges >= 65535 or args.max_span < 1 or args.cluster_iterations < 1 or args.bpe_samples < 1:
        p.error('Invalid vocabulary/merge/span/iteration size')
    if min(args.batch_size, args.max_patches, args.threads, args.cluster_chunk_size) < 1:
        p.error('Batch size, feature budget, threads and chunk size must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Choose a new output directory; preserve earlier vocabulary versions')
    args.output.mkdir(parents=True, exist_ok=True)
    device, _, world = runtime(args.device, args.threads)
    if world != 1:
        raise ValueError('Build a codebook in one process')
    seed_all(args.seed)
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
    # Distinct waveforms, with equal feature-pool quotas across task/length buckets.
    _, first = np.unique(train.meta['hash'][train.indices], return_index=True)
    unique_positions = first
    rng = np.random.default_rng(args.seed)
    buckets = {}
    for position in unique_positions:
        row = train.indices[position]
        key = (int(train.meta['task'][row]), int(train.meta['length'][row]))
        buckets.setdefault(key, []).append(int(position))
    quota = max(1, args.max_patches // len(buckets))
    pools = []
    for key, positions in buckets.items():
        needed_rows = min(len(positions), (quota + key[1] // patch - 1) // (key[1] // patch))
        chosen = rng.choice(positions, needed_rows, replace=False)
        loader = DataLoader(Subset(train, chosen.tolist()), batch_size=args.batch_size, collate_fn=collate)
        vectors = []
        for batch in loader:
            h, valid = features(to_device(batch, device), patch, teacher)
            vectors.append(h[valid].cpu())
        vectors = torch.cat(vectors)
        indices = rng.choice(len(vectors), min(quota, len(vectors)), replace=False)
        pools.append(vectors[indices])
    pool = torch.cat(pools)
    pool_size = len(pool)
    centers = cluster(pool, args.atomic_vocab, args.cluster_iterations, device, args.seed, args.cluster_chunk_size)
    del pool, pools
    centers_device = centers.to(device)
    maximum = 1024 // patch
    atomic = np.lib.format.open_memmap(args.output / 'atomic_labels.npy', mode='w+', dtype=np.uint16, shape=(info['rows'], maximum))
    atomic[:] = 65535
    all_data = PreparedDataset(args.data, 'train')
    all_data.indices = np.arange(info['rows'])
    loader = DataLoader(all_data, batch_size=args.batch_size, collate_fn=collate)
    for batch in loader:
        h, valid = features(to_device(batch, device), patch, teacher)
        assignment = (h.reshape(-1, h.shape[-1]) @ centers_device.T).argmax(-1).reshape(h.shape[:2]).cpu().numpy()
        for i, row in enumerate(batch['rows']):
            length = int(batch['lengths'][i]) // patch
            atomic[row, :length] = assignment[i, :length]
    atomic.flush()
    # Only TRAIN sequences fit BPE rules. Validation receives fixed-vocabulary labels.
    fitting_positions = rng.choice(unique_positions, min(args.bpe_samples, len(unique_positions)), replace=False)
    fitting_rows = train.indices[fitting_positions]
    sequences = [np.asarray(atomic[row, :int(train.meta['length'][row]) // patch], dtype=np.int32) for row in fitting_rows]
    merges, symbol_lengths = learn_merges(sequences, args.atomic_vocab, args.merges, args.max_span)
    tokenizer = StructuralTokenizer(merges)
    emergent = np.lib.format.open_memmap(args.output / 'emergent_labels.npy', mode='w+', dtype=np.uint16, shape=atomic.shape)
    emergent[:] = 65535
    for row, length in enumerate(train.meta['length']):
        count = int(length) // patch
        emergent[row, :count] = tokenizer.encode(atomic[row, :count])
    emergent.flush()
    config = dict(dataset_fingerprint=info['fingerprint'], patch_size=patch, atomic_vocab=args.atomic_vocab,
        emergent_vocab=args.atomic_vocab + len(merges), requested_merges=args.merges, merges=merges,
        max_span=args.max_span, symbol_lengths=symbol_lengths, feature_kind='encoder' if teacher else 'raw_iq',
        teacher_checkpoint=str(args.teacher_checkpoint.resolve()) if teacher else None,
        seed=args.seed, cluster_iterations=args.cluster_iterations, fitted_training_sequences=len(sequences),
        feature_pool_rows=pool_size, requested_max_patches=args.max_patches, requested_bpe_samples=args.bpe_samples,
        cluster_chunk_size=args.cluster_chunk_size, batch_size=args.batch_size,
        note='Project adaptation; sampled spherical clustering and bounded BPE, not a claimed reproduction of undisclosed author code.')
    # The artifact contains no live teacher dependency for training or inference.
    save_checkpoint(args.output / 'codebook.pt', {'centers': centers, 'config': config})
    write_json(args.output / 'codebook.json', config)
    print({key: config[key] for key in ['feature_kind', 'atomic_vocab', 'emergent_vocab', 'fitted_training_sequences']})


if __name__ == '__main__':
    main()
