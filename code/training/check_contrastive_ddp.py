"""Compare distributed SupCon gradients with an exact full-batch reference on CPU."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from utils.common import runtime, write_json
from utils.contrastive import supervised_contrastive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    device, rank, world = runtime(args.device, 2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if world != 2:
        raise ValueError('Run with torchrun --nproc_per_node=2')
    records = []
    for labels in [[0, 0, 1, 1, 2, 2], [-1, 0, 0, 1, 1, 1], [-1, -1, -1, 0, 0, 1]]:
        torch.manual_seed(42)
        encoder = torch.nn.Linear(5, 4).to(device)
        reference = deepcopy(encoder)
        model = DistributedDataParallel(encoder, device_ids=[device.index] if device.type=='cuda' else None)
        inputs = torch.randn(6, 5, device=device)
        target, rows = torch.tensor(labels,device=device), torch.arange(6,device=device)
        selected = slice(rank * 3, (rank + 1) * 3)
        loss, _ = supervised_contrastive(model(inputs[selected]), target[selected], rows[selected], .2, gather=True)
        loss.backward()
        exact, _ = supervised_contrastive(reference(inputs), target, rows, .2, gather=False)
        exact.backward()
        error = max(float((a.grad - b.grad).abs().max()) for a, b in zip(encoder.parameters(), reference.parameters()))
        maximum = torch.tensor(error,device=device)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if float(maximum) > 3e-6:
            raise AssertionError(f'Distributed gradient mismatch: {float(maximum)}')
        records.append({'labels': labels, 'maximum_gradient_error': float(maximum)})
    if rank == 0:
        write_json(args.output, {'passed': True, 'world_size': world, 'cases': records})
        print('Distributed SupCon gradients match full-batch reference in all three cases.', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
