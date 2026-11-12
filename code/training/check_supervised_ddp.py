"""Exact supervised DDP gradient check with unequal/missing labels and accumulation."""
import argparse
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.common import runtime,write_json
from utils.engine import supervised_loss


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=torch.nn.Linear(5,8)
        self.heads=torch.nn.ModuleDict({'modulation':torch.nn.Linear(8,3),'protocol':torch.nn.Linear(8,2),
                                      'regression':torch.nn.Linear(8,7)})
        self.grid=torch.nn.Linear(8,2)
    def forward(self,x):
        h=self.backbone(x).tanh()
        return {'modulation':self.heads['modulation'](h),'protocol':self.heads['protocol'](h),
                'parameters':self.heads['regression'](h),'grid_logits':{'0':self.grid(h)}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();device,rank,world=runtime(args.device,2)
    if world!=2:raise ValueError('Use torchrun with two processes')
    torch.backends.cuda.matmul.allow_tf32=False
    records=[]
    for accumulation in [1,2]:
        for empty_rank in [False,True]:
            torch.manual_seed(42);encoder=Toy().to(device);reference=deepcopy(encoder)
            model=DDP(encoder,device_ids=[device.index] if device.type=='cuda' else None)
            for micro in range(accumulation):
                torch.manual_seed(404+micro);x=torch.randn(6,5,device=device)
                modulation=torch.tensor([-1,-1,-1,0,0,1] if empty_rank else [0,-1,-1,1,0,1],device=device)
                protocol=torch.tensor([-1,-1,-1,0,1,0] if empty_rank else [-1,0,-1,-1,1,0],device=device)
                targets=torch.zeros(6,7,device=device);valid=torch.zeros(6,7,device=device,dtype=torch.bool)
                selected=[3,4,5] if empty_rank else [1,3,4]
                valid[selected,0]=True;targets[selected,0]=torch.tensor([-.2,.4,-.2],device=device)
                valid[5,3]=True;targets[5,3]=.6
                batch={'iq':x,'modulation':modulation,'protocol':protocol,'targets':targets,'target_valid':valid}
                local={k:v[rank*3:(rank+1)*3] for k,v in batch.items()}
                config={'parameter_weights':[1.,2.,1.,.5,1.,1.,1.], 'grid_loss_weight':.1}
                with model.no_sync() if micro+1<accumulation else nullcontext():
                    loss,_=supervised_loss(model(local['iq']),local,{'mean':[0.]*7,'std':[1.]*7},[[-.2,.4]]+[[]]*6,config)
                    (loss/accumulation).backward()
                exact,_=supervised_loss(reference(x),batch,{'mean':[0.]*7,'std':[1.]*7},[[-.2,.4]]+[[]]*6,config,distributed=False)
                (exact/accumulation).backward()
            error=max(float((a.grad-b.grad).abs().max()) for a,b in zip(encoder.parameters(),reference.parameters()))
            if error>3e-6:raise AssertionError(f'Gradient mismatch: {error}')
            records.append({'accumulation':accumulation,'rank_zero_without_labels':empty_rank,'maximum_gradient_error':error})
    if rank==0:
        write_json(args.output,{'passed':True,'world':world,'device':str(device),'cases':records})
        print('All supervised gradients match the full-batch reference.',flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
