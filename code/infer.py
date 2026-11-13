"""Unified C01 Parquet -> JSONL entry point. No labels or network required."""
import argparse
import json
from pathlib import Path
import time
import tempfile
import shutil

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from models.rf_model import RFModel
from utils.common import TARGETS, load_safetensors, read_json, runtime, write_json
from utils.engine import autocast
from utils.command_config import command_defaults, configured_launch
from utils.inference_output import merge_prediction_shards
from utils.process_lifecycle import managed_entrypoint


class Predictor:
    def __init__(self, weights, device='auto', precision='fp32', spectral_backend=None, threads=4):
        self.device, self.rank, self.world = runtime(device, threads)
        self.config = read_json(Path(weights) / 'config.json')
        bundle = {key: self.config[key] for key in ['model_config', 'schema', 'target_scaler', 'parameter_grids']}
        bundle['model_config'] = dict(bundle['model_config'])
        if spectral_backend:
            bundle['model_config']['spectral_backend'] = spectral_backend
        self.model = RFModel(**bundle).to(self.device).eval()
        self.model.load_state_dict(load_safetensors(Path(weights) / 'model.safetensors'))
        self.precision = precision
        self.mean = np.asarray(bundle['target_scaler']['mean'])
        self.std = np.asarray(bundle['target_scaler']['std'])

    @torch.inference_mode()
    def predict(self, records):
        result = [None] * len(records)
        buckets = {}
        for index, record in enumerate(records):
            if record['task'] not in ['modulation', 'protocol', 'parameter']:
                raise ValueError(f"Unknown task: {record['task']}")
            iq = np.asarray(record['iq'], dtype=np.float32)
            if iq.shape not in [(512, 2), (1024, 2)] or not np.isfinite(iq).all():
                raise ValueError(f"Invalid IQ shape/value for sample {record['sample_id']}")
            buckets.setdefault(len(iq), []).append((index, iq))
        for length, rows in buckets.items():
            x = torch.from_numpy(np.stack([iq for _, iq in rows])).to(self.device)
            lengths = torch.full((len(rows),), length, device=self.device, dtype=torch.long)
            with autocast(self.device, self.precision):
                output = self.model(x, lengths)
            classes = {task: output[task].argmax(-1).cpu().tolist() for task in ['modulation', 'protocol']}
            parameters = output['parameters'].float().cpu().numpy() * self.std + self.mean
            if not np.isfinite(parameters).all():
                raise FloatingPointError('Nonfinite model predictions')
            for offset, (index, _) in enumerate(rows):
                record = records[index]
                task = record['task']
                prediction = ({name: float(parameters[offset, j]) for j, name in enumerate(TARGETS)} if task == 'parameter'
                              else self.config['schema']['classification_labels'][task][classes[task][offset]])
                result[index] = {'task': task, 'sample_id': record['sample_id'], 'prediction': prediction}
        return result


@managed_entrypoint('推理')
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, help='Optional inference runtime settings; distinct from weights/config.json')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--weights', type=Path, default=Path(__file__).resolve().parents[1] / 'weights')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--precision', choices=['fp32', 'bf16', 'fp16'], default='fp32')
    p.add_argument('--spectral-backend', choices=['fft', 'matmul'])
    p.add_argument('--log', type=Path)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--gpus', default='')
    p.add_argument('--processes', type=int, choices=[1,2], default=1)
    command_defaults(p, {'run': {'input': 'input', 'output': 'output', 'weights': 'weights', 'log': 'log'},
                         'runtime': {'device': 'device', 'precision': 'precision', 'threads': 'threads', 'gpus': 'gpus', 'processes':'processes'},
                         'batching': {'batch_size': 'batch_size'}, 'spectrum': {'backend': 'spectral_backend'}})
    args = p.parse_args()
    configured_launch(args.gpus,args.processes,args.device,distributed=True)
    if args.batch_size < 1 or args.threads < 1 or args.input.resolve() == args.output.resolve():
        p.error('Positive batch size and different input/output paths required')
    if args.precision not in ['fp32', 'bf16', 'fp16'] or args.spectral_backend not in [None, 'fft', 'matmul']:
        p.error('Invalid precision or spectral backend')
    started = time.monotonic()
    predictor = Predictor(args.weights, args.device, args.precision, args.spectral_backend, args.threads)
    rank,world = predictor.rank,predictor.world
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if world > 2:
        p.error('Inference supports one or two processes')
    folder = [None]
    if world > 1:
        if rank == 0:
            folder[0] = tempfile.mkdtemp(prefix='.infer_',dir=args.output.parent)
        dist.broadcast_object_list(folder,src=0)
    rows = 0
    temporary = (Path(folder[0])/f'rank{rank}.jsonl' if world > 1 else args.output.with_suffix(args.output.suffix + '.tmp'))
    offset = 0
    with temporary.open('w', encoding='utf-8') as file:
        for batch in pq.ParquetFile(args.input).iter_batches(batch_size=args.batch_size, columns=['task', 'sample_id', 'iq']):
            records=batch.to_pylist()
            selected=[(offset+i,record) for i,record in enumerate(records) if (offset+i)%world==rank]
            for (index,_),result in zip(selected,predictor.predict([record for _,record in selected]),strict=True):
                value={'row':index,'result':result} if world>1 else result
                file.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
                rows += 1
            offset += len(records)
    if world > 1:
        dist.barrier()
        if rank == 0:
            rows=merge_prediction_shards([Path(folder[0])/f'rank{i}.jsonl' for i in range(world)],args.output)
            shutil.rmtree(folder[0])
        dist.barrier()
    else:
        temporary.replace(args.output)
    if rank != 0:
        return
    record = dict(rows=rows, elapsed_seconds=time.monotonic() - started, device=str(predictor.device), output=str(args.output),
                  weights=str(args.weights.resolve()), source_checkpoint=predictor.config.get('source_checkpoint'),
                  model_parameters=predictor.config.get('parameters', {}).get('final_parameters'),world_size=world)
    if args.log:
        write_json(args.log, record)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
