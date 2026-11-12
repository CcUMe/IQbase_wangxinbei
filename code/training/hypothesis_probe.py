"""Fixed-budget train-only standardized linear probes and paired IQ perturbations."""
import argparse
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.rf_model import RFModel
from training.diagnose_readout import feature_metrics
from utils.common import model_bundle, runtime, seed_all, save_checkpoint, write_json, log_json
from utils.config import make_scheduler
from utils.dataset import PreparedDataset, collate, to_device
from utils.metrics import classification_metrics
from utils.readable_log import ReadableRunLog
from utils.training_display import EpochProgress, classification_summary


PERTURBATIONS = ['clean', 'gain_half', 'gain_double', 'phase_90', 'cfo_plus', 'cfo_minus', 'shift_16']


def perturb(iq, lengths, kind):
    if kind == 'clean':
        return iq
    if kind == 'gain_half':
        return iq * .5
    if kind == 'gain_double':
        return iq * 2
    if kind == 'phase_90':
        return torch.stack((-iq[..., 1], iq[..., 0]), -1)
    coordinate = torch.arange(iq.shape[1], device=iq.device)[None]
    if kind == 'shift_16':
        index = (coordinate - 16) % lengths[:, None]
        moved = iq.gather(1, index[..., None].expand(-1, -1, 2))
        return moved * (coordinate < lengths[:, None])[..., None]
    if kind in ['cfo_plus', 'cfo_minus']:
        angle = coordinate * (2 * math.pi * .001 * (1 if kind == 'cfo_plus' else -1))
        real = iq[..., 0] * angle.cos() - iq[..., 1] * angle.sin()
        imag = iq[..., 0] * angle.sin() + iq[..., 1] * angle.cos()
        return torch.stack((real, imag), -1)
    raise ValueError(kind)


def group_metrics(metrics):
    matrix = np.asarray(metrics['confusion_matrix'])
    count = int(matrix[:14].sum())
    return {'support': count, 'accuracy': float(np.diag(matrix)[:14].sum() / max(1, count)),
            'macro_f1': float(np.mean(metrics['f1'][:14])),
            'routing_accuracy': float(matrix[:14, :14].sum() / max(1, count))}


def prepare_data(directory, split):
    data = PreparedDataset(directory, split)
    rows = data.indices[data.meta['modulation'][data.indices] >= 0]
    data.indices = rows[np.argsort(data.meta['length'][rows], kind='stable')]
    return data


@torch.inference_mode()
def extract(model, data, device, kind='clean', interventions=False):
    all_features, predictions, logits_sum, confidence_sum = [], [], 0., 0.
    for cpu in DataLoader(data, batch_size=256, collate_fn=collate, num_workers=2, pin_memory=True):
        batch = to_device(cpu, device)
        iq = perturb(batch['iq'], batch['lengths'], kind)
        if interventions:
            output = model(iq, batch['lengths'])['modulation'].float()
            predictions.append(output.argmax(-1).cpu().numpy())
            logits_sum += float(F.cross_entropy(output, batch['modulation'], reduction='sum'))
            confidence_sum += float(output.softmax(-1).amax(-1).sum())
        else:
            h, valid, _ = model.encode(iq, batch['lengths'])
            all_features.append((h.sum(1) / valid.sum(1, keepdim=True)).float().cpu())
    if interventions:
        labels = data.meta['modulation'][data.indices]
        prediction = np.concatenate(predictions)
        n = len(model.schema['classification_labels']['modulation'])
        cm = np.bincount(labels * n + prediction, minlength=n*n).reshape(n, n)
        metrics = classification_metrics(cm, model.schema['classification_labels']['modulation'])
        metrics.update(cross_entropy=logits_sum/len(labels), mean_confidence=confidence_sum/len(labels))
        metrics['hard14'] = group_metrics(metrics)
        return metrics, prediction
    return torch.cat(all_features)


def interventions(args, model, device):
    validation = prepare_data(args.data, 'validation')
    results, clean_predictions = {}, None
    for name, kind, scale, global_input in [
        ('clean', 'clean', True, True), ('drop_scale', 'clean', False, True),
        ('drop_global', 'clean', True, False), ('drop_both', 'clean', False, False),
        ('gain_half', 'gain_half', True, True), ('gain_double', 'gain_double', True, True),
        ('phase_90', 'phase_90', True, True), ('cfo_plus', 'cfo_plus', True, True),
        ('cfo_minus', 'cfo_minus', True, True)]:
        model.config.update(scale_embedding=scale, global_spectrum_input=global_input)
        metrics, predictions = extract(model, validation, device, kind, True)
        if clean_predictions is None:
            clean_predictions = predictions
        metrics['prediction_flip_fraction'] = float(np.mean(predictions != clean_predictions))
        results[name] = metrics
        np.save(args.output / f'{name}_predictions.npy', predictions)
        print(name, f'Accuracy={metrics["accuracy"]:.4%} | hard14={metrics["hard14"]["accuracy"]:.4%}', flush=True)
        write_json(args.output / 'summary.json', results)
    print('推理时关闭分支属于分布干预，不能代替从头消融；所有扰动仅评估调制任务。', flush=True)


def probes(args, model, device, source):
    train, validation = prepare_data(args.data, 'train'), prepare_data(args.data, 'validation')
    names = source['schema']['classification_labels']['modulation']
    print('提取冻结 FP32 编码器特征 ...', flush=True)
    x, validation_x = extract(model, train, device), extract(model, validation, device)
    labels = torch.from_numpy(train.meta['modulation'][train.indices].astype(np.int64))
    validation_y = torch.from_numpy(validation.meta['modulation'][validation.indices].astype(np.int64))
    np.save(args.output/'train_rows.npy', train.indices)
    np.save(args.output/'validation_rows.npy', validation.indices)
    np.save(args.output/'validation_labels.npy', validation_y.numpy())
    np.save(args.output/'train_features.npy', x.numpy())
    np.save(args.output/'validation_features.npy', validation_x.numpy())
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-4)
    normalized_x, clean_x = (x-mean)/std, (validation_x-mean)/std
    heads, results = [], []
    epochs, batch_size = 20, 1024
    for seed in [2026, 2027]:
        seed_all(seed)
        head = torch.nn.Linear(x.shape[1], len(names))
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=.01)
        scheduler, _ = make_scheduler(optimizer, dict(name='cosine', warmup_steps=0, warmup_ratio=0.,
            warmup_start_factor=0., min_lr_ratio=.05), math.ceil(len(x)/batch_size)*epochs)
        for epoch in range(epochs):
            order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed + epoch*100003))
            progress = EpochProgress(epoch, epochs, math.ceil(len(x)/batch_size), min_interval=1.)
            with progress:
                for begin in range(0, len(x), batch_size):
                    index = order[begin:begin+batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(head(normalized_x[index]), labels[index])
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                    rate = optimizer.param_groups[0]['lr']
                    optimizer.step(); scheduler.step()
                    progress.observe(float(loss.detach()), {'modulation':float(loss.detach())}, {'modulation'})
                    progress.advance({'head':rate},float(norm))
        training, _ = feature_metrics(head, normalized_x, labels, names)
        metric, prediction = feature_metrics(head, clean_x, validation_y, names)
        metric['hard14'] = group_metrics(metric)
        classification_summary({'classification':{'modulation':metric}})
        heads.append(head.eval())
        results.append({'seed':seed, 'training':training, 'clean':metric, 'perturbations':{}})
        np.save(args.output/f'clean_seed{seed}_predictions.npy', prediction)
        save_checkpoint(args.output/f'linear_seed{seed}.pt', dict(head=head.state_dict(), mean=mean, std=std))
    for kind in PERTURBATIONS[1:]:
        changed = extract(model, validation, device, kind)
        normalized = (changed-mean)/std
        cosine = float(F.cosine_similarity(validation_x, changed).mean())
        for head, result in zip(heads, results):
            metric, prediction = feature_metrics(head, normalized, validation_y, names)
            metric['hard14'] = group_metrics(metric)
            original = np.load(args.output/f'clean_seed{result["seed"]}_predictions.npy')
            metric.update(prediction_flip_fraction=float(np.mean(prediction!=original)), feature_cosine=cosine)
            result['perturbations'][kind] = metric
            np.save(args.output/f'{kind}_seed{result["seed"]}_predictions.npy', prediction)
        print(kind, 'Accuracies', [f'{v["perturbations"][kind]["accuracy"]:.4%}' for v in results],flush=True)
        write_json(args.output/'results.json',results)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='auto')
    p.add_argument('--interventions',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    device,_,world=runtime(args.device,4)
    if world!=1:raise ValueError('Single process per diagnostic')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    source=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    model=RFModel(**model_bundle(source)).to(device).eval().requires_grad_(False)
    model.load_state_dict(source['model'],strict=True)
    data=PreparedDataset(args.data)
    if data.info['fingerprint']!=source['dataset_fingerprint']:raise ValueError('Dataset mismatch')
    protocol={'checkpoint':str(args.checkpoint.resolve()),'dataset_fingerprint':source['dataset_fingerprint'],
        'mode':'interventions' if args.interventions else 'linear_probes','precision':'fp32',
        'frozen':True,'pooling':'mean','readout_epochs':20,'readout_seeds':[2026,2027],
        'readout_device':'cpu','readout_learning_rate':1e-3,'readout_weight_decay':.01,
        'readout_batch_size':1024,'selection':'fixed final epoch','feature_standardization':'train only',
        'perturbations':{'phase_90':'constant complex rotation','gain_half':.5,'gain_double':2.,
            'cfo_plus':.001,'cfo_minus':-.001,'cfo_units':'cycles/sample; stress test, not measured physical drift',
            'shift_16':'circular shift by 16 valid samples; modulation only'},
        'note':'No physical sampling-rate alignment or SNR claim; perturbation performance is not hidden-test accuracy.'}
    write_json(args.output/'protocol.json',protocol)
    started=time.monotonic()
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    with ReadableRunLog(args.output):
        print('诊断总览',protocol,flush=True)
        if args.interventions:interventions(args,model,device)
        else:probes(args,model,device,source)
    assert all(torch.equal(value,model.state_dict()[name].cpu()) for name,value in before.items())
    protocol.update(elapsed_seconds=time.monotonic()-started,backbone_and_heads_bitwise_unchanged=True)
    write_json(args.output/'protocol.json',protocol)


if __name__=='__main__':main()
