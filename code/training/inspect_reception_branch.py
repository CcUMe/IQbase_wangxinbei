"""CPU post-hoc branch ablation on a fixed small monitoring panel.

These interventions diagnose a trained branch; they never choose submission
weights or report full-validation performance. Factor 10 is intentionally a
stress test, not a calibrated inference recipe.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from models.reception import reception_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise ValueError('Run with CUDA_VISIBLE_DEVICES empty: CPU diagnostic only')
    torch.set_num_threads(4)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    model = reception_model(ck).eval()
    receiver = model.backbone.reception
    is_front = ck['reception_options'].get('kind') == 'carrier_input'
    meta = np.load(ROOT / 'artifacts/data_full/metadata.npz', allow_pickle=False)
    iq = np.load(ROOT / 'artifacts/data_full/iq.npy', mmap_mode='r')
    panel = np.load(ROOT / 'artifacts/c01_followup_validation/20261113_224434/selection.npz')['panel_validation']
    labels = meta['modulation'][panel]
    rows = panel[(labels >= 0) & (labels < 14)][::7]
    labels = torch.from_numpy(meta['modulation'][rows].astype(np.int64))
    context = {}; ratios = []; factor = 1.
    def change_delta(module, inputs, output):
        context['delta'] = output * factor
        return context['delta']
    def record_input(module, inputs):
        h, mask = inputs[:2]
        delta = context['delta']; original = (h - delta)[mask]
        ratios.append(float(delta[mask].square().mean().sqrt() / original.square().mean().sqrt()))
    def record_front(module, inputs, output):
        time = torch.arange(output.shape[-1])[None]
        mask = time < context['lengths'][:, None]
        delta = context['delta']; original = output - delta
        ratios.append(float((delta.square()*mask[:,None]).sum().sqrt() /
                            (original.square()*mask[:,None]).sum().sqrt()))
    handles = [receiver.register_forward_hook(change_delta)]
    handles.append(model.backbone.time[0].register_forward_hook(record_front) if is_front else
                   model.backbone.blocks[0].register_forward_pre_hook(record_input))
    scores = {}; preds = {}
    with torch.inference_mode():
        for factor in [0., 1., 10.]:
            logits = []; ratios.clear()
            for start in range(0, len(rows), 8):
                rr = rows[start:start+8]
                lengths = torch.from_numpy(meta['length'][rr].astype(np.int64))
                context['lengths'] = lengths
                logits.append(model(torch.from_numpy(np.array(iq[rr])), lengths)['modulation'])
            z = torch.cat(logits); prediction = z.argmax(1); preds[factor] = prediction
            scores[str(factor)] = dict(accuracy=float((prediction == labels).float().mean()),
                cross_entropy=float(F.cross_entropy(z, labels)),
                delta_to_original_token_rms_mean=float(np.mean(ratios)),
                changed_predictions_vs_removed=int((prediction != preds[0.]).sum()))
    for handle in handles:
        handle.remove()
    result = dict(checkpoint=str(args.checkpoint), records=len(rows), panel_rows=rows.tolist(),
        split='validation monitoring panel, deterministic first14 stride 7; diagnostic only',
        caveat='Not a complete-validation metric or an inference-gain selection experiment',
        branch_factors=scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'panel_rows'}), flush=True)


if __name__ == '__main__':
    main()
