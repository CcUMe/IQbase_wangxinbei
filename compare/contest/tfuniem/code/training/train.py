"""Round-1 supervised multi-task training of TF-UniEM (guide 6.3).

Single GPU:  CUDA_VISIBLE_DEVICES=4 python code/training/train.py --config code/configs/base_round1.json --run-dir runs/base_r1
Multi GPU:   NCCL_CUMEM_HOST_ENABLE=0 torchrun --nproc_per_node 2 code/training/train.py ...
Overfit check: add --overfit
Terminal shows a progress bar with coloured losses; train.log is plain text.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.decode import decode_parameters
from models.tfuniem import PARAM_SLICES, TFUniEM, count_parameters
from utils.common import TARGETS
from utils.data import BalancedBatches, Cache, gather, validation_batches
from utils.score import score_all

PARAM_ITEMS = ['group', 'combo', 'fs_cls', 'fs_reg', 'npulses', 'pw', 'pri', 'td']
STAT_KEYS = ['L_total', 'L_mod', 'L_pro', 'L_param'] + ['L_' + k for k in PARAM_ITEMS] + \
            ['mod_correct', 'mod_n', 'pro_correct', 'pro_n', 'gn']

# ---- terminal colours (only on a TTY; log file stays plain)
TTY = sys.stdout.isatty()


def c(text, code):
    return f'\033[{code}m{text}\033[0m' if TTY else text


YEL, CYA, GRN, MAG, DIM, BOLD = '33', '36', '32', '35', '2', '1'


class StepModule(nn.Module):
    """Wraps both length sub-batches into ONE forward call so DDP sees one forward per backward."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, subs):
        return [self.model(s['iq'], s['task']) for s in subs]


def step_denominators(subs):
    """Computed on CPU tensors (no GPU sync)."""
    cat = {k: torch.cat([s[k] for s in subs]) for k in ['task', 'group', 'fs_cls']}
    par = cat['task'] == 2
    d = {'mod': (cat['task'] == 0).sum(), 'pro': (cat['task'] == 1).sum(), 'group': par.sum(),
         'combo': (cat['group'] == 1).sum(), 'fs_cls': (cat['group'] == 1).sum(), 'fs_reg': (cat['fs_cls'] == 0).sum(),
         'npulses': (cat['group'] == 0).sum()}
    d['pw'] = d['pri'] = d['td'] = d['npulses']
    return {k: int(v) for k, v in d.items()}


def sub_losses(out, s):
    """Sums of each loss item over this sub-batch. Labels are -1 / NaN where not applicable, so no boolean
    indexing (and no GPU sync) is needed: ignore_index for CE, masked squares for regression."""
    sums = {}
    o = out['parameter'].float()

    def part(name):
        a, b = PARAM_SLICES[name]
        return o[:, a:b]

    ce = lambda logits, y: F.cross_entropy(logits.float(), y, ignore_index=-1, reduction='sum')
    sums['mod'] = ce(out['modulation'], s['mod'])
    sums['pro'] = ce(out['protocol'], s['pro'])
    sums['group'] = ce(part('group'), s['group'])
    sums['combo'] = ce(part('combo'), s['combo'])
    sums['fs_cls'] = ce(part('fs_cls'), s['fs_cls'])
    sums['npulses'] = ce(part('npulses'), s['npulses'])
    fs_t = s['fs_reg']
    m = torch.isfinite(fs_t)
    sums['fs_reg'] = torch.where(m, (part('fs_reg')[:, 0] - torch.nan_to_num(fs_t)) ** 2, 0.0).sum()
    tt = s['times']
    mt = torch.isfinite(tt)
    se = torch.where(mt, (part('times') - torch.nan_to_num(tt)) ** 2, 0.0)
    sums['pw'], sums['pri'], sums['td'] = se[:, 0].sum(), se[:, 1].sum(), se[:, 2].sum()
    return sums


def total_loss(outs, subs, den, item_weights=None):
    items = {}
    for out, s in zip(outs, subs):
        for k, v in sub_losses(out, s).items():
            items[k] = items.get(k, 0.0) + v
    means = {k: items[k] / den[k] for k in items if den[k] > 0}
    loss = 0.0
    for k in ('mod', 'pro'):
        if k in means:
            loss = loss + means[k]
    present = [k for k in PARAM_ITEMS if k in means]
    if present:
        w = item_weights or {}
        # weighted items, divisor = number of present items (weight 2 doubles that item's gradient, others unchanged)
        means['param'] = sum(w.get(k, 1.0) * means[k] for k in present) / len(present)
        loss = loss + means['param']
    means['total'] = loss
    return loss, {k: v.detach() for k, v in means.items()}  # GPU tensors, no sync


def lr_factor(step, warmup, total, floor):
    if step < warmup:
        return (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))


def param_groups(model, wd):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith('queries') or 'embedding' in name else decay).append(p)
    return [{'params': decay, 'weight_decay': wd}, {'params': no_decay, 'weight_decay': 0.0}]


def fmt_time(s):
    s = int(max(0, s))
    return f'{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}' if s >= 3600 else f'{s // 60}:{s % 60:02d}'


def bar(frac, width=28):
    n = int(round(frac * width))
    return '█' * n + '░' * (width - n)


def progress_line(epoch, epochs, done, total, sps, eta_ep, eta_all, lb):
    """One terminal line that never exceeds the terminal width (wrapping would break the in-place update)."""
    cols = shutil.get_terminal_size((120, 24)).columns - 1
    head = ('ep %d/%d' % (epoch, epochs), BOLD)
    stats = ' %d/%d %3.0f%% %4.0f/s ETA %s all %s' % (done, total, 100 * done / total, sps, fmt_time(eta_ep),
                                                     fmt_time(eta_all))
    loss_main = [(' | loss ', None), ('%.4f' % lb['L_total'], YEL)]
    loss_parts = [(' mod ', None), ('%.3f' % lb['L_mod'], YEL), (' pro ', None), ('%.3f' % lb['L_pro'], YEL),
                  (' param ', None), ('%.3f' % lb['L_param'], YEL)]
    acc_parts = [(' | acc mod ', None), ('%.3f' % lb['mod_acc'], GRN), (' pro ', None), ('%.3f' % lb['pro_acc'], GRN)]
    for tail in (loss_main + loss_parts + acc_parts, loss_main + loss_parts, loss_main, []):
        plain_len = len(head[0]) + 1 + len(stats) + sum(len(t) for t, _ in tail)
        width = cols - plain_len
        if width >= 10:
            break
    width = max(4, min(40, width))
    out = c(head[0], head[1]) + ' ' + c(bar(done / total, width), CYA) + stats
    return out + ''.join(c(t, col) if col else t for t, col in tail)


@torch.no_grad()
def predict_rows(model, cache, rows, device, batch_size=1024, show=False):
    res = {k: [] for k in ['row', 'mod', 'pro', 'par_expect', 'par_argmax', 'group']}
    lengths = cache.meta['length'][rows]
    n_batches = sum(math.ceil((lengths == L).sum() / batch_size) for L in (512, 1024))
    t0 = time.time()
    for i, b in enumerate(validation_batches(cache, rows, batch_size)):
        iq, task = b['iq'].to(device, non_blocking=True), b['task'].to(device)
        out = model(iq, task)
        res['row'].append(b['row'].numpy())
        res['mod'].append(out['modulation'].float().argmax(1).cpu().numpy())
        res['pro'].append(out['protocol'].float().argmax(1).cpu().numpy())
        res['group'].append(out['parameter'][:, 0:2].float().argmax(1).cpu().numpy())
        res['par_expect'].append(decode_parameters(out['parameter'], cache.info['head_stats'], 'expect', 'expect'))
        res['par_argmax'].append(decode_parameters(out['parameter'], cache.info['head_stats'], 'argmax', 'argmax'))
        if show and TTY:
            f = (i + 1) / n_batches
            print(f'\r  {c("validating", CYA)} {bar(f, 20)} {i + 1}/{n_batches} {fmt_time(time.time() - t0)}',
                  end='', flush=True)
    if show and TTY:
        print('\r' + ' ' * 70 + '\r', end='', flush=True)
    return {k: np.concatenate(v) for k, v in res.items()}


def evaluate(model, cache, rows, device, train_mode_bn=False, show=False):
    model.eval()
    saved = {}
    if train_mode_bn:  # diagnostic: BatchNorm uses batch statistics; running stats restored afterwards
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                saved[m] = {k: b.clone() for k, b in m.named_buffers(recurse=False)}
                m.train()
    t0 = time.time()
    pr = predict_rows(model, cache, rows, device, show=show)
    for m, bufs in saved.items():
        for k, b in m.named_buffers(recurse=False):
            b.copy_(bufs[k])
        m.eval()
    r = pr['row']
    meta, lab = cache.meta, cache.labels
    std = cache.info['target_scaler']['std']
    env_bins = cache.info['env_snr_train_quartiles_db']
    args = (meta['task'][r], pr['mod'], lab['mod'][r], pr['pro'], lab['pro'][r])
    main = score_all(*args, pr['par_expect'], meta['targets'][r], meta['valid'][r], cache.info['schema'], std,
                     meta['env_snr'][r], env_bins)
    alt = score_all(*args, pr['par_argmax'], meta['targets'][r], meta['valid'][r], cache.info['schema'], std)
    par = meta['task'][r] == 2
    main['param_group_accuracy'] = float((pr['group'][par] == lab['group'][r][par]).mean())
    main['argmax_decode'] = {'proxy_score': alt['proxy_score'], 'parameter_proxy': alt['parameter_proxy'],
                             'rmse': {k: alt['parameters'][k]['rmse'] for k in TARGETS}}
    main['eval_seconds'] = time.time() - t0
    return main


def brief(m):
    p = m['parameters']
    return (f"proxy={m['proxy_score']:.3f} pro={m['protocol_accuracy']:.4f} mod={m['modulation_accuracy']:.4f} "
            f"(nb14={m['modulation']['narrowband14_accuracy']:.4f} oth9={m['modulation']['other9_accuracy']:.4f}) "
            f"pp={m['parameter_proxy']:.4f} grp={m['param_group_accuracy']:.4f} | RMSE sr={p['symbol_rate']['rmse']:.1f} "
            f"bw={p['bandwidth']['rmse']:.1f} fs={p['frequency_shift']['rmse']:.2f} pw={p['pulse_width']['rmse'] * 1e6:.3f}us "
            f"pri={p['pri']['rmse'] * 1e6:.3f}us td={p['time_delay']['rmse'] * 1e6:.3f}us np={p['number_of_pulses']['rmse']:.3f}")


def to_device(subs, device):
    return [{k: v.to(device, non_blocking=True) for k, v in s.items()} for s in subs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--overfit', action='store_true')
    ap.add_argument('--max-steps', type=int, default=0, help='debug: stop early')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())

    ddp = 'RANK' in os.environ
    if ddp:
        dist.init_process_group('nccl')
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ['LOCAL_RANK'])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    seed = cfg['seed']
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    run = args.run_dir
    if rank == 0:
        run.mkdir(parents=True, exist_ok=True)
        (run / 'config.json').write_text(json.dumps(cfg, indent=2))
    logf = open(run / 'train.log', 'a') if rank == 0 else None

    def log(plain, colored=None, terminal=True):
        """Plain text to train.log; coloured version (if given) to the terminal."""
        if rank == 0:
            ts = time.strftime('%Y-%m-%d %H:%M:%S ')
            if terminal:
                if TTY:
                    print('\r\033[K', end='')
                print(c(ts, DIM) + (colored or plain), flush=True)
            logf.write(ts + plain + '\n')
            logf.flush()

    cache = Cache(cfg['cache'])
    log(f'torch {torch.__version__} cuda {torch.version.cuda} world {world} device '
        f'{os.environ.get("CUDA_VISIBLE_DEVICES", "?")} cache {cfg["cache"]}')
    model = TFUniEM(**cfg['model']).to(device)
    pc = count_parameters(model)
    log('parameters ' + json.dumps(pc),
        f"参数量 {pc['total'] / 1e6:.2f}M, 共享主干占比 {100 * pc['shared_fraction']:.1f}% (红线 > 80%)")
    if ddp:
        # SyncBatchNorm conversion would drop LenBN's per-length running statistics
        raise NotImplementedError('multi-GPU training is not supported with per-length BatchNorm (LenBN)')
    step_module = StepModule(model).to(device)
    if ddp:
        step_module = nn.parallel.DistributedDataParallel(step_module, device_ids=[local])
    opt_cfg = cfg['optim']
    opt = torch.optim.AdamW(param_groups(model, opt_cfg['weight_decay']), lr=opt_cfg['lr'], betas=(0.9, 0.999),
                            fused=True)
    steps_per_epoch, epochs = cfg['steps_per_epoch'], cfg['epochs']
    total_steps = steps_per_epoch * epochs
    if args.overfit:
        total_steps, steps_per_epoch, epochs = cfg['overfit']['steps'], cfg['overfit']['steps'], 1
    heads = {0: model.mod_head, 1: model.pro_head, 2: model.param_head}
    log_every = cfg['log_every'] if not args.overfit else 50
    bar_every = cfg.get('bar_every', 20)

    if args.overfit:
        rng = np.random.default_rng(seed)
        fixed = np.concatenate([rng.choice(cache.rows(0, t), k, replace=False)
                                for t, k in enumerate(cfg['per_task'])])
        fixed_batch = gather(cache, fixed)
        loader = None
    else:
        train_rows = np.flatnonzero((cache.meta['split'] == 0) &
                                    ~(cache.meta['tune'] if cfg.get('exclude_tune_from_train') else False))
        ds = BalancedBatches(cfg['cache'], cfg['per_task'], seed, rank, rows_override=train_rows)
        log(f'training rows {len(train_rows)} (tune subset excluded: {bool(cfg.get("exclude_tune_from_train"))})')
        loader = iter(torch.utils.data.DataLoader(ds, batch_size=None, num_workers=cfg['num_workers'],
                                                  pin_memory=True, prefetch_factor=4, persistent_workers=True))
    # per-epoch selection set: 'tune' (held-out tune subset, guide 6.1) or 'val' (comparison set, round 1 only)
    select_on = cfg.get('select_on', 'val')
    if select_on == 'tune':
        assert cfg.get('exclude_tune_from_train'), 'tune selection needs the tune subset excluded from training'
        val_rows = np.flatnonzero((cache.meta['split'] == 0) & cache.meta['tune'])
    else:
        val_rows = cache.rows(1)
    sel_tag = 'TUNE' if select_on == 'tune' else 'VAL'
    log(f'per-epoch selection on {sel_tag} set: {len(val_rows)} rows')
    best, best_epoch = -1.0, 0
    step = 0
    acc = torch.zeros(len(STAT_KEYS), device=device, dtype=torch.float64)  # running sums since last log
    acc_steps = 0
    last_bar = {}
    t_start = time.time()
    t_log = time.time()
    for epoch in range(1, epochs + 1):
        step_module.train()
        t_epoch = time.time()
        for i_ep in range(steps_per_epoch):
            if args.max_steps and step >= args.max_steps:
                break
            batch = fixed_batch if args.overfit else next(loader)
            subs_cpu = [batch[L] for L in (512, 1024) if L in batch]
            den = step_denominators(subs_cpu)
            subs = to_device(subs_cpu, device)
            for g in opt.param_groups:
                g['lr'] = opt_cfg['lr'] * lr_factor(step, opt_cfg['warmup'], total_steps, opt_cfg['final_lr_frac'])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                outs = step_module(subs)
            loss, items = total_loss(outs, subs, den, cfg.get('param_item_weights'))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # heads with no samples this step (globally): grads -> None so AdamW leaves them untouched
            counts = [den['mod'], den['pro'], den['group']]
            if ddp:
                ct = torch.tensor(counts, device=device)
                dist.all_reduce(ct)
                counts = ct.tolist()
            for t, head in heads.items():
                if counts[t] == 0:
                    for p in head.parameters():
                        p.grad = None
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg['grad_clip'])
            opt.step()
            step += 1
            # accumulate stats on GPU (no sync)
            with torch.no_grad():
                vec = []
                for k in STAT_KEYS:
                    if k.startswith('L_'):
                        v = items.get(k[2:])
                        vec.append(v.double() if v is not None else torch.zeros((), device=device, dtype=torch.float64))
                mc = sum((o['modulation'].argmax(1) == s['mod']).sum() for o, s in zip(outs, subs))
                pc_ = sum((o['protocol'].argmax(1) == s['pro']).sum() for o, s in zip(outs, subs))
                vec += [mc.double(), torch.tensor(float(den['mod']), device=device, dtype=torch.float64),
                        pc_.double(), torch.tensor(float(den['pro']), device=device, dtype=torch.float64),
                        gn.double()]
                acc += torch.stack(vec)
            acc_steps += 1

            # progress bar (one GPU sync every bar_every steps)
            if rank == 0 and (step % bar_every == 0 or i_ep == steps_per_epoch - 1):
                vals = dict(zip(STAT_KEYS, acc.tolist()))
                last_bar = {k: vals[k] / acc_steps for k in STAT_KEYS if k.startswith('L_') or k == 'gn'}
                last_bar['mod_acc'] = vals['mod_correct'] / max(vals['mod_n'], 1)
                last_bar['pro_acc'] = vals['pro_correct'] / max(vals['pro_n'], 1)
                if not math.isfinite(last_bar['gn']) or not math.isfinite(last_bar['L_total']):
                    raise FloatingPointError(f'non-finite loss/grad norm near step {step}')
                if TTY:
                    done = i_ep + 1
                    el = time.time() - t_epoch
                    sps = done * cfg['per_task_total'] * world / max(el, 1e-6)
                    eta_ep = el / done * (steps_per_epoch - done)
                    eta_all = (time.time() - t_start) / step * (total_steps - step)
                    print('\r\033[K' + progress_line(epoch, epochs, done, steps_per_epoch, sps, eta_ep, eta_all,
                                                     last_bar), end='', flush=True)

            if step % log_every == 0:
                dt = time.time() - t_log
                t_log = time.time()
                vals = dict(zip(STAT_KEYS, acc.tolist()))
                summary = {k: vals[k] / acc_steps for k in STAT_KEYS if k.startswith('L_') or k == 'gn'}
                summary['mod_acc'] = vals['mod_correct'] / max(vals['mod_n'], 1)
                summary['pro_acc'] = vals['pro_correct'] / max(vals['pro_n'], 1)
                sps = acc_steps * cfg['per_task_total'] * world / dt
                head = f'ep {epoch} step {step} lr {opt.param_groups[0]["lr"]:.2e} {sps:.0f} samples/s '
                plain = head + ' '.join(f'{k}={v:.4f}' for k, v in sorted(summary.items()))
                colored = head + ' '.join(
                    f'{k}={c(f"{v:.4f}", YEL if k.startswith("L_") else (GRN if k.endswith("acc") else DIM))}'
                    for k, v in sorted(summary.items()))
                log(plain, colored, terminal=False)
                if rank == 0:
                    with open(run / 'train_curve.jsonl', 'a') as f:
                        f.write(json.dumps({'step': step, 'epoch': epoch, 'lr': opt.param_groups[0]['lr'],
                                            'samples_per_s': sps, **summary}) + '\n')
                acc.zero_()
                acc_steps = 0
        if args.overfit:
            break
        # ---- validation (rank 0, fp32, eval-mode BN)
        if rank == 0:
            if TTY:
                print(flush=True)  # keep this epoch's finished bar on its own line
            m = evaluate(model, cache, val_rows, device, show=True)
            rec = {'epoch': epoch, 'step': step, **{k: m[k] for k in ['proxy_score', 'parameter_proxy',
                   'protocol_accuracy', 'modulation_accuracy', 'param_group_accuracy']},
                   'rmse': {k: m['parameters'][k]['rmse'] for k in TARGETS}, 'argmax_decode': m['argmax_decode'],
                   'epoch_seconds': time.time() - t_epoch}
            plain = (f'{sel_tag} ep {epoch}: ' + brief(m) + f' | argmax-decode proxy={m["argmax_decode"]["proxy_score"]:.3f}'
                     f' ({m["eval_seconds"]:.0f}s)')
            log(plain, c(plain, GRN))
            with open(run / 'val_history.jsonl', 'a') as f:
                f.write(json.dumps(rec) + '\n')
            state = {'model': {k: v for k, v in model.state_dict().items()}, 'model_cfg': model.cfg,
                     'head_stats': cache.info['head_stats'], 'schema': cache.info['schema'],
                     'target_scaler': cache.info['target_scaler'], 'epoch': epoch, 'step': step,
                     'val': {k: m[k] for k in ['proxy_score', 'protocol_accuracy', 'modulation_accuracy']}}
            torch.save(state, run / 'last.pt')
            if m['proxy_score'] > best:
                best, best_epoch = m['proxy_score'], epoch
                torch.save(state, run / 'best.pt')
                (run / 'best_metrics.json').write_text(json.dumps({'epoch': epoch, **m}, indent=1))
                msg = f'  new best proxy {best:.3f} at epoch {epoch}'
                log(msg, c(msg, MAG + ';1'))
            else:
                log(f'  best so far {best:.3f} at epoch {best_epoch}')
        if ddp:
            dist.barrier()
        if args.max_steps and step >= args.max_steps:
            break

    if args.overfit and rank == 0:
        step_module.train()
        subs = to_device([fixed_batch[L] for L in (512, 1024) if L in fixed_batch], device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            outs = step_module(subs)
        _, items = total_loss(outs, subs, step_denominators([fixed_batch[L] for L in (512, 1024) if L in fixed_batch]))
        accs = {}
        for key, tid, lk in [('modulation', 0, 'mod'), ('protocol', 1, 'pro')]:
            cnum = n = 0
            for out, s in zip(outs, subs):
                msk = s['task'] == tid
                cnum += int((out[key][msk].argmax(1) == s[lk][msk]).sum())
                n += int(msk.sum())
            accs[key] = cnum / n
        res = {'acc': accs, 'loss_items': {k: float(v) for k, v in items.items()}}
        log('OVERFIT final (train-mode BN): ' + json.dumps(res))
        (run / 'overfit_result.json').write_text(json.dumps(res, indent=1))
    if rank == 0 and not args.overfit:
        log(f'finished: best {sel_tag} proxy {best:.3f} at epoch {best_epoch}; total time {fmt_time(time.time() - t_start)}')
        if select_on == 'tune':
            # milestone: score the selected checkpoint ONCE on the comparison set (leader-comparable)
            st = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(st['model'])
            m = evaluate(model, cache, cache.rows(1), device, show=True)
            (run / 'comparison_metrics.json').write_text(json.dumps({'epoch': best_epoch, **m}, indent=1))
            plain = f'COMPARISON SET (best epoch {best_epoch}): ' + brief(m)
            log(plain, c(plain, MAG + ';1'))
    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
