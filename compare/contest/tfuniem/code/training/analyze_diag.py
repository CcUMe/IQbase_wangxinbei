"""Analyse dumped outputs (diagnose.py) -> plain-text diagnostic report (guide 10.2 D1/D3 + parameter error analysis).

Usage: python code/training/analyze_diag.py --diag runs/base_r1/diag_lenbn --cache /mnt/disk1/suyang/contest/artifacts/cache_v1
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.decode import decode_parameters
from models.tfuniem import PARAM_SLICES
from utils.common import NARROWBAND_MODS
from utils.data import Cache

US = 1e6
SAMPLE_US = 0.311  # pulse-group sampling interval (guide 2.3)


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def part(par, name):
    a, b = PARAM_SLICES[name]
    return par[:, a:b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag', type=Path, required=True)
    ap.add_argument('--cache', default='/mnt/disk1/suyang/contest/artifacts/cache_v1')
    a = ap.parse_args()
    cache = Cache(a.cache)
    hs = cache.info['head_stats']
    meta, lab = cache.meta, cache.labels
    lines = []
    P = lambda *x: lines.append(' '.join(str(v) for v in x))

    def analyse(tag, d):
        r, par = d['row'], d['par']
        dec = decode_parameters(torch.from_numpy(par), hs, 'expect', 'expect')
        T, V = meta['targets'][r], meta['valid'][r]
        rms = meta['rms'][r]
        comm = V[:, 0]
        pulse = V[:, 3]
        P(f'\n######## {tag}: {len(r)} rows ########')

        # ---- frequency shift
        P('\n=== 频偏 frequency_shift (通信组) ===')
        fs_t, fs_p = T[comm, 2], dec[comm, 2]
        err = fs_p - fs_t
        real = rms[comm] < 0.05  # real-capture narrowband (RMS ~1e-3) vs simulated wideband (RMS ~1.7)
        P(f'样本数 {comm.sum()}，其中实采窄带 {real.sum()}，仿真宽带 {(~real).sum()}')
        P(f'RMSE 全部 {np.sqrt(np.mean(err ** 2)):.2f}；实采 {np.sqrt(np.mean(err[real] ** 2)):.2f}；仿真 {np.sqrt(np.mean(err[~real] ** 2)):.2f}')
        pf = softmax(part(par[comm], 'fs_cls'))
        cls_t = lab['fs_cls'][r][comm]
        cls_p = pf.argmax(1)
        P('粗分类混淆 (行=真, 列=预测; small/100/1000):')
        cm = np.zeros((3, 3), int)
        np.add.at(cm, (cls_t, cls_p), 1)
        for i, n in enumerate(['small', '100', '1000']):
            P(f'  {n:>6}: {cm[i].tolist()}')
        big = np.abs(err) > 100
        P(f'|误差|>100 的样本 {big.sum()} 条，贡献了 {100 * (err[big] ** 2).sum() / (err ** 2).sum():.1f}% 的平方误差')
        if big.any():
            P('  这些样本的 (真值, 预测值, 来源) 前 12 条:',
              [(float(t), round(float(p_), 1), 'real' if rr else 'sim') for t, p_, rr in
               list(zip(fs_t[big], fs_p[big], real[big]))[:12]])
        small = cls_t == 0
        e_small = err[small & (cls_p == 0)]
        P(f'小频偏(-50..50)且粗分正确: RMSE {np.sqrt(np.mean(e_small ** 2)):.2f} kHz (若恒输出均值约 {np.std(fs_t[small]):.2f})')

        # ---- symbol rate / bandwidth
        P('\n=== 码元速率 / 带宽 (22 类联合分类) ===')
        combos = np.array(hs['sr_bw_combos'])
        pc = softmax(part(par[comm], 'combo'))
        ct = lab['combo'][r][comm]
        cp = pc.argmax(1)
        P(f'联合分类准确率 {np.mean(cp == ct):.4f}；实采 {np.mean(cp[real] == ct[real]):.4f}；仿真 {np.mean(cp[~real] == ct[~real]):.4f}')
        sr_ok = combos[cp, 0] == combos[ct, 0]
        P(f'码元速率判对的比例 {np.mean(sr_ok):.4f}；实采 {np.mean(sr_ok[real]):.4f}；仿真 {np.mean(sr_ok[~real]):.4f}')
        for j, name in enumerate(['symbol_rate', 'bandwidth']):
            e = dec[comm, j] - T[comm, j]
            P(f'{name}: RMSE {np.sqrt(np.mean(e ** 2)):.1f} (实采 {np.sqrt(np.mean(e[real] ** 2)):.1f} / 仿真 {np.sqrt(np.mean(e[~real] ** 2)):.1f})；'
              f'平方误差中来自联合分类判错样本的占比 {100 * (e[cp != ct] ** 2).sum() / (e ** 2).sum():.1f}%')
        conf = Counter((tuple(combos[t]), tuple(combos[p_])) for t, p_ in zip(ct, cp) if t != p_)
        P('最常见的错分 (真 sr,bw -> 预测 sr,bw): 次数')
        for (t, p_), n in conf.most_common(12):
            P(f'  {t} -> {p_}: {n}')
        P('按真码元速率的联合分类准确率 / 码元速率准确率:')
        for sr in sorted(set(combos[:, 0])):
            m = combos[ct, 0] == sr
            P(f'  sr={sr:>6}: n={m.sum():>6} combo_acc={np.mean(cp[m] == ct[m]):.4f} sr_acc={np.mean(sr_ok[m]):.4f}')
        maxp = pc.max(1)
        P(f'最大概率的平均值: 判对 {maxp[cp == ct].mean():.3f} / 判错 {maxp[cp != ct].mean():.3f}')

        # ---- pulse group
        P('\n=== 脉冲组 时间量 ===')
        env = meta['env_snr'][r][pulse]
        q = cache.info['env_snr_train_quartiles_db']
        seg = np.digitize(env, q)
        names = [f'<= {q[0]:.1f}dB', f'{q[0]:.1f}-{q[1]:.1f}', f'{q[1]:.1f}-{q[2]:.1f}', f'> {q[2]:.1f}dB']
        for j, name in [(3, 'pulse_width'), (4, 'pri'), (5, 'time_delay')]:
            t_, p_ = T[pulse, j] * US, dec[pulse, j] * US
            prior = np.std(t_)
            P(f'{name}: RMSE {np.sqrt(np.mean((p_ - t_) ** 2)):.3f} us (只输出均值时的 RMSE = {prior:.3f})')
            for s in range(4):
                m = seg == s
                e = p_[m] - t_[m]
                slope = np.polyfit(t_[m], p_[m], 1)[0]
                P(f'  {names[s]:>12}: n={m.sum():>5} RMSE {np.sqrt(np.mean(e ** 2)):.3f}us '
                  f'MAE {np.mean(np.abs(e)):.3f}us = {np.mean(np.abs(e)) / SAMPLE_US:.2f} 采样点; 偏差 {e.mean():+.3f}; '
                  f'预测对真值的斜率 {slope:.2f} (1=无收缩, 0=总输出均值)')
        npc = softmax(part(par[pulse], 'npulses')).argmax(1) + 2
        nt = np.rint(T[pulse, 6]).astype(int)
        P('脉冲数分类准确率 按信噪比段: ' + ', '.join(f'{names[s]} {np.mean(npc[seg == s] == nt[seg == s]):.3f}' for s in range(4)))
        P(f'子群分类(脉冲/通信)准确率 {np.mean(softmax(part(par[r >= 0][meta["task"][r] == 2], "group")).argmax(1) == lab["group"][r][meta["task"][r] == 2]):.5f}')
        return dec

    v = dict(np.load(a.diag / 'outputs_val.npz'))
    tr = dict(np.load(a.diag / 'outputs_train_subset.npz'))
    analyse('验证集 (对比集)', v)
    analyse('训练子集 (训练中见过的样本，用来看过拟合)', tr)

    # ---- D1: modulation accuracy vs symbol rate via cross-task duplicate waveforms
    P('\n######## D1: 调制准确率 按码元速率 (利用"调制+参数"重复波形) ########')
    h = meta['hash']
    par_rows = np.flatnonzero((meta['task'] == 2) & meta['valid'][:, 0])
    sr_by_hash = {h[i].tobytes(): (meta['targets'][i, 0], meta['targets'][i, 1]) for i in par_rows}
    r = v['row']
    is_mod = meta['task'][r] == 0
    names = cache.info['schema']['classification_labels']['modulation']
    nb_idx = [names.index(c_) for c_ in NARROWBAND_MODS]
    pred = v['mod'].argmax(1)
    by_sr = defaultdict(lambda: [0, 0])
    by_ratio = defaultdict(Counter)
    for k in np.flatnonzero(is_mod):
        row = r[k]
        hit = sr_by_hash.get(h[row].tobytes())
        if hit is None or meta['mod'][row] not in nb_idx:
            continue
        by_sr[hit[0]][0] += int(pred[k] == meta['mod'][row])
        by_sr[hit[0]][1] += 1
        by_ratio[names[meta['mod'][row]]][round(hit[1] / hit[0], 3)] += 1
    for sr in sorted(by_sr):
        c_, n = by_sr[sr]
        P(f'  sr={sr:>6} ({20000 / sr:.0f} 采样点/码元): 窄带调制样本 n={n:>4}  准确率 {c_ / n:.3f}')
    P('D2 (同批样本) 调制类别 x 带宽/码元速率之比:')
    for k_, cnt in sorted(by_ratio.items()):
        P(f'  {k_:>8}: {dict(sorted(cnt.items()))}')
    out = a.diag / 'analysis.txt'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
