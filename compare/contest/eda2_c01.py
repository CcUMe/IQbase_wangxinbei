#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C01 第二轮核实脚本：只读数据，输出 eda2_report.txt。

要回答的三个问题:
  A. symbol_rate / bandwidth / frequency_shift 是否联合取值（全量标签交叉表）
  B. 脉冲组的采样间隔，以及 time_delay / pri / pulse_width 与波形的对应关系
  C. 通信组的采样率，以及 frequency_shift 的物理含义（载频偏移还是频偏/调制指数）
另外: 全量统计 序列长度 x 任务/类别；各调制、体制类别的谱特征。

用法:
    python eda2_c01.py /path/to/C01
    python eda2_c01.py /path/to/C01 --groups 40 --skip-full-length

依赖: numpy, pandas, pyarrow
"""
import argparse
import io
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

OUT = io.StringIO()
PARAMS = ["symbol_rate", "bandwidth", "frequency_shift", "pulse_width", "pri", "time_delay", "number_of_pulses"]
BLOCKS = " ▁▂▃▄▅▆▇█"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 400)


def P(*a):
    s = " ".join(str(x) for x in a)
    for line in s.split("\n"):
        OUT.write((line[:3000] + " ...[截断]" if len(line) > 3000 else line) + "\n")


def H(title):
    P("\n" + "=" * 100)
    P("## " + title)
    P("=" * 100)
    sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), title))


def safe(title, fn, *args, **kw):
    H(title)
    try:
        return fn(*args, **kw)
    except Exception:
        P("!! 本节出错（脚本继续执行）:")
        P(traceback.format_exc())
        return None


def g6(v):
    try:
        return "%.6g" % float(v)
    except Exception:
        return str(v)


def pct(x, ps=(5, 25, 50, 75, 95)):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "-"
    return " ".join(g6(v) for v in np.percentile(x, ps))


# ----------------------------------------------------------------------------
def find_train_parquets(root):
    res = []
    for d, _, files in sorted(os.walk(root)):
        for f in sorted(files):
            if f.lower().endswith(".parquet"):
                p = os.path.join(d, f)
                try:
                    names = pq.ParquetFile(p).schema_arrow.names
                except Exception:
                    continue
                if "iq" in names and "symbol_rate" in names:
                    res.append(p)
    return res


def load_labels(paths):
    frames = {}
    for p in paths:
        names = pq.ParquetFile(p).schema_arrow.names
        cols = [n for n in names if n != "iq"]
        frames[p] = pq.read_table(p, columns=cols).to_pandas()
        P("%s: %d 行" % (os.path.basename(p), len(frames[p])))
    return frames


def iq_to_list(col):
    """Arrow 列 -> [np.ndarray(L,2) float32, ...]。优先走零拷贝的快速路径。"""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    try:
        lens = pc.list_value_length(col).to_numpy(zero_copy_only=False).astype(np.int64)
        flat = col.flatten().flatten().to_numpy(zero_copy_only=False)
        if flat.size != 2 * int(lens.sum()):
            raise ValueError("flat size mismatch")
        flat = flat.reshape(-1, 2)
        off = np.concatenate([[0], np.cumsum(lens)])
        return [flat[off[i]:off[i + 1]] for i in range(len(lens))], "fast"
    except Exception:
        return [np.asarray(v, dtype=np.float32).reshape(-1, 2) for v in col.to_pylist()], "slow"


# ----------------------------------------------------------------------------
# A. 标签联合分布（全量）
# ----------------------------------------------------------------------------
def part_a(frames):
    df = pd.concat(list(frames.values()), ignore_index=True)
    par = df[df["task"] == "parameter"]
    comm = par[par["symbol_rate_valid"]]
    pulse = par[par["pulse_width_valid"]]
    P("参数样本 %d = 通信组 %d + 脉冲组 %d" % (len(par), len(comm), len(pulse)))

    P("\n### A1 symbol_rate x bandwidth (通信组全量)")
    P(pd.crosstab(comm["symbol_rate"], comm["bandwidth"]).to_string())
    P("\n### A2 bandwidth / symbol_rate 比值的取值计数")
    P((comm["bandwidth"] / comm["symbol_rate"]).round(4).value_counts().sort_index().to_string())
    P("\n### A3 symbol_rate x frequency_shift")
    P(pd.crosstab(comm["symbol_rate"], comm["frequency_shift"]).to_string())
    P("\n### A4 bandwidth x frequency_shift")
    P(pd.crosstab(comm["bandwidth"], comm["frequency_shift"]).to_string())
    trip = comm.groupby(["symbol_rate", "bandwidth", "frequency_shift"]).size()
    P("\n### A5 (symbol_rate, bandwidth, frequency_shift) 三元组合总数 = %d" % len(trip))
    fcat = np.where(comm["frequency_shift"].abs() <= 50, "|f|<=50", comm["frequency_shift"].astype(int).astype(str))
    P(pd.crosstab([comm["symbol_rate"], comm["bandwidth"]], fcat).to_string())

    P("\n### A6 脉冲组四个参数的关系 (全量)")
    n = pulse["number_of_pulses"].to_numpy(np.float64)
    pw = pulse["pulse_width"].to_numpy(np.float64)
    pri = pulse["pri"].to_numpy(np.float64)
    td = pulse["time_delay"].to_numpy(np.float64)
    P("pulse_width/pri 分位(0/5/50/95/100):", pct(pw / pri, (0, 5, 50, 95, 100)), "  pw>=pri 的条数:", int((pw >= pri).sum()))
    span = td + (n - 1) * pri + pw
    P("末脉冲结束时刻 td+(N-1)*pri+pw 分位(0/5/50/95/100):", pct(span, (0, 5, 50, 95, 100)))
    P("td+N*pri 分位(0/5/50/95/100):", pct(td + n * pri, (0, 5, 50, 95, 100)))
    P("Pearson 相关:")
    P(pulse[["pulse_width", "pri", "time_delay", "number_of_pulses"]].corr().round(3).to_string())
    for c in ["pulse_width", "pri", "time_delay"]:
        v = pulse[c].to_numpy(np.float64) * 1e6
        hist, edges = np.histogram(v, bins=12)
        P("%s 直方图(µs): %s" % (c, "  ".join("%.1f:%d" % (edges[i], hist[i]) for i in range(12))))
    P("number_of_pulses 按 time_delay 三分位的均值:",
      pulse.groupby(pd.qcut(pulse["time_delay"], 3, labels=["低", "中", "高"]), observed=True)["number_of_pulses"].mean().round(3).to_dict())


# ----------------------------------------------------------------------------
# B0. 全量长度统计
# ----------------------------------------------------------------------------
def part_len(frames):
    rows = []
    for p, df in frames.items():
        pf = pq.ParquetFile(p)
        lens = []
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=["iq"])
            lens.append(pc.list_value_length(t.column(0)).to_numpy())
        lens = np.concatenate(lens)
        if len(lens) != len(df):
            P("!! 行数不一致:", p, len(lens), len(df))
            continue
        d = df[["task", "modulation_label", "protocol_label", "symbol_rate_valid", "pulse_width_valid"]].copy()
        d["L"] = lens
        rows.append(d)
        sys.stderr.write("   长度扫描完成: %s\n" % os.path.basename(p))
    d = pd.concat(rows, ignore_index=True)
    P("### 长度 x 任务 (全量)")
    P(pd.crosstab(d["L"], d["task"]).to_string())
    par = d[d["task"] == "parameter"]
    grp = np.where(par["pulse_width_valid"], "脉冲组", np.where(par["symbol_rate_valid"], "通信组", "其他"))
    P("\n### 长度 x 参数子群 (全量)")
    P(pd.crosstab(par["L"], grp).to_string())
    P("\n### 长度 x 调制类别 (全量)")
    P(pd.crosstab(d.loc[d["task"] == "modulation", "modulation_label"], d.loc[d["task"] == "modulation", "L"]).to_string())
    P("\n### 长度 x 体制类别 (全量)")
    P(pd.crosstab(d.loc[d["task"] == "protocol", "protocol_label"], d.loc[d["task"] == "protocol", "L"]).to_string())


# ----------------------------------------------------------------------------
# 信号特征
# ----------------------------------------------------------------------------
def spec_feat(x):
    """x: (L,2)。返回归一化频率量 (单位: 周期/采样点)。"""
    z = x[:, 0].astype(np.float64) + 1j * x[:, 1].astype(np.float64)
    L = z.size
    p = np.abs(z) ** 2
    f = {"L": L, "rms": float(np.sqrt(p.mean()))}
    if f["rms"] <= 0:
        return f
    S = np.abs(np.fft.fftshift(np.fft.fft(z))) ** 2
    fr = (np.arange(L) - L // 2) / float(L)
    cs = np.cumsum(S) / S.sum()
    f["centroid"] = float((fr * S).sum() / S.sum())
    f["bw99"] = float((np.searchsorted(cs, 0.995) - np.searchsorted(cs, 0.005)) / float(L))
    f["bw90"] = float((np.searchsorted(cs, 0.95) - np.searchsorted(cs, 0.05)) / float(L))
    k = np.ones(9) / 9.0
    Ss = np.convolve(S, k, mode="same")
    above = np.flatnonzero(Ss > Ss.max() / 2.0)
    f["bw3db"] = float((above[-1] - above[0] + 1) / float(L))
    f["snr_spec_db"] = float(10 * np.log10(Ss.max() / max(np.percentile(Ss, 10), 1e-30)))
    d = z[1:] * np.conj(z[:-1])
    fi = np.angle(d) / (2 * np.pi)
    f["f_mean"] = float(np.angle(d.sum()) / (2 * np.pi))
    q = np.percentile(fi, [10, 50, 90])
    f["f_inst_med"] = float(q[1])
    f["f_inst_spread"] = float(q[2] - q[0])
    e = p - p.mean()
    E = np.abs(np.fft.rfft(e))
    E[:3] = 0
    kk = int(E.argmax())
    f["env2_line"] = kk / float(L)
    f["env2_prom"] = float(E[kk] / max(np.median(E[3:]), 1e-30))
    for m in (2, 4):
        Z = np.abs(np.fft.fft(z ** m))
        j = int(Z.argmax())
        if j >= L // 2:
            j -= L
        f["pow%d_line" % m] = j / float(L) / m
        f["pow%d_prom" % m] = float(Z.max() / max(np.median(Z), 1e-30))
    return f


def runs(b):
    idx = np.flatnonzero(np.diff(b.astype(np.int8)) != 0) + 1
    starts = np.concatenate([[0], idx])
    ends = np.concatenate([idx, [b.size]])
    return starts, ends, b[starts]


def clean(on, min_gap=3, min_run=5):
    on = on.copy()
    s, e, v = runs(on)
    for a, b, val in zip(s, e, v):
        if (not val) and (b - a) < min_gap and a > 0 and b < on.size:
            on[a:b] = True
    s, e, v = runs(on)
    for a, b, val in zip(s, e, v):
        if val and (b - a) < min_run:
            on[a:b] = False
    return on


def pulse_measure(x):
    p = x[:, 0].astype(np.float64) ** 2 + x[:, 1].astype(np.float64) ** 2
    sm = np.convolve(p, np.ones(3) / 3.0, mode="same")
    lo, hi = np.percentile(sm, [10, 90])
    r = {"env_snr_db": float(10 * np.log10(hi / max(lo, 1e-30)))}
    e = sm - sm.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(e, 2 * e.size)) ** 2)[: e.size]
    a0, a1 = 30, min(120, e.size - 2)
    j = a0 + int(ac[a0:a1].argmax())
    den = ac[j - 1] - 2 * ac[j] + ac[j + 1]
    r["pri_ac"] = float(j + (0.5 * (ac[j - 1] - ac[j + 1]) / den if den != 0 else 0.0))
    r["pri_ac_strength"] = float(ac[j] / max(ac[0], 1e-30))
    thr = 0.5 * (lo + hi)
    on = clean(sm > thr)
    s, en, v = runs(on)
    s, en = s[v], en[v]
    r["n_det"] = int(s.size)
    r["touch_start"] = bool(on[0])
    r["touch_end"] = bool(on[-1])

    def cross(i):  # 阈值穿越点位于 i-1 与 i 之间
        a, b = sm[i - 1], sm[i]
        return (i - 1) + ((thr - a) / (b - a) if b != a else 0.5)

    rises = np.array([cross(i) for i in s if i > 0])
    falls = np.array([cross(i) for i in en if i < on.size])
    r["first_rise"] = float(rises[0]) if (rises.size and not on[0]) else np.nan
    r["spacing"] = float(np.mean(np.diff(rises))) if rises.size >= 2 else np.nan
    full = [(cross(a), cross(b)) for a, b in zip(s, en) if a > 0 and b < on.size]
    r["width"] = float(np.mean([b - a for a, b in full])) if full else np.nan
    r["last_fall"] = float(falls[-1]) if (falls.size and not on[-1]) else np.nan
    return r


def spark(v, width=128):
    v = np.asarray(v, dtype=np.float64)
    n = (v.size // width) * width
    v = v[:n].reshape(width, -1).max(axis=1)
    lo, hi = v.min(), v.max()
    if hi <= lo:
        return " " * width
    q = np.clip(((v - lo) / (hi - lo) * 8).astype(int), 0, 8)
    return "".join(BLOCKS[i] for i in q)


def fit_line(name, meas, lab, unit="s"):
    meas = np.asarray(meas, np.float64)
    lab = np.asarray(lab, np.float64)
    m = np.isfinite(meas) & np.isfinite(lab) & (meas > 0)
    if m.sum() < 20:
        P("  %s: 有效样本不足 (%d)" % (name, int(m.sum())))
        return
    x, y = meas[m], lab[m]
    P("  %s  n=%d  比值 label/实测 分位(5/25/50/75/95): %s" % (name, x.size, pct(y / x)))
    for it in range(2):
        a, b = np.polyfit(x, y, 1)
        res = y - (a * x + b)
        mad = np.median(np.abs(res - np.median(res))) + 1e-30
        if it == 0:
            keep = np.abs(res - np.median(res)) < 5 * 1.4826 * mad
            x2, y2 = x[keep], y[keep]
            P("     全部样本拟合: label = %s * 实测 + %s   R2=%.4f  残差std=%s" % (
                g6(a), g6(b), 1 - res.var() / max(y.var(), 1e-60), g6(res.std())))
            x, y = x2, y2
        else:
            P("     剔除离群后拟合(n=%d): label = %s * 实测 + %s   R2=%.4f  残差std=%s %s;  -截距/斜率=%s 采样点" % (
                x.size, g6(a), g6(b), 1 - res.var() / max(y.var(), 1e-60), g6(res.std()), unit, g6(-b / a if a else np.nan)))


# ----------------------------------------------------------------------------
# 抽样并计算特征
# ----------------------------------------------------------------------------
def sample_features(frames, groups):
    recs = []
    ex_pulse, ex_comm = [], {}
    modes = {}
    for p, df in frames.items():
        pf = pq.ParquetFile(p)
        nrg = pf.metadata.num_row_groups
        starts = np.concatenate([[0], np.cumsum([pf.metadata.row_group(i).num_rows for i in range(nrg)])])
        pick = sorted(set(np.linspace(0, nrg - 1, min(nrg, groups)).astype(int).tolist()))
        for rg in pick:
            t = pf.read_row_group(rg, columns=["iq"])
            xs, mode = iq_to_list(t.column(0))
            modes[mode] = modes.get(mode, 0) + 1
            lab = df.iloc[int(starts[rg]): int(starts[rg]) + len(xs)]
            for x, (_, r) in zip(xs, lab.iterrows()):
                f = spec_feat(x)
                f["task"] = r["task"]
                f["modulation_label"] = r["modulation_label"]
                f["protocol_label"] = r["protocol_label"]
                f["grp"] = "-"
                if r["task"] == "parameter":
                    for c in PARAMS:
                        f[c] = float(r[c])
                    if bool(r["pulse_width_valid"]):
                        f["grp"] = "pulse"
                        f.update(pulse_measure(x))
                        if len(ex_pulse) < 400:
                            ex_pulse.append((f["env_snr_db"], x.copy(), {c: float(r[c]) for c in PARAMS[3:]}, f.get("n_det")))
                    else:
                        f["grp"] = "comm"
                        key = (float(r["symbol_rate"]), float(r["bandwidth"]), float(r["frequency_shift"]))
                        if key not in ex_comm and len(ex_comm) < 400:
                            ex_comm[key] = x.copy()
                recs.append(f)
        sys.stderr.write("   特征抽样完成: %s (%d 个行组)\n" % (os.path.basename(p), len(pick)))
    P("IQ 解码路径统计 (fast=Arrow 零拷贝展平可用):", modes)
    return pd.DataFrame(recs), ex_pulse, ex_comm


def med_table(df, keys, cols):
    g = df.groupby(keys, observed=True)
    t = g[cols].median()
    t.insert(0, "n", g.size())
    return t


def part_b(sig, ex_pulse):
    pu = sig[sig["grp"] == "pulse"].copy()
    P("脉冲组抽样数: %d   长度分布: %s" % (len(pu), pu["L"].value_counts().to_dict()))
    if len(pu) == 0:
        return
    P("包络信噪比 env_snr_db (90分位功率/10分位功率) 分位(5/25/50/75/95):", pct(pu["env_snr_db"]))
    P("检测到的脉冲段数 n_det 与标签 number_of_pulses 的交叉表:")
    P(pd.crosstab(pu["number_of_pulses"], pu["n_det"].clip(upper=12)).to_string())
    P("窗口起点处已在脉冲内的比例: %.3f   窗口终点处仍在脉冲内的比例: %.3f" % (pu["touch_start"].astype(float).mean(), pu["touch_end"].astype(float).mean()))

    ok = pu[(pu["n_det"] == pu["number_of_pulses"]) & (pu["env_snr_db"] > 6)]
    P("\n### B1 用于拟合的干净样本 (n_det==标签 且 env_snr_db>6): %d 条 (占 %.1f%%)" % (len(ok), 100.0 * len(ok) / len(pu)))
    P("斜率即采样间隔 Ts (秒/采样点)；'-截距/斜率' 表示标签零点相对窗口起点的偏移（采样点）。")
    fit_line("pri         ~ 上升沿间距 spacing", ok["spacing"], ok["pri"])
    fit_line("pri         ~ 包络自相关峰 pri_ac (全部脉冲样本)", pu["pri_ac"], pu["pri"])
    fit_line("pulse_width ~ 脉冲宽度 width", ok["width"], ok["pulse_width"])
    fit_line("time_delay  ~ 首个上升沿 first_rise", ok["first_rise"], ok["time_delay"])
    P("\n### B2 一致性检查 (干净样本, 以采样点为单位)")
    P("  末脉冲下降沿 last_fall 分位(5/25/50/75/95):", pct(ok["last_fall"]))
    P("  first_rise 分位:", pct(ok["first_rise"]), "  spacing 分位:", pct(ok["spacing"]), "  width 分位:", pct(ok["width"]))
    P("  按 env_snr_db 四分位: 干净样本占比 =",
      pu.assign(ok=(pu["n_det"] == pu["number_of_pulses"])).groupby(pd.qcut(pu["env_snr_db"], 4, duplicates="drop"), observed=True)["ok"].mean().round(3).to_dict())

    P("\n### B3 脉冲组波形示例 (功率包络, 512 点压成 128 列, 每列=4 个采样点; 高/中/低信噪比各 4 条)")
    ex = sorted(ex_pulse, key=lambda t: -t[0])
    chosen = ex[:4] + ex[len(ex) // 2: len(ex) // 2 + 4] + ex[-4:]
    for snr, x, lab, ndet in chosen:
        p = x[:, 0].astype(np.float64) ** 2 + x[:, 1].astype(np.float64) ** 2
        P("  env_snr=%.1fdB n_det=%s  标签: pw=%.3fµs pri=%.3fµs td=%.3fµs N=%d" % (
            snr, ndet, lab["pulse_width"] * 1e6, lab["pri"] * 1e6, lab["time_delay"] * 1e6, int(lab["number_of_pulses"])))
        P("  |" + spark(p) + "|")


def part_c(sig, ex_comm):
    cm = sig[sig["grp"] == "comm"].copy()
    P("通信组抽样数: %d   长度分布: %s" % (len(cm), cm["L"].value_counts().to_dict()))
    if len(cm) == 0:
        return
    cols = ["rms", "snr_spec_db", "centroid", "f_mean", "f_inst_med", "f_inst_spread", "bw3db", "bw90", "bw99",
            "env2_line", "env2_prom", "pow2_line", "pow2_prom", "pow4_line", "pow4_prom"]
    P("列说明(均为每组中位数, 频率单位=周期/采样点): centroid=功率谱质心; f_mean=相邻采样相位差的加权平均(载频偏移估计);")
    P("  f_inst_spread=瞬时频率 90-10 分位差(对 FSK 反映频偏); bw3db/bw90/bw99=占用带宽; env2_line=|x|^2 谱线位置(常对应码元速率);")
    P("  powM_line=x^M 的谱线位置/M(M 次方法估载频); *_prom=谱线峰值/中位数(越大越可信)")

    P("\n### C1 按 frequency_shift 分组")
    t = med_table(cm, ["frequency_shift"], cols)
    for c in ("f_mean", "centroid", "pow2_line", "pow4_line"):
        t["fs?=shift/" + c] = t.index.to_numpy(np.float64) / t[c].to_numpy()
    t["fs?=shift/(spread/2)"] = t.index.to_numpy(np.float64) / (t["f_inst_spread"].to_numpy() / 2)
    P(t.to_string(float_format=lambda v: "%.5g" % v))

    P("\n### C2 按 (symbol_rate, bandwidth) 分组")
    t = med_table(cm, ["symbol_rate", "bandwidth"], cols)
    sr = t.index.get_level_values(0).to_numpy(np.float64)
    bw = t.index.get_level_values(1).to_numpy(np.float64)
    t["fs?=sr/env2_line"] = sr / t["env2_line"].to_numpy()
    for c in ("bw3db", "bw90", "bw99"):
        t["fs?=bw/" + c] = bw / t[c].to_numpy()
    P(t.to_string(float_format=lambda v: "%.5g" % v))

    P("\n### C3 按 (symbol_rate, bandwidth, frequency_shift 类) 分组")
    cm["fcat"] = np.where(cm["frequency_shift"].abs() <= 50, "|f|<=50", cm["frequency_shift"].astype(int).astype(str))
    P(med_table(cm, ["symbol_rate", "bandwidth", "fcat"], ["rms", "f_mean", "f_inst_spread", "bw3db", "bw99", "env2_line", "env2_prom"]
                ).to_string(float_format=lambda v: "%.5g" % v))

    P("\n### C4 env2_line 的集中度: 每个 symbol_rate 下最常见的 5 个谱线位置(乘以 L 后的频点号 : 占比)")
    for srv, g in cm.groupby("symbol_rate"):
        vc = (g["env2_line"] * g["L"]).round().astype(int).value_counts(normalize=True).head(5)
        P("  symbol_rate=%s (n=%d): %s" % (g6(srv), len(g), "  ".join("%d:%.2f" % (k, v) for k, v in vc.items())))

    P("\n### C5 小频偏能否分辨: frequency_shift 在 [-50,50] 的样本, f_mean 按标签分组的 25/50/75 分位")
    small = cm[cm["frequency_shift"].abs() <= 50]
    for fv, g in small.groupby("frequency_shift"):
        P("  shift=%5s n=%-5d f_mean: %s    pow4_line: %s" % (g6(fv), len(g), pct(g["f_mean"], (25, 50, 75)), pct(g["pow4_line"], (25, 50, 75))))

    P("\n### C6 通信组频谱示例 (对数功率谱, fftshift 后 1024 点压成 128 列, 中心=0 频; 每种三元组合 1 条, 最多 40 条)")
    for i, (key, x) in enumerate(sorted(ex_comm.items())):
        if i >= 40:
            break
        z = x[:, 0].astype(np.float64) + 1j * x[:, 1].astype(np.float64)
        S = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(z))) ** 2 + 1e-20)
        S = np.clip(S, S.max() - 50, None)
        P("  sr=%-6s bw=%-6s shift=%-5s |%s|" % (g6(key[0]), g6(key[1]), g6(key[2]), spark(S)))


def part_d(sig):
    cols = ["L", "rms", "snr_spec_db", "f_mean", "f_inst_spread", "bw3db", "bw90", "bw99", "env2_line", "env2_prom", "pow2_prom", "pow4_prom"]
    md = sig[sig["task"] == "modulation"]
    if len(md):
        P("### D1 调制任务各类别的特征中位数")
        P(med_table(md, ["modulation_label"], cols).to_string(float_format=lambda v: "%.5g" % v))
        P("\n### D2 调制任务: 每类 rms 的 5/50/95 分位 与 snr_spec_db 的 5/50/95 分位")
        for k, g in md.groupby("modulation_label"):
            P("  %-28s rms: %-36s snr_spec_db: %s" % (k, pct(g["rms"], (5, 50, 95)), pct(g["snr_spec_db"], (5, 50, 95))))
    pr = sig[sig["task"] == "protocol"]
    if len(pr):
        P("\n### D3 体制任务各类别的特征中位数")
        P(med_table(pr, ["protocol_label", "L"], cols[1:]).to_string(float_format=lambda v: "%.5g" % v))
    cm = sig[sig["grp"] == "comm"]
    if len(cm):
        P("\n### D4 参数任务通信组整体: rms 分位(5/50/95) = %s ; snr_spec_db 分位 = %s" % (
            pct(cm["rms"], (5, 50, 95)), pct(cm["snr_spec_db"], (5, 50, 95))))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="C01 数据根目录")
    ap.add_argument("--out", default="eda2_report.txt")
    ap.add_argument("--groups", type=int, default=40, help="每个文件抽多少个行组(每组约 1024 条)做信号分析")
    ap.add_argument("--skip-full-length", action="store_true", help="跳过全量长度扫描(该步需读完全部 IQ, 约几分钟)")
    args = ap.parse_args()
    t0 = time.time()
    P("C01 第二轮核实报告  生成时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
    P("python %s | numpy %s | pandas %s | pyarrow %s" % (sys.version.split()[0], np.__version__, pd.__version__, pa.__version__))
    try:
        paths = find_train_parquets(os.path.abspath(args.root))
        P("训练 Parquet:", [os.path.basename(p) for p in paths])
        frames = safe("0. 读取标签列", load_labels, paths) or {}
        if frames:
            safe("A. 标签联合分布 (全量)", part_a, frames)
            res = safe("信号抽样与特征计算", sample_features, frames, args.groups)
            if res is not None:
                sig, ex_pulse, ex_comm = res
                safe("B. 脉冲组: 采样间隔与参数定义", part_b, sig, ex_pulse)
                safe("C. 通信组: 采样率与 frequency_shift 含义", part_c, sig, ex_comm)
                safe("D. 调制/体制类别特征", part_d, sig)
            if not args.skip_full_length:
                safe("E. 序列长度全量统计", part_len, frames)
    finally:
        P("\n总耗时 %.1f 秒" % (time.time() - t0))
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(OUT.getvalue())
        sys.stderr.write("报告已写入 %s (%.1f KB)\n" % (os.path.abspath(args.out), len(OUT.getvalue().encode("utf-8")) / 1024.0))


if __name__ == "__main__":
    main()
