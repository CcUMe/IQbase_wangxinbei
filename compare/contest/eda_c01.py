#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C01 数据摸底脚本：只读数据，输出一份文本报告（默认 eda_report.txt）。

用法:
    python eda_c01.py /path/to/C01
    python eda_c01.py /path/to/C01 --out eda_report.txt --groups-per-file 12 --rows-per-group 512

依赖: numpy, pandas, pyarrow
报告内容: 目录树、小文件原文、Parquet 结构、标签/参数全量统计、信号抽样统计。
报告里不含完整 IQ 数据，只有统计量和每个任务几条样本的前几个采样点。
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

OUT = io.StringIO()
MAXCAT = 80
PCTS = [0, 1, 5, 25, 50, 75, 95, 99, 100]
TEXT_EXT = (".json", ".jsonl", ".txt", ".md", ".csv", ".yaml", ".yml")


def P(*a):
    s = " ".join(str(x) for x in a)
    for line in s.split("\n"):
        if len(line) > 3000:
            line = line[:3000] + " ...[截断]"
        OUT.write(line + "\n")


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


def fmt(v):
    if v is None:
        return "None"
    try:
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "nan"
            return "%.6g" % v
    except Exception:
        pass
    return str(v)


# ----------------------------------------------------------------------------
# 1. 目录与小文件
# ----------------------------------------------------------------------------
def list_tree(root):
    parquets, texts = [], []
    for d, _, files in sorted(os.walk(root)):
        for f in sorted(files):
            p = os.path.join(d, f)
            try:
                sz = os.path.getsize(p)
            except OSError:
                sz = -1
            P("%14d  %s" % (sz, os.path.relpath(p, root)))
            if f.lower().endswith(".parquet"):
                parquets.append(p)
            elif f.lower().endswith(TEXT_EXT):
                texts.append(p)
    return parquets, texts


def dump_texts(root, texts):
    for p in texts:
        name = os.path.basename(p).lower()
        limit = 40000 if ("schema" in name or name.endswith(".txt") or name.endswith(".md")) else 6000
        P("\n----- %s (前 %d 字符) -----" % (os.path.relpath(p, root), limit))
        raw = open(p, "rb").read(limit * 4)
        txt = None
        for enc in ("utf-8-sig", "gb18030"):
            for cut in (0, 1, 2, 3):  # 读到一半的多字节字符会解码失败，去掉末尾几个字节重试
                try:
                    txt = raw[:len(raw) - cut].decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if txt is not None:
                break
        if txt is None:
            txt = raw.decode("utf-8", errors="replace")
        P(txt[:limit])


# ----------------------------------------------------------------------------
# 2. Parquet 结构
# ----------------------------------------------------------------------------
def find_iq_col(pf):
    names = pf.schema_arrow.names
    for cand in ("iq", "IQ", "iq_data", "signal", "data", "samples"):
        if cand in names:
            return cand
    first = next(pf.iter_batches(batch_size=1))
    for n in names:
        t = pf.schema_arrow.field(n).type
        if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t) \
                or pa.types.is_struct(t) or pa.types.is_binary(t) or pa.types.is_large_binary(t):
            v = first.column(first.schema.get_field_index(n))[0].as_py()
            try:
                if v is not None and np.asarray(v, dtype=object).size >= 64 or isinstance(v, bytes):
                    return n
            except Exception:
                pass
    return None


def parquet_meta(root, parquets):
    info = {}
    for p in parquets:
        pf = pq.ParquetFile(p)
        md = pf.metadata
        P("\n----- %s -----" % os.path.relpath(p, root))
        P("行数=%d  行组数=%d  列数=%d  created_by=%s" % (md.num_rows, md.num_row_groups, md.num_columns, md.created_by))
        rg_rows = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
        P("各行组行数(前20):", rg_rows[:20])
        P("Arrow schema:")
        P(pf.schema_arrow.to_string(show_schema_metadata=False))
        meta = pf.schema_arrow.metadata or {}
        for k, v in meta.items():
            P("schema metadata [%s]: %s" % (k.decode("utf-8", "replace"), v.decode("utf-8", "replace")[:3000]))
        try:
            rg0 = md.row_group(0)
            for j in range(rg0.num_columns):
                c = rg0.column(j)
                P("  列块: %s  physical=%s  compression=%s  压缩后=%d  未压缩=%d" % (
                    c.path_in_schema, c.physical_type, c.compression, c.total_compressed_size, c.total_uncompressed_size))
        except Exception as e:
            P("  (列块信息读取失败: %s)" % e)
        iq = find_iq_col(pf)
        P("识别到的 IQ 列:", iq)
        info[p] = dict(iq=iq, num_rows=md.num_rows, rg_rows=rg_rows, names=pf.schema_arrow.names)
    return info


# ----------------------------------------------------------------------------
# 3. 标签表（除 IQ 外的所有列，全量）
# ----------------------------------------------------------------------------
SEQ = (list, tuple, np.ndarray)


def is_objlike(s):
    return s.dtype == object or isinstance(s.dtype, pd.StringDtype)


def _try_json(t):
    if isinstance(t, str) and t.strip()[:1] in ("{", "["):
        try:
            return json.loads(t)
        except Exception:
            return t
    return t


def _isnull(x):
    return x is None or x is pd.NA or (isinstance(x, float) and x != x)


def expand_df(df, notes):
    """把 JSON 字符串 / 字典 / 定长列表 列展开成普通列，便于统计。"""
    done = set()
    for _ in range(3):
        changed = False
        for col in list(df.columns):
            if col in done or col.startswith("__") or not is_objlike(df[col]):
                continue
            vals = [None if _isnull(x) else x for x in df[col].tolist()]
            probe = [x for x in vals[:20000] if x is not None][:2000]
            if not probe:
                continue
            if any(isinstance(x, str) and x.strip()[:1] in ("{", "[") for x in probe):
                vals = [_try_json(x) for x in vals]
                notes.append("列 %s 含 JSON 字符串，已解析" % col)
            isd = [isinstance(x, dict) for x in vals]
            isl = [isinstance(x, SEQ) for x in vals]
            if any(isd):
                sub = pd.json_normalize([x if d else {} for x, d in zip(vals, isd)])
                sub.columns = ["%s.%s" % (col, c) for c in sub.columns]
                sub.index = df.index
                rest = [None if d else x for x, d in zip(vals, isd)]
                df = df.drop(columns=[col])
                if any(x is not None for x in rest):
                    df[col] = pd.Series(rest, index=df.index, dtype=object)
                    notes.append("列 %s 同时含字典和非字典值；非字典值保留在原列" % col)
                df = pd.concat([df, sub], axis=1)
                notes.append("列 %s 的字典已展开为: %s" % (col, list(sub.columns)))
                changed = True
            elif any(isl):
                lens = [len(x) if l else -1 for x, l in zip(vals, isl)]
                good = set(n for n in lens if n >= 0)
                df[col + ".__len"] = lens
                if len(good) == 1 and 0 < list(good)[0] <= 16:
                    n = list(good)[0]
                    for i in range(n):
                        df["%s[%d]" % (col, i)] = pd.Series(
                            [x[i] if l else None for x, l in zip(vals, isl)], index=df.index, dtype=object)
                    notes.append("列 %s 是定长列表(长度 %d)，已按位置展开" % (col, n))
                df[col] = pd.Series(
                    [json.dumps(np.asarray(x).tolist(), ensure_ascii=False, default=str) if l else x
                     for x, l in zip(vals, isl)], index=df.index, dtype=object)
                done.add(col)
                changed = True
            else:
                done.add(col)
        if not changed:
            break
    for col in df.columns:
        if df[col].dtype == object and not col.startswith("__"):
            nn = df[col].dropna()
            if len(nn) == 0 or isinstance(nn.iloc[0], (str, bool, np.bool_)):
                continue
            conv = pd.to_numeric(df[col], errors="coerce")
            if conv.notna().sum() == len(nn):
                df[col] = conv
    return df


def load_labels(paths, info):
    frames = []
    notes = []
    for p in paths:
        iq = info[p]["iq"]
        cols = [n for n in info[p]["names"] if n != iq]
        t = pq.read_table(p, columns=cols)
        for _ in range(4):
            if any(pa.types.is_struct(f.type) for f in t.schema):
                t = t.flatten()
        df = t.to_pandas()
        df["__file"] = os.path.basename(p)
        df["__row"] = np.arange(len(df))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = expand_df(df, notes)
    for n in notes:
        P("注:", n)
    return df


def is_num(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def describe_series(s, name, indent="  "):
    n = len(s)
    x = s.dropna()
    nn = len(x)
    head = "%s- %s  dtype=%s  非空=%d/%d" % (indent, name, s.dtype, nn, n)
    if nn == 0:
        P(head + "  (全空)")
        return
    if not is_num(x):
        xs = x.astype(str)
        nu = xs.nunique()
        P(head + "  唯一值=%d" % nu)
        if nu <= MAXCAT:
            vc = xs.value_counts()
            for k, c in vc.items():
                P("%s    %-40s %9d  (%.2f%%)" % (indent, repr(k), c, 100.0 * c / nn))
        else:
            P("%s    示例: %s" % (indent, [v[:120] for v in xs.iloc[:5].tolist()]))
            ln = xs.str.len()
            P("%s    字符串长度: min=%d median=%d max=%d" % (indent, ln.min(), ln.median(), ln.max()))
        return
    xv = x.to_numpy(dtype=np.float64)
    fin = np.isfinite(xv)
    nonfin = int((~fin).sum())
    xv = xv[fin]
    if xv.size == 0:
        P(head + "  (无有限值, 非有限=%d)" % nonfin)
        return
    uniq, cnt = np.unique(xv, return_counts=True)
    nu = uniq.size
    P(head + "  唯一值=%d  非有限=%d  全为整数值=%s" % (nu, nonfin, bool(np.all(xv == np.round(xv)))))
    q = np.percentile(xv, PCTS)
    P("%s    分位 %s: %s" % (indent, PCTS, " ".join(fmt(v) for v in q)))
    P("%s    mean=%s std=%s 零值数=%d 负值数=%d" % (indent, fmt(xv.mean()), fmt(xv.std()), int((xv == 0).sum()), int((xv < 0).sum())))
    if nu <= 40:
        P("%s    全部取值(值:次数): %s" % (indent, "  ".join("%s:%d" % (fmt(u), c) for u, c in zip(uniq, cnt))))
    else:
        order = np.argsort(-cnt)[:15]
        cover = cnt[order].sum() / float(xv.size)
        P("%s    最常见15个取值(值:次数): %s" % (indent, "  ".join("%s:%d" % (fmt(uniq[i]), cnt[i]) for i in order)))
        P("%s    这15个取值覆盖率=%.2f%%  (接近100%% 说明该量是离散档位)" % (indent, 100 * cover))
        d = np.diff(uniq)
        P("%s    相邻唯一值间距: min=%s median=%s max=%s" % (indent, fmt(d.min()), fmt(np.median(d)), fmt(d.max())))


def find_task_col(df):
    if "task" in df.columns:
        return "task"
    for c in df.columns:
        if "task" in c.lower():
            return c
    return None


def split_cols(df, task_col):
    hidden = {"__file", "__row", task_col}
    vcols = [c for c in df.columns if c not in hidden and not c.endswith(".__len")
             and any(k in c.lower() for k in ("valid", "mask", "flag", "有效"))]
    num = [c for c in df.columns if c not in hidden and c not in vcols and is_num(df[c]) and not c.endswith(".__len")]
    cat = []
    for c in df.columns:
        if c in hidden or c in vcols or c in num or c.endswith(".__len"):
            continue
        nn = df[c].dropna()
        if len(nn) and 2 <= nn.astype(str).nunique() <= MAXCAT:
            cat.append(c)
    return vcols, num, cat


def label_report(df):
    task_col = find_task_col(df)
    P("总行数=%d  列=%s" % (len(df), [c for c in df.columns if not c.startswith("__")]))
    P("任务列:", task_col)
    P("\n### 3.1 全体样本逐列统计")
    for c in df.columns:
        if c == "__row":
            continue
        describe_series(df[c], c)

    for c in df.columns:
        if "id" in c.lower() and "valid" not in c.lower() and is_objlike(df[c]):
            dup = int(df[c].duplicated().sum())
            P("\n标识列 %s: 唯一=%d 重复=%d 示例=%s" % (c, df[c].nunique(), dup, df[c].iloc[:3].tolist()))

    vcols, num, cat = split_cols(df, task_col)
    P("\n识别: 有效性列=%s\n      数值列=%s\n      类别列=%s" % (vcols, num, cat))

    groups = [("(全体)", df)] if task_col is None else [(str(k), g) for k, g in df.groupby(task_col, dropna=False)]
    if task_col is not None:
        P("\n### 3.2 任务 x 文件 样本数")
        P(pd.crosstab(df[task_col].astype(str), df["__file"]).to_string())

    for tname, g in groups:
        P("\n### 3.3 任务 = %s  (样本数 %d)" % (tname, len(g)))
        P("  [原始样例 2 条]")
        for _, r in g.head(2).iterrows():
            P("   ", json.dumps({k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()
                                 if not k.startswith("__")}, ensure_ascii=False, default=str))
        P("  [逐列统计, 仅列出该任务下非空的列]")
        for c in g.columns:
            if c in ("__row", "__file", task_col) or g[c].notna().sum() == 0:
                continue
            describe_series(g[c], c, indent="    ")

        allc = [c for c in g.columns if not c.startswith("__") and c != task_col][:60]
        m = g[allc].notna().to_numpy()
        codes = np.zeros(len(g), dtype=object)
        for j in range(m.shape[1]):
            codes = codes * 2 + m[:, j].astype(int)
        pat = pd.Series(codes).value_counts().head(25)
        P("  [非空模式 Top25] (1=非空) 列顺序=%s" % allc)
        for k, c in pat.items():
            P("    %s  %9d  (%.2f%%)" % (format(int(k), "0%db" % len(allc)), c, 100.0 * c / len(g)))

        gv = [c for c in vcols if g[c].notna().sum() > 0]
        gnum = [c for c in num if g[c].notna().sum() > 0]
        if gv:
            vp = g[gv[0]].astype(str)
            for c in gv[1:]:
                vp = vp + "|" + g[c].astype(str)
            vc = vp.value_counts()
            P("  [有效性组合 Top40] 列顺序=%s  (共 %d 种组合)" % (gv, len(vc)))
            for k, c in vc.head(40).items():
                P("    %s  %9d  (%.2f%%)" % (k, c, 100.0 * c / len(g)))
            if gnum:
                P("  [各有效性组合下的数值列: 非空数/唯一值数/min/median/max] (Top12 组合)")
                for k in vc.head(12).index:
                    sub = g[vp == k]
                    P("    组合 %s  (n=%d)" % (k, len(sub)))
                    for c in gnum:
                        x = sub[c].dropna()
                        if len(x) == 0:
                            P("      %-28s 全空" % c)
                        else:
                            P("      %-28s n=%d uniq=%d min=%s med=%s max=%s" % (
                                c, len(x), x.nunique(), fmt(x.min()), fmt(x.median()), fmt(x.max())))
        gcat = [c for c in cat if g[c].notna().sum() > 0 and g[c].astype(str).nunique() >= 2]
        if gnum and gcat:
            P("  [数值列按类别列分组: n/uniq/min/median/max]")
            for cc in gcat:
                for c in gnum:
                    if g[c].nunique() < 2:
                        continue
                    P("    -- %s 按 %s 分组" % (c, cc))
                    for k, sub in g.groupby(g[cc].astype(str)):
                        x = sub[c].dropna()
                        if len(x):
                            P("       %-32s n=%-8d uniq=%-7d min=%-12s med=%-12s max=%s" % (
                                k[:32], len(x), x.nunique(), fmt(x.min()), fmt(x.median()), fmt(x.max())))
        if len(gcat) >= 2:
            P("  [类别列两两交叉表]")
            for i in range(len(gcat)):
                for j in range(i + 1, len(gcat)):
                    ct = pd.crosstab(g[gcat[i]].astype(str), g[gcat[j]].astype(str))
                    if ct.shape[0] * ct.shape[1] <= 2500:
                        P("    -- %s x %s" % (gcat[i], gcat[j]))
                        P(ct.to_string())
        if len(gnum) >= 2:
            P("  [数值列 Spearman 相关]")
            P(g[gnum].rank().corr().round(2).to_string())
    return task_col


# ----------------------------------------------------------------------------
# 4. 信号抽样
# ----------------------------------------------------------------------------
def to_complex(v):
    if isinstance(v, dict):
        v = list(v.values())
    if isinstance(v, (bytes, bytearray)):
        a = np.frombuffer(v, dtype=np.float32).astype(np.float64)
        return a[0::2] + 1j * a[1::2], "bytes(len=%d)->按float32交织解析" % len(v)
    a = np.asarray(v, dtype=np.float64)
    shp = str(a.shape)
    if a.ndim == 2 and a.shape[1] == 2:
        return a[:, 0] + 1j * a[:, 1], shp + " [L,2]"
    if a.ndim == 2 and a.shape[0] == 2:
        return a[0] + 1j * a[1], shp + " [2,L]"
    if a.ndim == 1:
        return a[0::2] + 1j * a[1::2], shp + " 一维,按交织解析(存疑)"
    a = a.reshape(-1, 2)
    return a[:, 0] + 1j * a[:, 1], shp + " 其他形状,强制reshape(存疑)"


def sample_rows(path, iq, groups_per_file, rows_per_group):
    """产出 (文件内行号, 原始IQ对象)。行组足够多时抽行组，否则顺序扫描按步长取 batch。"""
    pf = pq.ParquetFile(path)
    md = pf.metadata
    nrg = md.num_row_groups
    if nrg >= max(4, groups_per_file // 2):
        starts = np.concatenate([[0], np.cumsum([md.row_group(i).num_rows for i in range(nrg)])])
        pick = sorted(set(np.linspace(0, nrg - 1, min(nrg, groups_per_file)).astype(int).tolist()))
        for rg in pick:
            b = next(pf.iter_batches(batch_size=rows_per_group, row_groups=[rg], columns=[iq]))
            col = b.column(0).to_pylist()
            for i, v in enumerate(col):
                yield int(starts[rg]) + i, v
    else:
        total_batches = max(1, int(np.ceil(md.num_rows / float(rows_per_group))))
        pick = set(np.linspace(0, total_batches - 1, min(total_batches, groups_per_file)).astype(int).tolist())
        off = 0
        for bi, b in enumerate(pf.iter_batches(batch_size=rows_per_group, columns=[iq])):
            if bi in pick:
                col = b.column(0).to_pylist()
                for i, v in enumerate(col):
                    yield off + i, v
            off += b.num_rows


def features(x):
    L = x.size
    f = {"L": L}
    a = np.abs(x)
    p = a * a
    rms = float(np.sqrt(p.mean())) if L else 0.0
    f["rms"] = rms
    f["max_abs"] = float(a.max()) if L else 0.0
    f["papr_db"] = float(10 * np.log10(p.max() / p.mean())) if rms > 0 else np.nan
    f["dc_ratio"] = float(abs(x.mean()) / rms) if rms > 0 else np.nan
    f["frac_zero"] = float((a == 0).mean())
    nz = np.flatnonzero(a != 0)
    f["lead_zero"] = int(nz[0]) if nz.size else L
    f["trail_zero"] = int(L - 1 - nz[-1]) if nz.size else L
    f["nonfinite"] = int((~np.isfinite(a)).sum())
    if rms <= 0 or f["nonfinite"]:
        return f
    S = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    fr = (np.arange(L) - L // 2) / float(L)
    tot = S.sum()
    srt = np.sort(S)[::-1]
    cs = np.cumsum(srt) / tot
    f["occ90"] = float((np.searchsorted(cs, 0.90) + 1) / float(L))
    f["occ99"] = float((np.searchsorted(cs, 0.99) + 1) / float(L))
    f["centroid"] = float((fr * S).sum() / tot)
    f["peak_freq"] = float(fr[int(S.argmax())])
    f["spec_peak_to_median_db"] = float(10 * np.log10(S.max() / max(np.median(S), 1e-30)))
    sm = np.convolve(a, np.ones(4) / 4.0, mode="same")
    on = sm > 0.5 * sm.max()
    f["duty"] = float(on.mean())
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    rise = np.flatnonzero(d == 1)
    fall = np.flatnonzero(d == -1)
    f["n_runs"] = int(rise.size)
    f["first_on"] = int(rise[0]) if rise.size else -1
    f["longest_run"] = int((fall - rise).max()) if rise.size else 0
    f["median_run"] = float(np.median(fall - rise)) if rise.size else np.nan
    f["edge_spacing"] = float(np.median(np.diff(rise))) if rise.size >= 2 else np.nan
    return f


def pct_table(df, cols, by, pcts=(5, 50, 95)):
    rows = []
    for k, g in df.groupby(by):
        r = {"组": str(k)[:36], "n": len(g)}
        for c in cols:
            x = g[c].dropna().to_numpy(dtype=np.float64)
            r[c] = "/".join(fmt(v) for v in np.percentile(x, pcts)) if x.size else "-"
        rows.append(r)
    return pd.DataFrame(rows).to_string(index=False)


def signal_report(paths, info, labels, task_col, groups_per_file, rows_per_group):
    recs, shapes, hashes, examples = [], {}, {}, {}
    for p in paths:
        iq = info[p]["iq"]
        if iq is None:
            P("跳过(未识别到 IQ 列):", p)
            continue
        base = os.path.basename(p)
        n = 0
        for row, v in sample_rows(p, iq, groups_per_file, rows_per_group):
            try:
                x, shp = to_complex(v)
            except Exception as e:
                shapes["转换失败: %s" % str(e)[:80]] = shapes.get("转换失败: %s" % str(e)[:80], 0) + 1
                continue
            shapes[shp] = shapes.get(shp, 0) + 1
            f = features(x)
            f["__file"], f["__row"] = base, row
            f["__hash"] = hashlib.md5(np.round(x[:256], 6).astype(np.complex64).tobytes()).hexdigest()
            recs.append(f)
            if n < 3:
                examples.setdefault(base, []).append((row, np.round(np.c_[x.real, x.imag][:6], 6).tolist()))
            n += 1
        P("%s: 抽样 %d 条" % (base, n))
    if not recs:
        P("没有抽到任何信号")
        return
    sig = pd.DataFrame(recs)
    P("\nIQ 原始存储形状统计:", shapes)
    P("前几条样本的前 6 个采样点 [I,Q]:")
    for b, lst in examples.items():
        for row, pts in lst:
            P("  %s 行%d: %s" % (b, row, pts))

    df = sig.merge(labels, on=["__file", "__row"], how="left")
    tcol = task_col if task_col in df.columns else None
    if tcol is None:
        df["__task"] = "(全体)"
        tcol = "__task"
    df[tcol] = df[tcol].astype(str)

    P("\n### 4.1 长度 x 任务 (抽样)")
    P(pd.crosstab(df["L"], df[tcol]).to_string())
    P("\n### 4.2 重复检查 (抽样内, 按前256点哈希)")
    dup = df[df["__hash"].duplicated(keep=False)]
    P("抽样数=%d  哈希重复的样本数=%d" % (len(df), len(dup)))
    if len(dup):
        P("重复样本的任务分布:", dup[tcol].value_counts().to_dict())
        P("同一哈希跨任务的组数:", int((dup.groupby("__hash")[tcol].nunique() > 1).sum()))

    feat = [c for c in ["rms", "max_abs", "papr_db", "dc_ratio", "frac_zero", "lead_zero", "trail_zero",
                        "occ90", "occ99", "centroid", "peak_freq", "spec_peak_to_median_db",
                        "duty", "n_runs", "first_on", "longest_run", "median_run", "edge_spacing"] if c in df.columns]
    P("\n### 4.3 信号特征按任务 (每格为 5/50/95 分位)")
    P("说明: occ90/occ99=集中90%/99%功率所需频点占比; centroid/peak_freq 为归一化频率[-0.5,0.5);")
    P("      duty=包络超过半峰值的占比; n_runs=包络超阈值的段数; first_on/longest_run/edge_spacing 单位为采样点")
    P(pct_table(df, feat, tcol))
    P("非有限值样本数:", int((df["nonfinite"] > 0).sum()), " 全零样本数:", int((df["rms"] == 0).sum()))

    vcols, num, cat = split_cols(labels, task_col)
    for tname, g in df.groupby(tcol):
        gcat = [c for c in cat if c in g.columns and g[c].notna().sum() > 0 and g[c].astype(str).nunique() >= 2]
        for cc in gcat:
            P("\n### 4.4 任务=%s 按 %s 分组: 长度分布与特征中位数" % (tname, cc))
            P(pd.crosstab(g[cc].astype(str), g["L"]).to_string())
            key = [c for c in ["rms", "papr_db", "occ90", "occ99", "centroid", "spec_peak_to_median_db",
                               "duty", "n_runs", "longest_run"] if c in g.columns]
            P(pct_table(g, key, g[cc].astype(str), pcts=(50,)))

        gnum = [c for c in num if c in g.columns and g[c].notna().sum() > 0 and g[c].nunique() > 1]
        if gnum:
            P("\n### 4.5 任务=%s: 标签数值列 与 信号特征 的 Spearman 相关" % tname)
            fcols = [c for c in ["L", "rms", "occ90", "occ99", "centroid", "peak_freq", "duty", "n_runs",
                                 "first_on", "longest_run", "median_run", "edge_spacing"] if c in g.columns]
            rk = g[gnum + fcols].rank()
            P(rk.corr().loc[gnum, fcols].round(2).to_string())

            P("\n### 4.6 任务=%s: 推断采样率/采样间隔 (标签值 ÷ 信号实测量, 各分位; 若分位彼此接近说明采样率固定)" % tname)

            def col_like(key):
                for c in gnum:
                    if key in c.lower():
                        return c
                return None

            pairs = [("bandwidth", "occ99", "≈ 采样率 fs"), ("bandwidth", "occ90", "≈ 采样率 fs"),
                     ("frequency_shift", "centroid", "≈ 采样率 fs"), ("frequency_shift", "peak_freq", "≈ 采样率 fs"),
                     ("pulse_width", "median_run", "≈ 采样间隔 Ts"), ("pulse_width", "longest_run", "≈ 采样间隔 Ts"),
                     ("pri", "edge_spacing", "≈ 采样间隔 Ts"), ("time_delay", "first_on", "≈ 采样间隔 Ts")]
            for key, fc, meaning in pairs:
                c = col_like(key)
                if c is None or fc not in g.columns:
                    continue
                for Lval, gg in [("全部", g)] + [("L=%d" % k, s) for k, s in g.groupby("L")]:
                    m = gg[c].notna() & gg[fc].notna() & (gg[fc].abs() > 1e-9) & (gg[c] != 0)
                    if m.sum() < 5:
                        continue
                    r = (gg.loc[m, c] / gg.loc[m, fc]).to_numpy(dtype=np.float64)
                    P("  %-18s / %-13s %-14s [%s] n=%d  分位5/25/50/75/95: %s" % (
                        c[-18:], fc, meaning, Lval, int(m.sum()), " ".join(fmt(v) for v in np.percentile(r, [5, 25, 50, 75, 95]))))
            c = col_like("number_of_pulses") or col_like("pulses")
            if c is not None:
                P("  -- %s 与 实测包络段数 n_runs 的交叉表" % c)
                ct = pd.crosstab(g[c], g["n_runs"].clip(upper=20))
                if ct.shape[0] * ct.shape[1] <= 1500:
                    P(ct.to_string())


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="C01 数据根目录")
    ap.add_argument("--out", default="eda_report.txt")
    ap.add_argument("--groups-per-file", type=int, default=12, help="每个 Parquet 文件抽多少段")
    ap.add_argument("--rows-per-group", type=int, default=512, help="每段取多少行")
    ap.add_argument("--skip-signal", action="store_true", help="跳过信号抽样(只看标签)")
    args = ap.parse_args()
    t0 = time.time()
    root = os.path.abspath(args.root)
    P("C01 数据摸底报告  生成时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
    P("root =", root)
    P("python %s | numpy %s | pandas %s | pyarrow %s" % (sys.version.split()[0], np.__version__, pd.__version__, pa.__version__))
    try:
        res = safe("1. 目录树 (字节数, 相对路径)", list_tree, root)
        parquets, texts = res if res else ([], [])
        safe("2. 文本/JSON 小文件原文", dump_texts, root, texts)
        info = safe("3.0 Parquet 文件结构", parquet_meta, root, parquets) or {}
        by_dir = {}
        for p in parquets:
            if p in info:
                by_dir.setdefault(os.path.dirname(p), []).append(p)
        for d, paths in by_dir.items():
            rel = os.path.relpath(d, root)
            labels = safe("3. 标签/参数全量统计  [%s]" % rel, load_labels, paths, info)
            if labels is None:
                continue
            task_col = safe("3. 标签/参数明细  [%s]" % rel, label_report, labels)
            if not args.skip_signal:
                safe("4. 信号抽样统计  [%s]" % rel, signal_report, paths, info, labels, task_col,
                     args.groups_per_file, args.rows_per_group)
    finally:
        P("\n总耗时 %.1f 秒" % (time.time() - t0))
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(OUT.getvalue())
        sys.stderr.write("报告已写入 %s (%.1f KB)\n" % (os.path.abspath(args.out), len(OUT.getvalue().encode("utf-8")) / 1024.0))


if __name__ == "__main__":
    main()
