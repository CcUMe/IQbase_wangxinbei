"""Stream the original C01 Parquet files and write a reproducible audit/index.

No source data is modified. This is an audit, not a fitted training scaler or split.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


TARGETS = ["symbol_rate", "bandwidth", "frequency_shift", "pulse_width",
           "pri", "time_delay", "number_of_pulses"]
TASK_BITS = {"modulation": 1, "protocol": 2, "parameter": 4}


def waveform_hash(length, raw):
    h = hashlib.blake2b(digest_size=16)
    h.update(struct.pack("<I", length))
    h.update(raw)
    return h.digest()


def analyze_manifest(out):
    """Inspect hash groups without rereading waveforms or fitting a split."""
    table = pq.read_table(out / "sample_manifest.parquet")
    counts = pc.value_counts(table["iq_blake2b_128"])
    duplicate_hashes = pc.filter(counts.field("values"), pc.greater(counts.field("counts"), 1))
    duplicate_rows = table.filter(pc.is_in(table["iq_blake2b_128"], value_set=duplicate_hashes))
    pq.write_table(duplicate_rows, out / "duplicate_rows.parquet", compression="zstd")
    groups = defaultdict(list)
    for row in duplicate_rows.to_pylist():
        groups[row["iq_blake2b_128"]].append(row)
    combinations, repeated, conflicts = Counter(), Counter(), []
    for digest, rows in groups.items():
        bytask = defaultdict(list)
        for row in rows:
            bytask[row["task"]].append(row)
        combinations["+".join(sorted(bytask))] += 1
        for task, records in bytask.items():
            if len(records) > 1:
                repeated[task] += 1
            if task in ["modulation", "protocol"]:
                signatures = {r[f"{task}_label"] for r in records}
            else:
                signatures = {tuple((r[k + "_valid"], r[k] if r[k + "_valid"] else None)
                                    for k in TARGETS) for r in records}
            if len(signatures) > 1:
                conflicts.append({"hash": digest.hex(), "task": task,
                                  "sample_ids": [r["sample_id"] for r in records]})
    details = {"task_combinations": dict(combinations),
               "within_task_repeated_groups": dict(repeated),
               "within_task_label_conflict_groups": len(conflicts),
               "label_conflict_examples": conflicts[:10]}
    tuples = {}
    parameters = table.filter(pc.equal(table["task"], "parameter"))
    for kind, fields in [("communication", TARGETS[:3]), ("radar", TARGETS[3:])]:
        subset = parameters.filter(parameters[fields[0] + "_valid"])
        grouped = subset.group_by(fields).aggregate([("sample_id", "count")])
        histogram = pc.value_counts(grouped["sample_id_count"]).to_pylist()
        tuples[kind] = {"unique_combinations": grouped.num_rows,
                        "multiplicity_histogram": {str(r["values"]): r["counts"] for r in histogram}}
        if kind == "communication":
            pq.write_table(grouped.sort_by([(k, "ascending") for k in fields]),
                           out / "communication_grid.parquet", compression="zstd")
    return details, tuples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    root, out = args.data_root.resolve(), args.output_dir.resolve()
    if out == root or root in out.parents:
        parser.error("Audit output must be outside the original dataset directory")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    schema = json.loads((root / "rf_signal_benchmark_train_v1.0/task_schema.json").read_text())
    sample_path = root / "rf_signal_sample_io_v1.0/rf_signal_sample_io_input_v1.0.parquet"
    try:
        samples = pq.read_table(sample_path).to_pylist()
    except OSError as exc:
        raise SystemExit(f"Cannot read official sample with PyArrow {pa.__version__}: {exc}. "
                         "Use a compatible reader; PyArrow 25.0.0 was verified locally.")
    sample_by_id = {r["sample_id"]: r for r in samples}
    sample_dir = sample_path.parent
    expected = [json.loads(line) for line in
                (sample_dir / "rf_signal_sample_io_expected_output_v1.0.jsonl").read_text().splitlines()]
    truth = json.loads((sample_dir / "rf_signal_sample_io_truth_v1.0.json").read_text())
    if len(samples) != len(expected):
        raise ValueError("Sample input/expected-output row counts differ")
    for row, result in zip(samples, expected):
        task, sid = row["task"], row["sample_id"]
        target = truth[task][sid]
        wanted = {k: target[k] for k in TARGETS} if task == "parameter" else target
        if task != result["task"] or sid != result["sample_id"] or wanted != result["prediction"]:
            raise ValueError(f"Sample input/expected-output/truth mismatch: {sid}")
    sample_hashes = defaultdict(list)
    for r in samples:
        arr = np.asarray(r["iq"], dtype="<f4")
        sample_hashes[waveform_hash(len(arr), arr.tobytes())].append(r["sample_id"])
    sample_matches = {sid: {"id_matches": [], "waveform_hash_matches": 0}
                      for sid in sample_by_id}
    tasks, labels, lengths = Counter(), defaultdict(Counter), Counter()
    class_lengths, patterns, pattern_lengths = Counter(), Counter(), Counter()
    params = {name: [] for name in TARGETS}
    invalid_nonzero, invalid_null = Counter(), Counter()
    anomalies = Counter()
    ids, hashes = set(), {}
    task_runs, files = [], []
    previous_task = None
    manifest_writer = None
    total = 0
    rms_values = defaultdict(list)
    try:
        for path in sorted((root / "rf_signal_benchmark_train_v1.0").glob("*.parquet")):
            pf = pq.ParquetFile(path)
            per_file_tasks, row_start = Counter(), 0
            files.append({"path": str(path), "bytes": path.stat().st_size,
                          "rows": pf.metadata.num_rows, "row_groups": pf.num_row_groups,
                          "arrow_schema": str(pf.schema_arrow)})
            for batch in pf.iter_batches(batch_size=args.batch_size):
                table = pa.Table.from_batches([batch])
                scalar = table.drop(["iq"])
                d = scalar.to_pydict()
                task_list, sid_list = d["task"], d["sample_id"]
                n = len(task_list)
                for sid in sid_list:
                    anomalies["duplicate_sample_id"] += int(sid in ids)
                    ids.add(sid)
                tasks.update(task_list)
                per_file_tasks.update(task_list)
                for local, task in enumerate(task_list):
                    if task != previous_task:
                        task_runs.append({"task": task, "global_row": total + local,
                                          "file": path.name, "file_row": row_start + local})
                        previous_task = task
                for task in ["modulation", "protocol"]:
                    for t, label, valid in zip(task_list, d[f"{task}_label"], d[f"{task}_valid"]):
                        anomalies["classification_validity_mismatch"] += int(bool(valid) != (t == task))
                        if valid:
                            labels[task][label] += 1
                            anomalies["unknown_class_label"] += int(label not in schema["classification_labels"][task])
                valid_arrays = []
                for name in TARGETS:
                    values = scalar[name].combine_chunks().to_numpy(zero_copy_only=False)
                    valid = np.asarray(d[f"{name}_valid"], dtype=bool)
                    valid_arrays.append(valid)
                    selected = values[valid]
                    params[name].append(selected.copy())
                    anomalies[f"{name}_valid_nonfinite"] += int((~np.isfinite(selected)).sum())
                    invalid_nonzero[name] += int(((~valid) & np.isfinite(values) & (values != 0)).sum())
                    invalid_null[name] += int(((~valid) & ~np.isfinite(values)).sum())
                row_patterns = ["".join("1" if v else "0" for v in row)
                                for row in zip(*valid_arrays)]
                patterns.update(f"{t}:{m}" for t, m in zip(task_list, row_patterns))

                iq = batch.column(batch.schema.get_field_index("iq"))
                offsets = iq.offsets.to_numpy(zero_copy_only=False)
                points = iq.values
                inner_offsets = points.offsets.to_numpy(zero_copy_only=False)
                flat = points.values.to_numpy(zero_copy_only=False).astype("<f4", copy=False)
                lens = np.diff(offsets)
                anomalies["null_iq_rows"] += iq.null_count
                anomalies["null_iq_points"] += points.null_count
                anomalies["null_iq_values"] += points.values.null_count
                anomalies["iq_points_not_two_channels"] += int((np.diff(inner_offsets) != 2).sum())
                batch_hashes, batch_rms = [], []
                for i, (task, sid, length, pattern) in enumerate(zip(task_list, sid_list, lens, row_patterns)):
                    length = int(length)
                    lo, hi = int(inner_offsets[offsets[i]]), int(inner_offsets[offsets[i + 1]])
                    raw = flat[lo:hi]
                    anomalies["unexpected_iq_length"] += int(length not in [512, 1024])
                    anomalies["nonfinite_iq_rows"] += int(not np.isfinite(raw).all())
                    anomalies["zero_energy_iq_rows"] += int(not np.any(raw))
                    rms = float(np.sqrt(np.mean(raw.astype(np.float64) ** 2) * 2)) if length else 0.0
                    batch_rms.append(rms)
                    digest = waveform_hash(length, raw.tobytes())
                    packed = hashes.get(digest, 0)
                    hashes[digest] = (((packed >> 3) + 1) << 3) | (packed & 7) | TASK_BITS[task]
                    batch_hashes.append(digest)
                    lengths[f"{task}:{length}"] += 1
                    pattern_lengths[f"{task}:{pattern}:{length}"] += 1
                    if task in ["modulation", "protocol"]:
                        class_lengths[f"{task}:{d[f'{task}_label'][i]}:{length}"] += 1
                    for sample_id in sample_hashes.get(digest, []):
                        sample_matches[sample_id]["waveform_hash_matches"] += 1
                    if sid in sample_by_id:
                        original = np.asarray(sample_by_id[sid]["iq"], dtype="<f4")
                        sample_matches[sid]["id_matches"].append({
                            "file": path.name, "row": row_start + i,
                            "same_task": task == sample_by_id[sid]["task"],
                            "iq_equal": raw.size == original.size and np.array_equal(raw, original.ravel())})
                for task in set(task_list):
                    rms_values[task].append(np.asarray([r for t, r in zip(task_list, batch_rms) if t == task]))
                manifest = scalar.append_column("iq_length", pa.array(lens, type=pa.int32()))
                manifest = manifest.append_column("iq_blake2b_128", pa.array(batch_hashes, type=pa.binary(16)))
                manifest = manifest.append_column("source_file", pa.array([path.name] * n))
                manifest = manifest.append_column("source_row", pa.array(np.arange(row_start, row_start + n)))
                if manifest_writer is None:
                    manifest_writer = pq.ParquetWriter(out / "sample_manifest.parquet", manifest.schema, compression="zstd")
                manifest_writer.write_table(manifest)
                row_start += n
                total += n
            print(json.dumps({"file": path.name, "processed_rows": total,
                              "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
            files[-1]["task_counts"] = dict(per_file_tasks)
    finally:
        if manifest_writer is not None:
            manifest_writer.close()

    statistics = {}
    for name, chunks in params.items():
        arr = np.concatenate(chunks)
        finite = arr[np.isfinite(arr)]
        values, counts = np.unique(finite, return_counts=True)
        statistics[name] = {
            "valid_count": int(len(arr)), "finite_count": int(len(finite)),
            "zero_count": int((finite == 0).sum()), "unique_count": int(len(values)),
            "min": float(finite.min()), "max": float(finite.max()),
            "mean": float(finite.mean(dtype=np.float64)), "std_population": float(finite.std(dtype=np.float64)),
            "quantiles": {str(q): float(np.quantile(finite, q)) for q in [0.01, 0.1, 0.5, 0.9, 0.99]},
            "invalid_nonzero_count": invalid_nonzero[name], "invalid_null_or_nonfinite_count": invalid_null[name]}
        if len(values) <= 64:
            statistics[name]["value_counts"] = {str(float(v)): int(c) for v, c in zip(values, counts)}
    multiplicities = Counter(value >> 3 for value in hashes.values())
    report = {
        "data_root": str(root), "python": sys.version.split()[0], "pyarrow": pa.__version__,
        "audit_scope": "All scalar columns and every training IQ row; exact float32 byte hashes with length prefix. No near-duplicate/parent-record audit.",
        "rows": total, "bytes": sum(x["bytes"] for x in files), "files": files,
        "task_counts": dict(tasks), "class_counts": {k: dict(v) for k, v in labels.items()},
        "task_length_counts": dict(lengths), "class_length_counts": dict(class_lengths),
        "task_validity_patterns": dict(patterns), "pattern_length_counts": dict(pattern_lengths),
        "parameter_statistics": statistics, "anomalies": dict(anomalies),
        "exact_waveforms": {"unique_hashes": len(hashes),
                            "duplicate_groups": sum(v for k, v in multiplicities.items() if k > 1),
                            "excess_duplicate_rows": total - len(hashes),
                            "multiplicity_histogram": dict(multiplicities),
                            "cross_task_duplicate_groups": sum((v & 7).bit_count() > 1 for v in hashes.values())},
        "sample_io": {"rows": len(samples), "matches_to_training": sample_matches,
                      "input_expected_truth_consistency_verified": True},
        "rms_summary": {}, "elapsed_seconds": round(time.monotonic() - started, 2),
        "note": "Global audit statistics are descriptive. Fit training scalers/codebooks only after the split; manifest is not itself a train/validation split."}
    for task, chunks in rms_values.items():
        arr = np.concatenate(chunks)
        report["rms_summary"][task] = {str(q): float(np.quantile(arr, q)) for q in [0, 0.01, 0.5, 0.99, 1]}
    details, tuples = analyze_manifest(out)
    report["exact_waveforms"].update(details)
    report["parameter_tuple_structure"] = tuples
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    (out / "data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ["rows", "task_counts", "task_length_counts", "exact_waveforms", "elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
