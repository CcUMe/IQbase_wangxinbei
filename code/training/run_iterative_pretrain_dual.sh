#!/usr/bin/env bash
set -euo pipefail
c01_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
c01_python="${C01_PYTHON:-/home/yangqilin/anaconda3/envs/py313/bin/python}"
cd "$c01_root"
mkdir -p logs/shared_base_iterative_dual/launch
c01_stamp="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S_%N)"
c01_log="$c01_root/logs/shared_base_iterative_dual/launch/${c01_stamp}.log"
c01_gpu_args=()
if [[ -n "${C01_GPUS:-}" ]]; then c01_gpu_args=(--gpus "$C01_GPUS"); fi
nohup setsid env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
  "$c01_python" -u code/training/iterative_pretrain_dual.py \
  "${c01_gpu_args[@]}" "$@" >"$c01_log" 2>&1 < /dev/null &
c01_pid=$!
printf '%s\n' "$c01_pid" >"$c01_log.pid"
sleep 3
if ! kill -0 "$c01_pid" 2>/dev/null; then
  # --help / resuming a completed pipeline can succeed before this check.
  if wait "$c01_pid"; then
    cat "$c01_log"
    exit 0
  fi
  cat "$c01_log" >&2
  exit 1
fi
printf 'Dual-GPU iterative pipeline PID=%s\nLog: %s\n' "$c01_pid" "$c01_log"
