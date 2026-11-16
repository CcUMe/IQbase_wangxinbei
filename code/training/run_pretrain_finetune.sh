#!/usr/bin/env bash
set -euo pipefail
c01_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
c01_python="${C01_PYTHON:-/home/yangqilin/anaconda3/envs/py313/bin/python}"
c01_gpu="${C01_GPU:-0}"
cd "$c01_root"
mkdir -p logs/shared_base_pipeline/launch
c01_stamp="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S_%N)"
c01_log="$c01_root/logs/shared_base_pipeline/launch/${c01_stamp}.log"
nohup setsid env CUDA_VISIBLE_DEVICES="$c01_gpu" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  PYTHONUNBUFFERED=1 "$c01_python" -u code/training/pretrain_finetune_pipeline.py \
  --gpu "$c01_gpu" "$@" >"$c01_log" 2>&1 < /dev/null &
c01_pid=$!
printf '%s\n' "$c01_pid" >"$c01_log.pid"
sleep 1
if ! kill -0 "$c01_pid" 2>/dev/null; then
  cat "$c01_log" >&2
  exit 1
fi
printf 'Pipeline PID=%s | GPU=%s\nLog: %s\n' "$c01_pid" "$c01_gpu" "$c01_log"
