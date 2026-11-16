#!/usr/bin/env bash
# Start ONE training run in the background. Arguments are forwarded unchanged.
set -euo pipefail
c01_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
c01_python="${C01_PYTHON:-/home/yangqilin/anaconda3/envs/py313/bin/python}"
c01_gpu="${C01_GPU:-0}"
if [[ ! -x "$c01_python" ]]; then
  printf 'Python is not executable: %s\nSet C01_PYTHON to the project interpreter.\n' "$c01_python" >&2
  exit 1
fi
cd "$c01_root"
c01_launch_dir="$c01_root/logs/supervised_shift_consistent/launch"
mkdir -p "$c01_launch_dir"
c01_stamp="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S_%N)"
c01_launch_log="$c01_launch_dir/launch_${c01_stamp}.log"
nohup env CUDA_VISIBLE_DEVICES="$c01_gpu" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  PYTHONUNBUFFERED=1 "$c01_python" code/training/train_shift_consistent.py \
  --config code/configs/supervised/shift_consistent.json5 "$@" \
  >"$c01_launch_log" 2>&1 < /dev/null &
c01_pid=$!
printf '%s\n' "$c01_pid" > "$c01_launch_log.pid"
printf 'Started PID=%s on physical GPU=%s\nLog: %s\nMonitor: tail -f "%s"\n' \
  "$c01_pid" "$c01_gpu" "$c01_launch_log" "$c01_launch_log"
