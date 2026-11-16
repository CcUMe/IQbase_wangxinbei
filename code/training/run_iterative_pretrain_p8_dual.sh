#!/usr/bin/env bash
set -euo pipefail
c01_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$c01_root"
# P8 cache/raw vocabulary are prepared once; their commands are documented.
# The common launcher freezes this P8 config/source and defaults to GPU2,3.
exec bash code/training/run_iterative_pretrain_dual.sh \
  --config code/configs/pipeline/shared_base_iterative_p8_dual.json5 "$@"
