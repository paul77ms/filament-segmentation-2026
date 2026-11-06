#!/usr/bin/env bash
# AT_nested chain (CPU, 4 cores, TMPDIR in the project): sweep -> select -> combined -> loao -> analyse.
set -u
cd "$(dirname "$0")/.." && mkdir -p tmp logs   # repository root
export CUDA_VISIBLE_DEVICES= NUMPY_MADVISE_HUGEPAGE=0 TMPDIR="$PWD/tmp"
for c in sweep select combined loao analyse; do
  echo "$(date +%F_%T) start $c"
  taskset -c 0-3 .venv/bin/python scripts/tune_nested.py $c --workers 4 > logs/at_$c.log 2>&1 < /dev/null || { echo "$(date +%F_%T) $c FAILED"; exit 1; }
  echo "$(date +%F_%T) end $c"
done
