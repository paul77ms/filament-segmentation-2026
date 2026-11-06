#!/usr/bin/env bash
# SE2 inner R2 models: for each outer fold k, rebuild the nested cache refine_nested_o<k> (stage 1 of every file from
# models that never saw k), train R2nn_context/fold<k>_in<j> on the folds outside {k, j} and refine fold j (j != k),
# then delete the cache (8.4 GB; only one on disk at a time). One GPU, one job at a time.
# Usage: scripts/run_r2nn.sh <gpu id approved by the user>
set -u
GPU=$1
cd "$(dirname "$0")/.." && mkdir -p tmp logs   # repository root
export CUDA_DEVICE_ORDER=PCI_BUS_ID NUMPY_MADVISE_HUGEPAGE=0 TMPDIR="$PWD/tmp"
CORES=22-25; PY=.venv/bin/python
for k in 0 1 2 3 4; do
  C="$PWD/outputs/cache/refine_nested_o$k"
  need=0
  for j in 0 1 2 3 4; do
    [ $j -eq $k ] && continue
    [ -f outputs/exps/R2nn_context/fold${k}_in${j}/val_cand_probs.npz ] || need=1
  done
  [ $need -eq 0 ] && { echo "$(date +%F_%T) outer $k done, skip"; continue; }
  free=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9)
  [ "$free" -lt 20 ] && { echo "$(date +%F_%T) only ${free} GB free, STOP"; exit 1; }
  if [ ! -f $C/index.json ]; then
    echo "$(date +%F_%T) build cache outer $k"
    CUDA_VISIBLE_DEVICES= taskset -c $CORES $PY scripts/nested_stage1.py cache --outer $k --workers 4 \
      > logs/r2nn_cache_o$k.log 2>&1 < /dev/null || { echo "$(date +%F_%T) cache $k FAILED"; exit 1; }
  fi
  for j in 0 1 2 3 4; do
    [ $j -eq $k ] && continue
    if [ -f outputs/exps/R2nn_context/fold${k}_in${j}/val_cand_probs.npz ]; then echo "$(date +%F_%T) skip ($k,$j)"; continue; fi
    echo "$(date +%F_%T) start R2nn outer $k inner $j on GPU$GPU"
    CUDA_VISIBLE_DEVICES=$GPU taskset -c $CORES $PY scripts/train_refine.py --variant r2 --fold $k --inner $j --epochs 30 \
        --exp R2nn_context --cache $C > logs/r2nn_o${k}_i${j}.log 2>&1 < /dev/null \
      || { echo "$(date +%F_%T) R2nn ($k,$j) FAILED"; exit 1; }
    echo "$(date +%F_%T) end R2nn outer $k inner $j"
  done
  if [ -n "$C" ] && [ -d "$C" ]; then rm -rf -- "$C"; echo "$(date +%F_%T) deleted cache outer $k"; fi
done
echo "$(date +%F_%T) R2NN ALL DONE"
