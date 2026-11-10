#!/usr/bin/env bash
# SE2 after the inner R2 queue: inner data (CPU) -> nested + final scorers (GPU) -> eval / loao / analyse (CPU).
# Usage: scripts/run_se2_eval.sh <gpu id> <pid of run_r2nn.sh to wait for>
set -u
GPU=$1; WAIT=$2
cd "$(dirname "$0")/.." && mkdir -p tmp logs   # repository root
export CUDA_DEVICE_ORDER=PCI_BUS_ID NUMPY_MADVISE_HUGEPAGE=0 TMPDIR="$PWD/tmp"
CORES=22-25; PY=.venv/bin/python
while kill -0 "$WAIT" 2>/dev/null; do sleep 30; done
grep -q "R2NN ALL DONE" logs/run_r2nn.out || { echo "$(date +%F_%T) R2nn queue did not finish, STOP"; exit 1; }
echo "$(date +%F_%T) SE2 inner data (CPU)"
CUDA_VISIBLE_DEVICES= taskset -c $CORES $PY scripts/se2.py data --source inner --workers 4 > logs/se2_data_inner.log 2>&1 < /dev/null \
  || { echo "$(date +%F_%T) SE2 data FAILED"; exit 1; }
echo "$(date +%F_%T) SE2 train on GPU$GPU"
CUDA_VISIBLE_DEVICES=$GPU taskset -c $CORES $PY scripts/se2.py train > logs/se2_train.log 2>&1 < /dev/null \
  || { echo "$(date +%F_%T) SE2 train FAILED"; exit 1; }
for c in eval loao analyse; do
  CUDA_VISIBLE_DEVICES= taskset -c $CORES $PY scripts/se2.py $c --workers 4 > logs/se2_$c.log 2>&1 < /dev/null
  echo "$(date +%F_%T) SE2 $c rc=$?"
done
echo "$(date +%F_%T) SE2 ALL DONE"
