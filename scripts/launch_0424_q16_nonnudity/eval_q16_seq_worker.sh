#!/bin/bash
# Q16 sequential worker — handles N cells sequentially on one GPU.
# Usage: bash eval_q16_seq_worker.sh <gpu> "label1 dir1" "label2 dir2" ...
set -uo pipefail
GPU=$1; shift
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0424_q16_nonnudity
mkdir -p $LOGDIR
LOG=$LOGDIR/g${GPU}_seq.log

echo "[$(date)] [g$GPU] sequential worker start; $# cells" | tee -a "$LOG"
for spec in "$@"; do
  read LABEL CELL <<< "$spec"
  echo "[$(date)] [g$GPU] $LABEL  cell=$CELL" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU $PY $REPO/vlm/eval_q16.py "$CELL" \
    --threshold 0.7 \
    --save_path "$CELL/q16_results_thr0.7.txt" \
    >> "$LOG" 2>&1 || echo "[g$GPU] Q16 FAILED $LABEL" | tee -a "$LOG"
done
echo "[$(date)] [g$GPU] all done" | tee -a "$LOG"
