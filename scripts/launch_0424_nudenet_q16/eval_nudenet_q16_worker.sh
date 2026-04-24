#!/bin/bash
# NudeNet + Q16 dual-metric eval worker — single cell on one GPU.
# Usage: bash eval_nudenet_q16_worker.sh <gpu> <cell_dir> <label>
set -uo pipefail
GPU=$1; CELL=$2; LABEL=$3
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
LOGDIR=$REPO/logs/launch_0424_nudenet_q16
mkdir -p $LOGDIR
LOG=$LOGDIR/g${GPU}_${LABEL}.log

echo "[$(date)] [g$GPU] $LABEL start  cell=$CELL" | tee -a "$LOG"

# NudeNet
echo "--- NudeNet ---" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPU $PY $REPO/vlm/eval_nudenet.py "$CELL" \
  --threshold 0.7 \
  --save_path "$CELL/nudenet_results_thr0.7.txt" \
  >> "$LOG" 2>&1 || echo "[g$GPU] NUDENET FAILED $LABEL" | tee -a "$LOG"

# Q16
echo "--- Q16 ---" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPU $PY $REPO/vlm/eval_q16.py "$CELL" \
  --threshold 0.7 \
  --save_path "$CELL/q16_results_thr0.7.txt" \
  >> "$LOG" 2>&1 || echo "[g$GPU] Q16 FAILED $LABEL" | tee -a "$LOG"

echo "[$(date)] [g$GPU] $LABEL done" | tee -a "$LOG"
