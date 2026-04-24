#!/bin/bash
# v5 eval one single-pool cell on one GPU.
# Usage: bash eval_singlepool_v5.sh <gpu> <cell_dir> <concept_rubric>
set -uo pipefail
GPU=$1; CELL=$2; CONCEPT=$3
REPO=/mnt/home3/yhgil99/unlearning
PYEVAL=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
LOGDIR=$REPO/logs/launch_0424_singlepool_v5
mkdir -p $LOGDIR
LOG=$LOGDIR/g${GPU}_${CONCEPT}.log
echo "[$(date)] [g$GPU] eval $CELL rubric=$CONCEPT" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPU $PYEVAL $REPO/vlm/opensource_vlm_i2p_all_v5.py "$CELL" $CONCEPT qwen >> "$LOG" 2>&1 || echo "[g$GPU] EVAL FAILED" | tee -a "$LOG"
echo "[$(date)] [g$GPU] done" | tee -a "$LOG"
