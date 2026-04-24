#!/bin/bash
# SDXL-Lightning 4-step chunk worker — one (dataset, gpu, start, end) cell.
# Usage: bash sdxl_lightning_chunk.sh <gpu> <dataset> <start> <end>
set -uo pipefail
GPU=$1; DATASET=$2; START=$3; END=$4
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
SCRIPT=$REPO/scripts/launch_0425_sdxl_lightning/sdxl_lightning_gen.py

# Dataset -> prompt file mapping
case "$DATASET" in
  i2p)          PROMPTS=$REPO/CAS_SpatialCFG/prompts/i2p_full.txt ;;
  mja_sexual)   PROMPTS=$REPO/CAS_SpatialCFG/prompts/mja_sexual.txt ;;
  mja_violent)  PROMPTS=$REPO/CAS_SpatialCFG/prompts/mja_violent.txt ;;
  mja_illegal)  PROMPTS=$REPO/CAS_SpatialCFG/prompts/mja_illegal.txt ;;
  mja_disturbing) PROMPTS=$REPO/CAS_SpatialCFG/prompts/mja_disturbing.txt ;;
  unlearndiff)  PROMPTS=$REPO/CAS_SpatialCFG/prompts/unlearndiff.txt ;;
  ringabell)    PROMPTS=$REPO/CAS_SpatialCFG/prompts/ringabell.txt ;;
  *) echo "Unknown dataset: $DATASET"; exit 1 ;;
esac
OUTDIR=$REPO/CAS_SpatialCFG/outputs/launch_0425_sdxl_lightning_human_eval/${DATASET}
LOGDIR=$REPO/logs/launch_0425_sdxl_lightning
LOG=$LOGDIR/g${GPU}_${DATASET}_${START}_${END}.log
mkdir -p "$OUTDIR" "$LOGDIR"

echo "[$(date)] [g$GPU] $DATASET chunk $START-$END start" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPU $PY $SCRIPT --prompts "$PROMPTS" --outdir "$OUTDIR" --start_idx $START --end_idx $END --steps 4 --seed 42 >> "$LOG" 2>&1 || echo "[g$GPU] FAILED" | tee -a "$LOG"
echo "[$(date)] [g$GPU] $DATASET chunk $START-$END done" | tee -a "$LOG"
