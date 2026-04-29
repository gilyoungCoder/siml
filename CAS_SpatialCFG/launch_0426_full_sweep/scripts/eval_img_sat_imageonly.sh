#!/bin/bash
set -uo pipefail
GPU=${1:-0}
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL=$REPO/vlm/opensource_vlm_i2p_all_v5.py
BASE=$REPO/CAS_SpatialCFG/launch_0426_full_sweep
LOG=$BASE/logs/eval_img_sat_imageonly_$(date +%m%d_%H%M).log
echo "Starting eval on GPU $GPU" > $LOG
cd $REPO/vlm
for K in 1 2 4 8 16 32; do
  D=$BASE/outputs/phase_img_saturation_imageonly/sexual_K${K}
  J=$D/categories_qwen3_vl_nudity_v5.json
  if [ -f "$J" ]; then
    echo "[K=$K] SKIP (json exists)" | tee -a $LOG
    continue
  fi
  echo "[$(date +%H:%M:%S)] [K=$K] EVAL start" | tee -a $LOG
  CUDA_VISIBLE_DEVICES=$GPU $PY $EVAL "$D" nudity qwen >> $LOG 2>&1
  echo "[$(date +%H:%M:%S)] [K=$K] EVAL done" | tee -a $LOG
done
echo "[$(date)] all eval done" | tee -a $LOG
