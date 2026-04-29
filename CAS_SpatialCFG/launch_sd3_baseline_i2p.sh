#!/bin/bash
# Launch SD3 baseline on i2p_q16_top60 single concept.
# usage: ./launch_sd3_baseline_i2p.sh <gpu> <concept>
set -uo pipefail
G="$1"; C="$2"
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPTS=$REPO/CAS_SpatialCFG/prompts/i2p_q16_top60/${C}_q16_top60.txt
OUT=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_baseline_sd3/$C
LOGD=$REPO/CAS_SpatialCFG/outputs/launch_0429_i2p_baseline_sd3/_logs
mkdir -p "$OUT" "$LOGD"
setsid env CUDA_VISIBLE_DEVICES=$G $PY $REPO/scripts/sd3/generate_sd3_baseline.py \
  --prompts "$PROMPTS" --outdir "$OUT" --seed 42 \
  </dev/null > "$LOGD/g${G}_${C}.log" 2>&1 &
echo "started SD3 baseline $C on g$G pid=$!"
