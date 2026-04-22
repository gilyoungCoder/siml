#!/bin/bash
# Urgent v3 nudity eval for SD3+FLUX1 mja_sexual hybrid ss10/15/20.
# Usage: bash sd3_flux1_sexual_eval.sh <gpu> <slot> <nshards>
set -uo pipefail
GPU=$1; SLOT=$2; NSHARDS=$3
REPO=/mnt/home3/yhgil99/unlearning
PY=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
LOG=$REPO/logs/launch_0420/sd3flux_sexual_eval_g${GPU}.log
JOBS=()
for backbone in sd3 flux1; do
  for d in $REPO/CAS_SpatialCFG/outputs/launch_0420/ours_${backbone}/mja_sexual/hybrid_ss*_imgthr*_both; do
    [ -d "$d" ] || continue
    n=$(ls "$d"/*.png 2>/dev/null | wc -l)
    [ "$n" -lt 90 ] && continue
    [ -f "$d/categories_qwen3_vl_nudity_v3.json" ] && continue
    JOBS+=("$d")
  done
done
N=${#JOBS[@]}
echo "[g$GPU s$SLOT/$NSHARDS] $N pending eval" | tee -a "$LOG"
cd $REPO/vlm
for ((i=SLOT; i<N; i+=NSHARDS)); do
  d="${JOBS[$i]}"
  echo "[$(date)] [g$GPU] eval $d" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU $PY opensource_vlm_i2p_all_v3.py "$d" nudity qwen >> "$LOG" 2>&1 || echo "FAIL $d" | tee -a "$LOG"
done
