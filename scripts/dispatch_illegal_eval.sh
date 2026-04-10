#!/usr/bin/env bash
set -uo pipefail
export PYTHONNOUSERSITE=1
REPO=/mnt/home3/yhgil99/unlearning
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=$REPO/vlm

echo "[$(date)] Illegal activity Qwen eval"

# GPU 4: both_ainp configs (9)
(
  for d in $REPO/CAS_SpatialCFG/outputs/v27_final/c_illegal_activity_both_ainp_*; do
    [ -f "$d/results_qwen3_vl_illegal.txt" ] && continue
    echo "[$(date)] Eval: $(basename $d)"
    CUDA_VISIBLE_DEVICES=4 $VLP $VLD/opensource_vlm_i2p_all.py "$d" illegal qwen 2>&1 | tail -1
  done
) &

# GPU 5: text_ainp configs (9)
(
  for d in $REPO/CAS_SpatialCFG/outputs/v27_final/c_illegal_activity_text_ainp_*; do
    [ -f "$d/results_qwen3_vl_illegal.txt" ] && continue
    echo "[$(date)] Eval: $(basename $d)"
    CUDA_VISIBLE_DEVICES=5 $VLP $VLD/opensource_vlm_i2p_all.py "$d" illegal qwen 2>&1 | tail -1
  done
) &

wait
echo "[$(date)] Illegal eval all done"
