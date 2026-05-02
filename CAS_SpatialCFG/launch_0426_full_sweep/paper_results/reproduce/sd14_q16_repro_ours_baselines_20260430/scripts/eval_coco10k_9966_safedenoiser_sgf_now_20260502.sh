#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
BASE=$ROOT/outputs/coco10k_baseline_flat/all
CSV=$ROOT/prompts/coco/coco_10k_9966.csv
LOGDIR=$ROOT/logs/coco10k_9966_eval_now_20260502
mkdir -p "$LOGDIR" "$ROOT/summaries"
run_eval(){
  local method=$1 gpu=$2
  local FINAL=$ROOT/outputs/${method}_coco10k_9966/all
  local OUTTXT=$ROOT/summaries/coco_fid_clip_${method}_nudity_coco10k9966_vs_phasebaseline_20260502.txt
  echo "[$(date '+%F %T')] START eval $method gpu=$gpu final=$FINAL"
  CUDA_VISIBLE_DEVICES=$gpu /mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10 \
    "$ROOT/scripts/eval_fid_clip_coco10k.py" "$BASE" "$FINAL" "$CSV" "$OUTTXT" \
    2>&1 | tee "$LOGDIR/eval_${method}_gpu${gpu}.log"
  echo "[$(date '+%F %T')] DONE eval $method"
}
run_eval safedenoiser 0 > "$LOGDIR/safedenoiser_launcher.log" 2>&1 &
run_eval sgf 1 > "$LOGDIR/sgf_launcher.log" 2>&1 &
wait
echo "[$(date '+%F %T')] ALL DONE" | tee "$LOGDIR/done.log"
