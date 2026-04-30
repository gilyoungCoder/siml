#!/usr/bin/env bash
set -euo pipefail
GPU=${1:-0}
CAS=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG
ROOT=$CAS/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
BASE=$CAS/outputs/baselines_v2/coco250
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPTS=$CAS/prompts/coco_250.txt
for name in safedenoiser sgf; do
  if [ "$name" = safedenoiser ]; then METH=$ROOT/outputs/safedenoiser/nudity_coco/coco250x4_merged/all; else METH=$ROOT/outputs/sgf/nudity_coco/coco250x4_merged/all; fi
  echo "=== EVAL $name vs existing SD1.4 baseline ==="
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 "$PY" "$ROOT/scripts/eval_fid_clip_fixed.py" "$BASE" "$METH" "$PROMPTS" \
    | tee "$ROOT/logs/coco_fid_against_sd14/${name}_vs_baselines_v2_coco250.log"
  cp "$METH/results_fid_clip_fixed.txt" "$ROOT/summaries/coco_fid_clip_${name}_nudity_vs_baselines_v2_coco250.txt"
done
