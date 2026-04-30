#!/usr/bin/env bash
set -uo pipefail
GPU=${1:-0}
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
REPO=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
for spec in safedenoiser:$ROOT/outputs/safedenoiser/nudity_coco/coco250x4_merged/all sgf:$ROOT/outputs/sgf/nudity_coco/coco250x4_merged/all; do
  name=${spec%%:*}; dir=${spec#*:}
  echo "RUN fixed eval $name $dir"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/eval_fid_clip_fixed.py" "$REPO/outputs/coco_fid/baseline" "$dir" "$REPO/prompts/coco_250.txt"
  cp "$dir/results_fid_clip_fixed.txt" "$ROOT/summaries/coco_fid_clip_${name}_nudity_official_merged1000_fixed.txt"
done
