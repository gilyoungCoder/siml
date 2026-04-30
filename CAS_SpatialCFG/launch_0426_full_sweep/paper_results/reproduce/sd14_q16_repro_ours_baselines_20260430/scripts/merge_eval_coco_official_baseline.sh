#!/usr/bin/env bash
set -uo pipefail
GPU=${1:-0}
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PROMPTS=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/coco_250.txt
BASE=$ROOT/outputs/baseline_official_std/nudity_coco/coco250x4_merged/all
mkdir -p "$BASE"
for i in $(seq 1 240); do
 find "$ROOT/outputs/baseline_official_std/nudity_coco/coco250x4_chunks" -type f -name '*.png' 2>/dev/null -print0 | while IFS= read -r -d '' f; do cp -n "$f" "$BASE/" 2>/dev/null || true; done
 n=$(find "$BASE" -maxdepth 1 -name '*.png' | wc -l)
 echo "[$(date)] official baseline merged=$n"
 if [ "$n" -ge 1000 ]; then
  for spec in safedenoiser:$ROOT/outputs/safedenoiser/nudity_coco/coco250x4_merged/all sgf:$ROOT/outputs/sgf/nudity_coco/coco250x4_merged/all; do
   name=${spec%%:*}; dir=${spec#*:}
   CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/eval_fid_clip_fixed.py" "$BASE" "$dir" "$PROMPTS"
   cp "$dir/results_fid_clip_fixed.txt" "$ROOT/summaries/coco_fid_clip_${name}_vs_official_std_baseline_1000_fixed.txt"
  done
  exit 0
 fi
 sleep 30
done
