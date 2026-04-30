#!/usr/bin/env bash
set -euo pipefail
GPU=${1:-0}
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
CAS=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
ALL=$ROOT/outputs/ours/coco_family_best/all
mkdir -p "$ALL"
find "$ROOT/outputs/ours/coco_family_best/chunks" -mindepth 2 -maxdepth 2 -type f -name '*.png' | sort | while read f; do cp -n "$f" "$ALL/$(basename "$f")"; done
COUNT=$(find "$ALL" -maxdepth 1 -type f -name '*.png' | wc -l)
echo "merged count=$COUNT"
if [ "$COUNT" -eq 1000 ]; then
 CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 "$PY" "$ROOT/scripts/eval_fid_clip_fixed.py" \
   "$CAS/outputs/coco_fid/baseline" "$ALL" "$CAS/prompts/coco_250.txt" | tee "$ROOT/logs/ours_coco_family/eval_family_best.log"
 cp "$ALL/results_fid_clip_fixed.txt" "$ROOT/summaries/coco_fid_clip_ours_family_best_1000_fixed.txt"
fi
