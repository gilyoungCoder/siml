#!/usr/bin/env bash
set -uo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
REPO=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG
PY_EVAL=${PY_EVAL:-/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10}
GPU=${1:-0}
LOG=$ROOT/logs/coco_fid_official/merge_eval_coco1000.log
mkdir -p "$ROOT/logs/coco_fid_official"
merge_one() {
  local method=$1
  local merged srcmain chunkdir summary
  case "$method" in
    safedenoiser)
      merged=$ROOT/outputs/safedenoiser/nudity_coco/coco250x4_merged/all
      srcmain=$ROOT/outputs/safedenoiser/nudity_coco/coco250x4/all
      chunkdir=$ROOT/outputs/safedenoiser/nudity_coco/coco250x4_chunks
      summary=$ROOT/summaries/coco_fid_clip_safedenoiser_nudity_official_merged1000.txt
      ;;
    sgf)
      merged=$ROOT/outputs/sgf/nudity_coco/coco250x4_merged/all
      srcmain=$ROOT/outputs/sgf/nudity_coco/coco250x4_parallel/all
      chunkdir=$ROOT/outputs/sgf/nudity_coco/coco250x4_chunks
      summary=$ROOT/summaries/coco_fid_clip_sgf_nudity_official_merged1000.txt
      ;;
  esac
  mkdir -p "$merged"
  find "$srcmain" "$chunkdir" -type f -name '*.png' 2>/dev/null -print0 | while IFS= read -r -d '' f; do cp -n "$f" "$merged/" 2>/dev/null || true; done
  local n=$(find "$merged" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
  echo "[$(date '+%F %T')] merge $method n=$n" >> "$LOG"
  if [ "$n" -ge 1000 ] && [ ! -s "$summary" ]; then
    echo "[$(date '+%F %T')] EVAL $method n=$n" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU "$PY_EVAL" "$REPO/eval_fid_clip.py" "$REPO/outputs/coco_fid/baseline" "$merged" "$REPO/prompts/coco_250.txt" >> "$LOG" 2>&1
    cp "$merged/results_fid_clip.txt" "$summary" 2>/dev/null || true
  fi
}
for i in $(seq 1 240); do
  merge_one safedenoiser
  merge_one sgf
  [ -s "$ROOT/summaries/coco_fid_clip_safedenoiser_nudity_official_merged1000.txt" ] && [ -s "$ROOT/summaries/coco_fid_clip_sgf_nudity_official_merged1000.txt" ] && exit 0
  sleep 30
done
