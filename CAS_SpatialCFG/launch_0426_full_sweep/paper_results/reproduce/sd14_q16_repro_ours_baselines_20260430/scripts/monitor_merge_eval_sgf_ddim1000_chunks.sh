#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
CAS=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG
BASE=$CAS/outputs/baselines_v2/coco250
PROMPT_TXT=$ROOT/prompts/coco/coco_1000_from_coco250x4.txt
FINAL=$ROOT/outputs/sgf_ddim1000/nudity_coco/all
CHROOT=$ROOT/outputs/sgf_ddim1000_chunks
LOGDIR=$ROOT/logs/coco_ddim1000
mkdir -p "$LOGDIR" "$ROOT/summaries"
echo "[$(date)] monitor start"
while true; do
  final_count=0; [ -d "$FINAL" ] && final_count=$(find "$FINAL" -maxdepth 1 -type f -name '*.png' | wc -l)
  chunk_count=0; [ -d "$CHROOT" ] && chunk_count=$(find "$CHROOT" -path '*/all/*.png' -type f | wc -l)
  running=$(ssh siml-07 "pgrep -u yhgil99 -f 'generate_unsafe_sgf.py.*sgf_ddim1000_chunks' | wc -l" 2>/dev/null || echo 0)
  echo "[$(date)] final=$final_count chunk=$chunk_count running=$running"
  if [ "$running" = "0" ]; then break; fi
  sleep 60
done
# merge all chunk outputs into final dir; filenames are global case ids, so overwrite is safe for duplicates
find "$CHROOT" -path '*/all/*.png' -type f -print0 | xargs -0 -r cp -t "$FINAL"
count=$(find "$FINAL" -maxdepth 1 -type f -name '*.png' | wc -l)
echo "[$(date)] merged final_count=$count"
if [ "$count" -lt 1000 ]; then
  echo "[ERROR] SGF final incomplete after merge: $count"; exit 2
fi
CUDA_VISIBLE_DEVICES=0 /mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10 \
  "$ROOT/scripts/eval_fid_clip_fixed.py" "$BASE" "$FINAL" "$PROMPT_TXT" \
  | tee "$LOGDIR/eval_sgf_ddim1000.log"
cp "$FINAL/results_fid_clip_fixed.txt" "$ROOT/summaries/coco_fid_clip_sgf_nudity_ddim1000_vs_sd14ddim1000.txt"
echo "[DONE] SGF DDIM1000 merge/eval"
