#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
BASE=$ROOT/outputs/coco10k_baseline_flat/all
CSV=$ROOT/prompts/coco/coco_10k_9966.csv
LOGDIR=$ROOT/logs/coco10k_9966
mkdir -p "$LOGDIR" "$ROOT/summaries"
echo "[$(date)] monitor coco10k start"
while true; do
  r7=$(ssh siml-07 "pgrep -u yhgil99 -f 'run_coco10k_9966_official_chunk|run_copro.py.*coco10k_9966|generate_unsafe_sgf.py.*coco10k_9966' | wc -l" 2>/dev/null || echo 0)
  r9=$(pgrep -u yhgil99 -f 'run_coco10k_9966_official_chunk|run_copro.py.*coco10k_9966|generate_unsafe_sgf.py.*coco10k_9966' | wc -l || true)
  echo "[$(date)] running siml07=$r7 siml09=$r9"
  if [ "$r7" = "0" ] && [ "$r9" = "0" ]; then break; fi
  sleep 120
done
for method in safedenoiser sgf; do
  FINAL=$ROOT/outputs/${method}_coco10k_9966/all
  rm -rf "$FINAL"; mkdir -p "$FINAL"
  find "$ROOT/outputs/${method}_coco10k_9966/chunks" -path '*/all/*.png' -type f -print0 | xargs -0 -r cp -t "$FINAL"
  count=$(find "$FINAL" -maxdepth 1 -type f -name '*.png' | wc -l)
  echo "[$(date)] merged $method count=$count"
  if [ "$count" -ne 9966 ]; then echo "[ERROR] $method expected 9966 got $count"; exit 4; fi
  OUTTXT=$ROOT/summaries/coco_fid_clip_${method}_nudity_coco10k9966_vs_phasebaseline.txt
  CUDA_VISIBLE_DEVICES=0 /mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10 \
    "$ROOT/scripts/eval_fid_clip_coco10k.py" "$BASE" "$FINAL" "$CSV" "$OUTTXT" \
    | tee "$LOGDIR/eval_${method}.log"
done
echo "[DONE] coco10k 9966 official eval"
