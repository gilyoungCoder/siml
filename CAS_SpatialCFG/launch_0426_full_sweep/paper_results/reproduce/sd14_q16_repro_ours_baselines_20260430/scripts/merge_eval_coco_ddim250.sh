#!/usr/bin/env bash
set -euo pipefail
GPU=${1:-2}
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
CAS=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG
PY=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
BASE=$ROOT/outputs/baseline_sd14/coco250_ddim_first250
PROMPTS=$ROOT/prompts/coco/coco_250_first250.txt
python3 - <<PY
from pathlib import Path
src=Path('$CAS/prompts/coco_250.txt')
out=Path('$PROMPTS')
lines=[l.strip() for l in src.read_text().splitlines() if l.strip()]
# baseline subset is first 250 images = first 62 full prompts x4 + prompt 62 samples 0/1; CLIP script expects 250 prompts.
p=[]
for line in lines:
    for _ in range(4): p.append(line)
out.write_text('\n'.join(p[:250])+'\n')
print(out, len(p[:250]))
PY
for name in safedenoiser sgf; do
  if [ "$name" = safedenoiser ]; then SRC=$ROOT/outputs/safedenoiser_ddim250/nudity_coco/coco250; else SRC=$ROOT/outputs/sgf_ddim250/nudity_coco/coco250; fi
  ALL=$SRC/all
  COUNT=$(find "$ALL" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)
  echo "$name count=$COUNT all=$ALL"
  if [ "$COUNT" -ne 250 ]; then echo "SKIP $name incomplete"; continue; fi
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 "$PY" "$ROOT/scripts/eval_fid_clip_fixed.py" "$BASE" "$ALL" "$PROMPTS" \
    | tee "$ROOT/logs/coco_ddim250/eval_${name}_ddim250.log"
  cp "$ALL/results_fid_clip_fixed.txt" "$ROOT/summaries/coco_fid_clip_${name}_nudity_ddim250_vs_sd14ddim250.txt"
done
