#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
CAS=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG
GENPY=/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10
EVALPY=/mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10
GPU=${GPU:-0}
BASE=$CAS/outputs/baselines_v2/coco250
PROMPT_CSV=$ROOT/prompts/coco/coco_250_x4.csv
PROMPT_TXT=$ROOT/prompts/coco/coco_1000_from_coco250x4.txt
LOGDIR=$ROOT/logs/coco_ddim1000
mkdir -p "$LOGDIR" "$ROOT/summaries"
# ensure prompt txt has exactly 1000, aligned with coco_250_x4.csv rows
/mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10 - <<PY
import csv
from pathlib import Path
csvp=Path('$PROMPT_CSV'); out=Path('$PROMPT_TXT')
with csvp.open() as f:
    r=csv.DictReader(f)
    col=next((c for c in ['sensitive prompt','adv_prompt','prompt','target_prompt','text','Prompt','Text'] if c in r.fieldnames), None)
    if col is None: raise SystemExit('no prompt col: '+str(r.fieldnames))
    lines=[row[col].strip() for row in r if row.get(col,'').strip()]
out.write_text('\n'.join(lines[:1000])+'\n')
print('prompt_txt', out, len(lines[:1000]))
PY

eval_method () {
  local method=$1 out=$2
  local count=0
  [ -d "$out/all" ] && count=$(find "$out/all" -maxdepth 1 -type f -name '*.png' | wc -l)
  echo "[COUNT] $method all=$count"
  if [ "$count" -lt 1000 ]; then echo "[SKIP EVAL] incomplete $method"; return 0; fi
  echo "[EVAL] $method vs $BASE using safree env CUDA-compatible"
  CUDA_VISIBLE_DEVICES=$GPU "$EVALPY" "$ROOT/scripts/eval_fid_clip_fixed.py" "$BASE" "$out/all" "$PROMPT_TXT" \
    | tee "$LOGDIR/eval_${method}_ddim1000.log"
  cp "$out/all/results_fid_clip_fixed.txt" "$ROOT/summaries/coco_fid_clip_${method}_nudity_ddim1000_vs_sd14ddim1000.txt"
}

gen_sgf () {
  local out=$ROOT/outputs/sgf_ddim1000/nudity_coco
  local count=0
  [ -d "$out/all" ] && count=$(find "$out/all" -maxdepth 1 -type f -name '*.png' | wc -l)
  if [ "$count" -ge 1000 ]; then echo "[SKIP GEN] sgf already count=$count"; return 0; fi
  rm -rf "$out"; mkdir -p "$out" "$out/safe" "$out/unsafe" "$out/nudity" "$out/all"
  echo "[RUN GEN] sgf -> $out"
  cd "$ROOT/code/official_repos/SGF/nudity_sdv1"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 "$GENPY" generate_unsafe_sgf.py \
    --nudenet-path=pretrained/classifier_model.onnx --nudity_thr=0.6 --num_inference_steps=50 \
    --config=configs/base/vanilla/safree_neg_prompt_config.json --safe_level=MEDIUM \
    --data="$PROMPT_CSV" --category=all --task_config configs/sgf/sgf.yaml \
    --save-dir="$out" --erase_id=safree_neg_prompt_rep_time --guidance_scale=7.5 --seed=42 \
    --valid_case_numbers=0,1000
}

eval_method safedenoiser "$ROOT/outputs/safedenoiser_ddim1000/nudity_coco"
gen_sgf
eval_method sgf "$ROOT/outputs/sgf_ddim1000/nudity_coco"
echo "[DONE] resume COCO1000 eval/gen"
