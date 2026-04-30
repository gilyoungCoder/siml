#!/usr/bin/env bash
set -uo pipefail
GPU=${1:-2}
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
REPO=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG
PY_OFFICIAL=${PY_OFFICIAL:-/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10}
PY_EVAL=${PY_EVAL:-/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10}
DATA=$ROOT/prompts/coco/coco_250_x4.csv
BASELINE=$REPO/outputs/coco_fid/baseline
PROMPTS=$REPO/prompts/coco_250.txt
LOGDIR=$ROOT/logs/coco_fid_official
OUT=$ROOT/outputs/sgf/nudity_coco/coco250x4_parallel
mkdir -p "$LOGDIR" "$OUT/safe" "$OUT/unsafe" "$OUT/all"
count=$(find "$OUT/all" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
echo "[$(date)] SGF parallel all_count=$count/1000 out=$OUT gpu=$GPU"
if [ "$count" -lt 1000 ]; then
  rm -rf "$OUT"
  mkdir -p "$OUT/safe" "$OUT/unsafe" "$OUT/all"
  cd "$ROOT/code/official_repos/SGF/nudity_sdv1"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 "$PY_OFFICIAL" generate_unsafe_sgf.py \
    --nudenet-path=pretrained/classifier_model.onnx \
    --nudity_thr=0.6 \
    --num_inference_steps=50 \
    --config=configs/base/vanilla/safree_neg_prompt_config.json \
    --safe_level=MEDIUM \
    --data="$DATA" \
    --category=all \
    --task_config=configs/sgf/sgf.yaml \
    --save-dir="$OUT" \
    --erase_id=safree_neg_prompt_rep_time \
    --guidance_scale=7.5 \
    --seed=42 \
    --valid_case_numbers=0,100000 2>&1 | tee "$LOGDIR/sgf_parallel_gen.log"
fi
count=$(find "$OUT/all" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
echo "[$(date)] SGF parallel post_gen_count=$count"
if [ "$count" -ge 1000 ]; then
  CUDA_VISIBLE_DEVICES=$GPU "$PY_EVAL" "$REPO/eval_fid_clip.py" "$BASELINE" "$OUT/all" "$PROMPTS" 2>&1 | tee "$LOGDIR/sgf_parallel_fid_clip.log"
  cp "$OUT/all/results_fid_clip.txt" "$ROOT/summaries/coco_fid_clip_sgf_parallel_nudity_official.txt" 2>/dev/null || true
else
  echo "[$(date)] ERROR incomplete sgf parallel count=$count" | tee -a "$LOGDIR/sgf_parallel_gen.log"
  exit 3
fi
