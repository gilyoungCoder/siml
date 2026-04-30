#!/usr/bin/env bash
set -uo pipefail
START=$1; LEN=$2; GPU=$3
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
PY=${PY_OFFICIAL:-/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10}
DATA=$ROOT/prompts/coco/coco_250_x4.csv
OUT=$ROOT/outputs/baseline_official_std/nudity_coco/coco250x4_chunks/${START}_$((START+LEN))
mkdir -p "$OUT/safe" "$OUT/unsafe" "$OUT/all" "$ROOT/logs/coco_fid_official/chunks"
count=$(find "$OUT/all" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
[ "$count" -ge "$LEN" ] && { echo "SKIP official std $START $LEN"; exit 0; }
rm -rf "$OUT"; mkdir -p "$OUT/safe" "$OUT/unsafe" "$OUT/all"
cd "$ROOT/code/official_repos/Safe_Denoiser"
CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 "$PY" run_copro.py \
  --nudenet-path=pretrained/classifier_model.onnx --nudity_thr=0.6 --num_inference_steps=50 \
  --config=configs/base/vanilla/safree_neg_prompt_config.json --safe_level=MEDIUM \
  --data="$DATA" --category=all --save-dir="$OUT" --erase_id=std \
  --guidance_scale=7.5 --seed=42 --valid_case_numbers=${START},${LEN}
