#!/usr/bin/env bash
set -euo pipefail
GPU=${1:-2}
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep/paper_results/reproduce/sd14_q16_repro_ours_baselines_20260430
OREPO=$ROOT/code/official_repos/Safe_Denoiser
PY=/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10
OUT=$ROOT/outputs/safedenoiser_ddim250/nudity_coco/coco250
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$OREPO"
echo "[RUN SafeDenoiser DDIM250] gpu=$GPU out=$OUT"
CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 "$PY" run_copro.py \
  --nudenet-path=pretrained/classifier_model.onnx --nudity_thr=0.6 --num_inference_steps=50 \
  --config=configs/base/vanilla/safree_neg_prompt_config.json --safe_level=MEDIUM \
  --data="$ROOT/prompts/coco/coco_250_x4.csv" --category=all \
  --save-dir="$OUT" --erase_id=safree_neg_prompt_rep_threshold_time \
  --guidance_scale=7.5 --seed=42 --valid_case_numbers=0,250
