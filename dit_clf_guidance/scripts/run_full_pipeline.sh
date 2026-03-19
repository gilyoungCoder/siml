#!/bin/bash
# Full pipeline: wait for training -> run guided generation on multiple GPUs
# This script is designed to run with nohup so it continues after SSH disconnect.
#
# Usage:
#   nohup bash scripts/run_full_pipeline.sh > logs/pipeline.log 2>&1 &

set -e
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."

CLASSIFIER_CKPT="work_dirs/pony_z0_resnet18/checkpoint/step_1000/classifier.pth"
PRETRAINED_MODEL="purplesmartai/pony-v7-base"

echo "=== Full Pipeline Started at $(date) ==="

# ---- Step 1: Wait for training to finish ----
echo "[Pipeline] Waiting for classifier training to complete..."
while [ ! -f "$CLASSIFIER_CKPT" ]; do
    sleep 30
done
echo "[Pipeline] Classifier checkpoint found: $CLASSIFIER_CKPT"
sleep 10  # extra buffer for file write completion

# ---- Step 2: Guided generation for country_body prompts ----
echo ""
echo "=== Step 2: Guided generation (country prompts) ==="
for PROMPT_FILE in prompts/country_body.txt prompts/country_nude_body.txt; do
    BASENAME=$(basename "$PROMPT_FILE" .txt)
    for SCALE in 5 10 15 20; do
        OUTPUT_DIR="output_img/${BASENAME}_guided_s${SCALE}"
        echo ""
        echo "--- ${BASENAME}, scale=${SCALE} ---"
        CUDA_VISIBLE_DEVICES=3 python generate.py \
          --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
          --prompt_file "$PROMPT_FILE" \
          --output_dir "$OUTPUT_DIR" \
          --nsamples 1 \
          --cfg_scale 3.5 \
          --num_inference_steps 20 \
          --height 1024 \
          --width 1024 \
          --seed 1234 \
          --classifier_ckpt "$CLASSIFIER_CKPT" \
          --num_classes 3 \
          --guidance_scale "$SCALE" \
          --guidance_mode safe_minus_harm \
          --safe_classes 0 1 \
          --harm_classes 2 \
          --grad_clip_ratio 0.3 \
          --mixed_precision bf16
    done
done

# ---- Step 3: Ring-a-Bell guided generation ----
echo ""
echo "=== Step 3: Ring-a-Bell guided generation ==="
for SCALE in 5 10 15 20; do
    echo ""
    echo "--- Ring-a-Bell, scale=${SCALE} ---"
    CUDA_VISIBLE_DEVICES=3 python generate.py \
      --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
      --prompt_csv "/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv" \
      --csv_column "sensitive prompt" \
      --output_dir "output_img/ringabell_nudity_guided_s${SCALE}" \
      --nsamples 1 \
      --cfg_scale 3.5 \
      --num_inference_steps 20 \
      --height 1024 \
      --width 1024 \
      --seed 1234 \
      --classifier_ckpt "$CLASSIFIER_CKPT" \
      --num_classes 3 \
      --guidance_scale "$SCALE" \
      --guidance_mode safe_minus_harm \
      --safe_classes 0 1 \
      --harm_classes 2 \
      --grad_clip_ratio 0.3 \
      --mixed_precision bf16
done

echo ""
echo "=== Full Pipeline Completed at $(date) ==="
echo "Results saved in output_img/"
