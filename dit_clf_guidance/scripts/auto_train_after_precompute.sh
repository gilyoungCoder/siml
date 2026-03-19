#!/bin/bash
# Wait for precomputation to finish, then start classifier training.
# Usage:
#   nohup bash scripts/auto_train_after_precompute.sh > logs/auto_pipeline.log 2>&1 &

set -e
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."

PRECOMPUTED_PATH="precomputed/pony_z0hat/precomputed_organized.pt"
OUTPUT_DIR="work_dirs/pony_z0_resnet18_v2"

echo "=== Auto Pipeline Started at $(date) ==="

# ---- Wait for precomputation to finish ----
echo "[Pipeline] Waiting for precomputed data: $PRECOMPUTED_PATH"
while [ ! -f "$PRECOMPUTED_PATH" ]; do
    sleep 60
    echo "  Still waiting... $(date)"
done
echo "[Pipeline] Precomputed file found at $(date)"

# Wait extra time for file write to fully complete
sleep 30
FILE_SIZE=$(stat -c%s "$PRECOMPUTED_PATH" 2>/dev/null || echo 0)
echo "[Pipeline] File size: $FILE_SIZE bytes"

# Verify the file is valid by checking it's not growing
sleep 10
FILE_SIZE2=$(stat -c%s "$PRECOMPUTED_PATH" 2>/dev/null || echo 0)
if [ "$FILE_SIZE" != "$FILE_SIZE2" ]; then
    echo "[Pipeline] File still being written, waiting 60 more seconds..."
    sleep 60
fi

echo "[Pipeline] Starting classifier training at $(date)"

# ---- Train classifier (20000 epochs) ----
CUDA_VISIBLE_DEVICES=1 python train.py \
  --precomputed_path "$PRECOMPUTED_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --num_classes 3 \
  --train_batch_size 64 \
  --learning_rate 1e-3 \
  --weight_decay 1e-4 \
  --num_train_epochs 20000 \
  --lr_scheduler cosine \
  --lr_warmup_steps 500 \
  --save_ckpt_freq 5000 \
  --val_freq 500 \
  --seed 42 \
  --use_wandb \
  --wandb_project pony_clf_guidance \
  --wandb_run_name "resnet18_precomputed_20k_epochs_v2" \
  --log_freq 10

echo ""
echo "=== Training Completed at $(date) ==="

# ---- After training: run guided generation ----
BEST_CKPT="$OUTPUT_DIR/classifier_best.pth"
PRETRAINED_MODEL="purplesmartai/pony-v7-base"

if [ -f "$BEST_CKPT" ]; then
    echo ""
    echo "=== Guided Generation with Best Classifier ==="

    # Country nude body guided
    for SCALE in 5 10 20; do
        echo "--- country_nude_body, scale=$SCALE ---"
        CUDA_VISIBLE_DEVICES=1 python generate.py \
          --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
          --prompt_file prompts/country_nude_body.txt \
          --output_dir "output_img/country_nude_body_guided_v2_s${SCALE}" \
          --nsamples 1 \
          --cfg_scale 3.5 \
          --num_inference_steps 20 \
          --height 1024 --width 1024 \
          --seed 1234 \
          --classifier_ckpt "$BEST_CKPT" \
          --num_classes 3 \
          --guidance_scale "$SCALE" \
          --guidance_mode safe_minus_harm \
          --safe_classes 0 1 \
          --harm_classes 2 \
          --grad_clip_ratio 0.3 \
          --mixed_precision bf16
    done

    # Ring-a-Bell guided
    for SCALE in 5 10 20; do
        echo "--- ringabell, scale=$SCALE ---"
        CUDA_VISIBLE_DEVICES=1 python generate.py \
          --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
          --prompt_csv "/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv" \
          --csv_column "sensitive prompt" \
          --output_dir "output_img/ringabell_nudity_guided_v2_s${SCALE}" \
          --nsamples 1 \
          --cfg_scale 3.5 \
          --num_inference_steps 20 \
          --height 1024 --width 1024 \
          --seed 1234 \
          --classifier_ckpt "$BEST_CKPT" \
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
else
    echo "WARNING: Best checkpoint not found at $BEST_CKPT"
    echo "Training may have failed. Check logs."
fi
