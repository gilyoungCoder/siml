#!/bin/bash
# Full pipeline on GPU 0 only: Precompute -> Train -> Generate
# Usage:
#   nohup bash scripts/run_full_gpu0.sh > logs/full_gpu0.log 2>&1 &

set -e
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

cd "$(dirname "$0")/.."

OUTPUT_DIR="precomputed/pony_z0hat"
PRETRAINED_MODEL="purplesmartai/pony-v7-base"
BENIGN_DIR="/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k"
PERSON_DIR="/mnt/home/yhgil99/dataset/threeclassImg/clothed5k"
NUDITY_DIR="/mnt/home/yhgil99/dataset/threeclassImg/Wnudity5k"

mkdir -p logs "$OUTPUT_DIR"

echo "=== Full Pipeline (GPU 0) Started at $(date) ==="

# ---- Phase 1: Precompute z0_hat ----
echo ""
echo "=== Phase 1: Precomputing z0_hat ==="
python precompute_z0hat.py \
  --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
  --benign_data_path "$BENIGN_DIR" \
  --person_data_path "$PERSON_DIR" \
  --nudity_data_path "$NUDITY_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --n_sigma 10 \
  --batch_size 16 \
  --resolution 512 \
  --balance_classes \
  --seed 42 \
  --mixed_precision bf16

echo ""
echo "=== Precompute completed at $(date) ==="

# ---- Phase 2: Train classifier (20000 epochs) ----
echo ""
echo "=== Phase 2: Training classifier (20000 epochs) ==="
python train.py \
  --precomputed_path "$OUTPUT_DIR/precomputed_organized.pt" \
  --output_dir work_dirs/pony_z0_resnet18_v2 \
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
echo "=== Training completed at $(date) ==="

# ---- Phase 3: Guided generation with best classifier ----
BEST_CKPT="work_dirs/pony_z0_resnet18_v2/classifier_best.pth"

if [ -f "$BEST_CKPT" ]; then
    echo ""
    echo "=== Phase 3: Guided Generation with Best Classifier ==="

    for SCALE in 5 10 20; do
        echo "--- country_nude_body, scale=$SCALE ---"
        python generate.py \
          --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
          --prompt_file prompts/country_nude_body.txt \
          --output_dir "output_img/country_nude_body_guided_v2_s${SCALE}" \
          --nsamples 1 --cfg_scale 3.5 --num_inference_steps 20 \
          --height 1024 --width 1024 --seed 1234 \
          --classifier_ckpt "$BEST_CKPT" --num_classes 3 \
          --guidance_scale "$SCALE" --guidance_mode safe_minus_harm \
          --safe_classes 0 1 --harm_classes 2 \
          --grad_clip_ratio 0.3 --mixed_precision bf16
    done

    for SCALE in 5 10 20; do
        echo "--- ringabell, scale=$SCALE ---"
        python generate.py \
          --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
          --prompt_csv "/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv" \
          --csv_column "sensitive prompt" \
          --output_dir "output_img/ringabell_nudity_guided_v2_s${SCALE}" \
          --nsamples 1 --cfg_scale 3.5 --num_inference_steps 20 \
          --height 1024 --width 1024 --seed 1234 \
          --classifier_ckpt "$BEST_CKPT" --num_classes 3 \
          --guidance_scale "$SCALE" --guidance_mode safe_minus_harm \
          --safe_classes 0 1 --harm_classes 2 \
          --grad_clip_ratio 0.3 --mixed_precision bf16
    done
fi

echo ""
echo "=== Full Pipeline Completed at $(date) ==="
