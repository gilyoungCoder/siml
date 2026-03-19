#!/usr/bin/env bash
# Train z0 classifier for Pony V7 (AuraFlow) guidance
#
# Two-phase pipeline:
#   Phase 1: Precompute z0_hat (run once, ~1-2 hours)
#   Phase 2: Train classifier on precomputed data (20000+ epochs, ~hours)
#
# Usage:
#   bash scripts/train.sh           # Full pipeline
#   bash scripts/train.sh --skip-precompute  # Skip phase 1

set -e
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."

PRETRAINED_MODEL="purplesmartai/pony-v7-base"
BENIGN_DIR="/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k"
PERSON_DIR="/mnt/home/yhgil99/dataset/threeclassImg/clothed5k"
NUDITY_DIR="/mnt/home/yhgil99/dataset/threeclassImg/Wnudity5k"
PRECOMPUTED_DIR="precomputed/pony_z0hat"
PRECOMPUTED_PATH="${PRECOMPUTED_DIR}/precomputed_organized.pt"
OUTPUT_DIR="work_dirs/pony_z0_resnet18_v2"

mkdir -p logs

# ---- Phase 1: Precompute z0_hat ----
if [ "$1" != "--skip-precompute" ] && [ ! -f "$PRECOMPUTED_PATH" ]; then
    echo "=== Phase 1: Precomputing z0_hat ==="
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python precompute_z0hat.py \
      --pretrained_model_name_or_path "$PRETRAINED_MODEL" \
      --benign_data_path "$BENIGN_DIR" \
      --person_data_path "$PERSON_DIR" \
      --nudity_data_path "$NUDITY_DIR" \
      --output_dir "$PRECOMPUTED_DIR" \
      --n_sigma 10 \
      --batch_size 8 \
      --resolution 512 \
      --balance_classes \
      --seed 42 \
      --mixed_precision bf16 \
      2>&1 | tee logs/precompute.log
    echo "Phase 1 done."
else
    echo "Skipping precomputation (already exists or --skip-precompute)"
fi

# ---- Phase 2: Train classifier ----
echo ""
echo "=== Phase 2: Training classifier (20000 epochs) ==="
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train.py \
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
  --wandb_run_name "resnet18_precomputed_20k_epochs" \
  --log_freq 10 \
  2>&1 | tee logs/train_v2.log

echo "Phase 2 done."
