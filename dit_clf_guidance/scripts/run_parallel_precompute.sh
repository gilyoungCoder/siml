#!/bin/bash
# Run precompute across 8 GPUs in parallel, then merge and start training.
# Usage:
#   nohup bash scripts/run_parallel_precompute.sh > logs/parallel_pipeline.log 2>&1 &

set -e
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "$(dirname "$0")/.."

NUM_SHARDS=8
OUTPUT_DIR="precomputed/pony_z0hat"
PRETRAINED_MODEL="purplesmartai/pony-v7-base"
BENIGN_DIR="/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k"
PERSON_DIR="/mnt/home/yhgil99/dataset/threeclassImg/clothed5k"
NUDITY_DIR="/mnt/home/yhgil99/dataset/threeclassImg/Wnudity5k"

mkdir -p logs "$OUTPUT_DIR"

echo "=== 8-GPU Parallel Precompute Started at $(date) ==="

# ---- Phase 1: Launch 8 parallel precompute shards ----
PIDS=()
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    echo "Launching shard $SHARD_ID on GPU $SHARD_ID..."
    CUDA_VISIBLE_DEVICES=$SHARD_ID python precompute_z0hat.py \
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
      --mixed_precision bf16 \
      --shard_id "$SHARD_ID" \
      --num_shards "$NUM_SHARDS" \
      > "logs/precompute_shard_${SHARD_ID}.log" 2>&1 &
    PIDS+=($!)
done

echo "All ${NUM_SHARDS} shards launched. PIDs: ${PIDS[*]}"
echo "Waiting for all shards to complete..."

# Wait for all shards
FAILED=0
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    if wait ${PIDS[$i]}; then
        echo "  Shard $i completed successfully."
    else
        echo "  ERROR: Shard $i failed!"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo "ERROR: $FAILED shards failed. Aborting."
    exit 1
fi

echo ""
echo "=== All shards completed at $(date) ==="

# ---- Phase 2: Merge shards ----
echo ""
echo "=== Merging shards ==="
python precompute_z0hat.py \
  --benign_data_path "$BENIGN_DIR" \
  --person_data_path "$PERSON_DIR" \
  --nudity_data_path "$NUDITY_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --merge_shards \
  --num_shards "$NUM_SHARDS" \
  --seed 42

echo ""
echo "=== Merge completed at $(date) ==="

# ---- Phase 3: Train classifier (20000 epochs) ----
echo ""
echo "=== Training classifier (20000 epochs) ==="
CUDA_VISIBLE_DEVICES=0 python train.py \
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

# ---- Phase 4: Guided generation with best classifier ----
BEST_CKPT="work_dirs/pony_z0_resnet18_v2/classifier_best.pth"

if [ -f "$BEST_CKPT" ]; then
    echo ""
    echo "=== Guided Generation with Best Classifier ==="

    for SCALE in 5 10 20; do
        echo "--- country_nude_body, scale=$SCALE ---"
        CUDA_VISIBLE_DEVICES=0 python generate.py \
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
        CUDA_VISIBLE_DEVICES=0 python generate.py \
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
