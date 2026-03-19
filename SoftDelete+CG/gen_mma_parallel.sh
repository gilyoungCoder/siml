#!/bin/bash
# MMA 1000개 프롬프트를 8 GPU로 분할 생성
# Config: mon0.05_gs12.5_bs2.0_sp0.2-0.3

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

SRC_CSV="/mnt/home/yhgil99/unlearning/SAFREE/datasets/mma-diffusion-nsfw-adv-prompts.csv"
OUTPUT_BASE="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/fine_grid_mon4class/mma/mon0.05_gs12.5_bs2.0_sp0.2-0.3"
SPLIT_DIR="/tmp/mma_splits"
NUM_GPUS=8

# Kill existing mma generation
pkill -u yhgil99 -f "mma.*mon0.05_gs12.5" 2>/dev/null

# Split CSV
mkdir -p "$SPLIT_DIR" "$OUTPUT_BASE"
HEADER=$(head -1 "$SRC_CSV")
TOTAL=$(tail -n +2 "$SRC_CSV" | wc -l)
PER_GPU=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))

echo "Total prompts: $TOTAL, per GPU: $PER_GPU"

tail -n +2 "$SRC_CSV" | split -l $PER_GPU -d -a 1 - "$SPLIT_DIR/chunk_"

PIDS=()
for i in $(seq 0 $((NUM_GPUS-1))); do
    CHUNK="$SPLIT_DIR/chunk_$i"
    [ -f "$CHUNK" ] || continue

    # Add header
    CHUNK_CSV="$SPLIT_DIR/mma_gpu${i}.csv"
    echo "$HEADER" > "$CHUNK_CSV"
    cat "$CHUNK" >> "$CHUNK_CSV"

    CHUNK_OUT="${OUTPUT_BASE}_gpu${i}"
    mkdir -p "$CHUNK_OUT"

    echo "[GPU $i] $(wc -l < "$CHUNK") prompts -> $CHUNK_OUT"

    CUDA_VISIBLE_DEVICES=$i python generate_nudity_4class_sample_level_monitoring.py \
        --ckpt_path CompVis/stable-diffusion-v1-4 \
        --prompt_file "$CHUNK_CSV" \
        --output_dir "$CHUNK_OUT" \
        --classifier_ckpt work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth \
        --gradcam_stats_dir gradcam_stats/nudity_4class \
        --monitoring_threshold 0.05 --guidance_scale 12.5 --base_guidance_scale 2.0 \
        --spatial_threshold_start 0.2 --spatial_threshold_end 0.3 \
        --spatial_threshold_strategy cosine --num_inference_steps 50 --cfg_scale 7.5 --seed 42 --nsamples 1 \
        > "${OUTPUT_BASE}_gpu${i}.log" 2>&1 &

    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} processes. Waiting..."
for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null; done

# Merge results: copy all images to main output dir
echo "Merging results..."
for i in $(seq 0 $((NUM_GPUS-1))); do
    CHUNK_OUT="${OUTPUT_BASE}_gpu${i}"
    [ -d "$CHUNK_OUT" ] || continue
    cp "$CHUNK_OUT"/*.png "$OUTPUT_BASE/" 2>/dev/null
    # Merge generation_stats if needed
done

# Count final images
TOTAL_IMGS=$(ls "$OUTPUT_BASE"/*.png 2>/dev/null | wc -l)
echo "Done! Total images: $TOTAL_IMGS in $OUTPUT_BASE"
