#!/bin/bash
# Auto-evaluate all completed SDErasure versions
# Usage: bash auto_eval_all.sh [GPU_ID]

EVAL_GPU="${1:-7}"
PYTHON="PYTHONNOUSERSITE=1 /mnt/home/yhgil99/.conda/envs/sfgd/bin/python"
VLM_PYTHON="PYTHONNOUSERSITE=1 /mnt/home/yhgil99/.conda/envs/vlm/bin/python"
PROMPT_FILE="/mnt/home/yhgil99/unlearning/prompts/nudity_datasets/ringabell.txt"
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"

cd /mnt/home/yhgil99/unlearning/SDErasure

for v in v7 v8 v9 v10 v11 v12; do
    UNET_DIR="./outputs/sderasure_nudity_${v}/unet"
    IMG_DIR="./outputs/sderasure_nudity_${v}/ringabell_images"
    RESULT_FILE="${IMG_DIR}/results_qwen3_vl_nudity.txt"

    # Skip if already evaluated
    if [ -f "$RESULT_FILE" ]; then
        echo "[$v] Already evaluated, skipping."
        continue
    fi

    # Skip if UNet not yet saved (training not done)
    if [ ! -f "${UNET_DIR}/config.json" ]; then
        echo "[$v] UNet not ready yet, skipping."
        continue
    fi

    echo "============================================================"
    echo "[$v] Generating Ring-A-Bell images..."
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=$EVAL_GPU eval $PYTHON generate_from_prompts.py \
        --model_id "CompVis/stable-diffusion-v1-4" \
        --unet_dir "$UNET_DIR" \
        --prompt_file "$PROMPT_FILE" \
        --output_dir "$IMG_DIR" \
        --seed 42

    echo "[$v] Running Qwen3-VL evaluation..."
    cd /mnt/home/yhgil99/unlearning
    CUDA_VISIBLE_DEVICES=$EVAL_GPU eval $VLM_PYTHON "$VLM_SCRIPT" "$IMG_DIR" nudity qwen
    cd /mnt/home/yhgil99/unlearning/SDErasure

    echo "[$v] Results:"
    cat "$RESULT_FILE" 2>/dev/null
    echo ""
done

echo ""
echo "============================================================"
echo "FINAL SUMMARY"
echo "============================================================"
for v in v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12; do
    RESULT_FILE="./outputs/sderasure_nudity_${v}/ringabell_images/results_qwen3_vl_nudity.txt"
    if [ -f "$RESULT_FILE" ]; then
        SR=$(grep "SR" "$RESULT_FILE" | head -1)
        echo "  $v: $SR"
    fi
done
echo "  SGF (target): SR (Safe+Partial): 59/79 (74.7%)"
