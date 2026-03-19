#!/bin/bash
# Run SHGD on COCO-30k prompts for quality evaluation (FID, CLIP Score)
# Usage: bash run_coco.sh [config] [gpu_ids] [num_prompts]

PYTHON="PYTHONNOUSERSITE=1 /mnt/home/yhgil99/.conda/envs/sfgd/bin/python"

CONFIG=${1:-"configs/default.yaml"}
GPU_IDS=${2:-"0"}
NUM_PROMPTS=${3:-"1000"}  # Use subset for quick eval; full=10000
SAVE_DIR=${4:-"outputs/coco_$(basename ${CONFIG%.yaml})_$(date +%Y%m%d_%H%M%S)"}

PROMPT_FILE="../prompts/coco/coco_10k.txt"

# Create a subset prompt file if needed
if [ "$NUM_PROMPTS" -lt 10000 ]; then
    SUBSET_FILE="/tmp/coco_${NUM_PROMPTS}.txt"
    head -n "$NUM_PROMPTS" "$PROMPT_FILE" > "$SUBSET_FILE"
    PROMPT_FILE="$SUBSET_FILE"
    echo "Using first $NUM_PROMPTS COCO prompts"
fi

# Parse GPU IDs
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

echo "=============================================="
echo "SHGD - COCO Quality Evaluation"
echo "=============================================="
echo "Config: $CONFIG"
echo "GPUs: $GPU_IDS ($NUM_GPUS total)"
echo "Prompts: $NUM_PROMPTS"
echo "Save dir: $SAVE_DIR"
echo "=============================================="

if [ $NUM_GPUS -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=${GPUS[0]} $PYTHON generate.py \
        --config "$CONFIG" \
        --prompt_file "$PROMPT_FILE" \
        --save_dir "$SAVE_DIR" \
        --device "cuda:0" \
        --skip_eval
else
    PIDS=()
    for i in "${!GPUS[@]}"; do
        GPU=${GPUS[$i]}
        echo "Launching GPU $GPU (worker $i/$NUM_GPUS)..."
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON generate.py \
            --config "$CONFIG" \
            --prompt_file "$PROMPT_FILE" \
            --save_dir "$SAVE_DIR" \
            --device "cuda:0" \
            --gpu_id $i \
            --total_gpus $NUM_GPUS \
            --skip_eval &
        PIDS+=($!)
    done

    echo "Waiting for all workers..."
    for pid in "${PIDS[@]}"; do
        wait $pid
    done
fi

echo ""
echo "=============================================="
echo "Generation complete."
echo "=============================================="
echo ""
echo "To compute CLIP Score:"
echo "  python evaluate.py --image_dir $SAVE_DIR/all --eval_type clip --prompt_file $PROMPT_FILE"
echo ""
echo "To compute FID (requires reference stats):"
echo "  python -m pytorch_fid $SAVE_DIR/all <reference_dir>"
