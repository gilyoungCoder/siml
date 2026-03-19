#!/bin/bash
# Run SHGD on Ring-A-Bell dataset
# Usage: bash run_rab.sh [config] [gpu_ids]
# Example: bash run_rab.sh configs/default.yaml 0,1,2,3

PYTHON="PYTHONNOUSERSITE=1 /mnt/home/yhgil99/.conda/envs/sfgd/bin/python"
VLM_PYTHON="PYTHONNOUSERSITE=1 /mnt/home/yhgil99/.conda/envs/vlm/bin/python"

CONFIG=${1:-"configs/default.yaml"}
GPU_IDS=${2:-"0"}
SAVE_DIR=${3:-"outputs/rab_$(basename ${CONFIG%.yaml})_$(date +%Y%m%d_%H%M%S)"}

PROMPT_FILE="../rab_grid_search/data/ringabell_full.txt"

# Parse GPU IDs
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

echo "=============================================="
echo "SHGD - Ring-A-Bell Generation"
echo "=============================================="
echo "Config: $CONFIG"
echo "GPUs: $GPU_IDS ($NUM_GPUS total)"
echo "Save dir: $SAVE_DIR"
echo "Prompt file: $PROMPT_FILE"
echo "=============================================="

if [ $NUM_GPUS -eq 1 ]; then
    # Single GPU
    CUDA_VISIBLE_DEVICES=${GPUS[0]} $PYTHON generate.py \
        --config "$CONFIG" \
        --prompt_file "$PROMPT_FILE" \
        --save_dir "$SAVE_DIR" \
        --device "cuda:0"
else
    # Multi-GPU: launch parallel processes
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
            --total_gpus $NUM_GPUS &
        PIDS+=($!)
    done

    # Wait for all processes
    echo "Waiting for all workers..."
    for pid in "${PIDS[@]}"; do
        wait $pid
    done
fi

echo ""
echo "=============================================="
echo "Generation complete. Running evaluation..."
echo "=============================================="

# Run NudeNet evaluation
$PYTHON evaluate.py \
    --image_dir "$SAVE_DIR/all" \
    --eval_type nudenet \
    --output "$SAVE_DIR/eval_nudenet.json"

echo ""
echo "Results saved to: $SAVE_DIR"
echo "To run Qwen VLM evaluation:"
echo "  python evaluate.py --image_dir $SAVE_DIR/all --eval_type qwen"
