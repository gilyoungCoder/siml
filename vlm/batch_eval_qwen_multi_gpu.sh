#!/bin/bash
# ============================================================================
# Multi-GPU Batch Qwen2-VL Evaluation
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,3,4,5 ./batch_eval_qwen_multi_gpu.sh <dir1> [dir2] ...
# ============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs/qwen_eval_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

cd /mnt/home/yhgil99/unlearning

VLM_SCRIPT="vlm/opensource_vlm_i2p_all.py"

# Parse GPU list
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=(0 1 2 3)
fi
NUM_GPUS=${#GPU_LIST[@]}

echo "============================================================"
echo "Multi-GPU Qwen2-VL Evaluation"
echo "============================================================"
echo "GPUs: ${GPU_LIST[*]}"
echo "Log dir: ${LOG_DIR}"
echo ""

# Fixed concept for Ring-A-Bell evaluation
CONCEPT="nudity"

# Collect all directories to evaluate
declare -a ALL_DIRS=()

for base_dir in "$@"; do
    if [ ! -d "$base_dir" ]; then
        echo "Warning: Directory not found: $base_dir"
        continue
    fi

    # Check if base_dir itself contains images
    img_count=$(find "$base_dir" -maxdepth 1 -type f -name "*.png" 2>/dev/null | wc -l)
    if [ "$img_count" -gt 0 ]; then
        ALL_DIRS+=("$base_dir")
    fi

    # Check subdirectories
    for subdir in "$base_dir"/*/; do
        if [ -d "$subdir" ]; then
            # Skip logs folder
            if [[ "$(basename "$subdir")" == "logs" ]]; then
                continue
            fi
            img_count=$(find "$subdir" -maxdepth 1 -type f -name "*.png" 2>/dev/null | wc -l)
            if [ "$img_count" -gt 0 ]; then
                ALL_DIRS+=("$subdir")
            fi
        fi
    done
done

TOTAL_DIRS=${#ALL_DIRS[@]}
echo "Total directories to evaluate: ${TOTAL_DIRS}"
echo ""

if [ "$TOTAL_DIRS" -eq 0 ]; then
    echo "No directories with images found!"
    exit 1
fi

# Function to check if already evaluated
needs_evaluation() {
    local folder="$1"
    local concept="$2"
    local results_file="${folder}/results_qwen3_vl_${concept}.txt"
    if [ -f "$results_file" ]; then
        # Compare evaluated count vs actual image count
        local eval_total=$(grep -oP 'Total images: \K\d+' "$results_file" 2>/dev/null || echo "0")
        local actual_total=$(find "$folder" -maxdepth 1 -type f -name "*.png" 2>/dev/null | wc -l)
        if [ "$eval_total" -lt "$actual_total" ]; then
            echo "[WARN] ${folder}: evaluated=${eval_total}, actual=${actual_total}. Re-evaluating..."
            rm -f "$results_file"
            local categories_file="${folder}/categories_qwen3_vl_${concept}.json"
            rm -f "$categories_file"
            return 0
        fi
        return 1
    fi
    return 0
}

# Function to evaluate a directory
eval_dir() {
    local gpu_idx=$1
    local dir_path=$2
    local log_idx=$3

    local actual_gpu=${GPU_LIST[$gpu_idx]}
    local dir_name=$(basename "$dir_path")
    local log_file="${LOG_DIR}/eval_${log_idx}_${dir_name}.log"

    # Check if already evaluated
    if ! needs_evaluation "$dir_path" "$CONCEPT"; then
        echo "[GPU ${actual_gpu}] Skip (already done): ${dir_name}"
        return 0
    fi

    echo "[GPU ${actual_gpu}] Evaluating: ${dir_name} (concept: ${CONCEPT})"

    {
        echo "============================================================"
        echo "Directory: ${dir_path}"
        echo "Concept: ${CONCEPT}"
        echo "GPU: ${actual_gpu}"
        echo "Time: $(date)"
        echo "============================================================"
        echo ""

        CUDA_VISIBLE_DEVICES=${actual_gpu} python "$VLM_SCRIPT" "$dir_path" "$CONCEPT" qwen

        echo ""
        echo "Completed: $(date)"

    } > "$log_file" 2>&1

    echo "[GPU ${actual_gpu}] Done: ${dir_name}"
}

# Track running jobs per GPU
declare -A GPU_PIDS

# Process all directories
idx=0
for dir_path in "${ALL_DIRS[@]}"; do
    # Find available GPU
    while true; do
        for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
            pid="${GPU_PIDS[$gpu]:-}"

            # Check if slot is free
            if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
                # Run evaluation
                eval_dir $gpu "$dir_path" $idx &
                GPU_PIDS[$gpu]=$!

                idx=$((idx + 1))
                echo "Progress: ${idx}/${TOTAL_DIRS}"

                sleep 2
                break 2
            fi
        done

        sleep 5
    done
done

# Wait for all remaining jobs
echo ""
echo "Waiting for remaining jobs..."
wait

echo ""
echo "============================================================"
echo "QWEN EVALUATION COMPLETE"
echo "============================================================"
echo "Logs saved to: ${LOG_DIR}"
echo ""
echo "Done!"
