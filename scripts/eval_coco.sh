#!/bin/bash
# ============================================================================
# Evaluate COCO generation metrics: FID, CLIP
#
# Usage:
#   bash eval_coco.sh <IMG_DIR1> [IMG_DIR2] ...
#   bash eval_coco.sh --metrics "fid clip" <IMG_DIR1> [IMG_DIR2] ...
#   bash eval_coco.sh --gpus 0,1,2,3,4 <IMG_DIR1> [IMG_DIR2] ...
#
# Each directory runs on a separate GPU in parallel.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
METRICS="all"
OUTPUT=""
GPUS="0,1,2,3,4,5,6,7"
IMG_DIRS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --metrics) METRICS="$2"; shift 2;;
        --output) OUTPUT="$2"; shift 2;;
        --gpus) GPUS="$2"; shift 2;;
        --help|-h)
            echo "Usage: $0 [--gpus 0,1,2,...] [--metrics \"fid clip\"] <IMG_DIR1> [IMG_DIR2] ..."
            exit 0;;
        *) IMG_DIRS+=("$1"); shift;;
    esac
done

if [ ${#IMG_DIRS[@]} -eq 0 ]; then
    echo "ERROR: No image directories specified"
    exit 1
fi

IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

echo "=============================================="
echo "COCO Metrics Evaluation (Parallel)"
echo "=============================================="
echo "GPUs: ${GPU_LIST[*]} (${NUM_GPUS})"
echo "Metrics: ${METRICS}"
echo "Directories: ${#IMG_DIRS[@]}"
for dir in "${IMG_DIRS[@]}"; do
    N_IMAGES=$(find "${dir}" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
    echo "  - ${dir} (${N_IMAGES} images)"
done
echo "=============================================="
echo ""

# Launch each directory on a separate GPU
PIDS=()
for i in "${!IMG_DIRS[@]}"; do
    GPU_IDX=$((i % NUM_GPUS))
    GPU=${GPU_LIST[$GPU_IDX]}
    DIR="${IMG_DIRS[$i]}"
    DIR_NAME=$(basename "$DIR")
    LOG_FILE="$(dirname "$DIR")/logs/eval_${DIR_NAME}.log"
    mkdir -p "$(dirname "$LOG_FILE")"

    OUTPUT_ARG=""
    if [ -n "$OUTPUT" ]; then
        OUTPUT_ARG="--output ${OUTPUT%.csv}_${DIR_NAME}.csv"
    fi

    echo "[GPU $GPU] Evaluating: $DIR_NAME -> $LOG_FILE"
    CUDA_VISIBLE_DEVICES=$GPU python "${SCRIPT_DIR}/evaluate_coco_metrics.py" \
        --img_dirs "$DIR" \
        --device cuda:0 \
        --metrics ${METRICS} \
        --batch_size 64 \
        ${OUTPUT_ARG} \
        > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "--- Waiting for ${#PIDS[@]} jobs ---"
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || FAIL=$((FAIL + 1))
done

echo ""
echo "=============================================="
echo "Evaluation Complete!"
[ $FAIL -gt 0 ] && echo "WARNING: ${FAIL} jobs failed"
echo "=============================================="

# Print results summary
echo ""
echo "Results:"
for DIR in "${IMG_DIRS[@]}"; do
    JSON="${DIR}/eval_metrics.json"
    if [ -f "$JSON" ]; then
        DIR_NAME=$(basename "$DIR")
        FID=$(python3 -c "import json; d=json.load(open('$JSON')); print(f'{d.get(\"fid\", \"N/A\"):.2f}' if isinstance(d.get('fid'), float) else 'N/A')" 2>/dev/null)
        CLIP=$(python3 -c "import json; d=json.load(open('$JSON')); print(f'{d.get(\"clip_score\", \"N/A\"):.4f}' if isinstance(d.get('clip_score'), float) else 'N/A')" 2>/dev/null)
        printf "  %-20s FID: %8s  CLIP: %s\n" "$DIR_NAME" "$FID" "$CLIP"
    fi
done
