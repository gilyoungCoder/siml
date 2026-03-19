#!/bin/bash
# Compute FID only for mon01_grid/coco configs
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash vlm/eval_mon01_grid_fid.sh --nohup

cd /mnt/home/yhgil99/unlearning

EVAL_SCRIPT="scripts/evaluate_coco_metrics.py"
BASE_DIR="SoftDelete+CG/scg_outputs/mon01_grid/coco"
REF_DIR="SoftDelete+CG/scg_outputs/final_coco/sd_baseline"
NOHUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --nohup) NOHUP=true; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [ "$NOHUP" = true ]; then
    LOGFILE="results/eval_mon01_grid_fid_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p results
    echo "Running in background... Log: $LOGFILE"
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    nohup bash "$SCRIPT_PATH" > "$LOGFILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

# Collect coco dirs
DIRS=()
for config_dir in "$BASE_DIR"/*/; do
    [ -d "$config_dir" ] || continue
    IMG_COUNT=$(ls "$config_dir"/*.png 2>/dev/null | wc -l)
    [ "$IMG_COUNT" -gt 0 ] || continue
    DIRS+=("$config_dir")
done

TOTAL=${#DIRS[@]}
echo "=== mon01_grid/coco FID folders: $TOTAL ==="

if [ $TOTAL -eq 0 ]; then
    echo "Nothing to evaluate."
    exit 0
fi

# GPU setup
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=(0 1 2 3 4 5 6 7)
fi
NUM_GPUS=${#GPU_LIST[@]}

PIDS=()
GPU_IDX=0

for dir in "${DIRS[@]}"; do
    gpu=${GPU_LIST[$((GPU_IDX % NUM_GPUS))]}
    echo "[GPU $gpu] FID: $dir"
    CUDA_VISIBLE_DEVICES=$gpu python "$EVAL_SCRIPT" \
        --img_dirs "$dir" \
        --metrics fid \
        --device cuda:0 \
        --batch_size 512 &
    PIDS+=($!)
    GPU_IDX=$((GPU_IDX + 1))

    if [ $((GPU_IDX % NUM_GPUS)) -eq 0 ]; then
        echo "--- Waiting for batch (${#PIDS[@]} jobs) ---"
        for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null; done
        PIDS=()
    fi
done

if [ ${#PIDS[@]} -gt 0 ]; then
    echo "--- Waiting for final batch ---"
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null; done
fi

echo ""
echo "=== mon01_grid FID eval complete ==="
