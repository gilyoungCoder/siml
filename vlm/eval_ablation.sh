#!/bin/bash
# Eval ablation results with Qwen VLM
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash vlm/eval_ablation.sh --nohup

cd /mnt/home/yhgil99/unlearning

VLM_SCRIPT="vlm/opensource_vlm_i2p_all.py"
BASE_DIR="SoftDelete+CG/scg_outputs/ablation"
NOHUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --nohup) NOHUP=true; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [ "$NOHUP" = true ]; then
    LOGFILE="results/eval_ablation_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p results
    echo "Running in background... Log: $LOGFILE"
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    nohup bash "$SCRIPT_PATH" > "$LOGFILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

# Collect dirs needing eval
DIRS=()
for cond in full_cg full_cg_mon spatial_cg; do
    for ds_dir in "$BASE_DIR/$cond"/*/; do
        [ -d "$ds_dir" ] || continue
        for gs_dir in "$ds_dir"gs*/; do
            [ -d "$gs_dir" ] || continue
            # Must have images but no eval yet
            IMG_COUNT=$(ls "$gs_dir"/*.png 2>/dev/null | wc -l)
            [ "$IMG_COUNT" -gt 0 ] || continue
            [ -f "$gs_dir/categories_qwen3_vl_nudity.json" ] && continue
            DIRS+=("$gs_dir")
        done
    done
done

TOTAL=${#DIRS[@]}
echo "=== Ablation folders to eval: $TOTAL ==="

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
    echo "[GPU $gpu] Evaluating: $dir"
    CUDA_VISIBLE_DEVICES=$gpu python "$VLM_SCRIPT" "$dir" nudity qwen &
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
echo "=== Ablation eval complete ==="
