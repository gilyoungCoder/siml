#!/bin/bash
# Eval fine grid mon4class results with Qwen VLM (nudity)
# Usage:
#   bash vlm/eval_fine_grid_mon4class.sh --site A --nohup
#   bash vlm/eval_fine_grid_mon4class.sh --site B --nohup

cd /mnt/home/yhgil99/unlearning

NUM_GPUS=8
VLM_SCRIPT="vlm/opensource_vlm_i2p_all.py"
BASE_DIR="SoftDelete+CG/scg_outputs/fine_grid_mon4class"

SITE=""
NOHUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --site) SITE="$2"; shift 2;;
        --nohup) NOHUP=true; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [ -z "$SITE" ]; then
    echo "Usage: bash vlm/eval_fine_grid_mon4class.sh --site A|B [--nohup]"
    exit 1
fi

if [ "$NOHUP" = true ]; then
    LOGDIR="./results"
    mkdir -p "$LOGDIR"
    LOGFILE="$LOGDIR/eval_fine_grid_mon4class_site${SITE}_$(date +%Y%m%d_%H%M%S).log"
    echo "Running in background (site $SITE)..."
    echo "Log: $LOGFILE"
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    nohup bash "$SCRIPT_PATH" --site "$SITE" > "$LOGFILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

# Collect folders needing eval
DIRS=()
for ds in p4dn ringabell unlearndiff; do
    for cfg_dir in "$BASE_DIR/$ds"/mon*/; do
        [ -d "$cfg_dir" ] || continue
        # Must have images (generation_stats.json) but no eval yet
        [ -f "$cfg_dir/generation_stats.json" ] || continue
        [ -f "$cfg_dir/categories_qwen3_vl_nudity.json" ] && continue
        DIRS+=("$cfg_dir")
    done
done

TOTAL=${#DIRS[@]}
echo "=== Folders to eval: $TOTAL ==="

if [ $TOTAL -eq 0 ]; then
    echo "Nothing to evaluate."
    exit 0
fi

# Split
HALF=$(( (TOTAL + 1) / 2 ))
if [ "$SITE" = "A" ]; then
    START=0; END=$HALF
elif [ "$SITE" = "B" ]; then
    START=$HALF; END=$TOTAL
else
    echo "Invalid site: $SITE"; exit 1
fi

echo "Site $SITE: $START to $((END-1)) (total $((END-START)))"

# Parse GPU list
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=(0 1 2 3 4 5 6 7)
fi

PIDS=()
GPU_IDX=0

for (( i=START; i<END; i++ )); do
    dir="${DIRS[$i]}"
    gpu=${GPU_LIST[$((GPU_IDX % ${#GPU_LIST[@]}))]}
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
    echo "--- Waiting for final batch (${#PIDS[@]} jobs) ---"
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null; done
fi

echo ""
echo "=== Site $SITE eval complete ==="
echo "Run: python vlm/aggregate_fine_grid.py --base-dir SoftDelete+CG/scg_outputs/fine_grid_mon4class"
