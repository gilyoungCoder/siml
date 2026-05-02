#!/bin/bash
# Evaluate best7 SAFREE+Mon config folders with Qwen VLM (nudity)
# Usage:
#   bash vlm/eval_best7_safree_mon.sh --site A   (GPU 0-7, first half)
#   bash vlm/eval_best7_safree_mon.sh --site B   (GPU 0-7, second half)
#   bash vlm/eval_best7_safree_mon.sh --nohup --site A  (background)

cd /mnt/home/yhgil99/unlearning

NUM_GPUS=8
VLM_SCRIPT="vlm/opensource_vlm_i2p_all.py"
BASE_DIR="SoftDelete+CG/scg_outputs"

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
    echo "Usage: bash vlm/eval_best7_safree_mon.sh --site A|B [--nohup]"
    exit 1
fi

# Nohup handling
if [ "$NOHUP" = true ]; then
    LOGDIR="./results"
    mkdir -p "$LOGDIR"
    LOGFILE="$LOGDIR/eval_best7_site${SITE}_$(date +%Y%m%d_%H%M%S).log"
    echo "Running in background (site $SITE)..."
    echo "Log: $LOGFILE"
    nohup bash "$0" --site "$SITE" > "$LOGFILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

# Collect all config folders that need eval
DIRS=()
for ds in i2p mma p4dn ringabell unlearndiff; do
    for cfg_dir in "$BASE_DIR/final_${ds}/safree_mon"/mon*/; do
        [ -d "$cfg_dir" ] || continue
        if [ -f "$cfg_dir/categories_qwen3_vl_nudity.json" ]; then
            echo "[SKIP] $cfg_dir (already evaluated)"
            continue
        fi
        DIRS+=("$cfg_dir")
    done
done

TOTAL=${#DIRS[@]}
echo "=== Total folders to evaluate: $TOTAL ==="

if [ $TOTAL -eq 0 ]; then
    echo "Nothing to evaluate."
    exit 0
fi

# Split for site A (first half) / B (second half)
HALF=$(( (TOTAL + 1) / 2 ))
if [ "$SITE" = "A" ]; then
    START=0
    END=$HALF
elif [ "$SITE" = "B" ]; then
    START=$HALF
    END=$TOTAL
else
    echo "Invalid site: $SITE (use A or B)"
    exit 1
fi

echo "=== Site $SITE: folders $START to $((END-1)) (total $((END-START))) ==="

# Launch in batches of NUM_GPUS
PIDS=()
GPU_IDX=0

for (( i=START; i<END; i++ )); do
    dir="${DIRS[$i]}"
    gpu=$((GPU_IDX % NUM_GPUS))
    echo "[GPU $gpu] Evaluating: $dir"
    CUDA_VISIBLE_DEVICES=$gpu python "$VLM_SCRIPT" "$dir" nudity qwen &
    PIDS+=($!)
    GPU_IDX=$((GPU_IDX + 1))

    if [ $((GPU_IDX % NUM_GPUS)) -eq 0 ]; then
        echo "--- Waiting for batch to finish (${#PIDS[@]} jobs) ---"
        for pid in "${PIDS[@]}"; do
            wait "$pid"
        done
        PIDS=()
    fi
done

if [ ${#PIDS[@]} -gt 0 ]; then
    echo "--- Waiting for final batch (${#PIDS[@]} jobs) ---"
    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done
fi

echo ""
echo "=== Site $SITE evaluations complete ==="
echo ""

# Print summary
for ds in i2p mma p4dn ringabell unlearndiff; do
    echo "--- $ds ---"
    for cfg_dir in "$BASE_DIR/final_${ds}/safree_mon"/mon*/; do
        cfg=$(basename "$cfg_dir")
        result="$cfg_dir/results_qwen3_vl_nudity.txt"
        if [ -f "$result" ]; then
            echo "  $cfg: $(cat "$result" | tail -1)"
        else
            echo "  $cfg: NO RESULT"
        fi
    done
done
