#!/bin/bash
# ============================================================================
# Monitoring Trigger Rate Test (GradCAM CDF)
#
# Runs COCO (benign, FP) and ringabell (harmful, TP) through denoising
# without guidance, measuring P(harm) at each step.
#
# Usage:
#   bash scripts/run_monitoring_test.sh                  # dry run
#   bash scripts/run_monitoring_test.sh --run             # execute
#   bash scripts/run_monitoring_test.sh --run --nohup     # background
# ============================================================================

cd /mnt/home/yhgil99/unlearning/z0_clf_guidance
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd

# ============================================
# Parse Arguments
# ============================================
DRY_RUN=true
USE_NOHUP=false
NUM_GPUS=8
DATASETS="coco ringabell"

while [[ $# -gt 0 ]]; do
    case $1 in
        --run) DRY_RUN=false; shift;;
        --dry-run) DRY_RUN=true; shift;;
        --nohup) USE_NOHUP=true; shift;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        --datasets) DATASETS="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

# Nohup handling
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="./monitoring_test/logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/monitoring_test_${TIMESTAMP}.log"
    echo "Running in background..."
    echo "Log: $LOG_FILE"
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    nohup bash "$SCRIPT_PATH" --run --num-gpus "$NUM_GPUS" --datasets "$DATASETS" > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    echo "Monitor: tail -f $LOG_FILE"
    exit 0
fi

OUTPUT_DIR="./monitoring_test"
mkdir -p "$OUTPUT_DIR"

# GPU list
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_LIST+=($i); done
fi

echo "=============================================="
echo "MONITORING TRIGGER RATE TEST (GradCAM CDF)"
echo "=============================================="
echo "Datasets: $DATASETS"
echo "GPUs: ${GPU_LIST[*]} (${NUM_GPUS} total)"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY RUN] Would run:"
    for ds in $DATASETS; do
        echo "  - $ds on ${NUM_GPUS} GPUs"
    done
    echo ""
    echo "Add --run to execute."
    exit 0
fi

# ============================================
# Run tests
# ============================================

for ds in $DATASETS; do
    echo ""
    echo "=== Starting: $ds ==="

    # Launch one process per GPU
    PIDS=()
    for ((i=0; i<NUM_GPUS; i++)); do
        ACTUAL_GPU=${GPU_LIST[$i]}
        echo "[GPU ${ACTUAL_GPU}] Starting $ds (chunk $i/${NUM_GPUS})"
        CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python test_monitoring.py \
            --prompt_file "$ds" \
            --gpu_id $i \
            --num_gpus $NUM_GPUS \
            --output_dir "$OUTPUT_DIR" \
            > "${OUTPUT_DIR}/${ds}_gpu${i}.log" 2>&1 &
        PIDS+=($!)
    done

    # Wait for all GPUs
    echo "Waiting for $ds to finish (${#PIDS[@]} processes)..."
    FAILED=0
    for pid in "${PIDS[@]}"; do
        wait $pid
        if [ $? -ne 0 ]; then
            FAILED=$((FAILED + 1))
        fi
    done

    if [ $FAILED -gt 0 ]; then
        echo "WARNING: $FAILED GPU processes failed for $ds"
    fi

    # Merge results
    echo "Merging $ds results..."
    python test_monitoring.py --prompt_file "$ds" --merge --output_dir "$OUTPUT_DIR"
    echo ""
done

echo "=============================================="
echo "ALL TESTS COMPLETE"
echo "=============================================="
echo "Results in: $OUTPUT_DIR"
echo ""

# Print combined summary
echo "=== COMBINED SUMMARY ==="
for ds in $DATASETS; do
    merged="${OUTPUT_DIR}/${ds}_merged.json"
    if [ -f "$merged" ]; then
        echo ""
        python -c "
import json
import numpy as np
with open('$merged') as f:
    data = json.load(f)
label = data['label']
n = len(data['results'])
print(f'--- {\"$ds\"} ({label}, N={n}) ---')
print(f'  mu={data[\"mon_mu\"]:.4f}, sigma={data[\"mon_sigma\"]:.4f}')
print(f'{\"Threshold\":>12} | {\"Rate\":>10} | {\"Count\":>8} | {\"Avg Steps\":>12}')
print(f\"{\"-\"*12}-+-{\"-\"*10}-+-{\"-\"*8}-+-{\"-\"*12}\")
for thr in ['0.05','0.10','0.15','0.20','0.25','0.30','0.40','0.50','0.60','0.70','0.80','0.90']:
    if thr in data['trigger_stats']:
        s = data['trigger_stats'][thr]
        print(f'  {s[\"threshold\"]:>10.2f} | {s[\"trigger_rate\"]:>9.1%} | {s[\"trigger_count\"]:>8d} | {s[\"avg_guided_steps\"]:>12.1f}')
"
    fi
done
