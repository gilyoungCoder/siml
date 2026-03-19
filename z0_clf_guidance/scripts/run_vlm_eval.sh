#!/bin/bash
# ============================================================================
# VLM Evaluation for Grid Search Results (Qwen3-VL)
#
# Runs VLM eval on all generated experiment directories that have images
# but no evaluation results yet.
#
# Usage:
#   bash scripts/run_vlm_eval.sh --dataset ringabell              # dry run
#   bash scripts/run_vlm_eval.sh --dataset ringabell --run         # execute
#   bash scripts/run_vlm_eval.sh --dataset ringabell --run --nohup # background
#
# Multi-server MMA example:
#   Server 1: bash scripts/run_vlm_eval.sh --dataset mma --start-idx 0 --end-idx 250 --run --nohup
#   Server 2: bash scripts/run_vlm_eval.sh --dataset mma --start-idx 250 --end-idx 500 --run --nohup
# ============================================================================

cd /mnt/home/yhgil99/unlearning/z0_clf_guidance

# Activate conda environment (VLM eval uses vlm env)
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate vlm

# ============================================
# Parse Arguments
# ============================================
DRY_RUN=true
USE_NOHUP=false
NUM_GPUS=8
JOBS_PER_GPU=2
DATASET=""
START_CONFIG=0
END_CONFIG=-1

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2;;
        --run) DRY_RUN=false; shift;;
        --dry-run) DRY_RUN=true; shift;;
        --nohup) USE_NOHUP=true; shift;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2;;
        --start-idx) START_CONFIG="$2"; shift 2;;
        --end-idx) END_CONFIG="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "ERROR: --dataset is required."
    echo "Available datasets: ringabell, unlearndiff, mma"
    exit 1
fi

VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
OUTPUT_BASE="./grid_monitoring_output/${DATASET}"

if [ ! -d "$OUTPUT_BASE" ]; then
    echo "ERROR: Output dir not found: $OUTPUT_BASE"
    exit 1
fi

# Nohup handling
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="./grid_monitoring_output/logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/vlm_eval_${DATASET}_${TIMESTAMP}.log"
    echo "Running in background..."
    echo "Log: $LOG_FILE"
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    nohup bash "$SCRIPT_PATH" --dataset "$DATASET" --run --num-gpus "$NUM_GPUS" --jobs-per-gpu "$JOBS_PER_GPU" --start-idx "$START_CONFIG" --end-idx "$END_CONFIG" > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    echo "Monitor: tail -f $LOG_FILE"
    exit 0
fi

# ============================================
# COLLECT DIRECTORIES TO EVALUATE
# ============================================

declare -a EVAL_DIRS=()
declare -a SKIP_DIRS=()
declare -a NO_GEN_DIRS=()

for exp_dir in $(ls -d "${OUTPUT_BASE}"/mon* 2>/dev/null | sort); do
    if [ ! -d "$exp_dir" ]; then
        continue
    fi

    exp_name=$(basename "$exp_dir")

    # Skip if no generation_stats.json (not generated yet)
    if [ ! -f "${exp_dir}/generation_stats.json" ]; then
        NO_GEN_DIRS+=("$exp_name")
        continue
    fi

    # Skip if already evaluated
    if [ -f "${exp_dir}/categories_qwen3_vl_nudity.json" ]; then
        SKIP_DIRS+=("$exp_name")
        continue
    fi

    EVAL_DIRS+=("$exp_name")
done

# Apply start/end range
TOTAL_EVAL=${#EVAL_DIRS[@]}
if [ "$END_CONFIG" -eq -1 ]; then
    END_CONFIG=$TOTAL_EVAL
fi
if [ "$START_CONFIG" -gt 0 ] || [ "$END_CONFIG" -lt "$TOTAL_EVAL" ]; then
    EVAL_DIRS=("${EVAL_DIRS[@]:$START_CONFIG:$((END_CONFIG - START_CONFIG))}")
fi

TOTAL=${#EVAL_DIRS[@]}

echo "=============================================="
echo "VLM EVALUATION (Qwen3-VL, nudity)"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Output base: $OUTPUT_BASE"
echo "To evaluate: $TOTAL"
echo "Already done: ${#SKIP_DIRS[@]}"
echo "No generation: ${#NO_GEN_DIRS[@]}"
echo "GPUs: $NUM_GPUS (${JOBS_PER_GPU} jobs/GPU = $((NUM_GPUS * JOBS_PER_GPU)) concurrent)"
if [ "$START_CONFIG" -ne 0 ] || [ "$END_CONFIG" -ne "$TOTAL_EVAL" ]; then
    echo "Config range: [${START_CONFIG}:${END_CONFIG}] of ${TOTAL_EVAL}"
fi
echo "=============================================="

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY RUN] Directories to evaluate:"
    for (( i=0; i<TOTAL; i++ )); do
        GPU_IDX=$((i % NUM_GPUS))
        echo "  [$i] GPU${GPU_IDX} ${EVAL_DIRS[$i]}"
    done
    echo ""
    echo "Total: $TOTAL evals on $NUM_GPUS GPUs"
    echo "Add --run to execute."
    exit 0
fi

# ============================================
# GPU SETUP
# ============================================

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_LIST+=($i); done
fi

mkdir -p "${OUTPUT_BASE}/logs"

# ============================================
# RUN FUNCTION
# ============================================

run_vlm_eval() {
    local GPU_IDX=$1
    local EXP_NAME=$2

    local ACTUAL_GPU=${GPU_LIST[$GPU_IDX]}
    local EXP_DIR="${OUTPUT_BASE}/${EXP_NAME}"
    local LOG_FILE="${OUTPUT_BASE}/logs/vlm_${EXP_NAME}.log"

    # Double check not already done
    if [ -f "${EXP_DIR}/categories_qwen3_vl_nudity.json" ]; then
        echo "[GPU ${ACTUAL_GPU}] SKIP (done): ${EXP_NAME}"
        return 0
    fi

    echo "[GPU ${ACTUAL_GPU}] EVAL: ${EXP_NAME}"

    CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python "${VLM_SCRIPT}" "${EXP_DIR}" nudity qwen \
        >> "${LOG_FILE}" 2>&1

    if [ $? -eq 0 ]; then
        echo "[GPU ${ACTUAL_GPU}] DONE: ${EXP_NAME}"
    else
        echo "[GPU ${ACTUAL_GPU}] FAILED: ${EXP_NAME}"
        return 1
    fi

    return 0
}

# ============================================
# MAIN LOOP
# ============================================

TOTAL_SLOTS=$((NUM_GPUS * JOBS_PER_GPU))
declare -A RUNNING_JOBS
IDX=0

for (( ci=0; ci<TOTAL; ci++ )); do
    EXP_NAME="${EVAL_DIRS[$ci]}"

    while true; do
        for ((slot=0; slot<TOTAL_SLOTS; slot++)); do
            JOB_PID="${RUNNING_JOBS[$slot]}"
            if [ -z "$JOB_PID" ] || ! kill -0 "$JOB_PID" 2>/dev/null; then
                GPU_IDX=$((slot % NUM_GPUS))
                run_vlm_eval $GPU_IDX "$EXP_NAME" &
                RUNNING_JOBS[$slot]=$!
                IDX=$((IDX + 1))
                echo "Progress: ${IDX}/${TOTAL}"
                sleep 1
                break 2
            fi
        done
        sleep 5
    done
done

echo ""
echo "Waiting for remaining jobs..."
wait

# Summary
COMPLETED=0
for exp_name in "${EVAL_DIRS[@]}"; do
    if [ -f "${OUTPUT_BASE}/${exp_name}/categories_qwen3_vl_nudity.json" ]; then
        COMPLETED=$((COMPLETED + 1))
    fi
done

echo ""
echo "=============================================="
echo "VLM EVALUATION COMPLETE"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Evaluated: ${COMPLETED}/${TOTAL}"
echo "=============================================="
