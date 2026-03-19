#!/bin/bash
# ============================================================================
# Run all 5 datasets × 7 methods across 8 GPUs
#
# Each (dataset, method) pair is a separate job. Jobs are distributed
# round-robin across available GPUs. A semaphore limits concurrency to
# NUM_GPUS simultaneous jobs.
#
# Usage:
#   ./scripts/run_all_datasets.sh                    # all datasets, all methods, 8 GPUs
#   ./scripts/run_all_datasets.sh --gpus 0,1,2,3     # use specific GPUs
#   ./scripts/run_all_datasets.sh --methods "sd_baseline safree"  # specific methods only
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_SCRIPT="${SCRIPT_DIR}/generate_all_methods.sh"

# Defaults
GPU_LIST="0,1,2,3,4,5,6,7"
METHODS_ARG=""

# Parse args
while [ $# -gt 0 ]; do
    case "$1" in
        --gpus)     GPU_LIST="$2"; shift 2 ;;
        --methods)  METHODS_ARG="$2"; shift 2 ;;
        *)          echo "Unknown: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

DATASETS=("ringabell" "i2p" "p4dn" "unlearndiff" "mma")

if [ -n "$METHODS_ARG" ]; then
    read -ra METHODS <<< "$METHODS_ARG"
else
    METHODS=("sd_baseline" "safree" "ours_dual" "ours_mon4class" "ours_mon3class" "safree_dual" "safree_mon")
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/logs_all_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

# Build job list: (dataset, method)
JOBS=()
for ds in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        JOBS+=("${ds}|${method}")
    done
done

TOTAL_JOBS=${#JOBS[@]}

echo "============================================================"
echo "Parallel Generation: ${TOTAL_JOBS} jobs on ${NUM_GPUS} GPUs"
echo "============================================================"
echo "Datasets: ${DATASETS[*]}"
echo "Methods:  ${METHODS[*]}"
echo "GPUs:     ${GPUS[*]}"
echo "Logs:     ${LOG_DIR}"
echo "============================================================"
echo ""

# Semaphore: track running jobs per GPU
declare -A GPU_PIDS  # gpu -> current PID

PIDS=()
JOB_NAMES=()
JOB_LOGS=()

job_idx=0
for job in "${JOBS[@]}"; do
    IFS='|' read -r ds method <<< "$job"

    # Pick GPU round-robin
    gpu_idx=$((job_idx % NUM_GPUS))
    gpu="${GPUS[$gpu_idx]}"

    # Wait for previous job on this GPU to finish
    prev_pid="${GPU_PIDS[$gpu]:-}"
    if [ -n "$prev_pid" ]; then
        wait "$prev_pid" 2>/dev/null || true
    fi

    log="${LOG_DIR}/${ds}_${method}.log"
    echo "[${job_idx}/${TOTAL_JOBS}] ${ds}/${method} → GPU ${gpu}"

    bash "${GEN_SCRIPT}" --dataset "${ds}" --methods "${method}" --gpu "${gpu}" \
        > "${log}" 2>&1 &
    pid=$!

    GPU_PIDS[$gpu]=$pid
    PIDS+=($pid)
    JOB_NAMES+=("${ds}/${method}")
    JOB_LOGS+=("${log}")

    job_idx=$((job_idx + 1))
done

echo ""
echo "All ${TOTAL_JOBS} jobs launched. Waiting..."
echo "Monitor: tail -f ${LOG_DIR}/*.log"
echo ""

# Wait for all
FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" 2>/dev/null
    status=$?
    if [ ${status} -ne 0 ]; then
        echo "[FAIL] ${JOB_NAMES[$i]} (exit ${status}) — ${JOB_LOGS[$i]}"
        FAILED=$((FAILED + 1))
    else
        echo "[DONE] ${JOB_NAMES[$i]}"
    fi
done

echo ""
echo "============================================================"
if [ ${FAILED} -eq 0 ]; then
    echo "ALL ${TOTAL_JOBS} JOBS COMPLETE"
else
    echo "${FAILED}/${TOTAL_JOBS} jobs failed. Check: ${LOG_DIR}"
fi
echo "============================================================"
