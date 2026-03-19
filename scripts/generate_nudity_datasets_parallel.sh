#!/bin/bash
# ============================================================================
# Generate images for nudity datasets - PARALLEL execution
# Each method runs on a different GPU
#
# Usage: bash generate_nudity_datasets_parallel.sh <DATASET> [GPU_START]
# Example: bash generate_nudity_datasets_parallel.sh all 0
#          -> baseline on GPU0, safree on GPU1, ours on GPU2, safree_ours on GPU3
# ============================================================================

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <DATASET> [GPU_START]"
    echo ""
    echo "DATASET options:"
    echo "  ringabell  - nudity-ring-a-bell.csv (79 prompts)"
    echo "  mma        - mma-diffusion-nsfw-adv-prompts.csv (1000 prompts)"
    echo "  nudity     - nudity.csv (142 prompts)"
    echo "  all        - all datasets"
    echo ""
    echo "GPU_START: Starting GPU number (default: 0)"
    echo "  Methods will use GPU_START, GPU_START+1, GPU_START+2, GPU_START+3"
    exit 1
fi

DATASET=$1
GPU_START=${2:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/generate_nudity_datasets.sh"

# Calculate GPU assignments
GPU_BASELINE=$((GPU_START))
GPU_SAFREE=$((GPU_START + 1))
GPU_OURS=$((GPU_START + 2))
GPU_SAFREE_OURS=$((GPU_START + 3))

echo "=============================================="
echo "Parallel Nudity Dataset Generation"
echo "=============================================="
echo "Dataset: ${DATASET}"
echo ""
echo "GPU Assignment:"
echo "  GPU ${GPU_BASELINE}: SD Baseline"
echo "  GPU ${GPU_SAFREE}: SAFREE"
echo "  GPU ${GPU_OURS}: Ours 4class"
echo "  GPU ${GPU_SAFREE_OURS}: SAFREE+Ours 4class"
echo "=============================================="
echo ""

# Show GPU status
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader | head -8
echo ""

# Skip confirmation if running non-interactively (nohup)
if [ -t 0 ]; then
    read -p "Start generation on GPUs ${GPU_START}-$((GPU_START+3))? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
else
    echo "Running non-interactively, skipping confirmation..."
fi

# Create log directory
LOG_DIR="/mnt/home/yhgil99/unlearning/logs/nudity_datasets_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo ""
echo "Starting parallel generation..."
echo "Logs: ${LOG_DIR}/"
echo ""

# Run all methods in parallel with nohup
nohup bash "${BASE_SCRIPT}" ${GPU_BASELINE} ${DATASET} baseline > "${LOG_DIR}/baseline.log" 2>&1 &
PID_BASELINE=$!
echo "[GPU ${GPU_BASELINE}] Baseline started (PID: ${PID_BASELINE})"

nohup bash "${BASE_SCRIPT}" ${GPU_SAFREE} ${DATASET} safree > "${LOG_DIR}/safree.log" 2>&1 &
PID_SAFREE=$!
echo "[GPU ${GPU_SAFREE}] SAFREE started (PID: ${PID_SAFREE})"

nohup bash "${BASE_SCRIPT}" ${GPU_OURS} ${DATASET} ours > "${LOG_DIR}/ours.log" 2>&1 &
PID_OURS=$!
echo "[GPU ${GPU_OURS}] Ours 4class started (PID: ${PID_OURS})"

nohup bash "${BASE_SCRIPT}" ${GPU_SAFREE_OURS} ${DATASET} safree_ours > "${LOG_DIR}/safree_ours.log" 2>&1 &
PID_SAFREE_OURS=$!
echo "[GPU ${GPU_SAFREE_OURS}] SAFREE+Ours started (PID: ${PID_SAFREE_OURS})"

echo ""
echo "=============================================="
echo "All jobs started!"
echo "=============================================="
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOG_DIR}/baseline.log"
echo "  tail -f ${LOG_DIR}/safree.log"
echo "  tail -f ${LOG_DIR}/ours.log"
echo "  tail -f ${LOG_DIR}/safree_ours.log"
echo ""
echo "Or watch all:"
echo "  watch -n 5 'for f in ${LOG_DIR}/*.log; do echo \"=== \$f ===\"; tail -3 \$f; done'"
echo ""
echo "PIDs: ${PID_BASELINE} ${PID_SAFREE} ${PID_OURS} ${PID_SAFREE_OURS}"
echo ""
