#!/bin/bash
# ============================================================================
# Multi-GPU Grid Search: Sample-Level Monitoring + Spatial CG
# ============================================================================
# Distributes experiments across multiple GPUs using round-robin assignment
#
# Usage:
#   ./grid_search_multi_gpu.sh                    # Run all experiments
#   ./grid_search_multi_gpu.sh --dry-run          # Print commands without running
#   ./grid_search_multi_gpu.sh --concept sexual   # Use specific concept
#   ./grid_search_multi_gpu.sh --nohup            # Run in background with nohup
# ============================================================================

set -e

# === Parse arguments ===
DRY_RUN=false
CONCEPT="sexual"
NUM_GPUS=8
USE_NOHUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --concept)
            CONCEPT="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --nohup)
            USE_NOHUP=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# If --nohup flag is set, re-run this script in background
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="./scg_outputs/grid_search_3class_ringabell_${TIMESTAMP}.log"
    mkdir -p ./scg_outputs
    echo "Running in background with nohup..."
    echo "Log file: ${LOG_FILE}"
    nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$0" --concept "${CONCEPT}" --num-gpus "${NUM_GPUS}" > "${LOG_FILE}" 2>&1 &
    echo "PID: $!"
    echo "Use 'tail -f ${LOG_FILE}' to monitor progress"
    exit 0
fi

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# === Parse GPU list from CUDA_VISIBLE_DEVICES ===
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do
        GPU_LIST+=($i)
    done
fi
echo "Available GPUs: ${GPU_LIST[*]}"

# === Configuration (3-class adaptive spatial CG with skip_safe) ===
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
SD_MODEL="CompVis/stable-diffusion-v1-4"

# Prompt file
PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"

# Output base directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="./scg_outputs/grid_search_3class_ringabell_${TIMESTAMP}"

# Fixed parameters
NSAMPLES=1
CFG_SCALE=7.5
NUM_STEPS=50
THR_STRATEGY="cosine_anneal"

# ============================================================================
# Grid Search Parameters
# ============================================================================

GUIDANCE_SCALES=(7.5 10 12.5 15)
HARMFUL_SCALES=(1.5)
SPATIAL_THR_STARTS=(0.1)
SPATIAL_THR_ENDS=(0.4)
BASE_GUIDANCE_SCALES=(1.0 2.0)

# Calculate total experiments
TOTAL_GS=${#GUIDANCE_SCALES[@]}
TOTAL_HS=${#HARMFUL_SCALES[@]}
TOTAL_SS=${#SPATIAL_THR_STARTS[@]}
TOTAL_SE=${#SPATIAL_THR_ENDS[@]}
TOTAL_BS=${#BASE_GUIDANCE_SCALES[@]}
TOTAL_EXPERIMENTS=$((TOTAL_GS * TOTAL_HS * TOTAL_SS * TOTAL_SE * TOTAL_BS))

echo "=============================================="
echo "MULTI-GPU GRID SEARCH"
echo "=============================================="
echo "Concept: ${CONCEPT}"
echo "Prompt file: ${PROMPT_FILE}"
echo "Output base: ${BASE_OUTPUT_DIR}"
echo ""
echo "Parameters:"
echo "  guidance_scales: ${GUIDANCE_SCALES[*]}"
echo "  harmful_scales: ${HARMFUL_SCALES[*]}"
echo "  spatial_threshold_starts: ${SPATIAL_THR_STARTS[*]}"
echo "  spatial_threshold_ends: ${SPATIAL_THR_ENDS[*]}"
echo "  base_guidance_scales: ${BASE_GUIDANCE_SCALES[*]}"
echo "  spatial_threshold_strategy: ${THR_STRATEGY} (fixed)"
echo "  mode: 3-class skip_safe + bidirectional"
echo ""
echo "Total experiments: ${TOTAL_EXPERIMENTS}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "=============================================="
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN MODE]"
    echo ""
fi

# Create output directory
mkdir -p "${BASE_OUTPUT_DIR}"

# Save config
cat > "${BASE_OUTPUT_DIR}/config.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "concept": "${CONCEPT}",
    "prompt_file": "${PROMPT_FILE}",
    "total_experiments": ${TOTAL_EXPERIMENTS},
    "num_gpus": ${NUM_GPUS},
    "parameters": {
        "guidance_scales": [${GUIDANCE_SCALES[*]// /, }],
        "harmful_scales": [${HARMFUL_SCALES[*]// /, }],
        "spatial_threshold_starts": [${SPATIAL_THR_STARTS[*]// /, }],
        "spatial_threshold_ends": [${SPATIAL_THR_ENDS[*]// /, }],
        "base_guidance_scales": [${BASE_GUIDANCE_SCALES[*]// /, }],
        "spatial_threshold_strategy": "${THR_STRATEGY}"
    }
}
EOF

# Generate all experiment commands and distribute across GPUs
EXP_IDX=0
declare -a GPU_PIDS

# Function to wait for a GPU slot
wait_for_gpu_slot() {
    local gpu_id=$1
    if [ ! -z "${GPU_PIDS[$gpu_id]}" ]; then
        wait ${GPU_PIDS[$gpu_id]} 2>/dev/null || true
    fi
}

# Function to run experiment on GPU
run_on_gpu() {
    local gpu_idx=$1
    local exp_name=$2
    local gs=$3
    local hs=$4
    local sp_start=$5
    local sp_end=$6
    local base_gs=$7

    # Map GPU index to actual GPU ID
    local actual_gpu=${GPU_LIST[$gpu_idx]}

    local output_dir="${BASE_OUTPUT_DIR}/${exp_name}"
    local log_file="${BASE_OUTPUT_DIR}/logs/${exp_name}.log"

    mkdir -p "${BASE_OUTPUT_DIR}/logs"

    if [ "$DRY_RUN" = true ]; then
        echo "[GPU ${actual_gpu}] ${exp_name}"
        return 0
    fi

    echo "[GPU ${actual_gpu}] Starting: ${exp_name}"

    CUDA_VISIBLE_DEVICES=${actual_gpu} python generate_adaptive_spatial_cg.py \
        "${SD_MODEL}" \
        --prompt_file "${PROMPT_FILE}" \
        --output_dir "${output_dir}" \
        --classifier_ckpt "${CLASSIFIER_CKPT}" \
        --num_classes 3 \
        --harmful_class 2 \
        --safe_class 1 \
        --nsamples ${NSAMPLES} \
        --cfg_scale ${CFG_SCALE} \
        --num_inference_steps ${NUM_STEPS} \
        --seed 42 \
        --guidance_scale ${gs} \
        --harmful_scale ${hs} \
        --base_guidance_scale ${base_gs} \
        --spatial_threshold_start ${sp_start} \
        --spatial_threshold_end ${sp_end} \
        --threshold_strategy "${THR_STRATEGY}" \
        --guidance_start_step 0 \
        --guidance_end_step 50 \
        --use_bidirectional \
        --skip_safe \
        > "${log_file}" 2>&1 &

    GPU_PIDS[$gpu_idx]=$!
}

# Main loop - distribute experiments across GPUs
for GS in "${GUIDANCE_SCALES[@]}"; do
    for HS in "${HARMFUL_SCALES[@]}"; do
        for SP_START in "${SPATIAL_THR_STARTS[@]}"; do
            for SP_END in "${SPATIAL_THR_ENDS[@]}"; do
                for BASE_GS in "${BASE_GUIDANCE_SCALES[@]}"; do
                    # Compute GPU ID (round-robin)
                    GPU_ID=$((EXP_IDX % NUM_GPUS))

                    # Experiment name
                    EXP_NAME="gs${GS}_hs${HS}_sp${SP_START}-${SP_END}_bs${BASE_GS}"

                    # Wait for GPU slot if it's busy
                    wait_for_gpu_slot ${GPU_ID}

                    # Run experiment
                    run_on_gpu ${GPU_ID} "${EXP_NAME}" ${GS} ${HS} ${SP_START} ${SP_END} ${BASE_GS}

                    EXP_IDX=$((EXP_IDX + 1))

                    # Progress update every 10 experiments
                    if [ $((EXP_IDX % 10)) -eq 0 ] && [ "$DRY_RUN" = false ]; then
                        echo "Progress: ${EXP_IDX}/${TOTAL_EXPERIMENTS} experiments started"
                    fi
                done
            done
        done
    done
done

# Wait for all remaining jobs
echo ""
echo "Waiting for all experiments to complete..."
for pid in "${GPU_PIDS[@]}"; do
    if [ ! -z "$pid" ]; then
        wait $pid 2>/dev/null || true
    fi
done

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "Total: ${TOTAL_EXPERIMENTS} experiments would run on ${NUM_GPUS} GPUs"
    exit 0
fi

echo ""
echo "=============================================="
echo "GRID SEARCH COMPLETE!"
echo "=============================================="
echo "Results saved to: ${BASE_OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Run NudeNet evaluation on each output directory"
echo "  2. Compute FID/CLIP scores"
echo "  3. Aggregate results"
echo "=============================================="

# Aggregate results
echo ""
echo "Aggregating results..."

python -c "
import json
import os
from pathlib import Path

base_dir = Path('${BASE_OUTPUT_DIR}')
results = []

for exp_dir in sorted(base_dir.iterdir()):
    if not exp_dir.is_dir() or exp_dir.name == 'logs':
        continue
    stats_file = exp_dir / 'generation_stats.json'
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
        overall = stats.get('overall', {})
        results.append({
            'exp_name': exp_dir.name,
            'avg_guided_steps': overall.get('avg_guided_steps', 0),
            'avg_guidance_ratio': overall.get('avg_guidance_ratio', 0),
            'no_guidance': overall.get('no_guidance_count', 0),
            'light_guidance': overall.get('light_guidance_count', 0),
            'medium_guidance': overall.get('medium_guidance_count', 0),
            'heavy_guidance': overall.get('heavy_guidance_count', 0),
        })

# Save aggregated results
with open(base_dir / 'aggregated_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'Aggregated {len(results)} experiment results')
print(f'Saved to: {base_dir / \"aggregated_results.json\"}')
"
