#!/bin/bash
# Grid Search for Dual Classifier: 3-class monitoring + 4-class guidance
#
# Monitoring: argmax-based (no threshold to search)
# Search parameters: guidance_scale, base_guidance_scale, spatial_threshold

set -e

# ============================================
# Parse Arguments
# ============================================
DRY_RUN=false
USE_NOHUP=false
NUM_GPUS=1
CONCEPT="nudity"
PROMPT_FILE_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --nohup)
            USE_NOHUP=true
            shift
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --concept)
            CONCEPT="$2"
            shift 2
            ;;
        --prompt-file)
            PROMPT_FILE_ARG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ============================================
# CONFIGURATION
# ============================================

# Classifiers
CLASSIFIER_3CLASS="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
CLASSIFIER_4CLASS="./work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_DIR="./gradcam_stats/nudity_4class"

# Stable Diffusion
SD_CKPT="CompVis/stable-diffusion-v1-4"

# Prompt file (use --prompt-file if provided, otherwise default to Ring-A-Bell)
if [ -n "$PROMPT_FILE_ARG" ]; then
    PROMPT_FILE="$PROMPT_FILE_ARG"
else
    PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
fi

# Fixed parameters
NUM_STEPS=50
SEED=42
NSAMPLES=1

# ============================================
# GRID SEARCH PARAMETERS (matching 3-class best config)
# ============================================
# Best config from 3-class: gs15.0_hs1.5_st0.1-0.4_cosine_anneal

# Guidance scales for spatial weighting in harm regions
GUIDANCE_SCALES=(10 12.5 15)

# Harmful scales (multiplier for harm gradient: grad = g_safe - harmful_scale * g_harm)
# Fixed to 1.5 (best from 3-class)
HARMFUL_SCALES=(1.5)

# Base guidance scales (for non-harm regions)
BASE_SCALES=(0.0 1.0 2.0)

# Spatial thresholds (start end pairs)
# Fixed to 0.1 -> 0.4 (best from 3-class)
SPATIAL_THRESHOLDS=(
    "0.1 0.4"   # Best config from 3-class
)

# Threshold strategy (cosine to match cosine_anneal behavior)
THRESHOLD_STRATEGY="cosine"

# ============================================
# NOHUP HANDLING
# ============================================
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="./scg_outputs/grid_search_dual_ringabell_${TIMESTAMP}"
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/grid_search.log"

    echo "Running in background with nohup..."
    echo "Log file: ${LOG_FILE}"

    PROMPT_OPT=""
    [ -n "$PROMPT_FILE_ARG" ] && PROMPT_OPT="--prompt-file ${PROMPT_FILE_ARG}"
    nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$0" --concept "${CONCEPT}" --num-gpus "${NUM_GPUS}" ${PROMPT_OPT} > "${LOG_FILE}" 2>&1 &
    echo "Background PID: $!"
    exit 0
fi

# ============================================
# SETUP
# ============================================
# Parse GPU list from CUDA_VISIBLE_DEVICES or use default 0,1,2,...
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do
        GPU_LIST+=($i)
    done
fi
echo "Available GPUs: ${GPU_LIST[*]}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="./scg_outputs/grid_search_dual_ringabell_${TIMESTAMP}"
mkdir -p "${OUTPUT_BASE}/logs"

echo "=============================================="
echo "DUAL CLASSIFIER GRID SEARCH"
echo "=============================================="
echo "Concept: ${CONCEPT}"
echo "GPUs: ${NUM_GPUS}"
echo "3-class (monitoring): ${CLASSIFIER_3CLASS}"
echo "4-class (guidance): ${CLASSIFIER_4CLASS}"
echo "Output: ${OUTPUT_BASE}"
echo "=============================================="
echo ""

# Build parameter combinations
declare -a COMBINATIONS=()

for gs in "${GUIDANCE_SCALES[@]}"; do
    for hs in "${HARMFUL_SCALES[@]}"; do
        for bs in "${BASE_SCALES[@]}"; do
            for sp in "${SPATIAL_THRESHOLDS[@]}"; do
                COMBINATIONS+=("${gs}|${hs}|${bs}|${sp}")
            done
        done
    done
done

TOTAL=${#COMBINATIONS[@]}
echo "Total combinations: ${TOTAL}"
echo "  - guidance_scale: ${#GUIDANCE_SCALES[@]} values"
echo "  - harmful_scale: ${#HARMFUL_SCALES[@]} values"
echo "  - base_scale: ${#BASE_SCALES[@]} values"
echo "  - spatial_threshold: ${#SPATIAL_THRESHOLDS[@]} values"
echo "  - strategy: ${THRESHOLD_STRATEGY} (fixed)"
echo ""

# ============================================
# RUN GRID SEARCH
# ============================================
run_experiment() {
    local GPU_IDX=$1
    local GUIDANCE_SCALE=$2
    local HARMFUL_SCALE=$3
    local BASE_SCALE=$4
    local SPATIAL_START=$5
    local SPATIAL_END=$6
    local EXP_NAME=$7

    # Map GPU index to actual GPU ID
    local ACTUAL_GPU=${GPU_LIST[$GPU_IDX]}

    local OUTPUT_DIR="${OUTPUT_BASE}/${EXP_NAME}"
    local LOG_FILE="${OUTPUT_BASE}/logs/${EXP_NAME}.log"

    echo "[GPU ${ACTUAL_GPU}] Starting: ${EXP_NAME}"

    CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python generate_nudity_dual_classifier.py \
        --ckpt_path "${SD_CKPT}" \
        --prompt_file "${PROMPT_FILE}" \
        --output_dir "${OUTPUT_DIR}" \
        --classifier_3class_ckpt "${CLASSIFIER_3CLASS}" \
        --classifier_4class_ckpt "${CLASSIFIER_4CLASS}" \
        --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
        --guidance_scale ${GUIDANCE_SCALE} \
        --harmful_scale ${HARMFUL_SCALE} \
        --base_guidance_scale ${BASE_SCALE} \
        --spatial_threshold_start ${SPATIAL_START} \
        --spatial_threshold_end ${SPATIAL_END} \
        --spatial_threshold_strategy "${THRESHOLD_STRATEGY}" \
        --num_inference_steps ${NUM_STEPS} \
        --seed ${SEED} \
        --nsamples ${NSAMPLES} \
        > "${LOG_FILE}" 2>&1

    local STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "[GPU ${ACTUAL_GPU}] Completed: ${EXP_NAME}"
    else
        echo "[GPU ${ACTUAL_GPU}] FAILED: ${EXP_NAME} (exit code: ${STATUS})"
    fi

    return $STATUS
}

# Track running jobs
declare -A RUNNING_JOBS

# Process all combinations
IDX=0
for combo in "${COMBINATIONS[@]}"; do
    IFS='|' read -r GS HS BS SP <<< "$combo"

    # Parse spatial threshold (space separated: "start end")
    SP_START=$(echo $SP | awk '{print $1}')
    SP_END=$(echo $SP | awk '{print $2}')

    # Experiment name
    EXP_NAME="gs${GS}_hs${HS}_bs${BS}_sp${SP_START}-${SP_END}"

    # Find available GPU
    while true; do
        for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
            JOB_PID="${RUNNING_JOBS[$gpu]}"

            # Check if slot is free
            if [ -z "$JOB_PID" ] || ! kill -0 "$JOB_PID" 2>/dev/null; then
                # Run experiment
                run_experiment $gpu $GS $HS $BS $SP_START $SP_END "$EXP_NAME" &
                RUNNING_JOBS[$gpu]=$!

                IDX=$((IDX + 1))
                echo "Progress: ${IDX}/${TOTAL}"

                # Small delay to prevent race conditions
                sleep 2
                break 2
            fi
        done

        # Wait a bit before checking again
        sleep 5
    done
done

# Wait for all remaining jobs
echo ""
echo "Waiting for remaining jobs to complete..."
wait

echo ""
echo "=============================================="
echo "GRID SEARCH COMPLETE"
echo "=============================================="
echo "Results saved to: ${OUTPUT_BASE}"
echo ""

# Generate summary
echo "Generating summary..."
python -c "
import json
from pathlib import Path
import sys

base_dir = Path('${OUTPUT_BASE}')
results = []

for exp_dir in sorted(base_dir.iterdir()):
    if exp_dir.is_dir() and exp_dir.name != 'logs':
        stats_file = exp_dir / 'generation_stats.json'
        if stats_file.exists():
            with open(stats_file) as f:
                data = json.load(f)

            args = data.get('args', {})
            overall = data.get('overall', {})

            results.append({
                'experiment': exp_dir.name,
                'guidance_scale': args.get('guidance_scale'),
                'harmful_scale': args.get('harmful_scale'),
                'base_guidance_scale': args.get('base_guidance_scale'),
                'spatial_start': args.get('spatial_threshold_start'),
                'spatial_end': args.get('spatial_threshold_end'),
                'strategy': args.get('spatial_threshold_strategy'),
                'avg_guided_steps': overall.get('avg_guided_steps', 0),
                'avg_guidance_ratio': overall.get('avg_guidance_ratio', 0),
                'no_guidance_count': overall.get('no_guidance_count', 0),
                'heavy_guidance_count': overall.get('heavy_guidance_count', 0),
            })

# Sort by guidance ratio
results.sort(key=lambda x: x['avg_guidance_ratio'], reverse=True)

# Save summary
summary_file = base_dir / 'grid_search_summary.json'
with open(summary_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f'Summary saved to: {summary_file}')
print(f'Total experiments: {len(results)}')

# Print top 5
print()
print('Top 5 by guidance ratio:')
for i, r in enumerate(results[:5]):
    print(f\"  {i+1}. {r['experiment']}: {r['avg_guidance_ratio']*100:.1f}% guided\")
"

echo ""
echo "Done!"
