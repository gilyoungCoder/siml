#!/bin/bash
# ============================================================================
# Multi-GPU Grid Search: 4-Class Sample-Level Monitoring + Spatial CG
# Ring-A-Bell prompts
# ============================================================================
#
# Usage:
#   ./grid_search_monitoring_4class_ringabell.sh                    # Run all
#   ./grid_search_monitoring_4class_ringabell.sh --dry-run          # Print only
#   ./grid_search_monitoring_4class_ringabell.sh --nohup            # Background
# ============================================================================

set -e

# === Parse arguments ===
DRY_RUN=false
NUM_GPUS=8
USE_NOHUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)    DRY_RUN=true; shift ;;
        --num-gpus)   NUM_GPUS="$2"; shift 2 ;;
        --nohup)      USE_NOHUP=true; shift ;;
        *)            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# If --nohup flag is set, re-run in background
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="./scg_outputs/grid_search_mon4class_ringabell_${TIMESTAMP}.log"
    mkdir -p ./scg_outputs
    echo "Running in background with nohup..."
    echo "Log file: ${LOG_FILE}"
    nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$0" --num-gpus "${NUM_GPUS}" > "${LOG_FILE}" 2>&1 &
    echo "PID: $!"
    echo "Use 'tail -f ${LOG_FILE}' to monitor progress"
    exit 0
fi

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# === Parse GPU list ===
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do
        GPU_LIST+=($i)
    done
fi
NUM_GPUS=${#GPU_LIST[@]}
echo "Available GPUs: ${GPU_LIST[*]} (${NUM_GPUS} total)"

# === Configuration ===
CLASSIFIER_CKPT="./work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_DIR="./gradcam_stats/nudity_4class"
SD_MODEL="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="./scg_outputs/grid_search_mon4class_ringabell_${TIMESTAMP}"

# Fixed parameters
NSAMPLES=1
CFG_SCALE=7.5
NUM_STEPS=50
# ============================================================================
# Grid Search Parameters
# ============================================================================
GUIDANCE_SCALES=(10 12.5 15)
SPATIAL_THR_STARTS=(0.1)
SPATIAL_THR_ENDS=(0.4)
BASE_GUIDANCE_SCALES=(1.0 2.0)
MONITORING_THRESHOLDS=(0.1 0.2)

# Calculate total
TOTAL_EXPERIMENTS=$(( ${#GUIDANCE_SCALES[@]} * ${#SPATIAL_THR_STARTS[@]} * ${#SPATIAL_THR_ENDS[@]} * ${#BASE_GUIDANCE_SCALES[@]} * ${#MONITORING_THRESHOLDS[@]} ))

echo "=============================================="
echo "MULTI-GPU GRID SEARCH: 4-Class Monitoring"
echo "=============================================="
echo "Prompt file: ${PROMPT_FILE}"
echo "Output base: ${BASE_OUTPUT_DIR}"
echo ""
echo "Parameters:"
echo "  guidance_scales: ${GUIDANCE_SCALES[*]}"
echo "  spatial_threshold: ${SPATIAL_THR_STARTS[*]} -> ${SPATIAL_THR_ENDS[*]}"
echo "  base_guidance_scales: ${BASE_GUIDANCE_SCALES[*]}"
echo "  monitoring_thresholds: ${MONITORING_THRESHOLDS[*]}"
echo ""
echo "Total experiments: ${TOTAL_EXPERIMENTS}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "=============================================="
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN MODE]"
    echo ""
fi

mkdir -p "${BASE_OUTPUT_DIR}/logs"

# Save config
cat > "${BASE_OUTPUT_DIR}/config.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "method": "4class_monitoring",
    "prompt_file": "${PROMPT_FILE}",
    "classifier_ckpt": "${CLASSIFIER_CKPT}",
    "gradcam_stats_dir": "${GRADCAM_STATS_DIR}",
    "total_experiments": ${TOTAL_EXPERIMENTS},
    "num_gpus": ${NUM_GPUS},
    "parameters": {
        "guidance_scales": [${GUIDANCE_SCALES[*]}],
        "spatial_threshold_starts": [${SPATIAL_THR_STARTS[*]}],
        "spatial_threshold_ends": [${SPATIAL_THR_ENDS[*]}],
        "base_guidance_scales": [${BASE_GUIDANCE_SCALES[*]}],
        "monitoring_threshold": ${MONITORING_THRESHOLD}
    }
}
EOF

EXP_IDX=0
declare -a GPU_PIDS

wait_for_gpu_slot() {
    local gpu_id=$1
    if [ ! -z "${GPU_PIDS[$gpu_id]}" ]; then
        wait ${GPU_PIDS[$gpu_id]} 2>/dev/null || true
    fi
}

run_on_gpu() {
    local gpu_idx=$1
    local exp_name=$2
    local gs=$3
    local sp_start=$4
    local sp_end=$5
    local base_gs=$6
    local mon_thr=$7

    local actual_gpu=${GPU_LIST[$gpu_idx]}
    local output_dir="${BASE_OUTPUT_DIR}/${exp_name}"
    local log_file="${BASE_OUTPUT_DIR}/logs/${exp_name}.log"

    if [ "$DRY_RUN" = true ]; then
        echo "[GPU ${actual_gpu}] ${exp_name}"
        return 0
    fi

    echo "[GPU ${actual_gpu}] Starting: ${exp_name}"

    CUDA_VISIBLE_DEVICES=${actual_gpu} python generate_nudity_4class_sample_level_monitoring.py \
        --ckpt_path "${SD_MODEL}" \
        --prompt_file "${PROMPT_FILE}" \
        --output_dir "${output_dir}" \
        --nsamples ${NSAMPLES} \
        --cfg_scale ${CFG_SCALE} \
        --num_inference_steps ${NUM_STEPS} \
        --classifier_ckpt "${CLASSIFIER_CKPT}" \
        --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
        --monitoring_threshold ${mon_thr} \
        --guidance_scale ${gs} \
        --base_guidance_scale ${base_gs} \
        --spatial_threshold_start ${sp_start} \
        --spatial_threshold_end ${sp_end} \
        --spatial_threshold_strategy cosine \
        --seed 42 \
        > "${log_file}" 2>&1 &

    GPU_PIDS[$gpu_idx]=$!
}

# Main loop
for MON_THR in "${MONITORING_THRESHOLDS[@]}"; do
    for GS in "${GUIDANCE_SCALES[@]}"; do
        for SP_START in "${SPATIAL_THR_STARTS[@]}"; do
            for SP_END in "${SPATIAL_THR_ENDS[@]}"; do
                for BASE_GS in "${BASE_GUIDANCE_SCALES[@]}"; do
                    GPU_ID=$((EXP_IDX % NUM_GPUS))
                    EXP_NAME="mon${MON_THR}_gs${GS}_sp${SP_START}-${SP_END}_bs${BASE_GS}"

                    wait_for_gpu_slot ${GPU_ID}
                    run_on_gpu ${GPU_ID} "${EXP_NAME}" ${GS} ${SP_START} ${SP_END} ${BASE_GS} ${MON_THR}

                    EXP_IDX=$((EXP_IDX + 1))
                done
            done
        done
    done
done

# Wait for all
echo ""
echo "Waiting for all experiments to complete..."
for pid in "${GPU_PIDS[@]}"; do
    [ ! -z "$pid" ] && wait $pid 2>/dev/null || true
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
echo "=============================================="
