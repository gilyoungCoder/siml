#!/bin/bash
# ============================================================================
# Grid Search: 4-class Monitoring + Spatial CG
# Best from ringabell: mon=0.1, gs=12.5, sp=0.1-0.4, bs=1.0 (SR=77.2%)
# Search nearby values across 3 datasets
# ============================================================================

set -e

DRY_RUN=false
NUM_GPUS=8
USE_NOHUP=false
DATASET="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --nohup) USE_NOHUP=true; shift ;;
        --dataset) DATASET="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/mnt/home/yhgil99/unlearning"

if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${BASE_DIR}/scripts/logs/grid_mon4class_${TIMESTAMP}.log"
    mkdir -p "${BASE_DIR}/scripts/logs"
    echo "Running in background..."
    echo "Log: ${LOG_FILE}"
    nohup bash "$0" --num-gpus "${NUM_GPUS}" --dataset "${DATASET}" > "${LOG_FILE}" 2>&1 &
    echo "PID: $!"
    exit 0
fi

cd "${BASE_DIR}/SoftDelete+CG"

# Parse GPU list
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_LIST+=($i); done
fi
echo "GPUs: ${GPU_LIST[*]}"

# === Paths ===
CLASSIFIER_CKPT="./work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS="./gradcam_stats/nudity_4class"
SD_MODEL="CompVis/stable-diffusion-v1-4"
SCRIPT="generate_nudity_4class_sample_level_monitoring.py"

# === Datasets ===
declare -A PROMPTS
PROMPTS[ringabell]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
PROMPTS[p4dn]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/p4dn_16_prompt.csv"
PROMPTS[unlearndiff]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/unlearn_diff_nudity.csv"

if [ "$DATASET" = "all" ]; then
    DATASETS=(ringabell p4dn unlearndiff)
else
    DATASETS=($DATASET)
fi

# === Grid params (around best: mon=0.1, gs=12.5, sp=0.1-0.4, bs=1.0) ===
MONITORING_THRESHOLDS=(0.1 0.2 0.3)
GUIDANCE_SCALES=(10.0 12.5 15.0)
BASE_GUIDANCE_SCALES=(0.0 1.0 2.0)
SPATIAL_THR_STARTS=(0.1 0.3 0.5)
SPATIAL_THR_ENDS=(0.3 0.4 0.5)
THR_STRATEGY="cosine"

# Fixed
NSAMPLES=1
CFG_SCALE=7.5
NUM_STEPS=50
SEED=1234

# Count
TOTAL_PER_DS=$((${#MONITORING_THRESHOLDS[@]} * ${#GUIDANCE_SCALES[@]} * ${#BASE_GUIDANCE_SCALES[@]} * ${#SPATIAL_THR_STARTS[@]} * ${#SPATIAL_THR_ENDS[@]}))
TOTAL=$((TOTAL_PER_DS * ${#DATASETS[@]}))

echo "=============================================="
echo "GRID SEARCH: 4-class Monitoring"
echo "=============================================="
echo "Datasets: ${DATASETS[*]}"
echo "mon_thresholds: ${MONITORING_THRESHOLDS[*]}"
echo "guidance_scales: ${GUIDANCE_SCALES[*]}"
echo "base_guidance_scales: ${BASE_GUIDANCE_SCALES[*]}"
echo "spatial_starts: ${SPATIAL_THR_STARTS[*]}"
echo "spatial_ends: ${SPATIAL_THR_ENDS[*]}"
echo "Total: ${TOTAL} experiments (${TOTAL_PER_DS}/dataset)"
echo "=============================================="

declare -a GPU_PIDS

wait_for_gpu_slot() {
    local gpu_id=$1
    if [ -n "${GPU_PIDS[$gpu_id]:-}" ]; then
        wait ${GPU_PIDS[$gpu_id]} 2>/dev/null || true
    fi
}

EXP_IDX=0

for DS in "${DATASETS[@]}"; do
    PROMPT_FILE="${PROMPTS[$DS]}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_BASE="./scg_outputs/grid_mon4class_${DS}_${TIMESTAMP}"
    mkdir -p "${OUTPUT_BASE}/logs"

    echo ""
    echo "=== Dataset: ${DS} ==="
    echo "Output: ${OUTPUT_BASE}"

    for MON in "${MONITORING_THRESHOLDS[@]}"; do
        for GS in "${GUIDANCE_SCALES[@]}"; do
            for BS in "${BASE_GUIDANCE_SCALES[@]}"; do
                for SP_S in "${SPATIAL_THR_STARTS[@]}"; do
                    for SP_E in "${SPATIAL_THR_ENDS[@]}"; do
                        GPU_ID=$((EXP_IDX % ${#GPU_LIST[@]}))
                        ACTUAL_GPU=${GPU_LIST[$GPU_ID]}
                        EXP_NAME="mon${MON}_gs${GS}_sp${SP_S}-${SP_E}_bs${BS}"

                        wait_for_gpu_slot ${GPU_ID}

                        if [ "$DRY_RUN" = true ]; then
                            echo "[GPU ${ACTUAL_GPU}] ${DS}/${EXP_NAME}"
                        else
                            echo "[GPU ${ACTUAL_GPU}] ${DS}/${EXP_NAME}"
                            CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python ${SCRIPT} \
                                --ckpt_path "${SD_MODEL}" \
                                --prompt_file "${PROMPT_FILE}" \
                                --output_dir "${OUTPUT_BASE}/${EXP_NAME}" \
                                --classifier_ckpt "${CLASSIFIER_CKPT}" \
                                --gradcam_stats_dir "${GRADCAM_STATS}" \
                                --monitoring_threshold ${MON} \
                                --guidance_scale ${GS} \
                                --base_guidance_scale ${BS} \
                                --spatial_threshold_start ${SP_S} \
                                --spatial_threshold_end ${SP_E} \
                                --spatial_threshold_strategy "${THR_STRATEGY}" \
                                --guidance_start_step 0 \
                                --guidance_end_step 50 \
                                --nsamples ${NSAMPLES} \
                                --cfg_scale ${CFG_SCALE} \
                                --num_inference_steps ${NUM_STEPS} \
                                --seed ${SEED} \
                                > "${OUTPUT_BASE}/logs/${EXP_NAME}.log" 2>&1 &
                            GPU_PIDS[$GPU_ID]=$!
                        fi

                        EXP_IDX=$((EXP_IDX + 1))
                        if [ $((EXP_IDX % 10)) -eq 0 ] && [ "$DRY_RUN" = false ]; then
                            echo "Progress: ${EXP_IDX}/${TOTAL}"
                        fi
                    done
                done
            done
        done
    done

    # Wait for all before next dataset
    for pid in "${GPU_PIDS[@]}"; do
        [ -n "${pid:-}" ] && wait $pid 2>/dev/null || true
    done
    GPU_PIDS=()
    echo "=== ${DS} done ==="
done

echo ""
echo "=============================================="
echo "ALL GRID SEARCH COMPLETE (${EXP_IDX} experiments)"
echo "=============================================="
