#!/bin/bash
# ============================================================================
# Run text-based early exit (word-level CLIP cosine sim) on all 4 datasets
# Config: best monitoring config mon0.05_gs12.5_bs2.0_sp0.2-0.3
# + text_exit_threshold=0.50
# ============================================================================

set -e

NUM_GPUS=8
USE_NOHUP=false
DATASET="all"
TEXT_THR=0.50

while [[ $# -gt 0 ]]; do
    case $1 in
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --nohup) USE_NOHUP=true; shift ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --text-thr) TEXT_THR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

BASE_DIR="/mnt/home/yhgil99/unlearning"

if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${BASE_DIR}/scripts/logs/text_exit_${TIMESTAMP}.log"
    mkdir -p "${BASE_DIR}/scripts/logs"
    echo "Running in background..."
    echo "Log: ${LOG_FILE}"
    nohup bash "$0" --num-gpus "${NUM_GPUS}" --dataset "${DATASET}" --text-thr "${TEXT_THR}" > "${LOG_FILE}" 2>&1 &
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
SCRIPT="generate_nudity_4class_monitoring_early_exit.py"

# === Best config ===
MON=0.05
GS=12.5
BS=2.0
SP_S=0.2
SP_E=0.3

# Fixed
NSAMPLES=1
CFG_SCALE=7.5
NUM_STEPS=50
SEED=1234

# === Datasets ===
declare -A PROMPTS
declare -A NUM_IMAGES
PROMPTS[ringabell]="${BASE_DIR}/SAFREE/datasets/nudity-ring-a-bell.csv"
PROMPTS[unlearndiff]="${BASE_DIR}/SAFREE/datasets/unlearn_diff_nudity.csv"
PROMPTS[mma]="${BASE_DIR}/prompts/nudity_datasets/mma.txt"
PROMPTS[coco]="${BASE_DIR}/SAFREE/datasets/coco_30k_10k.csv"

if [ "$DATASET" = "all" ]; then
    DATASETS=(ringabell unlearndiff mma coco)
else
    DATASETS=($DATASET)
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="./scg_outputs/text_exit_${TIMESTAMP}"
mkdir -p "${OUTPUT_BASE}/logs"

CONFIG_NAME="mon${MON}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}_txt${TEXT_THR}"

echo "=============================================="
echo "TEXT EXIT: word-level CLIP cosine sim"
echo "=============================================="
echo "Datasets: ${DATASETS[*]}"
echo "Config: ${CONFIG_NAME}"
echo "text_exit_threshold: ${TEXT_THR}"
echo "Output: ${OUTPUT_BASE}"
echo "=============================================="

declare -a GPU_PIDS
EXP_IDX=0

for DS in "${DATASETS[@]}"; do
    PROMPT_FILE="${PROMPTS[$DS]}"

    # Count total prompts to split across GPUs
    if [[ "${PROMPT_FILE}" == *.csv ]]; then
        TOTAL_PROMPTS=$(($(wc -l < "${PROMPT_FILE}") - 1))
    else
        TOTAL_PROMPTS=$(wc -l < "${PROMPT_FILE}")
    fi

    CHUNK=$((TOTAL_PROMPTS / ${#GPU_LIST[@]}))
    REMAINDER=$((TOTAL_PROMPTS % ${#GPU_LIST[@]}))

    DS_OUTPUT="${OUTPUT_BASE}/${DS}/${CONFIG_NAME}"
    mkdir -p "${DS_OUTPUT}"
    mkdir -p "${OUTPUT_BASE}/logs"

    echo ""
    echo "=== Dataset: ${DS} (${TOTAL_PROMPTS} prompts, ${#GPU_LIST[@]} GPUs, ~${CHUNK}/GPU) ==="

    START=0
    for ((g=0; g<${#GPU_LIST[@]}; g++)); do
        ACTUAL_GPU=${GPU_LIST[$g]}

        # Distribute remainder to first GPUs
        if [ $g -lt $REMAINDER ]; then
            END=$((START + CHUNK + 1))
        else
            END=$((START + CHUNK))
        fi

        if [ $START -ge $TOTAL_PROMPTS ]; then
            break
        fi
        if [ $END -gt $TOTAL_PROMPTS ]; then
            END=$TOTAL_PROMPTS
        fi

        echo "  [GPU ${ACTUAL_GPU}] ${DS} prompts ${START}-${END}"

        CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python ${SCRIPT} \
            --ckpt_path "${SD_MODEL}" \
            --prompt_file "${PROMPT_FILE}" \
            --output_dir "${DS_OUTPUT}" \
            --classifier_ckpt "${CLASSIFIER_CKPT}" \
            --gradcam_stats_dir "${GRADCAM_STATS}" \
            --monitoring_threshold ${MON} \
            --guidance_scale ${GS} \
            --base_guidance_scale ${BS} \
            --spatial_threshold_start ${SP_S} \
            --spatial_threshold_end ${SP_E} \
            --spatial_threshold_strategy cosine \
            --guidance_start_step 0 \
            --guidance_end_step 50 \
            --text_exit_threshold ${TEXT_THR} \
            --nsamples ${NSAMPLES} \
            --cfg_scale ${CFG_SCALE} \
            --num_inference_steps ${NUM_STEPS} \
            --seed ${SEED} \
            --start_idx ${START} \
            --end_idx ${END} \
            > "${OUTPUT_BASE}/logs/${DS}_gpu${ACTUAL_GPU}.log" 2>&1 &
        GPU_PIDS[$g]=$!

        START=$END
    done

    # Wait for all GPUs before next dataset
    for pid in "${GPU_PIDS[@]}"; do
        [ -n "${pid:-}" ] && wait $pid 2>/dev/null || true
    done
    GPU_PIDS=()
    echo "=== ${DS} done ==="
done

echo ""
echo "=============================================="
echo "ALL DONE"
echo "Output: ${OUTPUT_BASE}"
echo "=============================================="
