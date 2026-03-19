#!/bin/bash
# ============================================================================
# Grid Search: Z0 ASCG Monitoring (3-class) — Multi-Dataset
#
# Adaptive Spatial Classifier Guidance with GradCAM monitoring for z0 pipeline.
# Uses nude 7700 3-class checkpoint.
#
# Supported datasets:
#   ringabell    — Ring-A-Bell (79 prompts)
#   unlearndiff  — UnlearnDiff (142 prompts)
#   mma          — MMA-Diffusion (1000 prompts)
#
# Sweep parameters:
#   monitoring_thresholds: [0.1, 0.3, 0.5]
#   guidance_scales: [10.0, 12.5, 15.0, 20.0, 25.0, 30.0]
#   base_scales: [0.0, 1.0, 2.0]
#   spatial_thresholds: 6 (start, end) pairs
#   harm_ratio: 1.0 (fixed)
#   Total: 324 configs per dataset
#
# Usage:
#   bash scripts/run_grid_monitoring.sh --dataset ringabell              # dry run
#   bash scripts/run_grid_monitoring.sh --dataset ringabell --run        # execute
#   bash scripts/run_grid_monitoring.sh --dataset ringabell --run --nohup  # background
# ============================================================================

cd /mnt/home/yhgil99/unlearning/z0_clf_guidance

# Activate conda environment
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd

# ============================================
# Parse Arguments
# ============================================
DRY_RUN=true
USE_NOHUP=false
NUM_GPUS=8
JOBS_PER_GPU=2
DATASET=""
START_IDX=0
END_IDX=-1

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2;;
        --run) DRY_RUN=false; shift;;
        --dry-run) DRY_RUN=true; shift;;
        --nohup) USE_NOHUP=true; shift;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2;;
        --start-idx) START_IDX="$2"; shift 2;;
        --end-idx) END_IDX="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

# ============================================
# DATASET CONFIGURATION
# ============================================

if [ -z "$DATASET" ]; then
    echo "ERROR: --dataset is required."
    echo "Available datasets: ringabell, unlearndiff, mma"
    echo ""
    echo "Usage:"
    echo "  bash scripts/run_grid_monitoring.sh --dataset ringabell              # dry run"
    echo "  bash scripts/run_grid_monitoring.sh --dataset ringabell --run        # execute"
    echo "  bash scripts/run_grid_monitoring.sh --dataset ringabell --run --nohup  # background"
    exit 1
fi

case $DATASET in
    ringabell)
        PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
        ;;
    unlearndiff)
        PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/unlearn_diff_nudity.csv"
        ;;
    mma)
        PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/mma-diffusion-nsfw-adv-prompts.csv"
        ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET'"
        echo "Available datasets: ringabell, unlearndiff, mma"
        exit 1
        ;;
esac

# Verify prompt file exists
if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: Prompt file not found: $PROMPT_FILE"
    exit 1
fi

# Nohup handling
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="./grid_monitoring_output/logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/grid_${DATASET}_${TIMESTAMP}.log"
    echo "Running in background..."
    echo "Log: $LOG_FILE"
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    nohup bash "$SCRIPT_PATH" --dataset "$DATASET" --run --num-gpus "$NUM_GPUS" --jobs-per-gpu "$JOBS_PER_GPU" --start-idx "$START_IDX" --end-idx "$END_IDX" > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    echo "Monitor: tail -f $LOG_FILE"
    exit 0
fi

# ============================================
# CONFIGURATION
# ============================================

CLASSIFIER_CKPT="./work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth"
HARMFUL_STATS_PATH="./harmful_stats.pt"
SD_CKPT="CompVis/stable-diffusion-v1-4"
GEN_SCRIPT="generate_monitoring.py"

# Fixed params
NUM_STEPS=50
SEED=42
NSAMPLES=1
CFG_SCALE=7.5
THRESHOLD_STRATEGY="cosine"
GRADCAM_LAYER="layer2"
HARM_RATIO=1.0

# ============================================
# GRID SEARCH PARAMETERS
# ============================================

MONITORING_THRESHOLDS=(0.1 0.2 0.3 0.4)
GUIDANCE_SCALES=(10.0 12.5 15.0 20.0 25.0 30.0)
BASE_SCALES=(0.0 1.0 2.0)
SPATIAL_THRESHOLDS=(
    "0.05 0.3"
    "0.05 0.5"
    "0.1 0.1"
    "0.1 0.3"
    "0.1 0.5"
    "0.15 0.3"
    "0.2 0.2"
    "0.2 0.3"
    "0.2 0.5"
    "0.3 0.3"
    "0.3 0.5"
    "0.5 0.5"
)

# VLM eval
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
OUTPUT_BASE="./grid_monitoring_output/${DATASET}"

# ============================================
# BUILD COMBINATIONS
# ============================================

declare -a COMBINATIONS=()
for mt in "${MONITORING_THRESHOLDS[@]}"; do
    for gs in "${GUIDANCE_SCALES[@]}"; do
        for bs in "${BASE_SCALES[@]}"; do
            for sp in "${SPATIAL_THRESHOLDS[@]}"; do
                COMBINATIONS+=("${mt}|${gs}|${bs}|${sp}")
            done
        done
    done
done

TOTAL=${#COMBINATIONS[@]}
echo "=============================================="
echo "Z0 ASCG MONITORING GRID SEARCH (3-class)"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Prompt file: $PROMPT_FILE"
echo "Total combinations: $TOTAL"
echo "mon_thr: ${MONITORING_THRESHOLDS[*]}"
echo "gs: ${GUIDANCE_SCALES[*]}"
echo "bs: ${BASE_SCALES[*]}"
echo "spatial: ${#SPATIAL_THRESHOLDS[@]} pairs"
echo "harm_ratio: ${HARM_RATIO}"
echo "GPUs: $NUM_GPUS (${JOBS_PER_GPU} jobs/GPU = $((NUM_GPUS * JOBS_PER_GPU)) concurrent)"
if [ "$START_IDX" -ne 0 ] || [ "$END_IDX" -ne -1 ]; then
    echo "Prompt range: [${START_IDX}:${END_IDX}]"
fi
echo "=============================================="

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY RUN] All experiments:"
    for (( i=0; i<TOTAL; i++ )); do
        combo="${COMBINATIONS[$i]}"
        IFS='|' read -r MT GS BS SP <<< "$combo"
        SP_START=$(echo $SP | awk '{print $1}')
        SP_END=$(echo $SP | awk '{print $2}')
        GPU_IDX=$((i % NUM_GPUS))
        echo "  [$i] GPU${GPU_IDX} ${DATASET} | mon${MT}_gs${GS}_bs${BS}_sp${SP_START}-${SP_END}"
    done
    echo ""
    echo "Total: $TOTAL experiments on $NUM_GPUS GPUs (~$((TOTAL / NUM_GPUS)) per GPU)"
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
echo "GPUs: ${GPU_LIST[*]}"

mkdir -p "${OUTPUT_BASE}/logs"

# ============================================
# RUN FUNCTION
# ============================================

run_experiment() {
    local GPU_IDX=$1
    local MON_THR=$2
    local GS=$3
    local BS=$4
    local SP_START=$5
    local SP_END=$6

    local ACTUAL_GPU=${GPU_LIST[$GPU_IDX]}
    local EXP_NAME="mon${MON_THR}_gs${GS}_bs${BS}_sp${SP_START}-${SP_END}"
    local OUTPUT_DIR="${OUTPUT_BASE}/${EXP_NAME}"
    local LOG_FILE="${OUTPUT_BASE}/logs/${EXP_NAME}.log"

    # Skip if already evaluated
    if [ -f "${OUTPUT_DIR}/categories_qwen3_vl_nudity.json" ]; then
        echo "[GPU ${ACTUAL_GPU}] SKIP (eval done): ${DATASET}/${EXP_NAME}"
        return 0
    fi

    echo "[GPU ${ACTUAL_GPU}] START: ${DATASET}/${EXP_NAME}"

    # Step 1: Generate images (skip if stats file exists)
    if [ ! -f "${OUTPUT_DIR}/generation_stats.json" ]; then
        local GEN_CMD="CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python ${GEN_SCRIPT} \
            --ckpt_path ${SD_CKPT} \
            --prompt_file ${PROMPT_FILE} \
            --output_dir ${OUTPUT_DIR} \
            --classifier_ckpt ${CLASSIFIER_CKPT} \
            --harmful_stats_path ${HARMFUL_STATS_PATH} \
            --gradcam_layer ${GRADCAM_LAYER} \
            --monitoring_mode classifier \
            --monitoring_threshold ${MON_THR} \
            --guidance_scale ${GS} \
            --base_guidance_scale ${BS} \
            --harm_ratio ${HARM_RATIO} \
            --spatial_threshold_start ${SP_START} \
            --spatial_threshold_end ${SP_END} \
            --spatial_threshold_strategy ${THRESHOLD_STRATEGY} \
            --num_inference_steps ${NUM_STEPS} \
            --cfg_scale ${CFG_SCALE} \
            --seed ${SEED} \
            --nsamples ${NSAMPLES}"

        # Add prompt range if specified
        if [ "$START_IDX" -ne 0 ]; then
            GEN_CMD="${GEN_CMD} --start_idx ${START_IDX}"
        fi
        if [ "$END_IDX" -ne -1 ]; then
            GEN_CMD="${GEN_CMD} --end_idx ${END_IDX}"
        fi

        eval ${GEN_CMD} >> "${LOG_FILE}" 2>&1

        if [ $? -ne 0 ]; then
            echo "[GPU ${ACTUAL_GPU}] FAILED generation: ${DATASET}/${EXP_NAME}"
            return 1
        fi
    else
        echo "[GPU ${ACTUAL_GPU}] SKIP generation (done): ${DATASET}/${EXP_NAME}"
    fi

    # Step 2: VLM eval (requires vlm conda env)
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate vlm
    CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python "${VLM_SCRIPT}" "${OUTPUT_DIR}" nudity qwen \
        >> "${LOG_FILE}" 2>&1
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd

    if [ $? -eq 0 ]; then
        echo "[GPU ${ACTUAL_GPU}] DONE: ${DATASET}/${EXP_NAME}"
    else
        echo "[GPU ${ACTUAL_GPU}] FAILED eval: ${DATASET}/${EXP_NAME}"
        return 1
    fi

    return 0
}

# ============================================
# MAIN LOOP (parallel GPU dispatch, multiple jobs per GPU)
# ============================================

TOTAL_SLOTS=$((NUM_GPUS * JOBS_PER_GPU))
declare -A RUNNING_JOBS  # key=slot_id, value=PID
IDX=0

for (( ci=0; ci<TOTAL; ci++ )); do
    combo="${COMBINATIONS[$ci]}"
    IFS='|' read -r MT GS BS SP <<< "$combo"
    SP_START=$(echo $SP | awk '{print $1}')
    SP_END=$(echo $SP | awk '{print $2}')

    # Find available slot
    while true; do
        for ((slot=0; slot<TOTAL_SLOTS; slot++)); do
            JOB_PID="${RUNNING_JOBS[$slot]}"
            if [ -z "$JOB_PID" ] || ! kill -0 "$JOB_PID" 2>/dev/null; then
                GPU_IDX=$((slot % NUM_GPUS))
                run_experiment $GPU_IDX "$MT" "$GS" "$BS" "$SP_START" "$SP_END" &
                RUNNING_JOBS[$slot]=$!
                IDX=$((IDX + 1))
                echo "Progress: ${IDX}/${TOTAL} [combo $ci]"
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

echo ""
echo "=============================================="
echo "GRID SEARCH COMPLETE"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Total experiments: $TOTAL"
echo "Results: ${OUTPUT_BASE}/"
echo ""

# Quick summary: count completed experiments
COMPLETED=0
FAILED=0
for (( ci=0; ci<TOTAL; ci++ )); do
    combo="${COMBINATIONS[$ci]}"
    IFS='|' read -r MT GS BS SP <<< "$combo"
    SP_START=$(echo $SP | awk '{print $1}')
    SP_END=$(echo $SP | awk '{print $2}')
    EXP_NAME="mon${MT}_gs${GS}_bs${BS}_sp${SP_START}-${SP_END}"
    if [ -f "${OUTPUT_BASE}/${EXP_NAME}/categories_qwen3_vl_nudity.json" ]; then
        COMPLETED=$((COMPLETED + 1))
    elif [ -f "${OUTPUT_BASE}/${EXP_NAME}/generation_stats.json" ]; then
        FAILED=$((FAILED + 1))  # generated but eval failed
    fi
done
echo "Completed (with eval): ${COMPLETED}/${TOTAL}"
if [ $FAILED -gt 0 ]; then
    echo "Generated but eval pending: ${FAILED}"
fi
echo ""
echo "Done!"
