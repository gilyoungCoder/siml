#!/bin/bash
# ============================================================================
# Grid Search V4: 4-class Z0 ASCG — Optimized monitoring (Step 13, mon=0.08)
#
# Changes from V3:
#   - 4-class z0 ResNet18 classifier (step_18900)
#   - num_classes = 4 (harm = nude + color)
#   - monitoring_threshold = 0.08 (RB 92.4%, COCO 8.0%, gap +84.4%)
#
# Fixed parameters:
#   sticky_trigger = True
#   grad_clip_ratio = 0.0
#   monitoring_threshold = 0.08
#   monitoring_start_step = 13
#   harm_ratio = 1.0
#   num_classes = 4
#   spatial_threshold_strategy = cosine
#
# Variable parameters:
#   guidance_scale: [5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0]
#   base_guidance_scale: [0.0, 1.0, 2.0]
#   spatial_mode: none / gradcam / cross_attn
#   spatial_soft: hard (binary) / soft (continuous CDF)
#   spatial_threshold pairs: 10 (start, end) for hard mask
#
# Total: 536 configs
#   A) none:             8
#   B) gradcam hard:   240  (8 gs × 3 base × 10 sp_pairs)
#   C) gradcam soft:    24  (8 gs × 3 base)
#   D) cross_attn hard: 240  (8 gs × 3 base × 10 sp_pairs)
#   E) cross_attn soft:  24  (8 gs × 3 base)
#
# Server split (2 servers, proportional to GPU count):
#   --server 0: siml1 (4 GPUs)  — 179 configs (indices 0-178)
#   --server 1: other (8 GPUs)  — 357 configs (indices 179-535)
#
# Usage:
#   bash scripts/run_grid_v4.sh --dataset ringabell --server 0 --num-gpus 4             # dry run siml1
#   bash scripts/run_grid_v4.sh --dataset ringabell --server 0 --num-gpus 4 --run --nohup  # siml1
#   bash scripts/run_grid_v4.sh --dataset ringabell --server 1 --run --nohup               # other
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
SERVER_ID=-1

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2;;
        --server) SERVER_ID="$2"; shift 2;;
        --run) DRY_RUN=false; shift;;
        --dry-run) DRY_RUN=true; shift;;
        --nohup) USE_NOHUP=true; shift;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

# Validate args
if [ -z "$DATASET" ]; then
    echo "ERROR: --dataset is required (ringabell / unlearndiff / mma)"
    exit 1
fi
if [ "$SERVER_ID" -lt 0 ] || [ "$SERVER_ID" -gt 1 ]; then
    echo "ERROR: --server is required (0=siml1 4GPU, 1=other 8GPU)"
    exit 1
fi

# Dataset config
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
        echo "ERROR: Unknown dataset '$DATASET'"; exit 1;;
esac

if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: Prompt file not found: $PROMPT_FILE"; exit 1
fi

# Nohup
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="./grid_v4_output/logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/grid_${DATASET}_server${SERVER_ID}_${TIMESTAMP}.log"
    echo "Running in background..."
    echo "Log: $LOG_FILE"
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    nohup bash "$SCRIPT_PATH" --dataset "$DATASET" --server "$SERVER_ID" --run \
        --num-gpus "$NUM_GPUS" --jobs-per-gpu "$JOBS_PER_GPU" > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    echo "Monitor: tail -f $LOG_FILE"
    exit 0
fi

# ============================================
# CONFIGURATION
# ============================================

# 4-class z0 ResNet18 classifier
CLASSIFIER_CKPT="./work_dirs/z0_resnet18_4class/checkpoint/step_18900/classifier.pth"
NUM_CLASSES=4
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
MON_THR=0.08
MON_START_STEP=13
GRAD_CLIP=0.0

# VLM eval
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
OUTPUT_BASE="./grid_v4_output/${DATASET}"

# ============================================
# BUILD ALL COMBINATIONS
# ============================================

GUIDANCE_SCALES=(5.0 7.5 10.0 12.5 15.0 20.0 25.0 30.0)
BASE_SCALES=(0.0 1.0 2.0)

# Spatial threshold pairs (start end) for hard mask
SP_PAIRS=(
    "0.1 0.1"
    "0.2 0.2"
    "0.3 0.3"
    "0.5 0.5"
    "0.7 0.7"
    "0.9 0.9"
    "0.3 0.1"
    "0.5 0.2"
    "0.7 0.3"
    "0.9 0.5"
)

declare -a ALL_COMBOS=()

# Group A: spatial_mode=none (8 configs)
for gs in "${GUIDANCE_SCALES[@]}"; do
    ALL_COMBOS+=("none|hard|${gs}|0.0|0.3|0.3")
done

# Group B: gradcam hard mask (8 × 3 × 10 = 240 configs)
for gs in "${GUIDANCE_SCALES[@]}"; do
    for bs in "${BASE_SCALES[@]}"; do
        for sp in "${SP_PAIRS[@]}"; do
            SP_S=$(echo $sp | awk '{print $1}')
            SP_E=$(echo $sp | awk '{print $2}')
            ALL_COMBOS+=("gradcam|hard|${gs}|${bs}|${SP_S}|${SP_E}")
        done
    done
done

# Group C: gradcam soft mask (8 × 3 = 24 configs)
for gs in "${GUIDANCE_SCALES[@]}"; do
    for bs in "${BASE_SCALES[@]}"; do
        ALL_COMBOS+=("gradcam|soft|${gs}|${bs}|0.3|0.3")
    done
done

# Group D: cross_attn hard mask (8 × 3 × 10 = 240 configs)
for gs in "${GUIDANCE_SCALES[@]}"; do
    for bs in "${BASE_SCALES[@]}"; do
        for sp in "${SP_PAIRS[@]}"; do
            SP_S=$(echo $sp | awk '{print $1}')
            SP_E=$(echo $sp | awk '{print $2}')
            ALL_COMBOS+=("cross_attn|hard|${gs}|${bs}|${SP_S}|${SP_E}")
        done
    done
done

# Group E: cross_attn soft mask (8 × 3 = 24 configs)
for gs in "${GUIDANCE_SCALES[@]}"; do
    for bs in "${BASE_SCALES[@]}"; do
        ALL_COMBOS+=("cross_attn|soft|${gs}|${bs}|0.3|0.3")
    done
done

TOTAL=${#ALL_COMBOS[@]}

# ============================================
# SERVER SPLIT
# ============================================
case $SERVER_ID in
    0) COMBO_START=0;   COMBO_END=179;;   # siml1 (4 GPUs): 179 configs
    1) COMBO_START=179; COMBO_END=${TOTAL};;  # other (8 GPUs): 357 configs
esac

SERVER_COUNT=$((COMBO_END - COMBO_START))

echo "=============================================="
echo "Z0 ASCG GRID SEARCH V4 (4-class z0 classifier)"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Server: $SERVER_ID (indices ${COMBO_START}-$((COMBO_END-1)))"
echo "Configs: $SERVER_COUNT / $TOTAL total"
echo "Classifier: ${CLASSIFIER_CKPT}"
echo "Fixed: num_classes=${NUM_CLASSES} mon=${MON_THR} start_step=${MON_START_STEP} sticky=T cl0=0 hr=${HARM_RATIO}"
echo "GPUs: $NUM_GPUS (${JOBS_PER_GPU} jobs/GPU = $((NUM_GPUS * JOBS_PER_GPU)) concurrent)"
echo "=============================================="

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY RUN] Experiments for server $SERVER_ID:"
    for (( ci=COMBO_START; ci<COMBO_END; ci++ )); do
        combo="${ALL_COMBOS[$ci]}"
        IFS='|' read -r MODE MASK GS BS SP_S SP_E <<< "$combo"
        GPU_IDX=$(( (ci - COMBO_START) % NUM_GPUS ))
        echo "  [$ci] GPU${GPU_IDX} ${MODE}_${MASK}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}"
    done
    echo ""
    echo "Total: $SERVER_COUNT experiments on $NUM_GPUS GPUs (~$((SERVER_COUNT / NUM_GPUS)) per GPU)"
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
    local MODE=$2
    local MASK=$3
    local GS=$4
    local BS=$5
    local SP_S=$6
    local SP_E=$7

    local ACTUAL_GPU=${GPU_LIST[$GPU_IDX]}
    local EXP_NAME="${MODE}_${MASK}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}"
    local OUTPUT_DIR="${OUTPUT_BASE}/${EXP_NAME}"
    local LOG_FILE="${OUTPUT_BASE}/logs/${EXP_NAME}.log"

    # Skip if already evaluated
    if [ -f "${OUTPUT_DIR}/categories_qwen3_vl_nudity.json" ]; then
        echo "[GPU ${ACTUAL_GPU}] SKIP (eval done): ${EXP_NAME}"
        return 0
    fi

    echo "[GPU ${ACTUAL_GPU}] START: ${EXP_NAME}"

    # Step 1: Generate images
    if [ ! -f "${OUTPUT_DIR}/generation_stats.json" ]; then
        local SOFT_FLAG=""
        if [ "$MASK" = "soft" ]; then
            SOFT_FLAG="--spatial_soft"
        fi

        local SPATIAL_MODE_ARG="$MODE"
        local BS_ARG="$BS"
        local SP_S_ARG="$SP_S"
        local SP_E_ARG="$SP_E"
        local STRATEGY_ARG="${THRESHOLD_STRATEGY}"

        # For mode=none: set base=gs so mask doesn't matter
        if [ "$MODE" = "none" ]; then
            SPATIAL_MODE_ARG="gradcam"
            BS_ARG="${GS}"
            SP_S_ARG="0.5"
            SP_E_ARG="0.5"
            STRATEGY_ARG="constant"
        fi

        CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python ${GEN_SCRIPT} \
            --ckpt_path ${SD_CKPT} \
            --prompt_file ${PROMPT_FILE} \
            --output_dir "${OUTPUT_DIR}" \
            --classifier_ckpt ${CLASSIFIER_CKPT} \
            --harmful_stats_path ${HARMFUL_STATS_PATH} \
            --num_classes ${NUM_CLASSES} \
            --gradcam_layer ${GRADCAM_LAYER} \
            --monitoring_mode classifier \
            --monitoring_threshold ${MON_THR} \
            --monitoring_start_step ${MON_START_STEP} \
            --guidance_scale ${GS} \
            --base_guidance_scale ${BS_ARG} \
            --harm_ratio ${HARM_RATIO} \
            --spatial_threshold_start ${SP_S_ARG} \
            --spatial_threshold_end ${SP_E_ARG} \
            --spatial_threshold_strategy ${STRATEGY_ARG} \
            --spatial_mode ${SPATIAL_MODE_ARG} \
            --grad_clip_ratio ${GRAD_CLIP} \
            --num_inference_steps ${NUM_STEPS} \
            --cfg_scale ${CFG_SCALE} \
            --seed ${SEED} \
            --nsamples ${NSAMPLES} \
            --sticky_trigger \
            ${SOFT_FLAG} \
            >> "${LOG_FILE}" 2>&1

        if [ $? -ne 0 ]; then
            echo "[GPU ${ACTUAL_GPU}] FAILED generation: ${EXP_NAME}"
            return 1
        fi
    else
        echo "[GPU ${ACTUAL_GPU}] SKIP generation (done): ${EXP_NAME}"
    fi

    # Step 2: VLM eval
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate vlm
    CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python "${VLM_SCRIPT}" "${OUTPUT_DIR}" nudity qwen \
        >> "${LOG_FILE}" 2>&1
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd

    if [ $? -eq 0 ]; then
        echo "[GPU ${ACTUAL_GPU}] DONE: ${EXP_NAME}"
    else
        echo "[GPU ${ACTUAL_GPU}] FAILED eval: ${EXP_NAME}"
        return 1
    fi

    return 0
}

# ============================================
# MAIN LOOP
# ============================================

TOTAL_SLOTS=$((NUM_GPUS * JOBS_PER_GPU))
declare -A RUNNING_JOBS
LOCAL_IDX=0

for (( ci=COMBO_START; ci<COMBO_END; ci++ )); do
    combo="${ALL_COMBOS[$ci]}"
    IFS='|' read -r MODE MASK GS BS SP_S SP_E <<< "$combo"

    # Find available slot
    while true; do
        for ((slot=0; slot<TOTAL_SLOTS; slot++)); do
            JOB_PID="${RUNNING_JOBS[$slot]}"
            if [ -z "$JOB_PID" ] || ! kill -0 "$JOB_PID" 2>/dev/null; then
                GPU_IDX=$((slot % NUM_GPUS))
                run_experiment $GPU_IDX "$MODE" "$MASK" "$GS" "$BS" "$SP_S" "$SP_E" &
                RUNNING_JOBS[$slot]=$!
                LOCAL_IDX=$((LOCAL_IDX + 1))
                echo "Progress: ${LOCAL_IDX}/${SERVER_COUNT} [combo $ci]"
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
echo "GRID SEARCH V4 COMPLETE"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Server: $SERVER_ID"
echo "Experiments: $SERVER_COUNT"
echo "Results: ${OUTPUT_BASE}/"
echo ""

# Quick summary
COMPLETED=0
for (( ci=COMBO_START; ci<COMBO_END; ci++ )); do
    combo="${ALL_COMBOS[$ci]}"
    IFS='|' read -r MODE MASK GS BS SP_S SP_E <<< "$combo"
    EXP_NAME="${MODE}_${MASK}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}"
    if [ -f "${OUTPUT_BASE}/${EXP_NAME}/categories_qwen3_vl_nudity.json" ]; then
        COMPLETED=$((COMPLETED + 1))
    fi
done
echo "Completed (with eval): ${COMPLETED}/${SERVER_COUNT}"
echo "Done!"
