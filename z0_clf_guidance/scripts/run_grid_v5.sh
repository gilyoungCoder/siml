#!/bin/bash
# ============================================================================
# Grid Search V5: 4-class Z0 ASCG — Dual Monitoring + harm_ratio + layer variation
#
# Changes from V4:
#   - monitoring_mode = dual (softmax sticky trigger + GradCAM CDF per-step)
#   - monitoring_threshold = 0.08 (softmax sticky), cdf_threshold = 0.05 (per-step)
#   - guidance_scale capped at 20
#   - harm_ratio variation: [1.0, 1.5, 2.0]
#   - gradcam_layer variation: [layer1 (16x16), layer2 (8x8)]
#   - per-layer harmful_stats files
#
# Fixed parameters:
#   sticky_trigger = True
#   grad_clip_ratio = 0.0
#   monitoring_threshold = 0.08 (softmax sticky)
#   cdf_threshold = 0.05 (GradCAM CDF per-step)
#   monitoring_mode = dual
#   monitoring_start_step = 13
#   num_classes = 4
#   spatial_threshold_strategy = cosine
#
# Variable parameters:
#   guidance_scale: [5.0, 7.5, 10.0, 12.5, 15.0, 20.0]    — 6
#   base_guidance_scale: [0.0, 1.0, 2.0]                    — 3
#   harm_ratio: [1.0, 1.5, 2.0]                             — 3
#   gradcam_layer: [layer1, layer2]                          — 2
#   spatial_mode + SP_PAIRS:
#     A) none:             1
#     B) gradcam hard:     10 SP pairs
#     C) gradcam soft:     1
#     D) cross_attn hard:  10 SP pairs
#     E) cross_attn soft:  1
#                                                 subtotal = 23
#
# Total: 6 × 3 × 3 × 2 × 23 = 2,484 configs
#   Group A (none):              6 × 3 × 2 = 36   (bs irrelevant, set bs=gs)
#   Group B (gradcam hard):  6 × 3 × 3 × 2 × 10 = 1,080
#   Group C (gradcam soft):  6 × 3 × 3 × 2      = 108
#   Group D (cross_attn hard): 6 × 3 × 3 × 2 × 10 = 1,080
#   Group E (cross_attn soft): 6 × 3 × 3 × 2    = 108
#   Total: 36 + 1080 + 108 + 1080 + 108 = 2,412
#
# Usage:
#   bash scripts/run_grid_v5.sh --dataset ringabell                          # dry run
#   bash scripts/run_grid_v5.sh --dataset ringabell --run --nohup            # launch
#   bash scripts/run_grid_v5.sh --dataset ringabell --run --nohup --num-gpus 4
# ============================================================================

cd /mnt/home/yhgil99/unlearning/z0_clf_guidance

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd

# ============================================
# Parse Arguments
# ============================================
DRY_RUN=true
USE_NOHUP=false
NUM_GPUS=8
JOBS_PER_GPU=1
DATASET=""
SITE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2;;
        --run) DRY_RUN=false; shift;;
        --dry-run) DRY_RUN=true; shift;;
        --nohup) USE_NOHUP=true; shift;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2;;
        --site) SITE="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "ERROR: --dataset is required (ringabell / coco / unlearndiff / mma)"
    exit 1
fi

# Dataset config
case $DATASET in
    ringabell)
        PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
        END_IDX=-1
        ;;
    coco)
        PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/coco_30k.csv"
        END_IDX=50
        ;;
    unlearndiff)
        PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/unlearn_diff_nudity.csv"
        END_IDX=-1
        ;;
    mma)
        PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/mma-diffusion-nsfw-adv-prompts.csv"
        END_IDX=-1
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
    LOG_DIR="./grid_v5_output/logs"
    mkdir -p "$LOG_DIR"
    SITE_TAG=""
    if [ -n "$SITE" ]; then SITE_TAG="_site${SITE}"; fi
    LOG_FILE="${LOG_DIR}/grid_${DATASET}${SITE_TAG}_${TIMESTAMP}.log"
    echo "Running in background..."
    echo "Log: $LOG_FILE"
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    SITE_ARG=""
    if [ -n "$SITE" ]; then SITE_ARG="--site $SITE"; fi
    nohup bash "$SCRIPT_PATH" --dataset "$DATASET" --run \
        --num-gpus "$NUM_GPUS" --jobs-per-gpu "$JOBS_PER_GPU" $SITE_ARG > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    echo "Monitor: tail -f $LOG_FILE"
    exit 0
fi

# ============================================
# CONFIGURATION
# ============================================

CLASSIFIER_CKPT="./work_dirs/z0_resnet18_4class/checkpoint/step_18900/classifier.pth"
NUM_CLASSES=4
SD_CKPT="CompVis/stable-diffusion-v1-4"
GEN_SCRIPT="generate_monitoring.py"

# Per-layer harmful stats (4-class classifier)
HARMFUL_STATS_LAYER1="./harmful_stats_4class_layer1.pt"
HARMFUL_STATS_LAYER2="./harmful_stats_4class_layer2.pt"

# Fixed params
NUM_STEPS=50
SEED=42
NSAMPLES=1
CFG_SCALE=7.5
THRESHOLD_STRATEGY="cosine"
MON_MODE="dual"
MON_THR=0.08
CDF_THR=0.05
MON_START_STEP=13
GRAD_CLIP=0.0

# VLM eval
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
OUTPUT_BASE="./grid_v5_output/${DATASET}"

# ============================================
# BUILD ALL COMBINATIONS
# ============================================

GUIDANCE_SCALES=(10.0 12.5 15.0 20.0)
BASE_SCALES=(0.0 1.0 2.0)
HARM_RATIOS=(1.0 1.5 2.0)
GRADCAM_LAYERS=(layer1 layer2)

# Spatial threshold pairs (start end) for hard mask
SP_PAIRS=(
    "0.1 0.1"
    "0.2 0.2"
    "0.3 0.3"
    "0.5 0.5"
    "0.7 0.7"
    "0.3 0.1"
    "0.5 0.2"
    "0.7 0.3"
)

declare -a ALL_COMBOS=()

# Group A: spatial_mode=none (6 gs × 3 hr × 2 layer = 36 configs)
# For none mode: bs=gs (uniform guidance), SP doesn't matter
for layer in "${GRADCAM_LAYERS[@]}"; do
    for gs in "${GUIDANCE_SCALES[@]}"; do
        for hr in "${HARM_RATIOS[@]}"; do
            ALL_COMBOS+=("none|hard|${gs}|0.0|0.3|0.3|${hr}|${layer}")
        done
    done
done

# Group B: gradcam hard mask (6 × 3 × 3 × 2 × 10 = 1080 configs)
for layer in "${GRADCAM_LAYERS[@]}"; do
    for gs in "${GUIDANCE_SCALES[@]}"; do
        for bs in "${BASE_SCALES[@]}"; do
            for hr in "${HARM_RATIOS[@]}"; do
                for sp in "${SP_PAIRS[@]}"; do
                    SP_S=$(echo $sp | awk '{print $1}')
                    SP_E=$(echo $sp | awk '{print $2}')
                    ALL_COMBOS+=("gradcam|hard|${gs}|${bs}|${SP_S}|${SP_E}|${hr}|${layer}")
                done
            done
        done
    done
done

# Group C: gradcam soft mask (6 × 3 × 3 × 2 = 108 configs)
for layer in "${GRADCAM_LAYERS[@]}"; do
    for gs in "${GUIDANCE_SCALES[@]}"; do
        for bs in "${BASE_SCALES[@]}"; do
            for hr in "${HARM_RATIOS[@]}"; do
                ALL_COMBOS+=("gradcam|soft|${gs}|${bs}|0.3|0.3|${hr}|${layer}")
            done
        done
    done
done

# Group D: cross_attn hard mask (6 × 3 × 3 × 2 × 10 = 1080 configs)
for layer in "${GRADCAM_LAYERS[@]}"; do
    for gs in "${GUIDANCE_SCALES[@]}"; do
        for bs in "${BASE_SCALES[@]}"; do
            for hr in "${HARM_RATIOS[@]}"; do
                for sp in "${SP_PAIRS[@]}"; do
                    SP_S=$(echo $sp | awk '{print $1}')
                    SP_E=$(echo $sp | awk '{print $2}')
                    ALL_COMBOS+=("cross_attn|hard|${gs}|${bs}|${SP_S}|${SP_E}|${hr}|${layer}")
                done
            done
        done
    done
done

# Group E: cross_attn soft mask (6 × 3 × 3 × 2 = 108 configs)
for layer in "${GRADCAM_LAYERS[@]}"; do
    for gs in "${GUIDANCE_SCALES[@]}"; do
        for bs in "${BASE_SCALES[@]}"; do
            for hr in "${HARM_RATIOS[@]}"; do
                ALL_COMBOS+=("cross_attn|soft|${gs}|${bs}|0.3|0.3|${hr}|${layer}")
            done
        done
    done
done

TOTAL=${#ALL_COMBOS[@]}

# Site splitting (A=60%, B=40% — A server is faster)
SPLIT=$(( TOTAL * 3 / 5 ))
if [ "$SITE" = "A" ]; then
    START=0; END=$SPLIT
elif [ "$SITE" = "B" ]; then
    START=$SPLIT; END=$TOTAL
else
    START=0; END=$TOTAL
fi
SITE_COUNT=$((END - START))

echo "=============================================="
echo "Z0 ASCG GRID SEARCH V5"
echo "  GradCAM CDF monitoring (thr=${MON_THR})"
echo "  harm_ratio variation + layer variation"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Total configs: $TOTAL (site ${SITE:-ALL}: $START to $((END-1)), running $SITE_COUNT)"
echo "Classifier: ${CLASSIFIER_CKPT}"
echo "Fixed: mon=${MON_MODE} thr=${MON_THR} cdf_thr=${CDF_THR} start_step=${MON_START_STEP} sticky=T clip=${GRAD_CLIP}"
echo "Variable: gs=[10-20] bs=[0,1,2] hr=[1,1.5,2] layer=[layer1,layer2]"
echo "GPUs: $NUM_GPUS (${JOBS_PER_GPU} jobs/GPU = $((NUM_GPUS * JOBS_PER_GPU)) concurrent)"
echo "=============================================="

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY RUN] $SITE_COUNT experiments (site ${SITE:-ALL}):"
    for (( ci=START; ci<END; ci++ )); do
        combo="${ALL_COMBOS[$ci]}"
        IFS='|' read -r MODE MASK GS BS SP_S SP_E HR LAYER <<< "$combo"
        echo "  [$ci] ${LAYER}_${MODE}_${MASK}_gs${GS}_bs${BS}_hr${HR}_sp${SP_S}-${SP_E}"
    done
    echo ""
    echo "Total: $SITE_COUNT experiments on $NUM_GPUS GPUs"
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
    local HR=$8
    local LAYER=$9

    local ACTUAL_GPU=${GPU_LIST[$GPU_IDX]}
    local EXP_NAME="${LAYER}_${MODE}_${MASK}_gs${GS}_bs${BS}_hr${HR}_sp${SP_S}-${SP_E}"
    local OUTPUT_DIR="${OUTPUT_BASE}/${EXP_NAME}"
    local LOG_FILE="${OUTPUT_BASE}/logs/${EXP_NAME}.log"

    # Select per-layer harmful stats
    local STATS_PATH="${HARMFUL_STATS_LAYER2}"
    if [ "$LAYER" = "layer1" ]; then
        STATS_PATH="${HARMFUL_STATS_LAYER1}"
    fi

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

        local END_IDX_ARG=""
        if [ "$END_IDX" != "-1" ]; then
            END_IDX_ARG="--end_idx ${END_IDX}"
        fi

        CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python ${GEN_SCRIPT} \
            --ckpt_path ${SD_CKPT} \
            --prompt_file ${PROMPT_FILE} \
            --output_dir "${OUTPUT_DIR}" \
            --classifier_ckpt ${CLASSIFIER_CKPT} \
            --harmful_stats_path ${STATS_PATH} \
            --num_classes ${NUM_CLASSES} \
            --gradcam_layer ${LAYER} \
            --monitoring_mode ${MON_MODE} \
            --monitoring_threshold ${MON_THR} \
            --cdf_threshold ${CDF_THR} \
            --monitoring_start_step ${MON_START_STEP} \
            --guidance_scale ${GS} \
            --base_guidance_scale ${BS_ARG} \
            --harm_ratio ${HR} \
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
            ${END_IDX_ARG} \
            >> "${LOG_FILE}" 2>&1

        if [ $? -ne 0 ]; then
            echo "[GPU ${ACTUAL_GPU}] FAILED generation: ${EXP_NAME}"
            return 1
        fi
    else
        echo "[GPU ${ACTUAL_GPU}] SKIP generation (done): ${EXP_NAME}"
    fi

    # Step 2: VLM eval (skip for coco)
    if [ "$DATASET" != "coco" ]; then
        eval "$(conda shell.bash hook 2>/dev/null)" && conda activate vlm
        CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python "${VLM_SCRIPT}" "${OUTPUT_DIR}" nudity qwen \
            >> "${LOG_FILE}" 2>&1
        eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
    fi

    echo "[GPU ${ACTUAL_GPU}] DONE: ${EXP_NAME}"
    return 0
}

# ============================================
# MAIN LOOP
# ============================================

TOTAL_SLOTS=$((NUM_GPUS * JOBS_PER_GPU))
declare -A RUNNING_JOBS
LOCAL_IDX=0

for (( ci=START; ci<END; ci++ )); do
    combo="${ALL_COMBOS[$ci]}"
    IFS='|' read -r MODE MASK GS BS SP_S SP_E HR LAYER <<< "$combo"

    # Find available slot
    while true; do
        for ((slot=0; slot<TOTAL_SLOTS; slot++)); do
            JOB_PID="${RUNNING_JOBS[$slot]}"
            if [ -z "$JOB_PID" ] || ! kill -0 "$JOB_PID" 2>/dev/null; then
                GPU_IDX=$((slot % NUM_GPUS))
                run_experiment $GPU_IDX "$MODE" "$MASK" "$GS" "$BS" "$SP_S" "$SP_E" "$HR" "$LAYER" &
                RUNNING_JOBS[$slot]=$!
                LOCAL_IDX=$((LOCAL_IDX + 1))
                echo "Progress: ${LOCAL_IDX}/${SITE_COUNT} [combo $ci]"
                sleep 2
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
echo "GRID SEARCH V5 COMPLETE"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Experiments: $TOTAL"
echo "Results: ${OUTPUT_BASE}/"
echo ""

# Quick summary
COMPLETED=0
for (( ci=0; ci<TOTAL; ci++ )); do
    combo="${ALL_COMBOS[$ci]}"
    IFS='|' read -r MODE MASK GS BS SP_S SP_E HR LAYER <<< "$combo"
    EXP_NAME="${LAYER}_${MODE}_${MASK}_gs${GS}_bs${BS}_hr${HR}_sp${SP_S}-${SP_E}"
    if [ -f "${OUTPUT_BASE}/${EXP_NAME}/generation_stats.json" ]; then
        COMPLETED=$((COMPLETED + 1))
    fi
done
echo "Completed: ${COMPLETED}/${TOTAL}"
echo "Done!"
