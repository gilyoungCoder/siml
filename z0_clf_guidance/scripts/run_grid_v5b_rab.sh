#!/bin/bash
# ============================================================================
# Grid Search V5b-RAB: Same as V5b but with RAB-trained classifier (step_15900)
#
# Part 1 (544 experiments): grad_clip_ratio=[0.1,0.3], HR=[1.5,2.5], BS=[1.0,3.0]
#   - monitoring same as V5 (dual, thr=0.08, cdf=0.05, start=13)
#
# Part 2 (320 experiments): monitoring variations
#   - dual: cdf_thr=[0.01,0.1], start=[10,20]
#   - classifier: softmax_thr=[0.05,0.15]
#   - gradcam: cdf_thr=[0.03,0.08]
#   - Fixed: HR=1.5, BS=1.0, grad_clip=0.0
#
# Total: 864 experiments
# ============================================================================

cd /mnt/home/yhgil99/unlearning/z0_clf_guidance

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
export PYTHONNOUSERSITE=1

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
    LOG_DIR="./grid_v5b_output/logs"
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

CLASSIFIER_CKPT="./work_dirs/z0_resnet18_4class_ringabell/checkpoint/step_15900/classifier.pth"
NUM_CLASSES=4
SD_CKPT="CompVis/stable-diffusion-v1-4"
GEN_SCRIPT="generate_monitoring.py"

# Per-layer harmful stats (RAB classifier - only layer2 available)
HARMFUL_STATS_LAYER1="./harmful_stats_4class_ringabell_layer2.pt"
HARMFUL_STATS_LAYER2="./harmful_stats_4class_ringabell_layer2.pt"

# Shared fixed params
NUM_STEPS=50
SEED=42
NSAMPLES=1
CFG_SCALE=7.5
THRESHOLD_STRATEGY="cosine"

# VLM eval
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
OUTPUT_BASE="./grid_v5b_rab_output/${DATASET}"

# ============================================
# BUILD ALL COMBINATIONS
# Combo format: MODE|MASK|GS|BS|SP_S|SP_E|HR|LAYER|CLIP|MON_MODE|MON_THR|CDF_THR|MON_START
# ============================================

GUIDANCE_SCALES=(10.0 12.5 15.0 20.0)
GRADCAM_LAYERS=(layer1 layer2)

declare -a ALL_COMBOS=()

# ============================================
# PART 1: grad_clip exploration (544 experiments)
# ============================================
P1_CLIPS=(0.1 0.3)
P1_HRS=(1.5 2.5)
P1_BSS=(1.0 3.0)
P1_SP_PAIRS=("0.3 0.3" "0.5 0.5" "0.7 0.3")
P1_MON_MODE="dual"
P1_MON_THR=0.08
P1_CDF_THR=0.05
P1_MON_START=13

# Part 1 Group A: none
for layer in "${GRADCAM_LAYERS[@]}"; do
    for gs in "${GUIDANCE_SCALES[@]}"; do
        for clip in "${P1_CLIPS[@]}"; do
            for hr in "${P1_HRS[@]}"; do
                ALL_COMBOS+=("none|hard|${gs}|0.0|0.3|0.3|${hr}|${layer}|${clip}|${P1_MON_MODE}|${P1_MON_THR}|${P1_CDF_THR}|${P1_MON_START}")
            done
        done
    done
done

# Part 1 Group B: gradcam hard
for layer in "${GRADCAM_LAYERS[@]}"; do
    for gs in "${GUIDANCE_SCALES[@]}"; do
        for bs in "${P1_BSS[@]}"; do
            for clip in "${P1_CLIPS[@]}"; do
                for hr in "${P1_HRS[@]}"; do
                    for sp in "${P1_SP_PAIRS[@]}"; do
                        SP_S=$(echo $sp | awk '{print $1}')
                        SP_E=$(echo $sp | awk '{print $2}')
                        ALL_COMBOS+=("gradcam|hard|${gs}|${bs}|${SP_S}|${SP_E}|${hr}|${layer}|${clip}|${P1_MON_MODE}|${P1_MON_THR}|${P1_CDF_THR}|${P1_MON_START}")
                    done
                done
            done
        done
    done
done

# Part 1 Group C: gradcam soft
for layer in "${GRADCAM_LAYERS[@]}"; do
    for gs in "${GUIDANCE_SCALES[@]}"; do
        for bs in "${P1_BSS[@]}"; do
            for clip in "${P1_CLIPS[@]}"; do
                for hr in "${P1_HRS[@]}"; do
                    ALL_COMBOS+=("gradcam|soft|${gs}|${bs}|0.3|0.3|${hr}|${layer}|${clip}|${P1_MON_MODE}|${P1_MON_THR}|${P1_CDF_THR}|${P1_MON_START}")
                done
            done
        done
    done
done

# Part 1 Group D: cross_attn hard
for layer in "${GRADCAM_LAYERS[@]}"; do
    for gs in "${GUIDANCE_SCALES[@]}"; do
        for bs in "${P1_BSS[@]}"; do
            for clip in "${P1_CLIPS[@]}"; do
                for hr in "${P1_HRS[@]}"; do
                    for sp in "${P1_SP_PAIRS[@]}"; do
                        SP_S=$(echo $sp | awk '{print $1}')
                        SP_E=$(echo $sp | awk '{print $2}')
                        ALL_COMBOS+=("cross_attn|hard|${gs}|${bs}|${SP_S}|${SP_E}|${hr}|${layer}|${clip}|${P1_MON_MODE}|${P1_MON_THR}|${P1_CDF_THR}|${P1_MON_START}")
                    done
                done
            done
        done
    done
done

# Part 1 Group E: cross_attn soft
for layer in "${GRADCAM_LAYERS[@]}"; do
    for gs in "${GUIDANCE_SCALES[@]}"; do
        for bs in "${P1_BSS[@]}"; do
            for clip in "${P1_CLIPS[@]}"; do
                for hr in "${P1_HRS[@]}"; do
                    ALL_COMBOS+=("cross_attn|soft|${gs}|${bs}|0.3|0.3|${hr}|${layer}|${clip}|${P1_MON_MODE}|${P1_MON_THR}|${P1_CDF_THR}|${P1_MON_START}")
                done
            done
        done
    done
done

# ============================================
# PART 2: monitoring exploration (320 experiments)
# Fixed: HR=1.5, BS=1.0, grad_clip=0.0
# ============================================
P2_HR=1.5
P2_BS=1.0
P2_CLIP=0.0
P2_SP_PAIRS=("0.3 0.3" "0.5 0.5")

# 8 monitoring combos: (mode, mon_thr, cdf_thr, start)
P2_MON_COMBOS=(
    "dual|0.08|0.01|10"
    "dual|0.08|0.01|20"
    "dual|0.08|0.1|10"
    "dual|0.08|0.1|20"
    "classifier|0.05|0.05|13"
    "classifier|0.15|0.05|13"
    "gradcam|0.03|0.05|13"
    "gradcam|0.08|0.05|13"
)

for mon_combo in "${P2_MON_COMBOS[@]}"; do
    IFS='|' read -r MMODE MTHR CDFTHR MSTART <<< "$mon_combo"

    for layer in "${GRADCAM_LAYERS[@]}"; do
        for gs in "${GUIDANCE_SCALES[@]}"; do
            # none
            ALL_COMBOS+=("none|hard|${gs}|0.0|0.3|0.3|${P2_HR}|${layer}|${P2_CLIP}|${MMODE}|${MTHR}|${CDFTHR}|${MSTART}")

            # gradcam hard
            for sp in "${P2_SP_PAIRS[@]}"; do
                SP_S=$(echo $sp | awk '{print $1}')
                SP_E=$(echo $sp | awk '{print $2}')
                ALL_COMBOS+=("gradcam|hard|${gs}|${P2_BS}|${SP_S}|${SP_E}|${P2_HR}|${layer}|${P2_CLIP}|${MMODE}|${MTHR}|${CDFTHR}|${MSTART}")
            done

            # cross_attn hard
            for sp in "${P2_SP_PAIRS[@]}"; do
                SP_S=$(echo $sp | awk '{print $1}')
                SP_E=$(echo $sp | awk '{print $2}')
                ALL_COMBOS+=("cross_attn|hard|${gs}|${P2_BS}|${SP_S}|${SP_E}|${P2_HR}|${layer}|${P2_CLIP}|${MMODE}|${MTHR}|${CDFTHR}|${MSTART}")
            done
        done
    done
done

TOTAL=${#ALL_COMBOS[@]}

# Site splitting (A=60%, B=40%)
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
echo "Z0 ASCG GRID SEARCH V5b"
echo "  Part 1: grad_clip exploration"
echo "  Part 2: monitoring variations"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Total configs: $TOTAL (site ${SITE:-ALL}: $START to $((END-1)), running $SITE_COUNT)"
echo "Classifier: ${CLASSIFIER_CKPT}"
echo "Part 1: clip=[0.1,0.3] hr=[1.5,2.5] bs=[1.0,3.0] mon=dual"
echo "Part 2: monitoring variations (dual/classifier/gradcam)"
echo "GPUs: $NUM_GPUS (${JOBS_PER_GPU} jobs/GPU = $((NUM_GPUS * JOBS_PER_GPU)) concurrent)"
echo "=============================================="

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY RUN] $SITE_COUNT experiments (site ${SITE:-ALL}):"
    for (( ci=START; ci<END; ci++ )); do
        combo="${ALL_COMBOS[$ci]}"
        IFS='|' read -r MODE MASK GS BS SP_S SP_E HR LAYER CLIP MMODE MTHR CDFTHR MSTART <<< "$combo"
        echo "  [$ci] ${LAYER}_${MODE}_${MASK}_gs${GS}_bs${BS}_hr${HR}_sp${SP_S}-${SP_E}_clip${CLIP}_${MMODE}_mt${MTHR}_cdf${CDFTHR}_ms${MSTART}"
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
    local CLIP=${10}
    local MMODE=${11}
    local MTHR=${12}
    local CDFTHR=${13}
    local MSTART=${14}

    local ACTUAL_GPU=${GPU_LIST[$GPU_IDX]}
    local EXP_NAME="${LAYER}_${MODE}_${MASK}_gs${GS}_bs${BS}_hr${HR}_sp${SP_S}-${SP_E}_clip${CLIP}_${MMODE}_mt${MTHR}_cdf${CDFTHR}_ms${MSTART}"
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
            --monitoring_mode ${MMODE} \
            --monitoring_threshold ${MTHR} \
            --cdf_threshold ${CDFTHR} \
            --monitoring_start_step ${MSTART} \
            --guidance_scale ${GS} \
            --base_guidance_scale ${BS_ARG} \
            --harm_ratio ${HR} \
            --spatial_threshold_start ${SP_S_ARG} \
            --spatial_threshold_end ${SP_E_ARG} \
            --spatial_threshold_strategy ${STRATEGY_ARG} \
            --spatial_mode ${SPATIAL_MODE_ARG} \
            --grad_clip_ratio ${CLIP} \
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
    IFS='|' read -r MODE MASK GS BS SP_S SP_E HR LAYER CLIP MMODE MTHR CDFTHR MSTART <<< "$combo"

    # Find available slot
    while true; do
        for ((slot=0; slot<TOTAL_SLOTS; slot++)); do
            JOB_PID="${RUNNING_JOBS[$slot]}"
            if [ -z "$JOB_PID" ] || ! kill -0 "$JOB_PID" 2>/dev/null; then
                GPU_IDX=$((slot % NUM_GPUS))
                run_experiment $GPU_IDX "$MODE" "$MASK" "$GS" "$BS" "$SP_S" "$SP_E" "$HR" "$LAYER" "$CLIP" "$MMODE" "$MTHR" "$CDFTHR" "$MSTART" &
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
echo "GRID SEARCH V5b COMPLETE"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Experiments: $TOTAL"
echo "Results: ${OUTPUT_BASE}/"
echo ""

# Quick summary
COMPLETED=0
for (( ci=0; ci<TOTAL; ci++ )); do
    combo="${ALL_COMBOS[$ci]}"
    IFS='|' read -r MODE MASK GS BS SP_S SP_E HR LAYER CLIP MMODE MTHR CDFTHR MSTART <<< "$combo"
    EXP_NAME="${LAYER}_${MODE}_${MASK}_gs${GS}_bs${BS}_hr${HR}_sp${SP_S}-${SP_E}_clip${CLIP}_${MMODE}_mt${MTHR}_cdf${CDFTHR}_ms${MSTART}"
    if [ -f "${OUTPUT_BASE}/${EXP_NAME}/generation_stats.json" ]; then
        COMPLETED=$((COMPLETED + 1))
    fi
done
echo "Completed: ${COMPLETED}/${TOTAL}"
echo "Done!"
