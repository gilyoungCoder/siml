#!/bin/bash
# ============================================================================
# Grid Search: SGF Time Window + ASCG Spatial CG
#
# Replaces monitoring-based "when" with SGF's critical time window.
# Keeps ASCG spatial masking + dual classifier gradient.
#
# Supports --dataset ringabell / coco
#   - ringabell: full 79 prompts, with VLM eval
#   - coco: 50 prompts, no VLM eval (FP check only)
# ============================================================================

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
export PYTHONNOUSERSITE=1

# ============================================
# Parse Arguments
# ============================================
DRY_RUN=true
USE_NOHUP=false
NUM_GPUS=8
SITE=""
DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2;;
        --run) DRY_RUN=false; shift;;
        --dry-run) DRY_RUN=true; shift;;
        --nohup) USE_NOHUP=true; shift;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        --site) SITE="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "ERROR: --dataset is required (ringabell / coco)"
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
    *)
        echo "ERROR: Unknown dataset '$DATASET' (use ringabell or coco)"; exit 1;;
esac

if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: Prompt file not found: $PROMPT_FILE"; exit 1
fi

# Nohup handling
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="./scg_outputs"
    mkdir -p "$LOG_DIR"
    SITE_TAG=""
    if [ -n "$SITE" ]; then SITE_TAG="_site${SITE}"; fi
    LOG_FILE="${LOG_DIR}/sgf_window_${DATASET}${SITE_TAG}_${TIMESTAMP}.log"
    echo "Running in background..."
    echo "Log: $LOG_FILE"
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    SITE_ARG=""
    if [ -n "$SITE" ]; then SITE_ARG="--site $SITE"; fi
    nohup bash "$SCRIPT_PATH" --dataset "$DATASET" --run --num-gpus "$NUM_GPUS" $SITE_ARG > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    echo "Monitor: tail -f $LOG_FILE"
    exit 0
fi

# ============================================
# CONFIGURATION
# ============================================

CLASSIFIER_CKPT="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth"
GRADCAM_STATS_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/nudity_4class_ringabell"
SD_CKPT="CompVis/stable-diffusion-v1-4"
GEN_SCRIPT="generate_sgf_window_unified.py"

# Fixed params
NUM_STEPS=50
SEED=42
NSAMPLES=1
CFG_SCALE=7.5

HARMFUL_KW="nude naked breast topless bare undress nsfw"

# VLM eval
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
OUTPUT_BASE="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/sgf_window_grid/${DATASET}"

# ============================================
# Variable parameters
# ============================================

# SGF time windows (start_t, end_t)
WINDOWS=("1000 800" "1000 600" "1000 400" "1000 200")

# Guidance schedules
SCHEDULES=("cosine" "constant" "linear")

# Guidance scales
GUIDANCE_SCALES=(10.0 20.0)

# Base scales
BASE_SCALES=(0.0 2.0 3.0)

# Spatial threshold pairs (start end)
SP_PAIRS_GC=("0.2 0.4" "0.3 0.5")
SP_PAIRS_ATTN=("0.2 0.3" "0.3 0.5")

# ============================================
# BUILD COMBINATIONS
# Format: SMODE|START_T|END_T|SCHED|GS|BS|SP_START|SP_END
# ============================================

declare -a ALL_COMBOS=()

# --- Group 1: gradcam spatial ---
for win in "${WINDOWS[@]}"; do
    START_T=$(echo $win | awk '{print $1}')
    END_T=$(echo $win | awk '{print $2}')
    for sched in "${SCHEDULES[@]}"; do
        for gs in "${GUIDANCE_SCALES[@]}"; do
            for bs in "${BASE_SCALES[@]}"; do
                for sp in "${SP_PAIRS_GC[@]}"; do
                    SP_S=$(echo $sp | awk '{print $1}')
                    SP_E=$(echo $sp | awk '{print $2}')
                    ALL_COMBOS+=("gradcam|${START_T}|${END_T}|${sched}|${gs}|${bs}|${SP_S}|${SP_E}")
                done
            done
        done
    done
done

# --- Group 2: attention spatial ---
for win in "${WINDOWS[@]}"; do
    START_T=$(echo $win | awk '{print $1}')
    END_T=$(echo $win | awk '{print $2}')
    for sched in "${SCHEDULES[@]}"; do
        for gs in "${GUIDANCE_SCALES[@]}"; do
            for bs in "${BASE_SCALES[@]}"; do
                for sp in "${SP_PAIRS_ATTN[@]}"; do
                    SP_S=$(echo $sp | awk '{print $1}')
                    SP_E=$(echo $sp | awk '{print $2}')
                    ALL_COMBOS+=("attention|${START_T}|${END_T}|${sched}|${gs}|${bs}|${SP_S}|${SP_E}")
                done
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
echo "SGF WINDOW + ASCG SPATIAL CG GRID SEARCH"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Total configs: $TOTAL (site ${SITE:-ALL}: $START to $((END-1)), running $SITE_COUNT)"
echo "Windows: ${WINDOWS[*]}"
echo "Schedules: ${SCHEDULES[*]}"
echo "Spatial: gradcam, attention"
echo "GPUs: $NUM_GPUS"
echo "=============================================="

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY RUN] $SITE_COUNT experiments (site ${SITE:-ALL}):"
    for (( ci=START; ci<END; ci++ )); do
        combo="${ALL_COMBOS[$ci]}"
        IFS='|' read -r SMODE ST ET SCHED GS BS SP_S SP_E <<< "$combo"
        echo "  [$ci] ${SMODE}_w${ST}-${ET}_${SCHED}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}"
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
    local SMODE=$2
    local ST=$3
    local ET=$4
    local SCHED=$5
    local GS=$6
    local BS=$7
    local SP_S=$8
    local SP_E=$9

    local ACTUAL_GPU=${GPU_LIST[$GPU_IDX]}
    local EXP_NAME="${SMODE}_w${ST}-${ET}_${SCHED}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}"
    local OUTPUT_DIR="${OUTPUT_BASE}/${EXP_NAME}"
    local LOG_FILE="${OUTPUT_BASE}/logs/${EXP_NAME}.log"

    # Skip if already done
    if [ -f "${OUTPUT_DIR}/generation_stats.json" ]; then
        if [ "$DATASET" = "ringabell" ]; then
            if [ -f "${OUTPUT_DIR}/categories_qwen3_vl_nudity.json" ]; then
                echo "[GPU ${ACTUAL_GPU}] SKIP (eval done): ${EXP_NAME}"
                return 0
            fi
        else
            echo "[GPU ${ACTUAL_GPU}] SKIP (gen done): ${EXP_NAME}"
            return 0
        fi
    fi

    echo "[GPU ${ACTUAL_GPU}] START: ${EXP_NAME}"

    # Build extra args for attention mode
    local EXTRA_ARGS=""
    if [ "$SMODE" = "attention" ] || [ "$SMODE" = "attention_gradcam" ]; then
        EXTRA_ARGS="--harmful_keywords ${HARMFUL_KW} --attn_resolutions 16 32"
    fi

    local END_IDX_ARG=""
    if [ "$END_IDX" != "-1" ]; then
        END_IDX_ARG="--end_idx ${END_IDX}"
    fi

    # Step 1: Generate
    if [ ! -f "${OUTPUT_DIR}/generation_stats.json" ]; then
        CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python "${GEN_SCRIPT}" \
            --ckpt_path "${SD_CKPT}" \
            --prompt_file "${PROMPT_FILE}" \
            --output_dir "${OUTPUT_DIR}" \
            --classifier_ckpt "${CLASSIFIER_CKPT}" \
            --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
            --guidance_start_t ${ST} \
            --guidance_end_t ${ET} \
            --guidance_schedule "${SCHED}" \
            --spatial_mode "${SMODE}" \
            --guidance_scale ${GS} \
            --base_guidance_scale ${BS} \
            --spatial_threshold_start ${SP_S} \
            --spatial_threshold_end ${SP_E} \
            --spatial_threshold_strategy "cosine" \
            --num_inference_steps ${NUM_STEPS} \
            --cfg_scale ${CFG_SCALE} \
            --seed ${SEED} \
            --nsamples ${NSAMPLES} \
            ${END_IDX_ARG} \
            ${EXTRA_ARGS} \
            >> "${LOG_FILE}" 2>&1

        if [ $? -ne 0 ]; then
            echo "[GPU ${ACTUAL_GPU}] FAILED generation: ${EXP_NAME}"
            return 1
        fi
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

TOTAL_SLOTS=$((NUM_GPUS * 1))
declare -A RUNNING_JOBS
LOCAL_IDX=0

for (( ci=START; ci<END; ci++ )); do
    combo="${ALL_COMBOS[$ci]}"
    IFS='|' read -r SMODE ST ET SCHED GS BS SP_S SP_E <<< "$combo"

    # Find available slot
    while true; do
        for ((slot=0; slot<TOTAL_SLOTS; slot++)); do
            JOB_PID="${RUNNING_JOBS[$slot]}"
            if [ -z "$JOB_PID" ] || ! kill -0 "$JOB_PID" 2>/dev/null; then
                GPU_IDX=$((slot % NUM_GPUS))
                run_experiment $GPU_IDX "$SMODE" "$ST" "$ET" "$SCHED" "$GS" "$BS" "$SP_S" "$SP_E" &
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
echo "SGF WINDOW GRID SEARCH COMPLETE"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Results: ${OUTPUT_BASE}/"

COMPLETED=0
for (( ci=0; ci<TOTAL; ci++ )); do
    combo="${ALL_COMBOS[$ci]}"
    IFS='|' read -r SMODE ST ET SCHED GS BS SP_S SP_E <<< "$combo"
    EXP_NAME="${SMODE}_w${ST}-${ET}_${SCHED}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}"
    if [ -f "${OUTPUT_BASE}/${EXP_NAME}/generation_stats.json" ]; then
        COMPLETED=$((COMPLETED + 1))
    fi
done
echo "Completed: ${COMPLETED}/${TOTAL}"
echo "Done!"
