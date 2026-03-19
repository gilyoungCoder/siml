#!/bin/bash
# ============================================================================
# Unified Grid Search V2: Non-overlapping with V1 (332 experiments)
#
# V1 covered: gs=[12.5,15,17.5], bs=1.0, start_step=0, datasets=[ringabell,coco]
#
# V2 explores (non-overlapping):
#   - GS: [10.0, 20.0]              (V1 had 12.5, 15, 17.5)
#   - BS: [0.0, 2.0, 3.0]           (V1 had 1.0)
#   - monitoring_start_step: [0, 10] (V1 had 0 only)
#   - Datasets: ringabell only
#
# Total: 360 experiments
# ============================================================================

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd

# ============================================
# Parse Arguments
# ============================================
DRY_RUN=true
USE_NOHUP=false
NUM_GPUS=8
SITE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --run) DRY_RUN=false; shift;;
        --dry-run) DRY_RUN=true; shift;;
        --nohup) USE_NOHUP=true; shift;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        --site) SITE="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

# Nohup handling
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="./scg_outputs"
    mkdir -p "$LOG_DIR"
    SITE_TAG=""
    if [ -n "$SITE" ]; then SITE_TAG="_site${SITE}"; fi
    LOG_FILE="${LOG_DIR}/unified_v2${SITE_TAG}_${TIMESTAMP}.log"
    echo "Running in background..."
    echo "Log: $LOG_FILE"
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    SITE_ARG=""
    if [ -n "$SITE" ]; then SITE_ARG="--site $SITE"; fi
    nohup bash "$SCRIPT_PATH" --run --num-gpus "$NUM_GPUS" $SITE_ARG > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    echo "Monitor: tail -f $LOG_FILE"
    exit 0
fi

# ============================================
# CONFIGURATION
# ============================================

CLASSIFIER_CKPT="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/nudity_4class"
SD_CKPT="CompVis/stable-diffusion-v1-4"
GEN_SCRIPT="generate_unified_monitoring.py"

PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
END_IDX=-1

# Fixed params
NUM_STEPS=50
SEED=42
NSAMPLES=1
CFG_SCALE=7.5
THRESHOLD_STRATEGY="cosine"

HARMFUL_KW="nude naked breast topless bare undress nsfw"

# VLM eval
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
OUTPUT_BASE="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/unified_grid_v2"

# ============================================
# Variable parameters
# ============================================
GUIDANCE_SCALES=(10.0 20.0)
BASE_SCALES=(0.0 2.0 3.0)
MON_STARTS=(0 10)
SP_PAIRS_GC=("0.2 0.4" "0.3 0.5")
SP_PAIRS_ATTN=("0.2 0.3" "0.3 0.5")

# ============================================
# BUILD COMBINATIONS
# Format: MON_MODE|SP_MODE|MON_THR|GS|BS|SP_START|SP_END|MON_START
# ============================================

declare -a ALL_COMBOS=()

# --- Group 1: gradcam monitoring + gradcam spatial ---
for ms in "${MON_STARTS[@]}"; do
    for mt in 0.05 0.1 0.15; do
        for gs in "${GUIDANCE_SCALES[@]}"; do
            for bs in "${BASE_SCALES[@]}"; do
                for sp in "${SP_PAIRS_GC[@]}"; do
                    SP_S=$(echo $sp | awk '{print $1}')
                    SP_E=$(echo $sp | awk '{print $2}')
                    ALL_COMBOS+=("gradcam|gradcam|${mt}|${gs}|${bs}|${SP_S}|${SP_E}|${ms}")
                done
            done
        done
    done
done

# --- Group 2: z0_sticky monitoring + gradcam spatial ---
for ms in "${MON_STARTS[@]}"; do
    for mt in 0.01 0.03 0.06; do
        for gs in "${GUIDANCE_SCALES[@]}"; do
            for bs in "${BASE_SCALES[@]}"; do
                for sp in "${SP_PAIRS_GC[@]}"; do
                    SP_S=$(echo $sp | awk '{print $1}')
                    SP_E=$(echo $sp | awk '{print $2}')
                    ALL_COMBOS+=("z0_sticky|gradcam|${mt}|${gs}|${bs}|${SP_S}|${SP_E}|${ms}")
                done
            done
        done
    done
done

# --- Group 3: z0 monitoring + gradcam spatial ---
for ms in "${MON_STARTS[@]}"; do
    for mt in 0.01 0.03 0.06; do
        for gs in "${GUIDANCE_SCALES[@]}"; do
            for bs in "${BASE_SCALES[@]}"; do
                for sp in "${SP_PAIRS_GC[@]}"; do
                    SP_S=$(echo $sp | awk '{print $1}')
                    SP_E=$(echo $sp | awk '{print $2}')
                    ALL_COMBOS+=("z0|gradcam|${mt}|${gs}|${bs}|${SP_S}|${SP_E}|${ms}")
                done
            done
        done
    done
done

# --- Group 4: gradcam monitoring + attention spatial ---
for ms in "${MON_STARTS[@]}"; do
    for mt in 0.05 0.1 0.15; do
        for gs in "${GUIDANCE_SCALES[@]}"; do
            for bs in "${BASE_SCALES[@]}"; do
                for sp in "${SP_PAIRS_ATTN[@]}"; do
                    SP_S=$(echo $sp | awk '{print $1}')
                    SP_E=$(echo $sp | awk '{print $2}')
                    ALL_COMBOS+=("gradcam|attention|${mt}|${gs}|${bs}|${SP_S}|${SP_E}|${ms}")
                done
            done
        done
    done
done

# --- Group 5: z0_sticky monitoring + attention spatial ---
for ms in "${MON_STARTS[@]}"; do
    for mt in 0.01 0.03 0.06; do
        for gs in "${GUIDANCE_SCALES[@]}"; do
            for bs in "${BASE_SCALES[@]}"; do
                for sp in "${SP_PAIRS_ATTN[@]}"; do
                    SP_S=$(echo $sp | awk '{print $1}')
                    SP_E=$(echo $sp | awk '{print $2}')
                    ALL_COMBOS+=("z0_sticky|attention|${mt}|${gs}|${bs}|${SP_S}|${SP_E}|${ms}")
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
echo "UNIFIED GRID SEARCH V2 (SoftDelete+CG)"
echo "  Non-overlapping with V1"
echo "=============================================="
echo "Total configs: $TOTAL (site ${SITE:-ALL}: $START to $((END-1)), running $SITE_COUNT)"
echo "New: GS=[10,20] BS=[0,2,3] start_step=[0,10]"
echo "Monitoring: gradcam, z0, z0_sticky"
echo "Spatial: gradcam, attention"
echo "GPUs: $NUM_GPUS"
echo "=============================================="

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY RUN] $SITE_COUNT experiments (site ${SITE:-ALL}):"
    for (( ci=START; ci<END; ci++ )); do
        combo="${ALL_COMBOS[$ci]}"
        IFS='|' read -r MMODE SMODE MT GS BS SP_S SP_E MS <<< "$combo"
        echo "  [$ci] ${MMODE}_${SMODE}_mt${MT}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}_ms${MS}"
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
    local MMODE=$2
    local SMODE=$3
    local MT=$4
    local GS=$5
    local BS=$6
    local SP_S=$7
    local SP_E=$8
    local MS=$9

    local ACTUAL_GPU=${GPU_LIST[$GPU_IDX]}
    local EXP_NAME="${MMODE}_${SMODE}_mt${MT}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}_ms${MS}"
    local OUTPUT_DIR="${OUTPUT_BASE}/ringabell/${EXP_NAME}"
    local LOG_FILE="${OUTPUT_BASE}/logs/${EXP_NAME}.log"

    # Skip if already evaluated
    if [ -f "${OUTPUT_DIR}/categories_qwen3_vl_nudity.json" ]; then
        echo "[GPU ${ACTUAL_GPU}] SKIP (eval done): ${EXP_NAME}"
        return 0
    fi

    echo "[GPU ${ACTUAL_GPU}] START: ${EXP_NAME}"

    # Build extra args
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
            --monitoring_mode "${MMODE}" \
            --monitoring_threshold ${MT} \
            --monitoring_start_step ${MS} \
            --spatial_mode "${SMODE}" \
            --guidance_scale ${GS} \
            --base_guidance_scale ${BS} \
            --spatial_threshold_start ${SP_S} \
            --spatial_threshold_end ${SP_E} \
            --spatial_threshold_strategy "${THRESHOLD_STRATEGY}" \
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
    else
        echo "[GPU ${ACTUAL_GPU}] SKIP generation (done): ${EXP_NAME}"
    fi

    # Step 2: VLM eval
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate vlm
    CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python "${VLM_SCRIPT}" "${OUTPUT_DIR}" nudity qwen \
        >> "${LOG_FILE}" 2>&1
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd

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
    IFS='|' read -r MMODE SMODE MT GS BS SP_S SP_E MS <<< "$combo"

    # Find available slot
    while true; do
        for ((slot=0; slot<TOTAL_SLOTS; slot++)); do
            JOB_PID="${RUNNING_JOBS[$slot]}"
            if [ -z "$JOB_PID" ] || ! kill -0 "$JOB_PID" 2>/dev/null; then
                GPU_IDX=$((slot % NUM_GPUS))
                run_experiment $GPU_IDX "$MMODE" "$SMODE" "$MT" "$GS" "$BS" "$SP_S" "$SP_E" "$MS" &
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
echo "UNIFIED GRID SEARCH V2 COMPLETE"
echo "=============================================="
echo "Results: ${OUTPUT_BASE}/"

COMPLETED=0
for (( ci=0; ci<TOTAL; ci++ )); do
    combo="${ALL_COMBOS[$ci]}"
    IFS='|' read -r MMODE SMODE MT GS BS SP_S SP_E MS <<< "$combo"
    EXP_NAME="${MMODE}_${SMODE}_mt${MT}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}_ms${MS}"
    if [ -f "${OUTPUT_BASE}/ringabell/${EXP_NAME}/generation_stats.json" ]; then
        COMPLETED=$((COMPLETED + 1))
    fi
done
echo "Completed: ${COMPLETED}/${TOTAL}"
echo "Done!"
