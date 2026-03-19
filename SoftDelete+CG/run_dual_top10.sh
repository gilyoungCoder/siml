#!/bin/bash
# ============================================================================
# Run top 10 SCG configs with DUAL monitoring (z0_sticky AND + GradCAM CDF)
#
# Based on aggregate results (SR-sorted top 10 from unified V1+V2).
# Each config is re-run with monitoring_mode=dual.
#
# z0 thresholds to try: 0.03, 0.06
# GradCAM CDF thresholds to try: 0.03, 0.05, 0.1
# = 6 dual combos per original config = 60 experiments total
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
    LOG_FILE="${LOG_DIR}/dual_top10${SITE_TAG}_${TIMESTAMP}.log"
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

NUM_STEPS=50
SEED=42
NSAMPLES=1
CFG_SCALE=7.5
THRESHOLD_STRATEGY="cosine"

HARMFUL_KW="nude naked breast topless bare undress nsfw"
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
OUTPUT_BASE="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/dual_top10"

# ============================================
# Top 10 configs (from aggregate, SR-sorted)
# Format: SPATIAL_MODE|GS|BS|SP_START|SP_END|MON_START
# (monitoring_mode will be dual for all)
# ============================================
declare -a TOP_CONFIGS=(
    # 1. gradcam_gradcam mon0.05 gs17.5 bs1.0 sp0.2-0.4 (SR 87.3%)
    "gradcam|17.5|1.0|0.2|0.4|0"
    # 2. z0_sticky_gradcam mon0.03 gs15 bs1.0 sp0.1-0.3 (SR 84.8%)
    "gradcam|15.0|1.0|0.1|0.3|0"
    # 3. gradcam_gradcam mt0.05 gs10 bs2 sp0.3-0.5 ms0 (SR 84.8%)
    "gradcam|10.0|2.0|0.3|0.5|0"
    # 4. gradcam_gradcam mon0.05 gs15 bs1.0 sp0.1-0.4 (SR 83.5%)
    "gradcam|15.0|1.0|0.1|0.4|0"
    # 5. gradcam_gradcam mon0.1 gs17.5 bs1.0 sp0.1-0.3 (SR 83.5%)
    "gradcam|17.5|1.0|0.1|0.3|0"
    # 6. gradcam_gradcam mon0.05 gs12.5 bs1.0 sp0.2-0.4 (SR 82.3%)
    "gradcam|12.5|1.0|0.2|0.4|0"
    # 7. z0_sticky_gradcam mon0.01 gs12.5 bs1.0 sp0.1-0.3 (SR 82.3%)
    "gradcam|12.5|1.0|0.1|0.3|0"
    # 8. gradcam_gradcam mon0.05 gs12.5 bs1.0 sp0.1-0.4 (SR 81.0%)
    "gradcam|12.5|1.0|0.1|0.4|0"
    # 9. attention top1: gs10 bs0 sp0.2-0.3 ms10 (SR 81.0%)
    "attention|10.0|0.0|0.2|0.3|10"
    # 10. attention top2: gs10 bs3 sp0.3-0.5 ms10 (SR 81.0%)
    "attention|10.0|3.0|0.3|0.5|10"
)

# Dual monitoring thresholds to try
Z0_THRESHOLDS=(0.03 0.06)
CDF_THRESHOLDS=(0.03 0.05 0.1)

# ============================================
# BUILD COMBINATIONS
# ============================================
declare -a ALL_COMBOS=()

for config in "${TOP_CONFIGS[@]}"; do
    IFS='|' read -r SMODE GS BS SP_S SP_E MS <<< "$config"
    for z0_thr in "${Z0_THRESHOLDS[@]}"; do
        for cdf_thr in "${CDF_THRESHOLDS[@]}"; do
            ALL_COMBOS+=("${SMODE}|${GS}|${BS}|${SP_S}|${SP_E}|${MS}|${z0_thr}|${cdf_thr}")
        done
    done
done

TOTAL=${#ALL_COMBOS[@]}

# Site splitting: A=24(40%), B=24(40%), C=12(20%) — C gets 1.5x of A,B
# Actually: 60 total. A=17, B=17, C=26 (C ≈ 1.5× A/B)
if [ "$SITE" = "A" ]; then
    START=0; END=17
elif [ "$SITE" = "B" ]; then
    START=17; END=34
elif [ "$SITE" = "C" ]; then
    START=34; END=$TOTAL
else
    START=0; END=$TOTAL
fi
SITE_COUNT=$((END - START))

echo "=============================================="
echo "DUAL MONITORING TOP 10 CONFIGS"
echo "=============================================="
echo "Total experiments: $TOTAL (site ${SITE:-ALL}: $START to $((END-1)), running $SITE_COUNT)"
echo "Configs: 10 base × 2 z0_thr × 3 cdf_thr = 60"
echo "Split: A=17, B=17, C=26 (C ≈ 1.5× A/B)"
echo "GPUs: $NUM_GPUS"
echo "=============================================="

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY RUN] $SITE_COUNT experiments (site ${SITE:-ALL}):"
    for (( ci=START; ci<END; ci++ )); do
        combo="${ALL_COMBOS[$ci]}"
        IFS='|' read -r SMODE GS BS SP_S SP_E MS Z0T CDFT <<< "$combo"
        echo "  [$ci] dual_${SMODE}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}_ms${MS}_z0t${Z0T}_cdf${CDFT}"
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
    local GS=$3
    local BS=$4
    local SP_S=$5
    local SP_E=$6
    local MS=$7
    local Z0T=$8
    local CDFT=$9

    local ACTUAL_GPU=${GPU_LIST[$GPU_IDX]}
    local EXP_NAME="dual_${SMODE}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}_ms${MS}_z0t${Z0T}_cdf${CDFT}"
    local OUTPUT_DIR="${OUTPUT_BASE}/ringabell/${EXP_NAME}"
    local LOG_FILE="${OUTPUT_BASE}/logs/${EXP_NAME}.log"

    # Skip if already evaluated
    if [ -f "${OUTPUT_DIR}/categories_qwen3_vl_nudity.json" ]; then
        echo "[GPU ${ACTUAL_GPU}] SKIP (eval done): ${EXP_NAME}"
        return 0
    fi

    echo "[GPU ${ACTUAL_GPU}] START: ${EXP_NAME}"

    # Build extra args for attention mode
    local EXTRA_ARGS=""
    if [ "$SMODE" = "attention" ] || [ "$SMODE" = "attention_gradcam" ]; then
        EXTRA_ARGS="--harmful_keywords ${HARMFUL_KW} --attn_resolutions 16 32"
    fi

    # Step 1: Generate
    if [ ! -f "${OUTPUT_DIR}/generation_stats.json" ]; then
        CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python "${GEN_SCRIPT}" \
            --ckpt_path "${SD_CKPT}" \
            --prompt_file "${PROMPT_FILE}" \
            --output_dir "${OUTPUT_DIR}" \
            --classifier_ckpt "${CLASSIFIER_CKPT}" \
            --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
            --monitoring_mode dual \
            --monitoring_threshold ${Z0T} \
            --cdf_threshold ${CDFT} \
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
    IFS='|' read -r SMODE GS BS SP_S SP_E MS Z0T CDFT <<< "$combo"

    while true; do
        for ((slot=0; slot<TOTAL_SLOTS; slot++)); do
            JOB_PID="${RUNNING_JOBS[$slot]}"
            if [ -z "$JOB_PID" ] || ! kill -0 "$JOB_PID" 2>/dev/null; then
                GPU_IDX=$((slot % NUM_GPUS))
                run_experiment $GPU_IDX "$SMODE" "$GS" "$BS" "$SP_S" "$SP_E" "$MS" "$Z0T" "$CDFT" &
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
echo "DUAL MONITORING TOP 10 COMPLETE"
echo "=============================================="
echo "Results: ${OUTPUT_BASE}/"

COMPLETED=0
for (( ci=0; ci<TOTAL; ci++ )); do
    combo="${ALL_COMBOS[$ci]}"
    IFS='|' read -r SMODE GS BS SP_S SP_E MS Z0T CDFT <<< "$combo"
    EXP_NAME="dual_${SMODE}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}_ms${MS}_z0t${Z0T}_cdf${CDFT}"
    if [ -f "${OUTPUT_BASE}/ringabell/${EXP_NAME}/categories_qwen3_vl_nudity.json" ]; then
        COMPLETED=$((COMPLETED + 1))
    fi
done
echo "Completed: ${COMPLETED}/${TOTAL}"
echo "Done!"
