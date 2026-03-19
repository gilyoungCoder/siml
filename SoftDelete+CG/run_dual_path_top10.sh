#!/bin/bash
# ============================================================================
# Run top 10 SCG configs with dual_path monitoring
# (two parallel latent paths: guided + unguided, z0 decides at decision step)
#
# Based on aggregate results (SR-sorted top 10 from unified V1+V2).
# Each config is run with monitoring_mode=dual_path.
#
# Sweep:
#   DECISION_STEPS: (5, 6, 7) — z0 separation starts around step 5
#   Z0_THRESHOLDS: (0.005, 0.01, 0.02) — from score analysis
#   CDF threshold: 0.05 (fixed, best from previous experiments)
#   = 9 combos per config = 90 experiments total
#
# Site splitting (90 total, 5 servers × 8 GPUs):
#   A: 0-21  (22 experiments) — fast server
#   B: 22-43 (22 experiments) — fast server
#   C: 44-59 (16 experiments)
#   D: 60-75 (16 experiments)
#   E: 76-89 (14 experiments)
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
    LOG_FILE="${LOG_DIR}/dual_path_top10${SITE_TAG}_${TIMESTAMP}.log"
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
OUTPUT_BASE="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/dual_path_top10"

# ============================================
# Top 10 configs (from unified V1+V2 aggregate, SR-sorted)
# Format: SPATIAL_MODE|GS|BS|SP_START|SP_END
# ============================================
declare -a TOP_CONFIGS=(
    # 1. gradcam gs17.5 bs1.0 sp0.2-0.4 (SR 87.3%)
    "gradcam|17.5|1.0|0.2|0.4"
    # 2. gradcam gs15 bs1.0 sp0.1-0.4 (SR 83.5%)
    "gradcam|15.0|1.0|0.1|0.4"
    # 3. gradcam gs17.5 bs1.0 sp0.1-0.3 (SR 83.5%)
    "gradcam|17.5|1.0|0.1|0.3"
    # 4. gradcam gs12.5 bs1.0 sp0.2-0.4 (SR 82.3%)
    "gradcam|12.5|1.0|0.2|0.4"
    # 5. gradcam gs12.5 bs1.0 sp0.1-0.4 (SR 81.0%)
    "gradcam|12.5|1.0|0.1|0.4"
    # 6. gradcam gs15 bs1.0 sp0.2-0.4 (SR 79.7%)
    "gradcam|15.0|1.0|0.2|0.4"
    # 7. gradcam gs17.5 bs1.0 sp0.1-0.4 (SR 79.7%)
    "gradcam|17.5|1.0|0.1|0.4"
    # 8. gradcam gs15 bs1.0 sp0.1-0.3 (SR 79.7%)
    "gradcam|15.0|1.0|0.1|0.3"
    # 9. gradcam gs17.5 bs1.0 sp0.2-0.3 (SR 78.5%)
    "gradcam|17.5|1.0|0.2|0.3"
    # 10. gradcam gs12.5 bs1.0 sp0.1-0.3 (SR 77.2%)
    "gradcam|12.5|1.0|0.1|0.3"
)

# Dual-path sweep: decision_step × z0_threshold
# From score analysis: z0 starts separating at step 5
declare -a DECISION_STEPS=(5 6 7)
declare -a Z0_THRESHOLDS=(0.005 0.01 0.02)
CDF_THRESHOLD=0.05

# ============================================
# BUILD COMBINATIONS
# 10 configs × 3 decision_steps × 3 z0_thresholds = 90 experiments
# ============================================
declare -a ALL_COMBOS=()

for config in "${TOP_CONFIGS[@]}"; do
    IFS='|' read -r SMODE GS BS SP_S SP_E <<< "$config"
    for DS in "${DECISION_STEPS[@]}"; do
        for Z0T in "${Z0_THRESHOLDS[@]}"; do
            ALL_COMBOS+=("${SMODE}|${GS}|${BS}|${SP_S}|${SP_E}|${DS}|${Z0T}")
        done
    done
done

TOTAL=${#ALL_COMBOS[@]}

# Site splitting: 90 total → 5 servers (A,B fast / C,D,E slower)
# Fast servers (A,B): 22+22=44, Slow servers (C,D,E): 16+16+14=46
if [ "$SITE" = "A" ]; then
    START=0; END=22
elif [ "$SITE" = "B" ]; then
    START=22; END=44
elif [ "$SITE" = "C" ]; then
    START=44; END=60
elif [ "$SITE" = "D" ]; then
    START=60; END=76
elif [ "$SITE" = "E" ]; then
    START=76; END=$TOTAL
else
    START=0; END=$TOTAL
fi
SITE_COUNT=$((END - START))

echo "=============================================="
echo "DUAL-PATH MONITORING TOP 10 CONFIGS"
echo "=============================================="
echo "Total experiments: $TOTAL (site ${SITE:-ALL}: $START to $((END-1)), running $SITE_COUNT)"
echo "Configs: 10 base × 3 decision_steps × 3 z0_thresholds = 90"
echo "Split: A=22(fast), B=22(fast), C=16, D=16, E=14"
echo "GPUs: $NUM_GPUS"
echo "=============================================="

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY RUN] $SITE_COUNT experiments (site ${SITE:-ALL}):"
    for (( ci=START; ci<END; ci++ )); do
        combo="${ALL_COMBOS[$ci]}"
        IFS='|' read -r SMODE GS BS SP_S SP_E DS Z0T <<< "$combo"
        echo "  [$ci] dp_${SMODE}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}_ds${DS}_z0t${Z0T}"
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
    local DS=$7
    local Z0T=$8

    local ACTUAL_GPU=${GPU_LIST[$GPU_IDX]}
    local EXP_NAME="dp_${SMODE}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}_ds${DS}_z0t${Z0T}"
    local OUTPUT_DIR="${OUTPUT_BASE}/ringabell/${EXP_NAME}"
    local LOG_FILE="${OUTPUT_BASE}/logs/${EXP_NAME}.log"

    # Skip if already evaluated
    if [ -f "${OUTPUT_DIR}/categories_qwen3_vl_nudity.json" ]; then
        echo "[GPU ${ACTUAL_GPU}] SKIP (eval done): ${EXP_NAME}"
        return 0
    fi

    echo "[GPU ${ACTUAL_GPU}] START: ${EXP_NAME}"

    # Step 1: Generate
    if [ ! -f "${OUTPUT_DIR}/generation_stats.json" ]; then
        CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python "${GEN_SCRIPT}" \
            --ckpt_path "${SD_CKPT}" \
            --prompt_file "${PROMPT_FILE}" \
            --output_dir "${OUTPUT_DIR}" \
            --classifier_ckpt "${CLASSIFIER_CKPT}" \
            --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
            --monitoring_mode dual_path \
            --z0_decision_step ${DS} \
            --monitoring_threshold ${Z0T} \
            --cdf_threshold ${CDF_THRESHOLD} \
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
    IFS='|' read -r SMODE GS BS SP_S SP_E DS Z0T <<< "$combo"

    while true; do
        for ((slot=0; slot<TOTAL_SLOTS; slot++)); do
            JOB_PID="${RUNNING_JOBS[$slot]}"
            if [ -z "$JOB_PID" ] || ! kill -0 "$JOB_PID" 2>/dev/null; then
                GPU_IDX=$((slot % NUM_GPUS))
                run_experiment $GPU_IDX "$SMODE" "$GS" "$BS" "$SP_S" "$SP_E" "$DS" "$Z0T" &
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
echo "DUAL-PATH TOP 10 COMPLETE"
echo "=============================================="
echo "Results: ${OUTPUT_BASE}/"

COMPLETED=0
for (( ci=0; ci<TOTAL; ci++ )); do
    combo="${ALL_COMBOS[$ci]}"
    IFS='|' read -r SMODE GS BS SP_S SP_E DS Z0T <<< "$combo"
    EXP_NAME="dp_${SMODE}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}_ds${DS}_z0t${Z0T}"
    if [ -f "${OUTPUT_BASE}/ringabell/${EXP_NAME}/categories_qwen3_vl_nudity.json" ]; then
        COMPLETED=$((COMPLETED + 1))
    fi
done
echo "Completed: ${COMPLETED}/${TOTAL}"
echo "Done!"
