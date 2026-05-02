#!/bin/bash
# ============================================================================
# Run top 10 SCG configs with z0_trigger_cdf monitoring
# (single-step z0 decision + GradCAM CDF per-step guidance)
#
# Based on aggregate results (SR-sorted top 10 from unified V1+V2).
# Each config is re-run with monitoring_mode=z0_trigger_cdf.
#
# Sweep:
#   (decision_step, z0_threshold) pairs: 6 combos from score analysis
#   CDF threshold: 0.05 (fixed, best from previous experiments)
#   = 6 combos per config = 60 experiments total
#
# Site splitting (60 total, 5 servers × 8 GPUs):
#   A: 0-15  (16 experiments) — fast server
#   B: 16-31 (16 experiments) — fast server
#   C: 32-41 (10 experiments)
#   D: 42-51 (10 experiments)
#   E: 52-59 (8 experiments)
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
    LOG_FILE="${LOG_DIR}/z0_trigger_cdf_top10${SITE_TAG}_${TIMESTAMP}.log"
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
OUTPUT_BASE="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/z0_trigger_cdf_top10"

# ============================================
# Top 10 configs (from aggregate, SR-sorted)
# Format: SPATIAL_MODE|GS|BS|SP_START|SP_END
# ============================================
declare -a TOP_CONFIGS=(
    # 1. gradcam gs17.5 bs1.0 sp0.2-0.4 (SR 87.3%)
    "gradcam|17.5|1.0|0.2|0.4"
    # 2. gradcam gs15 bs1.0 sp0.1-0.3 (SR 84.8%)
    "gradcam|15.0|1.0|0.1|0.3"
    # 3. gradcam gs10 bs2 sp0.3-0.5 (SR 84.8%)
    "gradcam|10.0|2.0|0.3|0.5"
    # 4. gradcam gs15 bs1.0 sp0.1-0.4 (SR 83.5%)
    "gradcam|15.0|1.0|0.1|0.4"
    # 5. gradcam gs17.5 bs1.0 sp0.1-0.3 (SR 83.5%)
    "gradcam|17.5|1.0|0.1|0.3"
    # 6. gradcam gs12.5 bs1.0 sp0.2-0.4 (SR 82.3%)
    "gradcam|12.5|1.0|0.2|0.4"
    # 7. gradcam gs12.5 bs1.0 sp0.1-0.3 (SR 82.3%)
    "gradcam|12.5|1.0|0.1|0.3"
    # 8. gradcam gs12.5 bs1.0 sp0.1-0.4 (SR 81.0%)
    "gradcam|12.5|1.0|0.1|0.4"
    # 9. attention gs10 bs0 sp0.2-0.3 (SR 81.0%)
    "attention|10.0|0.0|0.2|0.3"
    # 10. attention gs10 bs3 sp0.3-0.5 (SR 81.0%)
    "attention|10.0|3.0|0.3|0.5"
)

# z0_trigger_cdf sweep: (decision_step, z0_threshold) pairs from score analysis
# CDF threshold fixed at 0.05 (best from previous experiments)
# step=13, z0>0.005: 86.4% Full, 10% FP
# step=14, z0>0.005: 84.7% Full, 6% FP
# step=12, z0>0.005: 84.7% Full, 8% FP
# step=12, z0>0.01:  83.1% Full, 6% FP
# step=13, z0>0.01:  81.4% Full, 4% FP
# step=7,  z0>0.005: 83.1% Full, 8% FP
declare -a Z0_PAIRS=(
    "13|0.005"
    "14|0.005"
    "12|0.005"
    "12|0.01"
    "13|0.01"
    "7|0.005"
)
CDF_THRESHOLD=0.05

# ============================================
# BUILD COMBINATIONS
# 10 configs × 6 z0_pairs × 1 cdf_thr = 60 experiments
# ============================================
declare -a ALL_COMBOS=()

for config in "${TOP_CONFIGS[@]}"; do
    IFS='|' read -r SMODE GS BS SP_S SP_E <<< "$config"
    for z0_pair in "${Z0_PAIRS[@]}"; do
        IFS='|' read -r DS Z0T <<< "$z0_pair"
        ALL_COMBOS+=("${SMODE}|${GS}|${BS}|${SP_S}|${SP_E}|${DS}|${Z0T}|${CDF_THRESHOLD}")
    done
done

TOTAL=${#ALL_COMBOS[@]}

# Site splitting: 60 total → 5 servers (A,B fast / C,D,E slower)
# Fast servers (A,B): 16+16=32, Slow servers (C,D,E): 10+10+8=28
if [ "$SITE" = "A" ]; then
    START=0; END=16
elif [ "$SITE" = "B" ]; then
    START=16; END=32
elif [ "$SITE" = "C" ]; then
    START=32; END=42
elif [ "$SITE" = "D" ]; then
    START=42; END=52
elif [ "$SITE" = "E" ]; then
    START=52; END=$TOTAL
else
    START=0; END=$TOTAL
fi
SITE_COUNT=$((END - START))

echo "=============================================="
echo "z0_trigger_cdf MONITORING TOP 10 CONFIGS"
echo "=============================================="
echo "Total experiments: $TOTAL (site ${SITE:-ALL}: $START to $((END-1)), running $SITE_COUNT)"
echo "Configs: 10 base × 6 z0_pairs × 1 cdf_thr = 60"
echo "Split: A=16(fast), B=16(fast), C=10, D=10, E=8"
echo "GPUs: $NUM_GPUS"
echo "=============================================="

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY RUN] $SITE_COUNT experiments (site ${SITE:-ALL}):"
    for (( ci=START; ci<END; ci++ )); do
        combo="${ALL_COMBOS[$ci]}"
        IFS='|' read -r SMODE GS BS SP_S SP_E DS Z0T CDFT <<< "$combo"
        echo "  [$ci] z0tc_${SMODE}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}_ds${DS}_z0t${Z0T}_cdf${CDFT}"
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
    local CDFT=$9

    local ACTUAL_GPU=${GPU_LIST[$GPU_IDX]}
    local EXP_NAME="z0tc_${SMODE}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}_ds${DS}_z0t${Z0T}_cdf${CDFT}"
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
            --monitoring_mode z0_trigger_cdf \
            --z0_decision_step ${DS} \
            --monitoring_threshold ${Z0T} \
            --cdf_threshold ${CDFT} \
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
    IFS='|' read -r SMODE GS BS SP_S SP_E DS Z0T CDFT <<< "$combo"

    while true; do
        for ((slot=0; slot<TOTAL_SLOTS; slot++)); do
            JOB_PID="${RUNNING_JOBS[$slot]}"
            if [ -z "$JOB_PID" ] || ! kill -0 "$JOB_PID" 2>/dev/null; then
                GPU_IDX=$((slot % NUM_GPUS))
                run_experiment $GPU_IDX "$SMODE" "$GS" "$BS" "$SP_S" "$SP_E" "$DS" "$Z0T" "$CDFT" &
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
echo "z0_SINGLE TOP 10 COMPLETE"
echo "=============================================="
echo "Results: ${OUTPUT_BASE}/"

COMPLETED=0
for (( ci=0; ci<TOTAL; ci++ )); do
    combo="${ALL_COMBOS[$ci]}"
    IFS='|' read -r SMODE GS BS SP_S SP_E DS Z0T CDFT <<< "$combo"
    EXP_NAME="z0tc_${SMODE}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}_ds${DS}_z0t${Z0T}_cdf${CDFT}"
    if [ -f "${OUTPUT_BASE}/ringabell/${EXP_NAME}/categories_qwen3_vl_nudity.json" ]; then
        COMPLETED=$((COMPLETED + 1))
    fi
done
echo "Completed: ${COMPLETED}/${TOTAL}"
echo "Done!"
