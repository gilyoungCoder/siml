#!/bin/bash
# ============================================================================
# Run top-7 SAFREE+Monitoring configs on all 5 datasets
#
# Usage:
#   # Site A (GPUs 0-7):
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_best7_all_datasets.sh --site A
#
#   # Site B (GPUs 0-7):
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_best7_all_datasets.sh --site B
#
#   # Dry run:
#   bash run_best7_all_datasets.sh --dry-run
#
#   # Nohup:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_best7_all_datasets.sh --site A --nohup
# ============================================================================

set -e

SITE=""
DRY_RUN=false
USE_NOHUP=false
DATASETS_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --site) SITE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --nohup) USE_NOHUP=true; shift ;;
        --datasets) DATASETS_ARG="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ "$DRY_RUN" = false ] && [ -z "$SITE" ]; then
    echo "ERROR: Must specify --site A or --site B"
    echo "  Site A runs configs 1-4 (18 experiments on 8 GPUs)"
    echo "  Site B runs configs 5-7 (17 experiments on 8 GPUs)"
    exit 1
fi

if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="./results/best7_site${SITE}_${TIMESTAMP}.log"
    mkdir -p ./results
    echo "Running in background (site ${SITE})..."
    echo "Log: ${LOG_FILE}"
    EXTRA_ARGS=""
    [ -n "$DATASETS_ARG" ] && EXTRA_ARGS="--datasets $DATASETS_ARG"
    nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$0" --site "$SITE" $EXTRA_ARGS > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

cd /mnt/home/yhgil99/unlearning/SAFREE

# ============================================
# FIXED CONFIG
# ============================================
CLASSIFIER_CKPT="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/nudity_4class"
SD_CKPT="CompVis/stable-diffusion-v1-4"
OUTPUT_BASE="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs"

NUM_STEPS=50
SEED=42
NSAMPLES=1
CFG_SCALE=7.5
SAFREE_ALPHA=0.01
SVF_UP_T=10
CATEGORY="nudity"
THRESHOLD_STRATEGY="cosine"
HARMFUL_SCALE=1.0

# ============================================
# DATASETS
# ============================================
declare -A DATASET_PROMPTS
DATASET_PROMPTS=(
    ["i2p"]="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/i2p/sexual.csv"
    ["mma"]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/mma-diffusion-nsfw-adv-prompts.csv"
    ["p4dn"]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/p4dn_16_prompt.csv"
    ["ringabell"]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
    ["unlearndiff"]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/unlearn_diff_nudity.csv"
)

declare -A DATASET_DIRS
DATASET_DIRS=(
    ["i2p"]="final_i2p"
    ["mma"]="final_mma"
    ["p4dn"]="final_p4dn"
    ["ringabell"]="final_ringabell"
    ["unlearndiff"]="final_unlearndiff"
)

if [ -n "$DATASETS_ARG" ]; then
    IFS=' ' read -ra DATASETS <<< "$DATASETS_ARG"
else
    DATASETS=(i2p mma p4dn ringabell unlearndiff)
fi

# ============================================
# TOP 7 CONFIGS
# mon_threshold | guidance_scale | base_scale | sp_start | sp_end
# ============================================
declare -a CONFIGS=(
    "0.2|5|1.0|0.3|0.3"       # 1. mon0.2_gs5_bs1.0_sp0.3-0.3     (81.1)
    "0.2|10.0|2.0|0.5|0.5"    # 2. mon0.2_gs10.0_bs2.0_sp0.5-0.5  (80.9)
    "0.2|7.5|2.0|0.7|0.3"     # 3. mon0.2_gs7.5_bs2.0_sp0.7-0.3   (80.3)
    "0.2|7.5|2.0|0.5|0.5"     # 4. mon0.2_gs7.5_bs2.0_sp0.5-0.5   (80.3)
    "0.2|5|2.0|0.7|0.3"       # 5. mon0.2_gs5_bs2.0_sp0.7-0.3     (80.3)
    "0.2|5|2.0|0.5|0.5"       # 6. mon0.2_gs5_bs2.0_sp0.5-0.5     (80.3)
    "0.2|5|2.0|0.3|0.3"       # 7. mon0.2_gs5_bs2.0_sp0.3-0.3     (80.3)
)

# ============================================
# SITE SPLIT: 7 configs across 3 sites (8 GPUs each)
# A: configs 0-2 (3×5=15), B: configs 3-4 (2×5=10), C: configs 5-6 (2×5=10)
# ============================================
if [ "$SITE" = "A" ]; then
    START_CFG=0; END_CFG=3
elif [ "$SITE" = "B" ]; then
    START_CFG=3; END_CFG=5
elif [ "$SITE" = "C" ]; then
    START_CFG=5; END_CFG=7
else
    START_CFG=0; END_CFG=7  # dry-run shows all
fi

# ============================================
# GPU SETUP
# ============================================
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=(0 1 2 3 4 5 6 7)
fi
NUM_GPUS=${#GPU_LIST[@]}

# ============================================
# BUILD COMBINATIONS
# ============================================
declare -a COMBINATIONS=()

for ((ci=START_CFG; ci<END_CFG; ci++)); do
    IFS='|' read -r MT GS BS SP_S SP_E <<< "${CONFIGS[$ci]}"
    EXP_NAME="mon${MT}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}"
    for ds in "${DATASETS[@]}"; do
        COMBINATIONS+=("${ds}|${MT}|${GS}|${BS}|${SP_S}|${SP_E}|${EXP_NAME}")
    done
done

TOTAL=${#COMBINATIONS[@]}

echo "=============================================="
echo "SAFREE+MON Best-7 Generation (Site ${SITE:-ALL})"
echo "=============================================="
echo "Configs: ${START_CFG}-$((END_CFG-1)) (${#CONFIGS[@]} total)"
echo "Datasets: ${DATASETS[*]}"
echo "Experiments: ${TOTAL}"
echo "GPUs: ${GPU_LIST[*]} (${NUM_GPUS})"
echo "=============================================="

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN]"
    for combo in "${COMBINATIONS[@]}"; do
        IFS='|' read -r DS MT GS BS SP_S SP_E EXP <<< "$combo"
        echo "  ${DS}/safree_mon/${EXP}"
    done
    echo ""
    echo "Total: ${TOTAL} experiments"
    echo "Site A (configs 0-3): $((4*5)) experiments"
    echo "Site B (configs 4-6): $((3*5)) experiments"
    exit 0
fi

# ============================================
# RUN
# ============================================

run_experiment() {
    local GPU_IDX=$1
    local DS=$2
    local MT=$3
    local GS=$4
    local BS=$5
    local SP_S=$6
    local SP_E=$7
    local EXP_NAME=$8

    local ACTUAL_GPU=${GPU_LIST[$GPU_IDX]}
    local DS_DIR="${DATASET_DIRS[$DS]:-final_${DS}}"
    local OUTPUT_DIR="${OUTPUT_BASE}/${DS_DIR}/safree_mon/${EXP_NAME}"
    local PROMPT_FILE="${DATASET_PROMPTS[$DS]}"

    echo "[GPU ${ACTUAL_GPU}] Starting: ${DS}/${EXP_NAME}"

    CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python generate_safree_monitoring.py \
        --ckpt_path "${SD_CKPT}" \
        --prompt_file "${PROMPT_FILE}" \
        --output_dir "${OUTPUT_DIR}" \
        --classifier_ckpt "${CLASSIFIER_CKPT}" \
        --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
        --safree \
        --safree_alpha ${SAFREE_ALPHA} \
        --svf \
        --svf_up_t ${SVF_UP_T} \
        --category "${CATEGORY}" \
        --monitoring_threshold ${MT} \
        --guidance_scale ${GS} \
        --harmful_scale ${HARMFUL_SCALE} \
        --base_guidance_scale ${BS} \
        --spatial_threshold_start ${SP_S} \
        --spatial_threshold_end ${SP_E} \
        --spatial_threshold_strategy "${THRESHOLD_STRATEGY}" \
        --num_inference_steps ${NUM_STEPS} \
        --seed ${SEED} \
        --nsamples ${NSAMPLES} \
        --cfg_scale ${CFG_SCALE} \
        > "${OUTPUT_DIR}/run.log" 2>&1

    local STATUS=$?
    if [ $STATUS -eq 0 ]; then
        echo "[GPU ${ACTUAL_GPU}] Done: ${DS}/${EXP_NAME}"
    else
        echo "[GPU ${ACTUAL_GPU}] FAILED: ${DS}/${EXP_NAME} (exit ${STATUS})"
    fi
    return $STATUS
}

declare -A RUNNING_JOBS
IDX=0

for combo in "${COMBINATIONS[@]}"; do
    IFS='|' read -r DS MT GS BS SP_S SP_E EXP_NAME <<< "$combo"

    # Make output dir early so log can be written
    DS_DIR="${DATASET_DIRS[$DS]:-final_${DS}}"
    mkdir -p "${OUTPUT_BASE}/${DS_DIR}/safree_mon/${EXP_NAME}"

    while true; do
        for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
            JOB_PID="${RUNNING_JOBS[$gpu]:-}"
            if [ -z "$JOB_PID" ] || ! kill -0 "$JOB_PID" 2>/dev/null; then
                run_experiment $gpu "$DS" "$MT" "$GS" "$BS" "$SP_S" "$SP_E" "$EXP_NAME" &
                RUNNING_JOBS[$gpu]=$!
                IDX=$((IDX + 1))
                echo "Progress: ${IDX}/${TOTAL}"
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
echo "ALL DONE (Site ${SITE})"
echo "=============================================="
echo "Output: ${OUTPUT_BASE}/final_*/safree_mon/"
echo ""

# Quick summary of failures
FAILED=0
for combo in "${COMBINATIONS[@]}"; do
    IFS='|' read -r DS MT GS BS SP_S SP_E EXP_NAME <<< "$combo"
    DS_DIR="${DATASET_DIRS[$DS]:-final_${DS}}"
    LOG="${OUTPUT_BASE}/${DS_DIR}/safree_mon/${EXP_NAME}/run.log"
    if [ -f "$LOG" ] && grep -q "GENERATION COMPLETE" "$LOG"; then
        :
    else
        echo "FAILED: ${DS}/${EXP_NAME}"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -eq 0 ]; then
    echo "All ${TOTAL} experiments completed successfully!"
else
    echo "${FAILED}/${TOTAL} experiments failed. Check logs."
fi
