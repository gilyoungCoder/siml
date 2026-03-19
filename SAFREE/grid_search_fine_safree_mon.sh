#!/bin/bash
# ============================================================================
# Fine Grid Search: SAFREE + Monitoring around best configs
# Datasets: p4dn, ringabell, unlearndiff
# Eval: Qwen VLM SR (nudity)
#
# Usage:
#   bash grid_search_fine_safree_mon.sh --site A          # first half
#   bash grid_search_fine_safree_mon.sh --site B          # second half
#   bash grid_search_fine_safree_mon.sh --site A --nohup  # background
#   bash grid_search_fine_safree_mon.sh --dry-run         # print only
# ============================================================================

cd /mnt/home/yhgil99/unlearning/SAFREE

# ============================================
# Parse Arguments
# ============================================
DRY_RUN=false
USE_NOHUP=false
NUM_GPUS=8
SITE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift;;
        --nohup) USE_NOHUP=true; shift;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        --site) SITE="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

if [ -z "$SITE" ] && [ "$DRY_RUN" = false ]; then
    echo "Usage: bash grid_search_fine_safree_mon.sh --site A|B [--nohup] [--dry-run]"
    exit 1
fi

# Nohup handling
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="./results"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/fine_grid_site${SITE}_${TIMESTAMP}.log"
    echo "Running in background (site $SITE)..."
    echo "Log: $LOG_FILE"
    nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$0" --site "$SITE" --num-gpus "$NUM_GPUS" > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

# ============================================
# CONFIGURATION
# ============================================

CLASSIFIER_CKPT="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/nudity_4class"
SD_CKPT="CompVis/stable-diffusion-v1-4"

# Datasets
declare -A DATASET_PROMPTS
DATASET_PROMPTS[p4dn]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/p4dn_16_prompt.csv"
DATASET_PROMPTS[ringabell]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
DATASET_PROMPTS[unlearndiff]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/unlearn_diff_nudity.csv"

DATASETS=(p4dn ringabell unlearndiff)

# Fixed params
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
# FINE GRID SEARCH PARAMETERS
# ============================================

MONITORING_THRESHOLDS=(0.1 0.15 0.2 0.25)
GUIDANCE_SCALES=(3 5 7.5 10)
BASE_SCALES=(0.5 1.0 1.5 2.0)
SPATIAL_THRESHOLDS=(
    "0.3 0.3"
    "0.5 0.3"
    "0.5 0.5"
    "0.7 0.3"
)

# VLM eval script
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
OUTPUT_BASE="/mnt/home/yhgil99/unlearning/SAFREE/results/fine_grid"

# ============================================
# BUILD COMBINATIONS
# ============================================

declare -a COMBINATIONS=()
for ds in "${DATASETS[@]}"; do
    for mt in "${MONITORING_THRESHOLDS[@]}"; do
        for gs in "${GUIDANCE_SCALES[@]}"; do
            for bs in "${BASE_SCALES[@]}"; do
                for sp in "${SPATIAL_THRESHOLDS[@]}"; do
                    COMBINATIONS+=("${ds}|${mt}|${gs}|${bs}|${sp}")
                done
            done
        done
    done
done

TOTAL=${#COMBINATIONS[@]}
echo "=============================================="
echo "FINE GRID SEARCH: SAFREE + MONITORING"
echo "=============================================="
echo "Total combinations: $TOTAL (256 configs x 3 datasets)"
echo "Datasets: ${DATASETS[*]}"
echo "mon_thr: ${MONITORING_THRESHOLDS[*]}"
echo "gs: ${GUIDANCE_SCALES[*]}"
echo "bs: ${BASE_SCALES[*]}"
echo "spatial: 0.3-0.3, 0.5-0.3, 0.5-0.5, 0.7-0.3"
echo "=============================================="

# Split for sites
HALF=$(( (TOTAL + 1) / 2 ))
if [ "$SITE" = "A" ]; then
    START=0; END=$HALF
elif [ "$SITE" = "B" ]; then
    START=$HALF; END=$TOTAL
else
    START=0; END=$TOTAL  # dry-run shows all
fi
echo "Site $SITE: experiments $START to $((END-1)) (total $((END-START)))"
echo ""

if [ "$DRY_RUN" = true ]; then
    for (( i=START; i<END; i++ )); do
        combo="${COMBINATIONS[$i]}"
        IFS='|' read -r DS MT GS BS SP <<< "$combo"
        SP_START=$(echo $SP | awk '{print $1}')
        SP_END=$(echo $SP | awk '{print $2}')
        echo "[$i] ${DS} | mon${MT}_gs${GS}_bs${BS}_sp${SP_START}-${SP_END}"
    done
    echo ""
    echo "Total: $((END-START)) experiments"
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
# RUN FUNCTION (generate + eval)
# ============================================

run_experiment() {
    local GPU_IDX=$1
    local DS=$2
    local MON_THR=$3
    local GS=$4
    local BS=$5
    local SP_START=$6
    local SP_END=$7

    local ACTUAL_GPU=${GPU_LIST[$GPU_IDX]}
    local EXP_NAME="mon${MON_THR}_gs${GS}_bs${BS}_sp${SP_START}-${SP_END}"
    local OUTPUT_DIR="${OUTPUT_BASE}/${DS}/${EXP_NAME}"
    local LOG_FILE="${OUTPUT_BASE}/logs/${DS}_${EXP_NAME}.log"
    local PROMPT_FILE="${DATASET_PROMPTS[$DS]}"

    # Skip if already evaluated
    if [ -f "${OUTPUT_DIR}/categories_qwen3_vl_nudity.json" ]; then
        echo "[GPU ${ACTUAL_GPU}] SKIP (eval done): ${DS}/${EXP_NAME}"
        return 0
    fi

    echo "[GPU ${ACTUAL_GPU}] START: ${DS}/${EXP_NAME}"

    # Step 1: Generate images (skip if already done)
    if [ ! -f "${OUTPUT_DIR}/generation_stats.json" ]; then
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
            --monitoring_threshold ${MON_THR} \
            --guidance_scale ${GS} \
            --harmful_scale ${HARMFUL_SCALE} \
            --base_guidance_scale ${BS} \
            --spatial_threshold_start ${SP_START} \
            --spatial_threshold_end ${SP_END} \
            --spatial_threshold_strategy "${THRESHOLD_STRATEGY}" \
            --num_inference_steps ${NUM_STEPS} \
            --seed ${SEED} \
            --nsamples ${NSAMPLES} \
            --cfg_scale ${CFG_SCALE} \
            >> "${LOG_FILE}" 2>&1

        if [ $? -ne 0 ]; then
            echo "[GPU ${ACTUAL_GPU}] FAILED generation: ${DS}/${EXP_NAME}"
            return 1
        fi
    else
        echo "[GPU ${ACTUAL_GPU}] SKIP generation (already done): ${DS}/${EXP_NAME}"
    fi

    # Step 2: Qwen VLM eval
    CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python "${VLM_SCRIPT}" "${OUTPUT_DIR}" nudity qwen \
        >> "${LOG_FILE}" 2>&1

    if [ $? -eq 0 ]; then
        echo "[GPU ${ACTUAL_GPU}] DONE: ${DS}/${EXP_NAME}"
    else
        echo "[GPU ${ACTUAL_GPU}] FAILED eval: ${DS}/${EXP_NAME}"
        return 1
    fi

    return 0
}

# ============================================
# MAIN LOOP
# ============================================

declare -A RUNNING_JOBS
IDX=0

for (( ci=START; ci<END; ci++ )); do
    combo="${COMBINATIONS[$ci]}"
    IFS='|' read -r DS MT GS BS SP <<< "$combo"
    SP_START=$(echo $SP | awk '{print $1}')
    SP_END=$(echo $SP | awk '{print $2}')

    # Find available GPU
    while true; do
        for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
            JOB_PID="${RUNNING_JOBS[$gpu]}"
            if [ -z "$JOB_PID" ] || ! kill -0 "$JOB_PID" 2>/dev/null; then
                run_experiment $gpu "$DS" "$MT" "$GS" "$BS" "$SP_START" "$SP_END" &
                RUNNING_JOBS[$gpu]=$!
                IDX=$((IDX + 1))
                echo "Progress: ${IDX}/$((END-START)) [combo $ci]"
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
echo "FINE GRID SEARCH COMPLETE (Site $SITE)"
echo "=============================================="
echo "Results: ${OUTPUT_BASE}/"
echo ""
echo "Run aggregation:"
echo "  python vlm/aggregate_fine_grid.py"
echo ""
echo "Done!"
