#!/bin/bash
# ============================================================================
# Fine Grid Search: Monitoring-Only (4class) — NO SAFREE
# Best config: mon0.1_gs15_sp0.1-0.4_bs1.0 (ringabell SR=77.2%)
# Datasets: p4dn, ringabell, unlearndiff
# Eval: Qwen VLM SR (nudity)
#
# Usage:
#   bash grid_search_fine_mon4class.sh --site A          # first half
#   bash grid_search_fine_mon4class.sh --site B          # second half
#   bash grid_search_fine_mon4class.sh --site A --nohup  # background
#   bash grid_search_fine_mon4class.sh --dry-run         # print only
# ============================================================================

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

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
    echo "Usage: bash grid_search_fine_mon4class.sh --site A|B [--nohup] [--dry-run]"
    exit 1
fi

# Nohup handling
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="./scg_outputs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/fine_grid_mon4class_site${SITE}_${TIMESTAMP}.log"
    echo "Running in background (site $SITE)..."
    echo "Log: $LOG_FILE"
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$SCRIPT_PATH" --site "$SITE" --num-gpus "$NUM_GPUS" > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

# ============================================
# CONFIGURATION
# ============================================

CLASSIFIER_CKPT="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/nudity_4class"
SD_CKPT="CompVis/stable-diffusion-v1-4"
GEN_SCRIPT="generate_nudity_4class_sample_level_monitoring.py"

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
THRESHOLD_STRATEGY="cosine"

# ============================================
# FINE GRID SEARCH PARAMETERS
# (around best: mon0.1_gs15_sp0.1-0.4_bs1.0)
# ============================================

MONITORING_THRESHOLDS=(0.05 0.1 0.15)
GUIDANCE_SCALES=(12.5 15 17.5)
BASE_SCALES=(0.5 1.0 2.0)
SPATIAL_THRESHOLDS=(
    "0.1 0.2"
    "0.1 0.3"
    "0.1 0.4"
    "0.1 0.5"
    "0.2 0.3"
    "0.2 0.4"
    "0.3 0.3"
    "0.3 0.5"
)
# Total per dataset: 3 x 3 x 3 x 8 = 72

# VLM eval script
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
OUTPUT_BASE="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/fine_grid_mon4class"

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
CONFIGS_PER_DS=$(( ${#MONITORING_THRESHOLDS[@]} * ${#GUIDANCE_SCALES[@]} * ${#BASE_SCALES[@]} * ${#SPATIAL_THRESHOLDS[@]} ))
echo "=============================================="
echo "FINE GRID SEARCH: MONITORING-ONLY (4class)"
echo "=============================================="
echo "Total combinations: $TOTAL ($CONFIGS_PER_DS configs x ${#DATASETS[@]} datasets)"
echo "Datasets: ${DATASETS[*]}"
echo "mon_thr: ${MONITORING_THRESHOLDS[*]}"
echo "gs: ${GUIDANCE_SCALES[*]}"
echo "bs: ${BASE_SCALES[*]}"
echo "spatial: 0.1-0.2, 0.1-0.3, 0.1-0.4, 0.1-0.5, 0.2-0.3, 0.2-0.4, 0.3-0.3, 0.3-0.5"
echo "=============================================="

# Split for sites
HALF=$(( (TOTAL + 1) / 2 ))
if [ "$SITE" = "A" ]; then
    START=0; END=$HALF
elif [ "$SITE" = "B" ]; then
    START=$HALF; END=$TOTAL
else
    START=0; END=$TOTAL
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
        CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python "${GEN_SCRIPT}" \
            --ckpt_path "${SD_CKPT}" \
            --prompt_file "${PROMPT_FILE}" \
            --output_dir "${OUTPUT_DIR}" \
            --classifier_ckpt "${CLASSIFIER_CKPT}" \
            --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
            --monitoring_threshold ${MON_THR} \
            --guidance_scale ${GS} \
            --base_guidance_scale ${BS} \
            --spatial_threshold_start ${SP_START} \
            --spatial_threshold_end ${SP_END} \
            --spatial_threshold_strategy "${THRESHOLD_STRATEGY}" \
            --num_inference_steps ${NUM_STEPS} \
            --cfg_scale ${CFG_SCALE} \
            --seed ${SEED} \
            --nsamples ${NSAMPLES} \
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
echo "  python vlm/aggregate_fine_grid.py --base-dir ${OUTPUT_BASE}"
echo ""
echo "Done!"
