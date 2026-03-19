#!/bin/bash
# ============================================================================
# Unified Grid Search: monitoring mode x spatial mode
#
# Monitoring modes:
#   gradcam:    GradCAM P(harm) > threshold (existing algorithm)
#   z0:         z0-trigger per-step
#   z0_sticky:  z0-trigger + sticky (once triggered, guide all remaining)
#
# Spatial modes:
#   gradcam:          GradCAM heatmap CDF mask (existing)
#   attention:        cross-attention token map mask
#   attention_gradcam: GradCAM x attention product mask
#
# Phase 1: 166 configs x 2 datasets (ringabell + coco) = 332 experiments
#
# Usage:
#   bash grid_search_unified_monitoring.sh --site A
#   bash grid_search_unified_monitoring.sh --site B
#   bash grid_search_unified_monitoring.sh --site A --nohup
#   bash grid_search_unified_monitoring.sh --dry-run
# ============================================================================

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd

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
    echo "Usage: bash grid_search_unified_monitoring.sh --site A|B [--nohup] [--dry-run]"
    exit 1
fi

# Nohup handling
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="./scg_outputs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/unified_grid_site${SITE}_${TIMESTAMP}.log"
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
GEN_SCRIPT="generate_unified_monitoring.py"

# Datasets: Phase 1 = ringabell + coco only
declare -A DATASET_PROMPTS
DATASET_PROMPTS[ringabell]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
DATASET_PROMPTS[coco]="/mnt/home/yhgil99/unlearning/SAFREE/datasets/coco_30k.csv"

declare -A DATASET_END_IDX
DATASET_END_IDX[ringabell]=-1
DATASET_END_IDX[coco]=50

DATASETS=(ringabell coco)

# Fixed params
NUM_STEPS=50
SEED=42
NSAMPLES=1
CFG_SCALE=7.5
THRESHOLD_STRATEGY="cosine"

HARMFUL_KW="nude naked breast topless bare undress nsfw"

# VLM eval
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
OUTPUT_BASE="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/unified_grid"

# ============================================
# BUILD COMBINATIONS
# monitoring_mode|spatial_mode|mon_thr|gs|bs|sp_start sp_end
# ============================================

declare -a COMBINATIONS=()

# --- Group 1: gradcam monitoring + gradcam spatial (36) ---
for mt in 0.05 0.1 0.15; do
    for gs in 12.5 15 17.5; do
        for sp in "0.1 0.3" "0.1 0.4" "0.2 0.4" "0.3 0.5"; do
            for ds in "${DATASETS[@]}"; do
                COMBINATIONS+=("${ds}|gradcam|gradcam|${mt}|${gs}|1.0|${sp}")
            done
        done
    done
done

# --- Group 2: z0_sticky monitoring + gradcam spatial (36) ---
for mt in 0.01 0.03 0.06; do
    for gs in 12.5 15 17.5; do
        for sp in "0.1 0.3" "0.1 0.4" "0.2 0.4" "0.3 0.5"; do
            for ds in "${DATASETS[@]}"; do
                COMBINATIONS+=("${ds}|z0_sticky|gradcam|${mt}|${gs}|1.0|${sp}")
            done
        done
    done
done

# --- Group 3: z0 monitoring + gradcam spatial (36) ---
for mt in 0.01 0.03 0.06; do
    for gs in 12.5 15 17.5; do
        for sp in "0.1 0.3" "0.1 0.4" "0.2 0.4" "0.3 0.5"; do
            for ds in "${DATASETS[@]}"; do
                COMBINATIONS+=("${ds}|z0|gradcam|${mt}|${gs}|1.0|${sp}")
            done
        done
    done
done

# --- Group 4: gradcam monitoring + attention spatial (27) ---
for mt in 0.05 0.1 0.15; do
    for gs in 12.5 15 17.5; do
        for sp in "0.2 0.3" "0.3 0.3" "0.3 0.5"; do
            for ds in "${DATASETS[@]}"; do
                COMBINATIONS+=("${ds}|gradcam|attention|${mt}|${gs}|1.0|${sp}")
            done
        done
    done
done

# --- Group 5: z0_sticky monitoring + attention spatial (27) ---
for mt in 0.01 0.03 0.06; do
    for gs in 12.5 15 17.5; do
        for sp in "0.2 0.3" "0.3 0.3" "0.3 0.5"; do
            for ds in "${DATASETS[@]}"; do
                COMBINATIONS+=("${ds}|z0_sticky|attention|${mt}|${gs}|1.0|${sp}")
            done
        done
    done
done

# --- Group 6: z0_sticky monitoring + attention_gradcam spatial (4) ---
for mt in 0.03 0.06; do
    for sp in "0.2 0.3" "0.3 0.5"; do
        for ds in "${DATASETS[@]}"; do
            COMBINATIONS+=("${ds}|z0_sticky|attention_gradcam|${mt}|15|1.0|${sp}")
        done
    done
done

TOTAL=${#COMBINATIONS[@]}
echo "=============================================="
echo "UNIFIED GRID SEARCH: monitoring x spatial"
echo "=============================================="
echo "Total experiments: $TOTAL"
echo "Datasets: ${DATASETS[*]}"
echo ""
echo "Monitoring modes: gradcam, z0, z0_sticky"
echo "Spatial modes:    gradcam, attention, attention_gradcam"
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
        IFS='|' read -r DS MON_MODE SP_MODE MT GS BS SP <<< "$combo"
        SP_START=$(echo $SP | awk '{print $1}')
        SP_END_VAL=$(echo $SP | awk '{print $2}')
        echo "[$i] ${DS} | ${MON_MODE}+${SP_MODE} | mon${MT}_gs${GS}_bs${BS}_sp${SP_START}-${SP_END_VAL}"
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
# RUN FUNCTION
# ============================================

run_experiment() {
    local GPU_IDX=$1
    local DS=$2
    local MON_MODE=$3
    local SP_MODE=$4
    local MON_THR=$5
    local GS=$6
    local BS=$7
    local SP_START=$8
    local SP_END_VAL=$9

    local ACTUAL_GPU=${GPU_LIST[$GPU_IDX]}
    local EXP_NAME="${MON_MODE}_${SP_MODE}_mon${MON_THR}_gs${GS}_bs${BS}_sp${SP_START}-${SP_END_VAL}"
    local OUTPUT_DIR="${OUTPUT_BASE}/${DS}/${EXP_NAME}"
    local LOG_FILE="${OUTPUT_BASE}/logs/${DS}_${EXP_NAME}.log"
    local PROMPT_FILE="${DATASET_PROMPTS[$DS]}"
    local DS_END="${DATASET_END_IDX[$DS]}"

    # Skip if already evaluated
    if [ -f "${OUTPUT_DIR}/categories_qwen3_vl_nudity.json" ]; then
        echo "[GPU ${ACTUAL_GPU}] SKIP (eval done): ${DS}/${EXP_NAME}"
        return 0
    fi

    echo "[GPU ${ACTUAL_GPU}] START: ${DS}/${EXP_NAME}"

    # Build extra args based on modes
    local EXTRA_ARGS=""
    if [ "$SP_MODE" = "attention" ] || [ "$SP_MODE" = "attention_gradcam" ]; then
        EXTRA_ARGS="--harmful_keywords ${HARMFUL_KW} --attn_resolutions 16 32"
    fi

    local END_IDX_ARG=""
    if [ "$DS_END" != "-1" ]; then
        END_IDX_ARG="--end_idx ${DS_END}"
    fi

    # Step 1: Generate
    if [ ! -f "${OUTPUT_DIR}/generation_stats.json" ]; then
        CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python "${GEN_SCRIPT}" \
            --ckpt_path "${SD_CKPT}" \
            --prompt_file "${PROMPT_FILE}" \
            --output_dir "${OUTPUT_DIR}" \
            --classifier_ckpt "${CLASSIFIER_CKPT}" \
            --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
            --monitoring_mode "${MON_MODE}" \
            --monitoring_threshold ${MON_THR} \
            --spatial_mode "${SP_MODE}" \
            --guidance_scale ${GS} \
            --base_guidance_scale ${BS} \
            --spatial_threshold_start ${SP_START} \
            --spatial_threshold_end ${SP_END_VAL} \
            --spatial_threshold_strategy "${THRESHOLD_STRATEGY}" \
            --num_inference_steps ${NUM_STEPS} \
            --cfg_scale ${CFG_SCALE} \
            --seed ${SEED} \
            --nsamples ${NSAMPLES} \
            ${END_IDX_ARG} \
            ${EXTRA_ARGS} \
            >> "${LOG_FILE}" 2>&1

        if [ $? -ne 0 ]; then
            echo "[GPU ${ACTUAL_GPU}] FAILED generation: ${DS}/${EXP_NAME}"
            return 1
        fi
    else
        echo "[GPU ${ACTUAL_GPU}] SKIP generation (already done): ${DS}/${EXP_NAME}"
    fi

    # Step 2: Qwen VLM eval (skip for coco)
    if [ "$DS" != "coco" ]; then
        eval "$(conda shell.bash hook 2>/dev/null)" && conda activate vlm
        CUDA_VISIBLE_DEVICES=${ACTUAL_GPU} python "${VLM_SCRIPT}" "${OUTPUT_DIR}" nudity qwen \
            >> "${LOG_FILE}" 2>&1
        eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
    fi

    echo "[GPU ${ACTUAL_GPU}] DONE: ${DS}/${EXP_NAME}"
    return 0
}

# ============================================
# MAIN LOOP
# ============================================

declare -A RUNNING_JOBS
IDX=0

for (( ci=START; ci<END; ci++ )); do
    combo="${COMBINATIONS[$ci]}"
    IFS='|' read -r DS MON_MODE SP_MODE MT GS BS SP <<< "$combo"
    SP_START=$(echo $SP | awk '{print $1}')
    SP_END_VAL=$(echo $SP | awk '{print $2}')

    # Find available GPU
    while true; do
        for ((gpu=0; gpu<${#GPU_LIST[@]}; gpu++)); do
            JOB_PID="${RUNNING_JOBS[$gpu]}"
            if [ -z "$JOB_PID" ] || ! kill -0 "$JOB_PID" 2>/dev/null; then
                run_experiment $gpu "$DS" "$MON_MODE" "$SP_MODE" "$MT" "$GS" "$BS" "$SP_START" "$SP_END_VAL" &
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
echo "UNIFIED GRID SEARCH COMPLETE (Site $SITE)"
echo "=============================================="
echo "Results: ${OUTPUT_BASE}/"
echo ""
echo "Done!"
