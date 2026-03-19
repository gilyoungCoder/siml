#!/bin/bash
# ============================================================================
# Generate datasets using 3-class Adaptive Spatial CG (SKIP mode)
# Parallel execution on multiple GPUs
#
# Datasets:
#   1. nudity-ring-a-bell (79 prompts)
#   2. mma-diffusion-nsfw-adv-prompts (1000 prompts)
#   3. nudity.csv (142 prompts)
#   4. COCO 10k (10000 prompts)
#
# Method: 3-class Adaptive Spatial CG with SKIP mode
# Best Config: gs15.0_hs1.5_st0.1-0.4_cosine_anneal (from GPT-4o eval)
#
# Usage: bash generate_nudity_3class_parallel.sh [GPU_START]
# Example: bash generate_nudity_3class_parallel.sh 0
#          -> ringabell on GPU0, mma on GPU1, nudity on GPU2, coco on GPU3
# ============================================================================

set -e

GPU_START=${1:-0}

# ============================================================================
# Paths
# ============================================================================
BASE_DIR="/mnt/home/yhgil99/unlearning"
DATASET_DIR="${BASE_DIR}/SAFREE/datasets"
COCO_DIR="${BASE_DIR}/datasets"
PROMPT_DIR="${BASE_DIR}/prompts/nudity_datasets"
OUTPUT_BASE="${BASE_DIR}/outputs/nudity_3class_skip"

SD_MODEL="CompVis/stable-diffusion-v1-4"
CLASSIFIER_CKPT="${BASE_DIR}/SoftDelete+CG/work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"

# Create directories
mkdir -p "${PROMPT_DIR}"
mkdir -p "${OUTPUT_BASE}"

# ============================================================================
# Generation parameters (Best Config from GPT-4o evaluation)
# ============================================================================
SEED=42
STEPS=50
CFG_SCALE=7.5
NSAMPLES=1

# 3-class Adaptive Spatial CG params
GUIDANCE_SCALE=15.0
HARMFUL_SCALE=1.5
SPATIAL_THR_START=0.1
SPATIAL_THR_END=0.4
THRESHOLD_STRATEGY="cosine_anneal"
GUIDANCE_START_STEP=0
GUIDANCE_END_STEP=50

# ============================================================================
# Extract prompts from CSV to txt
# ============================================================================
extract_prompts() {
    local csv_name=$1
    local txt_name=$2
    local column=$3

    local csv_path="${DATASET_DIR}/${csv_name}"
    local txt_path="${PROMPT_DIR}/${txt_name}"

    if [ ! -f "${txt_path}" ]; then
        echo "Extracting prompts from ${csv_name}..."
        python "${BASE_DIR}/scripts/extract_prompts_from_csv.py" \
            "${csv_path}" \
            --output "${txt_path}" \
            --column "${column}"
    else
        echo "Prompts already extracted: ${txt_path}"
    fi
}

# ============================================================================
# Generate function
# ============================================================================
generate_3class_skip() {
    local gpu=$1
    local prompt_file=$2
    local output_dir=$3
    local dataset_name=$4

    echo ""
    echo "=============================================="
    echo "[3class SKIP] ${dataset_name}"
    echo "GPU: ${gpu}"
    echo "=============================================="

    export CUDA_VISIBLE_DEVICES=${gpu}
    mkdir -p "${output_dir}"
    cd "${BASE_DIR}/SoftDelete+CG"

    python generate_adaptive_spatial_cg.py \
        "${SD_MODEL}" \
        --prompt_file "${prompt_file}" \
        --output_dir "${output_dir}" \
        --classifier_ckpt "${CLASSIFIER_CKPT}" \
        --num_classes 3 \
        --harmful_class 2 \
        --safe_class 1 \
        --nsamples ${NSAMPLES} \
        --cfg_scale ${CFG_SCALE} \
        --num_inference_steps ${STEPS} \
        --seed ${SEED} \
        --guidance_scale ${GUIDANCE_SCALE} \
        --spatial_threshold_start ${SPATIAL_THR_START} \
        --spatial_threshold_end ${SPATIAL_THR_END} \
        --threshold_strategy "${THRESHOLD_STRATEGY}" \
        --guidance_start_step ${GUIDANCE_START_STEP} \
        --guidance_end_step ${GUIDANCE_END_STEP} \
        --harmful_scale ${HARMFUL_SCALE} \
        --use_bidirectional \
        --skip_safe

    echo "Done: ${output_dir}"
}

# ============================================================================
# Main
# ============================================================================

# Calculate GPU assignments
GPU_RINGABELL=$((GPU_START))
GPU_MMA=$((GPU_START + 1))
GPU_NUDITY=$((GPU_START + 2))
GPU_COCO=$((GPU_START + 3))

echo "=============================================="
echo "Parallel Dataset Generation (3class SKIP)"
echo "=============================================="
echo ""
echo "Datasets:"
echo "  1. nudity-ring-a-bell (79 prompts)"
echo "  2. mma-diffusion-nsfw-adv-prompts (1000 prompts)"
echo "  3. nudity.csv (142 prompts)"
echo "  4. COCO 10k (10000 prompts)"
echo ""
echo "GPU Assignment:"
echo "  GPU ${GPU_RINGABELL}: Ring-a-Bell (79)"
echo "  GPU ${GPU_MMA}: MMA Adversarial (1000)"
echo "  GPU ${GPU_NUDITY}: Nudity (142)"
echo "  GPU ${GPU_COCO}: COCO 10k (10000)"
echo ""
echo "Method: 3-class Adaptive Spatial CG (SKIP mode)"
echo "Parameters:"
echo "  guidance_scale: ${GUIDANCE_SCALE}"
echo "  harmful_scale: ${HARMFUL_SCALE}"
echo "  spatial_threshold: ${SPATIAL_THR_START} → ${SPATIAL_THR_END}"
echo "  strategy: ${THRESHOLD_STRATEGY}"
echo "=============================================="
echo ""

# Show GPU status
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader | head -8
echo ""

# Skip confirmation if running non-interactively (nohup)
if [ -t 0 ]; then
    read -p "Start generation on GPUs ${GPU_START}-$((GPU_START+3))? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
else
    echo "Running non-interactively, skipping confirmation..."
fi

# Extract prompts
echo ""
echo "Extracting prompts..."
extract_prompts "nudity-ring-a-bell.csv" "ringabell.txt" "prompt"
extract_prompts "mma-diffusion-nsfw-adv-prompts.csv" "mma.txt" "adv_prompt"
extract_prompts "nudity.csv" "nudity.txt" "prompt"

# COCO prompts (already exists)
COCO_PROMPT="/mnt/home/yhgil99/safeguard-with-human/prompts/coco_10k.txt"
if [ -f "${COCO_PROMPT}" ]; then
    echo "Using existing COCO prompts: ${COCO_PROMPT}"
else
    echo "ERROR: COCO prompts not found at ${COCO_PROMPT}"
    exit 1
fi

# Create log directory
LOG_DIR="${BASE_DIR}/logs/nudity_3class_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo ""
echo "Starting parallel generation..."
echo "Logs: ${LOG_DIR}/"
echo ""

# Run all datasets in parallel with nohup
nohup bash -c "$(declare -f extract_prompts generate_3class_skip); \
    export BASE_DIR='${BASE_DIR}'; \
    export SD_MODEL='${SD_MODEL}'; \
    export CLASSIFIER_CKPT='${CLASSIFIER_CKPT}'; \
    export NSAMPLES=${NSAMPLES}; \
    export CFG_SCALE=${CFG_SCALE}; \
    export STEPS=${STEPS}; \
    export SEED=${SEED}; \
    export GUIDANCE_SCALE=${GUIDANCE_SCALE}; \
    export SPATIAL_THR_START=${SPATIAL_THR_START}; \
    export SPATIAL_THR_END=${SPATIAL_THR_END}; \
    export THRESHOLD_STRATEGY='${THRESHOLD_STRATEGY}'; \
    export GUIDANCE_START_STEP=${GUIDANCE_START_STEP}; \
    export GUIDANCE_END_STEP=${GUIDANCE_END_STEP}; \
    export HARMFUL_SCALE=${HARMFUL_SCALE}; \
    generate_3class_skip ${GPU_RINGABELL} '${PROMPT_DIR}/ringabell.txt' '${OUTPUT_BASE}/ringabell' 'Ring-a-Bell (79)'" \
    > "${LOG_DIR}/ringabell.log" 2>&1 &
PID_RINGABELL=$!
echo "[GPU ${GPU_RINGABELL}] Ring-a-Bell started (PID: ${PID_RINGABELL})"

nohup bash -c "$(declare -f extract_prompts generate_3class_skip); \
    export BASE_DIR='${BASE_DIR}'; \
    export SD_MODEL='${SD_MODEL}'; \
    export CLASSIFIER_CKPT='${CLASSIFIER_CKPT}'; \
    export NSAMPLES=${NSAMPLES}; \
    export CFG_SCALE=${CFG_SCALE}; \
    export STEPS=${STEPS}; \
    export SEED=${SEED}; \
    export GUIDANCE_SCALE=${GUIDANCE_SCALE}; \
    export SPATIAL_THR_START=${SPATIAL_THR_START}; \
    export SPATIAL_THR_END=${SPATIAL_THR_END}; \
    export THRESHOLD_STRATEGY='${THRESHOLD_STRATEGY}'; \
    export GUIDANCE_START_STEP=${GUIDANCE_START_STEP}; \
    export GUIDANCE_END_STEP=${GUIDANCE_END_STEP}; \
    export HARMFUL_SCALE=${HARMFUL_SCALE}; \
    generate_3class_skip ${GPU_MMA} '${PROMPT_DIR}/mma.txt' '${OUTPUT_BASE}/mma' 'MMA Adversarial (1000)'" \
    > "${LOG_DIR}/mma.log" 2>&1 &
PID_MMA=$!
echo "[GPU ${GPU_MMA}] MMA Adversarial started (PID: ${PID_MMA})"

nohup bash -c "$(declare -f extract_prompts generate_3class_skip); \
    export BASE_DIR='${BASE_DIR}'; \
    export SD_MODEL='${SD_MODEL}'; \
    export CLASSIFIER_CKPT='${CLASSIFIER_CKPT}'; \
    export NSAMPLES=${NSAMPLES}; \
    export CFG_SCALE=${CFG_SCALE}; \
    export STEPS=${STEPS}; \
    export SEED=${SEED}; \
    export GUIDANCE_SCALE=${GUIDANCE_SCALE}; \
    export SPATIAL_THR_START=${SPATIAL_THR_START}; \
    export SPATIAL_THR_END=${SPATIAL_THR_END}; \
    export THRESHOLD_STRATEGY='${THRESHOLD_STRATEGY}'; \
    export GUIDANCE_START_STEP=${GUIDANCE_START_STEP}; \
    export GUIDANCE_END_STEP=${GUIDANCE_END_STEP}; \
    export HARMFUL_SCALE=${HARMFUL_SCALE}; \
    generate_3class_skip ${GPU_NUDITY} '${PROMPT_DIR}/nudity.txt' '${OUTPUT_BASE}/nudity' 'Nudity (142)'" \
    > "${LOG_DIR}/nudity.log" 2>&1 &
PID_NUDITY=$!
echo "[GPU ${GPU_NUDITY}] Nudity started (PID: ${PID_NUDITY})"

nohup bash -c "$(declare -f extract_prompts generate_3class_skip); \
    export BASE_DIR='${BASE_DIR}'; \
    export SD_MODEL='${SD_MODEL}'; \
    export CLASSIFIER_CKPT='${CLASSIFIER_CKPT}'; \
    export NSAMPLES=${NSAMPLES}; \
    export CFG_SCALE=${CFG_SCALE}; \
    export STEPS=${STEPS}; \
    export SEED=${SEED}; \
    export GUIDANCE_SCALE=${GUIDANCE_SCALE}; \
    export SPATIAL_THR_START=${SPATIAL_THR_START}; \
    export SPATIAL_THR_END=${SPATIAL_THR_END}; \
    export THRESHOLD_STRATEGY='${THRESHOLD_STRATEGY}'; \
    export GUIDANCE_START_STEP=${GUIDANCE_START_STEP}; \
    export GUIDANCE_END_STEP=${GUIDANCE_END_STEP}; \
    export HARMFUL_SCALE=${HARMFUL_SCALE}; \
    generate_3class_skip ${GPU_COCO} '${COCO_PROMPT}' '${OUTPUT_BASE}/coco_10k' 'COCO 10k (10000)'" \
    > "${LOG_DIR}/coco.log" 2>&1 &
PID_COCO=$!
echo "[GPU ${GPU_COCO}] COCO 10k started (PID: ${PID_COCO})"

echo ""
echo "=============================================="
echo "All jobs started!"
echo "=============================================="
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOG_DIR}/ringabell.log"
echo "  tail -f ${LOG_DIR}/mma.log"
echo "  tail -f ${LOG_DIR}/nudity.log"
echo "  tail -f ${LOG_DIR}/coco.log"
echo ""
echo "Or watch all:"
echo "  watch -n 5 'for f in ${LOG_DIR}/*.log; do echo \"=== \$f ===\"; tail -3 \$f; done'"
echo ""
echo "PIDs: ${PID_RINGABELL} ${PID_MMA} ${PID_NUDITY} ${PID_COCO}"
echo ""
echo "Output: ${OUTPUT_BASE}/"
echo "=============================================="
