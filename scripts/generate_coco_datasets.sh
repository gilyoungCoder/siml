#!/bin/bash
# ============================================================================
# Generate images for COCO dataset with all methods (using nudity settings)
#
# Dataset:
#   - coco_30k_10k.csv (10000 prompts)
#
# Methods:
#   1. SD baseline
#   2. SAFREE
#   3. Ours (4class nudity)
#   4. SAFREE+Ours (4class nudity)
#
# Best parameters from best_configs.txt (nudity):
#   - Ours 4class: gs7.5_thr0.7-0.3_hs1.5_bgs2.0_cosine_anneal
#   - SAFREE+Ours 4class: cgs5.0_st0.5-0.3_hs0.5_bgs2.0
#
# Usage: bash generate_coco_datasets.sh <GPU> <METHOD>
# Example: bash generate_coco_datasets.sh 0 baseline
# Example: bash generate_coco_datasets.sh 0 all
# ============================================================================

set -e

# ============================================================================
# Find available GPU with most free memory
# ============================================================================
find_free_gpu() {
    python3 -c "
import subprocess
import re

result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True)
gpus = []
for line in result.stdout.strip().split('\n'):
    idx, mem = line.split(',')
    gpus.append((int(idx.strip()), int(mem.strip())))

gpus.sort(key=lambda x: x[1], reverse=True)
print(gpus[0][0])
"
}

if [ $# -lt 2 ]; then
    echo "Usage: $0 <GPU|auto> <METHOD>"
    echo ""
    echo "GPU options:"
    echo "  0,1,2,...  - specific GPU number"
    echo "  auto       - automatically select GPU with most free memory"
    echo ""
    echo "METHOD options:"
    echo "  baseline   - SD 1.4 baseline"
    echo "  safree     - SAFREE"
    echo "  ours       - Ours 4class (nudity settings)"
    echo "  safree_ours - SAFREE+Ours 4class (nudity settings)"
    echo "  all        - all methods"
    exit 1
fi

GPU_ARG=$1
METHOD=$2

# Handle auto GPU selection
if [ "${GPU_ARG}" = "auto" ]; then
    GPU=$(find_free_gpu)
    echo "Auto-selected GPU: ${GPU}"
else
    GPU=${GPU_ARG}
fi

export CUDA_VISIBLE_DEVICES=${GPU}

# ============================================================================
# Paths
# ============================================================================
BASE_DIR="/mnt/home/yhgil99/unlearning"
DATASET_DIR="${BASE_DIR}/SAFREE/datasets"
PROMPT_DIR="${BASE_DIR}/prompts/coco"
OUTPUT_BASE="${BASE_DIR}/outputs/coco"

SD_MODEL="CompVis/stable-diffusion-v1-4"
CLASSIFIER_CKPT="${BASE_DIR}/SoftDelete+CG/work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_DIR="${BASE_DIR}/SoftDelete+CG/gradcam_stats/nudity_4class"

# Create directories
mkdir -p "${PROMPT_DIR}"
mkdir -p "${OUTPUT_BASE}"

# ============================================================================
# Generation parameters
# ============================================================================
SEED=42
STEPS=50
CFG_SCALE=7.5
NSAMPLES=1

# Ours 4class best params (nudity): gs7.5_thr0.7-0.3_hs1.5_bgs2.0_cosine_anneal
OURS_GS=7.5
OURS_THR_START=0.7
OURS_THR_END=0.3
OURS_HS=1.5
OURS_BGS=2.0
OURS_STRATEGY="cosine_anneal"

# SAFREE+Ours best params (nudity): cgs5.0_st0.5-0.3_hs0.5_bgs2.0
SAFREE_OURS_CGS=5.0
SAFREE_OURS_ST_START=0.5
SAFREE_OURS_ST_END=0.3
SAFREE_OURS_HS=0.5
SAFREE_OURS_BGS=2.0

# SAFREE params
SAFREE_ALPHA=0.01
SVF_UP_T=10

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
# Generate functions
# ============================================================================
generate_baseline() {
    local prompt_file=$1
    local output_dir=$2
    local dataset_name=$3

    echo ""
    echo "=============================================="
    echo "[SD Baseline] ${dataset_name}"
    echo "=============================================="

    mkdir -p "${output_dir}"
    cd "/mnt/home/yhgil99/guided2-safe-diffusion"

    python generate.py \
        --pretrained_model_name_or_path "${SD_MODEL}" \
        --image_dir "${output_dir}" \
        --prompt_path "${prompt_file}" \
        --num_images_per_prompt ${NSAMPLES} \
        --use_fp16 \
        --seed ${SEED} \
        --device "cuda:0"

    echo "Done: ${output_dir}"
}

generate_safree() {
    local prompt_file=$1
    local output_dir=$2
    local dataset_name=$3

    echo ""
    echo "=============================================="
    echo "[SAFREE] ${dataset_name}"
    echo "=============================================="

    mkdir -p "${output_dir}"
    cd "${BASE_DIR}/SAFREE"

    python gen_safree_simple.py \
        --txt "${prompt_file}" \
        --outdir "${output_dir}" \
        --model_id "${SD_MODEL}" \
        --num_images ${NSAMPLES} \
        --steps ${STEPS} \
        --guidance ${CFG_SCALE} \
        --seed ${SEED} \
        --device "cuda:0" \
        --safree \
        --svf \
        --lra \
        --sf_alpha ${SAFREE_ALPHA} \
        --re_attn_t="-1,4" \
        --up_t ${SVF_UP_T} \
        --freeu_hyp "1.0-1.0-0.9-0.2"

    echo "Done: ${output_dir}"
}

generate_ours() {
    local prompt_file=$1
    local output_dir=$2
    local dataset_name=$3

    echo ""
    echo "=============================================="
    echo "[Ours 4class] ${dataset_name}"
    echo "=============================================="

    mkdir -p "${output_dir}"
    cd "${BASE_DIR}/SoftDelete+CG"

    python generate_nudity_4class_spatial_cg.py \
        "${SD_MODEL}" \
        --prompt_file "${prompt_file}" \
        --output_dir "${output_dir}" \
        --classifier_ckpt "${CLASSIFIER_CKPT}" \
        --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
        --nsamples ${NSAMPLES} \
        --cfg_scale ${CFG_SCALE} \
        --num_inference_steps ${STEPS} \
        --seed ${SEED} \
        --guidance_scale ${OURS_GS} \
        --spatial_threshold_start ${OURS_THR_START} \
        --spatial_threshold_end ${OURS_THR_END} \
        --threshold_strategy "${OURS_STRATEGY}" \
        --use_bidirectional \
        --harmful_scale ${OURS_HS} \
        --base_guidance_scale ${OURS_BGS}

    echo "Done: ${output_dir}"
}

generate_safree_ours() {
    local prompt_file=$1
    local output_dir=$2
    local dataset_name=$3

    echo ""
    echo "=============================================="
    echo "[SAFREE+Ours 4class] ${dataset_name}"
    echo "=============================================="

    mkdir -p "${output_dir}"
    cd "${BASE_DIR}/SAFREE"

    python generate_safree_spatial_cg.py \
        --ckpt_path "${SD_MODEL}" \
        --prompt_file "${prompt_file}" \
        --output_dir "${output_dir}" \
        --classifier_ckpt "${CLASSIFIER_CKPT}" \
        --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
        --nsamples ${NSAMPLES} \
        --cfg_scale ${CFG_SCALE} \
        --num_inference_steps ${STEPS} \
        --seed ${SEED} \
        --safree \
        --safree_alpha ${SAFREE_ALPHA} \
        --svf \
        --svf_up_t ${SVF_UP_T} \
        --spatial_cg \
        --cg_guidance_scale ${SAFREE_OURS_CGS} \
        --spatial_threshold_start ${SAFREE_OURS_ST_START} \
        --spatial_threshold_end ${SAFREE_OURS_ST_END} \
        --threshold_strategy "linear_decrease" \
        --harmful_scale ${SAFREE_OURS_HS} \
        --base_guidance_scale ${SAFREE_OURS_BGS} \
        --category "nudity"

    echo "Done: ${output_dir}"
}

# ============================================================================
# Main
# ============================================================================
# Show GPU status
echo "=============================================="
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader | head -8
echo "=============================================="
echo ""
echo "=============================================="
echo "COCO Dataset Generation (Nudity Settings)"
echo "=============================================="
echo "Selected GPU: ${GPU}"
echo "Method: ${METHOD}"
echo "Dataset: COCO 10k"
echo "=============================================="

PROMPT_FILE="${PROMPT_DIR}/coco_10k.txt"
DATASET_NAME="COCO (10000 prompts)"

# Extract prompts from coco_30k_10k.csv
extract_prompts "coco_30k_10k.csv" "coco_10k.txt" "prompt"

# Run generation
case ${METHOD} in
    baseline)
        generate_baseline "${PROMPT_FILE}" "${OUTPUT_BASE}/baseline" "${DATASET_NAME}"
        ;;
    safree)
        generate_safree "${PROMPT_FILE}" "${OUTPUT_BASE}/safree" "${DATASET_NAME}"
        ;;
    ours)
        generate_ours "${PROMPT_FILE}" "${OUTPUT_BASE}/ours_4class" "${DATASET_NAME}"
        ;;
    safree_ours)
        generate_safree_ours "${PROMPT_FILE}" "${OUTPUT_BASE}/safree_ours_4class" "${DATASET_NAME}"
        ;;
    all)
        generate_baseline "${PROMPT_FILE}" "${OUTPUT_BASE}/baseline" "${DATASET_NAME}"
        generate_safree "${PROMPT_FILE}" "${OUTPUT_BASE}/safree" "${DATASET_NAME}"
        generate_ours "${PROMPT_FILE}" "${OUTPUT_BASE}/ours_4class" "${DATASET_NAME}"
        generate_safree_ours "${PROMPT_FILE}" "${OUTPUT_BASE}/safree_ours_4class" "${DATASET_NAME}"
        ;;
    *)
        echo "Unknown method: ${METHOD}"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Generation Complete!"
echo "=============================================="
echo "Output: ${OUTPUT_BASE}"
echo "=============================================="
