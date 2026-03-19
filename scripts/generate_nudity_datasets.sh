#!/bin/bash
# ============================================================================
# Generate images for nudity datasets with all methods
#
# Datasets:
#   1. nudity-ring-a-bell (79 prompts)
#   2. mma-diffusion-nsfw-adv-prompts (1000 prompts)
#   3. nudity.csv (142 prompts)
#
# Methods:
#   1. SD baseline
#   2. SAFREE
#   3. Ours (4class)
#   4. SAFREE+Ours (4class)
#
# Best parameters from best_configs.txt:
#   - Ours 4class: gs7.5_thr0.7-0.3_hs1.5_bgs2.0_cosine_anneal
#   - SAFREE+Ours 4class: cgs5.0_st0.5-0.3_hs0.5_bgs2.0
#
# Usage: bash generate_nudity_datasets.sh <GPU> <DATASET> <METHOD>
# Example: bash generate_nudity_datasets.sh 0 ringabell baseline
# Example: bash generate_nudity_datasets.sh 0 all all
# ============================================================================

set -e

# ============================================================================
# Find available GPU with most free memory
# ============================================================================
find_free_gpu() {
    # Get GPU with most free memory using nvidia-smi
    python3 -c "
import subprocess
import re

result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True)
gpus = []
for line in result.stdout.strip().split('\n'):
    idx, mem = line.split(',')
    gpus.append((int(idx.strip()), int(mem.strip())))

# Sort by free memory (descending) and get the one with most free memory
gpus.sort(key=lambda x: x[1], reverse=True)
print(gpus[0][0])
"
}

if [ $# -lt 2 ]; then
    echo "Usage: $0 <GPU|auto> <DATASET> <METHOD>"
    echo ""
    echo "GPU options:"
    echo "  0,1,2,...  - specific GPU number"
    echo "  auto       - automatically select GPU with most free memory"
    echo ""
    echo "DATASET options:"
    echo "  ringabell  - nudity-ring-a-bell.csv (79 prompts)"
    echo "  mma        - mma-diffusion-nsfw-adv-prompts.csv (1000 prompts)"
    echo "  nudity     - nudity.csv (142 prompts)"
    echo "  all        - all datasets"
    echo ""
    echo "METHOD options:"
    echo "  baseline   - SD 1.4 baseline"
    echo "  safree     - SAFREE"
    echo "  ours       - Ours 4class"
    echo "  safree_ours - SAFREE+Ours 4class"
    echo "  all        - all methods"
    exit 1
fi

GPU_ARG=$1
DATASET=$2
METHOD=${3:-all}  # Default to 'all' if not specified

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
PROMPT_DIR="${BASE_DIR}/prompts/nudity_datasets"
OUTPUT_BASE="${BASE_DIR}/outputs/nudity_datasets"

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

# Ours 4class best params: gs7.5_thr0.7-0.3_hs1.5_bgs2.0_cosine_anneal
OURS_GS=7.5
OURS_THR_START=0.7
OURS_THR_END=0.3
OURS_HS=1.5
OURS_BGS=2.0
OURS_STRATEGY="cosine_anneal"

# SAFREE+Ours best params: cgs5.0_st0.5-0.3_hs0.5_bgs2.0
SAFREE_OURS_CGS=5.0
SAFREE_OURS_ST_START=0.5
SAFREE_OURS_ST_END=0.3
SAFREE_OURS_HS=0.5
SAFREE_OURS_BGS=2.0

# SAFREE params
SAFREE_ALPHA=0.01
SVF_UP_T=10

# ============================================================================
# Extract prompts from CSV to txt (for baseline and Ours)
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

    # Use gen_safree_single.py with --category for proper negative prompt space
    python gen_safree_single.py \
        --txt "${prompt_file}" \
        --save-dir "${output_dir}" \
        --model_id "${SD_MODEL}" \
        --num-samples ${NSAMPLES} \
        --num_inference_steps ${STEPS} \
        --guidance_scale ${CFG_SCALE} \
        --seed ${SEED} \
        --device "cuda:0" \
        --category "nudity" \
        --safree \
        -svf \
        -lra \
        --sf_alpha ${SAFREE_ALPHA} \
        --re_attn_t="-1,4" \
        --up_t ${SVF_UP_T} \
        --freeu_hyp "1.0-1.0-0.9-0.2"

    # gen_safree_single.py saves to {save-dir}/generated/, move files up
    if [ -d "${output_dir}/generated" ]; then
        mv "${output_dir}/generated/"* "${output_dir}/" 2>/dev/null || true
        rmdir "${output_dir}/generated" 2>/dev/null || true
    fi

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
# Run generation for a dataset
# ============================================================================
run_dataset() {
    local dataset_key=$1
    local csv_file=$2
    local prompt_file=$3
    local dataset_name=$4

    case ${METHOD} in
        baseline)
            generate_baseline "${prompt_file}" "${OUTPUT_BASE}/${dataset_key}/baseline" "${dataset_name}"
            ;;
        safree)
            generate_safree "${prompt_file}" "${OUTPUT_BASE}/${dataset_key}/safree" "${dataset_name}"
            ;;
        ours)
            generate_ours "${prompt_file}" "${OUTPUT_BASE}/${dataset_key}/ours_4class" "${dataset_name}"
            ;;
        safree_ours)
            generate_safree_ours "${prompt_file}" "${OUTPUT_BASE}/${dataset_key}/safree_ours_4class" "${dataset_name}"
            ;;
        all)
            generate_baseline "${prompt_file}" "${OUTPUT_BASE}/${dataset_key}/baseline" "${dataset_name}"
            generate_safree "${prompt_file}" "${OUTPUT_BASE}/${dataset_key}/safree" "${dataset_name}"
            generate_ours "${prompt_file}" "${OUTPUT_BASE}/${dataset_key}/ours_4class" "${dataset_name}"
            generate_safree_ours "${prompt_file}" "${OUTPUT_BASE}/${dataset_key}/safree_ours_4class" "${dataset_name}"
            ;;
        *)
            echo "Unknown method: ${METHOD}"
            exit 1
            ;;
    esac
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
echo "Nudity Dataset Generation"
echo "=============================================="
echo "Selected GPU: ${GPU}"
echo "Dataset: ${DATASET}"
echo "Method: ${METHOD}"
echo "=============================================="

# Extract prompts for txt-based methods
extract_prompts "nudity-ring-a-bell.csv" "ringabell.txt" "sensitive prompt"
extract_prompts "mma-diffusion-nsfw-adv-prompts.csv" "mma.txt" "adv_prompt"
extract_prompts "nudity.csv" "nudity.txt" "prompt"

# Dataset info
RINGABELL_CSV="${DATASET_DIR}/nudity-ring-a-bell.csv"
RINGABELL_TXT="${PROMPT_DIR}/ringabell.txt"

MMA_CSV="${DATASET_DIR}/mma-diffusion-nsfw-adv-prompts.csv"
MMA_TXT="${PROMPT_DIR}/mma.txt"

NUDITY_CSV="${DATASET_DIR}/nudity.csv"
NUDITY_TXT="${PROMPT_DIR}/nudity.txt"

# Run generation
case ${DATASET} in
    ringabell)
        run_dataset "ringabell" "${RINGABELL_CSV}" "${RINGABELL_TXT}" "Ring-a-Bell (79 prompts)"
        ;;
    mma)
        run_dataset "mma" "${MMA_CSV}" "${MMA_TXT}" "MMA-Diffusion (1000 prompts)"
        ;;
    nudity)
        run_dataset "nudity" "${NUDITY_CSV}" "${NUDITY_TXT}" "Nudity.csv (142 prompts)"
        ;;
    all)
        run_dataset "ringabell" "${RINGABELL_CSV}" "${RINGABELL_TXT}" "Ring-a-Bell (79 prompts)"
        run_dataset "mma" "${MMA_CSV}" "${MMA_TXT}" "MMA-Diffusion (1000 prompts)"
        run_dataset "nudity" "${NUDITY_CSV}" "${NUDITY_TXT}" "Nudity.csv (142 prompts)"
        ;;
    *)
        echo "Unknown dataset: ${DATASET}"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Generation Complete!"
echo "=============================================="
echo "Output: ${OUTPUT_BASE}"
echo "=============================================="
