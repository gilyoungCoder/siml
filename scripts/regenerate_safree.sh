#!/bin/bash
# ============================================================================
# Regenerate SAFREE images using gen_safree_single.py
# (with proper --category flag for automatic negative prompt space)
#
# Datasets:
#   1. COCO (10k prompts)
#   2. Nudity datasets: i2p (142), mma (1000), ringabell (79)
#
# Usage: bash regenerate_safree.sh <GPU> <DATASET>
# Example: bash regenerate_safree.sh 0 all
#          bash regenerate_safree.sh 0 coco
#          bash regenerate_safree.sh 0 nudity
# ============================================================================

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <GPU> <DATASET>"
    echo ""
    echo "DATASET options:"
    echo "  coco     - COCO 10k prompts"
    echo "  nudity   - All nudity datasets (i2p, mma, ringabell)"
    echo "  i2p      - i2p only (142 prompts)"
    echo "  mma      - mma only (1000 prompts)"
    echo "  ringabell - ringabell only (79 prompts)"
    echo "  all      - everything"
    exit 1
fi

GPU=$1
DATASET=$2

export CUDA_VISIBLE_DEVICES=${GPU}

# ============================================================================
# Paths
# ============================================================================
BASE_DIR="/mnt/home/yhgil99/unlearning"
SAFREE_DIR="${BASE_DIR}/SAFREE"
PROMPT_DIR="${BASE_DIR}/prompts"
OUTPUT_DIR="${BASE_DIR}/outputs"

SD_MODEL="CompVis/stable-diffusion-v1-4"

# Generation params
SEED=42
STEPS=50
CFG_SCALE=7.5
NSAMPLES=1

# SAFREE params
SAFREE_ALPHA=0.01
SVF_UP_T=10

# ============================================================================
# Generate function using gen_safree_single.py
# ============================================================================
generate_safree() {
    local prompt_file=$1
    local output_dir=$2
    local name=$3
    local category=$4  # nudity, violence, etc.

    echo ""
    echo "=============================================="
    echo "[SAFREE] ${name}"
    echo "=============================================="
    echo "Prompt: ${prompt_file}"
    echo "Output: ${output_dir}"
    echo "Category: ${category}"
    echo "=============================================="

    mkdir -p "${output_dir}"
    cd "${SAFREE_DIR}"

    python gen_safree_single.py \
        --txt "${prompt_file}" \
        --save-dir "${output_dir}" \
        --model_id "${SD_MODEL}" \
        --num-samples ${NSAMPLES} \
        --num_inference_steps ${STEPS} \
        --guidance_scale ${CFG_SCALE} \
        --seed ${SEED} \
        --device "cuda:0" \
        --category "${category}" \
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

# ============================================================================
# Main
# ============================================================================
echo "=============================================="
echo "SAFREE Regeneration (using gen_safree_single.py)"
echo "=============================================="
echo "GPU: ${GPU}"
echo "Dataset: ${DATASET}"
echo "=============================================="

case ${DATASET} in
    coco)
        generate_safree \
            "${PROMPT_DIR}/coco/coco_10k.txt" \
            "${OUTPUT_DIR}/coco/safree" \
            "COCO 10k" \
            "nudity"
        ;;
    i2p)
        generate_safree \
            "${PROMPT_DIR}/nudity_datasets/nudity.txt" \
            "${OUTPUT_DIR}/nudity_datasets/i2p/safree" \
            "I2P (142 prompts)" \
            "nudity"
        ;;
    mma)
        generate_safree \
            "${PROMPT_DIR}/nudity_datasets/mma.txt" \
            "${OUTPUT_DIR}/nudity_datasets/mma/safree" \
            "MMA (1000 prompts)" \
            "nudity"
        ;;
    ringabell)
        generate_safree \
            "${PROMPT_DIR}/nudity_datasets/ringabell.txt" \
            "${OUTPUT_DIR}/nudity_datasets/ringabell/safree" \
            "Ring-a-Bell (79 prompts)" \
            "nudity"
        ;;
    nudity)
        generate_safree \
            "${PROMPT_DIR}/nudity_datasets/nudity.txt" \
            "${OUTPUT_DIR}/nudity_datasets/i2p/safree" \
            "I2P (142 prompts)" \
            "nudity"
        generate_safree \
            "${PROMPT_DIR}/nudity_datasets/mma.txt" \
            "${OUTPUT_DIR}/nudity_datasets/mma/safree" \
            "MMA (1000 prompts)" \
            "nudity"
        generate_safree \
            "${PROMPT_DIR}/nudity_datasets/ringabell.txt" \
            "${OUTPUT_DIR}/nudity_datasets/ringabell/safree" \
            "Ring-a-Bell (79 prompts)" \
            "nudity"
        ;;
    all)
        generate_safree \
            "${PROMPT_DIR}/coco/coco_10k.txt" \
            "${OUTPUT_DIR}/coco/safree" \
            "COCO 10k" \
            "nudity"
        generate_safree \
            "${PROMPT_DIR}/nudity_datasets/nudity.txt" \
            "${OUTPUT_DIR}/nudity_datasets/i2p/safree" \
            "I2P (142 prompts)" \
            "nudity"
        generate_safree \
            "${PROMPT_DIR}/nudity_datasets/mma.txt" \
            "${OUTPUT_DIR}/nudity_datasets/mma/safree" \
            "MMA (1000 prompts)" \
            "nudity"
        generate_safree \
            "${PROMPT_DIR}/nudity_datasets/ringabell.txt" \
            "${OUTPUT_DIR}/nudity_datasets/ringabell/safree" \
            "Ring-a-Bell (79 prompts)" \
            "nudity"
        ;;
    *)
        echo "Unknown dataset: ${DATASET}"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "SAFREE Regeneration Complete!"
echo "=============================================="
