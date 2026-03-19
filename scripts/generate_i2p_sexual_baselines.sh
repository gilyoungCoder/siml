#!/bin/bash
# ============================================================================
# Generate SAFREE and SD Baseline for I2P Sexual Data (Full CSV)
# ============================================================================
#
# Usage:
#   ./scripts/generate_i2p_sexual_baselines.sh <METHOD> <GPU>
#   ./scripts/generate_i2p_sexual_baselines.sh baseline 0   # SD baseline
#   ./scripts/generate_i2p_sexual_baselines.sh safree 0     # SAFREE
#   ./scripts/generate_i2p_sexual_baselines.sh all 0        # Both methods
#
# ============================================================================

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <METHOD> <GPU>"
    echo ""
    echo "METHOD: baseline, safree, all"
    exit 1
fi

METHOD=$1
GPU=$2

export CUDA_VISIBLE_DEVICES=$GPU

# ============================================================================
# Paths
# ============================================================================
BASE_DIR="/mnt/home/yhgil99/unlearning"
PROMPT_CSV="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/i2p/sexual.csv"
OUTPUT_BASE="${BASE_DIR}/SoftDelete+CG/scg_outputs/baselines_i2p_sexual"

SD_MODEL="CompVis/stable-diffusion-v1-4"

# Generation params
STEPS=50
CFG_SCALE=7.5
NSAMPLES=1

# SAFREE params
SAFREE_ALPHA=0.01
SVF_UP_T=10

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================================
# Generate SD Baseline
# ============================================================================
generate_baseline() {
    local output_dir="${OUTPUT_BASE}/sd_baseline"

    echo -e "${CYAN}[SD Baseline] sexual (full CSV)${NC}"
    echo "  Prompt: $PROMPT_CSV"
    echo "  Output: $output_dir"

    mkdir -p "$output_dir"
    cd "/mnt/home/yhgil99/guided2-safe-diffusion"

    python generate.py \
        --pretrained_model_name_or_path "${SD_MODEL}" \
        --image_dir "${output_dir}" \
        --prompt_path "${PROMPT_CSV}" \
        --num_images_per_prompt ${NSAMPLES} \
        --device "cuda:0"

    echo -e "${GREEN}Done: ${output_dir}${NC}"
}

# ============================================================================
# Generate SAFREE
# ============================================================================
generate_safree() {
    local output_dir="${OUTPUT_BASE}/safree"

    echo -e "${CYAN}[SAFREE] sexual (full CSV)${NC}"
    echo "  Prompt: $PROMPT_CSV"
    echo "  Output: $output_dir"

    mkdir -p "$output_dir"
    cd "${BASE_DIR}/SAFREE"

    python gen_safree_i2p_concepts.py \
        --prompt_file "${PROMPT_CSV}" \
        --concepts "sexual" \
        --outdir "${output_dir}" \
        --no_concept_subdir \
        --model_id "${SD_MODEL}" \
        --num_images ${NSAMPLES} \
        --steps ${STEPS} \
        --guidance ${CFG_SCALE} \
        --device "cuda:0" \
        --safree \
        --svf \
        --lra \
        --sf_alpha ${SAFREE_ALPHA} \
        --re_attn_t="-1,4" \
        --up_t ${SVF_UP_T} \
        --freeu_hyp "1.0-1.0-0.9-0.2"

    echo -e "${GREEN}Done: ${output_dir}${NC}"
}

# ============================================================================
# Main
# ============================================================================
echo -e "${GREEN}=============================================="
echo -e "I2P Sexual Baseline Generation (Full CSV)"
echo -e "==============================================${NC}"
echo "Method: $METHOD"
echo "GPU: $GPU"
echo "Prompt CSV: $PROMPT_CSV"
echo "Total prompts: $(($(wc -l < "$PROMPT_CSV") - 1))"
echo ""

case $METHOD in
    baseline)
        generate_baseline
        ;;
    safree)
        generate_safree
        ;;
    all)
        generate_baseline
        generate_safree
        ;;
    *)
        echo -e "${YELLOW}Unknown method: $METHOD${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}=============================================="
echo -e "Generation Complete!"
echo -e "==============================================${NC}"
echo "Output: ${OUTPUT_BASE}"
