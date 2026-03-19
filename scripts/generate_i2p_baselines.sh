#!/bin/bash
# ============================================================================
# Generate SAFREE and SD Baseline for ALL I2P Concepts (high_tox + low_tox)
# ============================================================================
#
# Usage:
#   ./scripts/generate_i2p_baselines.sh <METHOD> <GPU> [CONCEPT]
#   ./scripts/generate_i2p_baselines.sh baseline 0 all      # SD baseline all concepts
#   ./scripts/generate_i2p_baselines.sh safree 0 all        # SAFREE all concepts
#   ./scripts/generate_i2p_baselines.sh baseline 0 nudity   # SD baseline nudity only
#   ./scripts/generate_i2p_baselines.sh all 0 all           # Both methods, all concepts
#
# ============================================================================

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <METHOD> <GPU> [CONCEPT]"
    echo ""
    echo "METHOD: baseline, safree, all"
    echo "CONCEPT: nudity, violence, harassment, hate, shocking, illegal, selfharm, all"
    exit 1
fi

METHOD=$1
GPU=$2
CONCEPT=${3:-all}

export CUDA_VISIBLE_DEVICES=$GPU

# ============================================================================
# Paths
# ============================================================================
BASE_DIR="/mnt/home/yhgil99/unlearning"
I2P_PROMPT_DIR="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/i2p"
OUTPUT_BASE="${BASE_DIR}/SoftDelete+CG/scg_outputs/baselines_i2p"

SD_MODEL="CompVis/stable-diffusion-v1-4"

# Generation params
SEED=1234
STEPS=50
CFG_SCALE=7.5
NSAMPLES=1

# SAFREE params
SAFREE_ALPHA=0.01
SVF_UP_T=10

# ============================================================================
# Prompt file mapping
# ============================================================================
declare -A PROMPT_HIGH
PROMPT_HIGH["nudity"]="sexual_high_tox.txt"
PROMPT_HIGH["violence"]="violence_high_tox.txt"
PROMPT_HIGH["harassment"]="harassment_high_tox.txt"
PROMPT_HIGH["hate"]="hate_high_tox.txt"
PROMPT_HIGH["shocking"]="shocking_high_tox.txt"
PROMPT_HIGH["illegal"]="illegal_activity_high_tox.txt"
PROMPT_HIGH["selfharm"]="self-harm_high_tox.txt"

declare -A PROMPT_LOW
PROMPT_LOW["nudity"]="sexual_low_tox.txt"
PROMPT_LOW["violence"]="violence_low_tox.txt"
PROMPT_LOW["harassment"]="harassment_low_tox.txt"
PROMPT_LOW["hate"]="hate_low_tox.txt"
PROMPT_LOW["shocking"]="shocking_low_tox.txt"
PROMPT_LOW["illegal"]="illegal_activity_low_tox.txt"
PROMPT_LOW["selfharm"]="self-harm_low_tox.txt"

# SAFREE category mapping (must match CONCEPT_KEYWORDS in gen_safree_i2p_concepts.py)
declare -A SAFREE_CATEGORY
SAFREE_CATEGORY["nudity"]="sexual"
SAFREE_CATEGORY["violence"]="violence"
SAFREE_CATEGORY["harassment"]="harassment"
SAFREE_CATEGORY["hate"]="hate"
SAFREE_CATEGORY["shocking"]="shocking"
SAFREE_CATEGORY["illegal"]="illegal activity"
SAFREE_CATEGORY["selfharm"]="self-harm"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================================
# Generate SD Baseline
# ============================================================================
generate_baseline() {
    local concept=$1
    local tox=$2
    
    if [ "$tox" == "high" ]; then
        local prompt_file="${I2P_PROMPT_DIR}/${PROMPT_HIGH[$concept]}"
        local output_dir="${OUTPUT_BASE}/sd_baseline/${concept}/high_tox"
    else
        local prompt_file="${I2P_PROMPT_DIR}/${PROMPT_LOW[$concept]}"
        local output_dir="${OUTPUT_BASE}/sd_baseline/${concept}/low_tox"
    fi
    
    if [ ! -f "$prompt_file" ]; then
        echo -e "${YELLOW}[SKIP] Prompt not found: $prompt_file${NC}"
        return
    fi
    
    echo -e "${CYAN}[SD Baseline] ${concept} (${tox}_tox)${NC}"
    echo "  Prompt: $prompt_file"
    echo "  Output: $output_dir"
    
    mkdir -p "$output_dir"
    cd "/mnt/home/yhgil99/guided2-safe-diffusion"
    
    python generate.py \
        --pretrained_model_name_or_path "${SD_MODEL}" \
        --image_dir "${output_dir}" \
        --prompt_path "${prompt_file}" \
        --num_images_per_prompt ${NSAMPLES} \
        --seed ${SEED} \
        --device "cuda:0"
    
    echo -e "${GREEN}Done: ${output_dir}${NC}"
}

# ============================================================================
# Generate SAFREE (using gen_safree_i2p_concepts.py)
# ============================================================================
generate_safree() {
    local concept=$1
    local tox=$2

    if [ "$tox" == "high" ]; then
        local prompt_file="${I2P_PROMPT_DIR}/${PROMPT_HIGH[$concept]}"
        local output_dir="${OUTPUT_BASE}/safree/${concept}/high_tox"
    else
        local prompt_file="${I2P_PROMPT_DIR}/${PROMPT_LOW[$concept]}"
        local output_dir="${OUTPUT_BASE}/safree/${concept}/low_tox"
    fi

    local category="${SAFREE_CATEGORY[$concept]}"

    if [ ! -f "$prompt_file" ]; then
        echo -e "${YELLOW}[SKIP] Prompt not found: $prompt_file${NC}"
        return
    fi

    echo -e "${CYAN}[SAFREE] ${concept} (${tox}_tox)${NC}"
    echo "  Prompt: $prompt_file"
    echo "  Output: $output_dir"
    echo "  Category: $category"

    mkdir -p "$output_dir"
    cd "${BASE_DIR}/SAFREE"

    python gen_safree_i2p_concepts.py \
        --prompt_file "${prompt_file}" \
        --concepts "${category}" \
        --outdir "${output_dir}" \
        --no_concept_subdir \
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

    echo -e "${GREEN}Done: ${output_dir}${NC}"
}

# ============================================================================
# Run for a concept
# ============================================================================
run_concept() {
    local concept=$1
    
    echo -e "\n${GREEN}=============================================="
    echo -e "Processing: ${concept^^}"
    echo -e "==============================================${NC}"
    
    for tox in "high" "low"; do
        case $METHOD in
            baseline)
                generate_baseline "$concept" "$tox"
                ;;
            safree)
                generate_safree "$concept" "$tox"
                ;;
            all)
                generate_baseline "$concept" "$tox"
                generate_safree "$concept" "$tox"
                ;;
        esac
    done
}

# ============================================================================
# Main
# ============================================================================
echo -e "${GREEN}=============================================="
echo -e "I2P Baseline Generation"
echo -e "==============================================${NC}"
echo "Method: $METHOD"
echo "GPU: $GPU"
echo "Concept: $CONCEPT"
echo ""

ALL_CONCEPTS=("nudity" "violence" "harassment" "hate" "shocking" "illegal" "selfharm")

if [ "$CONCEPT" == "all" ]; then
    for c in "${ALL_CONCEPTS[@]}"; do
        run_concept "$c"
    done
else
    if [[ ! " ${ALL_CONCEPTS[*]} " =~ " ${CONCEPT} " ]]; then
        echo -e "${YELLOW}Unknown concept: $CONCEPT${NC}"
        exit 1
    fi
    run_concept "$CONCEPT"
fi

echo -e "\n${GREEN}=============================================="
echo -e "Generation Complete!"
echo -e "==============================================${NC}"
echo "Output: ${OUTPUT_BASE}"
