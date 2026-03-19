#!/bin/bash
# ============================================================================
# Violence 13-Class Top 6 - Parallel Execution on GPU 1-6
#
# Usage:
#   ./run_violence_top6_parallel.sh
# ============================================================================

ROOT_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG"
cd "$ROOT_DIR" || exit 1

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Paths
STEP=28400
CKPT_PATH="CompVis/stable-diffusion-v1-4"
CLASSIFIER_PATH="./work_dirs/violence_13class/checkpoint/step_${STEP}/classifier.pth"
GRADCAM_STATS_DIR="./gradcam_stats/violence_13class_step${STEP}"
PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/i2p/violence_top50.txt"
OUTPUT_BASE_DIR="./scg_outputs/violence_i2p_top6_configs"

# Fixed Params
NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
NSAMPLES=1
SEED=1234

# GradCAM flag
if [ -d "$GRADCAM_STATS_DIR" ]; then
    GRADCAM_FLAG="--gradcam_stats_dir ${GRADCAM_STATS_DIR}"
else
    GRADCAM_FLAG=""
fi

echo -e "${CYAN}===== Violence Top 6 Parallel Execution (GPU 1-6) =====${NC}"
echo ""

# Function to run single config
run_config() {
    local GPU=$1
    local GS=$2
    local ST_START=$3
    local ST_END=$4
    local HS=$5
    local BGS=$6
    local TS=$7
    local SKIP_FLAG=$8

    if [ "$SKIP_FLAG" == "skip" ]; then
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/gs${GS}_st${ST_START}-${ST_END}_hs${HS}_bgs${BGS}_${TS}_skip"
        SKIP_ARG="--skip_if_safe"
    else
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/gs${GS}_st${ST_START}-${ST_END}_hs${HS}_bgs${BGS}_${TS}"
        SKIP_ARG=""
    fi

    mkdir -p "$OUTPUT_DIR"

    echo -e "${GREEN}[GPU $GPU]${NC} Starting: gs${GS}_st${ST_START}-${ST_END}_hs${HS}_bgs${BGS}_${TS} (skip=$SKIP_FLAG)"

    CUDA_VISIBLE_DEVICES=$GPU python generate_violence_13class_spatial_cg.py \
        "$CKPT_PATH" \
        --prompt_file "$PROMPT_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --nsamples $NSAMPLES \
        --cfg_scale $CFG_SCALE \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --seed $SEED \
        --classifier_ckpt "$CLASSIFIER_PATH" \
        $GRADCAM_FLAG \
        --guidance_scale $GS \
        --spatial_threshold_start $ST_START \
        --spatial_threshold_end $ST_END \
        --threshold_strategy $TS \
        --harmful_scale $HS \
        --base_guidance_scale $BGS \
        --use_bidirectional \
        --gradcam_layer "encoder_model.middle_block.2" \
        $SKIP_ARG \
        > "${OUTPUT_DIR}/log.txt" 2>&1

    echo -e "${YELLOW}[GPU $GPU]${NC} Done: $OUTPUT_DIR"
}

# Run all 6 configs in parallel on GPU 1-6
# Config 1: gs10.0_st0.5-0.3_hs1.0_bgs2.0_cosine_anneal (80%) - noskip
run_config 1 10.0 0.5 0.3 1.0 2.0 cosine_anneal noskip &

# Config 2: gs10.0_st0.5-0.3_hs1.0_bgs2.0_linear_decrease (78%) - noskip
run_config 2 10.0 0.5 0.3 1.0 2.0 linear_decrease noskip &

# Config 3: gs10.0_st0.3-0.7_hs1.0_bgs1.0_cosine_anneal (72%) - noskip
run_config 3 10.0 0.3 0.7 1.0 1.0 cosine_anneal noskip &

# Config 4: gs7.5_st0.6-0.4_hs1.0_bgs2.0_linear_decrease (74%) - skip
run_config 4 7.5 0.6 0.4 1.0 2.0 linear_decrease skip &

# Config 5: gs12.5_st0.7-0.3_hs1.0_bgs2.0_linear_decrease (72%) - skip
run_config 5 12.5 0.7 0.3 1.0 2.0 linear_decrease skip &

# Config 6: gs7.5_st0.5-0.5_hs1.0_bgs2.0_cosine_anneal (70%) - skip
run_config 6 7.5 0.5 0.5 1.0 2.0 cosine_anneal skip &

echo ""
echo -e "${CYAN}All 6 jobs launched on GPU 1-6. Waiting for completion...${NC}"
echo ""

wait

echo ""
echo -e "${GREEN}===== ALL COMPLETE =====${NC}"
echo "Output: $OUTPUT_BASE_DIR"
ls -la "$OUTPUT_BASE_DIR"
