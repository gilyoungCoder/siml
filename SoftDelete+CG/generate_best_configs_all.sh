#!/bin/bash
# ============================================================================
# Generate Images using BEST Configurations for ALL Concepts
# ============================================================================
#
# Uses the best configurations from best_configs.txt
# Generates for both high_tox and low_tox prompts
#
# Usage:
#   ./generate_best_configs_all.sh <CONCEPT> <GPU> [TOX_LEVEL]
#   ./generate_best_configs_all.sh violence 0          # Both high and low
#   ./generate_best_configs_all.sh violence 0 high     # High toxicity only
#   ./generate_best_configs_all.sh violence 0 low      # Low toxicity only
#   ./generate_best_configs_all.sh all 0               # All concepts (sequential)
#
# Parallel execution for all concepts:
#   ./generate_best_configs_all.sh parallel 0          # GPU 0-6 for 7 concepts
#
# ============================================================================

set -u

if [ $# -lt 2 ]; then
    echo "Usage: $0 <CONCEPT|all|parallel> <GPU> [TOX_LEVEL]"
    echo ""
    echo "Concepts: nudity, violence, harassment, hate, shocking, illegal, selfharm, all, parallel"
    echo "TOX_LEVEL: high, low, or both (default: both)"
    echo ""
    echo "Examples:"
    echo "  $0 violence 0           # Violence on GPU 0, both tox levels"
    echo "  $0 violence 0 high      # Violence high_tox only"
    echo "  $0 all 0                # All concepts sequentially on GPU 0"
    echo "  $0 parallel 0           # All concepts in parallel (GPU 0-6)"
    exit 1
fi

CONCEPT=$1
START_GPU=$2
TOX_LEVEL=${3:-"both"}

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# ============================================================================
# Prompt directory
# ============================================================================
I2P_PROMPT_DIR="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/i2p"

# ============================================================================
# Best configurations from best_configs.txt
# ============================================================================
# Format: CONCEPT -> "GUIDANCE_SCALE|ST_START|ST_END|HARMFUL_SCALE|BASE_GUIDANCE|STRATEGY|CLASS_TYPE|STEP"

declare -A BEST_CONFIGS
# nudity: gs7.5_thr0.7-0.3_hs1.5_bgs2.0_cosine_anneal (4class)
BEST_CONFIGS["nudity"]="7.5|0.7|0.3|1.5|2.0|cosine_anneal|4class|17100"
# violence: gs7.5_st0.6-0.4_hs1.0_bgs2.0_cosine_anneal (13class) [BEST]
BEST_CONFIGS["violence"]="7.5|0.6|0.4|1.0|2.0|cosine_anneal|13class|28400"
# harassment: gs7.5_st0.4-0.6_hs1.5_bgs2.0_lin (9class)
BEST_CONFIGS["harassment"]="7.5|0.4|0.6|1.5|2.0|cosine_anneal|9class|24300"
# hate: gs10.0_st0.2-0.4_hs1.0_bgs1.0_cos (9class)
BEST_CONFIGS["hate"]="10.0|0.2|0.4|1.0|1.0|cosine_anneal|9class|20800"
# shocking: gs10.0_st0.7-0.3_hs1.0_bgs2.0_lin (9class)
BEST_CONFIGS["shocking"]="10.0|0.7|0.3|1.0|2.0|cosine_anneal|9class|23700"
# illegal: gs10.0_st0.3-0.5_hs1.5_bgs2.0_cos (9class)
BEST_CONFIGS["illegal"]="10.0|0.3|0.5|1.5|2.0|cosine_anneal|9class|22600"
# selfharm: gs7.5_st0.4-0.6_hs2.0_bgs1.0_lin (9class)
BEST_CONFIGS["selfharm"]="7.5|0.4|0.6|2.0|1.0|cosine_anneal|9class|20700"

# Prompt file mapping
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

# ============================================================================
# Colors
# ============================================================================
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

# ============================================================================
# Fixed parameters
# ============================================================================
SD_MODEL="CompVis/stable-diffusion-v1-4"
NSAMPLES=1
CFG_SCALE=7.5
NUM_INFERENCE_STEPS=50
SEED=1234

# ============================================================================
# Function: Run generation for a single concept
# ============================================================================
run_generation() {
    local CONCEPT=$1
    local GPU=$2
    local TOX=$3  # high or low

    export CUDA_VISIBLE_DEVICES=$GPU

    # Parse config
    IFS='|' read -r GS ST_START ST_END HS BGS STRATEGY CLASS_TYPE STEP <<< "${BEST_CONFIGS[$CONCEPT]}"

    # Determine paths based on class type
    if [ "$CLASS_TYPE" == "4class" ]; then
        CLASSIFIER_CKPT="./work_dirs/nudity_4class_safe_combined/checkpoint/step_${STEP}/classifier.pth"
        GRADCAM_STATS_DIR="./gradcam_stats/nudity_4class"
        PYTHON_SCRIPT="generate_nudity_4class_spatial_cg.py"
    elif [ "$CLASS_TYPE" == "13class" ]; then
        CLASSIFIER_CKPT="./work_dirs/violence_13class/checkpoint/step_${STEP}/classifier.pth"
        GRADCAM_STATS_DIR="./gradcam_stats/violence_13class_step${STEP}"
        PYTHON_SCRIPT="generate_violence_13class_spatial_cg.py"
    else
        # 9class
        CLASSIFIER_CKPT="./work_dirs/${CONCEPT}_9class/checkpoint/step_${STEP}/classifier.pth"
        GRADCAM_STATS_DIR="./gradcam_stats/${CONCEPT}_9class_step${STEP}"
        PYTHON_SCRIPT="generate_i2p_9class_spatial_cg.py"
    fi

    # Select prompt file
    if [ "$TOX" == "high" ]; then
        PROMPT_FILE="${I2P_PROMPT_DIR}/${PROMPT_HIGH[$CONCEPT]}"
        TOX_SUFFIX="high_tox"
    else
        PROMPT_FILE="${I2P_PROMPT_DIR}/${PROMPT_LOW[$CONCEPT]}"
        TOX_SUFFIX="low_tox"
    fi

    # Output directory (all have skip logic - 4class built-in, others via --skip_if_safe)
    OUTPUT_DIR="./scg_outputs/best_configs/${CONCEPT}_${CLASS_TYPE}_skip_ca/${TOX_SUFFIX}"

    # Validate files
    if [ ! -f "$CLASSIFIER_CKPT" ]; then
        echo -e "${RED}[ERROR] Classifier not found: $CLASSIFIER_CKPT${NC}"
        return 1
    fi

    if [ ! -f "$PROMPT_FILE" ]; then
        echo -e "${RED}[ERROR] Prompt file not found: $PROMPT_FILE${NC}"
        return 1
    fi

    # GradCAM flag
    if [ -d "$GRADCAM_STATS_DIR" ]; then
        GRADCAM_FLAG="--gradcam_stats_dir ${GRADCAM_STATS_DIR}"
    else
        GRADCAM_FLAG=""
        echo -e "${YELLOW}[WARN] GradCAM stats not found: $GRADCAM_STATS_DIR${NC}"
    fi

    echo -e "${CYAN}=============================================="
    echo -e "Generating: ${CONCEPT} (${TOX_SUFFIX})"
    echo -e "==============================================${NC}"
    echo -e "  ${GREEN}Class type:${NC}      $CLASS_TYPE"
    echo -e "  ${GREEN}Classifier:${NC}      $CLASSIFIER_CKPT"
    echo -e "  ${GREEN}Prompt file:${NC}     $PROMPT_FILE"
    echo -e "  ${GREEN}Output:${NC}          $OUTPUT_DIR"
    echo -e "  ${GREEN}GPU:${NC}             $GPU"
    echo ""
    echo -e "  ${GREEN}Best Config:${NC}"
    echo -e "    guidance_scale:     $GS"
    echo -e "    spatial_threshold:  $ST_START -> $ST_END"
    echo -e "    harmful_scale:      $HS"
    echo -e "    base_guidance:      $BGS"
    echo -e "    strategy:           $STRATEGY"
    echo ""

    mkdir -p "$OUTPUT_DIR"

    # Build command based on class type
    # NOTE: 4class has skip logic built-in (no --skip_if_safe flag needed)
    if [ "$CLASS_TYPE" == "4class" ]; then
        python $PYTHON_SCRIPT \
            "$SD_MODEL" \
            --prompt_file "$PROMPT_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --classifier_ckpt "$CLASSIFIER_CKPT" \
            $GRADCAM_FLAG \
            --nsamples $NSAMPLES \
            --cfg_scale $CFG_SCALE \
            --num_inference_steps $NUM_INFERENCE_STEPS \
            --seed $SEED \
            --guidance_scale $GS \
            --spatial_threshold_start $ST_START \
            --spatial_threshold_end $ST_END \
            --threshold_strategy $STRATEGY \
            --use_bidirectional \
            --harmful_scale $HS \
            --base_guidance_scale $BGS

    elif [ "$CLASS_TYPE" == "13class" ]; then
        python $PYTHON_SCRIPT \
            "$SD_MODEL" \
            --prompt_file "$PROMPT_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --classifier_ckpt "$CLASSIFIER_CKPT" \
            $GRADCAM_FLAG \
            --nsamples $NSAMPLES \
            --cfg_scale $CFG_SCALE \
            --num_inference_steps $NUM_INFERENCE_STEPS \
            --seed $SEED \
            --guidance_scale $GS \
            --spatial_threshold_start $ST_START \
            --spatial_threshold_end $ST_END \
            --threshold_strategy $STRATEGY \
            --use_bidirectional \
            --harmful_scale $HS \
            --base_guidance_scale $BGS \
            --gradcam_layer "encoder_model.middle_block.2" \
            --skip_if_safe

    else
        # 9class
        python $PYTHON_SCRIPT \
            "$SD_MODEL" \
            --concept "$CONCEPT" \
            --prompt_file "$PROMPT_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --classifier_ckpt "$CLASSIFIER_CKPT" \
            $GRADCAM_FLAG \
            --nsamples $NSAMPLES \
            --cfg_scale $CFG_SCALE \
            --num_inference_steps $NUM_INFERENCE_STEPS \
            --seed $SEED \
            --guidance_scale $GS \
            --spatial_threshold_start $ST_START \
            --spatial_threshold_end $ST_END \
            --threshold_strategy $STRATEGY \
            --use_bidirectional \
            --harmful_scale $HS \
            --base_guidance_scale $BGS \
            --skip_if_safe
    fi

    echo -e "${GREEN}Completed: ${CONCEPT} (${TOX_SUFFIX})${NC}"
    echo ""
}

# ============================================================================
# Run single concept
# ============================================================================
run_single_concept() {
    local CONCEPT=$1
    local GPU=$2
    local TOX_LEVEL=$3

    if [ "$TOX_LEVEL" == "both" ]; then
        run_generation "$CONCEPT" "$GPU" "high"
        run_generation "$CONCEPT" "$GPU" "low"
    elif [ "$TOX_LEVEL" == "high" ]; then
        run_generation "$CONCEPT" "$GPU" "high"
    else
        run_generation "$CONCEPT" "$GPU" "low"
    fi
}

# ============================================================================
# Main execution
# ============================================================================

ALL_CONCEPTS=("nudity" "violence" "harassment" "hate" "shocking" "illegal" "selfharm")

if [ "$CONCEPT" == "parallel" ]; then
    # Parallel execution: each concept on a different GPU
    echo -e "${GREEN}=============================================="
    echo -e "Launching ALL concepts in PARALLEL"
    echo -e "==============================================${NC}"
    echo ""

    mkdir -p logs

    for i in "${!ALL_CONCEPTS[@]}"; do
        C="${ALL_CONCEPTS[$i]}"
        GPU=$((START_GPU + i))
        echo -e "${YELLOW}Launching: ${C} on GPU ${GPU}${NC}"

        nohup bash -c "
            export CUDA_VISIBLE_DEVICES=$GPU
            cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
            source $(dirname $0)/generate_best_configs_all.sh
            run_single_concept $C $GPU both
        " > "logs/generate_best_${C}.log" 2>&1 &

        echo "  PID: $!"
        echo "  Log: logs/generate_best_${C}.log"
        echo ""
    done

    echo -e "${GREEN}=============================================="
    echo -e "All 7 concepts launched!"
    echo -e "==============================================${NC}"
    echo ""
    echo "GPU Assignments:"
    for i in "${!ALL_CONCEPTS[@]}"; do
        C="${ALL_CONCEPTS[$i]}"
        GPU=$((START_GPU + i))
        echo "  GPU $GPU: $C"
    done
    echo ""
    echo "Monitor: tail -f logs/generate_best_*.log"

elif [ "$CONCEPT" == "all" ]; then
    # Sequential execution
    echo -e "${GREEN}Running ALL concepts sequentially on GPU ${START_GPU}${NC}"
    for C in "${ALL_CONCEPTS[@]}"; do
        run_single_concept "$C" "$START_GPU" "$TOX_LEVEL"
    done

else
    # Single concept
    if [[ ! " ${ALL_CONCEPTS[*]} " =~ " ${CONCEPT} " ]]; then
        echo -e "${RED}Error: Invalid concept '${CONCEPT}'${NC}"
        echo "Available concepts: ${ALL_CONCEPTS[*]}"
        exit 1
    fi

    run_single_concept "$CONCEPT" "$START_GPU" "$TOX_LEVEL"
fi

echo -e "${GREEN}Done!${NC}"
