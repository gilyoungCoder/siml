#!/bin/bash
# ============================================================================
# Violence 13-Class Top 6 Configurations Inference
#
# Runs inference with the 6 best parameter settings from grid search
# Using violence_top50.txt prompts from I2P benchmark
#
# Usage:
#   ./run_violence_top6_inference.sh [GPU]
#   ./run_violence_top6_inference.sh 0
# ============================================================================

set -u

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

ROOT_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG"
cd "$ROOT_DIR" || exit 1

# ============================================================================
# Colors
# ============================================================================
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================================
# Paths
# ============================================================================
STEP=28400
CKPT_PATH="CompVis/stable-diffusion-v1-4"
CLASSIFIER_PATH="./work_dirs/violence_13class/checkpoint/step_${STEP}/classifier.pth"
GRADCAM_STATS_DIR="./gradcam_stats/violence_13class_step${STEP}"

# I2P violence prompts
PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/i2p/violence_top50.txt"

OUTPUT_BASE_DIR="./scg_outputs/violence_i2p_top6_configs"

# ============================================================================
# Fixed Params
# ============================================================================
NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
NSAMPLES=1
SEED=1234

# ============================================================================
# Helpers
# ============================================================================
print_header()  { echo -e "\n${CYAN}===== $1 =====${NC}"; }
print_kv()      { printf "  ${GREEN}%-22s${NC} : %s\n" "$1" "$2"; }
print_info()    { echo -e "${YELLOW}[INFO]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================================
# Validate
# ============================================================================
if [ ! -f "$CLASSIFIER_PATH" ]; then
  print_error "Classifier checkpoint not found: $CLASSIFIER_PATH"
  exit 1
fi

if [ ! -f "$PROMPT_FILE" ]; then
  print_error "Prompt file not found: $PROMPT_FILE"
  exit 1
fi

# ============================================================================
# Show Config
# ============================================================================
print_header "Violence 13-Class Top 6 Configurations Inference"
print_kv "STEP" "$STEP"
print_kv "GPU" "$GPU"
print_kv "CLASSIFIER" "$CLASSIFIER_PATH"
print_kv "GRADCAM_STATS" "$GRADCAM_STATS_DIR"
print_kv "PROMPT_FILE" "$PROMPT_FILE"
print_kv "OUTPUT_BASE" "$OUTPUT_BASE_DIR"
echo ""

# Check GradCAM stats
if [ -d "$GRADCAM_STATS_DIR" ]; then
    GRADCAM_FLAG="--gradcam_stats_dir ${GRADCAM_STATS_DIR}"
    print_info "GradCAM stats directory found"
else
    GRADCAM_FLAG=""
    print_info "GradCAM stats not found, will use per-image normalization"
fi

# ============================================================================
# Top 6 Configurations (from grid search results)
# ============================================================================
# Config format: "GS,ST_START,ST_END,HS,BGS,TS,SKIP"
#
# From grid_search_violence_13class_step28400 (no skip):
#   1. gs10.0_st0.5-0.3_hs1.0_bgs2.0_cosine_anneal (80%)
#   2. gs10.0_st0.5-0.3_hs1.0_bgs2.0_linear_decrease (78%)
#   3. gs10.0_st0.3-0.7_hs1.0_bgs1.0_cosine_anneal (72%)
#
# From grid_search_violence_13class_step28400_skip:
#   4. gs7.5_st0.6-0.4_hs1.0_bgs2.0_linear_decrease (74%)
#   5. gs12.5_st0.7-0.3_hs1.0_bgs2.0_linear_decrease (72%)
#   6. gs7.5_st0.5-0.5_hs1.0_bgs2.0_cosine_anneal (70%)

CONFIGS=(
  "10.0,0.5,0.3,1.0,2.0,cosine_anneal,noskip"
  "10.0,0.5,0.3,1.0,2.0,linear_decrease,noskip"
  "10.0,0.3,0.7,1.0,1.0,cosine_anneal,noskip"
  "7.5,0.6,0.4,1.0,2.0,linear_decrease,skip"
  "12.5,0.7,0.3,1.0,2.0,linear_decrease,skip"
  "7.5,0.5,0.5,1.0,2.0,cosine_anneal,skip"
)

total_experiments=${#CONFIGS[@]}
print_kv "TOTAL_EXPERIMENTS" "$total_experiments"
echo ""

# ============================================================================
# Run Inference
# ============================================================================
current_experiment=0

for config in "${CONFIGS[@]}"; do
    IFS=',' read -r GS ST_START ST_END HS BGS TS SKIP_MODE <<< "$config"
    current_experiment=$((current_experiment + 1))

    # Determine output directory name and skip flag
    if [ "$SKIP_MODE" == "skip" ]; then
        SKIP_FLAG="--skip_if_safe"
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/gs${GS}_st${ST_START}-${ST_END}_hs${HS}_bgs${BGS}_${TS}_skip"
    else
        SKIP_FLAG=""
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/gs${GS}_st${ST_START}-${ST_END}_hs${HS}_bgs${BGS}_${TS}"
    fi

    mkdir -p "$OUTPUT_DIR"

    print_header "Experiment ${current_experiment}/${total_experiments}"
    print_kv "GUIDANCE_SCALE" "$GS"
    print_kv "SPATIAL_THRESHOLD" "${ST_START} -> ${ST_END}"
    print_kv "HARMFUL_SCALE" "$HS"
    print_kv "BASE_GUIDANCE_SCALE" "$BGS"
    print_kv "THRESHOLD_STRATEGY" "$TS"
    print_kv "SKIP_MODE" "$SKIP_MODE"
    print_kv "OUTPUT_DIR" "$OUTPUT_DIR"
    echo ""

    python generate_violence_13class_spatial_cg.py \
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
        $SKIP_FLAG

    ret=$?
    if [ $ret -ne 0 ]; then
        print_error "Experiment failed (exit code=$ret)"
        exit $ret
    fi

    print_info "Completed: $OUTPUT_DIR"
done

print_header "ALL EXPERIMENTS COMPLETE"
echo "Total: $total_experiments experiments"
echo "Output: $OUTPUT_BASE_DIR"
echo ""
echo "Generated folders:"
for config in "${CONFIGS[@]}"; do
    IFS=',' read -r GS ST_START ST_END HS BGS TS SKIP_MODE <<< "$config"
    if [ "$SKIP_MODE" == "skip" ]; then
        echo "  - gs${GS}_st${ST_START}-${ST_END}_hs${HS}_bgs${BGS}_${TS}_skip"
    else
        echo "  - gs${GS}_st${ST_START}-${ST_END}_hs${HS}_bgs${BGS}_${TS}"
    fi
done
