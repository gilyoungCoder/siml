#!/bin/bash
# ============================================================================
# Grid Search for Violence 13-Class Dynamic Spatial Classifier Guidance
# ============================================================================
#
# Usage:
#   ./grid_search_violence_13class_spatial_cg.sh <STEP> <GPU> [SKIP_MODE]
#   ./grid_search_violence_13class_spatial_cg.sh 20000 0
#   ./grid_search_violence_13class_spatial_cg.sh 20000 0 skip   # with --skip_if_safe
# ============================================================================

set -u   # undefined variable 방지 (set -e는 사용 안 함)

# ============================================================================
# STEP / GPU / SKIP_MODE (args or default)
# ============================================================================
STEP=${1:-28400}
GPU=${2:-7}
SKIP_MODE=${3:-""}  # Optional: "skip" to enable --skip_if_safe

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
CKPT_PATH="CompVis/stable-diffusion-v1-4"
CLASSIFIER_PATH="./work_dirs/violence_13class/checkpoint/step_${STEP}/classifier.pth"
GRADCAM_STATS_DIR="./gradcam_stats/violence_13class_step${STEP}"

# Handle skip_if_safe mode
if [ "$SKIP_MODE" == "skip" ]; then
    SKIP_FLAG="--skip_if_safe"
    OUTPUT_BASE_DIR="./scg_outputs/grid_search_violence_13class_step${STEP}_skip"
    echo -e "${CYAN}Mode: SKIP if safe/benign/artifact${NC}"
else
    SKIP_FLAG=""
    OUTPUT_BASE_DIR="./scg_outputs/grid_search_violence_13class_step${STEP}"
    echo -e "${CYAN}Mode: ALWAYS guide (no skip)${NC}"
fi

PROMPT_FILE="./prompts/violence_50.txt"

# ============================================================================
# Fixed Params
# ============================================================================
NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
NSAMPLES=1
SEED=1234

# ============================================================================
# Grid Params
# ============================================================================
GUIDANCE_SCALES=(7.5 10.0 12.5)
SPATIAL_THRESHOLDS=(
  "0.7,0.3"
  "0.6,0.4"
  "0.5,0.3"
  "0.5,0.5"
  "0.4,0.2"
  "0.3,0.3"
  "0.3,0.7"
  "0.4,0.6"
  "0.3,0.5"
  "0.2,0.4"
)
HARMFUL_SCALES=(1.0 1.5 2.0)
BASE_GUIDANCE_SCALES=(1.0 2.0)
THRESHOLD_STRATEGIES=("cosine_anneal" "linear_decrease")

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
print_header "Violence 13-Class Spatial CG Grid Search"
print_kv "STEP" "$STEP"
print_kv "GPU" "$GPU"
print_kv "CLASSIFIER" "$CLASSIFIER_PATH"
print_kv "GRADCAM_STATS" "$GRADCAM_STATS_DIR"
print_kv "PROMPT_FILE" "$PROMPT_FILE"
print_kv "OUTPUT_BASE" "$OUTPUT_BASE_DIR"
print_kv "SKIP_MODE" "${SKIP_MODE:-none}"

# Count total
total_experiments=0
for GS in "${GUIDANCE_SCALES[@]}"; do
  for ST in "${SPATIAL_THRESHOLDS[@]}"; do
    for HS in "${HARMFUL_SCALES[@]}"; do
      for BGS in "${BASE_GUIDANCE_SCALES[@]}"; do
        for TS in "${THRESHOLD_STRATEGIES[@]}"; do
          total_experiments=$((total_experiments + 1))
        done
      done
    done
  done
done
print_kv "TOTAL_EXPERIMENTS" "$total_experiments"
echo ""

# ============================================================================
# Check GradCAM stats
# ============================================================================
if [ -d "$GRADCAM_STATS_DIR" ]; then
    GRADCAM_FLAG="--gradcam_stats_dir ${GRADCAM_STATS_DIR}"
    print_info "GradCAM stats directory found"
else
    GRADCAM_FLAG=""
    print_info "GradCAM stats not found, will use per-image normalization"
fi

# ============================================================================
# Run Grid
# ============================================================================
current_experiment=0

for GS in "${GUIDANCE_SCALES[@]}"; do
  for ST in "${SPATIAL_THRESHOLDS[@]}"; do
    IFS=',' read -r ST_START ST_END <<< "$ST"

    for HS in "${HARMFUL_SCALES[@]}"; do
      for BGS in "${BASE_GUIDANCE_SCALES[@]}"; do
        for TS in "${THRESHOLD_STRATEGIES[@]}"; do
          current_experiment=$((current_experiment + 1))

          OUTPUT_DIR="${OUTPUT_BASE_DIR}/gs${GS}_st${ST_START}-${ST_END}_hs${HS}_bgs${BGS}_${TS}"
          mkdir -p "$OUTPUT_DIR"

          print_header "Experiment ${current_experiment}/${total_experiments}"
          print_kv "GUIDANCE_SCALE" "$GS"
          print_kv "SPATIAL_THRESHOLD" "${ST_START} -> ${ST_END}"
          print_kv "HARMFUL_SCALE" "$HS"
          print_kv "BASE_GUIDANCE_SCALE" "$BGS"
          print_kv "THRESHOLD_STRATEGY" "$TS"
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
      done
    done
  done
done

print_header "ALL EXPERIMENTS COMPLETE"
echo "Total: $total_experiments experiments"
echo "Output: $OUTPUT_BASE_DIR"
