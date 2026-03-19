#!/bin/bash
# ============================================================================
# Grid Search for Violence 9-Class Dynamic Spatial Classifier Guidance
# ============================================================================
#
# Usage:
#   ./grid_search_violence_9class_spatial_cg.sh <STEP> <GPU> [SKIP_MODE]
#   ./grid_search_violence_9class_spatial_cg.sh 25400 0
#   ./grid_search_violence_9class_spatial_cg.sh 25400 0 skip   # with --skip_if_safe
# ============================================================================

set -u   # undefined variable 방지 (set -e는 사용 안 함)

# ============================================================================
# STEP / GPU / SKIP_MODE (args or default)
# ============================================================================
STEP=${1:-25400}
GPU=${2:-6}
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
CLASSIFIER_PATH="./work_dirs/violence_9class/checkpoint/step_${STEP}/classifier.pth"
GRADCAM_STATS_DIR="./gradcam_stats_violence_9class_step${STEP}"

# Handle skip_if_safe mode
if [ "$SKIP_MODE" == "skip" ]; then
    SKIP_FLAG="--skip_if_safe"
    OUTPUT_BASE_DIR="./scg_outputs/grid_search_violence_9class_step${STEP}_skip"
    echo -e "${CYAN}Mode: SKIP if safe/benign${NC}"
else
    SKIP_FLAG=""
    OUTPUT_BASE_DIR="./scg_outputs/grid_search_violence_9class_step${STEP}_aug"
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
  "0.6,0.4"
  "0.5,0.3"
  "0.4,0.6"
  "0.3,0.5"
)
HARMFUL_SCALES=(1.0 1.5 2.0)
BASE_GUIDANCE_SCALES=(1.0 0.0)

# ============================================================================
# Utils
# ============================================================================
print_header () {
  echo -e "${GREEN}============================================================${NC}"
  echo -e "${GREEN}$1${NC}"
  echo -e "${GREEN}============================================================${NC}"
}

print_info () {
  echo -e "${YELLOW}$1${NC}"
}

print_error () {
  echo -e "${RED}[ERROR] $1${NC}"
}

print_kv () {
  echo -e "${CYAN}$1:${NC} $2"
}

# ============================================================================
# Pre-flight Checks
# ============================================================================
print_header "Violence 9-Class Dynamic Spatial CG - Grid Search"

print_kv "STEP" "$STEP"
print_kv "GPU" "$GPU"
print_kv "Classifier" "$CLASSIFIER_PATH"
print_kv "GradCAM Stats Dir" "$GRADCAM_STATS_DIR"
print_kv "Output Base" "$OUTPUT_BASE_DIR"
print_kv "Prompt File" "$PROMPT_FILE"
echo ""

if [ ! -f "generate_violence_9class_spatial_cg.py" ]; then
  print_error "generate_violence_9class_spatial_cg.py not found"
  exit 1
fi

if [ ! -f "$CLASSIFIER_PATH" ]; then
  print_error "Classifier not found: $CLASSIFIER_PATH"
  exit 1
fi

if [ ! -d "$GRADCAM_STATS_DIR" ]; then
  print_error "GradCAM stats dir not found: $GRADCAM_STATS_DIR"
  exit 1
fi

for f in \
  gradcam_stats_fighting_class1.json \
  gradcam_stats_weapon_class3.json \
  gradcam_stats_blood_class5.json \
  gradcam_stats_war_class7.json
do
  if [ ! -f "$GRADCAM_STATS_DIR/$f" ]; then
    print_error "Missing GradCAM stats file: $f"
    exit 1
  fi
done

mkdir -p "$OUTPUT_BASE_DIR"

# ============================================================================
# Count experiments (절대 죽지 않는 방식)
# ============================================================================
total_experiments=0
for GS in "${GUIDANCE_SCALES[@]}"; do
  for ST in "${SPATIAL_THRESHOLDS[@]}"; do
    for HS in "${HARMFUL_SCALES[@]}"; do
      for BGS in "${BASE_GUIDANCE_SCALES[@]}"; do
        total_experiments=$((total_experiments + 1))
      done
    done
  done
done

print_info "Total experiments to run: $total_experiments"
echo ""

# ============================================================================
# Run Grid
# ============================================================================
current_experiment=0

for GS in "${GUIDANCE_SCALES[@]}"; do
  for ST in "${SPATIAL_THRESHOLDS[@]}"; do
    IFS=',' read -r ST_START ST_END <<< "$ST"

    for HS in "${HARMFUL_SCALES[@]}"; do
      for BGS in "${BASE_GUIDANCE_SCALES[@]}"; do
        current_experiment=$((current_experiment + 1))

        OUTPUT_DIR="${OUTPUT_BASE_DIR}/gs${GS}_st${ST_START}-${ST_END}_hs${HS}_bgs${BGS}"
        mkdir -p "$OUTPUT_DIR"

        print_header "Experiment ${current_experiment}/${total_experiments}"
        print_kv "GUIDANCE_SCALE" "$GS"
        print_kv "SPATIAL_THRESHOLD" "${ST_START} -> ${ST_END}"
        print_kv "HARMFUL_SCALE" "$HS"
        print_kv "BASE_GUIDANCE_SCALE" "$BGS"
        echo ""

        python generate_violence_9class_spatial_cg.py \
          "$CKPT_PATH" \
          --prompt_file "$PROMPT_FILE" \
          --output_dir "$OUTPUT_DIR" \
          --nsamples $NSAMPLES \
          --cfg_scale $CFG_SCALE \
          --num_inference_steps $NUM_INFERENCE_STEPS \
          --seed $SEED \
          --classifier_ckpt "$CLASSIFIER_PATH" \
          --gradcam_stats_dir "$GRADCAM_STATS_DIR" \
          --guidance_scale $GS \
          --spatial_threshold_start $ST_START \
          --spatial_threshold_end $ST_END \
          --threshold_strategy cosine_anneal \
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
        echo ""
      done
    done
  done
done

print_header "Grid Search Complete"
echo "Results saved to: $OUTPUT_BASE_DIR"
exit 0
