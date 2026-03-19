#!/bin/bash
# ============================================================================
# Grid Search for I2P 9-Class Spatial Classifier Guidance
#
# Usage: bash grid_search_i2p_9class_spatial_cg.sh <CONCEPT> <STEP> <GPU>
# Example: bash grid_search_i2p_9class_spatial_cg.sh harassment 25000 0
#
# Concepts: harassment, hate, illegal, selfharm, shocking, violence
# ============================================================================

if [ $# -lt 3 ]; then
    echo "Usage: $0 <CONCEPT> <STEP> <GPU> [SKIP_IF_SAFE]"
    echo "Example: $0 harassment 25000 0"
    echo "Example: $0 harassment 25000 0 skip    # with --skip_if_safe"
    echo ""
    echo "Available concepts: harassment, hate, illegal, selfharm, shocking, violence"
    exit 1
fi

CONCEPT=$1
STEP=$2
GPU=$3
SKIP_MODE=${4:-""}  # Optional: "skip" to enable --skip_if_safe

export CUDA_VISIBLE_DEVICES=${GPU}

# Validate concept
if [[ ! "$CONCEPT" =~ ^(harassment|hate|illegal|selfharm|shocking|violence)$ ]]; then
    echo "Error: Invalid concept '$CONCEPT'"
    echo "Available concepts: harassment, hate, illegal, selfharm, shocking, violence"
    exit 1
fi

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# Paths
SD_MODEL="CompVis/stable-diffusion-v1-4"
CLASSIFIER_CKPT="./work_dirs/${CONCEPT}_9class/checkpoint/step_${STEP}/classifier.pth"
GRADCAM_STATS_DIR="./gradcam_stats/${CONCEPT}_9class_step${STEP}"

# Prompt file path (in guided2-safe-diffusion)
I2P_PROMPT_DIR="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/i2p"

# Map concept name to prompt filename
declare -A PROMPT_MAP
PROMPT_MAP["violence"]="violence_top50.txt"
PROMPT_MAP["shocking"]="shocking_top50.txt"
PROMPT_MAP["illegal"]="illegal activity_top50.txt"
PROMPT_MAP["selfharm"]="self-harm_top50.txt"
PROMPT_MAP["harassment"]="harassment_top50.txt"
PROMPT_MAP["hate"]="hate_top50.txt"

PROMPT_FILE="${I2P_PROMPT_DIR}/${PROMPT_MAP[$CONCEPT]}"

# Handle skip_if_safe mode
if [ "$SKIP_MODE" == "skip" ]; then
    SKIP_FLAG="--skip_if_safe"
    OUTPUT_BASE="./scg_outputs/grid_search_results/${CONCEPT}_9class_step${STEP}_skip"
    echo "Mode: SKIP if safe/benign"
else
    SKIP_FLAG=""
    OUTPUT_BASE="./scg_outputs/grid_search_results/${CONCEPT}_9class_step${STEP}"
    echo "Mode: ALWAYS guide (no skip)"
fi

# Check if classifier exists
if [ ! -f "$CLASSIFIER_CKPT" ]; then
    echo "Error: Classifier checkpoint not found: $CLASSIFIER_CKPT"
    exit 1
fi

# Check if prompt file exists
if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: Prompt file not found: $PROMPT_FILE"
    exit 1
fi

# Check GradCAM stats
if [ ! -d "$GRADCAM_STATS_DIR" ]; then
    echo "Warning: GradCAM stats directory not found: $GRADCAM_STATS_DIR"
    echo "Will use per-image normalization (not recommended)"
    GRADCAM_FLAG=""
else
    GRADCAM_FLAG="--gradcam_stats_dir ${GRADCAM_STATS_DIR}"
fi

echo "=============================================="
echo "I2P 9-Class Spatial CG Grid Search"
echo "=============================================="
echo "Concept: ${CONCEPT}"
echo "Classifier: ${CLASSIFIER_CKPT}"
echo "Prompt file: ${PROMPT_FILE}"
echo "Output base: ${OUTPUT_BASE}"
echo "GradCAM stats: ${GRADCAM_STATS_DIR}"
echo "GPU: ${GPU}"
echo "=============================================="

# Grid search parameters
GUIDANCE_SCALES=(7.5 10.0 12.5)
# Original + Reverse order (HIGH LOW) pairs
SPATIAL_THRESHOLDS=("0.7 0.3" "0.6 0.4" "0.5 0.3" "0.5 0.5" "0.4 0.2" "0.3 0.3" "0.3 0.7" "0.4 0.6" "0.3 0.5" "0.2 0.4")
HARMFUL_SCALES=(1.0 1.5 2.0)
BASE_GUIDANCE_SCALES=(1.0 2.0)
THRESHOLD_STRATEGIES=("cosine_anneal" "linear_decrease")

NSAMPLES=1
NUM_INFERENCE_STEPS=50
SEED=1234
CFG_SCALE=7.5

total_configs=0
for gs in "${GUIDANCE_SCALES[@]}"; do
    for st in "${SPATIAL_THRESHOLDS[@]}"; do
        for hs in "${HARMFUL_SCALES[@]}"; do
            for bgs in "${BASE_GUIDANCE_SCALES[@]}"; do
                for ts in "${THRESHOLD_STRATEGIES[@]}"; do
                    total_configs=$((total_configs + 1))
                done
            done
        done
    done
done

echo "Total configurations to run: ${total_configs}"
echo ""

current=0
for GUIDANCE_SCALE in "${GUIDANCE_SCALES[@]}"; do
    for SPATIAL_THRESHOLD in "${SPATIAL_THRESHOLDS[@]}"; do
        read -r ST_START ST_END <<< "$SPATIAL_THRESHOLD"
        for HARMFUL_SCALE in "${HARMFUL_SCALES[@]}"; do
            for BASE_GUIDANCE_SCALE in "${BASE_GUIDANCE_SCALES[@]}"; do
                for THRESHOLD_STRATEGY in "${THRESHOLD_STRATEGIES[@]}"; do
                    current=$((current + 1))

                    # Short name for strategy
                    if [ "$THRESHOLD_STRATEGY" == "cosine_anneal" ]; then
                        TS_SHORT="cos"
                    else
                        TS_SHORT="lin"
                    fi

                    CONFIG_NAME="gs${GUIDANCE_SCALE}_st${ST_START}-${ST_END}_hs${HARMFUL_SCALE}_bgs${BASE_GUIDANCE_SCALE}_${TS_SHORT}"
                    OUTPUT_DIR="${OUTPUT_BASE}/${CONFIG_NAME}"

                    echo ""
                    echo "[$current/$total_configs] Running: $CONFIG_NAME"
                    echo "  guidance_scale=${GUIDANCE_SCALE}"
                    echo "  spatial_threshold=${ST_START} -> ${ST_END}"
                    echo "  harmful_scale=${HARMFUL_SCALE}"
                    echo "  base_guidance_scale=${BASE_GUIDANCE_SCALE}"
                    echo "  threshold_strategy=${THRESHOLD_STRATEGY}"
                    echo ""

                    python generate_i2p_9class_spatial_cg.py \
                        "${SD_MODEL}" \
                        --concept "${CONCEPT}" \
                        --prompt_file "${PROMPT_FILE}" \
                        --output_dir "${OUTPUT_DIR}" \
                        --classifier_ckpt "${CLASSIFIER_CKPT}" \
                        ${GRADCAM_FLAG} \
                        ${SKIP_FLAG} \
                        --nsamples ${NSAMPLES} \
                        --cfg_scale ${CFG_SCALE} \
                        --num_inference_steps ${NUM_INFERENCE_STEPS} \
                        --seed ${SEED} \
                        --guidance_scale ${GUIDANCE_SCALE} \
                        --spatial_threshold_start ${ST_START} \
                        --spatial_threshold_end ${ST_END} \
                        --threshold_strategy ${THRESHOLD_STRATEGY} \
                        --use_bidirectional \
                        --harmful_scale ${HARMFUL_SCALE} \
                        --base_guidance_scale ${BASE_GUIDANCE_SCALE}

                    echo "Completed: $CONFIG_NAME"
                done
            done
        done
    done
done

echo ""
echo "=============================================="
echo "Grid Search Complete!"
echo "=============================================="
echo "Results saved to: ${OUTPUT_BASE}"
echo "=============================================="
