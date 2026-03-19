#!/bin/bash
# ============================================================================
# Grid Search for SAFREE++ 9-Class Spatial CG
#
# Usage: bash grid_search_safree_9class.sh <CONCEPT> <GPU>
# Example: bash grid_search_safree_9class.sh harassment 0
#
# Concepts: harassment, hate, illegal, selfharm, shocking, violence
# ============================================================================

if [ $# -lt 2 ]; then
    echo "Usage: $0 <CONCEPT> <GPU>"
    echo "Example: $0 harassment 0"
    echo ""
    echo "Available concepts: harassment, hate, illegal, selfharm, shocking, violence"
    exit 1
fi

CONCEPT=$1
GPU=$2

export CUDA_VISIBLE_DEVICES=${GPU}

# Validate concept and set step
declare -A STEP_MAP
STEP_MAP["harassment"]="24300"
STEP_MAP["hate"]="20800"
STEP_MAP["illegal"]="22600"
STEP_MAP["selfharm"]="20700"
STEP_MAP["shocking"]="23700"
STEP_MAP["violence"]="15500"

if [[ -z "${STEP_MAP[$CONCEPT]}" ]]; then
    echo "Error: Invalid concept '$CONCEPT'"
    echo "Available concepts: harassment, hate, illegal, selfharm, shocking, violence"
    exit 1
fi

STEP=${STEP_MAP[$CONCEPT]}

cd /mnt/home/yhgil99/unlearning/SAFREE

# Paths
SD_MODEL="CompVis/stable-diffusion-v1-4"
CLASSIFIER_CKPT="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/${CONCEPT}_9class/checkpoint/step_${STEP}/classifier.pth"
GRADCAM_STATS_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/${CONCEPT}_9class_step${STEP}"

# Prompt file path
I2P_PROMPT_DIR="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/i2p"

declare -A PROMPT_MAP
PROMPT_MAP["violence"]="violence_top50.txt"
PROMPT_MAP["shocking"]="shocking_top50.txt"
PROMPT_MAP["illegal"]="illegal activity_top50.txt"
PROMPT_MAP["selfharm"]="self-harm_top50.txt"
PROMPT_MAP["harassment"]="harassment_top50.txt"
PROMPT_MAP["hate"]="hate_top50.txt"

PROMPT_FILE="${I2P_PROMPT_DIR}/${PROMPT_MAP[$CONCEPT]}"

OUTPUT_BASE="./results/grid_search_safree_9class/${CONCEPT}_step${STEP}"

# Check paths
if [ ! -f "$CLASSIFIER_CKPT" ]; then
    echo "Error: Classifier checkpoint not found: $CLASSIFIER_CKPT"
    exit 1
fi

if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: Prompt file not found: $PROMPT_FILE"
    exit 1
fi

GRADCAM_FLAG=""
if [ -d "$GRADCAM_STATS_DIR" ]; then
    GRADCAM_FLAG="--gradcam_stats_dir ${GRADCAM_STATS_DIR}"
else
    echo "Warning: GradCAM stats not found, using per-image normalization"
fi

echo "=============================================="
echo "SAFREE++ 9-Class Spatial CG Grid Search"
echo "=============================================="
echo "Concept: ${CONCEPT}"
echo "Step: ${STEP}"
echo "Classifier: ${CLASSIFIER_CKPT}"
echo "Prompt file: ${PROMPT_FILE}"
echo "Output base: ${OUTPUT_BASE}"
echo "GPU: ${GPU}"
echo "=============================================="

# Grid search parameters (compact - SAFREE already does text-level erasing)
# CG guidance scales: 7.5 as baseline, with smaller and larger options
CG_GUIDANCE_SCALES=(5.0 7.5 10.0)

# Spatial thresholds: (start end) - including reverse directions
SPATIAL_THRESHOLDS=("0.7 0.3" "0.5 0.3" "0.5 0.5" "0.3 0.7" "0.3 0.5")

# Harmful scales: lower values since SAFREE already erases
HARMFUL_SCALES=(0.5 1.0 1.5)

# Base guidance (safe direction)
BASE_GUIDANCE_SCALES=(0.0 1.0 2.0)

# Threshold strategy
THRESHOLD_STRATEGIES=("linear_decrease")

# SAFREE parameters (fixed)
SAFREE_ALPHA=0.01
SVF_UP_T=10

# Generation parameters
NSAMPLES=1
NUM_INFERENCE_STEPS=50
SEED=42
CFG_SCALE=7.5

# Count total configurations
total_configs=0
for cgs in "${CG_GUIDANCE_SCALES[@]}"; do
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

echo "Total configurations: ${total_configs}"
echo ""

current=0
for CG_GUIDANCE_SCALE in "${CG_GUIDANCE_SCALES[@]}"; do
    for SPATIAL_THRESHOLD in "${SPATIAL_THRESHOLDS[@]}"; do
        read -r ST_START ST_END <<< "$SPATIAL_THRESHOLD"
        for HARMFUL_SCALE in "${HARMFUL_SCALES[@]}"; do
            for BASE_GUIDANCE_SCALE in "${BASE_GUIDANCE_SCALES[@]}"; do
                for THRESHOLD_STRATEGY in "${THRESHOLD_STRATEGIES[@]}"; do
                    current=$((current + 1))

                    CONFIG_NAME="cgs${CG_GUIDANCE_SCALE}_st${ST_START}-${ST_END}_hs${HARMFUL_SCALE}_bgs${BASE_GUIDANCE_SCALE}"
                    OUTPUT_DIR="${OUTPUT_BASE}/${CONFIG_NAME}"

                    # Skip if already done
                    if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR 2>/dev/null)" ]; then
                        echo "[$current/$total_configs] Skipping (exists): $CONFIG_NAME"
                        continue
                    fi

                    echo ""
                    echo "[$current/$total_configs] Running: $CONFIG_NAME"
                    echo "  cg_guidance_scale=${CG_GUIDANCE_SCALE}"
                    echo "  spatial_threshold=${ST_START} -> ${ST_END}"
                    echo "  harmful_scale=${HARMFUL_SCALE}"
                    echo "  base_guidance_scale=${BASE_GUIDANCE_SCALE}"
                    echo ""

                    python generate_safree_9class_spatial_cg.py \
                        --ckpt_path "${SD_MODEL}" \
                        --prompt_file "${PROMPT_FILE}" \
                        --output_dir "${OUTPUT_DIR}" \
                        --classifier_ckpt "${CLASSIFIER_CKPT}" \
                        ${GRADCAM_FLAG} \
                        --category "${CONCEPT}" \
                        --nsamples ${NSAMPLES} \
                        --cfg_scale ${CFG_SCALE} \
                        --num_inference_steps ${NUM_INFERENCE_STEPS} \
                        --seed ${SEED} \
                        --safree \
                        --safree_alpha ${SAFREE_ALPHA} \
                        --svf \
                        --svf_up_t ${SVF_UP_T} \
                        --spatial_cg \
                        --cg_guidance_scale ${CG_GUIDANCE_SCALE} \
                        --spatial_threshold_start ${ST_START} \
                        --spatial_threshold_end ${ST_END} \
                        --threshold_strategy "${THRESHOLD_STRATEGY}" \
                        --harmful_scale ${HARMFUL_SCALE} \
                        --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
                        --skip_if_safe

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
echo "Results: ${OUTPUT_BASE}"
echo "Total configs: ${total_configs}"
echo "=============================================="
