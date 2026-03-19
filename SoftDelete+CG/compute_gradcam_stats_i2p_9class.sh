#!/bin/bash
# ============================================================================
# Compute GradCAM Statistics for I2P 9-Class (All 4 Harm Classes)
# ============================================================================
#
# Harm classes structure:
#   - Class 1 (harm0): harm0/
#   - Class 3 (harm1): harm1/
#   - Class 5 (harm2): harm2/
#   - Class 7 (harm3): harm3/
#
# Best checkpoint steps per concept:
#   - Violence: 15500
#   - Shocking: 23700
#   - Illegal: 22600
#   - Self-harm: 20700
#   - Harassment: 24300
#   - Hate: 20800
#
# Usage:
#   ./compute_gradcam_stats_i2p_9class.sh <CONCEPT> <STEP> <GPU>
#   ./compute_gradcam_stats_i2p_9class.sh harassment 24300 0
# ============================================================================

set -e

if [ $# -lt 3 ]; then
    echo "Usage: $0 <CONCEPT> <STEP> <GPU>"
    echo "Example: $0 harassment 24300 0"
    echo ""
    echo "Available concepts: harassment, hate, illegal, selfharm, shocking, violence"
    echo ""
    echo "Best steps per concept:"
    echo "  - Violence: 15500"
    echo "  - Shocking: 23700"
    echo "  - Illegal: 22600"
    echo "  - Self-harm: 20700"
    echo "  - Harassment: 24300"
    echo "  - Hate: 20800"
    exit 1
fi

CONCEPT=$1
STEP=$2
GPU=$3

# Validate concept
if [[ ! "$CONCEPT" =~ ^(harassment|hate|illegal|selfharm|shocking|violence)$ ]]; then
    echo "Error: Invalid concept '$CONCEPT'"
    echo "Available concepts: harassment, hate, illegal, selfharm, shocking, violence"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Computing GradCAM Statistics for ${CONCEPT} 9-Class${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "${YELLOW}Classifier Step: ${STEP}${NC}"
echo -e "${YELLOW}GPU: ${GPU}${NC}"
echo ""

# Classifier path
CLASSIFIER_CKPT="./work_dirs/${CONCEPT}_9class/checkpoint/step_${STEP}/classifier.pth"

# Check if classifier exists
if [ ! -f "$CLASSIFIER_CKPT" ]; then
    echo -e "\033[0;31mError: Classifier not found: $CLASSIFIER_CKPT\033[0m"
    exit 1
fi

# Dataset path
DATASET_BASE="/mnt/home/yhgil99/dataset/threeclassImg/i2p/${CONCEPT}_8class"

# Check if dataset exists
if [ ! -d "$DATASET_BASE" ]; then
    echo -e "\033[0;31mError: Dataset not found: $DATASET_BASE\033[0m"
    exit 1
fi

# Output directory
OUTPUT_DIR="./gradcam_stats/${CONCEPT}_9class_step${STEP}"
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}Output directory: ${OUTPUT_DIR}${NC}"
echo ""

# Compute stats for each harm class (1, 3, 5, 7)
for HARM_CLASS in 1 3 5 7; do
    # harm0 -> class 1, harm1 -> class 3, harm2 -> class 5, harm3 -> class 7
    HARM_IDX=$((($HARM_CLASS - 1) / 2))
    DATA_DIR="${DATASET_BASE}/harm${HARM_IDX}"
    OUTPUT_FILE="${OUTPUT_DIR}/gradcam_stats_harm${HARM_IDX}_class${HARM_CLASS}.json"

    echo -e "${GREEN}============================================================${NC}"
    echo -e "${YELLOW}Computing stats for: harm${HARM_IDX} (class ${HARM_CLASS})${NC}"
    echo -e "Data: ${DATA_DIR}"
    echo -e "Output: ${OUTPUT_FILE}"
    echo -e "${GREEN}============================================================${NC}"

    # Check if data directory exists
    if [ ! -d "$DATA_DIR" ]; then
        echo -e "\033[0;31mWarning: Data directory not found: $DATA_DIR - Skipping\033[0m"
        continue
    fi

    python compute_gradcam_statistics.py \
        --data_dir "$DATA_DIR" \
        --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
        --classifier_ckpt "$CLASSIFIER_CKPT" \
        --output_file "$OUTPUT_FILE" \
        --num_samples 1000 \
        --harmful_class $HARM_CLASS \
        --gradcam_layer "encoder_model.middle_block.2" \
        --num_classes 9

    echo ""
done

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}All GradCAM statistics computed for ${CONCEPT}!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Output files:"
for HARM_CLASS in 1 3 5 7; do
    HARM_IDX=$((($HARM_CLASS - 1) / 2))
    echo "  - ${OUTPUT_DIR}/gradcam_stats_harm${HARM_IDX}_class${HARM_CLASS}.json"
done
echo ""
