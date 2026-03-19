#!/bin/bash
# ============================================================================
# Compute GradCAM Statistics for Violence 9-Class (All 4 Harm Classes)
# ============================================================================
#
# Harm classes and their data:
#   - Class 1 (harm_fighting): street_fight
#   - Class 3 (harm_weapon): attacking_with_knife
#   - Class 5 (harm_blood): bloody_wounds
#   - Class 7 (harm_war): soldiers_battlefield
#
# Usage:
#   ./compute_gradcam_stats_violence_9class.sh [STEP] [GPU]
#   ./compute_gradcam_stats_violence_9class.sh 20000 0
# ============================================================================

set -e

# Default values
STEP=25400
GPU=5

export CUDA_VISIBLE_DEVICES=$GPU

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Computing GradCAM Statistics for Violence 9-Class${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "${YELLOW}Classifier Step: ${STEP}${NC}"
echo -e "${YELLOW}GPU: ${GPU}${NC}"
echo ""

# Classifier path
CLASSIFIER_CKPT="./work_dirs/violence_9class/checkpoint/step_${STEP}/classifier.pth"

# Check if classifier exists
if [ ! -f "$CLASSIFIER_CKPT" ]; then
    echo -e "\033[0;31mError: Classifier not found: $CLASSIFIER_CKPT\033[0m"
    exit 1
fi

# Dataset paths for each harm class
DATASET_BASE="/mnt/home/yhgil99/dataset/threeclassImg/violence_9class"

# Harm class mapping
declare -A HARM_CLASSES
HARM_CLASSES[1]="street_fight"          # harm_fighting
HARM_CLASSES[3]="attacking_with_knife"  # harm_weapon
HARM_CLASSES[5]="bloody_wounds"         # harm_blood
HARM_CLASSES[7]="soldiers_battlefield"  # harm_war

declare -A HARM_NAMES
HARM_NAMES[1]="fighting"
HARM_NAMES[3]="weapon"
HARM_NAMES[5]="blood"
HARM_NAMES[7]="war"

# Output directory
OUTPUT_DIR="./gradcam_stats_violence_9class_step${STEP}"
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}Output directory: ${OUTPUT_DIR}${NC}"
echo ""

# Compute stats for each harm class
for HARM_CLASS in 1 3 5 7; do
    DATA_FOLDER="${HARM_CLASSES[$HARM_CLASS]}"
    HARM_NAME="${HARM_NAMES[$HARM_CLASS]}"
    DATA_DIR="${DATASET_BASE}/${DATA_FOLDER}"
    OUTPUT_FILE="${OUTPUT_DIR}/gradcam_stats_${HARM_NAME}_class${HARM_CLASS}.json"

    echo -e "${GREEN}============================================================${NC}"
    echo -e "${YELLOW}Computing stats for: ${HARM_NAME} (class ${HARM_CLASS})${NC}"
    echo -e "Data: ${DATA_DIR}"
    echo -e "Output: ${OUTPUT_FILE}"
    echo -e "${GREEN}============================================================${NC}"

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
echo -e "${GREEN}All GradCAM statistics computed!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Output files:"
for HARM_CLASS in 1 3 5 7; do
    HARM_NAME="${HARM_NAMES[$HARM_CLASS]}"
    echo "  - ${OUTPUT_DIR}/gradcam_stats_${HARM_NAME}_class${HARM_CLASS}.json"
done
echo ""
