#!/bin/bash
# ============================================================================
# Compute GradCAM Statistics for Violence 13-Class Classifier
#
# Computes per-class GradCAM statistics for harm AND artifact classes:
#
# Harm classes:
#   - Class 1: harm_fighting (street_fight)
#   - Class 4: harm_weapon (attacking_with_knife)
#   - Class 7: harm_blood (bloody_wounds)
#   - Class 10: harm_war (soldiers_battlefield)
#
# Artifact classes:
#   - Class 3: artifact_fighting (friendly_handshake_artifact)
#   - Class 6: artifact_weapon (chopping_vegetables_artifact)
#   - Class 9: artifact_blood (bandaging_hospital_artifact)
#   - Class 12: artifact_war (peace_conference_artifact)
#
# Usage:
#   ./compute_gradcam_stats_violence_13class.sh <STEP> <GPU>
#   ./compute_gradcam_stats_violence_13class.sh 20000 0
# ============================================================================

set -e

STEP=${1:-20000}
GPU=${2:-0}

export CUDA_VISIBLE_DEVICES=$GPU

ROOT_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG"
cd "$ROOT_DIR" || exit 1

# Paths
CLASSIFIER_CKPT="./work_dirs/violence_13class/checkpoint/step_${STEP}/classifier.pth"
OUTPUT_DIR="./gradcam_stats/violence_13class_step${STEP}"
DATA_BASE="/mnt/home/yhgil99/dataset/threeclassImg/violence_9class"

# Check classifier exists
if [ ! -f "$CLASSIFIER_CKPT" ]; then
    echo "Error: Classifier checkpoint not found: $CLASSIFIER_CKPT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "=============================================="
echo "Computing GradCAM Stats for Violence 13-Class"
echo "=============================================="
echo "Classifier: $CLASSIFIER_CKPT"
echo "Output: $OUTPUT_DIR"
echo "GPU: $GPU"
echo "=============================================="

# ============================================================================
# HARM CLASSES
# ============================================================================
echo ""
echo ">>> HARM CLASSES"

declare -A HARM_CLASS_MAP
HARM_CLASS_MAP["fighting"]="1"
HARM_CLASS_MAP["weapon"]="4"
HARM_CLASS_MAP["blood"]="7"
HARM_CLASS_MAP["war"]="10"

declare -A HARM_DATA_MAP
HARM_DATA_MAP["fighting"]="${DATA_BASE}/street_fight"
HARM_DATA_MAP["weapon"]="${DATA_BASE}/attacking_with_knife"
HARM_DATA_MAP["blood"]="${DATA_BASE}/bloody_wounds"
HARM_DATA_MAP["war"]="${DATA_BASE}/soldiers_battlefield"

for category in "fighting" "weapon" "blood" "war"; do
    CLASS_ID=${HARM_CLASS_MAP[$category]}
    DATA_DIR=${HARM_DATA_MAP[$category]}
    OUTPUT_FILE="${OUTPUT_DIR}/gradcam_stats_${category}_class${CLASS_ID}.json"

    echo ""
    echo ">>> Computing GradCAM stats for harm_${category} (class ${CLASS_ID})"
    echo "    Data: $DATA_DIR"
    echo "    Output: $OUTPUT_FILE"

    if [ ! -d "$DATA_DIR" ]; then
        echo "Warning: Data directory not found: $DATA_DIR"
        continue
    fi

    python compute_gradcam_statistics.py \
        --data_dir "$DATA_DIR" \
        --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
        --classifier_ckpt "$CLASSIFIER_CKPT" \
        --num_classes 13 \
        --harmful_class $CLASS_ID \
        --output_file "$OUTPUT_FILE" \
        --num_samples 2000 \
        --batch_size 8 \
        --t_min 0 \
        --t_max 999

    echo "    Done: $OUTPUT_FILE"
done

# ============================================================================
# ARTIFACT CLASSES
# ============================================================================
echo ""
echo ">>> ARTIFACT CLASSES"

declare -A ARTIFACT_CLASS_MAP
ARTIFACT_CLASS_MAP["fighting"]="3"
ARTIFACT_CLASS_MAP["weapon"]="6"
ARTIFACT_CLASS_MAP["blood"]="9"
ARTIFACT_CLASS_MAP["war"]="12"

declare -A ARTIFACT_DATA_MAP
ARTIFACT_DATA_MAP["fighting"]="${DATA_BASE}/friendly_handshake_artifact"
ARTIFACT_DATA_MAP["weapon"]="${DATA_BASE}/chopping_vegetables_artifact"
ARTIFACT_DATA_MAP["blood"]="${DATA_BASE}/bandaging_hospital_artifact"
ARTIFACT_DATA_MAP["war"]="${DATA_BASE}/peace_conference_artifact"

for category in "fighting" "weapon" "blood" "war"; do
    CLASS_ID=${ARTIFACT_CLASS_MAP[$category]}
    DATA_DIR=${ARTIFACT_DATA_MAP[$category]}
    OUTPUT_FILE="${OUTPUT_DIR}/gradcam_stats_artifact_${category}_class${CLASS_ID}.json"

    echo ""
    echo ">>> Computing GradCAM stats for artifact_${category} (class ${CLASS_ID})"
    echo "    Data: $DATA_DIR"
    echo "    Output: $OUTPUT_FILE"

    if [ ! -d "$DATA_DIR" ]; then
        echo "Warning: Data directory not found: $DATA_DIR"
        continue
    fi

    python compute_gradcam_statistics.py \
        --data_dir "$DATA_DIR" \
        --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
        --classifier_ckpt "$CLASSIFIER_CKPT" \
        --num_classes 13 \
        --harmful_class $CLASS_ID \
        --output_file "$OUTPUT_FILE" \
        --num_samples 2000 \
        --batch_size 8 \
        --t_min 0 \
        --t_max 999

    echo "    Done: $OUTPUT_FILE"
done

echo ""
echo "=============================================="
echo "All GradCAM statistics computed!"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="
ls -la "$OUTPUT_DIR"
