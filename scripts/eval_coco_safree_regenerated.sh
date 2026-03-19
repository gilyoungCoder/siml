#!/bin/bash
# COCO safree_regenerated FID + CLIP Score 평가
# Usage: bash eval_coco_safree_regenerated.sh <GPU>

GPU=${1:-7}
export CUDA_VISIBLE_DEVICES=${GPU}

BASE="/mnt/home/yhgil99/unlearning"
cd "${BASE}"

# Target directory
DIR="${BASE}/outputs/coco/safree_regenerated"
if [ -d "${DIR}/generated" ]; then
    TARGET="${DIR}/generated"
else
    TARGET="${DIR}"
fi

# Reference images for FID
REF_DIR="${BASE}/SAFREE/datasets/coco_30k_images"

echo "=============================================="
echo "COCO Evaluation (FID + CLIP)"
echo "GPU: ${GPU}"
echo "Target: ${TARGET}"
echo "Reference: ${REF_DIR}"
echo "=============================================="

if [ ! -d "${TARGET}" ]; then
    echo "ERROR: Target directory not found: ${TARGET}"
    exit 1
fi

# Count images
NUM_IMAGES=$(ls -1 "${TARGET}"/*.png 2>/dev/null | wc -l)
echo "Images: ${NUM_IMAGES}"
echo ""

# Run evaluation
python scripts/evaluate_coco_metrics.py \
    --gen_dir "${TARGET}" \
    --ref_dir "${REF_DIR}" \
    --prompt_file "${BASE}/prompts/coco/coco_10k.txt" \
    --output_file "${TARGET}/coco_metrics.json" \
    --device "cuda:0"

echo ""
echo "=============================================="
echo "Done! Results: ${TARGET}/coco_metrics.json"
echo "=============================================="
