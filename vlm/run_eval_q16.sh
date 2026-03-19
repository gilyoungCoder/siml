#!/bin/bash
# Q16 Classifier 평가
# Usage: ./vlm/run_eval_q16.sh <folder> [threshold] [gpu_id]
#
# Examples:
#   ./vlm/run_eval_q16.sh /path/to/images
#   ./vlm/run_eval_q16.sh /path/to/images 0.5
#   ./vlm/run_eval_q16.sh /path/to/images 0.7 4

set -e

if [ -z "$1" ]; then
    echo "Usage: ./vlm/run_eval_q16.sh <folder> [threshold] [gpu_id]"
    echo "  threshold: default 0.7 (Q16 paper recommendation)"
    echo "  gpu_id: default 0"
    exit 1
fi

FOLDER=$1
THRESHOLD=${2:-0.5}
GPU_ID=${3:-0}

# Activate conda environment
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate sdd

cd /mnt/home/yhgil99/unlearning

echo "============================================================"
echo "Q16 Classifier Evaluation"
echo "Folder: $FOLDER"
echo "Threshold: $THRESHOLD"
echo "GPU: $GPU_ID"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU_ID python vlm/eval_q16.py "$FOLDER" --threshold "$THRESHOLD"

echo "✅ Q16 evaluation complete!"
