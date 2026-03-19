#!/bin/bash
# Run SHGD Grid Search
# Usage: bash run_grid_search.sh [grid_type] [gpu_ids]
# Example: bash run_grid_search.sh quick 0,1,2,3

PYTHON="PYTHONNOUSERSITE=1 /mnt/home/yhgil99/.conda/envs/sfgd/bin/python"

GRID_TYPE=${1:-"quick"}
GPU_IDS=${2:-"0,1,2,3"}

echo "=============================================="
echo "SHGD Grid Search"
echo "=============================================="
echo "Grid: $GRID_TYPE"
echo "GPUs: $GPU_IDS"
echo "=============================================="

cd "$(dirname "$0")"

$PYTHON grid_search.py \
    --config configs/default.yaml \
    --prompt_file "../rab_grid_search/data/ringabell_full.txt" \
    --gpu_ids "$GPU_IDS" \
    --grid "$GRID_TYPE"
