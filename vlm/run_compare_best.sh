#!/bin/bash
# Best config 비교 그리드 생성
# Usage: ./vlm/run_compare_best.sh [concept|all]
#
# Examples:
#   ./vlm/run_compare_best.sh nudity
#   ./vlm/run_compare_best.sh all

set -e

cd /mnt/home/yhgil99/unlearning

# Create output directory
mkdir -p vlm/grids

if [ "$1" == "all" ] || [ -z "$1" ]; then
    echo "Generating comparison grids for ALL concepts..."
    python vlm/compare_best_configs.py --all --output-dir vlm/grids/
else
    CONCEPT=$1
    echo "Generating comparison grid for: $CONCEPT"
    python vlm/compare_best_configs.py --concept "$CONCEPT" --output "vlm/grids/${CONCEPT}_comparison.png"
fi

echo ""
echo "✅ Grids saved to: vlm/grids/"
