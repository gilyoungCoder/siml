#!/bin/bash
# 이미지 그리드 생성
# Usage: ./vlm/run_make_grid.sh <folder> [output_name]
#
# Examples:
#   ./vlm/run_make_grid.sh /path/to/images
#   ./vlm/run_make_grid.sh /path/to/images my_grid.png

set -e

if [ -z "$1" ]; then
    echo "Usage: ./vlm/run_make_grid.sh <folder> [output_name]"
    exit 1
fi

FOLDER=$1
OUTPUT=${2:-"grid_$(basename $FOLDER).png"}

cd /mnt/home/yhgil99/unlearning

echo "Creating 4x4 grid from: $FOLDER"
python vlm/make_grid.py "$FOLDER" --grid-size 4x4 --output "$OUTPUT"

echo "✅ Grid saved to: $OUTPUT"
