#!/bin/bash
# Aggregate mon01_grid VLM results (ringabell, unlearndiff, mma)
# Usage: bash vlm/aggregate_mon01_grid_vlm.sh

BASE="SoftDelete+CG/scg_outputs/mon01_grid"

bash vlm/aggregate_grid_cross_dataset.sh \
    "$BASE/ringabell" \
    "$BASE/unlearndiff" \
    "$BASE/mma"
