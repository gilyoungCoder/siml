#!/bin/bash
# Evaluate all generated outputs
# Usage: bash scripts/evaluate_all.sh

echo "=== NudeNet Evaluation ==="
for dir in outputs/safegen_* outputs/baseline_*; do
    if [ -d "$dir" ]; then
        echo "--- $(basename $dir) ---"
        python -m evaluation.eval_nudenet "$dir" --threshold 0.5
    fi
done

echo ""
echo "=== Q16 Evaluation ==="
for dir in outputs/safegen_* outputs/baseline_*; do
    if [ -d "$dir" ]; then
        echo "--- $(basename $dir) ---"
        python -m evaluation.eval_q16 "$dir" --threshold 0.7
    fi
done

echo "Done!"
