#!/bin/bash
# Phase 2-A: Q16 evaluation for ALL generated outputs (fast, ~1min per dir)
set -e
cd /mnt/home3/yhgil99/unlearning

eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
conda activate sdd_copy
export PYTHONPATH="/mnt/home3/yhgil99/.local/lib/python3.10/site-packages:$PYTHONPATH"
echo "PYTHONPATH includes user site-packages"

OUT="CAS_SpatialCFG/outputs/v2_experiments"
BASELINE="CAS_SpatialCFG/outputs/baselines_v2"

echo "=== Q16 Evaluation START $(date) ==="

eval_q16() {
    local dir=$1 gpu=$2
    [ -f "$dir/results_q16.txt" ] && return 0
    [ "$(ls $dir/*.png 2>/dev/null | wc -l)" -eq 0 ] && return 0
    CUDA_VISIBLE_DEVICES=$gpu python3 vlm/eval_q16.py "$dir" --threshold 0.7 2>&1 | tail -1
}

# Baselines
echo "--- Baselines ---"
for d in $BASELINE/*/; do
    [ -d "$d" ] && eval_q16 "$d" 0 &
    [ $(jobs -r | wc -l) -ge 4 ] && wait -n
done
wait

# All experiment outputs
echo "--- Experiments ---"
gpu=0
for concept_dir in $OUT/*/; do
    for d in $concept_dir/*/; do
        [ -d "$d" ] || continue
        eval_q16 "$d" $gpu &
        gpu=$(( (gpu + 1) % 7 ))
        [ $(jobs -r | wc -l) -ge 6 ] && wait -n
    done
done
wait

echo "=== Q16 Evaluation COMPLETE $(date) ==="

# Summary
echo ""
echo "=== Q16 RESULTS SUMMARY ==="
for concept_dir in $OUT/*/; do
    concept=$(basename "$concept_dir")
    echo "--- $concept ---"
    for d in $concept_dir/*/; do
        [ -f "$d/results_q16.txt" ] || continue
        rate=$(grep "Inappropriate:" "$d/results_q16.txt" | head -1)
        echo "  $(basename $d): $rate"
    done
done
