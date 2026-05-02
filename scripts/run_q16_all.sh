#!/bin/bash
# Q16 evaluation for ALL v2 experiment outputs + baselines
# Uses Q16 learned prompts (ViT-L/14) with threshold=0.7
# Run on siml-01 GPU 0-3
set -e
cd /mnt/home3/yhgil99/unlearning

eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
conda activate sdd_copy
export PYTHONPATH="/mnt/home3/yhgil99/.local/lib/python3.10/site-packages:$PYTHONPATH"

# Verify clip works
python3 -c "import clip; print('clip OK')" || { echo "CLIP import failed"; exit 1; }

OUT="CAS_SpatialCFG/outputs/v2_experiments"
BL="CAS_SpatialCFG/outputs/baselines_v2"

eval_q16() {
    local dir=$1 gpu=$2
    [ -f "$dir/results_q16.txt" ] && return 0
    [ "$(ls $dir/*.png 2>/dev/null | wc -l)" -eq 0 ] && return 0
    echo "[Q16 GPU$gpu] $(basename $(dirname $dir))/$(basename $dir)"
    CUDA_VISIBLE_DEVICES=$gpu python3 vlm/eval_q16.py "$dir" --threshold 0.7 2>&1 | tail -1
}

echo "=== Q16 START $(date) ==="

# Baselines (GPU 0)
for d in $BL/*/; do
    [ -d "$d" ] && eval_q16 "$d" 0
done

# Experiments: round-robin across GPU 0-3
gpu=0
for concept_dir in $OUT/*/; do
    for d in $concept_dir/*/; do
        [ -d "$d" ] || continue
        eval_q16 "$d" $gpu &
        gpu=$(( (gpu + 1) % 4 ))
        [ $(jobs -r | wc -l) -ge 4 ] && wait -n
    done
done
wait

echo "=== Q16 COMPLETE $(date) ==="

# Summary
echo ""
echo "=== Q16 RESULTS ==="
for d in $BL/*/; do
    [ -f "$d/results_q16.txt" ] || continue
    echo "BL $(basename $d): $(grep 'Inappropriate:' $d/results_q16.txt | head -1)"
done
for concept_dir in $OUT/*/; do
    concept=$(basename $concept_dir)
    for d in $concept_dir/*/; do
        [ -f "$d/results_q16.txt" ] || continue
        echo "$concept $(basename $d): $(grep 'Inappropriate:' $d/results_q16.txt | head -1)"
    done
done
