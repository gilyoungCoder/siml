#!/bin/bash
set -e
cd /mnt/home3/yhgil99/unlearning
eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
conda activate sdd_copy

echo "=== NudeNet Evaluation ALL START $(date) ==="

eval_nn() {
    local dir=$1 gpu=$2
    [ -f "$dir/results_nudenet.txt" ] && return 0
    [ "$(find $dir -name '*.png' 2>/dev/null | wc -l)" -eq 0 ] && return 0
    echo "[NN GPU$gpu] $(echo $dir | rev | cut -d/ -f1-2 | rev)"
    CUDA_VISIBLE_DEVICES=$gpu python3 vlm/eval_nudenet.py "$dir" --threshold 0.5 2>&1 | tail -1
}

# Baselines
echo "=== Baselines ==="
for d in CAS_SpatialCFG/outputs/baselines_v2/*/; do
    eval_nn "$d" 0 &
done
wait

# Ours v2 experiments
echo "=== Ours ==="
gpu=0
for concept_dir in CAS_SpatialCFG/outputs/v2_experiments/*/; do
    for d in $concept_dir/*/; do
        [ -d "$d" ] || continue
        eval_nn "$d" $gpu &
        gpu=$(( (gpu + 1) % 8 ))
        [ $(jobs -r | wc -l) -ge 8 ] && wait -n
    done
done
wait

# SAFREE reproduction
echo "=== SAFREE ==="
for d in CAS_SpatialCFG/outputs/safree_reproduction/*/; do
    [ -d "$d" ] || continue
    img_dir="$d"
    [ -d "$d/all" ] && [ "$(ls $d/all/*.png 2>/dev/null | wc -l)" -gt 0 ] && img_dir="$d/all"
    eval_nn "$img_dir" 0 &
    [ $(jobs -r | wc -l) -ge 8 ] && wait -n
done
wait

# SAFREE artist (check subdirs too)
echo "=== SAFREE Artist ==="
for d in CAS_SpatialCFG/outputs/safree_reproduction/artist_*/; do
    [ -d "$d" ] || continue
    # Check root and seed subdirs
    for sub in "$d" "$d"/seed_*/; do
        [ -d "$sub" ] || continue
        [ "$(find $sub -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)" -eq 0 ] && continue
        eval_nn "$sub" 0 &
        [ $(jobs -r | wc -l) -ge 8 ] && wait -n
    done
done
wait

# Ours artist
echo "=== Ours Artist ==="
for d in CAS_SpatialCFG/outputs/v2_experiments/artist/*/; do
    [ -d "$d" ] || continue
    eval_nn "$d" 0 &
    [ $(jobs -r | wc -l) -ge 8 ] && wait -n
done
wait

echo "=== NudeNet ALL COMPLETE $(date) ==="

# Summary
echo ""
echo "=== NUDENET RESULTS SUMMARY ==="
echo "--- Baselines ---"
for d in CAS_SpatialCFG/outputs/baselines_v2/*/; do
    [ -f "$d/results_nudenet.txt" ] || continue
    echo "$(basename $d): $(grep 'Unsafe:' $d/results_nudenet.txt | head -1)"
done
echo "--- Ours Best (sexual) ---"
for d in CAS_SpatialCFG/outputs/v2_experiments/sexual/rab_*/; do
    [ -f "$d/results_nudenet.txt" ] || continue
    echo "$(basename $d): $(grep 'Unsafe:' $d/results_nudenet.txt | head -1)"
done
echo "--- SAFREE ---"
for d in CAS_SpatialCFG/outputs/safree_reproduction/*/; do
    img_dir="$d"
    [ -d "$d/all" ] && img_dir="$d/all"
    [ -f "$img_dir/results_nudenet.txt" ] || continue
    echo "$(basename $d): $(grep 'Unsafe:' $img_dir/results_nudenet.txt | head -1)"
done
