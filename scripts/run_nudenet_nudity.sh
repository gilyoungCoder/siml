#!/bin/bash
set -e
cd /mnt/home3/yhgil99/unlearning
eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
conda activate sdd_copy

echo "=== NudeNet (Nudity Only) START $(date) ==="

eval_nn() {
    local dir=$1 gpu=$2
    [ -f "$dir/results_nudenet.txt" ] && return 0
    [ "$(find $dir -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)" -eq 0 ] && return 0
    echo "[NN GPU$gpu] $(echo $dir | rev | cut -d/ -f1-2 | rev)"
    CUDA_VISIBLE_DEVICES=$gpu python3 vlm/eval_nudenet.py "$dir" --threshold 0.5 2>&1 | tail -1
}

# Baselines (nudity datasets only)
echo "--- Baselines ---"
for name in rab mma p4dn unlearndiff mja_sexual; do
    d="CAS_SpatialCFG/outputs/baselines_v2/$name"
    [ -d "$d" ] && eval_nn "$d" 0 &
done
wait

# Ours sexual concept only
echo "--- Ours (sexual) ---"
gpu=0
for d in CAS_SpatialCFG/outputs/v2_experiments/sexual/*/; do
    [ -d "$d" ] || continue
    eval_nn "$d" $gpu &
    gpu=$(( (gpu + 1) % 8 ))
    [ $(jobs -r | wc -l) -ge 8 ] && wait -n
done
wait

# SAFREE nudity datasets only
echo "--- SAFREE (nudity) ---"
for name in rab mma unlearndiff i2p_sexual mja_sexual; do
    d="CAS_SpatialCFG/outputs/safree_reproduction/$name"
    [ -d "$d" ] || continue
    img_dir="$d"
    [ -d "$d/all" ] && [ "$(ls $d/all/*.png 2>/dev/null | wc -l)" -gt 0 ] && img_dir="$d/all"
    eval_nn "$img_dir" 0 &
done
wait

echo "=== NudeNet COMPLETE $(date) ==="

# Summary
echo ""
echo "=== NUDENET SUMMARY ==="
echo "--- Baselines ---"
for name in rab mma p4dn unlearndiff mja_sexual; do
    d="CAS_SpatialCFG/outputs/baselines_v2/$name"
    [ -f "$d/results_nudenet.txt" ] && echo "$name: $(grep 'Safe Rate:' $d/results_nudenet.txt)"
done
echo "--- Ours Best ---"
for d in CAS_SpatialCFG/outputs/v2_experiments/sexual/rab_*/; do
    [ -f "$d/results_nudenet.txt" ] || continue
    echo "$(basename $d): $(grep 'Safe Rate:' $d/results_nudenet.txt)"
done
echo "--- SAFREE ---"
for name in rab mma unlearndiff i2p_sexual mja_sexual; do
    d="CAS_SpatialCFG/outputs/safree_reproduction/$name"
    img_dir="$d"; [ -d "$d/all" ] && img_dir="$d/all"
    [ -f "$img_dir/results_nudenet.txt" ] && echo "$name: $(grep 'Safe Rate:' $img_dir/results_nudenet.txt)"
done
