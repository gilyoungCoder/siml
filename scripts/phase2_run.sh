#!/bin/bash
# Phase 2: Q16 + Qwen evaluation on ALL outputs
set -e
cd /mnt/home3/yhgil99/unlearning

eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"

OUT="CAS_SpatialCFG/outputs/v2_experiments"
BL="CAS_SpatialCFG/outputs/baselines_v2"

# ============================================================
# Step 1: Q16 (fast, sdd_copy env)
# ============================================================
echo "=== Q16 EVALUATION START $(date) ==="
conda activate sdd_copy

eval_q16() {
    local dir=$1 gpu=$2
    [ -f "$dir/results_q16.txt" ] && return 0
    [ "$(ls $dir/*.png 2>/dev/null | wc -l)" -lt 1 ] && return 0
    echo "[Q16 GPU$gpu] $(basename $(dirname $dir))/$(basename $dir)"
    CUDA_VISIBLE_DEVICES=$gpu python3 vlm/eval_q16.py "$dir" --threshold 0.7 2>&1 | tail -1
}

# Baselines
for d in $BL/*/; do
    eval_q16 "$d" 0 &
done
wait

# Ours - parallel across GPUs
gpu=0
for concept_dir in $OUT/*/; do
    for d in $concept_dir/*/; do
        [ -d "$d" ] || continue
        eval_q16 "$d" $gpu &
        gpu=$(( (gpu + 1) % 8 ))
        [ $(jobs -r | wc -l) -ge 8 ] && wait -n
    done
done
wait
echo "=== Q16 COMPLETE $(date) ==="

# ============================================================
# Step 2: Qwen VLM (slower, vlm env)
# ============================================================
echo "=== QWEN EVALUATION START $(date) ==="
conda activate vlm

eval_qwen() {
    local dir=$1 concept=$2 gpu=$3
    local rf="$dir/results_qwen_${concept}.txt"
    [ -f "$rf" ] && return 0
    [ "$(ls $dir/*.png 2>/dev/null | wc -l)" -lt 1 ] && return 0
    echo "[QWEN GPU$gpu] $(basename $(dirname $dir))/$(basename $dir) ($concept)"
    CUDA_VISIBLE_DEVICES=$gpu python3 vlm/opensource_vlm_i2p_all.py "$dir" "$concept" qwen 2>&1 | tail -2
}

# Baselines - nudity datasets
(
eval_qwen "$BL/rab" nudity 0
eval_qwen "$BL/mma" nudity 0
eval_qwen "$BL/p4dn" nudity 0
eval_qwen "$BL/unlearndiff" nudity 0
echo "[BL nudity DONE]"
) &

# Baselines - MJA
(
eval_qwen "$BL/mja_sexual" nudity 1
eval_qwen "$BL/mja_violent" violence 1
eval_qwen "$BL/mja_disturbing" shocking 1
eval_qwen "$BL/mja_illegal" illegal 1
echo "[BL MJA DONE]"
) &
wait

# Ours - map concept dirs to qwen concept names
eval_concept_dir() {
    local concept_dir=$1 qwen_concept=$2 gpu=$3
    for d in $concept_dir/*/; do
        [ -d "$d" ] || continue
        eval_qwen "$d" "$qwen_concept" $gpu
    done
}

# Run 7 concepts in parallel across 7 GPUs
eval_concept_dir "$OUT/sexual"     "nudity"    0 &
eval_concept_dir "$OUT/violent"    "violence"  1 &
eval_concept_dir "$OUT/disturbing" "shocking"  2 &
eval_concept_dir "$OUT/illegal"    "illegal"   3 &
eval_concept_dir "$OUT/harassment" "harassment" 4 &
eval_concept_dir "$OUT/hate"       "hate"      5 &
eval_concept_dir "$OUT/selfharm"   "self_harm" 6 &
wait

echo "=== QWEN COMPLETE $(date) ==="
echo "=== PHASE 2 ALL DONE $(date) ==="
