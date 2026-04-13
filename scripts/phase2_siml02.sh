#!/bin/bash
# Phase 2 on siml-02: Q16 + Qwen for completed experiments
set -e
cd /mnt/home3/yhgil99/unlearning

eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"

OUT="CAS_SpatialCFG/outputs/v2_experiments"

# Q16 first (fast)
echo "=== Q16 START $(date) ==="
conda activate sdd_copy
gpu=3
for concept_dir in $OUT/violent $OUT/disturbing $OUT/selfharm $OUT/illegal $OUT/hate; do
    for d in $concept_dir/*/; do
        [ -d "$d" ] || continue
        [ -f "$d/results_q16.txt" ] && continue
        [ "$(ls $d/*.png 2>/dev/null | wc -l)" -lt 1 ] && continue
        echo "[Q16 GPU$gpu] $(basename $(dirname $d))/$(basename $d)"
        CUDA_VISIBLE_DEVICES=$gpu python3 vlm/eval_q16.py "$d" --threshold 0.7 2>&1 | tail -1 &
        gpu=$(( (gpu % 6) + 3 ))
        [ $(jobs -r | wc -l) -ge 4 ] && wait -n
    done
done
wait
echo "=== Q16 DONE $(date) ==="

# Qwen eval
echo "=== QWEN START $(date) ==="
conda activate vlm

eval_qwen() {
    local dir=$1 concept=$2 gpu=$3
    [ -f "$dir/results_qwen_${concept}.txt" ] && return 0
    [ "$(ls $dir/*.png 2>/dev/null | wc -l)" -lt 1 ] && return 0
    echo "[QWEN GPU$gpu] $(basename $(dirname $dir))/$(basename $dir) ($concept)"
    CUDA_VISIBLE_DEVICES=$gpu python3 vlm/opensource_vlm_i2p_all.py "$dir" "$concept" qwen 2>&1 | tail -2
}

# Parallel across GPUs 3-6
(for d in $OUT/violent/*/; do eval_qwen "$d" violence 3; done; echo "[violent DONE]") &
(for d in $OUT/disturbing/*/; do eval_qwen "$d" shocking 4; done; echo "[disturbing DONE]") &
(for d in $OUT/selfharm/*/; do eval_qwen "$d" self_harm 5; done; echo "[selfharm DONE]") &
(for d in $OUT/illegal/*/; do eval_qwen "$d" illegal 6; done; echo "[illegal DONE]") &
wait

(for d in $OUT/hate/*/; do eval_qwen "$d" hate 3; done; echo "[hate DONE]") &
(for d in $OUT/harassment/*/; do eval_qwen "$d" harassment 4; done; echo "[harassment DONE]") &
wait

echo "=== SIML-02 PHASE 2 DONE $(date) ==="
