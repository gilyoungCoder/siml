#!/bin/bash
# Run SafeGen on all nudity benchmarks (Ring-A-Bell, MMA, P4DN, UnlearnDiff)
# Usage: bash scripts/run_nudity.sh [GPU_ID]

GPU=${1:-0}
CLIP_EMB="configs/exemplars/sexual/clip_exemplar_projected.pt"

echo "=== SafeGen Nudity Erasing (GPU $GPU) ==="

for dataset in ringabell mma p4dn unlearndiff; do
    echo "--- $dataset ---"
    CUDA_VISIBLE_DEVICES=$GPU python -m safegen.generate \
        --prompts prompts/${dataset}.txt \
        --outdir outputs/safegen_${dataset} \
        --probe_mode both \
        --clip_embeddings $CLIP_EMB \
        --how_mode anchor_inpaint \
        --safety_scale 1.0 \
        --cas_threshold 0.6 \
        --attn_threshold 0.3 \
        --seed 42
done

echo "=== Baseline ==="
for dataset in ringabell mma p4dn unlearndiff; do
    echo "--- Baseline $dataset ---"
    CUDA_VISIBLE_DEVICES=$GPU python -m safegen.generate_baseline \
        --prompts prompts/${dataset}.txt \
        --outdir outputs/baseline_${dataset} \
        --seed 42
done

echo "Done!"
