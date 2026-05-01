#!/bin/bash
# Run SafeGen on I2P multi-concept benchmarks
# Usage: bash scripts/run_i2p_multi.sh [GPU_ID]

GPU=${1:-0}

CONCEPTS=(sexual violence harassment hate shocking illegal_activity self-harm)

for concept in "${CONCEPTS[@]}"; do
    echo "=== $concept ==="

    # Map concept to CSV name
    csv_name="i2p_${concept}.csv"

    # Determine probe mode based on concept pack
    CLIP_EMB="configs/exemplars/${concept}/clip_exemplar_projected.pt"
    if [ -f "$CLIP_EMB" ]; then
        PROBE_MODE="both"
        EXTRA_ARGS="--clip_embeddings $CLIP_EMB"
    else
        PROBE_MODE="text"
        EXTRA_ARGS=""
    fi

    CUDA_VISIBLE_DEVICES=$GPU python -m safegen.generate \
        --prompts prompts/${csv_name} \
        --outdir outputs/safegen_${concept} \
        --probe_mode $PROBE_MODE \
        $EXTRA_ARGS \
        --how_mode anchor_inpaint \
        --safety_scale 1.0 \
        --seed 42
done

echo "Done!"
