#!/bin/bash
set -e
PYTHON="/mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10"
SAFREE="/mnt/home3/yhgil99/unlearning/unlearning-baselines/SAFREE_github"
SAVE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/safree_reproduction"
PROMPTS="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/artists"
cd $SAFREE

echo "=== SAFREE Artist START $(date) ==="

run_artist() {
    local gpu=$1 artist=$2
    local outdir="$SAVE/artist_${artist}"
    mkdir -p "$outdir/safe" "$outdir/unsafe" "$outdir/all"
    [ "$(find $outdir -name '*.png' 2>/dev/null | wc -l)" -gt 5 ] && echo "[SKIP] $artist" && return 0
    echo "[GPU$gpu] SAFREE artist: $artist"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -s generate_safree.py \
        --config ./configs/sd_config.json \
        --data "$PROMPTS/${artist}.txt" \
        --nudenet-path ./pretrained/nudenet_classifier_model.onnx \
        --num-samples 1 --erase-id std \
        --model_id CompVis/stable-diffusion-v1-4 \
        --category artist-VanGogh \
        --save-dir "$outdir" \
        --safree -svf -lra 2>&1 | tail -3
    echo "[DONE GPU$gpu] artist_$artist ($(find $outdir -name '*.png' 2>/dev/null | wc -l) imgs)"
}

run_artist 3 vangogh &
run_artist 4 picasso &
run_artist 5 monet &
run_artist 6 rembrandt &
run_artist 7 warhol &
wait

# Hopper on freed GPU
run_artist 3 hopper
wait

echo "=== SAFREE Artist COMPLETE $(date) ==="
