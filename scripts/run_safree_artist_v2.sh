#!/bin/bash
set -e
PYTHON="/mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10"
SAFREE="/mnt/home3/yhgil99/unlearning/SAFREE"
SAVE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/safree_reproduction"
PROMPTS="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/artists"
cd $SAFREE

echo "=== SAFREE Artist v2 START $(date) ==="

run_artist() {
    local gpu=$1 artist=$2 concept=$3
    local outdir="$SAVE/artist_${artist}"
    mkdir -p "$outdir"
    [ "$(find $outdir -name '*.png' 2>/dev/null | wc -l)" -gt 80 ] && echo "[SKIP] $artist" && return 0
    
    # Run 3 times with different seeds for 3 samples per prompt
    for seed in 42 123 456; do
        echo "[GPU$gpu] $artist seed=$seed"
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON -s gen_safree_i2p_concepts.py \
            --prompt_file "$PROMPTS/${artist}.txt" \
            --concepts "$concept" \
            --outdir "$outdir" \
            --no_concept_subdir \
            --seed $seed \
            --model_id CompVis/stable-diffusion-v1-4 \
            --safree --svf --lra --device cuda:0 2>&1 | tail -3
    done
    echo "[DONE GPU$gpu] artist_$artist ($(find $outdir -name '*.png' 2>/dev/null | wc -l) imgs)"
}

run_artist 0 vangogh "artist-vangogh" &
run_artist 1 picasso "artist-picasso" &
run_artist 2 monet "artist-monet" &
run_artist 3 rembrandt "artist-rembrandt" &
run_artist 4 warhol "artist-warhol" &
run_artist 5 hopper "artist-hopper" &
wait

echo "=== SAFREE Artist v2 COMPLETE $(date) ==="
