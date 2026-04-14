#!/bin/bash
set -e
cd /mnt/home3/yhgil99/unlearning
SAVE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/safree_reproduction"

# Wait for SAFREE artist to finish
echo "Waiting for SAFREE artist generation..."
while ps aux | grep yhgil99 | grep gen_safree_i2p | grep -v grep > /dev/null 2>&1; do
    sleep 30
done
echo "SAFREE artist done! Starting eval..."

# Q16
eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
conda activate sdd_copy
export PYTHONPATH="/mnt/home3/yhgil99/.local/lib/python3.10/site-packages:$PYTHONPATH"

echo "=== SAFREE Artist Q16 ==="
for d in $SAVE/artist_*/; do
    [ "$(find $d -name '*.png' 2>/dev/null | wc -l)" -eq 0 ] && continue
    [ -f "$d/results_q16.txt" ] && continue
    echo "[Q16] $(basename $d)"
    CUDA_VISIBLE_DEVICES=6 python3 vlm/eval_q16.py "$d" --threshold 0.7 2>&1 | tail -1
done

# VQA
echo "=== SAFREE Artist VQA ==="
for artist in vangogh picasso monet rembrandt warhol hopper; do
    d="$SAVE/artist_${artist}"
    [ "$(find $d -name '*.png' 2>/dev/null | wc -l)" -eq 0 ] && continue
    [ -f "$d/results_vqascore.txt" ] && continue
    echo "[VQA] artist_$artist"
    CUDA_VISIBLE_DEVICES=7 python3 vlm/eval_vqascore.py "$d" \
        --prompts "CAS_SpatialCFG/prompts/artists/${artist}.txt" 2>&1 | tail -2
done

# Qwen
echo "=== SAFREE Artist Qwen ==="
conda activate vlm 2>/dev/null || true
for artist in vangogh picasso monet rembrandt warhol hopper; do
    d="$SAVE/artist_${artist}"
    style="style_${artist}"
    [ "$(find $d -name '*.png' 2>/dev/null | wc -l)" -eq 0 ] && continue
    [ -f "$d/results_qwen3_vl_${style}.txt" ] && continue
    echo "[Qwen] artist_$artist"
    CUDA_VISIBLE_DEVICES=6 python3 vlm/opensource_vlm_i2p_all.py "$d" "$style" qwen 2>&1 | tail -2
done

echo "=== SAFREE Artist Eval COMPLETE $(date) ==="
