#!/bin/bash
# Run ALL missing evaluations
# Usage: bash eval_all_missing.sh <gpu_id>
# Designed to be run on any server
set -e
cd /mnt/home3/yhgil99/unlearning
eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"

GPU=$1
TASK=$2  # q16_ours, q16_safree, qwen_safree_artist, vqa_safree_artist

case $TASK in
q16_ours)
    conda activate sdd_copy
    export PYTHONPATH="/mnt/home3/yhgil99/.local/lib/python3.10/site-packages:$PYTHONPATH"
    echo "=== Q16 Ours missing (GPU $GPU) ==="
    for d in CAS_SpatialCFG/outputs/v2_experiments/*/*/; do
        [ "$(ls "$d"*.png 2>/dev/null | wc -l)" -eq 0 ] && continue
        [ -f "$d/results_q16.txt" ] && continue
        echo "[Q16] $d"
        CUDA_VISIBLE_DEVICES=$GPU python3 vlm/eval_q16.py "$d" --threshold 0.7 2>&1 | tail -1
    done
    ;;

q16_safree)
    conda activate sdd_copy
    export PYTHONPATH="/mnt/home3/yhgil99/.local/lib/python3.10/site-packages:$PYTHONPATH"
    echo "=== Q16 SAFREE missing (GPU $GPU) ==="
    for d in CAS_SpatialCFG/outputs/safree_reproduction/*/; do
        imgs=$(find "$d" -name '*.png' 2>/dev/null | wc -l)
        [ "$imgs" -eq 0 ] && continue
        # Check root and subdirs
        has_q16=0
        [ -f "$d/results_q16.txt" ] && has_q16=1
        for sub in "$d"/*/; do [ -f "$sub/results_q16.txt" ] && has_q16=1; done
        [ "$has_q16" -eq 1 ] && continue
        # Find where images are
        img_dir="$d"
        [ -d "$d/all" ] && [ "$(ls $d/all/*.png 2>/dev/null | wc -l)" -gt 0 ] && img_dir="$d/all"
        [ -d "$d/seed_42" ] && img_dir="$d/seed_42"  # artist uses seed subdirs
        echo "[Q16 SAFREE] $(basename $d) -> $img_dir"
        CUDA_VISIBLE_DEVICES=$GPU python3 vlm/eval_q16.py "$img_dir" --threshold 0.7 2>&1 | tail -1
    done
    ;;

vqa_safree_artist)
    conda activate sdd_copy
    echo "=== VQA SAFREE artist (GPU $GPU) ==="
    for artist in vangogh picasso monet rembrandt warhol hopper; do
        d="CAS_SpatialCFG/outputs/safree_reproduction/artist_${artist}"
        [ ! -d "$d" ] && continue
        img_dir="$d/seed_42"
        [ ! -d "$img_dir" ] && img_dir="$d"
        [ -f "$img_dir/results_vqascore.txt" ] && continue
        echo "[VQA] artist_$artist"
        CUDA_VISIBLE_DEVICES=$GPU python3 vlm/eval_vqascore.py "$img_dir" \
            --prompts "CAS_SpatialCFG/prompts/artists/${artist}.txt" 2>&1 | tail -2
    done
    ;;

qwen_safree_artist)
    conda activate vlm 2>/dev/null || conda activate sdd_copy
    echo "=== Qwen SAFREE artist (GPU $GPU) ==="
    for artist in vangogh picasso monet rembrandt warhol hopper; do
        d="CAS_SpatialCFG/outputs/safree_reproduction/artist_${artist}"
        [ ! -d "$d" ] && continue
        img_dir="$d/seed_42"
        [ ! -d "$img_dir" ] && img_dir="$d"
        style="style_${artist}"
        [ -f "$img_dir/results_qwen3_vl_${style}.txt" ] && continue
        echo "[Qwen] artist_$artist ($style)"
        CUDA_VISIBLE_DEVICES=$GPU python3 vlm/opensource_vlm_i2p_all.py "$img_dir" "$style" qwen 2>&1 | tail -2
    done
    ;;
esac

echo "=== $TASK COMPLETE $(date) ==="
