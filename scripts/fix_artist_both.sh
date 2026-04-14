#!/bin/bash
set -e
cd /mnt/home3/yhgil99/unlearning
eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
conda activate sdd_copy
export PYTHONPATH="/mnt/home3/yhgil99/unlearning/SafeGen:$PYTHONPATH"

OUT="CAS_SpatialCFG/outputs/v2_experiments/artist"
PROMPTS="CAS_SpatialCFG/prompts/artists"

# Artist text probe worked fine. For "both" probe, artists don't have image exemplars.
# So we skip "both" for artists — text probe is the correct approach for style.
# Instead, let's run the missing Qwen evaluations for existing artist outputs.

echo "=== Qwen eval for all artist outputs ==="
conda activate vlm 2>/dev/null || true

for d in $OUT/*/; do
    [ -d "$d" ] || continue
    [ "$(ls $d/*.png 2>/dev/null | wc -l)" -eq 0 ] && continue
    artist=$(basename $d | cut -d'_' -f1)
    style="style_${artist}"
    [ -f "$d/results_qwen3_vl_${style}.txt" ] && continue
    echo "[Qwen] $(basename $d)"
    CUDA_VISIBLE_DEVICES=0 python3 vlm/opensource_vlm_i2p_all.py "$d" "$style" qwen 2>&1 | tail -2
done

echo "=== Q16 eval for all artist outputs ==="
conda activate sdd_copy
export PYTHONPATH="/mnt/home3/yhgil99/.local/lib/python3.10/site-packages:$PYTHONPATH"

for d in $OUT/*/; do
    [ -d "$d" ] || continue
    [ "$(ls $d/*.png 2>/dev/null | wc -l)" -eq 0 ] && continue
    [ -f "$d/results_q16.txt" ] && continue
    echo "[Q16] $(basename $d)"
    CUDA_VISIBLE_DEVICES=1 python3 vlm/eval_q16.py "$d" --threshold 0.7 2>&1 | tail -1
done

echo "=== Artist fix COMPLETE $(date) ==="
