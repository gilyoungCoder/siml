#!/bin/bash
set -e
cd /mnt/home3/yhgil99/unlearning

# Wait for multi-concept generation to finish
echo "Waiting for multi-concept generation..."
while ps aux | grep yhgil99 | grep 'safegen.generate_family' | grep -v grep > /dev/null 2>&1; do
    sleep 30
done
echo "Multi-concept generation done! Starting eval... $(date)"

OUT="CAS_SpatialCFG/outputs/v2_experiments/multi"

# Q16
eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
conda activate sdd_copy
export PYTHONPATH="/mnt/home3/yhgil99/.local/lib/python3.10/site-packages:$PYTHONPATH"

echo "=== Multi Q16 ==="
for d in $OUT/*/; do
    [ "$(ls $d/*.png 2>/dev/null | wc -l)" -eq 0 ] && continue
    [ -f "$d/results_q16.txt" ] && continue
    echo "[Q16] $(basename $d)"
    CUDA_VISIBLE_DEVICES=6 python3 vlm/eval_q16.py "$d" --threshold 0.7 2>&1 | tail -1
done

# Qwen (nudity + 2nd concept)
echo "=== Multi Qwen ==="
conda activate vlm 2>/dev/null || true
for d in $OUT/*/; do
    [ "$(ls $d/*.png 2>/dev/null | wc -l)" -eq 0 ] && continue
    name=$(basename $d)
    for concept in nudity violence shocking harassment hate; do
        [ -f "$d/results_qwen3_vl_${concept}.txt" ] && continue
        echo "[Qwen] $name ($concept)"
        CUDA_VISIBLE_DEVICES=7 python3 vlm/opensource_vlm_i2p_all.py "$d" "$concept" qwen 2>&1 | tail -2
    done
done

echo "=== Multi Eval COMPLETE $(date) ==="
