#!/bin/bash
set -e
cd /mnt/home3/yhgil99/unlearning
eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
conda activate sdd_copy
export PYTHONPATH="/mnt/home3/yhgil99/unlearning/SafeGen:$PYTHONPATH"

EXEMPLAR="CAS_SpatialCFG/exemplars/concepts_v2"
OUT="CAS_SpatialCFG/outputs/v2_experiments/multi"
mkdir -p $OUT

echo "=== Multi-Concept v2 START $(date) ==="

# GPU 6: nudity+violence, nudity+shocking
(
echo "[GPU6] nudity+violence"
PYTHONPATH=/mnt/home3/yhgil99/unlearning/SafeGen:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=6 python3 -m safegen.generate_family \
    --prompts SAFREE/datasets/i2p_categories/i2p_sexual.csv \
    --outdir "$OUT/nude_violence_both_ainp" \
    --probe_mode text --how_mode anchor_inpaint \
    --cas_threshold 0.4 --safety_scale 1.2 \
    --target_concepts nudity nude naked violence blood knife weapon \
    --anchor_concepts clothed dressed clean healthy handshake toy \
    --steps 50 --seed 42 --cfg_scale 7.5 2>&1 | tail -3
echo "[DONE GPU6] nude+violence"

echo "[GPU6] nudity+shocking"
PYTHONPATH=/mnt/home3/yhgil99/unlearning/SafeGen:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=6 python3 -m safegen.generate_family \
    --prompts SAFREE/datasets/i2p_categories/i2p_sexual.csv \
    --outdir "$OUT/nude_shocking_both_ainp" \
    --probe_mode text --how_mode anchor_inpaint \
    --cas_threshold 0.4 --safety_scale 1.2 \
    --target_concepts nudity nude naked grotesque horror monster scream \
    --anchor_concepts clothed dressed beautiful gentle calm smile \
    --steps 50 --seed 42 --cfg_scale 7.5 2>&1 | tail -3
echo "[DONE GPU6] nude+shocking"
) &

# GPU 7: nudity+harassment, nudity+hate
(
echo "[GPU7] nudity+harassment"
PYTHONPATH=/mnt/home3/yhgil99/unlearning/SafeGen:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=7 python3 -m safegen.generate_family \
    --prompts SAFREE/datasets/i2p_categories/i2p_sexual.csv \
    --outdir "$OUT/nude_harassment_both_ainp" \
    --probe_mode text --how_mode anchor_inpaint \
    --cas_threshold 0.4 --safety_scale 1.2 \
    --target_concepts nudity nude naked threatening bully intimidation \
    --anchor_concepts clothed dressed friendly waving greeting smile \
    --steps 50 --seed 42 --cfg_scale 7.5 2>&1 | tail -3
echo "[DONE GPU7] nude+harassment"

echo "[GPU7] nudity+hate"
PYTHONPATH=/mnt/home3/yhgil99/unlearning/SafeGen:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=7 python3 -m safegen.generate_family \
    --prompts SAFREE/datasets/i2p_categories/i2p_sexual.csv \
    --outdir "$OUT/nude_hate_both_ainp" \
    --probe_mode text --how_mode anchor_inpaint \
    --cas_threshold 0.4 --safety_scale 1.0 \
    --target_concepts nudity nude naked nazi swastika hate discrimination \
    --anchor_concepts clothed dressed peace diversity unity community \
    --steps 50 --seed 42 --cfg_scale 7.5 2>&1 | tail -3
echo "[DONE GPU7] nude+hate"
) &

wait
echo "=== Multi-Concept v2 COMPLETE $(date) ==="
