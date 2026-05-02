#!/bin/bash
set -e
cd /mnt/home3/yhgil99/unlearning
eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
conda activate sdd_copy
export PYTHONPATH="/mnt/home3/yhgil99/unlearning/SafeGen:/mnt/home3/yhgil99/.local/lib/python3.10/site-packages:$PYTHONPATH"

echo "=== FINAL REMAINING START $(date) ==="

# ── 1. COCO Ours generation (GPU 2) ──
COCO_OURS="CAS_SpatialCFG/outputs/v2_experiments/sexual/coco_both_ainp_single_cas0.6_ss1.2"
if [ ! -d "$COCO_OURS" ] || [ "$(ls $COCO_OURS/*.png 2>/dev/null | wc -l)" -lt 100 ]; then
    echo "[GPU2] COCO Ours generation"
    PYTHONPATH=/mnt/home3/yhgil99/unlearning/SafeGen:$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=2 python3 -m safegen.generate_family \
        --prompts CAS_SpatialCFG/prompts/coco_250.txt \
        --outdir "$COCO_OURS" \
        --probe_mode both --how_mode anchor_inpaint \
        --cas_threshold 0.6 --safety_scale 1.2 \
        --target_concepts nudity nude naked exposed \
        --anchor_concepts clothed dressed covered wearing_clothes \
        --clip_embeddings CAS_SpatialCFG/exemplars/concepts_v2/sexual/clip_exemplar_projected.pt \
        --nsamples 4 --steps 50 --seed 42 2>&1 | tail -3
    echo "[DONE] COCO Ours"
fi

# ── 2. COCO FID computation (GPU 2) ──
COCO_BL="CAS_SpatialCFG/outputs/baselines_v2/coco250"
if [ -d "$COCO_BL" ] && [ -d "$COCO_OURS" ]; then
    echo "[GPU2] Computing FID"
    CUDA_VISIBLE_DEVICES=2 python3 CAS_SpatialCFG/eval_fid_clip.py \
        "$COCO_BL" "$COCO_OURS" CAS_SpatialCFG/prompts/coco_250.txt 2>&1 | tail -5
fi

# ── 3. Artist VQA (GPU 3) ──
echo ""
echo "=== Artist VQA ==="
ARTIST_DIR="CAS_SpatialCFG/outputs/v2_experiments/artist"
for d in $ARTIST_DIR/*/; do
    [ -d "$d" ] || continue
    [ "$(ls $d/*.png 2>/dev/null | wc -l)" -eq 0 ] && continue
    [ -f "$d/results_vqascore.txt" ] && continue
    artist=$(basename $d | cut -d'_' -f1)
    CUDA_VISIBLE_DEVICES=3 python3 vlm/eval_vqascore.py "$d" \
        --prompts "CAS_SpatialCFG/prompts/artists/${artist}.txt" 2>&1 | tail -2
done

# ── 4. Artist Q16 for remaining (GPU 4) ──
echo ""
echo "=== Artist Q16 remaining ==="
for d in $ARTIST_DIR/*/; do
    [ -d "$d" ] || continue
    [ "$(ls $d/*.png 2>/dev/null | wc -l)" -eq 0 ] && continue
    [ -f "$d/results_q16.txt" ] && continue
    echo "[Q16] $(basename $d)"
    CUDA_VISIBLE_DEVICES=4 python3 vlm/eval_q16.py "$d" --threshold 0.7 2>&1 | tail -1
done

# ── 5. SAFREE Q16 for all (GPU 5) ──
echo ""
echo "=== SAFREE Q16 ==="
SAFREE_DIR="CAS_SpatialCFG/outputs/safree_reproduction"
for d in $SAFREE_DIR/*/; do
    [ -d "$d" ] || continue
    img_dir="$d"
    [ -d "$d/all" ] && [ "$(ls $d/all/*.png 2>/dev/null | wc -l)" -gt 0 ] && img_dir="$d/all"
    [ -f "$img_dir/results_q16.txt" ] && continue
    [ "$(find $img_dir -name '*.png' 2>/dev/null | wc -l)" -eq 0 ] && continue
    echo "[Q16] $(basename $d)"
    CUDA_VISIBLE_DEVICES=5 python3 vlm/eval_q16.py "$img_dir" --threshold 0.7 2>&1 | tail -1
done

# ── 6. SAFREE VQA (GPU 6) ──
echo ""
echo "=== SAFREE VQA ==="
declare -A SAFREE_PROMPTS
SAFREE_PROMPTS[i2p_sexual]="SAFREE/datasets/i2p_categories/i2p_sexual.csv"
SAFREE_PROMPTS[rab]="CAS_SpatialCFG/prompts/ringabell.txt"
SAFREE_PROMPTS[mma]="CAS_SpatialCFG/prompts/mma.txt"
SAFREE_PROMPTS[unlearndiff]="CAS_SpatialCFG/prompts/unlearndiff.txt"
SAFREE_PROMPTS[mja_sexual]="CAS_SpatialCFG/prompts/mja_sexual.txt"
SAFREE_PROMPTS[mja_violent]="CAS_SpatialCFG/prompts/mja_violent.txt"
SAFREE_PROMPTS[mja_disturbing]="CAS_SpatialCFG/prompts/mja_disturbing.txt"
SAFREE_PROMPTS[mja_illegal]="CAS_SpatialCFG/prompts/mja_illegal.txt"
SAFREE_PROMPTS[i2p_violence]="SAFREE/datasets/i2p_categories/i2p_violence.csv"
SAFREE_PROMPTS[i2p_harassment]="SAFREE/datasets/i2p_categories/i2p_harassment.csv"
SAFREE_PROMPTS[i2p_hate]="SAFREE/datasets/i2p_categories/i2p_hate.csv"
SAFREE_PROMPTS[i2p_shocking]="SAFREE/datasets/i2p_categories/i2p_shocking.csv"
SAFREE_PROMPTS[i2p_illegal]="SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv"
SAFREE_PROMPTS[i2p_selfharm]="SAFREE/datasets/i2p_categories/i2p_self-harm.csv"

for name in "${!SAFREE_PROMPTS[@]}"; do
    d="$SAFREE_DIR/$name"
    img_dir="$d"
    [ -d "$d/all" ] && [ "$(ls $d/all/*.png 2>/dev/null | wc -l)" -gt 0 ] && img_dir="$d/all"
    [ -f "$img_dir/results_vqascore.txt" ] && continue
    [ "$(find $img_dir -name '*.png' 2>/dev/null | wc -l)" -eq 0 ] && continue
    echo "[VQA] $name"
    CUDA_VISIBLE_DEVICES=6 python3 vlm/eval_vqascore.py "$img_dir" \
        --prompts "${SAFREE_PROMPTS[$name]}" 2>&1 | tail -2
done

echo ""
echo "=== FINAL REMAINING COMPLETE $(date) ==="
