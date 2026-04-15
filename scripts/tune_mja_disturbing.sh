#!/bin/bash
# MJA-disturbing tuning: current best 88%, SAFREE 95%, need to beat it
set -e
cd /mnt/home3/yhgil99/unlearning
eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
conda activate sdd_copy
export PYTHONPATH="/mnt/home3/yhgil99/unlearning/SafeGen:/mnt/home3/yhgil99/.local/lib/python3.10/site-packages:$PYTHONPATH"

OUT="CAS_SpatialCFG/outputs/v2_experiments/disturbing"
EXEMPLAR="CAS_SpatialCFG/exemplars/concepts_v2"
PROMPTS="CAS_SpatialCFG/prompts/mja_disturbing.txt"
TGT="grotesque horror monster demon scream blood"
ANC="beautiful gentle friendly calm smile"

echo "=== MJA-Disturbing Tuning START $(date) ==="

run() {
    local gpu=$1 probe=$2 how=$3 fam=$4 cas=$5 ss=$6
    local sfx="${probe}_${how}"
    [ "$fam" = "1" ] && sfx="${sfx}_fam" || sfx="${sfx}_single"
    sfx="${sfx}_cas${cas}_ss${ss}"
    local outdir="$OUT/mja_${sfx}"
    
    [ -d "$outdir" ] && [ "$(ls $outdir/*.png 2>/dev/null | wc -l)" -gt 5 ] && return 0

    local fa=""
    if [ "$fam" = "1" ]; then
        fa="--family_guidance --family_config ${EXEMPLAR}/disturbing/clip_grouped.pt"
    elif [ "$probe" != "text" ]; then
        fa="--clip_embeddings ${EXEMPLAR}/disturbing/clip_exemplar_projected.pt"
    fi

    PYTHONPATH=/mnt/home3/yhgil99/unlearning/SafeGen:$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=$gpu python3 -m safegen.generate_family \
        --prompts "$PROMPTS" --outdir "$outdir" \
        --probe_mode $probe --how_mode $how \
        --cas_threshold $cas --safety_scale $ss \
        --target_concepts $TGT --anchor_concepts $ANC \
        --steps 50 --seed 42 --cfg_scale 7.5 $fa 2>&1 | tail -2
    echo "[GEN GPU$gpu] $sfx ($(ls $outdir/*.png 2>/dev/null | wc -l) imgs)"
}

# Aggressive grid: higher ss, lower cas
# GPU 0: both ainp, aggressive ss
(
for cas in 0.3 0.4; do
    for ss in 1.5 2.0 2.5; do
        run 0 both anchor_inpaint 0 $cas $ss
    done
done
) &

# GPU 1: both ainp fam, aggressive ss
(
for cas in 0.3 0.4; do
    for ss in 1.5 2.0 2.5; do
        run 1 both anchor_inpaint 1 $cas $ss
    done
done
) &

# GPU 2: text ainp, aggressive ss
(
for cas in 0.3 0.4; do
    for ss in 1.5 2.0 2.5; do
        run 2 text anchor_inpaint 0 $cas $ss
    done
done
) &

# GPU 3: text ainp fam, aggressive ss
(
for cas in 0.3 0.4; do
    for ss in 1.5 2.0 2.5; do
        run 3 text anchor_inpaint 1 $cas $ss
    done
done
) &

# GPU 4: image probe, aggressive
(
for cas in 0.3 0.4; do
    for ss in 1.5 2.0 2.5; do
        run 4 image anchor_inpaint 0 $cas $ss
        run 4 image anchor_inpaint 1 $cas $ss
    done
done
) &

# GPU 5: both with very aggressive ss
(
for cas in 0.2 0.3; do
    for ss in 3.0 4.0; do
        run 5 both anchor_inpaint 0 $cas $ss
        run 5 both anchor_inpaint 1 $cas $ss
    done
done
) &

# GPU 6: existing missed combos
(
for cas in 0.5 0.6; do
    for ss in 1.5 2.0; do
        run 6 both anchor_inpaint 0 $cas $ss
        run 6 both anchor_inpaint 1 $cas $ss
    done
done
) &

# GPU 7: text with very aggressive
(
for cas in 0.2 0.3; do
    for ss in 3.0 4.0; do
        run 7 text anchor_inpaint 0 $cas $ss
        run 7 text anchor_inpaint 1 $cas $ss
    done
done
) &

wait
echo "=== Generation COMPLETE $(date) ==="

# Auto eval: Qwen for all new dirs
echo "=== Qwen Eval ==="
conda activate vlm 2>/dev/null || true
for d in $OUT/mja_*/; do
    [ -f "$d/results_qwen3_vl_shocking.txt" ] && continue
    [ "$(ls $d/*.png 2>/dev/null | wc -l)" -eq 0 ] && continue
    echo "[Qwen] $(basename $d)"
    CUDA_VISIBLE_DEVICES=0 python3 vlm/opensource_vlm_i2p_all.py "$d" shocking qwen 2>&1 | tail -2
done

# Summary
echo ""
echo "=== MJA-DISTURBING RESULTS (sorted by SR) ==="
for d in $OUT/mja_*/; do
    f="$d/results_qwen3_vl_shocking.txt"
    [ -f "$f" ] || continue
    sr=$(grep 'SR (Safe+Partial)' "$f" | head -1)
    echo "$(basename $d): $sr"
done | sort -t'(' -k2 -rn

echo "=== TUNING COMPLETE $(date) ==="
