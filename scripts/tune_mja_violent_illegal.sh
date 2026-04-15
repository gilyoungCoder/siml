#!/bin/bash
set -e
cd /mnt/home3/yhgil99/unlearning
eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
conda activate sdd_copy
export PYTHONPATH="/mnt/home3/yhgil99/unlearning/SafeGen:/mnt/home3/yhgil99/.local/lib/python3.10/site-packages:$PYTHONPATH"

EXEMPLAR="CAS_SpatialCFG/exemplars/concepts_v2"
OUT="CAS_SpatialCFG/outputs/v2_experiments"

run() {
    local gpu=$1 concept=$2 prompts=$3 probe=$4 how=$5 fam=$6 cas=$7 ss=$8
    shift 8; local tgt="$1" anc="$2"
    local sfx="${probe}_${how}"; [ "$fam" = "1" ] && sfx="${sfx}_fam" || sfx="${sfx}_single"
    sfx="${sfx}_cas${cas}_ss${ss}"
    local outdir="$OUT/$concept/mja_${sfx}"
    [ -d "$outdir" ] && [ "$(ls $outdir/*.png 2>/dev/null | wc -l)" -gt 5 ] && return 0
    local fa=""
    if [ "$fam" = "1" ]; then fa="--family_guidance --family_config ${EXEMPLAR}/${concept}/clip_grouped.pt"
    elif [ "$probe" != "text" ]; then fa="--clip_embeddings ${EXEMPLAR}/${concept}/clip_exemplar_projected.pt"; fi
    PYTHONPATH=/mnt/home3/yhgil99/unlearning/SafeGen:$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=$gpu python3 -m safegen.generate_family \
        --prompts "$prompts" --outdir "$outdir" --probe_mode $probe --how_mode $how \
        --cas_threshold $cas --safety_scale $ss --target_concepts $tgt --anchor_concepts $anc \
        --steps 50 --seed 42 --cfg_scale 7.5 $fa 2>&1 | tail -2
    echo "[DONE GPU$gpu] $concept $sfx ($(ls $outdir/*.png 2>/dev/null | wc -l) imgs)"
}

VT="blood knife gun weapon wound fight"
VA="clean toy_gun handshake healthy"
IT="drug pills marijuana smoke trafficking"
IA="medicine vitamins tea coffee portrait"

echo "=== MJA Violent+Illegal Tuning START $(date) ==="

# GPU 3-4: Violent tuning (current best 59%, SAFREE 52%)
(
for cas in 0.3 0.4 0.5; do
    for ss in 1.0 1.5 2.0 2.5; do
        run 3 violent CAS_SpatialCFG/prompts/mja_violent.txt both anchor_inpaint 0 $cas $ss "$VT" "$VA"
        run 3 violent CAS_SpatialCFG/prompts/mja_violent.txt both anchor_inpaint 1 $cas $ss "$VT" "$VA"
    done
done
) &

(
for cas in 0.3 0.4 0.5; do
    for ss in 1.0 1.5 2.0 2.5; do
        run 4 violent CAS_SpatialCFG/prompts/mja_violent.txt image anchor_inpaint 0 $cas $ss "$VT" "$VA"
        run 4 violent CAS_SpatialCFG/prompts/mja_violent.txt image anchor_inpaint 1 $cas $ss "$VT" "$VA"
    done
done
) &

(
for cas in 0.3 0.4 0.5; do
    for ss in 1.0 1.5 2.0 2.5; do
        run 5 violent CAS_SpatialCFG/prompts/mja_violent.txt text anchor_inpaint 0 $cas $ss "$VT" "$VA"
        run 5 violent CAS_SpatialCFG/prompts/mja_violent.txt text anchor_inpaint 1 $cas $ss "$VT" "$VA"
    done
done
) &

# GPU 6-7: Illegal tuning (current best 78%, SAFREE 77%)
(
for cas in 0.3 0.4 0.5; do
    for ss in 1.0 1.5 2.0 2.5; do
        run 6 illegal CAS_SpatialCFG/prompts/mja_illegal.txt both anchor_inpaint 0 $cas $ss "$IT" "$IA"
        run 6 illegal CAS_SpatialCFG/prompts/mja_illegal.txt both anchor_inpaint 1 $cas $ss "$IT" "$IA"
    done
done
) &

(
for cas in 0.3 0.4 0.5; do
    for ss in 1.0 1.5 2.0 2.5; do
        run 7 illegal CAS_SpatialCFG/prompts/mja_illegal.txt image anchor_inpaint 0 $cas $ss "$IT" "$IA"
        run 7 illegal CAS_SpatialCFG/prompts/mja_illegal.txt image anchor_inpaint 1 $cas $ss "$IT" "$IA"
    done
done
) &

wait
echo "=== Generation COMPLETE $(date) ==="

# Qwen eval
echo "=== Qwen Eval ==="
conda activate vlm 2>/dev/null || true
for d in $OUT/violent/mja_*/; do
    [ -f "$d/results_qwen3_vl_violence.txt" ] && continue
    [ "$(ls $d/*.png 2>/dev/null | wc -l)" -eq 0 ] && continue
    echo "[Qwen] $(basename $d)"
    CUDA_VISIBLE_DEVICES=3 python3 vlm/opensource_vlm_i2p_all.py "$d" violence qwen 2>&1 | tail -2
done
for d in $OUT/illegal/mja_*/; do
    [ -f "$d/results_qwen3_vl_illegal.txt" ] && continue
    [ "$(ls $d/*.png 2>/dev/null | wc -l)" -eq 0 ] && continue
    echo "[Qwen] $(basename $d)"
    CUDA_VISIBLE_DEVICES=3 python3 vlm/opensource_vlm_i2p_all.py "$d" illegal qwen 2>&1 | tail -2
done

echo ""
echo "=== MJA VIOLENT RESULTS ==="
for d in $OUT/violent/mja_*/; do
    f="$d/results_qwen3_vl_violence.txt"; [ -f "$f" ] || continue
    echo "$(basename $d): $(grep 'SR (Safe+Partial)' $f)"
done | sort -t'(' -k2 -rn | head -10

echo "=== MJA ILLEGAL RESULTS ==="
for d in $OUT/illegal/mja_*/; do
    f="$d/results_qwen3_vl_illegal.txt"; [ -f "$f" ] || continue
    echo "$(basename $d): $(grep 'SR (Safe+Partial)' $f)"
done | sort -t'(' -k2 -rn | head -10

echo "=== TUNING COMPLETE $(date) ==="
