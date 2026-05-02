#!/bin/bash
# Phase 1 PARALLEL: Maximize GPU usage across all available GPUs
# Each GPU runs one experiment at a time, moves to next when done
set -e
cd /mnt/home3/yhgil99/unlearning

eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
conda activate sdd_copy
export PYTHONPATH="/mnt/home3/yhgil99/unlearning/SafeGen:$PYTHONPATH"

EXEMPLAR="CAS_SpatialCFG/exemplars/concepts_v2"
OUT="CAS_SpatialCFG/outputs/v2_experiments"
mkdir -p $OUT

run() {
    local gpu=$1 name=$2 prompts=$3 concept=$4 probe=$5 how=$6 fam=$7 cas=$8 ss=$9
    shift 9
    local tgt="$1" anc="$2"

    local sfx="${probe}_${how}"
    [ "$fam" = "1" ] && sfx="${sfx}_fam" || sfx="${sfx}_single"
    sfx="${sfx}_cas${cas}_ss${ss}"
    local out="${OUT}/${concept}/${name}_${sfx}"

    [ -d "$out" ] && [ "$(ls $out/*.png 2>/dev/null | wc -l)" -gt 5 ] && return 0

    local fa=""
    if [ "$fam" = "1" ]; then
        fa="--family_guidance --family_config ${EXEMPLAR}/${concept}/clip_grouped.pt"
    elif [ "$probe" != "text" ]; then
        fa="--clip_embeddings ${EXEMPLAR}/${concept}/clip_exemplar_projected.pt"
    fi

    PYTHONPATH=/mnt/home3/yhgil99/unlearning/SafeGen:$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=$gpu python3 -m safegen.generate_family \
        --prompts "$prompts" --outdir "$out" \
        --probe_mode $probe --how_mode $how \
        --cas_threshold $cas --safety_scale $ss \
        --target_concepts $tgt --anchor_concepts $anc \
        --steps 50 --seed 42 --cfg_scale 7.5 $fa 2>&1 | tail -3
    echo "[DONE GPU$gpu] $(basename $out) $(ls $out/*.png 2>/dev/null | wc -l) imgs"
}

NT="nudity nude naked exposed"
NA="clothed dressed covered wearing_clothes"
VT="blood knife gun weapon wound fight"
VA="clean toy_gun handshake healthy"
DT="grotesque horror monster demon scream blood"
DA="beautiful gentle friendly calm smile"
IT="drug pills marijuana smoke trafficking"
IA="medicine vitamins tea coffee portrait"

echo "=== Phase 1 PARALLEL START $(date) ==="

# ── GPU 0: RAB grid (nudity, 79 prompts) ──
(
for how in anchor_inpaint hybrid; do
  for probe in text both; do
    for fam in 0 1; do
      for cas in 0.4 0.6; do
        for ss in 1.0 1.2; do
          run 0 rab CAS_SpatialCFG/prompts/ringabell.txt sexual $probe $how $fam $cas $ss "$NT" "$NA"
        done
      done
    done
  done
done
echo "[GPU0 COMPLETE]"
) &

# ── GPU 1: MMA best configs (nudity, 1000 prompts) ──
(
for how in anchor_inpaint hybrid; do
  for probe in text both; do
    run 1 mma CAS_SpatialCFG/prompts/mma.txt sexual $probe $how 1 0.6 1.2 "$NT" "$NA"
  done
done
echo "[GPU1 COMPLETE]"
) &

# ── GPU 2: P4DN + UnlearnDiff (nudity) ──
(
for probe in text both; do
  for fam in 0 1; do
    run 2 p4dn CAS_SpatialCFG/prompts/p4dn.txt sexual $probe anchor_inpaint $fam 0.6 1.2 "$NT" "$NA"
    run 2 udiff CAS_SpatialCFG/prompts/unlearndiff.txt sexual $probe anchor_inpaint $fam 0.6 1.2 "$NT" "$NA"
  done
done
echo "[GPU2 COMPLETE]"
) &

# ── GPU 3: MJA violent grid ──
(
for how in anchor_inpaint hybrid; do
  for probe in text image both; do
    for fam in 0 1; do
      run 3 mja CAS_SpatialCFG/prompts/mja_violent.txt violent $probe $how $fam 0.4 1.5 "$VT" "$VA"
    done
  done
done
echo "[GPU3 COMPLETE]"
) &

# ── GPU 5: MJA disturbing grid ──
(
for probe in text image both; do
  for fam in 0 1; do
    for cas in 0.4 0.6; do
      run 5 mja CAS_SpatialCFG/prompts/mja_disturbing.txt disturbing $probe anchor_inpaint $fam $cas 1.0 "$DT" "$DA"
    done
  done
done
echo "[GPU5 COMPLETE]"
) &

# ── GPU 6: MJA sexual + illegal ──
(
for probe in text image both; do
  for fam in 0 1; do
    run 6 mja CAS_SpatialCFG/prompts/mja_sexual.txt sexual $probe anchor_inpaint $fam 0.6 1.2 "$NT" "$NA"
  done
done
for probe in text image both; do
  for fam in 0 1; do
    run 6 mja CAS_SpatialCFG/prompts/mja_illegal.txt illegal $probe anchor_inpaint $fam 0.5 1.0 "$IT" "$IA"
  done
done
echo "[GPU6 COMPLETE]"
) &

# ── GPU 7: RAB image-only ablation + extra configs ──
(
for cas in 0.4 0.6; do
  for ss in 1.0 1.2; do
    run 7 rab CAS_SpatialCFG/prompts/ringabell.txt sexual image anchor_inpaint 0 $cas $ss "$NT" "$NA"
    run 7 rab CAS_SpatialCFG/prompts/ringabell.txt sexual image anchor_inpaint 1 $cas $ss "$NT" "$NA"
  done
done
echo "[GPU7 COMPLETE]"
) &

wait
echo "=== Phase 1 PARALLEL COMPLETE $(date) ==="
