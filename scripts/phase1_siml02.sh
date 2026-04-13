#!/bin/bash
# Phase 1 on siml-02: I2P concepts + additional configs
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

# Concept keywords
HT="threatening pointing aggressive bully intimidation"
HA="friendly waving greeting handshake smile"
HAT="nazi swastika slur hate discrimination"
HAA="peace diversity unity community respect"
ST="grotesque horror monster scream agony burning"
SA="beautiful gentle calm smile serene"
SHT="hanging noose cutting self-harm suicide depression"
SHA="standing healthy yoga meditation reading"
VT="blood knife gun weapon wound fight"
VA="clean toy_gun handshake healthy"
IT="drug pills marijuana smoke trafficking"
IA="medicine vitamins tea coffee portrait"

echo "=== siml-02 Phase 1 START $(date) ==="

# ── GPU 0: I2P Violence (756 prompts) ──
(
for probe in text both; do
  for fam in 0 1; do
    run 0 i2p SAFREE/datasets/i2p_categories/i2p_violence.csv violent $probe anchor_inpaint $fam 0.4 1.5 "$VT" "$VA"
  done
done
echo "[GPU0 COMPLETE]"
) &

# ── GPU 1: I2P Harassment (824 prompts) ──
(
for probe in text both; do
  for fam in 0 1; do
    run 1 i2p SAFREE/datasets/i2p_categories/i2p_harassment.csv harassment $probe anchor_inpaint $fam 0.4 1.2 "$HT" "$HA"
  done
done
echo "[GPU1 COMPLETE]"
) &

# ── GPU 2: I2P Hate (231 prompts) ──
(
for probe in text both; do
  for fam in 0 1; do
    run 2 i2p SAFREE/datasets/i2p_categories/i2p_hate.csv hate $probe anchor_inpaint $fam 0.4 1.0 "$HAT" "$HAA"
  done
done
echo "[GPU2 COMPLETE]"
) &

# ── GPU 3: I2P Shocking (856 prompts) ──
(
for probe in text both; do
  for fam in 0 1; do
    run 3 i2p SAFREE/datasets/i2p_categories/i2p_shocking.csv disturbing $probe anchor_inpaint $fam 0.5 1.0 "$ST" "$SA"
  done
done
echo "[GPU3 COMPLETE]"
) &

# ── GPU 4: I2P Illegal (727 prompts) ──
(
for probe in text both; do
  for fam in 0 1; do
    run 4 i2p SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv illegal $probe anchor_inpaint $fam 0.5 1.0 "$IT" "$IA"
  done
done
echo "[GPU4 COMPLETE]"
) &

# ── GPU 5: I2P Self-harm (801 prompts) ──
(
for probe in text both; do
  for fam in 0 1; do
    run 5 i2p SAFREE/datasets/i2p_categories/i2p_self-harm.csv selfharm $probe anchor_inpaint $fam 0.4 1.0 "$SHT" "$SHA"
  done
done
echo "[GPU5 COMPLETE]"
) &

# ── GPU 6: I2P Sexual (931 prompts) ──
(
NT="nudity nude naked exposed"
NA="clothed dressed covered wearing_clothes"
for probe in text both; do
  for fam in 0 1; do
    run 6 i2p SAFREE/datasets/i2p_categories/i2p_sexual.csv sexual $probe anchor_inpaint $fam 0.6 1.2 "$NT" "$NA"
  done
done
echo "[GPU6 COMPLETE]"
) &

# GPU 7 reserved for SAFREE baseline (already running)

wait
echo "=== siml-02 Phase 1 COMPLETE $(date) ==="
