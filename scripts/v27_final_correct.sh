#!/usr/bin/env bash
set -euo pipefail

P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=/mnt/home3/yhgil99/unlearning/vlm
V27=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/generate_v27.py
REPO=/mnt/home3/yhgil99/unlearning
CLIP_NUDE=$REPO/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_embeddings.pt
CEXDIR=$REPO/CAS_SpatialCFG/exemplars/concepts
ADIR=$REPO/CAS_SpatialCFG/exemplars/artists
APDIR=$REPO/CAS_SpatialCFG/prompts/artists
I2P=$REPO/SAFREE/datasets/i2p_categories
OUT=$REPO/CAS_SpatialCFG/outputs/v27_final
mkdir -p $OUT

BOTH="--probe_mode both --attn_threshold 0.1 --img_attn_threshold 0.4"
NS="--nsamples 1 --steps 50 --seed 42"

# ── Per-concept target_words (from families.json + concept packs) ──
TW_NUDE="nude naked nudity nsfw bare body breast genital"
TW_VIOLENCE="blood bleeding wound gun rifle knife sword weapon fight fighting punch combat gore violent"
TW_HARASSMENT="harassment bully threatening creepy angry screaming vicious intimidation"
TW_HATE="hate racist nazi slur discrimination monster zombie killer"
TW_SHOCKING="screaming agony grotesque horror demonic burning scary disturbing"
TW_ILLEGAL="drugs pills powder smoking weed prison mugshot theft crime illegal weapon"
TW_SELFHARM="hanging noose suicide self-harm depression anorexic cutting injury"
TW_VANGOGH="van gogh style brushstrokes impasto swirling painting"
TW_MONET="monet impressionist impressionism style painting"
TW_PICASSO="picasso cubist cubism geometric style painting"
TW_REMBRANDT="rembrandt chiaroscuro style painting"
TW_CARAVAGGIO="caravaggio chiaroscuro style dramatic painting"
TW_WARHOL="warhol pop art style screen print"

run() {
  local gpu=$1 name=$2 prompts=$3; shift 3
  local outdir=$OUT/$name
  [ -f "${outdir}/generation_stats.json" ] && return
  echo "[$(date +%H:%M)] GPU $gpu: $name"
  CUDA_VISIBLE_DEVICES=$gpu $P $V27 --prompts "$prompts" --outdir "$outdir" $NS "$@" 2>&1 | tail -1
}

eval_one() {
  local gpu=$1 name=$2 concept=$3
  local outdir=$OUT/$name
  [ -f "${outdir}/categories_qwen3_vl_${concept}.json" ] && return
  local imgs=$(find "$outdir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  [ $imgs -lt 8 ] && return
  CUDA_VISIBLE_DEVICES=$gpu $VLP $VLD/opensource_vlm_i2p_all.py "$outdir" "$concept" qwen 2>&1 | tail -1
}

# =====================================================================
# GPU 0: Nudity — Ring-A-Bell × text/img/both × ainp/hybrid (3 samples)
# =====================================================================
(
RB=$REPO/CAS_SpatialCFG/prompts/ringabell.txt
for probe in both text image; do
  clip_arg=""; [ "$probe" != "text" ] && clip_arg="--clip_embeddings $CLIP_NUDE"
  ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
  run 0 "nude_rb_${probe}_ainp_ss12" "$RB" --probe_mode $probe --how_mode anchor_inpaint \
    --safety_scale 1.2 --cas_threshold 0.6 --target_words $TW_NUDE --nsamples 3 $clip_arg $ithr --attn_threshold 0.1
  run 0 "nude_rb_${probe}_hyb_ts15" "$RB" --probe_mode $probe --how_mode hybrid \
    --target_scale 15 --anchor_scale 15 --cas_threshold 0.6 --target_words $TW_NUDE --nsamples 3 $clip_arg $ithr --attn_threshold 0.1
done
for d in $OUT/nude_rb_*; do [ -d "$d" ] && eval_one 0 "$(basename $d)" nudity; done
echo "GPU0 DONE"
) &

# =====================================================================
# GPU 1: Nudity — other datasets (MMA, P4DN, UnlearnDiff, I2P, COCO)
# =====================================================================
(
for ds_info in \
  "mma|$REPO/CAS_SpatialCFG/prompts/mma-diffusion-nsfw-adv-prompts.csv" \
  "p4dn|$REPO/CAS_SpatialCFG/prompts/p4dn_16_prompt.csv" \
  "unlearndiff|$REPO/CAS_SpatialCFG/prompts/unlearn_diff_nudity.csv" \
  "i2p|$I2P/i2p_sexual.csv" \
  "coco|$REPO/CAS_SpatialCFG/prompts/coco_250.txt"; do
  IFS='|' read -r ds prompts <<< "$ds_info"
  for probe in both text; do
    clip_arg=""; [ "$probe" = "both" ] && clip_arg="--clip_embeddings $CLIP_NUDE"
    ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
    run 1 "nude_${ds}_${probe}_ainp" "$prompts" --probe_mode $probe --how_mode anchor_inpaint \
      --safety_scale 1.2 --cas_threshold 0.6 --target_words $TW_NUDE $clip_arg $ithr --attn_threshold 0.1
  done
done
for d in $OUT/nude_*; do [ -d "$d" ] && eval_one 1 "$(basename $d)" nudity; done
echo "GPU1 DONE"
) &

# =====================================================================
# GPU 2: Violence sweep
# =====================================================================
(
for probe in both text; do
  clip_arg=""; [ "$probe" = "both" ] && clip_arg="--clip_embeddings $CEXDIR/violence/clip_exemplar_projected.pt"
  ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
  for ss in 0.8 1.0 1.2 1.5; do
    for cas in 0.4 0.5 0.6; do
      run 2 "c_violence_${probe}_ainp_ss${ss}_cas${cas}" "$I2P/i2p_violence.csv" \
        --probe_mode $probe --how_mode anchor_inpaint --safety_scale $ss --cas_threshold $cas \
        --target_words $TW_VIOLENCE --target_concepts violence gore bloodshed fighting weapon \
        --anchor_concepts "toy weapon" "toy gun" "peaceful scene" "friendly interaction" \
        $clip_arg $ithr --attn_threshold 0.1
    done
  done
done
for d in $OUT/c_violence_*; do [ -d "$d" ] && eval_one 2 "$(basename $d)" violence; done
echo "GPU2 DONE"
) &

# =====================================================================
# GPU 3: Harassment + Hate sweep
# =====================================================================
(
# Harassment
for probe in both text; do
  clip_arg=""; [ "$probe" = "both" ] && clip_arg="--clip_embeddings $CEXDIR/harassment/clip_exemplar_projected.pt"
  ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
  for ss in 0.8 1.0 1.2; do
    for cas in 0.4 0.5 0.6; do
      run 3 "c_harassment_${probe}_ainp_ss${ss}_cas${cas}" "$I2P/i2p_harassment.csv" \
        --probe_mode $probe --how_mode anchor_inpaint --safety_scale $ss --cas_threshold $cas \
        --target_words $TW_HARASSMENT --target_concepts harassment bullying threatening intimidation \
        --anchor_concepts "friendly conversation" cooperation "respectful interaction" \
        $clip_arg $ithr --attn_threshold 0.1
    done
  done
done
# Hate
for probe in both text; do
  clip_arg=""; [ "$probe" = "both" ] && clip_arg="--clip_embeddings $CEXDIR/hate/clip_exemplar_projected.pt"
  ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
  for ss in 0.8 1.0 1.2; do
    for cas in 0.4 0.5 0.6; do
      run 3 "c_hate_${probe}_ainp_ss${ss}_cas${cas}" "$I2P/i2p_hate.csv" \
        --probe_mode $probe --how_mode anchor_inpaint --safety_scale $ss --cas_threshold $cas \
        --target_words $TW_HATE --target_concepts hate_speech discrimination "racist symbol" \
        --anchor_concepts diversity harmony "cultural respect" equality \
        $clip_arg $ithr --attn_threshold 0.1
    done
  done
done
for d in $OUT/c_harassment_* $OUT/c_hate_*; do
  [ -d "$d" ] || continue
  concept=harassment; [[ $(basename $d) == c_hate_* ]] && concept=hate
  eval_one 3 "$(basename $d)" $concept
done
echo "GPU3 DONE"
) &

# =====================================================================
# GPU 4: Shocking + Illegal + Self-harm sweep
# =====================================================================
(
for concept_info in \
  "shocking|$I2P/i2p_shocking.csv|$CEXDIR/shocking/clip_exemplar_projected.pt|grotesque body_horror disturbing_imagery|serene_scene beautiful_art harmonious_composition|$TW_SHOCKING" \
  "illegal_activity|$I2P/i2p_illegal_activity.csv|$CEXDIR/illegal_activity/clip_exemplar_projected.pt|drugs crime illegal_weapons theft|legal_activity professional_work social_norm|$TW_ILLEGAL" \
  "self-harm|$I2P/i2p_self-harm.csv|$CEXDIR/self-harm/clip_exemplar_projected.pt|self-harm suicide self-injury|healing meditation healthy_activity|$TW_SELFHARM"; do
  IFS='|' read -r concept prompts clip targets anchors tw <<< "$concept_info"
  for probe in both text; do
    clip_arg=""; [ "$probe" = "both" ] && clip_arg="--clip_embeddings $clip"
    ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
    for ss in 0.8 1.0 1.2; do
      for cas in 0.4 0.5 0.6; do
        run 4 "c_${concept}_${probe}_ainp_ss${ss}_cas${cas}" "$prompts" \
          --probe_mode $probe --how_mode anchor_inpaint --safety_scale $ss --cas_threshold $cas \
          --target_words $tw --target_concepts $targets --anchor_concepts $anchors \
          $clip_arg $ithr --attn_threshold 0.1
      done
    done
  done
  for d in $OUT/c_${concept}_*; do
    [ -d "$d" ] && eval_one 4 "$(basename $d)" "$concept"
  done
done
echo "GPU4 DONE"
) &

# =====================================================================
# GPU 5: Van Gogh + Monet (big 20 prompts)
# =====================================================================
(
for artist_info in \
  "vangogh|big_vangogh|$ADIR/vangogh/clip_exemplar.pt|Van_Gogh_style Van_Gogh_brushstrokes Van_Gogh_painting|painting artwork oil_painting|$TW_VANGOGH|style_vangogh" \
  "monet|big_monet|$ADIR/monet/clip_exemplar.pt|Monet_style impressionist_Monet Claude_Monet_painting|painting artwork oil_painting|$TW_MONET|style_monet"; do
  IFS='|' read -r artist pfile clip targets anchors tw eval_concept <<< "$artist_info"
  for probe in both text image; do
    clip_arg=""; [ "$probe" != "text" ] && clip_arg="--clip_embeddings $clip"
    ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
    for ss in 0.8 1.0 1.2 1.5; do
      for cas in 0.3 0.4 0.5; do
        run 5 "${artist}_${probe}_ainp_ss${ss}_cas${cas}" "$APDIR/$pfile.txt" \
          --probe_mode $probe --how_mode anchor_inpaint --safety_scale $ss --cas_threshold $cas \
          --target_words $tw --target_concepts $targets --anchor_concepts $anchors \
          $clip_arg $ithr --attn_threshold 0.1
      done
    done
  done
  for d in $OUT/${artist}_*; do [ -d "$d" ] && eval_one 5 "$(basename $d)" "$eval_concept"; done
done
echo "GPU5 DONE"
) &

# =====================================================================
# GPU 6: Picasso + Warhol (big 20 prompts)
# =====================================================================
(
for artist_info in \
  "picasso|big_pablopicasso|$ADIR/picasso/clip_exemplar.pt|Picasso_style cubist_Picasso Pablo_Picasso_painting|painting artwork modern_art|$TW_PICASSO|style_picasso" \
  "warhol|big_andywarhol|$ADIR/andywarhol/clip_exemplar.pt|Andy_Warhol_style Warhol_pop_art pop_art|painting artwork modern_art|$TW_WARHOL|style_vangogh"; do
  IFS='|' read -r artist pfile clip targets anchors tw eval_concept <<< "$artist_info"
  for probe in both text; do
    clip_arg=""; [ "$probe" = "both" ] && clip_arg="--clip_embeddings $clip"
    ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
    for ss in 0.8 1.0 1.2; do
      for cas in 0.3 0.4 0.5; do
        run 6 "${artist}_${probe}_ainp_ss${ss}_cas${cas}" "$APDIR/$pfile.txt" \
          --probe_mode $probe --how_mode anchor_inpaint --safety_scale $ss --cas_threshold $cas \
          --target_words $tw --target_concepts $targets --anchor_concepts $anchors \
          $clip_arg $ithr --attn_threshold 0.1
      done
    done
  done
  for d in $OUT/${artist}_*; do [ -d "$d" ] && eval_one 6 "$(basename $d)" "$eval_concept"; done
done
echo "GPU6 DONE"
) &

# =====================================================================
# GPU 7: Rembrandt + Caravaggio + Multi-concept + Artist baselines
# =====================================================================
(
for artist_info in \
  "rembrandt|big_rembrandt|$ADIR/rembrandt/clip_exemplar.pt|Rembrandt_style Rembrandt_chiaroscuro Rembrandt_painting|painting artwork oil_painting|$TW_REMBRANDT|style_vangogh" \
  "caravaggio|big_caravaggio|$ADIR/caravaggio/clip_exemplar.pt|Caravaggio_style Caravaggio_chiaroscuro Caravaggio_painting|painting artwork oil_painting|$TW_CARAVAGGIO|style_vangogh"; do
  IFS='|' read -r artist pfile clip targets anchors tw eval_concept <<< "$artist_info"
  for probe in both text; do
    clip_arg=""; [ "$probe" = "both" ] && clip_arg="--clip_embeddings $clip"
    ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
    for ss in 0.8 1.0 1.2; do
      for cas in 0.3 0.4 0.5; do
        run 7 "${artist}_${probe}_ainp_ss${ss}_cas${cas}" "$APDIR/$pfile.txt" \
          --probe_mode $probe --how_mode anchor_inpaint --safety_scale $ss --cas_threshold $cas \
          --target_words $tw --target_concepts $targets --anchor_concepts $anchors \
          $clip_arg $ithr --attn_threshold 0.1
      done
    done
  done
done
# Multi-concept: nudity+violence
for ss in 1.0 1.2; do
  for cas in 0.4 0.5; do
    run 7 "multi_nude_violence_ss${ss}_cas${cas}" "$I2P/i2p_sexual.csv" \
      $BOTH --clip_embeddings $CLIP_NUDE --how_mode anchor_inpaint --safety_scale $ss --cas_threshold $cas \
      --target_words $TW_NUDE $TW_VIOLENCE \
      --target_concepts nudity "nude person" violence gore weapon \
      --anchor_concepts "clothed person" "toy weapon" "peaceful scene"
  done
done
# Baselines
BASELINE=$REPO/CAS_SpatialCFG/generate_baseline.py
for artist in vangogh pablopicasso andywarhol rembrandt caravaggio; do
  [ -f "$APDIR/big_${artist}.txt" ] || continue
  outdir=$OUT/${artist}_baseline
  [ -f "${outdir}/generation_stats.json" ] && continue
  CUDA_VISIBLE_DEVICES=7 $P $BASELINE --prompts "$APDIR/big_${artist}.txt" --outdir "$outdir" --nsamples 1 --steps 50 --seed 42 2>&1 | tail -1
done
echo "GPU7 DONE"
) &

echo "=============================================="
echo "  v27 FINAL CORRECT — $(date)"
echo "  ALL target_words properly set per concept!"
echo "  GPU 0: Nudity Ring-A-Bell (text/img/both × ainp/hybrid)"
echo "  GPU 1: Nudity all datasets"
echo "  GPU 2: Violence sweep"
echo "  GPU 3: Harassment + Hate"
echo "  GPU 4: Shocking + Illegal + Self-harm"
echo "  GPU 5: Van Gogh + Monet"
echo "  GPU 6: Picasso + Warhol"
echo "  GPU 7: Rembrandt + Caravaggio + Multi-concept + Baselines"
echo "=============================================="

wait
echo "ALL DONE — $(date)"
