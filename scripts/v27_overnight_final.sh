#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# v27 OVERNIGHT FINAL — 프롬프트당 3장, 모든 artist + nudity + concepts
# =============================================================================

P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=/mnt/home3/yhgil99/unlearning/vlm
V27=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/generate_v27.py
BASELINE=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/generate_baseline.py
REPO=/mnt/home3/yhgil99/unlearning
ADIR=$REPO/CAS_SpatialCFG/exemplars/artists
APDIR=$REPO/CAS_SpatialCFG/prompts/artists
CLIP_NUDE=$REPO/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_embeddings.pt
OUT=$REPO/CAS_SpatialCFG/outputs/v27_overnight_final
I2P=$REPO/SAFREE/datasets/i2p_categories
CEXDIR=$REPO/CAS_SpatialCFG/exemplars/concepts
mkdir -p $OUT

BOTH_COMMON="--attn_threshold 0.1 --img_attn_threshold 0.4 --nsamples 3 --steps 50 --seed 42"
TXT_COMMON="--attn_threshold 0.1 --nsamples 3 --steps 50 --seed 42"

run() {
  local gpu=$1 name=$2 prompts=$3; shift 3
  local outdir=$OUT/$name
  [ -f "${outdir}/generation_stats.json" ] && return
  echo "[$(date +%H:%M)] GPU $gpu: $name"
  CUDA_VISIBLE_DEVICES=$gpu $P $V27 --prompts "$prompts" --outdir "$outdir" "$@" 2>&1 | tail -1
}

eval_one() {
  local gpu=$1 name=$2 concept=$3
  local outdir=$OUT/$name
  [ -f "${outdir}/categories_qwen3_vl_${concept}.json" ] && return
  local imgs=$(find "$outdir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  [ $imgs -lt 10 ] && return
  CUDA_VISIBLE_DEVICES=$gpu $VLP $VLD/opensource_vlm_i2p_all.py "$outdir" "$concept" qwen 2>&1 | tail -1
}

# =============================================================================
# GPU 0: Nudity Ring-A-Bell best configs × 3 samples
# =============================================================================
(
RB=$REPO/CAS_SpatialCFG/prompts/ringabell.txt
for probe in both text image; do
  clip_arg=""; [ "$probe" != "text" ] && clip_arg="--clip_embeddings $CLIP_NUDE"
  ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
  # anchor_inpaint
  run 0 "nude_rb_${probe}_ainp_ss12" "$RB" --probe_mode $probe --how_mode anchor_inpaint \
    --safety_scale 1.2 --attn_threshold 0.1 $ithr --cas_threshold 0.6 --nsamples 3 --steps 50 --seed 42 $clip_arg
  # hybrid
  run 0 "nude_rb_${probe}_hyb_ts15as15" "$RB" --probe_mode $probe --how_mode hybrid \
    --target_scale 15 --anchor_scale 15 --attn_threshold 0.1 $ithr --cas_threshold 0.6 --nsamples 3 --steps 50 --seed 42 $clip_arg
done
for d in $OUT/nude_rb_*; do [ -d "$d" ] && eval_one 0 "$(basename $d)" nudity; done
echo "GPU0 DONE"
) &

# =============================================================================
# GPU 1: Nudity other datasets (best config: both ainp ss=1.2)
# =============================================================================
(
for ds_info in \
  "mma|$REPO/CAS_SpatialCFG/prompts/mma-diffusion-nsfw-adv-prompts.csv" \
  "p4dn|$REPO/CAS_SpatialCFG/prompts/p4dn_16_prompt.csv" \
  "unlearndiff|$REPO/CAS_SpatialCFG/prompts/unlearn_diff_nudity.csv" \
  "i2p|$I2P/i2p_sexual.csv"; do
  IFS='|' read -r ds prompts <<< "$ds_info"
  for probe in both text; do
    clip_arg=""; [ "$probe" = "both" ] && clip_arg="--clip_embeddings $CLIP_NUDE"
    ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
    run 1 "nude_${ds}_${probe}_ainp" "$prompts" --probe_mode $probe --how_mode anchor_inpaint \
      --safety_scale 1.2 --attn_threshold 0.1 $ithr --cas_threshold 0.6 --nsamples 3 --steps 50 --seed 42 $clip_arg
  done
done
for d in $OUT/nude_*; do [ -d "$d" ] && eval_one 1 "$(basename $d)" nudity; done
echo "GPU1 DONE"
) &

# =============================================================================
# GPU 2: Van Gogh (big 20 prompts × 3 samples)
# =============================================================================
(
for probe in both text image; do
  clip_arg=""; [ "$probe" != "text" ] && clip_arg="--clip_embeddings $ADIR/vangogh/clip_exemplar.pt"
  ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
  for ss in 0.8 1.0 1.2; do
    for cas in 0.3 0.4 0.5; do
      run 2 "vg_big_${probe}_ainp_ss${ss}_cas${cas}" "$APDIR/big_vangogh.txt" \
        --probe_mode $probe --how_mode anchor_inpaint --safety_scale $ss \
        --attn_threshold 0.1 $ithr --cas_threshold $cas --nsamples 3 --steps 50 --seed 42 $clip_arg \
        --target_concepts "Van Gogh style" "Van Gogh brushstrokes" "Van Gogh painting" \
        --anchor_concepts painting artwork "oil painting"
    done
  done
done
for d in $OUT/vg_big_*; do [ -d "$d" ] && eval_one 2 "$(basename $d)" style_vangogh; done
echo "GPU2 DONE"
) &

# =============================================================================
# GPU 3: Picasso (big 20 prompts × 3 samples)
# =============================================================================
(
for probe in both text; do
  clip_arg=""; [ "$probe" = "both" ] && clip_arg="--clip_embeddings $ADIR/picasso/clip_exemplar.pt"
  ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
  for ss in 0.8 1.0 1.2; do
    for cas in 0.3 0.4 0.5; do
      run 3 "picasso_big_${probe}_ainp_ss${ss}_cas${cas}" "$APDIR/big_pablopicasso.txt" \
        --probe_mode $probe --how_mode anchor_inpaint --safety_scale $ss \
        --attn_threshold 0.1 $ithr --cas_threshold $cas --nsamples 3 --steps 50 --seed 42 $clip_arg \
        --target_concepts "Picasso style" "cubist Picasso" "Pablo Picasso painting" \
        --anchor_concepts painting artwork "modern art"
    done
  done
done
for d in $OUT/picasso_big_*; do [ -d "$d" ] && eval_one 3 "$(basename $d)" style_picasso; done
echo "GPU3 DONE"
) &

# =============================================================================
# GPU 4: Rembrandt + Caravaggio (big 20 each × 3 samples)
# =============================================================================
(
# Wait for exemplar generation to finish
while [ ! -f "$ADIR/rembrandt/clip_exemplar.pt" ] || [ ! -f "$ADIR/caravaggio/clip_exemplar.pt" ]; do
  sleep 60
done

for artist_info in \
  "rembrandt|big_rembrandt|Rembrandt style|Rembrandt painting|painting|artwork|style_vangogh" \
  "caravaggio|big_caravaggio|Caravaggio style|Caravaggio chiaroscuro|painting|artwork|style_vangogh"; do
  IFS='|' read -r artist pfile t1 t2 a1 a2 eval_style <<< "$artist_info"
  for probe in both text; do
    clip_arg=""; [ "$probe" = "both" ] && clip_arg="--clip_embeddings $ADIR/$artist/clip_exemplar.pt"
    ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
    for ss in 0.8 1.0 1.2; do
      for cas in 0.3 0.4 0.5; do
        run 4 "${artist}_big_${probe}_ainp_ss${ss}_cas${cas}" "$APDIR/$pfile.txt" \
          --probe_mode $probe --how_mode anchor_inpaint --safety_scale $ss \
          --attn_threshold 0.1 $ithr --cas_threshold $cas --nsamples 3 --steps 50 --seed 42 $clip_arg \
          --target_concepts "$t1" "$t2" --anchor_concepts "$a1" "$a2"
      done
    done
  done
done
echo "GPU4 DONE"
) &

# =============================================================================
# GPU 5: Andy Warhol (big 20 × 3 samples)
# =============================================================================
(
while [ ! -f "$ADIR/andywarhol/clip_exemplar.pt" ]; do sleep 60; done

for probe in both text; do
  clip_arg=""; [ "$probe" = "both" ] && clip_arg="--clip_embeddings $ADIR/andywarhol/clip_exemplar.pt"
  ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
  for ss in 0.8 1.0 1.2; do
    for cas in 0.3 0.4 0.5; do
      run 5 "warhol_big_${probe}_ainp_ss${ss}_cas${cas}" "$APDIR/big_andywarhol.txt" \
        --probe_mode $probe --how_mode anchor_inpaint --safety_scale $ss \
        --attn_threshold 0.1 $ithr --cas_threshold $cas --nsamples 3 --steps 50 --seed 42 $clip_arg \
        --target_concepts "Andy Warhol style" "Warhol pop art" "pop art" \
        --anchor_concepts painting artwork "modern art"
    done
  done
done
echo "GPU5 DONE"
) &

# =============================================================================
# GPU 6: All 6 safety concepts (best nudity config: both ainp ss=1.2)
# =============================================================================
(
for concept_info in \
  "violence|$I2P/i2p_violence.csv|$CEXDIR/violence/clip_exemplar_projected.pt|violence gore bloodshed fighting weapon|toy_weapon toy_gun peaceful_scene friendly_interaction" \
  "harassment|$I2P/i2p_harassment.csv|$CEXDIR/harassment/clip_exemplar_projected.pt|harassment bullying threatening intimidation|friendly_conversation cooperation respectful_interaction" \
  "hate|$I2P/i2p_hate.csv|$CEXDIR/hate/clip_exemplar_projected.pt|hate_speech discrimination racist_symbol|diversity harmony cultural_respect equality"; do
  IFS='|' read -r concept prompts clip targets anchors <<< "$concept_info"
  for probe in both text; do
    clip_arg=""; [ "$probe" = "both" ] && clip_arg="--clip_embeddings $clip"
    ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
    for cas in 0.4 0.5 0.6; do
      run 6 "c_${concept}_${probe}_ainp_cas${cas}" "$prompts" \
        --probe_mode $probe --how_mode anchor_inpaint --safety_scale 1.2 \
        --attn_threshold 0.1 $ithr --cas_threshold $cas --nsamples 3 --steps 50 --seed 42 $clip_arg \
        --target_concepts $targets --anchor_concepts $anchors
    done
  done
  for d in $OUT/c_${concept}_*; do [ -d "$d" ] && eval_one 6 "$(basename $d)" "$concept"; done
done
echo "GPU6 DONE"
) &

# =============================================================================
# GPU 7: Remaining concepts + baselines
# =============================================================================
(
for concept_info in \
  "shocking|$I2P/i2p_shocking.csv|$CEXDIR/shocking/clip_exemplar_projected.pt|grotesque body_horror disturbing_imagery|serene_scene beautiful_art harmonious_composition" \
  "illegal_activity|$I2P/i2p_illegal_activity.csv|$CEXDIR/illegal_activity/clip_exemplar_projected.pt|drugs crime illegal_weapons theft|legal_activity professional_work social_norm" \
  "self-harm|$I2P/i2p_self-harm.csv|$CEXDIR/self-harm/clip_exemplar_projected.pt|self-harm suicide self-injury|healing meditation healthy_activity mental_wellness"; do
  IFS='|' read -r concept prompts clip targets anchors <<< "$concept_info"
  for probe in both text; do
    clip_arg=""; [ "$probe" = "both" ] && clip_arg="--clip_embeddings $clip"
    ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
    for cas in 0.4 0.5 0.6; do
      run 7 "c_${concept}_${probe}_ainp_cas${cas}" "$prompts" \
        --probe_mode $probe --how_mode anchor_inpaint --safety_scale 1.2 \
        --attn_threshold 0.1 $ithr --cas_threshold $cas --nsamples 3 --steps 50 --seed 42 $clip_arg \
        --target_concepts $targets --anchor_concepts $anchors
    done
  done
  for d in $OUT/c_${concept}_*; do [ -d "$d" ] && eval_one 7 "$(basename $d)" "$concept"; done
done

# Artist baselines (3 samples)
for artist in vangogh picasso rembrandt caravaggio andywarhol; do
  pfile="big_${artist}"
  [ -f "$APDIR/$pfile.txt" ] || continue
  outdir=$OUT/${artist}_baseline
  [ -f "${outdir}/generation_stats.json" ] && continue
  CUDA_VISIBLE_DEVICES=7 $P $BASELINE --prompts "$APDIR/$pfile.txt" --outdir "$outdir" \
    --nsamples 3 --steps 50 --seed 42 2>&1 | tail -1
done
echo "GPU7 DONE"
) &

echo "=============================================="
echo "  v27 OVERNIGHT FINAL — $(date)"
echo "  ALL 8 GPUs, nsamples=3"
echo "  GPU 0: Nudity Ring-A-Bell (text/img/both × ainp/hybrid)"
echo "  GPU 1: Nudity other datasets"
echo "  GPU 2: Van Gogh big (20 prompts × 3)"
echo "  GPU 3: Picasso big (20 × 3)"
echo "  GPU 4: Rembrandt + Caravaggio big (20+20 × 3)"
echo "  GPU 5: Andy Warhol big (20 × 3)"
echo "  GPU 6: Violence + Harassment + Hate"
echo "  GPU 7: Shocking + Illegal + Self-harm + Baselines"
echo "=============================================="

wait
echo "ALL OVERNIGHT FINAL DONE — $(date)"
