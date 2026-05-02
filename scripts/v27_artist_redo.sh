#!/usr/bin/env bash
set -euo pipefail

P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=/mnt/home3/yhgil99/unlearning/vlm
V27=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/generate_v27.py
BASELINE=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/generate_baseline.py
ADIR=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/artists
APDIR=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/artists
OUT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v27_final
NS="--nsamples 3 --steps 50 --seed 42"

run() {
  local gpu=$1 name=$2 prompts=$3; shift 3
  local outdir=$OUT/$name
  [ -f "${outdir}/generation_stats.json" ] && return
  echo "[$(date +%H:%M)] GPU $gpu: $name"
  CUDA_VISIBLE_DEVICES=$gpu $P $V27 --prompts "$prompts" --outdir "$outdir" $NS "$@" 2>&1 | tail -1
}

eval_art() {
  local gpu=$1 name=$2 concept=$3
  local outdir=$OUT/$name
  [ -f "${outdir}/categories_qwen3_vl_${concept}.json" ] && return
  local imgs=$(find "$outdir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  [ $imgs -lt 10 ] && return
  CUDA_VISIBLE_DEVICES=$gpu $VLP $VLD/opensource_vlm_i2p_all.py "$outdir" "$concept" qwen 2>&1 | tail -1
}

# Artist definitions: slug|promptfile|clip|targets|anchors|eval_concept|target_words
declare -a ARTISTS=(
  "vangogh|big_vangogh|$ADIR/vangogh/clip_exemplar.pt|Van_Gogh_style Van_Gogh_brushstrokes Van_Gogh_painting|painting artwork oil_painting|style_vangogh|van gogh style brushstrokes impasto swirling painting"
  "picasso|big_pablopicasso|$ADIR/picasso/clip_exemplar.pt|Picasso_style cubist_Picasso Pablo_Picasso_painting|painting artwork modern_art|style_picasso|picasso cubist cubism geometric style painting"
  "warhol|big_andywarhol|$ADIR/andywarhol/clip_exemplar.pt|Andy_Warhol_style Warhol_pop_art pop_art|painting artwork modern_art|style_vangogh|warhol pop art style screen print"
  "rembrandt|big_rembrandt|$ADIR/rembrandt/clip_exemplar.pt|Rembrandt_style Rembrandt_chiaroscuro Rembrandt_painting|painting artwork oil_painting|style_vangogh|rembrandt chiaroscuro style painting"
  "caravaggio|big_caravaggio|$ADIR/caravaggio/clip_exemplar.pt|Caravaggio_style Caravaggio_chiaroscuro|painting artwork oil_painting|style_vangogh|caravaggio chiaroscuro style dramatic painting"
)

# =====================================================================
# GPU 0: Van Gogh full sweep (most important, hardest)
# =====================================================================
(
IFS='|' read -r slug pfile clip targets anchors eval_concept tw <<< "${ARTISTS[0]}"
BOTH="--probe_mode both --clip_embeddings $clip --attn_threshold 0.1 --img_attn_threshold 0.4"
for probe in both text image; do
  clip_arg=""; [ "$probe" != "text" ] && clip_arg="--clip_embeddings $clip"
  ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
  for ss in 0.8 1.0 1.2 1.5 2.0; do
    for cas in 0.3 0.4 0.5; do
      run 0 "${slug}_${probe}_ainp_ss${ss}_cas${cas}" "$APDIR/$pfile.txt" \
        --probe_mode $probe --how_mode anchor_inpaint --safety_scale $ss --cas_threshold $cas \
        --target_words $tw --target_concepts $targets --anchor_concepts $anchors \
        --attn_threshold 0.1 $ithr $clip_arg
    done
  done
done
for d in $OUT/${slug}_*; do [ -d "$d" ] && eval_art 0 "$(basename $d)" "$eval_concept"; done
echo "GPU0 DONE"
) &

# =====================================================================
# GPU 1: Picasso full sweep
# =====================================================================
(
IFS='|' read -r slug pfile clip targets anchors eval_concept tw <<< "${ARTISTS[1]}"
for probe in both text image; do
  clip_arg=""; [ "$probe" != "text" ] && clip_arg="--clip_embeddings $clip"
  ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
  for ss in 0.8 1.0 1.2 1.5; do
    for cas in 0.3 0.4 0.5; do
      run 1 "${slug}_${probe}_ainp_ss${ss}_cas${cas}" "$APDIR/$pfile.txt" \
        --probe_mode $probe --how_mode anchor_inpaint --safety_scale $ss --cas_threshold $cas \
        --target_words $tw --target_concepts $targets --anchor_concepts $anchors \
        --attn_threshold 0.1 $ithr $clip_arg
    done
  done
done
for d in $OUT/${slug}_*; do [ -d "$d" ] && eval_art 1 "$(basename $d)" "$eval_concept"; done
echo "GPU1 DONE"
) &

# =====================================================================
# GPU 2: Warhol full sweep
# =====================================================================
(
IFS='|' read -r slug pfile clip targets anchors eval_concept tw <<< "${ARTISTS[2]}"
for probe in both text; do
  clip_arg=""; [ "$probe" = "both" ] && clip_arg="--clip_embeddings $clip"
  ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
  for ss in 0.8 1.0 1.2 1.5; do
    for cas in 0.3 0.4 0.5; do
      run 2 "${slug}_${probe}_ainp_ss${ss}_cas${cas}" "$APDIR/$pfile.txt" \
        --probe_mode $probe --how_mode anchor_inpaint --safety_scale $ss --cas_threshold $cas \
        --target_words $tw --target_concepts $targets --anchor_concepts $anchors \
        --attn_threshold 0.1 $ithr $clip_arg
    done
  done
done
for d in $OUT/${slug}_*; do [ -d "$d" ] && eval_art 2 "$(basename $d)" "$eval_concept"; done
echo "GPU2 DONE"
) &

# =====================================================================
# GPU 3: Rembrandt + Caravaggio
# =====================================================================
(
for idx in 3 4; do
  IFS='|' read -r slug pfile clip targets anchors eval_concept tw <<< "${ARTISTS[$idx]}"
  for probe in both text; do
    clip_arg=""; [ "$probe" = "both" ] && clip_arg="--clip_embeddings $clip"
    ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
    for ss in 0.8 1.0 1.2; do
      for cas in 0.3 0.4 0.5; do
        run 3 "${slug}_${probe}_ainp_ss${ss}_cas${cas}" "$APDIR/$pfile.txt" \
          --probe_mode $probe --how_mode anchor_inpaint --safety_scale $ss --cas_threshold $cas \
          --target_words $tw --target_concepts $targets --anchor_concepts $anchors \
          --attn_threshold 0.1 $ithr $clip_arg
      done
    done
  done
  for d in $OUT/${slug}_*; do [ -d "$d" ] && eval_art 3 "$(basename $d)" "$eval_concept"; done
done
echo "GPU3 DONE"
) &

# =====================================================================
# GPU 4: Monet (10 prompts) + Baselines for all artists
# =====================================================================
(
# Monet
for probe in both text image; do
  clip_arg=""; [ "$probe" != "text" ] && clip_arg="--clip_embeddings $ADIR/monet/clip_exemplar.pt"
  ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
  for ss in 0.8 1.0 1.2 1.5; do
    for cas in 0.3 0.4 0.5; do
      run 4 "monet_${probe}_ainp_ss${ss}_cas${cas}" "$APDIR/monet.txt" \
        --probe_mode $probe --how_mode anchor_inpaint --safety_scale $ss --cas_threshold $cas \
        --target_words monet impressionist impressionism style painting \
        --target_concepts "Monet style" "impressionist Monet" "Claude Monet painting" \
        --anchor_concepts painting artwork "oil painting" \
        --attn_threshold 0.1 $ithr $clip_arg
    done
  done
done
for d in $OUT/monet_*; do [ -d "$d" ] && eval_art 4 "$(basename $d)" style_monet; done

# SD Baselines (nsamples=3)
for artist_pfile in "vangogh|big_vangogh" "picasso|big_pablopicasso" "warhol|big_andywarhol" "rembrandt|big_rembrandt" "caravaggio|big_caravaggio" "monet|monet"; do
  IFS='|' read -r a pf <<< "$artist_pfile"
  outdir=$OUT/${a}_baseline
  [ -f "${outdir}/generation_stats.json" ] && continue
  echo "[$(date +%H:%M)] GPU4: ${a}_baseline"
  CUDA_VISIBLE_DEVICES=4 $P $BASELINE --prompts "$APDIR/$pf.txt" --outdir "$outdir" --nsamples 3 --steps 50 --seed 42 2>&1 | tail -1
done
# Eval baselines
for a in vangogh picasso warhol rembrandt caravaggio monet; do
  style="style_${a}"
  [ "$a" = "warhol" ] && style="style_vangogh"
  [ "$a" = "rembrandt" ] && style="style_vangogh"
  [ "$a" = "caravaggio" ] && style="style_vangogh"
  eval_art 4 "${a}_baseline" "$style"
done
echo "GPU4 DONE"
) &

# =====================================================================
# GPU 5: Van Gogh hybrid sweep (need stronger for 85%→90%+)
# =====================================================================
(
IFS='|' read -r slug pfile clip targets anchors eval_concept tw <<< "${ARTISTS[0]}"
for probe in both text; do
  clip_arg=""; [ "$probe" = "both" ] && clip_arg="--clip_embeddings $clip"
  ithr=""; [ "$probe" = "both" ] && ithr="--img_attn_threshold 0.4"
  for ts_as in "10|10" "15|15" "20|20" "25|25"; do
    IFS='|' read -r ts as <<< "$ts_as"
    for cas in 0.3 0.4 0.5; do
      run 5 "${slug}_${probe}_hyb_ts${ts}_as${as}_cas${cas}" "$APDIR/$pfile.txt" \
        --probe_mode $probe --how_mode hybrid --target_scale $ts --anchor_scale $as --cas_threshold $cas \
        --target_words $tw --target_concepts $targets --anchor_concepts $anchors \
        --attn_threshold 0.1 $ithr $clip_arg
    done
  done
done
for d in $OUT/${slug}_*hyb*; do [ -d "$d" ] && eval_art 5 "$(basename $d)" "$eval_concept"; done
echo "GPU5 DONE"
) &

# =====================================================================
# GPU 6: Multi-artist erasing (VG+Monet, VG+Picasso, all 3)
# =====================================================================
(
for combo_info in \
  "multi_vg_monet|big_vangogh|$ADIR/vangogh/clip_exemplar.pt|Van_Gogh_style Monet_style impressionist|painting artwork|van gogh monet impressionist style" \
  "multi_vg_picasso|big_vangogh|$ADIR/vangogh/clip_exemplar.pt|Van_Gogh_style Picasso_style cubist|painting artwork modern_art|van gogh picasso cubist style" \
  "multi_3artists|big_vangogh|$ADIR/vangogh/clip_exemplar.pt|Van_Gogh_style Monet_style Picasso_style|painting artwork|van gogh monet picasso style"; do
  IFS='|' read -r name pfile clip targets anchors tw <<< "$combo_info"
  for ss in 0.8 1.0 1.2; do
    for cas in 0.3 0.4 0.5; do
      run 6 "${name}_ss${ss}_cas${cas}" "$APDIR/$pfile.txt" \
        --probe_mode both --how_mode anchor_inpaint --safety_scale $ss --cas_threshold $cas \
        --clip_embeddings $clip --attn_threshold 0.1 --img_attn_threshold 0.4 \
        --target_words $tw --target_concepts $targets --anchor_concepts $anchors
    done
  done
done
for d in $OUT/multi_*artist* $OUT/multi_vg_*; do [ -d "$d" ] && eval_art 6 "$(basename $d)" style_vangogh; done
echo "GPU6 DONE"
) &

# =====================================================================
# GPU 7: Picasso + Warhol hybrid + artist proj_replace
# =====================================================================
(
for idx in 1 2; do
  IFS='|' read -r slug pfile clip targets anchors eval_concept tw <<< "${ARTISTS[$idx]}"
  for ts_as in "15|15" "20|20"; do
    IFS='|' read -r ts as <<< "$ts_as"
    for cas in 0.3 0.4 0.5; do
      run 7 "${slug}_both_hyb_ts${ts}_as${as}_cas${cas}" "$APDIR/$pfile.txt" \
        --probe_mode both --how_mode hybrid --target_scale $ts --anchor_scale $as --cas_threshold $cas \
        --clip_embeddings $clip --attn_threshold 0.1 --img_attn_threshold 0.4 \
        --target_words $tw --target_concepts $targets --anchor_concepts $anchors
    done
  done
  for d in $OUT/${slug}_*hyb*; do [ -d "$d" ] && eval_art 7 "$(basename $d)" "$eval_concept"; done
done
echo "GPU7 DONE"
) &

echo "=============================================="
echo "  ARTIST REDO — nsamples=3, new 3-class eval"
echo "  $(date)"
echo "=============================================="

wait
echo "ALL ARTIST REDO DONE — $(date)"
