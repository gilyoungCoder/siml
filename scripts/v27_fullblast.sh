#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# v27 FULL BLAST — 모든 데이터셋 × 모든 concept × 최적 config
# =============================================================================
# Nudity: best config (both, ainp, ss=1.2, txt_thr=0.1, img_thr=0.4)
#         + text-only baseline 동시
#         × 5 datasets (ringabell, mma, p4dn, unlearndiff, i2p_sexual)
#         + COCO FP check
#
# Concepts: per-concept best (both + text, ainp + hybrid)
#           × 6 concepts × cas_thr sweep
# =============================================================================

P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=/mnt/home3/yhgil99/unlearning/vlm
V27=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/generate_v27.py
CLIP_NUDE=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_embeddings.pt
CEXDIR=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/concepts
OUT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v27_full
I2P=/mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories
REPO=/mnt/home3/yhgil99/unlearning
mkdir -p $OUT

# Nudity datasets
declare -A NUDE_DS=(
  [ringabell]="$REPO/CAS_SpatialCFG/prompts/ringabell.txt"
  [mma]="$REPO/CAS_SpatialCFG/prompts/mma-diffusion-nsfw-adv-prompts.csv"
  [p4dn]="$REPO/CAS_SpatialCFG/prompts/p4dn_16_prompt.csv"
  [unlearndiff]="$REPO/CAS_SpatialCFG/prompts/unlearn_diff_nudity.csv"
  [i2p_sexual]="$I2P/i2p_sexual.csv"
  [coco]="$REPO/CAS_SpatialCFG/prompts/coco_250.txt"
)

run() {
  local gpu=$1 name=$2 prompts=$3; shift 3
  local outdir=$OUT/$name
  [ -f "${outdir}/generation_stats.json" ] && return
  echo "[$(date +%H:%M)] GPU $gpu: $name"
  CUDA_VISIBLE_DEVICES=$gpu $P $V27 --prompts "$prompts" --outdir "$outdir" \
    --nsamples 1 --steps 50 --seed 42 "$@" 2>&1 | tail -1
}

eval_one() {
  local gpu=$1 name=$2 concept=$3
  local outdir=$OUT/$name
  [ -f "${outdir}/categories_qwen3_vl_${concept}.json" ] && return
  local imgs=$(find "$outdir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  [ $imgs -lt 50 ] && return
  CUDA_VISIBLE_DEVICES=$gpu $VLP $VLD/opensource_vlm_i2p_all.py "$outdir" "$concept" qwen 2>&1 | tail -1
}

# =============================================================================
# GPU 0: Nudity BEST (both, ainp, ss=1.2, txt_thr=0.1, img_thr=0.4) × all datasets
# =============================================================================
(
for ds in ringabell mma p4dn unlearndiff i2p_sexual coco; do
  run 0 "nude_both_${ds}" "${NUDE_DS[$ds]}" \
    --probe_mode both --how_mode anchor_inpaint --safety_scale 1.2 \
    --attn_threshold 0.1 --img_attn_threshold 0.4 --cas_threshold 0.6 \
    --clip_embeddings $CLIP_NUDE
done
for ds in ringabell mma p4dn unlearndiff i2p_sexual coco; do
  eval_one 0 "nude_both_${ds}" nudity
done
echo "GPU0 DONE"
) &

# =============================================================================
# GPU 1: Nudity TEXT-ONLY baseline × all datasets
# =============================================================================
(
for ds in ringabell mma p4dn unlearndiff i2p_sexual coco; do
  run 1 "nude_text_${ds}" "${NUDE_DS[$ds]}" \
    --probe_mode text --how_mode anchor_inpaint --safety_scale 1.2 \
    --attn_threshold 0.1 --cas_threshold 0.6
done
for ds in ringabell mma p4dn unlearndiff i2p_sexual coco; do
  eval_one 1 "nude_text_${ds}" nudity
done
echo "GPU1 DONE"
) &

# =============================================================================
# GPU 2: Nudity IMG-ONLY × all datasets (for ablation)
# =============================================================================
(
for ds in ringabell mma p4dn unlearndiff i2p_sexual; do
  run 2 "nude_img_${ds}" "${NUDE_DS[$ds]}" \
    --probe_mode image --how_mode anchor_inpaint --safety_scale 1.2 \
    --attn_threshold 0.4 --cas_threshold 0.6 \
    --clip_embeddings $CLIP_NUDE
done
for ds in ringabell mma p4dn unlearndiff i2p_sexual; do
  eval_one 2 "nude_img_${ds}" nudity
done
echo "GPU2 DONE"
) &

# =============================================================================
# GPU 3: Nudity 4-sample BEST (both) on key datasets
# =============================================================================
(
for ds in ringabell mma p4dn unlearndiff i2p_sexual; do
  run 3 "nude_both_4s_${ds}" "${NUDE_DS[$ds]}" \
    --probe_mode both --how_mode anchor_inpaint --safety_scale 1.2 \
    --attn_threshold 0.1 --img_attn_threshold 0.4 --cas_threshold 0.6 \
    --nsamples 4 --clip_embeddings $CLIP_NUDE
done
for ds in ringabell mma p4dn unlearndiff i2p_sexual; do
  eval_one 3 "nude_both_4s_${ds}" nudity
done
echo "GPU3 DONE"
) &

# =============================================================================
# GPU 4: Violence + Harassment (both + text × ainp, ss/cas sweep)
# =============================================================================
(
for concept_info in \
  "violence|$I2P/i2p_violence.csv|$CEXDIR/violence/clip_exemplar_projected.pt|violence gore bloodshed fighting weapon|peaceful_scene nature_landscape friendly_interaction" \
  "harassment|$I2P/i2p_harassment.csv|$CEXDIR/harassment/clip_exemplar_projected.pt|harassment bullying threatening intimidation|friendly_conversation cooperation respectful_interaction"; do
  IFS='|' read -r concept prompts clip targets anchors <<< "$concept_info"
  ca="--target_concepts $targets --anchor_concepts $anchors"
  for probe in both text; do
    for ss in 0.8 1.0 1.2 1.5; do
      for cas in 0.4 0.5 0.6; do
        clip_arg=""; [ "$probe" != "text" ] && clip_arg="--clip_embeddings $clip"
        thr=0.1; ithr=0.4
        [ "$probe" = "text" ] && ithr=0.1
        run 4 "c_${concept}_${probe}_ss${ss}_cas${cas}" "$prompts" \
          --probe_mode $probe --how_mode anchor_inpaint --safety_scale $ss \
          --attn_threshold $thr --img_attn_threshold $ithr --cas_threshold $cas \
          $clip_arg $ca
      done
    done
  done
  for d in $OUT/c_${concept}_*; do
    [ -d "$d" ] && eval_one 4 "$(basename $d)" "$concept"
  done
done
echo "GPU4 DONE"
) &

# =============================================================================
# GPU 5: Hate + Shocking (same grid)
# =============================================================================
(
for concept_info in \
  "hate|$I2P/i2p_hate.csv|$CEXDIR/hate/clip_exemplar_projected.pt|hate_speech discrimination racist_symbol|diversity harmony cultural_respect equality" \
  "shocking|$I2P/i2p_shocking.csv|$CEXDIR/shocking/clip_exemplar_projected.pt|grotesque body_horror disturbing_imagery|serene_scene beautiful_art harmonious_composition"; do
  IFS='|' read -r concept prompts clip targets anchors <<< "$concept_info"
  ca="--target_concepts $targets --anchor_concepts $anchors"
  for probe in both text; do
    for ss in 0.8 1.0 1.2 1.5; do
      for cas in 0.4 0.5 0.6; do
        clip_arg=""; [ "$probe" != "text" ] && clip_arg="--clip_embeddings $clip"
        thr=0.1; ithr=0.4
        [ "$probe" = "text" ] && ithr=0.1
        run 5 "c_${concept}_${probe}_ss${ss}_cas${cas}" "$prompts" \
          --probe_mode $probe --how_mode anchor_inpaint --safety_scale $ss \
          --attn_threshold $thr --img_attn_threshold $ithr --cas_threshold $cas \
          $clip_arg $ca
      done
    done
  done
  for d in $OUT/c_${concept}_*; do
    [ -d "$d" ] && eval_one 5 "$(basename $d)" "$concept"
  done
done
echo "GPU5 DONE"
) &

# =============================================================================
# GPU 6: Illegal + Self-harm (same grid)
# =============================================================================
(
for concept_info in \
  "illegal_activity|$I2P/i2p_illegal_activity.csv|$CEXDIR/illegal_activity/clip_exemplar_projected.pt|drugs crime illegal_weapons theft|legal_activity professional_work social_norm" \
  "self-harm|$I2P/i2p_self-harm.csv|$CEXDIR/self-harm/clip_exemplar_projected.pt|self-harm suicide self-injury|healing meditation healthy_activity mental_wellness"; do
  IFS='|' read -r concept prompts clip targets anchors <<< "$concept_info"
  ca="--target_concepts $targets --anchor_concepts $anchors"
  for probe in both text; do
    for ss in 0.8 1.0 1.2 1.5; do
      for cas in 0.4 0.5 0.6; do
        clip_arg=""; [ "$probe" != "text" ] && clip_arg="--clip_embeddings $clip"
        thr=0.1; ithr=0.4
        [ "$probe" = "text" ] && ithr=0.1
        run 6 "c_${concept}_${probe}_ss${ss}_cas${cas}" "$prompts" \
          --probe_mode $probe --how_mode anchor_inpaint --safety_scale $ss \
          --attn_threshold $thr --img_attn_threshold $ithr --cas_threshold $cas \
          $clip_arg $ca
      done
    done
  done
  for d in $OUT/c_${concept}_*; do
    [ -d "$d" ] && eval_one 6 "$(basename $d)" "$concept"
  done
done
echo "GPU6 DONE"
) &

# =============================================================================
# GPU 7: Nudity hybrid/target_sub × datasets + concept hybrid variants
# =============================================================================
(
# Nudity hybrid on key datasets
for ds in ringabell mma i2p_sexual; do
  for ss in 1.5 2.0 3.0; do
    run 7 "nude_both_hyb_ss${ss}_${ds}" "${NUDE_DS[$ds]}" \
      --probe_mode both --how_mode hybrid --safety_scale $ss \
      --attn_threshold 0.1 --img_attn_threshold 0.4 --cas_threshold 0.6 \
      --clip_embeddings $CLIP_NUDE
  done
done
# Concept hybrid for all 6
for concept_info in \
  "violence|$I2P/i2p_violence.csv|$CEXDIR/violence/clip_exemplar_projected.pt|violence gore bloodshed fighting weapon|peaceful_scene nature_landscape friendly_interaction" \
  "harassment|$I2P/i2p_harassment.csv|$CEXDIR/harassment/clip_exemplar_projected.pt|harassment bullying threatening intimidation|friendly_conversation cooperation respectful_interaction" \
  "hate|$I2P/i2p_hate.csv|$CEXDIR/hate/clip_exemplar_projected.pt|hate_speech discrimination racist_symbol|diversity harmony cultural_respect equality" \
  "shocking|$I2P/i2p_shocking.csv|$CEXDIR/shocking/clip_exemplar_projected.pt|grotesque body_horror disturbing_imagery|serene_scene beautiful_art harmonious_composition" \
  "illegal_activity|$I2P/i2p_illegal_activity.csv|$CEXDIR/illegal_activity/clip_exemplar_projected.pt|drugs crime illegal_weapons theft|legal_activity professional_work social_norm" \
  "self-harm|$I2P/i2p_self-harm.csv|$CEXDIR/self-harm/clip_exemplar_projected.pt|self-harm suicide self-injury|healing meditation healthy_activity mental_wellness"; do
  IFS='|' read -r concept prompts clip targets anchors <<< "$concept_info"
  ca="--target_concepts $targets --anchor_concepts $anchors"
  for ss in 1.5 2.0; do
    run 7 "c_${concept}_both_hyb_ss${ss}" "$prompts" \
      --probe_mode both --how_mode hybrid --safety_scale $ss \
      --attn_threshold 0.1 --img_attn_threshold 0.4 --cas_threshold 0.5 \
      --clip_embeddings $clip $ca
  done
done
# Eval all GPU7
for d in $OUT/nude_both_hyb_* $OUT/c_*_hyb_*; do
  [ -d "$d" ] || continue
  concept=nudity
  name=$(basename $d)
  case $name in *violence*) concept=violence;; *harassment*) concept=harassment;; *hate*) concept=hate;; *shocking*) concept=shocking;; *illegal*) concept=illegal_activity;; *self-harm*) concept=self-harm;; esac
  eval_one 7 "$name" "$concept"
done
echo "GPU7 DONE"
) &

echo "=============================================="
echo "  v27 FULL BLAST — $(date)"
echo "  GPU 0: Nudity BEST (both) × 6 datasets"
echo "  GPU 1: Nudity TEXT baseline × 6 datasets"
echo "  GPU 2: Nudity IMG ablation × 5 datasets"
echo "  GPU 3: Nudity 4-sample × 5 datasets"
echo "  GPU 4: Violence + Harassment sweep"
echo "  GPU 5: Hate + Shocking sweep"
echo "  GPU 6: Illegal + Self-harm sweep"
echo "  GPU 7: Hybrid variants + concept hybrid"
echo "=============================================="

wait
echo "ALL FULL BLAST DONE — $(date)"
