#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# v27 Overnight Massive Sweep — siml-01, 8 GPUs
# =============================================================================
# Grid: probe(text/img/both) × HOW(ainp/hybrid/tsub) × ss × attn_thr
# + multi-concept (violence, harassment, hate, shocking, illegal, self-harm)
# All with CLEAN 16-image exemplars
# =============================================================================

P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=/mnt/home3/yhgil99/unlearning/vlm
V27=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/generate_v27.py
RB=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/ringabell.txt
CLIP_NUDE=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_embeddings.pt
OUT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v27_clean
I2P=/mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories
CEXDIR=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/concepts
COMMON="--cas_threshold 0.6 --nsamples 1 --steps 50 --seed 42"

run() {
  local gpu=$1 name=$2 probe=$3 how=$4 ss=$5 thr=$6 prompts=$7 clip=$8
  shift 8
  local extra="$@"
  local outdir=$OUT/$name
  [ -f "${outdir}/generation_stats.json" ] && { echo "[$(date +%H:%M)] SKIP: $name"; return; }
  local clip_arg=""
  [ "$probe" != "text" ] && [ -n "$clip" ] && clip_arg="--clip_embeddings $clip"
  echo "[$(date +%H:%M)] GPU $gpu: $name"
  CUDA_VISIBLE_DEVICES=$gpu $P $V27 --prompts "$prompts" --outdir "$outdir" \
    --probe_mode $probe --how_mode $how --safety_scale $ss --attn_threshold $thr \
    $COMMON $clip_arg $extra 2>&1 | tail -1
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
# GPU 0: Ring-A-Bell — both × anchor_inpaint full sweep
# =============================================================================
(
for ss in 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.5; do
  for thr in 0.1 0.2 0.3 0.4 0.5; do
    run 0 "both_ainp_ss${ss}_t${thr}" both anchor_inpaint $ss $thr $RB $CLIP_NUDE
  done
done
# Eval all
for d in $OUT/both_ainp_*; do eval_one 0 "$(basename $d)" nudity; done
echo "GPU0 DONE"
) &

# =============================================================================
# GPU 1: Ring-A-Bell — both × hybrid full sweep
# =============================================================================
(
for ss in 0.5 1.0 1.5 2.0 3.0 5.0; do
  for thr in 0.1 0.2 0.3 0.4 0.5; do
    run 1 "both_hyb_ss${ss}_t${thr}" both hybrid $ss $thr $RB $CLIP_NUDE
  done
done
for d in $OUT/both_hyb_*; do eval_one 1 "$(basename $d)" nudity; done
echo "GPU1 DONE"
) &

# =============================================================================
# GPU 2: Ring-A-Bell — both × target_sub full sweep
# =============================================================================
(
for ss in 1.0 2.0 3.0 5.0 7.0 10.0; do
  for thr in 0.1 0.2 0.3 0.4 0.5; do
    run 2 "both_tsub_ss${ss}_t${thr}" both target_sub $ss $thr $RB $CLIP_NUDE
  done
done
for d in $OUT/both_tsub_*; do eval_one 2 "$(basename $d)" nudity; done
echo "GPU2 DONE"
) &

# =============================================================================
# GPU 3: Ring-A-Bell — text-only × 3 HOW × sweep
# =============================================================================
(
for how_ss in "anchor_inpaint|0.8" "anchor_inpaint|1.0" "anchor_inpaint|1.2" \
              "hybrid|1.0" "hybrid|1.5" "hybrid|2.0" "hybrid|3.0" \
              "target_sub|2.0" "target_sub|3.0" "target_sub|5.0"; do
  IFS='|' read -r how ss <<< "$how_ss"
  short=$(echo $how | cut -c1-4)
  for thr in 0.1 0.3 0.5; do
    run 3 "txt_${short}_ss${ss}_t${thr}" text $how $ss $thr $RB ""
  done
done
for d in $OUT/txt_*; do eval_one 3 "$(basename $d)" nudity; done
echo "GPU3 DONE"
) &

# =============================================================================
# GPU 4: Ring-A-Bell — img-only × 3 HOW × sweep
# =============================================================================
(
for how_ss in "anchor_inpaint|0.8" "anchor_inpaint|1.0" "anchor_inpaint|1.2" \
              "hybrid|1.0" "hybrid|1.5" "hybrid|2.0" "hybrid|3.0" \
              "target_sub|2.0" "target_sub|3.0" "target_sub|5.0"; do
  IFS='|' read -r how ss <<< "$how_ss"
  short=$(echo $how | cut -c1-4)
  for thr in 0.1 0.3 0.5; do
    run 4 "img_${short}_ss${ss}_t${thr}" image $how $ss $thr $RB $CLIP_NUDE
  done
done
for d in $OUT/img_*; do eval_one 4 "$(basename $d)" nudity; done
echo "GPU4 DONE"
) &

# =============================================================================
# GPU 5: Ring-A-Bell — both + fusion/gate variants
# =============================================================================
(
for fusion in union soft_union mean; do
  for ss in 0.8 1.0 1.2; do
    run 5 "both_ainp_ss${ss}_t03_${fusion}" both anchor_inpaint $ss 0.3 $RB $CLIP_NUDE "--probe_fusion $fusion"
  done
done
# With noise gate
for ss in 0.8 1.0 1.2; do
  run 5 "both_ainp_ss${ss}_t03_gate" both anchor_inpaint $ss 0.3 $RB $CLIP_NUDE "--probe_fusion union --noise_gate --noise_gate_threshold 0.1"
done
for d in $OUT/both_ainp_*fusion* $OUT/both_ainp_*gate*; do
  [ -d "$d" ] && eval_one 5 "$(basename $d)" nudity
done
echo "GPU5 DONE"
) &

# =============================================================================
# GPU 6: Multi-concept — violence, harassment, hate (best nudity config)
# =============================================================================
(
for concept_info in \
  "violence|$I2P/i2p_violence.csv|$CEXDIR/violence/clip_exemplar_projected.pt|violence gore bloodshed fighting weapon|peaceful_scene nature_landscape friendly_interaction" \
  "harassment|$I2P/i2p_harassment.csv|$CEXDIR/harassment/clip_exemplar_projected.pt|harassment bullying threatening intimidation|friendly_conversation cooperation respectful_interaction" \
  "hate|$I2P/i2p_hate.csv|$CEXDIR/hate/clip_exemplar_projected.pt|hate_speech discrimination racist_symbol|diversity harmony cultural_respect equality"; do
  IFS='|' read -r concept prompts clip targets anchors <<< "$concept_info"
  for how in anchor_inpaint hybrid; do
    for probe in both text; do
      ss=1.0; [ "$how" = "hybrid" ] && ss=1.5
      short=$(echo $how | cut -c1-4)
      name="concept_${concept}_${probe}_${short}"
      run 6 "$name" $probe $how $ss 0.3 "$prompts" "$clip" \
        "--target_concepts $targets --anchor_concepts $anchors --cas_threshold 0.5"
    done
  done
  # Eval
  for d in $OUT/concept_${concept}_*; do
    [ -d "$d" ] && eval_one 6 "$(basename $d)" "$concept"
  done
done
echo "GPU6 DONE"
) &

# =============================================================================
# GPU 7: Multi-concept — shocking, illegal, self-harm
# =============================================================================
(
for concept_info in \
  "shocking|$I2P/i2p_shocking.csv|$CEXDIR/shocking/clip_exemplar_projected.pt|grotesque body_horror disturbing_imagery|serene_scene beautiful_art harmonious_composition" \
  "illegal_activity|$I2P/i2p_illegal_activity.csv|$CEXDIR/illegal_activity/clip_exemplar_projected.pt|drugs crime illegal_weapons theft|legal_activity professional_work social_norm" \
  "self-harm|$I2P/i2p_self-harm.csv|$CEXDIR/self-harm/clip_exemplar_projected.pt|self-harm suicide self-injury|healing meditation healthy_activity mental_wellness"; do
  IFS='|' read -r concept prompts clip targets anchors <<< "$concept_info"
  for how in anchor_inpaint hybrid; do
    for probe in both text; do
      ss=1.0; [ "$how" = "hybrid" ] && ss=1.5
      short=$(echo $how | cut -c1-4)
      name="concept_${concept}_${probe}_${short}"
      run 7 "$name" $probe $how $ss 0.3 "$prompts" "$clip" \
        "--target_concepts $targets --anchor_concepts $anchors --cas_threshold 0.5"
    done
  done
  for d in $OUT/concept_${concept}_*; do
    [ -d "$d" ] && eval_one 7 "$(basename $d)" "$concept"
  done
done
echo "GPU7 DONE"
) &

echo "=============================================="
echo "  OVERNIGHT v27 MASSIVE SWEEP — $(date)"
echo "  GPU 0: both×ainp 40 configs"
echo "  GPU 1: both×hybrid 30 configs"
echo "  GPU 2: both×target_sub 30 configs"
echo "  GPU 3: text×3HOW 30 configs"
echo "  GPU 4: img×3HOW 30 configs"
echo "  GPU 5: fusion/gate variants 12 configs"
echo "  GPU 6: violence/harassment/hate 12 configs"
echo "  GPU 7: shocking/illegal/self-harm 12 configs"
echo "  Total: ~196 configs + auto Qwen eval"
echo "=============================================="

wait
echo ""
echo "ALL OVERNIGHT DONE — $(date)"
