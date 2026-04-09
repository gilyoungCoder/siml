#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# v27 Overnight EXTRA — GPU 5,6,7 확장 (각 100+ configs)
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

run() {
  local gpu=$1 name=$2 probe=$3 how=$4 ss=$5 thr=$6 prompts=$7 clip=$8
  shift 8
  local extra="$@"
  local outdir=$OUT/$name
  [ -f "${outdir}/generation_stats.json" ] && return
  local clip_arg=""
  [ "$probe" != "text" ] && [ -n "$clip" ] && clip_arg="--clip_embeddings $clip"
  echo "[$(date +%H:%M)] GPU $gpu: $name"
  CUDA_VISIBLE_DEVICES=$gpu $P $V27 --prompts "$prompts" --outdir "$outdir" \
    --probe_mode $probe --how_mode $how --safety_scale $ss --attn_threshold $thr \
    --cas_threshold 0.6 --nsamples 1 --steps 50 --seed 42 $clip_arg $extra 2>&1 | tail -1
}

eval_one() {
  local gpu=$1 outdir=$2 concept=$3
  [ -f "${outdir}/categories_qwen3_vl_${concept}.json" ] && return
  local imgs=$(find "$outdir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  [ $imgs -lt 50 ] && return
  CUDA_VISIBLE_DEVICES=$gpu $VLP $VLD/opensource_vlm_i2p_all.py "$outdir" "$concept" qwen 2>&1 | tail -1
}

# =============================================================================
# GPU 5: Ring-A-Bell massive — fusion/gate × all HOW × all probe × ss sweep
# ~120 configs
# =============================================================================
(
# Fusion variants × HOW × ss
for fusion in union soft_union mean; do
  for how_ss in "anchor_inpaint|0.7" "anchor_inpaint|0.8" "anchor_inpaint|0.9" \
                "anchor_inpaint|1.0" "anchor_inpaint|1.1" "anchor_inpaint|1.2" "anchor_inpaint|1.3" \
                "hybrid|0.5" "hybrid|1.0" "hybrid|1.5" "hybrid|2.0" "hybrid|3.0" \
                "target_sub|1.0" "target_sub|2.0" "target_sub|3.0" "target_sub|5.0"; do
    IFS='|' read -r how ss <<< "$how_ss"
    short=$(echo $how | cut -c1-4)
    for thr in 0.2 0.3 0.4; do
      run 5 "both_${short}_ss${ss}_t${thr}_${fusion}" both $how $ss $thr $RB $CLIP_NUDE "--probe_fusion $fusion"
    done
  done
done
# Noise gate variants
for how_ss in "anchor_inpaint|0.8" "anchor_inpaint|1.0" "anchor_inpaint|1.2" \
              "hybrid|1.0" "hybrid|1.5" "hybrid|2.0" \
              "target_sub|2.0" "target_sub|3.0" "target_sub|5.0"; do
  IFS='|' read -r how ss <<< "$how_ss"
  short=$(echo $how | cut -c1-4)
  for thr in 0.2 0.3 0.4; do
    for gate_thr in 0.05 0.1 0.2; do
      run 5 "both_${short}_ss${ss}_t${thr}_gate${gate_thr}" both $how $ss $thr $RB $CLIP_NUDE \
        "--probe_fusion union --noise_gate --noise_gate_threshold $gate_thr"
    done
  done
done
# Eval all GPU5 results
for d in $OUT/both_*_union $OUT/both_*_soft_union $OUT/both_*_mean $OUT/both_*_gate*; do
  [ -d "$d" ] && eval_one 5 "$d" nudity
done
echo "GPU5 DONE — $(date)"
) &

# =============================================================================
# GPU 6: Multi-concept — violence, harassment, hate
# probe(text/img/both) × HOW(ainp/hybrid/tsub) × ss sweep × thr sweep
# ~100+ configs
# =============================================================================
(
for concept_info in \
  "violence|$I2P/i2p_violence.csv|$CEXDIR/violence/clip_exemplar_projected.pt|violence gore bloodshed fighting weapon|peaceful_scene nature_landscape friendly_interaction" \
  "harassment|$I2P/i2p_harassment.csv|$CEXDIR/harassment/clip_exemplar_projected.pt|harassment bullying threatening intimidation|friendly_conversation cooperation respectful_interaction" \
  "hate|$I2P/i2p_hate.csv|$CEXDIR/hate/clip_exemplar_projected.pt|hate_speech discrimination racist_symbol|diversity harmony cultural_respect equality"; do

  IFS='|' read -r concept prompts clip targets anchors <<< "$concept_info"
  concept_args="--target_concepts $targets --anchor_concepts $anchors"

  for probe in both text image; do
    for how in anchor_inpaint hybrid target_sub; do
      short=$(echo $how | cut -c1-4)
      # ss depends on HOW
      case $how in
        anchor_inpaint) ss_list="0.8 1.0 1.2" ;;
        hybrid) ss_list="1.0 1.5 2.0 3.0" ;;
        target_sub) ss_list="2.0 3.0 5.0" ;;
      esac
      for ss in $ss_list; do
        for thr in 0.2 0.3 0.4 0.5; do
          for cas_thr in 0.4 0.5 0.6; do
            clip_use=$clip
            [ "$probe" = "text" ] && clip_use=""
            name="c_${concept}_${probe}_${short}_ss${ss}_t${thr}_cas${cas_thr}"
            outdir=$OUT/$name
            [ -f "${outdir}/generation_stats.json" ] && continue
            local_clip=""
            [ "$probe" != "text" ] && [ -n "$clip_use" ] && local_clip="--clip_embeddings $clip_use"
            echo "[$(date +%H:%M)] GPU 6: $name"
            CUDA_VISIBLE_DEVICES=6 $P $V27 --prompts "$prompts" --outdir "$outdir" \
              --probe_mode $probe --how_mode $how --safety_scale $ss --attn_threshold $thr \
              --cas_threshold $cas_thr --nsamples 1 --steps 50 --seed 42 \
              $local_clip $concept_args 2>&1 | tail -1
          done
        done
      done
    done
  done
  # Eval this concept
  for d in $OUT/c_${concept}_*; do
    [ -d "$d" ] && eval_one 6 "$d" "$concept"
  done
done
echo "GPU6 DONE — $(date)"
) &

# =============================================================================
# GPU 7: Multi-concept — shocking, illegal, self-harm (same grid as GPU 6)
# ~100+ configs
# =============================================================================
(
for concept_info in \
  "shocking|$I2P/i2p_shocking.csv|$CEXDIR/shocking/clip_exemplar_projected.pt|grotesque body_horror disturbing_imagery|serene_scene beautiful_art harmonious_composition" \
  "illegal_activity|$I2P/i2p_illegal_activity.csv|$CEXDIR/illegal_activity/clip_exemplar_projected.pt|drugs crime illegal_weapons theft|legal_activity professional_work social_norm" \
  "self-harm|$I2P/i2p_self-harm.csv|$CEXDIR/self-harm/clip_exemplar_projected.pt|self-harm suicide self-injury|healing meditation healthy_activity mental_wellness"; do

  IFS='|' read -r concept prompts clip targets anchors <<< "$concept_info"
  concept_args="--target_concepts $targets --anchor_concepts $anchors"

  for probe in both text image; do
    for how in anchor_inpaint hybrid target_sub; do
      short=$(echo $how | cut -c1-4)
      case $how in
        anchor_inpaint) ss_list="0.8 1.0 1.2" ;;
        hybrid) ss_list="1.0 1.5 2.0 3.0" ;;
        target_sub) ss_list="2.0 3.0 5.0" ;;
      esac
      for ss in $ss_list; do
        for thr in 0.2 0.3 0.4 0.5; do
          for cas_thr in 0.4 0.5 0.6; do
            clip_use=$clip
            [ "$probe" = "text" ] && clip_use=""
            name="c_${concept}_${probe}_${short}_ss${ss}_t${thr}_cas${cas_thr}"
            outdir=$OUT/$name
            [ -f "${outdir}/generation_stats.json" ] && continue
            local_clip=""
            [ "$probe" != "text" ] && [ -n "$clip_use" ] && local_clip="--clip_embeddings $clip_use"
            echo "[$(date +%H:%M)] GPU 7: $name"
            CUDA_VISIBLE_DEVICES=7 $P $V27 --prompts "$prompts" --outdir "$outdir" \
              --probe_mode $probe --how_mode $how --safety_scale $ss --attn_threshold $thr \
              --cas_threshold $cas_thr --nsamples 1 --steps 50 --seed 42 \
              $local_clip $concept_args 2>&1 | tail -1
          done
        done
      done
    done
  done
  for d in $OUT/c_${concept}_*; do
    [ -d "$d" ] && eval_one 7 "$d" "$concept"
  done
done
echo "GPU7 DONE — $(date)"
) &

echo "=============================================="
echo "  EXTRA SWEEP — $(date)"
echo "  GPU 5: ~120 fusion/gate Ring-A-Bell configs"
echo "  GPU 6: ~100+ violence/harassment/hate configs"
echo "  GPU 7: ~100+ shocking/illegal/self-harm configs"
echo "=============================================="

wait
echo "ALL EXTRA DONE — $(date)"
