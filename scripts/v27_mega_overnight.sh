#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# v27 MEGA OVERNIGHT — siml-01 + siml-02 전부 활용
# =============================================================================
# 1. proj_replace sweep 완료 + eval
# 2. hybrid_proj sweep (v13 best 재현)
# 3. 전체 데이터셋 × best configs (ainp, hybrid, hybrid_proj, proj_replace)
# 4. 모든 concept × best config sweep
# 5. 4-sample best candidates
# =============================================================================

SERVER=$1  # siml-01 or siml-02
P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=/mnt/home3/yhgil99/unlearning/vlm
V27=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/generate_v27.py
CLIP=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_embeddings.pt
REPO=/mnt/home3/yhgil99/unlearning
OUT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/v27_mega
I2P=$REPO/SAFREE/datasets/i2p_categories
CEXDIR=$REPO/CAS_SpatialCFG/exemplars/concepts
mkdir -p $OUT

BOTH="--probe_mode both --clip_embeddings $CLIP --attn_threshold 0.1 --img_attn_threshold 0.4"
TXT="--probe_mode text --attn_threshold 0.1"
BASE="--cas_threshold 0.6 --nsamples 1 --steps 50 --seed 42"

# Datasets
declare -A DS=(
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
  CUDA_VISIBLE_DEVICES=$gpu $P $V27 --prompts "$prompts" --outdir "$outdir" $BASE "$@" 2>&1 | tail -1
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
# GPU 0: Nudity best 4 HOW modes × all datasets (1s)
# =============================================================================
(
for ds in ringabell mma p4dn unlearndiff i2p_sexual coco; do
  # anchor_inpaint (best: ss=1.2)
  run 0 "nude_ainp_${ds}" "${DS[$ds]}" --how_mode anchor_inpaint --safety_scale 1.2 $BOTH
  # hybrid (best: ts=15 as=15)
  run 0 "nude_hyb_${ds}" "${DS[$ds]}" --how_mode hybrid --target_scale 15 --anchor_scale 15 $BOTH
  # hybrid_proj (v13 best: ss=15 as=15)
  run 0 "nude_hybproj_${ds}" "${DS[$ds]}" --how_mode hybrid_proj --safety_scale 15 --anchor_scale 15 $BOTH
  # proj_replace (ps=1.5 as=1.5 tentative)
  run 0 "nude_projrep_${ds}" "${DS[$ds]}" --how_mode proj_replace --proj_scale 1.5 --anchor_scale 1.5 $BOTH
done
for ds in ringabell mma p4dn unlearndiff i2p_sexual coco; do
  for how in ainp hyb hybproj projrep; do
    eval_one 0 "nude_${how}_${ds}" nudity
  done
done
echo "GPU0 DONE"
) &

# =============================================================================
# GPU 1: Nudity text-only × all datasets × 4 HOW (baseline)
# =============================================================================
(
for ds in ringabell mma p4dn unlearndiff i2p_sexual; do
  run 1 "nude_txt_ainp_${ds}" "${DS[$ds]}" --how_mode anchor_inpaint --safety_scale 1.2 $TXT
  run 1 "nude_txt_hyb_${ds}" "${DS[$ds]}" --how_mode hybrid --target_scale 15 --anchor_scale 15 $TXT
  run 1 "nude_txt_hybproj_${ds}" "${DS[$ds]}" --how_mode hybrid_proj --safety_scale 15 --anchor_scale 15 $TXT
done
for ds in ringabell mma p4dn unlearndiff i2p_sexual; do
  for how in ainp hyb hybproj; do
    eval_one 1 "nude_txt_${how}_${ds}" nudity
  done
done
echo "GPU1 DONE"
) &

# =============================================================================
# GPU 2: proj_replace massive sweep (Ring-A-Bell)
# =============================================================================
(
for ps in 0.5 0.7 1.0 1.2 1.5 2.0 3.0; do
  for as in 0.3 0.5 0.7 1.0 1.5 2.0 3.0; do
    run 2 "projrep_ps${ps}_as${as}" "${DS[ringabell]}" --how_mode proj_replace --proj_scale $ps --anchor_scale $as $BOTH
  done
done
for d in $OUT/projrep_*; do [ -d "$d" ] && eval_one 2 "$(basename $d)" nudity; done
echo "GPU2 DONE"
) &

# =============================================================================
# GPU 3: hybrid_proj sweep (Ring-A-Bell, reproduce v13 results)
# =============================================================================
(
for ss in 5 8 10 12 15 20; do
  for as in 5 10 15 20; do
    run 3 "hybproj_ss${ss}_as${as}" "${DS[ringabell]}" --how_mode hybrid_proj --safety_scale $ss --anchor_scale $as $BOTH
  done
done
for d in $OUT/hybproj_*; do [ -d "$d" ] && eval_one 3 "$(basename $d)" nudity; done
echo "GPU3 DONE"
) &

# =============================================================================
# GPU 4: 4-sample best candidates (all HOW)
# =============================================================================
(
run 4 "4s_ainp_both" "${DS[ringabell]}" --how_mode anchor_inpaint --safety_scale 1.2 --nsamples 4 $BOTH
run 4 "4s_hyb_both_ts15as15" "${DS[ringabell]}" --how_mode hybrid --target_scale 15 --anchor_scale 15 --nsamples 4 $BOTH
run 4 "4s_hybproj_both_ss15as15" "${DS[ringabell]}" --how_mode hybrid_proj --safety_scale 15 --anchor_scale 15 --nsamples 4 $BOTH
run 4 "4s_projrep_both_ps15as15" "${DS[ringabell]}" --how_mode proj_replace --proj_scale 1.5 --anchor_scale 1.5 --nsamples 4 $BOTH
run 4 "4s_ainp_txt" "${DS[ringabell]}" --how_mode anchor_inpaint --safety_scale 1.2 --nsamples 4 $TXT
for d in $OUT/4s_*; do [ -d "$d" ] && eval_one 4 "$(basename $d)" nudity; done
echo "GPU4 DONE"
) &

# =============================================================================
# GPU 5: Violence + Harassment concept sweep
# =============================================================================
(
for concept_info in \
  "violence|$I2P/i2p_violence.csv|$CEXDIR/violence/clip_exemplar_projected.pt|violence gore bloodshed fighting weapon|peaceful_scene nature_landscape friendly_interaction" \
  "harassment|$I2P/i2p_harassment.csv|$CEXDIR/harassment/clip_exemplar_projected.pt|harassment bullying threatening intimidation|friendly_conversation cooperation respectful_interaction"; do
  IFS='|' read -r concept prompts clip targets anchors <<< "$concept_info"
  ca="--target_concepts $targets --anchor_concepts $anchors"
  for probe in both text; do
    for how_cfg in "anchor_inpaint|1.2|1.2" "hybrid|15|15" "hybrid_proj|15|15"; do
      IFS='|' read -r how ts as <<< "$how_cfg"
      clip_arg=""; [ "$probe" != "text" ] && clip_arg="--clip_embeddings $clip"
      thr="--attn_threshold 0.1"; [ "$probe" = "both" ] && thr="--attn_threshold 0.1 --img_attn_threshold 0.4"
      for cas in 0.4 0.5 0.6; do
        short=$(echo $how | cut -c1-4)
        name="c_${concept}_${probe}_${short}_ts${ts}_as${as}_cas${cas}"
        run 5 "$name" "$prompts" --how_mode $how --target_scale $ts --anchor_scale $as --cas_threshold $cas $clip_arg $thr $ca
      done
    done
  done
  for d in $OUT/c_${concept}_*; do [ -d "$d" ] && eval_one 5 "$(basename $d)" "$concept"; done
done
echo "GPU5 DONE"
) &

# =============================================================================
# GPU 6: Hate + Shocking + Illegal concept sweep
# =============================================================================
(
for concept_info in \
  "hate|$I2P/i2p_hate.csv|$CEXDIR/hate/clip_exemplar_projected.pt|hate_speech discrimination racist_symbol|diversity harmony cultural_respect equality" \
  "shocking|$I2P/i2p_shocking.csv|$CEXDIR/shocking/clip_exemplar_projected.pt|grotesque body_horror disturbing_imagery|serene_scene beautiful_art harmonious_composition" \
  "illegal_activity|$I2P/i2p_illegal_activity.csv|$CEXDIR/illegal_activity/clip_exemplar_projected.pt|drugs crime illegal_weapons theft|legal_activity professional_work social_norm"; do
  IFS='|' read -r concept prompts clip targets anchors <<< "$concept_info"
  ca="--target_concepts $targets --anchor_concepts $anchors"
  for probe in both text; do
    for how_cfg in "anchor_inpaint|1.2|1.2" "hybrid|15|15" "hybrid_proj|15|15"; do
      IFS='|' read -r how ts as <<< "$how_cfg"
      clip_arg=""; [ "$probe" != "text" ] && clip_arg="--clip_embeddings $clip"
      thr="--attn_threshold 0.1"; [ "$probe" = "both" ] && thr="--attn_threshold 0.1 --img_attn_threshold 0.4"
      for cas in 0.4 0.5 0.6; do
        short=$(echo $how | cut -c1-4)
        name="c_${concept}_${probe}_${short}_ts${ts}_as${as}_cas${cas}"
        run 6 "$name" "$prompts" --how_mode $how --target_scale $ts --anchor_scale $as --cas_threshold $cas $clip_arg $thr $ca
      done
    done
  done
  for d in $OUT/c_${concept}_*; do [ -d "$d" ] && eval_one 6 "$(basename $d)" "$concept"; done
done
echo "GPU6 DONE"
) &

# =============================================================================
# GPU 7: Self-harm + hybrid ts/as fine-grained sweep
# =============================================================================
(
# Self-harm
concept_info="self-harm|$I2P/i2p_self-harm.csv|$CEXDIR/self-harm/clip_exemplar_projected.pt|self-harm suicide self-injury|healing meditation healthy_activity mental_wellness"
IFS='|' read -r concept prompts clip targets anchors <<< "$concept_info"
ca="--target_concepts $targets --anchor_concepts $anchors"
for probe in both text; do
  for how_cfg in "anchor_inpaint|1.2|1.2" "hybrid|15|15" "hybrid_proj|15|15"; do
    IFS='|' read -r how ts as <<< "$how_cfg"
    clip_arg=""; [ "$probe" != "text" ] && clip_arg="--clip_embeddings $clip"
    thr="--attn_threshold 0.1"; [ "$probe" = "both" ] && thr="--attn_threshold 0.1 --img_attn_threshold 0.4"
    for cas in 0.4 0.5 0.6; do
      short=$(echo $how | cut -c1-4)
      name="c_${concept}_${probe}_${short}_ts${ts}_as${as}_cas${cas}"
      run 7 "$name" "$prompts" --how_mode $how --target_scale $ts --anchor_scale $as --cas_threshold $cas $clip_arg $thr $ca
    done
  done
done
for d in $OUT/c_self-harm_*; do [ -d "$d" ] && eval_one 7 "$(basename $d)" "self-harm"; done

# Fine-grained hybrid ts/as on Ring-A-Bell
for ts in 12 13 14 15 16 17 18; do
  for as in 12 13 14 15 16 17 18; do
    run 7 "hyb_fine_ts${ts}_as${as}" "${DS[ringabell]}" --how_mode hybrid --target_scale $ts --anchor_scale $as $BOTH
  done
done
for d in $OUT/hyb_fine_*; do [ -d "$d" ] && eval_one 7 "$(basename $d)" nudity; done
echo "GPU7 DONE"
) &

echo "=============================================="
echo "  v27 MEGA OVERNIGHT — $SERVER — $(date)"
echo "  GPU 0: Nudity 4 HOW × 6 datasets"
echo "  GPU 1: Nudity text-only baseline × 5 datasets"
echo "  GPU 2: proj_replace sweep (49 configs)"
echo "  GPU 3: hybrid_proj sweep (24 configs)"
echo "  GPU 4: 4-sample best (5 configs)"
echo "  GPU 5: Violence + Harassment"
echo "  GPU 6: Hate + Shocking + Illegal"
echo "  GPU 7: Self-harm + hybrid fine-tune"
echo "=============================================="

wait
echo "ALL MEGA DONE — $(date)"
