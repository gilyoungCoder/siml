#!/usr/bin/env bash
set -uo pipefail
export PYTHONNOUSERSITE=1
REPO=/mnt/home3/yhgil99/unlearning
P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=$REPO/vlm
LOG=$REPO/logs/siml03_dispatch_$(date +%Y%m%d_%H%M).log

echo "[$(date)] Starting siml-03 dispatch" | tee -a $LOG
mkdir -p $REPO/logs

# GPU 0: SAFREE nudity_rab + nudity_ud
(
  bash $REPO/scripts/run_safree_official.sh 0 nudity_rab 2>&1 | tee -a $REPO/logs/safree_nudity_rab.log
  bash $REPO/scripts/run_safree_official.sh 0 nudity_ud  2>&1 | tee -a $REPO/logs/safree_nudity_ud.log
) &

# GPU 1: SAFREE nudity_p4dn + nudity_mma
(
  bash $REPO/scripts/run_safree_official.sh 1 nudity_p4dn 2>&1 | tee -a $REPO/logs/safree_nudity_p4dn.log
  bash $REPO/scripts/run_safree_official.sh 1 nudity_mma  2>&1 | tee -a $REPO/logs/safree_nudity_mma.log
) &

# GPU 2: SAFREE nudity_i2p
(
  bash $REPO/scripts/run_safree_official.sh 2 nudity_i2p 2>&1 | tee -a $REPO/logs/safree_nudity_i2p.log
) &

# GPU 3-5: v27 Illegal Activity (6 configs, 2 per GPU)
(
  for CAS in 0.4 0.5; do
    OUTDIR=$REPO/CAS_SpatialCFG/outputs/v27_final/c_illegal_both_ainp_ss0.8_cas${CAS}
    [ -f "$OUTDIR/results_qwen3_vl_illegal.txt" ] && continue
    mkdir -p "$OUTDIR"
    echo "[$(date)] v27 illegal both_ainp ss0.8 cas$CAS"
    CUDA_VISIBLE_DEVICES=3 $P $REPO/CAS_SpatialCFG/generate_v27.py \
      --prompts $REPO/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv \
      --outdir "$OUTDIR" \
      --cas_threshold $CAS \
      --probe_mode both \
      --how_mode anchor_inpaint \
      --safety_scale 0.8 --target_scale 0.8 --anchor_scale 0.8 \
      --attn_threshold 0.1 --img_attn_threshold 0.1 \
      --target_concepts "illegal" "drugs" "theft" "crime" "weapon" "contraband" "vandalism" "arson" \
      --anchor_concepts "legal activity" "daily life" "work" "community" "education" "profession"
    CUDA_VISIBLE_DEVICES=3 $VLP $VLD/opensource_vlm_i2p_all.py "$OUTDIR" illegal qwen
  done
) &

(
  for CAS in 0.4 0.5; do
    OUTDIR=$REPO/CAS_SpatialCFG/outputs/v27_final/c_illegal_both_ainp_ss1.0_cas${CAS}
    [ -f "$OUTDIR/results_qwen3_vl_illegal.txt" ] && continue
    mkdir -p "$OUTDIR"
    echo "[$(date)] v27 illegal both_ainp ss1.0 cas$CAS"
    CUDA_VISIBLE_DEVICES=4 $P $REPO/CAS_SpatialCFG/generate_v27.py \
      --prompts $REPO/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv \
      --outdir "$OUTDIR" \
      --cas_threshold $CAS \
      --probe_mode both \
      --how_mode anchor_inpaint \
      --safety_scale 1.0 --target_scale 1.0 --anchor_scale 1.0 \
      --attn_threshold 0.1 --img_attn_threshold 0.1 \
      --target_concepts "illegal" "drugs" "theft" "crime" "weapon" "contraband" "vandalism" "arson" \
      --anchor_concepts "legal activity" "daily life" "work" "community" "education" "profession"
    CUDA_VISIBLE_DEVICES=4 $VLP $VLD/opensource_vlm_i2p_all.py "$OUTDIR" illegal qwen
  done
) &

(
  for CAS in 0.4 0.5; do
    OUTDIR=$REPO/CAS_SpatialCFG/outputs/v27_final/c_illegal_both_ainp_ss1.2_cas${CAS}
    [ -f "$OUTDIR/results_qwen3_vl_illegal.txt" ] && continue
    mkdir -p "$OUTDIR"
    echo "[$(date)] v27 illegal both_ainp ss1.2 cas$CAS"
    CUDA_VISIBLE_DEVICES=5 $P $REPO/CAS_SpatialCFG/generate_v27.py \
      --prompts $REPO/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv \
      --outdir "$OUTDIR" \
      --cas_threshold $CAS \
      --probe_mode both \
      --how_mode anchor_inpaint \
      --safety_scale 1.2 --target_scale 1.2 --anchor_scale 1.2 \
      --attn_threshold 0.1 --img_attn_threshold 0.1 \
      --target_concepts "illegal" "drugs" "theft" "crime" "weapon" "contraband" "vandalism" "arson" \
      --anchor_concepts "legal activity" "daily life" "work" "community" "education" "profession"
    CUDA_VISIBLE_DEVICES=5 $VLP $VLD/opensource_vlm_i2p_all.py "$OUTDIR" illegal qwen
  done
) &

# GPU 6: Q16 on I2P (baseline + v27 + SDE)
(
  echo "[$(date)] Starting Q16 evaluations"
  
  # Baseline I2P concepts
  for concept_dir in $REPO/unlearning-baselines/official_rerun/baseline_guided/nudity_i2p \
                     $REPO/unlearning-baselines/official_rerun/baseline_guided/violence \
                     $REPO/unlearning-baselines/official_rerun/baseline_guided/harassment \
                     $REPO/unlearning-baselines/official_rerun/baseline_guided/hate \
                     $REPO/unlearning-baselines/official_rerun/baseline_guided/shocking \
                     $REPO/unlearning-baselines/official_rerun/baseline_guided/illegal_activity \
                     $REPO/unlearning-baselines/official_rerun/baseline_guided/self_harm; do
    [ -f "$concept_dir/results_q16.txt" ] && continue
    N=$(find "$concept_dir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
    [ $N -lt 10 ] && continue
    echo "  Q16 baseline: $(basename $concept_dir)"
    CUDA_VISIBLE_DEVICES=6 $P $VLD/eval_q16.py "$concept_dir" 2>&1 | tail -3
  done

  # v27 I2P best configs
  for v27dir in $REPO/CAS_SpatialCFG/outputs/v27_final/nude_i2p_both_ainp \
                $REPO/CAS_SpatialCFG/outputs/v27_final/c_violence_text_ainp_ss1.5_cas0.4 \
                $REPO/CAS_SpatialCFG/outputs/v27_final/c_harassment_text_ainp_ss1.2_cas0.4 \
                $REPO/CAS_SpatialCFG/outputs/v27_final/c_hate_text_ainp_ss1.2_cas0.4 \
                $REPO/CAS_SpatialCFG/outputs/v27_final/c_shocking_both_ainp_ss0.8_cas0.6 \
                $REPO/CAS_SpatialCFG/outputs/v27_final/c_selfharm_both_ainp_ss0.8_cas0.4; do
    [ -f "$v27dir/results_q16.txt" ] && continue
    N=$(find "$v27dir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
    [ $N -lt 10 ] && continue
    echo "  Q16 v27: $(basename $v27dir)"
    CUDA_VISIBLE_DEVICES=6 $P $VLD/eval_q16.py "$v27dir" 2>&1 | tail -3
  done

  # SDE I2P
  for sdedir in $REPO/unlearning-baselines/outputs/sderasure_v12/nudity_i2p \
                $REPO/unlearning-baselines/outputs/sderasure_v12/violence \
                $REPO/unlearning-baselines/outputs/sderasure_v12/harassment \
                $REPO/unlearning-baselines/outputs/sderasure_v12/hate \
                $REPO/unlearning-baselines/outputs/sderasure_v12/shocking \
                $REPO/unlearning-baselines/outputs/sderasure_v12/illegal_activity \
                $REPO/unlearning-baselines/outputs/sderasure_v12/self_harm; do
    [ -f "$sdedir/results_q16.txt" ] && continue
    N=$(find "$sdedir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
    [ $N -lt 10 ] && continue
    echo "  Q16 SDE: $(basename $sdedir)"
    CUDA_VISIBLE_DEVICES=6 $P $VLD/eval_q16.py "$sdedir" 2>&1 | tail -3
  done

  # SLD I2P  
  for slddir in $REPO/unlearning-baselines/outputs/sld_official/nudity_i2p \
                $REPO/unlearning-baselines/outputs/sld_official/violence \
                $REPO/unlearning-baselines/outputs/sld_official/harassment \
                $REPO/unlearning-baselines/outputs/sld_official/hate \
                $REPO/unlearning-baselines/outputs/sld_official/shocking \
                $REPO/unlearning-baselines/outputs/sld_official/illegal \
                $REPO/unlearning-baselines/outputs/sld_official/self_harm; do
    [ -f "$slddir/results_q16.txt" ] && continue
    N=$(find "$slddir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
    [ $N -lt 10 ] && continue
    echo "  Q16 SLD: $(basename $slddir)"
    CUDA_VISIBLE_DEVICES=6 $P $VLD/eval_q16.py "$slddir" 2>&1 | tail -3
  done

  echo "[$(date)] Q16 all done"
) &

# GPU 7: VQAScore v27_final best configs
(
  echo "[$(date)] Starting VQAScore evaluations"
  VQAP=/mnt/home3/yhgil99/.conda/envs/vqascore/bin/python
  
  for cfg_pair in \
    "nude_rb_both_hyb_ts15:$REPO/CAS_SpatialCFG/prompts/ringabell.txt" \
    "nude_p4dn_text_ainp:$REPO/CAS_SpatialCFG/prompts/p4dn_16_prompt.csv" \
    "nude_unlearndiff_both_ainp:$REPO/CAS_SpatialCFG/prompts/unlearn_diff_nudity.csv" \
    "nude_mma_text_ainp:$REPO/CAS_SpatialCFG/prompts/mma-diffusion-nsfw-adv-prompts.csv" \
    "nude_i2p_both_ainp:$REPO/SAFREE/datasets/i2p_categories/i2p_sexual.csv"; do
    
    CFG=$(echo $cfg_pair | cut -d: -f1)
    PFILE=$(echo $cfg_pair | cut -d: -f2)
    IMGDIR=$REPO/CAS_SpatialCFG/outputs/v27_final/$CFG
    
    [ -f "$IMGDIR/results_vqascore.txt" ] && continue
    [ ! -d "$IMGDIR" ] && continue
    
    echo "  VQAScore: $CFG"
    CUDA_VISIBLE_DEVICES=7 $VQAP $VLD/eval_vqascore.py "$IMGDIR" --prompts "$PFILE" 2>&1 | tail -3
  done
  
  echo "[$(date)] VQAScore all done"
) &

wait
echo "[$(date)] All siml-03 jobs done" | tee -a $LOG
