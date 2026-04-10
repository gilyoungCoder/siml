#!/usr/bin/env bash
set -uo pipefail
export PYTHONNOUSERSITE=1
REPO=/mnt/home3/yhgil99/unlearning
P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=$REPO/vlm

echo "[$(date)] Starting siml-03 FIX dispatch"

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

# GPU 4: v27 illegal ss1.0 generation + eval
(
  for CAS in 0.4 0.5; do
    OUTDIR=$REPO/CAS_SpatialCFG/outputs/v27_final/c_illegal_both_ainp_ss1.0_cas${CAS}
    [ -f "$OUTDIR/results_qwen3_vl_illegal.txt" ] && continue
    rm -rf "$OUTDIR"
    mkdir -p "$OUTDIR"
    echo "[$(date)] v27 illegal both_ainp ss1.0 cas$CAS GENERATION"
    CUDA_VISIBLE_DEVICES=4 $P $REPO/CAS_SpatialCFG/generate_v27.py \
      --prompts $REPO/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv \
      --outdir "$OUTDIR" \
      --cas_threshold $CAS \
      --probe_mode both \
      --how_mode anchor_inpaint \
      --safety_scale 1.0 --target_scale 1.0 --anchor_scale 1.0 \
      --attn_threshold 0.1 --img_attn_threshold 0.1 \
      --target_concepts "illegal" "drugs" "theft" "crime" "weapon" "contraband" "vandalism" "arson" \
      --anchor_concepts "legal activity" "daily life" "work" "community" "education" "profession" 2>&1 | tail -5
    N=$(find "$OUTDIR" -maxdepth 1 -name "*.png" | wc -l)
    echo "[$(date)] Generated $N images. Running eval..."
    [ $N -ge 10 ] && CUDA_VISIBLE_DEVICES=4 $VLP $VLD/opensource_vlm_i2p_all.py "$OUTDIR" illegal qwen 2>&1 | tail -3
  done
) &

# GPU 5: v27 illegal ss1.2 generation + eval
(
  for CAS in 0.4 0.5; do
    OUTDIR=$REPO/CAS_SpatialCFG/outputs/v27_final/c_illegal_both_ainp_ss1.2_cas${CAS}
    [ -f "$OUTDIR/results_qwen3_vl_illegal.txt" ] && continue
    rm -rf "$OUTDIR"
    mkdir -p "$OUTDIR"
    echo "[$(date)] v27 illegal both_ainp ss1.2 cas$CAS GENERATION"
    CUDA_VISIBLE_DEVICES=5 $P $REPO/CAS_SpatialCFG/generate_v27.py \
      --prompts $REPO/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv \
      --outdir "$OUTDIR" \
      --cas_threshold $CAS \
      --probe_mode both \
      --how_mode anchor_inpaint \
      --safety_scale 1.2 --target_scale 1.2 --anchor_scale 1.2 \
      --attn_threshold 0.1 --img_attn_threshold 0.1 \
      --target_concepts "illegal" "drugs" "theft" "crime" "weapon" "contraband" "vandalism" "arson" \
      --anchor_concepts "legal activity" "daily life" "work" "community" "education" "profession" 2>&1 | tail -5
    N=$(find "$OUTDIR" -maxdepth 1 -name "*.png" | wc -l)
    echo "[$(date)] Generated $N images. Running eval..."
    [ $N -ge 10 ] && CUDA_VISIBLE_DEVICES=5 $VLP $VLD/opensource_vlm_i2p_all.py "$OUTDIR" illegal qwen 2>&1 | tail -3
  done
) &

# GPU 7: VQAScore (using sdd_copy python with t2v_metrics if available)
(
  echo "[$(date)] Starting VQAScore evaluations"
  # Check if vqascore works with sdd_copy
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
    CUDA_VISIBLE_DEVICES=7 $P $VLD/eval_vqascore.py "$IMGDIR" --prompts "$PFILE" 2>&1 | tail -3
  done
  echo "[$(date)] VQAScore done"
) &

wait
echo "[$(date)] All siml-03 FIX jobs done"
