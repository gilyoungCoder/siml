#!/usr/bin/env bash
set -uo pipefail
export PYTHONNOUSERSITE=1
REPO=/mnt/home3/yhgil99/unlearning
P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=$REPO/vlm
CLIP_EMB=$REPO/CAS_SpatialCFG/exemplars/concepts/illegal_activity/clip_exemplar_projected.pt

echo "[$(date)] v27 illegal fix - both probe with clip_embeddings"

# GPU 4: ss0.8 + ss1.0 (text probe - no clip needed)
(
  for SS in 0.8 1.0; do
    for CAS in 0.4 0.5; do
      OUTDIR=$REPO/CAS_SpatialCFG/outputs/v27_final/c_illegal_text_ainp_ss${SS}_cas${CAS}
      [ -f "$OUTDIR/results_qwen3_vl_illegal.txt" ] && continue
      mkdir -p "$OUTDIR"
      echo "[$(date)] v27 illegal TEXT ss$SS cas$CAS"
      CUDA_VISIBLE_DEVICES=4 $P $REPO/CAS_SpatialCFG/generate_v27.py \
        --prompts $REPO/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv \
        --outdir "$OUTDIR" \
        --cas_threshold $CAS --probe_mode text \
        --how_mode anchor_inpaint \
        --safety_scale $SS --target_scale $SS --anchor_scale $SS \
        --attn_threshold 0.1 \
        --target_concepts "illegal" "drugs" "theft" "crime" "weapon" "contraband" "vandalism" "arson" \
        --anchor_concepts "legal activity" "daily life" "work" "community" "education" "profession" 2>&1 | tail -3
      N=$(find "$OUTDIR" -maxdepth 1 -name "*.png" | wc -l)
      echo "[$(date)] TEXT ss$SS cas$CAS: $N images"
      [ $N -ge 10 ] && CUDA_VISIBLE_DEVICES=4 $VLP $VLD/opensource_vlm_i2p_all.py "$OUTDIR" illegal qwen 2>&1 | tail -3
    done
  done
) &

# GPU 5: ss0.8 + ss1.0 + ss1.2 (both probe with clip_embeddings)
(
  for SS in 0.8 1.0 1.2; do
    for CAS in 0.4 0.5; do
      OUTDIR=$REPO/CAS_SpatialCFG/outputs/v27_final/c_illegal_both_ainp_ss${SS}_cas${CAS}
      [ -f "$OUTDIR/results_qwen3_vl_illegal.txt" ] && continue
      mkdir -p "$OUTDIR"
      echo "[$(date)] v27 illegal BOTH ss$SS cas$CAS"
      CUDA_VISIBLE_DEVICES=5 $P $REPO/CAS_SpatialCFG/generate_v27.py \
        --prompts $REPO/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv \
        --outdir "$OUTDIR" \
        --cas_threshold $CAS --probe_mode both \
        --clip_embeddings $CLIP_EMB \
        --how_mode anchor_inpaint \
        --safety_scale $SS --target_scale $SS --anchor_scale $SS \
        --attn_threshold 0.1 --img_attn_threshold 0.1 \
        --target_concepts "illegal" "drugs" "theft" "crime" "weapon" "contraband" "vandalism" "arson" \
        --anchor_concepts "legal activity" "daily life" "work" "community" "education" "profession" 2>&1 | tail -3
      N=$(find "$OUTDIR" -maxdepth 1 -name "*.png" | wc -l)
      echo "[$(date)] BOTH ss$SS cas$CAS: $N images"
      [ $N -ge 10 ] && CUDA_VISIBLE_DEVICES=5 $VLP $VLD/opensource_vlm_i2p_all.py "$OUTDIR" illegal qwen 2>&1 | tail -3
    done
  done
) &

wait
echo "[$(date)] v27 illegal all done"
