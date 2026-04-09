#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# REMAINING TASKS — Fill ALL gaps + Probe viz + VQAScore + COCO FID
# Run on: siml-02 GPU 0,3,6,7
# =============================================================================

P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=/mnt/home3/yhgil99/unlearning/vlm
REPO=/mnt/home3/yhgil99/unlearning
OUT=$REPO/unlearning-baselines/outputs
I2P=$REPO/SAFREE/datasets/i2p_categories
V27=$REPO/CAS_SpatialCFG/generate_v27.py
CLIP=$REPO/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_embeddings.pt

eval_qwen() {
  local gpu=$1 dir=$2 concept=$3
  [ -f "${dir}/categories_qwen3_vl_${concept}.json" ] && return
  local n=$(find "$dir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  [ $n -lt 10 ] && return
  echo "[$(date +%H:%M)] Eval $(basename $dir) / $concept ($n imgs)"
  CUDA_VISIBLE_DEVICES=$gpu $VLP $VLD/opensource_vlm_i2p_all.py "$dir" "$concept" qwen 2>&1 | tail -1
}

# =============================================================================
# GPU 0: Ours — illegal_activity + self_harm eval (v27_final)
# =============================================================================
(
echo "=== GPU 0: Our method eval (illegal, self_harm) ==="
V27F=$REPO/CAS_SpatialCFG/outputs/v27_final
for d in $V27F/c_illegal_activity_*; do
  [ -d "$d" ] && eval_qwen 0 "$d" illegal
done
for d in $V27F/c_selfharm_*; do
  [ -d "$d" ] && eval_qwen 0 "$d" self_harm
done
echo "GPU0 DONE — $(date)"
) &

# =============================================================================
# GPU 3: SDErasure v12 remaining eval + gen (i2p, hate, shocking)
# =============================================================================
(
echo "=== GPU 3: SDErasure v12 remaining ==="
SDE_UNET=$REPO/SDErasure/outputs/sderasure_nudity_v12/unet
GEN=$REPO/SDErasure/generate_from_prompts.py

# Eval I2P Sexual (already generated)
eval_qwen 3 "$OUT/sderasure_v12/nudity_i2p" nudity

# Eval harassment (already generated)
eval_qwen 3 "$OUT/sderasure_v12/harassment" harassment

# Gen + eval hate
ODIR=$OUT/sderasure_v12/hate
mkdir -p "$ODIR"
N=$(find "$ODIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
if [ $N -lt 50 ] && [ -d "$REPO/SDErasure/outputs/sderasure_hate/unet" ]; then
  echo "[$(date +%H:%M)] GPU 3: SDErasure hate"
  CUDA_VISIBLE_DEVICES=3 $P $GEN --model_id CompVis/stable-diffusion-v1-4 \
    --unet_dir $REPO/SDErasure/outputs/sderasure_hate/unet \
    --prompt_file $I2P/i2p_hate.csv --output_dir "$ODIR" \
    --seed 42 --num_inference_steps 50 --guidance_scale 7.5
fi
eval_qwen 3 "$ODIR" hate

# Gen + eval shocking
ODIR=$OUT/sderasure_v12/shocking
mkdir -p "$ODIR"
N=$(find "$ODIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
if [ $N -lt 50 ] && [ -d "$REPO/SDErasure/outputs/sderasure_shocking/unet" ]; then
  echo "[$(date +%H:%M)] GPU 3: SDErasure shocking"
  CUDA_VISIBLE_DEVICES=3 $P $GEN --model_id CompVis/stable-diffusion-v1-4 \
    --unet_dir $REPO/SDErasure/outputs/sderasure_shocking/unet \
    --prompt_file $I2P/i2p_shocking.csv --output_dir "$ODIR" \
    --seed 42 --num_inference_steps 50 --guidance_scale 7.5
fi
eval_qwen 3 "$ODIR" shocking
echo "GPU3 DONE — $(date)"
) &

# =============================================================================
# GPU 6: Baseline self_harm + Probe visualization
# =============================================================================
(
echo "=== GPU 6: Baseline self_harm + Probe viz ==="

# Baseline self_harm
ODIR=$OUT/baseline/self_harm
mkdir -p "$ODIR"
N=$(find "$ODIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
if [ $N -lt 50 ]; then
  echo "[$(date +%H:%M)] GPU 6: Baseline self_harm"
  CUDA_VISIBLE_DEVICES=6 $P $V27 \
    --prompts $I2P/i2p_self-harm.csv --outdir "$ODIR" \
    --nsamples 1 --steps 50 --seed 42 --cas_threshold 99.0 --safety_scale 0.0 \
    --how_mode anchor_inpaint --probe_mode text --attn_threshold 0.1
fi
eval_qwen 6 "$ODIR" self_harm

# Probe visualization — generate with save_maps=True
echo "[$(date +%H:%M)] GPU 6: Probe visualization"
VIZDIR=$REPO/CAS_SpatialCFG/outputs/probe_viz
mkdir -p "$VIZDIR"
CUDA_VISIBLE_DEVICES=6 $P $V27 \
  --prompts $REPO/CAS_SpatialCFG/prompts/ringabell.txt \
  --outdir "$VIZDIR" \
  --nsamples 1 --steps 50 --seed 42 --cas_threshold 0.6 \
  --how_mode hybrid --target_scale 15 --anchor_scale 15 \
  --probe_mode both --clip_embeddings $CLIP \
  --attn_threshold 0.1 --img_attn_threshold 0.4 \
  --save_maps --debug \
  --start_idx 0 --end_idx 10 2>&1 | tail -5

echo "GPU6 DONE — $(date)"
) &

# =============================================================================
# GPU 7: COCO FID for v27 best config
# =============================================================================
(
echo "=== GPU 7: COCO FID v27 hyb ts15 as15 ==="
bash $REPO/scripts/run_coco_fid_v27.sh 7 2>&1 | tail -10
echo "GPU7 DONE — $(date)"
) &

echo "=============================================="
echo "  siml-02: GPU 0(our eval) 3(SDE remaining) 6(baseline+viz) 7(FID)"
echo "=============================================="

wait
echo "ALL DONE — $(date)"
