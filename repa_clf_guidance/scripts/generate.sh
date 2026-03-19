#!/usr/bin/env bash
# Generate images: baseline (no guidance) + guided (with classifier)
# Uses SDE sampler + cfg=4.0 (REPA paper Figure 15 setting for sharp visuals)
export CUDA_VISIBLE_DEVICES=0

cd /mnt/home/yhgil99/unlearning/repa_clf_guidance

CLASSIFIER_CKPT="work_dirs/repa_z0_classifier/classifier_final.pth"
SAMPLER="sde"
CFG=4.0
GHIGH=1.0

# ================================================================
# 1. Baseline: no classifier guidance
# ================================================================
echo "=== Generating baseline images (no guidance, cfg=${CFG}, ${SAMPLER}) ==="
python generate.py \
  --output_dir output_img/baseline_cfg${CFG}_${SAMPLER} \
  --nsamples 4 \
  --num_steps 250 \
  --cfg_scale $CFG \
  --guidance_high $GHIGH \
  --sampler_mode "$SAMPLER" \
  --seed 42

# ================================================================
# 2. Guided: classifier guidance toward clothed (class 1)
# ================================================================
echo ""
echo "=== Generating guided images (target=clothed, scale=5.0, cfg=${CFG}, ${SAMPLER}) ==="
python generate.py \
  --classifier_ckpt "$CLASSIFIER_CKPT" \
  --output_dir output_img/guided_scale5_cfg${CFG}_${SAMPLER} \
  --nsamples 4 \
  --num_steps 250 \
  --cfg_scale $CFG \
  --guidance_high $GHIGH \
  --guidance_scale 5.0 \
  --target_class 1 \
  --guidance_mode target \
  --sampler_mode "$SAMPLER" \
  --seed 42

# ================================================================
# 3. Guided: guidance scale sweep
# ================================================================
for scale in 1.0 3.0 10.0 20.0; do
  echo ""
  echo "=== Generating guided images (scale=${scale}, cfg=${CFG}, ${SAMPLER}) ==="
  python generate.py \
    --classifier_ckpt "$CLASSIFIER_CKPT" \
    --output_dir "output_img/guided_scale${scale}_cfg${CFG}_${SAMPLER}" \
    --nsamples 4 \
    --num_steps 250 \
    --cfg_scale $CFG \
    --guidance_high $GHIGH \
    --guidance_scale "$scale" \
    --target_class 1 \
    --guidance_mode target \
    --sampler_mode "$SAMPLER" \
    --seed 42
done

# ================================================================
# 4. Safe minus harm mode
# ================================================================
echo ""
echo "=== Generating guided images (safe_minus_harm, scale=5.0, cfg=${CFG}, ${SAMPLER}) ==="
python generate.py \
  --classifier_ckpt "$CLASSIFIER_CKPT" \
  --output_dir output_img/guided_safe_minus_harm_cfg${CFG}_${SAMPLER} \
  --nsamples 4 \
  --num_steps 250 \
  --cfg_scale $CFG \
  --guidance_high $GHIGH \
  --guidance_scale 5.0 \
  --guidance_mode safe_minus_harm \
  --safe_classes 0 1 \
  --harm_classes 2 \
  --sampler_mode "$SAMPLER" \
  --seed 42

echo ""
echo "Done! Check output_img/ for results."
