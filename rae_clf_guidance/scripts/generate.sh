#!/usr/bin/env bash
# Generate images: baseline (no guidance) + guided (with classifier)
# Uses ODE sampler (RAE default) + cfg=1.75
export CUDA_VISIBLE_DEVICES=0

cd /mnt/home/yhgil99/unlearning/rae_clf_guidance

CLASSIFIER_CKPT="work_dirs/rae_z0_classifier/classifier_final.pth"
DITDH_CKPT="pretrained_models/stage2_model.pt"
DECODER_CKPT="pretrained_models/decoder_model.pt"
STAT_PATH="pretrained_models/stat.pt"
SAMPLER="ode"
CFG=1.75
STEPS=50

# ================================================================
# 1. Baseline: no classifier guidance
# ================================================================
echo "=== Generating baseline images (no guidance, cfg=${CFG}, ${SAMPLER}) ==="
python generate.py \
  --ditdh_ckpt "$DITDH_CKPT" \
  --decoder_ckpt "$DECODER_CKPT" \
  --stat_path "$STAT_PATH" \
  --output_dir output_img/baseline_cfg${CFG}_${SAMPLER} \
  --nsamples 4 \
  --num_steps $STEPS \
  --cfg_scale $CFG \
  --sampler_mode "$SAMPLER" \
  --seed 42

# ================================================================
# 2. Guided: classifier guidance toward clothed (class 1)
# ================================================================
echo ""
echo "=== Generating guided images (target=clothed, scale=5.0, cfg=${CFG}, ${SAMPLER}) ==="
python generate.py \
  --ditdh_ckpt "$DITDH_CKPT" \
  --decoder_ckpt "$DECODER_CKPT" \
  --stat_path "$STAT_PATH" \
  --classifier_ckpt "$CLASSIFIER_CKPT" \
  --output_dir output_img/guided_scale5_cfg${CFG}_${SAMPLER} \
  --nsamples 4 \
  --num_steps $STEPS \
  --cfg_scale $CFG \
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
    --ditdh_ckpt "$DITDH_CKPT" \
    --decoder_ckpt "$DECODER_CKPT" \
    --stat_path "$STAT_PATH" \
    --classifier_ckpt "$CLASSIFIER_CKPT" \
    --output_dir "output_img/guided_scale${scale}_cfg${CFG}_${SAMPLER}" \
    --nsamples 4 \
    --num_steps $STEPS \
    --cfg_scale $CFG \
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
  --ditdh_ckpt "$DITDH_CKPT" \
  --decoder_ckpt "$DECODER_CKPT" \
  --stat_path "$STAT_PATH" \
  --classifier_ckpt "$CLASSIFIER_CKPT" \
  --output_dir output_img/guided_safe_minus_harm_cfg${CFG}_${SAMPLER} \
  --nsamples 4 \
  --num_steps $STEPS \
  --cfg_scale $CFG \
  --guidance_scale 5.0 \
  --guidance_mode safe_minus_harm \
  --safe_classes 0 1 \
  --harm_classes 2 \
  --sampler_mode "$SAMPLER" \
  --seed 42

echo ""
echo "Done! Check output_img/ for results."
