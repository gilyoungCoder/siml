#!/usr/bin/env bash
# run_sdxl_guidance.sh – SDXL + Classifier Guidance 실행 예시

export CUDA_VISIBLE_DEVICES=3
SDXL_BASE="stabilityai/stable-diffusion-xl-base-1.0"
PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/unconditional.txt"
OUTPUT_DIR="./sdxl_guidance_out/Unconditional1"
NSAMPLES=9
STEPS=50
CFG_SCALE=7.5
DEVICE="cuda:0"
FREEDOM="--freedom"
FREEDOM_ARGS="./configs/models/time_dependent_discriminator.yaml"
FREEDOM_CKPT="./work_dirs/31cls1024_v2/checkpoint/step_00200/classifier.pth"
FREEDOM_SCALE=25  
GUIDE_START=1
SEED=42

python gensdxl.py \
  --sdxl_base "$SDXL_BASE" \
  --prompt_file "$PROMPT_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --nsamples "$NSAMPLES" \
  --num_inference_steps "$STEPS" \
  --cfg_scale "$CFG_SCALE" \
  $FREEDOM \
  --freedom_model_args "$FREEDOM_ARGS" \
  --freedom_model_ckpt "$FREEDOM_CKPT" \
  --freedom_scale "$FREEDOM_SCALE" \
  --guide_start "$GUIDE_START" \
  --device "$DEVICE" \
  --seed "$SEED"

echo "SDXL + Classifier Guidance 시작 (로그: $LOG_FILE)"
