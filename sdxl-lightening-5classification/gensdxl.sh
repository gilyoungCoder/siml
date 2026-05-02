#!/usr/bin/env bash
# run_sdxl_guidance.sh – SDXL + Classifier Guidance nohup 실행 예시

export CUDA_VISIBLE_DEVICES=7
DATA=/mnt/home/yhgil99/dataset/sdxlLight
SDXL_BASE="stabilityai/stable-diffusion-xl-base-1.0"
PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/nullprompt.txt"
OUTPUT_DIR="./nullguidance"
NSAMPLES=1
STEPS=50
CFG_SCALE=7.5
DEVICE="cuda:0"
FREEDOM="--freedom"
FREEDOM_ARGS="./configs/models/time_dependent_discriminator.yaml"
FREEDOM_CKPT="./work_dirs/sdxl1024/classifier_final.pth"
FREEDOM_SCALE=15  
GUIDE_START=1
SEED=42

LOG_FILE="./nohup_$(date +%Y%m%d_%H%M%S).log"

nohup python gensdxl.py \
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
  --seed "$SEED" \
  > "$LOG_FILE" 2>&1 &

echo "SDXL + Classifier Guidance 시작됨 (로그: $LOG_FILE)"
