#!/usr/bin/env bash
# run_sdxl_guidance_loop.sh – 여러 freedom_scale 값으로 SDXL+Classifier Guidance 연속 실행

export CUDA_VISIBLE_DEVICES=0

SDXL_BASE="stabilityai/stable-diffusion-xl-base-1.0"
PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/country_naked_body.txt"
FREEDOM_MODEL_ARGS="./configs/models/time_dependent_discriminator.yaml"
FREEDOM_MODEL_CKPT="./work_dirs/multisdxl/bikini/checkpoint/step_08800/classifier.pth,\
./work_dirs/multisdxl/lingerie/checkpoint/step_17700/classifier.pth,\
./work_dirs/sdxl1024/classifier_final.pth"
GUIDE_START=1
NSAMPLES=1
STEPS=50
TXT_CFG_SCALE=7.5
DEVICE="cuda:0"
SEED=42

# 동시에 실행할 최대 프로세스 수
MAX_CONCURRENCY=1

# 5,10,15,...,30 까지 freedom_scale 을 바꿔가며 실행
for FREEDOM_SCALE in $(seq 5 5 25); do
  OUTPUT_DIR="./sdxl_guidance_out/CountryNudeBody/partial_multi/${FREEDOM_SCALE}"
  mkdir -p "$OUTPUT_DIR"

  echo "▶ Starting SDXL guidance with FREEDOM_SCALE=${FREEDOM_SCALE}"

  nohup python gensdxlv2.py \
    --sdxl_base         "$SDXL_BASE" \
    --prompt_file       "$PROMPT_FILE" \
    --output_dir        "$OUTPUT_DIR" \
    --nsamples          "$NSAMPLES" \
    --num_inference_steps "$STEPS" \
    --cfg_scale         "$TXT_CFG_SCALE" \
    --freedom \
    --freedom_model_args "$FREEDOM_MODEL_ARGS" \
    --freedom_model_ckpt "$FREEDOM_MODEL_CKPT" \
    --freedom_scale     "$FREEDOM_SCALE" \
    --guide_start       "$GUIDE_START" \
    --device            "$DEVICE" \
    --seed              "$SEED" \
    > "${OUTPUT_DIR}/nohup.log" 2>&1 &

  # 최대 MAX_CONCURRENCY 개의 백그라운드 잡만 유지
  while [ "$(jobs -pr | wc -l)" -ge "$MAX_CONCURRENCY" ]; do
    wait -n
  done
done

# 남아있는 모든 백그라운드 잡이 끝날 때까지 대기
wait
echo "✅ All SDXL guidance jobs finished."
