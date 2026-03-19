#!/bin/bash
# 공통 인자들 설정
export CUDA_VISIBLE_DEVICES=1

LIGHTNING_REPO="ByteDance/SDXL-Lightning"
SDXL_BASE="stabilityai/stable-diffusion-xl-base-1.0"
PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/country_naked_body.txt"
FREEDOM_MODEL_ARGS="configs/models/time_dependent_discriminator.yaml"
FREEDOM_MODEL_CKPT="./work_dirs/31cls1024_v2/checkpoint/step_00200/classifier.pth"
GUIDE_START=0
NSAMPLES=1
STEPS=4
TXT_CFG_SCALE=0.0
SEED=42
MAX_CONCURRENCY=1

# freedom scale loop
for FREEDOM_SCALE in $(seq 5 5 30); do
    OUTPUT_DIR="./Continual/CountryNudeBody/${FREEDOM_SCALE}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "Running with FREEDOM_SCALE=${FREEDOM_SCALE}"

    nohup python generate_sdxl_lightning_guidance.py \
      $LIGHTNING_REPO \
      --sdxl_base         $SDXL_BASE \
      --prompt_file       $PROMPT_FILE \
      --output_dir        $OUTPUT_DIR \
      --nsamples          $NSAMPLES \
      --num_inference_steps $STEPS \
      --cfg_scale         $TXT_CFG_SCALE \
      --device            "cuda:0" \
      --seed              $SEED \
      --freedom \
      --freedom_model_args $FREEDOM_MODEL_ARGS \
      --freedom_model_ckpt $FREEDOM_MODEL_CKPT \
      --freedom_scale     $FREEDOM_SCALE \
      --guide_start       $GUIDE_START > "${OUTPUT_DIR}/nohup.log" 2>&1 &

    while [ $(jobs -pr | wc -l) -ge $MAX_CONCURRENCY ]; do
        wait -n
    done
done
