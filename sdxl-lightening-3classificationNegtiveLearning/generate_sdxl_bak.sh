#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1 

LIGHTNING_REPO="ByteDance/SDXL-Lightning"
SDXL_BASE="stabilityai/stable-diffusion-xl-base-1.0"
PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/country_body.txt"

# Guidance
FREEDOM_MODEL_ARGS="configs/models/time_dependent_discriminator.yaml"
FREEDOM_MODEL_CKPT="/mnt/home/yhgil99/unlearning/sdxl-lightening-3classification/work_dirs/sdxl/classifier_final.pth"
FREEDOM_SCALE=5
GUIDE_START=0

# Generation
NSAMPLES=1
STEPS=4
TXT_CFG_SCALE=0.0
SEED=42

OUTPUT_DIR="./Continual/CountryBody/$FREEDOM_SCALE"


python generate_sdxl_lightning_guidance.py \
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
  --guide_start       $GUIDE_START
