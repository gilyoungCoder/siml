#!/bin/bash
#
# run.sh
# StableDiffusion + Classifier/CLIP 혼합 가이던스 실행 스크립트
#

# 1) 사용할 GPU 지정
export CUDA_VISIBLE_DEVICES=4

# 2) 필수 인자 설정
CKPT_PATH="runwayml/stable-diffusion-v1-5"
PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/country_body.txt"

# 3) 선택적 인자 설정
NSAMPLES=1
CFG_SCALE=5.0
NUM_INFERENCE_STEPS=50
# --mixed 플래그를 없애면 classifier/CLIP 혼합 가이던스 없이 실행됩니다
MIXED="--mixed"
CLS_SCALE=1.0
CLIP_SCALE=0.1
FREEDOM_SCALE=15.0
SWITCH_STEP=25
OUTPUT_DIR="output_img"

# 4) 출력 디렉토리 생성
mkdir -p "${OUTPUT_DIR}"

# 5) 실행
python test_clip.py "${CKPT_PATH}" \
    --prompt_file "${PROMPT_FILE}" \
    --nsamples ${NSAMPLES} \
    --cfg_scale ${CFG_SCALE} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    ${MIXED} \
    --cls_scale ${CLS_SCALE} \
    --clip_scale ${CLIP_SCALE} \
    --freedom_scale ${FREEDOM_SCALE} \
    --switch_step ${SWITCH_STEP} \
    --output_dir "${OUTPUT_DIR}"
