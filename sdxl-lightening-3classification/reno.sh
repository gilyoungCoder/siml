#!/bin/bash

# GPU 지정
export CUDA_VISIBLE_DEVICES=7

# (필요시) 가상환경 활성화
# source /path/to/venv/bin/activate

########################
# 필수 인자 설정
########################
CKPT_PATH="runwayml/stable-diffusion-v1-5"   # Stable Diffusion 1.5 체크포인트
PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/country_body_small.txt"
OUTPUT_DIR="Continual/reno510/"  # 원하는 출력 디렉토리
NSAMPLES=1                                   # 이미지 개수
NUM_INFERENCE_STEPS=50                       # 디퓨전 스텝 수

########################
# ReNO 관련 인자
########################
CLIP_SCALE=5            # CLIPScore guidance 세기 (--clip_scale)
RENO_STEPS=10             # 각 디퓨전 스텝당 반복 ascent 횟수 (--reno_steps)
NEG_PROMPT="nudity"  # 억제할 개념 텍스트 (--negative_prompt)

########################
# 실행
########################
python reno_guidance.py \
    --ckpt_path         "$CKPT_PATH" \
    --prompt_file       "$PROMPT_FILE" \
    --output_dir        "$OUTPUT_DIR" \
    --nsamples          $NSAMPLES \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --clip_scale        $CLIP_SCALE \
    --reno_steps        $RENO_STEPS \
    --negative_prompt   "$NEG_PROMPT"
