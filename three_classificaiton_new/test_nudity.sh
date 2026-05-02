#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# 활성화할 가상 환경 설정 (필요시)
# source /path/to/your/venv/bin/activate
F_SCALE=3
G_Start=1
# 필수 인자들 설정
CKPT_PATH="runwayml/stable-diffusion-v1-5"   # 모델 체크포인트 경로
OUTPUT_DIR="Continual/timedependent/discriminator"+$F_SCALE                # 출력 디렉토리 경로
PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/country_body.txt"  # 프롬프트 파일 경로
# PROMPT_FILE=    "/mnt/home/yhgil99/guided2-safe-diffusion/prompts/coco_30.txt"  # 프롬프트 파일 경로
CYCLE=0                                # 사이클 번호 (선택 사항)

# 선택적 인자들 설정
NSAMPLES=2                              # 생성할 샘플 수
CFG_SCALE=5                             # classifier-free guidance scale
NUM_INFERENCE_STEPS=50                  # inference steps
TRAINED_TEXT_ENCODER="--trained_text_encoder"   # text encoder가 학습된 경우
FREEDOM="--freedom"                     # guidance model을 사용할 경우
FREEDOM_MODEL_CKPT="/mnt/home/yhgil99/guided3-safe-diffusion/geodiffusion-features-code_refactor/work_dirs/nudity_classifier_balanced_time_dependent/checkpoint/iter_2801/nudity_classifier.pth"  # 자유 모델 체크포인트
# FREEDOM_MODEL_CKPT="/mnt/home/yhgil99/guided3-safe-diffusion/geodiffusion-features-code_refactor/work_dirs/monet_classifier_chekpoint/checkpoint/iter_4801/nudity_classifier.pth"  # 자유 모델 체크포인트

# CONFIG_PATH="/mnt/home/yhgil99/guided3-safe-diffusion/geodiffusion-features-code_refactor/configs/models/nudity_classifier.yaml"
CONFIG_PATH="/mnt/home/yhgil99/guided3-safe-diffusion/geodiffusion-features-code_refactor/configs/models/time_dependent_discriminator.yaml"

# 실행할 명령어
python test_nudity.py \
    --ckpt_path $CKPT_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_file $PROMPT_FILE \
    --cycle $CYCLE \
    --nsamples $NSAMPLES \
    --cfg_scale $CFG_SCALE \
    $FREEDOM \
    --freedom_scale $F_SCALE \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --freedom_model_args_file $CONFIG_PATH \
    --freedom_model_ckpt $FREEDOM_MODEL_CKPT\
    --guide_start $G_Start   

