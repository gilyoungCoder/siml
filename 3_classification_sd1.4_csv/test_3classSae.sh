#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
# 활성화할 가상 환경 설정 (필요시)
# source /path/to/your/venv/bin/activate
F_SCALE=15
G_Start=1
# 필수 인자들 설정
CKPT_PATH="CompVis/stable-diffusion-v1-4"   # 모델 체크포인트 경로
OUTPUT_DIR="Continual/CountryNudeBody/cg_0_single_7.5initial"               # 출력 디렉토리 경로
PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/country_naked_body.txt"  # 프롬프트 파일 경로
# PROMPT_FILE="./prompts/null.txt"  # 프롬프트 파일 경로
# PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/coco_30.txt"  # 프롬프트 파일 경로
CYCLE=0                                # 사이클 번호 (선택 사항)

# 선택적 인자들 설정
NSAMPLES=1                              # 생성할 샘플 수
CFG_SCALE=5                             # classifier-free guidance scale
NUM_INFERENCE_STEPS=50                  # inference steps
FREEDOM="--freedom"                     # guidance model을 사용할 경우
FREEDOM_MODEL_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"  # 자유 모델 체크포인트

# FREEDOM_MODEL_CKPT="/mnt/home/yhgil99/unlearning/three_classificaiton_new/work_dirs/nudity_three_class_syn/checkpoint/step_3200/classifier.pth"  # 자유 모델 체크포인트
# FREEDOM_MODEL_CKPT="/mnt/home/yhgil99/guided3-safe-diffusion/geodiffusion-features-code_refactor/work_dirs/monet_classifier_chekpoint/checkpoint/iter_4801/nudity_classifier.pth"  # 자유 모델 체크포인트

# CONFIG_PATH="/mnt/home/yhgil99/guided3-safe-diffusion/geodiffusion-features-code_refactor/configs/models/nudity_classifier.yaml"
CONFIG_PATH="./configs/models/time_dependent_discriminator.yaml"

# 실행할 명령어
nohup python test_3classSae.py \
    $CKPT_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_file $PROMPT_FILE \
    --cycle $CYCLE \
    --nsamples $NSAMPLES \
    --cfg_scale $CFG_SCALE \
    $FREEDOM \
    --freedom_scale $F_SCALE \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --freedom_model_args_file $CONFIG_PATH \
    --freedom_model_ckpt $FREEDOM_MODEL_CKPT \
    --guide_start $G_Start\
    --sae_probe \
    --sae_csv "$OUTPUT_DIR/fai_log.csv" \
    --sae_calibrate \
    --sae_neutral_prompts ~/unlearning/SAeUron/UnlearnCanvas_resources/anchor_prompts/finetune_prompts/sd_prompt_Human.txt\
    --sae_nudity_prompts ~/unlearning/SAeUron/UnlearnCanvas_resources/anchor_prompts/finetune_prompts/sd_prompt_Nudity.txt\
    > Test2.log 2>&1 & 
