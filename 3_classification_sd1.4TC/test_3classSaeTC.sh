#!/usr/bin/env bash
set -euo pipefail

##############################################
# 기본 설정
##############################################
export CUDA_VISIBLE_DEVICES=5          # 사용할 GPU
PY="test_3classSaeTC.py"                       # 이전에 제공한 '전역 억제 버전' 파이썬 스크립트 파일명
SEED=1234                              # 재현성

# 모델 & 입출력 경로
CKPT_PATH="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="Continual/CountryNudeBody/token_control_scale_5_nog"
PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/country_naked_body.txt"

# 생성 파라미터
NSAMPLES=1
CFG_SCALE=5
NUM_INFERENCE_STEPS=50

##############################################
# Freedom(Classifier) Guidance 설정
##############################################
USE_FREEDOM=false               # 사용하지 않으려면 false
FREEDOM_SCALE=15               # guidance scale
GUIDE_START=1                  # guidance 시작 step index
FREEDOM_MODEL_ARGS="./configs/models/time_dependent_discriminator.yaml"
FREEDOM_MODEL_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"

##############################################
# SAE Probe 설정
##############################################
USE_SAE=true                   # 사용하지 않으려면 false
SAE_REPO="bcywinski/SAeUron"
SAE_HOOKPOINT="unet.up_blocks.1.attentions.1"
SAE_TOPK=32
SAE_CSV="${OUTPUT_DIR}/fai_log.csv"
SAE_CALIBRATE=true
# anchor 프롬프트 파일 (중복 지정 오류 수정: nudity/neutral 분리)
SAE_NUDITY_PROMPTS="/mnt/home/yhgil99/unlearning/SAeUron/UnlearnCanvas_resources/anchor_prompts/finetune_prompts/sd_prompt_Nudity.txt"
SAE_NEUTRAL_PROMPTS="/mnt/home/yhgil99/unlearning/SAeUron/UnlearnCanvas_resources/anchor_prompts/finetune_prompts/sd_prompt_Human.txt"

##############################################
# 전역 억제(Global suppression) 설정
##############################################
USE_GLOBAL_SUPPRESS=true       # 전역 억제 ON/OFF
HARM_GLOBAL_TXTS="./configs/harm_global.txt"  # 한 줄당 하나의 개념(예: nudity/sexual content 등)
HARM_TAU=0.50
HARM_GAMMA_START=5
HARM_GAMMA_END=2.5

##############################################
# 준비
##############################################
mkdir -p "${OUTPUT_DIR}"
LOG="./Test_$(date +%Y%m%d_%H%M%S).log"

# Freedom 인자 구성
FREEDOM_ARGS=()
if [ "${USE_FREEDOM}" = true ]; then
  FREEDOM_ARGS+=("--freedom")
  FREEDOM_ARGS+=("--freedom_scale" "${FREEDOM_SCALE}")
  FREEDOM_ARGS+=("--freedom_model_args_file" "${FREEDOM_MODEL_ARGS}")
  FREEDOM_ARGS+=("--freedom_model_ckpt" "${FREEDOM_MODEL_CKPT}")
  FREEDOM_ARGS+=("--guide_start" "${GUIDE_START}")
fi

# SAE 인자 구성
SAE_ARGS=()
if [ "${USE_SAE}" = true ]; then
  SAE_ARGS+=("--sae_probe")
  SAE_ARGS+=("--sae_repo" "${SAE_REPO}")
  SAE_ARGS+=("--sae_hookpoint" "${SAE_HOOKPOINT}")
  SAE_ARGS+=("--sae_topk" "${SAE_TOPK}")
  SAE_ARGS+=("--sae_csv" "${SAE_CSV}")
  if [ "${SAE_CALIBRATE}" = true ]; then
    SAE_ARGS+=("--sae_calibrate")
    # 두 파일 모두 존재할 때만 명시(둘 중 하나만 있으면 파이프라인 기본 동작 사용)
    SAE_ARGS+=("--sae_nudity_prompts" "${SAE_NUDITY_PROMPTS}")
    SAE_ARGS+=("--sae_neutral_prompts" "${SAE_NEUTRAL_PROMPTS}")
  fi
fi

# 전역 억제 인자 구성
HARM_ARGS=()
if [ "${USE_GLOBAL_SUPPRESS}" = true ]; then
  HARM_ARGS+=("--harm_suppress")
  HARM_ARGS+=("--harm_global_texts" "${HARM_GLOBAL_TXTS}")
  HARM_ARGS+=("--harm_tau" "${HARM_TAU}")
  HARM_ARGS+=("--harm_gamma_start" "${HARM_GAMMA_START}")
  HARM_ARGS+=("--harm_gamma_end" "${HARM_GAMMA_END}")
fi

##############################################
# 실행
##############################################
echo "[RUN] $(date) -> Log: ${LOG}"
nohup python "${PY}" \
  "${CKPT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --prompt_file "${PROMPT_FILE}" \
  --nsamples "${NSAMPLES}" \
  --cfg_scale "${CFG_SCALE}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --seed "${SEED}" \
  "${FREEDOM_ARGS[@]}" \
  "${SAE_ARGS[@]}" \
  "${HARM_ARGS[@]}" \
  > "${LOG}" 2>&1 &

echo "[PID] $!"
echo "실행을 백그라운드로 시작했습니다. tail -f \"${LOG}\" 로 로그를 확인하세요."
