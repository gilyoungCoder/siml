#!/usr/bin/env bash
set -euo pipefail

# =========================
# 런 설정
# =========================

# GPU
export CUDA_VISIBLE_DEVICES=5

PY="genTCMM.py"            # ← 이 파이썬 파일은 SOT soft-exempt + (harm - allow) soft 가중치를 내부적으로 적용함
SEED=1234

# 모델 & 입출력
CKPT_PATH="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="Continual/new/CNNB3class/fullyclothed30allow"
# PROMPT_FILE="./prompts/sexual_50.txt"
PROMPT_FILE="./prompts/country_nude_body_clothed.txt"
# PROMPT_FILE="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/country_naked_body.txt"

# 생성 파라미터
NSAMPLES=1
CFG_SCALE=5
NUM_INFERENCE_STEPS=50

# =========================
# Freedom (옵션)
# =========================
USE_FREEDOM=false
FREEDOM_SCALE=5
FREEDOM_SCALE_EARLY=10
FREEDOM_EARLY_STEPS=5
GUIDE_START=1
FREEDOM_MODEL_ARGS="./configs/models/time_dependent_discriminator.yaml"
# FREEDOM_MODEL_ARGS="./configs/models/time_dependent_discriminator2class.yaml"
FREEDOM_MODEL_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
# FREEDOM_MODEL_CKPT="./work_dirs/nudity_two_classSD/checkpoint/step_8200/classifier.pth"

# =========================
# SAE Probe (옵션)
# =========================
USE_SAE=false
SAE_REPO="bcywinski/SAeUron"
SAE_HOOKPOINT="unet.up_blocks.1.attentions.1"
SAE_TOPK=32
SAE_CSV="${OUTPUT_DIR}/fai_log.csv"
SAE_CALIBRATE=true
SAE_NUDITY_PROMPTS="/mnt/home/yhgil99/unlearning/SAeUron/UnlearnCanvas_resources/anchor_prompts/finetune_prompts/sd_prompt_Nudity.txt"
SAE_NEUTRAL_PROMPTS="/mnt/home/yhgil99/unlearning/SAeUron/UnlearnCanvas_resources/anchor_prompts/finetune_prompts/sd_prompt_Human.txt"

# =========================
# 전역 억제 (Global Suppress)
# =========================
USE_GLOBAL_SUPPRESS=true
# HARM_GLOBAL_TXTS="./configs/null.txt"
HARM_GLOBAL_TXTS="./configs/harm_global.txt"

# 기본 골자: soft는 harm≥τ AND allow≤τ_allow일 때만 적용
# soft 가중치 = γ * max(0, harm - allow)   ← (코드 내부 적용)
# SOT(BOS)은 soft delete에서 항상 보존(예외 처리) ← (코드 내부 적용)
# hard delete는 기존 규칙 유지(dual/hard_block 경로 동일)
HARM_TAU=0.20                 # soft threshold (harm 코사인)
HARM_HARD_TAU=0.20            # hard threshold (>= soft)
HARM_GAMMA_START=30           # early γ
HARM_GAMMA_END=0.5            # late  γ
HARM_ALLOW_TXTS="./configs/allowlist.txt"
HARM_ALLOW_TAU=0.35           # allow 면제 기준 (allow_max ≥ 0.35면 safe)
HARM_DUAL=false               # soft + hard 동시 적용
HARM_HARD_BLOCK=false         # dual=false일 때만 의미
HARM_DEBUG_PRINT=true
HARM_DEBUG_PROMPT=""          # 빈 문자열이면 전체 프롬프트 디버그

########### 새로 추가된 억제 옵션(이미 코드 지원) ###########
HARM_LAYER_INDEX=-2           # -1=last, -2/-3도 시험 가능
HARM_VEC_MODE="masked_mean"   # masked_mean | token | prompt_token
HARM_TARGET_WORDS="nude"      # token/prompt_token 모드에서 사용 (콤마구분)
INCLUDE_SPECIAL=false         # 평균 산출 시 SOT/EOT/PAD 제외(권장)

# =========================
# 준비
# =========================
mkdir -p "${OUTPUT_DIR}"
LOG="./Test_$(date +%Y%m%d_%H%M%S).log"

# Freedom 인자
FREEDOM_ARGS=()
if [ "${USE_FREEDOM}" = true ]; then
  FREEDOM_ARGS+=("--freedom")
  FREEDOM_ARGS+=("--freedom_scale" "${FREEDOM_SCALE}")
  FREEDOM_ARGS+=("--freedom_scale_early" "${FREEDOM_SCALE_EARLY}")
  FREEDOM_ARGS+=("--freedom_early_steps" "${FREEDOM_EARLY_STEPS}")
  FREEDOM_ARGS+=("--freedom_model_args_file" "${FREEDOM_MODEL_ARGS}")
  FREEDOM_ARGS+=("--freedom_model_ckpt" "${FREEDOM_MODEL_CKPT}")
  FREEDOM_ARGS+=("--guide_start" "${GUIDE_START}")
fi

# SAE 인자
SAE_ARGS=()
if [ "${USE_SAE}" = true ]; then
  SAE_ARGS+=("--sae_probe")
  SAE_ARGS+=("--sae_repo" "${SAE_REPO}")
  SAE_ARGS+=("--sae_hookpoint" "${SAE_HOOKPOINT}")
  SAE_ARGS+=("--sae_topk" "${SAE_TOPK}")
  SAE_ARGS+=("--sae_csv" "${SAE_CSV}")
  if [ "${SAE_CALIBRATE}" = true ]; then
    SAE_ARGS+=("--sae_calibrate")
    SAE_ARGS+=("--sae_nudity_prompts" "${SAE_NUDITY_PROMPTS}")
    SAE_ARGS+=("--sae_neutral_prompts" "${SAE_NEUTRAL_PROMPTS}")
  fi
fi

# 전역 억제 인자
HARM_ARGS=()
if [ "${USE_GLOBAL_SUPPRESS}" = true ]; then
  HARM_ARGS+=("--harm_suppress")
  HARM_ARGS+=("--harm_global_texts" "${HARM_GLOBAL_TXTS}")
  HARM_ARGS+=("--harm_tau" "${HARM_TAU}")
  HARM_ARGS+=("--harm_gamma_start" "${HARM_GAMMA_START}")
  HARM_ARGS+=("--harm_gamma_end" "${HARM_GAMMA_END}")
  HARM_ARGS+=("--harm_allowlist_texts" "${HARM_ALLOW_TXTS}")
  HARM_ARGS+=("--harm_allow_tau" "${HARM_ALLOW_TAU}")
  # 새 인자들(벡터 산출/레이어 선택)
  HARM_ARGS+=("--harm_layer_index" "${HARM_LAYER_INDEX}")
  HARM_ARGS+=("--harm_vec_mode" "${HARM_VEC_MODE}")
  if [ -n "${HARM_TARGET_WORDS}" ]; then
    HARM_ARGS+=("--harm_target_words" "${HARM_TARGET_WORDS}")
  fi
  if [ "${INCLUDE_SPECIAL}" = true ]; then
    HARM_ARGS+=("--include_special_tokens")
  fi

  if [ "${HARM_DUAL}" = true ]; then
    HARM_ARGS+=("--harm_dual")
    HARM_ARGS+=("--harm_hard_tau" "${HARM_HARD_TAU}")
  else
    if [ "${HARM_HARD_BLOCK}" = true ]; then
      HARM_ARGS+=("--harm_hard_block")
    fi
  fi
  if [ "${HARM_DEBUG_PRINT}" = true ]; then
    HARM_ARGS+=("--harm_debug_print")
    HARM_ARGS+=("--harm_debug_prompt" "${HARM_DEBUG_PROMPT}")
  fi
fi

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
echo "tail -f \"${LOG}\" 로 로그 확인하세요."
