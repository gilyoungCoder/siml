#!/usr/bin/env bash
set -euo pipefail

# =========================
# 런 설정
# =========================

# GPU
export CUDA_VISIBLE_DEVICES=5

PY="genTCMMM.py"    # ← 위 파이썬 파일 이름
SEED=1234

# 모델 & 입출력
CKPT_PATH="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="Continual/compare/CNB/addclothed40,0,0.0.1,harmtau0.1EOTblock_true"
PROMPT_FILE="./prompts/country_nude_body.txt"

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
FREEDOM_MODEL_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"

# =========================
# SAE Probe (옵션)
# =========================
USE_SAE=false
SAE_REPO="bcywinski/SAeUron"
SAE_HOOKPOINT="unet.up_blocks.1.attentions.1"
SAE_TOPK=32
SAE_CSV="${OUTPUT_DIR}/fai_log.csv"
SAE_CALIBRATE=true
SAE_NUDITY_PROMPTS="./configs/sd_prompt_Nudity.txt"
SAE_NEUTRAL_PROMPTS="./configs/sd_prompt_Human.txt"

# =========================
# 전역 억제 (Global Suppress: soft-only)
# =========================
USE_GLOBAL_SUPPRESS=true
HARM_GLOBAL_TXTS="./configs/harm_global.txt"  # 줄당 harmful 키워드 (예: nude, naked, nudity ...)
HARM_TAU=0.15
HARM_GAMMA_START=40
HARM_GAMMA_END=0.5

# 새 옵션(벡터/레이어)
HARM_LAYER_INDEX=-2
HARM_VEC_MODE="masked_mean"   # masked_mean | token | prompt_token
HARM_TARGET_WORDS="nude,naked,nudity"
INCLUDE_SPECIAL=false

# =========================
# ADD-LIST (긍정/안전 개념 강조)
# =========================
ADD_LIST_TXTS="./configs/addlist.txt"     # 예: fully clothed person / modest clothing / wearing a jacket ...
ADD_TAU=0.10
ADD_GAMMA_START=0
ADD_GAMMA_END=0.5
ADD_LAYER_INDEX=-2

# 문자열 보강 옵션
ADD_APPEND=true            # add_tau 매칭 기반 append
ADD_TOPK_APPEND=2

# harm 기반 강제 주입/append
ADD_ON_HARM=true
ADD_ON_HARM_TAU=0.20
ADD_APPEND_ON_HARM=true
ADD_FORCE_APPEND_N=1       # harm 트리거 시 문자열에 강제 추가할 add 항목 수

# =========================
# EOT 하드블록 토글
# =========================
EOT_HARD_BLOCK=true   # ← EOT(EOS)만 하드블록 (원치 않으면 false)

# =========================
# 준비
# =========================
mkdir -p "${OUTPUT_DIR}"
LOG="./Run_$(date +%Y%m%d_%H%M%S).log"

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

# 전역 억제 인자 (soft-only)
HARM_ARGS=()
if [ "${USE_GLOBAL_SUPPRESS}" = true ]; then
  HARM_ARGS+=("--harm_suppress")
  HARM_ARGS+=("--harm_global_texts" "${HARM_GLOBAL_TXTS}")
  HARM_ARGS+=("--harm_tau" "${HARM_TAU}")
  HARM_ARGS+=("--harm_gamma_start" "${HARM_GAMMA_START}")
  HARM_ARGS+=("--harm_gamma_end" "${HARM_GAMMA_END}")
  HARM_ARGS+=("--harm_layer_index" "${HARM_LAYER_INDEX}")
  HARM_ARGS+=("--harm_vec_mode" "${HARM_VEC_MODE}")
  if [ -n "${HARM_TARGET_WORDS}" ]; then
    HARM_ARGS+=("--harm_target_words" "${HARM_TARGET_WORDS}")
  fi
  if [ "${INCLUDE_SPECIAL}" = true ]; then
    HARM_ARGS+=("--include_special_tokens")
  fi
fi

# ADD 인자
ADD_ARGS=()
if [ -n "${ADD_LIST_TXTS}" ]; then
  ADD_ARGS+=("--add_list_texts" "${ADD_LIST_TXTS}")
  ADD_ARGS+=("--add_tau" "${ADD_TAU}")
  ADD_ARGS+=("--add_gamma_start" "${ADD_GAMMA_START}")
  ADD_ARGS+=("--add_gamma_end" "${ADD_GAMMA_END}")
  ADD_ARGS+=("--add_layer_index" "${ADD_LAYER_INDEX}")
  if [ "${ADD_APPEND}" = true ]; then
    ADD_ARGS+=("--add_append")
    ADD_ARGS+=("--add_topk_append" "${ADD_TOPK_APPEND}")
  fi
  if [ "${ADD_ON_HARM}" = true ]; then
    ADD_ARGS+=("--add_on_harm")
    ADD_ARGS+=("--add_on_harm_tau" "${ADD_ON_HARM_TAU}")
    if [ "${ADD_APPEND_ON_HARM}" = true ]; then
      ADD_ARGS+=("--add_append_on_harm")
      ADD_ARGS+=("--add_force_append_n" "${ADD_FORCE_APPEND_N}")
    fi
  fi
  # 디버깅 원하면 켜기
  ADD_ARGS+=("--add_debug_print")
fi

# EOT 하드블록 인자
EOT_ARGS=()
if [ "${EOT_HARD_BLOCK}" = true ]; then
  EOT_ARGS+=("--eot_hard_block")
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
  "${ADD_ARGS[@]}" \
  "${EOT_ARGS[@]}" \
  > "${LOG}" 2>&1 &

echo "[PID] $!"
echo "tail -f \"${LOG}\" 로 로그 확인하세요."
