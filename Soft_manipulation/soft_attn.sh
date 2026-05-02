#!/usr/bin/env bash
set -euo pipefail

# =========================
# 런 설정
# =========================

# GPU
export CUDA_VISIBLE_DEVICES=2

PY="soft_attn.py"   # ← 새 파이프라인(표준 StableDiffusionPipeline + GhostContextAttnProcessor)
SEED=1234

# 모델 & 입출력
CKPT_PATH="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="output_img/CNBfcpEot/ghost_attn_debug60.1"
PROMPT_FILE="./prompts/country_nude_body.txt"

# 생성 파라미터
NSAMPLES=1
CFG_SCALE=7.5
NUM_INFERENCE_STEPS=50
HEIGHT=512
WIDTH=512

# =========================
# Harmful 억제 + Anti-harm 주입
# =========================
ENABLE_HARM=true

# 개념 텍스트 파일(한 줄당 하나)
HARM_TXTS="./configs/harm_global.txt"       # 예: nude, naked, topless, ...
ANTI_TXTS="./configs/anti_clothed.txt"      # 예: fully clothed, modest apparel, ...

# 임계/스케줄
HARM_TAU=0.3
LAMBDA_START=6
LAMBDA_END=0.5
GHOST_BOOST_MU=0.1

# 임베딩 추출/레이어
LAYER_INDEX=-1
INCLUDE_SPECIAL=false
HARM_VEC_MODE="masked_mean"                 # masked_mean | token | prompt_token
HARM_TARGET_WORDS="nude,naked,topless"      # token/prompt_token 모드일 때만 의미

# =========================
# 디버깅 옵션 (NEW)
# =========================
DEBUG_COS=true                 # 프롬프트 시작 시 토큰별 cos(harm), cos(anti) 표 출력
DEBUG_ATTN=true                # 스텝별 교차-어텐션 재분배 통계 표 출력
DEBUG_INTERVAL=5               # N스텝마다 출력 (0,5,10,15, ...)
DEBUG_TOPK=8                   # 샘플0 기준 상위 harmful 컬럼 top-k 표시
DEBUG_PRECISION=4              # 소수점 자리수

# =========================
# 준비
# =========================
mkdir -p "${OUTPUT_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="Run_${STAMP}.log"

# Harm 인자 구성
HARM_ARGS=()
if [ "${ENABLE_HARM}" = true ]; then
  HARM_ARGS+=("--enable_harm")
  [ -n "${HARM_TXTS}" ] && HARM_ARGS+=("--harm_texts" "${HARM_TXTS}")
  [ -n "${ANTI_TXTS}" ] && HARM_ARGS+=("--anti_texts" "${ANTI_TXTS}")
  HARM_ARGS+=("--harm_tau" "${HARM_TAU}")
  HARM_ARGS+=("--lambda_start" "${LAMBDA_START}")
  HARM_ARGS+=("--lambda_end" "${LAMBDA_END}")
  HARM_ARGS+=("--ghost_boost_mu" "${GHOST_BOOST_MU}")
  HARM_ARGS+=("--layer_index" "${LAYER_INDEX}")
  HARM_ARGS+=("--harm_vec_mode" "${HARM_VEC_MODE}")
  if [ -n "${HARM_TARGET_WORDS}" ]; then
    HARM_ARGS+=("--harm_target_words" "${HARM_TARGET_WORDS}")
  fi
  if [ "${INCLUDE_SPECIAL}" = true ]; then
    HARM_ARGS+=("--include_special_tokens")
  fi
fi

# Debug 인자 구성
DBG_ARGS=()
if [ "${DEBUG_COS}" = true ]; then
  DBG_ARGS+=("--debug_cos")
fi
if [ "${DEBUG_ATTN}" = true ]; then
  DBG_ARGS+=("--debug_attn")
  DBG_ARGS+=("--debug_interval" "${DEBUG_INTERVAL}")
  DBG_ARGS+=("--debug_topk" "${DEBUG_TOPK}")
  DBG_ARGS+=("--debug_precision" "${DEBUG_PRECISION}")
fi

# 요약 에코
echo "================= RUN CONFIG (${STAMP}) =================" | tee "${LOG}"
echo "CKPT_PATH         : ${CKPT_PATH}"        | tee -a "${LOG}"
echo "PROMPT_FILE       : ${PROMPT_FILE}"      | tee -a "${LOG}"
echo "OUTPUT_DIR        : ${OUTPUT_DIR}"       | tee -a "${LOG}"
echo "NSAMPLES          : ${NSAMPLES}"         | tee -a "${LOG}"
echo "CFG_SCALE         : ${CFG_SCALE}"        | tee -a "${LOG}"
echo "NUM_STEPS         : ${NUM_INFERENCE_STEPS}" | tee -a "${LOG}"
echo "SIZE (HxW)        : ${HEIGHT}x${WIDTH}"  | tee -a "${LOG}"
echo "ENABLE_HARM       : ${ENABLE_HARM}"      | tee -a "${LOG}"
echo "HARM_TAU          : ${HARM_TAU}"         | tee -a "${LOG}"
echo "LAMBDA (start->end): ${LAMBDA_START} -> ${LAMBDA_END}" | tee -a "${LOG}"
echo "GHOST_BOOST_MU    : ${GHOST_BOOST_MU}"   | tee -a "${LOG}"
echo "LAYER_INDEX       : ${LAYER_INDEX}"      | tee -a "${LOG}"
echo "HARM_VEC_MODE     : ${HARM_VEC_MODE}"    | tee -a "${LOG}"
echo "DEBUG_COS         : ${DEBUG_COS}"        | tee -a "${LOG}"
echo "DEBUG_ATTN        : ${DEBUG_ATTN}"       | tee -a "${LOG}"
echo "DEBUG_INTERVAL    : ${DEBUG_INTERVAL}"   | tee -a "${LOG}"
echo "DEBUG_TOPK        : ${DEBUG_TOPK}"       | tee -a "${LOG}"
echo "DEBUG_PRECISION   : ${DEBUG_PRECISION}"  | tee -a "${LOG}"
echo "=========================================================" | tee -a "${LOG}"

# 실행
nohup python "${PY}" \
  "${CKPT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --prompt_file "${PROMPT_FILE}" \
  --nsamples "${NSAMPLES}" \
  --cfg_scale "${CFG_SCALE}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --anti_use_eot \
  --seed "${SEED}" \
  "${HARM_ARGS[@]}" \
  "${DBG_ARGS[@]}" \
  >> "${LOG}" 2>&1 &

PID=$!
echo "[PID] ${PID}"
echo "tail -f \"${LOG}\"  # 실시간 로그 확인"