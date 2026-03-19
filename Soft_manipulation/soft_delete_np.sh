#!/usr/bin/env bash
set -euo pipefail

### =========================
### 기본 GPU 선택 (필요시 덮어쓰기)
### =========================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

### =========================
### 기본값 (파이썬과 동일)
### =========================
PY="soft_delete_np.py"
CKPT="CompVis/stable-diffusion-v1-4"
PROMPTS="./prompts/country_nude_body.txt"
HARMS="./configs/harm_global.txt"

OUTROOT="output_img/softdelete_new"
SEED=1234

# 생성 파라미터
NS=1
CFG=7.5
STEPS=50
H=512
W=512

# 네거티브(uncond) 주입 옵션
UNCOND_MODE="concat"      # concat | mean | sample
UNCOND_JOIN=", "
UNCOND_K=10
DEDUP=true
ALPHA=1.0                 # 0.0=빈 네거티브, 1.0=텍스트 기반 임베딩 (동일 문자열로 생성)

# 문자열 네거티브 프롬프트 (임베딩 비교도 동일 문자열 사용)
NEG_TEXT="nudity"

# Soft Delete
ENABLE_SD=false
TAU=0.12
GAMMA_START=0.0
GAMMA_END=0.0

# 텍스트 인코더 / 해밍 벡터 구성
LAYER_IDX=-1
INCLUDE_SPECIAL=false
HARM_VEC_MODE="masked_mean"    # masked_mean | token | prompt_token
HARM_TARGET_WORDS=""

# 로그/디버그
PRINT_STEPS=5
DEBUG=true

### =========================
### 사용법
### =========================
usage() {
  cat <<'USAGE'
Usage: ./run_softdelete_np.sh [options]

[필수 경로]
  --ckpt PATH               SD 체크포인트 (기본: CompVis/stable-diffusion-v1-4)
  --prompts PATH            프롬프트 파일
  --harms PATH              harm 텍스트 파일 (줄당 1개)

[출력/시드]
  --outroot DIR             출력 루트 디렉토리 (기본: output_img/softdelete_np_runs)
  --seed INT                랜덤 시드 (기본: 1234)

[생성]
  --ns INT                  num_images_per_prompt (기본: 1)
  --cfg FLOAT               guidance scale (기본: 7.5)
  --steps INT               스텝 수 (기본: 50)
  --H INT                   높이 (기본: 512)
  --W INT                   너비 (기본: 512)

[네거티브 주입/비교]
  --alpha FLOAT             0.0이면 문자열 경로 사용, >0이면 임베딩 경로 사용 (기본: 1.0)
  --neg_text STR            negative_prompt 문자열 (기본: "nudity")
  --uncond_mode MODE        concat|mean|sample (기본: concat)
  --uncond_join STR         concat 시 조인 문자열 (기본: ", ")
  --uncond_k INT            sample 모드에서 샘플링 k (기본: 10)
  --dedup BOOL              harm 텍스트 dedup (true/false, 기본: true)

[Soft Delete]
  --enable_sd BOOL          true/false (기본: false)
  --tau FLOAT               cosine threshold (기본: 0.12)
  --gamma_start FLOAT       시작 gamma (기본: 0.0)
  --gamma_end FLOAT         끝 gamma (기본: 0.0)

[텍스트 인코더]
  --layer_idx INT           hidden_states 레이어 인덱스 (기본: -1)
  --include_special BOOL    special 토큰 포함 (true/false, 기본: false)
  --harm_vec_mode MODE      masked_mean|token|prompt_token (기본: masked_mean)
  --harm_target_words STR   콤마 구분 타겟단어 (기본: "")

[로그]
  --print_steps INT         스텝별 출력 간격 (기본: 5)
  --debug BOOL              true/false (기본: true)

예)
  ./run_softdelete_np.sh --prompts ./prompts/a.txt --harms ./configs/harm.txt \
    --alpha 1.0 --neg_text nudity --enable_sd false

  ./run_softdelete_np.sh --alpha 0.0              # 문자열 negative_prompt 경로 비교
  ./run_softdelete_np.sh --alpha 1.0              # 임베딩 직접 주입 경로 비교
USAGE
}

### =========================
### 인자 파싱
### =========================
to_bool() {
  case "${1,,}" in
    true|1|yes|y) echo "true" ;;
    false|0|no|n) echo "false" ;;
    *) echo "false" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2;;
    --prompts) PROMPTS="$2"; shift 2;;
    --harms) HARMS="$2"; shift 2;;
    --outroot) OUTROOT="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --ns) NS="$2"; shift 2;;
    --cfg) CFG="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --H) H="$2"; shift 2;;
    --W) W="$2"; shift 2;;
    --alpha) ALPHA="$2"; shift 2;;
    --neg_text) NEG_TEXT="$2"; shift 2;;
    --uncond_mode) UNCOND_MODE="$2"; shift 2;;
    --uncond_join) UNCOND_JOIN="$2"; shift 2;;
    --uncond_k) UNCOND_K="$2"; shift 2;;
    --dedup) DEDUP="$(to_bool "$2")"; shift 2;;
    --enable_sd) ENABLE_SD="$(to_bool "$2")"; shift 2;;
    --tau) TAU="$2"; shift 2;;
    --gamma_start) GAMMA_START="$2"; shift 2;;
    --gamma_end) GAMMA_END="$2"; shift 2;;
    --layer_idx) LAYER_IDX="$2"; shift 2;;
    --include_special) INCLUDE_SPECIAL="$(to_bool "$2")"; shift 2;;
    --harm_vec_mode) HARM_VEC_MODE="$2"; shift 2;;
    --harm_target_words) HARM_TARGET_WORDS="$2"; shift 2;;
    --print_steps) PRINT_STEPS="$2"; shift 2;;
    --debug) DEBUG="$(to_bool "$2")"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[ERR] Unknown arg: $1"; usage; exit 1;;
  esac
done

### =========================
### 출력 경로/로그
### =========================
mkdir -p "${OUTROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"

# 설명에 들어가기 좋은 케이스명 자동 구성
MODE_STR="$(awk -v a="$ALPHA" 'BEGIN{ if (a+0>0) print "embed"; else print "string"; }')"
CASE_NAME="${MODE_STR}_a${ALPHA}_cfg${CFG}_s${STEPS}"
OUTDIR="${OUTROOT}/${CASE_NAME}"
mkdir -p "${OUTDIR}"
LOG="${OUTDIR}/Run_${STAMP}.log"

### =========================
### 헤더 출력
### =========================
{
  echo "=================================================================="
  echo "=                          RUN CONFIG                            ="
  echo "=================================================================="
  echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
  echo "CKPT                 : ${CKPT}"
  echo "PROMPTS              : ${PROMPTS}"
  echo "HARMS                : ${HARMS}"
  echo "OUTDIR               : ${OUTDIR}"
  echo "------------------------------------------------------------------"
  echo "NSAMPLES             : ${NS}"
  echo "CFG_SCALE            : ${CFG}"
  echo "STEPS                : ${STEPS}"
  echo "SIZE(HxW)            : ${H}x${W}"
  echo "SEED                 : ${SEED}"
  echo "------------------------------------------------------------------"
  echo "NEG_TEXT             : ${NEG_TEXT@Q}"
  echo "ALPHA                : ${ALPHA}"
  echo "UNCOND_MODE          : ${UNCOND_MODE}"
  echo "UNCOND_JOIN          : ${UNCOND_JOIN@Q}"
  echo "UNCOND_K             : ${UNCOND_K}"
  echo "DEDUP                : ${DEDUP}"
  echo "------------------------------------------------------------------"
  echo "ENABLE_SOFTDELETE    : ${ENABLE_SD}"
  echo "TAU                  : ${TAU}"
  echo "GAMMA (start->end)   : ${GAMMA_START} -> ${GAMMA_END}"
  echo "LAYER_INDEX          : ${LAYER_IDX}"
  echo "INCLUDE_SPECIAL      : ${INCLUDE_SPECIAL}"
  echo "HARM_VEC_MODE        : ${HARM_VEC_MODE}"
  echo "HARM_TARGET_WORDS    : ${HARM_TARGET_WORDS}"
  echo "------------------------------------------------------------------"
  echo "PRINT_STEPS          : ${PRINT_STEPS}"
  echo "DEBUG                : ${DEBUG}"
  echo "=================================================================="
} | tee "${LOG}"

### =========================
### 커맨드 구성 & 실행
### =========================
CMD=(python "${PY}"
  "${CKPT}"
  --prompt_file "${PROMPTS}"
  --harm_texts "${HARMS}"
  --output_dir "${OUTDIR}"
  --seed "${SEED}"
  --nsamples "${NS}"
  --cfg_scale "${CFG}"
  --num_inference_steps "${STEPS}"
  --height "${H}"
  --width "${W}"
  --uncond_mode "${UNCOND_MODE}"
  --uncond_join "${UNCOND_JOIN}"
  --uncond_k "${UNCOND_K}"
  --alpha "${ALPHA}"
  --neg_text "${NEG_TEXT}"
  --tau "${TAU}"
  --gamma_start "${GAMMA_START}"
  --gamma_end "${GAMMA_END}"
  --layer_index "${LAYER_IDX}"
  --harm_vec_mode "${HARM_VEC_MODE}"
  --print_steps "${PRINT_STEPS}"
)

# boolean 플래그
[[ "${DEDUP}" == "true" ]] && CMD+=(--dedup)
[[ "${ENABLE_SD}" == "true" ]] && CMD+=(--enable_soft_delete)
[[ "${INCLUDE_SPECIAL}" == "true" ]] && CMD+=(--include_special_tokens)
[[ -n "${HARM_TARGET_WORDS}" ]] && CMD+=(--harm_target_words "${HARM_TARGET_WORDS}")
[[ "${DEBUG}" == "true" ]] && CMD+=(--debug)

echo "== RUN => ${LOG}"
# nohup 실행
nohup "${CMD[@]}" >> "${LOG}" 2>&1 &

PID=$!
echo "[PID] ${PID}  (tail -f '${LOG}')" | tee -a "${LOG}"

# 편의용 심볼릭 링크
ln -sfn "${OUTDIR}" "${OUTROOT}/latest"
ln -sfn "${LOG}"    "${OUTROOT}/latest.log"

# tail 힌트
echo
echo "tail -f '${LOG}'"
