# run_sd14_softdelete_prob_nohup.sh
#!/usr/bin/env bash
set -euo pipefail

# ===== GPU =====
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ===== Script & IO =====
PY="gen_delete.py"

MODEL="CompVis/stable-diffusion-v1-4"
PROMPTS="./prompts/country_nude_body.txt"
OUTDIR="./outputs/sd14_prob5  "
NEG="nudity"                 # "" 주면 negative prompt 끔
DTYPE="auto"                 # auto | fp16 | fp32

# ===== Gen params =====
STEPS=50
SCALE=7.5
H=512
W=512
SEED=1234
MAX_PROMPTS=0               # 0이면 전부

# ===== Soft Delete (probability redistribute) =====
ENABLE_SD=true
HARMS="./configs/harm_global.txt"
TAU=0.12
GAMMA_START=10
GAMMA_END=0.20
LAYER_IDX=-1
INCLUDE_SPECIAL=false

# ===== Debug =====
PRINT_STEPS=5
DEBUG=true

# ===== Prepare =====
mkdir -p "${OUTDIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${OUTDIR}/Run_${STAMP}.log"

echo "== RUN (sd14_from_file_softdelete_prob.py) => ${LOG}"
{
  echo "=================================================================="
  echo "=        sd14_from_file_softdelete_prob.py (nohup launch)        ="
  echo "=================================================================="
  echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
  echo "MODEL               : ${MODEL}"
  echo "PROMPTS             : ${PROMPTS}"
  echo "OUTDIR              : ${OUTDIR}"
  echo "NEGATIVE_PROMPT     : ${NEG@Q}"
  echo "DTYPE               : ${DTYPE}"
  echo "STEPS / SCALE       : ${STEPS} / ${SCALE}"
  echo "SIZE (HxW)          : ${H}x${W}"
  echo "SEED                : ${SEED}"
  echo "MAX_PROMPTS         : ${MAX_PROMPTS}"
  echo "------------------------------------------------------------------"
  echo "SOFT_DELETE(prob)   : ${ENABLE_SD}"
  echo "HARMS               : ${HARMS}"
  echo "TAU                 : ${TAU}"
  echo "GAMMA start->end    : ${GAMMA_START} -> ${GAMMA_END}"
  echo "LAYER_INDEX         : ${LAYER_IDX}"
  echo "INCLUDE_SPECIAL     : ${INCLUDE_SPECIAL}"
  echo "PRINT_STEPS         : ${PRINT_STEPS}"
  echo "DEBUG               : ${DEBUG}"
  echo "STAMP               : ${STAMP}"
  echo "=================================================================="
} | tee "${LOG}"

# ===== Build args =====
ARGS=(
  --model "${MODEL}"
  --prompt_file "${PROMPTS}"
  --output_dir "${OUTDIR}"
  --negative_prompt "${NEG}"
  --num_inference_steps "${STEPS}"
  --guidance_scale "${SCALE}"
  --height "${H}"
  --width "${W}"
  --seed "${SEED}"
  --dtype "${DTYPE}"
  --max_prompts "${MAX_PROMPTS}"
)

if [ "${ENABLE_SD}" = true ]; then
  ARGS+=( --enable_soft_delete --harm_texts "${HARMS}" --tau "${TAU}" \
          --gamma_start "${GAMMA_START}" --gamma_end "${GAMMA_END}" \
          --layer_index "${LAYER_IDX}" )
  [ "${INCLUDE_SPECIAL}" = true ] && ARGS+=( --include_special_tokens )
fi

[ "${PRINT_STEPS}" -gt 0 ] && ARGS+=( --print_steps "${PRINT_STEPS}" )
[ "${DEBUG}" = true ] && ARGS+=( --debug )

# ===== Run =====
nohup python "${PY}" "${ARGS[@]}" >> "${LOG}" 2>&1 &

PID=$!
echo "[PID] ${PID}" | tee -a "${LOG}"

# 편의용 심볼릭 링크
ln -sfn "${LOG}" "${OUTDIR}/latest.log"
ln -sfn "${OUTDIR}" "$(dirname "${OUTDIR}")/latest_sd14_prob"

echo "tail -f '${OUTDIR}/latest.log'"
