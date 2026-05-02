#!/usr/bin/env bash
set -euo pipefail

# =========================
# GPU
# =========================
export CUDA_VISIBLE_DEVICES=6
# =========================
# Python / Seed
# =========================
PY="generateTC.py"
SEED=1234

# =========================
# Model / IO
# =========================
CKPT_PATH="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="Continual/new6/CNB/addclothed__40,7.5_tau0.2_rel-alpha0.2_EOTblock_true"
PROMPT_FILE="./prompts/country_nude_body.txt"

# =========================
# Generation
# =========================
NSAMPLES=1
CFG_SCALE=5
NUM_INFERENCE_STEPS=50

# =========================
# Harm suppression
# =========================
USE_HARM=true
HARM_GLOBAL_TXTS="./configs/harm_global.txt"
HARM_LAYER_INDEX=-2
HARM_TAU=0.20
HARM_REL_PARAM=0.20      # alpha if RELATIVE_MODE=alpha, tau_rel if RELATIVE_MODE=ratio
HARM_GAMMA_START=40
HARM_GAMMA_END=0.5

# =========================
# ADD-LIST
# =========================
ADD_LIST_TXTS="./configs/addlist.txt"
ADD_LAYER_INDEX=-2
ADD_TAU=0.20
ADD_REL_PARAM=0.20
ADD_GAMMA_START=7.5
ADD_GAMMA_END=0.5

# =========================
# Append (suppress ≥ 1 일 때만)
# =========================
ADD_APPEND=true
ADD_FORCE_APPEND_N=1

# =========================
# Relative thresholds mode: alpha | ratio | off
# =========================
RELATIVE_MODE="alpha"

# =========================
# EOT hard block
# =========================
EOT_HARD_BLOCK=true

# =========================
# Debug
# =========================
DEBUG=true

# =========================
# Output & Log
# =========================
mkdir -p "${OUTPUT_DIR}"
LOG="./Run_$(date +%Y%m%d_%H%M%S).log"

# =========================
# Build args
# =========================
HARM_ARGS=()
if [ "${USE_HARM}" = true ]; then
  HARM_ARGS+=("--harm_suppress")
  HARM_ARGS+=("--harm_global_texts" "${HARM_GLOBAL_TXTS}")
  HARM_ARGS+=("--harm_layer_index" "${HARM_LAYER_INDEX}")
  HARM_ARGS+=("--harm_tau" "${HARM_TAU}")
  case "${RELATIVE_MODE}" in
    alpha) HARM_ARGS+=("--harm_alpha" "${HARM_REL_PARAM}") ;;
    ratio) HARM_ARGS+=("--harm_tau_rel" "${HARM_REL_PARAM}") ;;
    off)   : ;;
    *) echo "RELATIVE_MODE must be one of: alpha | ratio | off" >&2; exit 1 ;;
  esac
  HARM_ARGS+=("--harm_gamma_start" "${HARM_GAMMA_START}")
  HARM_ARGS+=("--harm_gamma_end" "${HARM_GAMMA_END}")
fi

ADD_ARGS=()
if [ -n "${ADD_LIST_TXTS}" ]; then
  ADD_ARGS+=("--add_list_texts" "${ADD_LIST_TXTS}")
  ADD_ARGS+=("--add_layer_index" "${ADD_LAYER_INDEX}")
  ADD_ARGS+=("--add_tau" "${ADD_TAU}")
  case "${RELATIVE_MODE}" in
    alpha) ADD_ARGS+=("--add_alpha" "${ADD_REL_PARAM}") ;;
    ratio) ADD_ARGS+=("--add_tau_rel" "${ADD_REL_PARAM}") ;;
    off)   : ;;
  esac
  ADD_ARGS+=("--add_gamma_start" "${ADD_GAMMA_START}")
  ADD_ARGS+=("--add_gamma_end" "${ADD_GAMMA_END}")
fi

APPEND_ARGS=()
if [ "${ADD_APPEND}" = true ]; then
  APPEND_ARGS+=("--add_append")
  APPEND_ARGS+=("--add_force_append_n" "${ADD_FORCE_APPEND_N}")
fi

EOT_ARGS=()
if [ "${EOT_HARD_BLOCK}" = true ]; then
  EOT_ARGS+=("--eot_hard_block")
fi

DBG_ARGS=()
if [ "${DEBUG}" = true ]; then
  DBG_ARGS+=("--add_debug_print")
fi

# =========================
# Run
# =========================
echo "[RUN] $(date) -> Log: ${LOG}"
nohup python "${PY}" \
  "${CKPT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --prompt_file "${PROMPT_FILE}" \
  --nsamples "${NSAMPLES}" \
  --cfg_scale "${CFG_SCALE}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --seed "${SEED}" \
  "${HARM_ARGS[@]}" \
  "${ADD_ARGS[@]}" \
  "${APPEND_ARGS[@]}" \
  "${EOT_ARGS[@]}" \
  "${DBG_ARGS[@]}" \
  > "${LOG}" 2>&1 &

echo "[PID] $!"
echo "tail -f \"${LOG}\" 로 로그 확인하세요."
