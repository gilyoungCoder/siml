#!/usr/bin/env bash
set -euo pipefail
IMG_DIR=${1:?Usage: bash RUN_ONE_EXAMPLE.sh /path/to/images concept [gpu]}
CONCEPT=${2:?Usage: bash RUN_ONE_EXAMPLE.sh /path/to/images concept [gpu]}
GPU=${3:-0}
export PY_VLM=${PY_VLM:-$(which python)}
export VLM_SCRIPT=${VLM_SCRIPT:-$(pwd)/vlm/opensource_vlm_i2p_all_v5.py}
GPU=$GPU python code/SafeGen/evaluation/eval_vlm.py "$IMG_DIR" "$CONCEPT" qwen --gpu "$GPU"
