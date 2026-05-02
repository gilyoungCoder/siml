#!/usr/bin/env bash
set -euo pipefail
REPRO_ROOT=${REPRO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${PY_VLM:?Set PY_VLM=/path/to/vlm/python}
VLM=${VLM_SCRIPT:?Set VLM_SCRIPT=/path/to/opensource_vlm_i2p_all_v5.py}
GPU=${GPU:-0}
run_eval(){ local dir=$1 concept=$2; [ -d "$dir" ] || { echo "[MISS] $dir"; return 0; }; CUDA_VISIBLE_DEVICES=$GPU "$PY" "$VLM" "$dir" "$concept" qwen; }
outdir(){ python - "$1" <<'PYOUT'
import json, os, sys
j=json.load(open(sys.argv[1])); print(os.path.expandvars(j["outdir"]))
PYOUT
}
export OUT_ROOT=${OUT_ROOT:-$REPRO_ROOT}
run_eval "$(outdir "$REPRO_ROOT/configs/ours_best/i2p_q16/sexual.json")" nudity
run_eval "$(outdir "$REPRO_ROOT/configs/ours_best/i2p_q16/violence.json")" violence
run_eval "$(outdir "$REPRO_ROOT/configs/ours_best/i2p_q16/self-harm.json")" self_harm
run_eval "$(outdir "$REPRO_ROOT/configs/ours_best/i2p_q16/shocking.json")" shocking
run_eval "$(outdir "$REPRO_ROOT/configs/ours_best/i2p_q16/illegal_activity.json")" illegal
run_eval "$(outdir "$REPRO_ROOT/configs/ours_best/i2p_q16/harassment.json")" harassment
run_eval "$(outdir "$REPRO_ROOT/configs/ours_best/i2p_q16/hate.json")" hate
