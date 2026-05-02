#!/usr/bin/env bash
set -euo pipefail
REPRO_ROOT=${REPRO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT_ROOT=${OUT_ROOT:-$REPRO_ROOT}
GPU=${GPU:-0}
python "$REPRO_ROOT/scripts/run_from_config.py" --gpu "$GPU" --config "$REPRO_ROOT/configs/ours_best/i2p_q16/illegal_activity.json" "$@"
