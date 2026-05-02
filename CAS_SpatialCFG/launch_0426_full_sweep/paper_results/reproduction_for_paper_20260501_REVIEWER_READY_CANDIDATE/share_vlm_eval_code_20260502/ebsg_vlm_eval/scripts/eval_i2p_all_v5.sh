#!/usr/bin/env bash
set -euo pipefail
REPRO_ROOT=${REPRO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
export REPRO_ROOT
export OUT_ROOT=${OUT_ROOT:-$REPRO_ROOT}
GPU=${GPU:-0}
CONCEPTS=(${CONCEPTS:-sexual violence self-harm shocking illegal_activity harassment hate})
for c in "${CONCEPTS[@]}"; do
  echo "==== EVAL I2P concept=$c gpu=$GPU ===="
  python3 "$REPRO_ROOT/scripts/eval_from_config.py" \
    --gpu "$GPU" \
    --config "$REPRO_ROOT/configs/ours_best/i2p_q16/$c.json" \
    "$@"
done
