#!/usr/bin/env bash
set -euo pipefail
export REPRO_ROOT=${REPRO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
export OUT_ROOT=${OUT_ROOT:-$REPRO_ROOT}
GPU=${GPU:-0}
for c in sexual violence self-harm shocking illegal_activity harassment hate; do
  python "$REPRO_ROOT/scripts/run_from_config.py" --gpu "$GPU" --config "$REPRO_ROOT/configs/ours_best/i2p_q16/$c.json" "$@"
done
