#!/usr/bin/env bash
set -euo pipefail
# Backward-compatible entrypoint: evaluate all final I2P q16 top-60 outputs.
REPRO_ROOT=${REPRO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
exec bash "$REPRO_ROOT/scripts/eval_i2p_all_v5.sh" "$@"
