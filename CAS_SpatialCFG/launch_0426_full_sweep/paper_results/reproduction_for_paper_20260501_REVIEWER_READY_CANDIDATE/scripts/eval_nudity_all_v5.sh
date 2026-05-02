#!/usr/bin/env bash
set -euo pipefail
REPRO_ROOT=${REPRO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
export REPRO_ROOT
export OUT_ROOT=${OUT_ROOT:-$REPRO_ROOT}
GPU=${GPU:-0}
BENCHES=(${BENCHES:-unlearndiff rab mma p4dn})
for b in "${BENCHES[@]}"; do
  echo "==== EVAL nudity benchmark=$b gpu=$GPU ===="
  python3 "$REPRO_ROOT/scripts/eval_from_config.py" \
    --gpu "$GPU" \
    --concept nudity \
    --config "$REPRO_ROOT/configs/ours_best/nudity/$b.json" \
    "$@"
done
