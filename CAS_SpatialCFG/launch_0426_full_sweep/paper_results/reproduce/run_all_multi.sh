#!/bin/bash
# Run all 4 multi-concept best configs sequentially.
set -uo pipefail
GPU=${1:-0}
DIR="$(cd "$(dirname "$0")" && pwd)"
for label in 1c_sexual 2c_sexvio_v3 3c_sexvioshock 7c_all; do
  bash "$DIR/run_${label}.sh" $GPU "./reproduce_out/${label}"
done
