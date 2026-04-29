#!/bin/bash
# Run all 6 single-concept i2p best configs sequentially.
# Usage: bash run_all_single.sh [GPU=0]
set -uo pipefail
GPU=${1:-0}
DIR="$(cd "$(dirname "$0")" && pwd)"
for label in violence self-harm shocking illegal harassment hate; do
  bash "$DIR/run_${label}.sh" $GPU "./reproduce_out/${label}"
done
