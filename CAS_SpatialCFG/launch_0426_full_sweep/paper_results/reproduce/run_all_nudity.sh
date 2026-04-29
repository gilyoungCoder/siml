#!/bin/bash
# Run nudity benchmark hybrid reproduction scripts sequentially.
# Usage: bash run_all_nudity.sh [GPU=0]
set -uo pipefail
GPU=${1:-0}
DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$DIR/run_nudity_ud.sh" "$GPU" "./reproduce_out/nudity_ud"
bash "$DIR/run_nudity_rab.sh" "$GPU" "./reproduce_out/nudity_rab"
bash "$DIR/run_nudity_p4dn.sh" "$GPU" "./reproduce_out/nudity_p4dn"
# MMA is longer; included for completeness. Split manually with run_nudity_mma.sh START END if desired.
bash "$DIR/run_nudity_mma.sh" "$GPU" "./reproduce_out/nudity_mma"
