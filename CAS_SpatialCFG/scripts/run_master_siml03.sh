#!/bin/bash
# ============================================================================
# Master Script for SIML-03: v14 → v15 → v16 (순차 chaining)
# GPU 0-7 (8개) 사용
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "============================================"
echo " SIML-03 Master: v14 → v15 → v16"
echo " Started: $(date)"
echo "============================================"

echo ""
echo ">>> [1/3] v14 Grid Search + Eval"
echo "    Started: $(date)"
bash scripts/run_v14.sh
echo "    Finished: $(date)"

echo ""
echo ">>> [2/3] v15 Grid Search + Eval"
echo "    Started: $(date)"
bash scripts/run_v15.sh
echo "    Finished: $(date)"

echo ""
echo ">>> [3/3] v16 Grid Search + Eval"
echo "    Started: $(date)"
bash scripts/run_v16.sh
echo "    Finished: $(date)"

echo ""
echo "============================================"
echo " SIML-03 Master COMPLETE"
echo " Finished: $(date)"
echo "============================================"
