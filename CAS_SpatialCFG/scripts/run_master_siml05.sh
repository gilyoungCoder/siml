#!/bin/bash
# ============================================================================
# Master Script for SIML-05: v17 → v18 → v19 (순차 chaining)
# GPU 0-7 (8개) 사용
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "============================================"
echo " SIML-05 Master: v17 → v18 → v19"
echo " Started: $(date)"
echo "============================================"

echo ""
echo ">>> [1/3] v17 Grid Search + Eval"
echo "    Started: $(date)"
bash scripts/run_v17.sh
echo "    Finished: $(date)"

echo ""
echo ">>> [2/3] v18 Grid Search + Eval"
echo "    Started: $(date)"
bash scripts/run_v18.sh
echo "    Finished: $(date)"

echo ""
echo ">>> [3/3] v19 Grid Search + Eval"
echo "    Started: $(date)"
bash scripts/run_v19.sh
echo "    Finished: $(date)"

echo ""
echo "============================================"
echo " SIML-05 Master COMPLETE"
echo " Finished: $(date)"
echo "============================================"
