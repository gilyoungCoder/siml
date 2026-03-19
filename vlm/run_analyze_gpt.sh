#!/bin/bash
# GPT-4o 평가 결과 분석
# Usage: ./vlm/run_analyze_gpt.sh

set -e

echo "============================================================"
echo "GPT-4o Evaluation Results Analysis"
echo "============================================================"

cd /mnt/home/yhgil99/unlearning

# Run analysis
python vlm/analyze_gpt_results.py --export vlm/gpt_results_summary.csv

echo ""
echo "✅ Analysis complete!"
echo "   CSV exported to: vlm/gpt_results_summary.csv"
