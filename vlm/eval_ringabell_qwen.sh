#!/bin/bash
# ============================================================================
# Ring-A-Bell Qwen3-VL Evaluation
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,3,4,5 bash eval_ringabell_qwen.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ring-A-Bell directories to evaluate
DIRS=(
    "/mnt/home/yhgil99/unlearning/outputs/nudity_3class_skip/ringabell"
    "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_ringabell_20260128_201546"
    "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_dual_ringabell_20260128_201546"
    "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_dual_ringabell_20260128_223640"
    "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_ringabell_20260128_200910"
    "/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/grid_search_dual_ringabell_20260129_022351"
)

echo "============================================================"
echo "Ring-A-Bell Qwen3-VL Evaluation"
echo "============================================================"
echo ""
echo "Directories:"
for d in "${DIRS[@]}"; do
    echo "  - $(basename $d)"
done
echo ""

# Run evaluation
cd "${SCRIPT_DIR}"
bash batch_eval_qwen_multi_gpu.sh "${DIRS[@]}"
