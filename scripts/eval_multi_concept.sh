#!/usr/bin/env bash
# =============================================================================
# Multi-Concept Qwen VLM Evaluation
# =============================================================================
# Evaluates all generated images across concepts and methods with Qwen VLM.
# Separate from generation so evaluation can be run independently.
#
# Usage:
#   bash scripts/eval_multi_concept.sh --gpus 0 --dry-run
#   bash scripts/eval_multi_concept.sh --concepts "violence harassment" --gpus 0,1
#   bash scripts/eval_multi_concept.sh --all --gpus 0
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VLM_DIR="${REPO_ROOT}/vlm"
OUTPUT_BASE="${REPO_ROOT}/CAS_SpatialCFG/outputs/multi_concept"
VLM_PYTHON="conda run -n vlm --no-capture-output python"

ALL_CONCEPTS="sexual violence harassment hate shocking illegal_activity self-harm"

# Map concept names to VLM eval names
get_eval_concept() {
    case "$1" in
        sexual)           echo "nudity" ;;
        illegal_activity) echo "illegal" ;;
        self-harm)        echo "self_harm" ;;
        *)                echo "$1" ;;
    esac
}

# Check if evaluation already exists
has_eval_result() {
    local method_dir="$1" eval_concept="$2"
    for fname in "categories_qwen3_vl_${eval_concept}.json" \
                 "categories_qwen_${eval_concept}.json"; do
        if [[ -f "${method_dir}/${fname}" ]]; then
            return 0
        fi
    done
    return 1
}

# ============================================================================
usage() {
    cat <<EOF
Multi-Concept Qwen VLM Evaluation

Usage:
  $0 [OPTIONS]

Options:
  --concepts "c1 c2 ..."   Concepts to evaluate (default: all)
  --gpus 0,1               Comma-separated GPU IDs
  --all                    Evaluate all concepts
  --dry-run                Print commands without executing
  --base-dir DIR           Override output base directory
  -h, --help               Show this help
EOF
    exit 0
}

DRY_RUN="false"
CONCEPTS=""
GPU_LIST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --concepts)  CONCEPTS="$2"; shift 2 ;;
        --gpus)      GPU_LIST="$2"; shift 2 ;;
        --all)       CONCEPTS="$ALL_CONCEPTS"; shift ;;
        --dry-run)   DRY_RUN="true"; shift ;;
        --base-dir)  OUTPUT_BASE="$2"; shift 2 ;;
        -h|--help)   usage ;;
        *)           echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$CONCEPTS" ]]; then
    CONCEPTS="$ALL_CONCEPTS"
fi

if [[ -z "$GPU_LIST" ]]; then
    echo "ERROR: --gpus is required."
    exit 1
fi

IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

echo "=============================================="
echo "Multi-Concept Qwen VLM Evaluation"
echo "=============================================="
echo "  Concepts: $CONCEPTS"
echo "  GPUs:     ${GPUS[*]}"
echo "  Base dir: $OUTPUT_BASE"
echo "  Dry-run:  $DRY_RUN"
echo "=============================================="

total_eval=0
skipped=0
gpu_idx=0

for concept in $CONCEPTS; do
    eval_concept=$(get_eval_concept "$concept")
    concept_dir="${OUTPUT_BASE}/${concept}"

    if [[ ! -d "$concept_dir" ]]; then
        echo "[SKIP] No output directory for ${concept}"
        continue
    fi

    echo ""
    echo "--- ${concept} (eval as: ${eval_concept}) ---"

    for method_dir in "${concept_dir}"/*/; do
        [[ -d "$method_dir" ]] || continue
        method_name=$(basename "$method_dir")

        # Skip if already evaluated
        if has_eval_result "$method_dir" "$eval_concept"; then
            echo "  [SKIP] ${method_name} (already evaluated)"
            skipped=$((skipped + 1))
            continue
        fi

        # Check for images
        img_count=$(find "$method_dir" -maxdepth 1 -name "*.png" | wc -l)
        if [[ "$img_count" -eq 0 ]]; then
            echo "  [SKIP] ${method_name} (no images)"
            continue
        fi

        local_gpu="${GPUS[$((gpu_idx % NUM_GPUS))]}"
        echo "  [EVAL] ${method_name} (${img_count} imgs) -> GPU ${local_gpu}"

        cmd="cd ${VLM_DIR} && CUDA_VISIBLE_DEVICES=${local_gpu} ${VLM_PYTHON} opensource_vlm_i2p_all.py \
            ${method_dir} ${eval_concept} qwen"

        if [[ "$DRY_RUN" == "true" ]]; then
            echo "    CMD: $cmd"
        else
            eval "$cmd" || echo "    [WARN] eval failed for ${concept}/${method_name}"
        fi

        total_eval=$((total_eval + 1))
        gpu_idx=$((gpu_idx + 1))
    done
done

echo ""
echo "=============================================="
echo "Evaluation complete: ${total_eval} evaluated, ${skipped} skipped"
echo "=============================================="
echo ""
echo "Run aggregation:"
echo "  python scripts/aggregate_multi_concept_results.py --base-dir ${OUTPUT_BASE}"
