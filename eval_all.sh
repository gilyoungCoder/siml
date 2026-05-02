#!/bin/bash
# Full eval pipeline for V3 + AMG: NudeNet + Qwen3-VL on 8 GPUs
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/repo_env.sh
source "${SCRIPT_DIR}/scripts/lib/repo_env.sh"

REPO_ROOT="${UNLEARNING_REPO_ROOT}"
NUDENET_PY="${UNLEARNING_SDD_COPY_PYTHON}"
VLM_PY="${UNLEARNING_VLM_PYTHON}"
NUDENET_SCRIPT="${REPO_ROOT}/vlm/eval_nudenet.py"
VLM_SCRIPT="${REPO_ROOT}/vlm/opensource_vlm_i2p_all.py"
V3_OUTPUT_ROOT="${REPO_ROOT}/CAS_SpatialCFG/outputs/v3"
AMG_OUTPUT_ROOT="${REPO_ROOT}/AMG/outputs"

run_parallel_8gpu() {
    local script="$1"
    local python="$2"
    local category="$3"
    shift 3
    local dirs=("$@")

    local idx=0
    while [ $idx -lt ${#dirs[@]} ]; do
        local pids=()
        for gpu in 0 1 2 3 4 5 6 7; do
            if [ $idx -lt ${#dirs[@]} ]; then
                local d="${dirs[$idx]}"
                local name=$(basename "$d")
                echo "  [GPU$gpu] $category $name"
                if [ "$category" = "NN" ]; then
                    CUDA_VISIBLE_DEVICES=$gpu $python $script "$d" > /dev/null 2>&1 &
                else
                    CUDA_VISIBLE_DEVICES=$gpu $python $script "$d" nudity qwen > /dev/null 2>&1 &
                fi
                pids+=($!)
                idx=$((idx + 1))
            fi
        done
        for pid in "${pids[@]}"; do
            wait $pid
        done
        echo "  Batch done! ($idx/${#dirs[@]})"
    done
}

echo "============================================================"
echo "EVAL PIPELINE START: $(date)"
echo "============================================================"

# Collect dirs needing NudeNet
NN_DIRS=()
for d in "${V3_OUTPUT_ROOT}"/*/ "${AMG_OUTPUT_ROOT}"/*/; do
    if [ ! -f "$d/results_nudenet.txt" ] && ls "$d"/*.png &>/dev/null; then
        NN_DIRS+=("$d")
    fi
done
echo ">>> NudeNet: ${#NN_DIRS[@]} dirs"
if [ ${#NN_DIRS[@]} -gt 0 ]; then
    run_parallel_8gpu "$NUDENET_SCRIPT" "$NUDENET_PY" "NN" "${NN_DIRS[@]}"
fi
echo "NudeNet DONE! $(date)"

# Collect dirs needing Qwen3-VL
VLM_DIRS=()
for d in "${V3_OUTPUT_ROOT}"/*/ "${AMG_OUTPUT_ROOT}"/*/; do
    if ! unlearning_find_qwen_result_txt "$d" >/dev/null 2>&1 && ls "$d"/*.png &>/dev/null; then
        VLM_DIRS+=("$d")
    fi
done
echo ">>> Qwen3-VL: ${#VLM_DIRS[@]} dirs"
if [ ${#VLM_DIRS[@]} -gt 0 ]; then
    run_parallel_8gpu "$VLM_SCRIPT" "$VLM_PY" "VLM" "${VLM_DIRS[@]}"
fi
echo "Qwen3-VL DONE! $(date)"

# Final results
echo ""
echo "============================================================"
echo "FINAL RESULTS (SR = (Safe+Partial) / Total)"
echo "============================================================"
echo ""
echo "--- V3 (CAS + Spatial CFG) ---"
printf "%-30s %6s %6s %6s %6s %8s %10s\n" "Config" "NotRel" "Safe" "Part" "Full" "SR(%)" "NN_Unsafe%"
echo "------------------------------------------------------------------------------------"
for d in "${V3_OUTPUT_ROOT}"/*/; do
    name=$(basename "$d")
    qwen_txt="$(unlearning_find_qwen_result_txt "$d" || true)"
    if [ -n "$qwen_txt" ] && [ -f "$d/results_nudenet.txt" ]; then
        nr="$(unlearning_qwen_count "$d" NotRel || echo 0)"
        safe="$(unlearning_qwen_count "$d" Safe || echo 0)"
        part="$(unlearning_qwen_count "$d" Partial || echo 0)"
        full="$(unlearning_qwen_count "$d" Full || echo 0)"
        sr="$(unlearning_qwen_percent_value "$d" SR || echo N/A)"
        nn="$(unlearning_nudenet_percent "$d" || echo -)"
        printf "%-30s %6s %6s %6s %6s %8s %10s\n" "$name" "$nr" "$safe" "$part" "$full" "$sr" "$nn"
    fi
done

echo ""
echo "--- AMG (Activation Matching Guidance) ---"
printf "%-30s %6s %6s %6s %6s %8s %10s\n" "Config" "NotRel" "Safe" "Part" "Full" "SR(%)" "NN_Unsafe%"
echo "------------------------------------------------------------------------------------"
for d in "${AMG_OUTPUT_ROOT}"/*/; do
    name=$(basename "$d")
    qwen_txt="$(unlearning_find_qwen_result_txt "$d" || true)"
    if [ -n "$qwen_txt" ] && [ -f "$d/results_nudenet.txt" ]; then
        nr="$(unlearning_qwen_count "$d" NotRel || echo 0)"
        safe="$(unlearning_qwen_count "$d" Safe || echo 0)"
        part="$(unlearning_qwen_count "$d" Partial || echo 0)"
        full="$(unlearning_qwen_count "$d" Full || echo 0)"
        sr="$(unlearning_qwen_percent_value "$d" SR || echo N/A)"
        nn="$(unlearning_nudenet_percent "$d" || echo -)"
        printf "%-30s %6s %6s %6s %6s %8s %10s\n" "$name" "$nr" "$safe" "$part" "$full" "$sr" "$nn"
    fi
done

echo ""
echo "============================================================"
echo "ALL EVALUATIONS COMPLETE! $(date)"
echo "============================================================"
