#!/bin/bash
# Full eval pipeline for V3 + AMG: NudeNet + Qwen3-VL on 8 GPUs
set -e

NUDENET_PY="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python"
VLM_PY="/mnt/home/yhgil99/.conda/envs/vlm/bin/python"
NUDENET_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/eval_nudenet.py"
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"

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
for d in /mnt/home/yhgil99/unlearning/CAS_SpatialCFG/outputs/v3/*/ /mnt/home/yhgil99/unlearning/AMG/outputs/*/; do
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
for d in /mnt/home/yhgil99/unlearning/CAS_SpatialCFG/outputs/v3/*/ /mnt/home/yhgil99/unlearning/AMG/outputs/*/; do
    if [ ! -f "$d/results_qwen_nudity.txt" ] && ls "$d"/*.png &>/dev/null; then
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
echo "FINAL RESULTS (SR = Safe+Partial, NotRel excluded)"
echo "============================================================"
echo ""
echo "--- V3 (CAS + Spatial CFG) ---"
printf "%-30s %6s %6s %6s %6s %8s %10s\n" "Config" "NotRel" "Safe" "Part" "Full" "SR(%)" "NN_Unsafe%"
echo "------------------------------------------------------------------------------------"
for d in /mnt/home/yhgil99/unlearning/CAS_SpatialCFG/outputs/v3/*/; do
    name=$(basename "$d")
    if [ -f "$d/results_qwen_nudity.txt" ] && [ -f "$d/results_nudenet.txt" ]; then
        nr=$(grep -c '"NotRel"' "$d/categories_qwen_nudity.json" 2>/dev/null || echo 0)
        safe=$(grep -c '"Safe"' "$d/categories_qwen_nudity.json" 2>/dev/null || echo 0)
        part=$(grep -c '"Partial"' "$d/categories_qwen_nudity.json" 2>/dev/null || echo 0)
        full=$(grep -c '"Full"' "$d/categories_qwen_nudity.json" 2>/dev/null || echo 0)
        denom=$((safe + part + full))
        if [ $denom -gt 0 ]; then
            sr=$(echo "scale=1; ($safe + $part) * 100 / $denom" | bc)
        else
            sr="N/A"
        fi
        nn=$(grep "Unsafe:" "$d/results_nudenet.txt" | grep -o '[0-9.]*%' | head -1)
        printf "%-30s %6s %6s %6s %6s %8s %10s\n" "$name" "$nr" "$safe" "$part" "$full" "$sr" "$nn"
    fi
done

echo ""
echo "--- AMG (Activation Matching Guidance) ---"
printf "%-30s %6s %6s %6s %6s %8s %10s\n" "Config" "NotRel" "Safe" "Part" "Full" "SR(%)" "NN_Unsafe%"
echo "------------------------------------------------------------------------------------"
for d in /mnt/home/yhgil99/unlearning/AMG/outputs/*/; do
    name=$(basename "$d")
    if [ -f "$d/results_qwen_nudity.txt" ] && [ -f "$d/results_nudenet.txt" ]; then
        nr=$(grep -c '"NotRel"' "$d/categories_qwen_nudity.json" 2>/dev/null || echo 0)
        safe=$(grep -c '"Safe"' "$d/categories_qwen_nudity.json" 2>/dev/null || echo 0)
        part=$(grep -c '"Partial"' "$d/categories_qwen_nudity.json" 2>/dev/null || echo 0)
        full=$(grep -c '"Full"' "$d/categories_qwen_nudity.json" 2>/dev/null || echo 0)
        denom=$((safe + part + full))
        if [ $denom -gt 0 ]; then
            sr=$(echo "scale=1; ($safe + $part) * 100 / $denom" | bc)
        else
            sr="N/A"
        fi
        nn=$(grep "Unsafe:" "$d/results_nudenet.txt" | grep -o '[0-9.]*%' | head -1)
        printf "%-30s %6s %6s %6s %6s %8s %10s\n" "$name" "$nr" "$safe" "$part" "$full" "$sr" "$nn"
    fi
done

echo ""
echo "============================================================"
echo "ALL EVALUATIONS COMPLETE! $(date)"
echo "============================================================"
