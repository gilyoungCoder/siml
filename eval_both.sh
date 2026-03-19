#!/bin/bash
# =============================================================================
# Watch V3 + AMG generation → run eval for BOTH when done
# =============================================================================
set -euo pipefail

LOG="/mnt/home/yhgil99/unlearning/eval_both.log"
exec > >(tee -a "$LOG") 2>&1

PYTHON_NN="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python"
PYTHON_VLM="/mnt/home/yhgil99/.conda/envs/vlm/bin/python"
NN_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/eval_nudenet.py"
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
V3OUT="/mnt/home/yhgil99/unlearning/CAS_SpatialCFG/outputs/v3"
AMGOUT="/mnt/home/yhgil99/unlearning/AMG/outputs"

get_free_gpus() {
    local free=()
    while IFS=, read -r idx used total; do
        idx=$(echo "$idx" | xargs)
        used=$(echo "$used" | xargs | sed 's/ MiB//')
        if [ "$used" -lt 1000 ]; then free+=("$idx"); fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader)
    echo "${free[@]}"
}

run_eval() {
    local OUTBASE="$1"
    local LABEL="$2"

    echo ""
    echo "============================================================"
    echo "[$(date)] $LABEL: NudeNet Evaluation"
    echo "============================================================"

    local FREE_GPUS=($(get_free_gpus))
    [ ${#FREE_GPUS[@]} -eq 0 ] && { echo "No free GPUs, waiting 60s..."; sleep 60; FREE_GPUS=($(get_free_gpus)); }

    for dir in "$OUTBASE"/*/; do
        name=$(basename "$dir")
        [ -f "$dir/results_nudenet.txt" ] && { echo "  [SKIP] $name"; continue; }
        n_imgs=$(ls "$dir"/*.png 2>/dev/null | wc -l)
        [ "$n_imgs" -eq 0 ] && continue
        echo "  [EVAL] $name ($n_imgs images)"
        CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} $PYTHON_NN "$NN_SCRIPT" "$dir" 2>/dev/null || true
    done

    echo ""
    echo "============================================================"
    echo "[$(date)] $LABEL: Qwen3-VL Evaluation"
    echo "============================================================"

    FREE_GPUS=($(get_free_gpus))
    local NUM_GPUS=${#FREE_GPUS[@]}

    local EVAL_DIRS=()
    for dir in "$OUTBASE"/*/; do
        [ -f "$dir/results_qwen_nudity.txt" ] && continue
        n_imgs=$(ls "$dir"/*.png 2>/dev/null | wc -l)
        [ "$n_imgs" -eq 0 ] && continue
        EVAL_DIRS+=("$dir")
    done

    local eval_idx=0
    while [ $eval_idx -lt ${#EVAL_DIRS[@]} ]; do
        local pids=() names=() gpu_idx=0
        while [ $eval_idx -lt ${#EVAL_DIRS[@]} ] && [ $gpu_idx -lt $NUM_GPUS ]; do
            dir="${EVAL_DIRS[$eval_idx]}"
            name=$(basename "$dir")
            GPU=${FREE_GPUS[$gpu_idx]}
            echo "  [GPU $GPU] VLM: $name"
            CUDA_VISIBLE_DEVICES=$GPU $PYTHON_VLM "$VLM_SCRIPT" "$dir" nudity qwen \
                > "$OUTBASE/${name}_vlm.log" 2>&1 &
            pids+=($!); names+=("$name"); gpu_idx=$((gpu_idx+1)); eval_idx=$((eval_idx+1))
        done
        for i in "${!pids[@]}"; do wait ${pids[$i]} || echo "  WARN: ${names[$i]}"; done
    done

    # Print results
    echo ""
    echo "============================================================"
    echo "$LABEL RESULTS (SR = Safe + Partial, NotRel excluded)"
    echo "============================================================"
    printf "%-30s %6s %6s %6s %6s %8s %10s\n" "Config" "NotRel" "Safe" "Part" "Full" "SR(%)" "NN_Unsafe%"
    echo "--------------------------------------------------------------------------------------------"
    for dir in "$OUTBASE"/*/; do
        name=$(basename "$dir")
        nn_pct="-"
        [ -f "$dir/results_nudenet.txt" ] && nn_pct=$(grep -oP '\d+\.\d+%' "$dir/results_nudenet.txt" | head -1)
        [ -z "$nn_pct" ] && nn_pct="-"
        nr="-"; safe="-"; part="-"; full="-"; sr="-"
        if [ -f "$dir/categories_qwen_nudity.json" ]; then
            nr=$(grep -c "NotRel" "$dir/categories_qwen_nudity.json" 2>/dev/null || echo 0)
            safe=$(grep -c '"Safe"' "$dir/categories_qwen_nudity.json" 2>/dev/null || echo 0)
            part=$(grep -c "Partial" "$dir/categories_qwen_nudity.json" 2>/dev/null || echo 0)
            full=$(grep -c '"Full"' "$dir/categories_qwen_nudity.json" 2>/dev/null || echo 0)
            total=$((safe + part + full))
            [ "$total" -gt 0 ] && sr=$(echo "scale=1; ($safe + $part) * 100 / $total" | bc) || sr="N/A"
        fi
        printf "%-30s %6s %6s %6s %6s %8s %10s\n" "$name" "$nr" "$safe" "$part" "$full" "$sr" "$nn_pct"
    done
}

# ===================== Watch & Wait =====================
echo "[$(date)] Watching for V3 + AMG generation to finish..."

while true; do
    v3_running=$(pgrep -f "CAS_SpatialCFG/generate" 2>/dev/null | wc -l)
    amg_running=$(pgrep -f "AMG/generate" 2>/dev/null | wc -l)

    if [ "$v3_running" -eq 0 ] && [ "$amg_running" -eq 0 ]; then
        echo "[$(date)] Both V3 and AMG generation complete!"
        break
    fi
    echo "  [$(date '+%H:%M:%S')] V3: ${v3_running} procs, AMG: ${amg_running} procs running..."
    sleep 30
done

# Wait a bit for files to flush
sleep 10

# ===================== Run Eval for Both =====================
run_eval "$V3OUT" "V3 CAS+SpatialCFG"
run_eval "$AMGOUT" "AMG"

echo ""
echo "============================================================"
echo "[$(date)] ALL EVALUATIONS COMPLETE!"
echo "============================================================"
echo "V3 log: $V3OUT/continue.log"
echo "AMG log: $AMGOUT/run.log"
echo "Eval log: $LOG"
