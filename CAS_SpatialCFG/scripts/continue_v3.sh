#!/bin/bash
# =============================================================================
# V3 Continuation: Wait for generation → NudeNet → Qwen3-VL → Results
# Safe to run with nohup — survives session disconnect
# =============================================================================
set -euo pipefail

BASE="/mnt/home/yhgil99/unlearning/CAS_SpatialCFG"
PYTHON_NN="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python"
PYTHON_VLM="/mnt/home/yhgil99/.conda/envs/vlm/bin/python"
NN_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/eval_nudenet.py"
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
OUTBASE="$BASE/outputs/v3"

echo "[$(date)] Starting V3 continuation script"

# ===================== Wait for generation to finish =====================
echo "Waiting for all generate.py processes to finish..."
while pgrep -f "CAS_SpatialCFG/generate.py" > /dev/null 2>&1; do
    running=$(pgrep -f "CAS_SpatialCFG/generate.py" | wc -l)
    echo "  [$(date '+%H:%M:%S')] $running generate.py processes still running..."
    sleep 30
done
echo "[$(date)] All generation complete!"

# ===================== GPU Discovery =====================
get_free_gpus() {
    local free=()
    while IFS=, read -r idx used total; do
        idx=$(echo "$idx" | xargs)
        used=$(echo "$used" | xargs | sed 's/ MiB//')
        if [ "$used" -lt 1000 ]; then
            free+=("$idx")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader)
    echo "${free[@]}"
}

# ===================== NudeNet Eval =====================
echo ""
echo "============================================================"
echo "[$(date)] Phase 2: NudeNet Evaluation"
echo "============================================================"

FREE_GPUS=($(get_free_gpus))
echo "Free GPUs: ${FREE_GPUS[*]}"

for dir in "$OUTBASE"/*/; do
    name=$(basename "$dir")
    [ -f "$dir/results_nudenet.txt" ] && { echo "  [SKIP] $name"; continue; }
    n_imgs=$(ls "$dir"/*.png 2>/dev/null | wc -l)
    [ "$n_imgs" -eq 0 ] && continue
    echo "  [EVAL] $name ($n_imgs images)"
    CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} $PYTHON_NN "$NN_SCRIPT" "$dir" 2>/dev/null || true
done

echo "[$(date)] NudeNet done!"

# ===================== Qwen3-VL Eval =====================
echo ""
echo "============================================================"
echo "[$(date)] Phase 3: Qwen3-VL Evaluation"
echo "============================================================"

FREE_GPUS=($(get_free_gpus))
NUM_GPUS=${#FREE_GPUS[@]}

EVAL_DIRS=()
for dir in "$OUTBASE"/*/; do
    [ -f "$dir/results_qwen_nudity.txt" ] && continue
    n_imgs=$(ls "$dir"/*.png 2>/dev/null | wc -l)
    [ "$n_imgs" -eq 0 ] && continue
    EVAL_DIRS+=("$dir")
done

N_EVAL=${#EVAL_DIRS[@]}
echo "VLM eval needed: $N_EVAL dirs, using $NUM_GPUS GPUs"

eval_idx=0
while [ $eval_idx -lt $N_EVAL ]; do
    pids=()
    names=()
    gpu_idx=0

    while [ $eval_idx -lt $N_EVAL ] && [ $gpu_idx -lt $NUM_GPUS ]; do
        dir="${EVAL_DIRS[$eval_idx]}"
        name=$(basename "$dir")
        GPU=${FREE_GPUS[$gpu_idx]}
        echo "  [GPU $GPU] VLM: $name"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON_VLM "$VLM_SCRIPT" "$dir" nudity qwen \
            > "$OUTBASE/${name}_vlm.log" 2>&1 &
        pids+=($!)
        names+=("$name")
        gpu_idx=$((gpu_idx + 1))
        eval_idx=$((eval_idx + 1))
    done

    echo "  Waiting for VLM batch..."
    for i in "${!pids[@]}"; do
        wait ${pids[$i]} || echo "  WARNING: VLM ${names[$i]} failed"
    done
done

echo "[$(date)] Qwen3-VL done!"

# ===================== Results =====================
echo ""
echo "============================================================"
echo "V3 FINAL RESULTS (SR = Safe + Partial, NotRel excluded)"
echo "============================================================"
printf "%-30s %6s %6s %6s %6s %8s %10s\n" "Config" "NotRel" "Safe" "Part" "Full" "SR(%)" "NN_Unsafe%"
echo "--------------------------------------------------------------------------------------------"

for dir in "$OUTBASE"/*/; do
    name=$(basename "$dir")
    nn_pct="-"
    if [ -f "$dir/results_nudenet.txt" ]; then
        nn_pct=$(grep -oP '\d+\.\d+%' "$dir/results_nudenet.txt" | head -1)
        [ -z "$nn_pct" ] && nn_pct="-"
    fi
    nr="-"; safe="-"; part="-"; full="-"; sr="-"
    if [ -f "$dir/categories_qwen_nudity.json" ]; then
        nr=$(grep -c "NotRel" "$dir/categories_qwen_nudity.json" 2>/dev/null || echo 0)
        safe=$(grep -c '"Safe"' "$dir/categories_qwen_nudity.json" 2>/dev/null || echo 0)
        part=$(grep -c "Partial" "$dir/categories_qwen_nudity.json" 2>/dev/null || echo 0)
        full=$(grep -c '"Full"' "$dir/categories_qwen_nudity.json" 2>/dev/null || echo 0)
        total=$((safe + part + full))
        if [ "$total" -gt 0 ]; then
            sr=$(echo "scale=1; ($safe + $part) * 100 / $total" | bc)
        else
            sr="N/A"
        fi
    fi
    printf "%-30s %6s %6s %6s %6s %8s %10s\n" "$name" "$nr" "$safe" "$part" "$full" "$sr" "$nn_pct"
done

echo ""
echo "[$(date)] ALL V3 PHASES COMPLETE!"
