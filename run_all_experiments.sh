#!/bin/bash
# =============================================================================
# Master Script: V3 eval 완료 대기 → AMG 전체 실행 → 최종 비교
# nohup으로 실행하면 세션 끊겨도 자동 진행
# =============================================================================
set -euo pipefail

LOG="/mnt/home/yhgil99/unlearning/master_run.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "[$(date)] MASTER SCRIPT START"
echo "============================================================"

# ===================== Phase 1: Wait for V3 generation =====================
echo ""
echo "[$(date)] Phase 1: Waiting for V3 CAS+SpatialCFG generation..."
while pgrep -f "CAS_SpatialCFG/generate" > /dev/null 2>&1; do
    running=$(pgrep -f "CAS_SpatialCFG/generate" | wc -l)
    echo "  [$(date '+%H:%M:%S')] $running V3 processes running..."
    sleep 30
done
echo "[$(date)] V3 generation complete!"

# ===================== Phase 2: V3 NudeNet =====================
echo ""
echo "============================================================"
echo "[$(date)] Phase 2: V3 NudeNet Evaluation"
echo "============================================================"

PYTHON_NN="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python"
PYTHON_VLM="/mnt/home/yhgil99/.conda/envs/vlm/bin/python"
NN_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/eval_nudenet.py"
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
V3OUT="/mnt/home/yhgil99/unlearning/CAS_SpatialCFG/outputs/v3"

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

FREE_GPUS=($(get_free_gpus))
echo "Free GPUs: ${FREE_GPUS[*]}"

for dir in "$V3OUT"/*/; do
    name=$(basename "$dir")
    [ -f "$dir/results_nudenet.txt" ] && { echo "  [SKIP] $name"; continue; }
    n_imgs=$(ls "$dir"/*.png 2>/dev/null | wc -l)
    [ "$n_imgs" -eq 0 ] && continue
    echo "  [EVAL] $name ($n_imgs images)"
    CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} $PYTHON_NN "$NN_SCRIPT" "$dir" 2>/dev/null || true
done

# ===================== Phase 3: V3 Qwen3-VL =====================
echo ""
echo "============================================================"
echo "[$(date)] Phase 3: V3 Qwen3-VL Evaluation"
echo "============================================================"

FREE_GPUS=($(get_free_gpus))
NUM_GPUS=${#FREE_GPUS[@]}

EVAL_DIRS=()
for dir in "$V3OUT"/*/; do
    [ -f "$dir/results_qwen_nudity.txt" ] && continue
    n_imgs=$(ls "$dir"/*.png 2>/dev/null | wc -l)
    [ "$n_imgs" -eq 0 ] && continue
    EVAL_DIRS+=("$dir")
done

eval_idx=0
while [ $eval_idx -lt ${#EVAL_DIRS[@]} ]; do
    pids=(); names=(); gpu_idx=0
    while [ $eval_idx -lt ${#EVAL_DIRS[@]} ] && [ $gpu_idx -lt $NUM_GPUS ]; do
        dir="${EVAL_DIRS[$eval_idx]}"
        name=$(basename "$dir")
        GPU=${FREE_GPUS[$gpu_idx]}
        echo "  [GPU $GPU] VLM: $name"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON_VLM "$VLM_SCRIPT" "$dir" nudity qwen \
            > "$V3OUT/${name}_vlm.log" 2>&1 &
        pids+=($!); names+=("$name"); gpu_idx=$((gpu_idx+1)); eval_idx=$((eval_idx+1))
    done
    for i in "${!pids[@]}"; do wait ${pids[$i]} || echo "  WARN: ${names[$i]}"; done
done

# ===================== V3 Results =====================
echo ""
echo "============================================================"
echo "V3 CAS+SpatialCFG RESULTS"
echo "============================================================"
printf "%-30s %6s %6s %6s %6s %8s %10s\n" "Config" "NotRel" "Safe" "Part" "Full" "SR(%)" "NN_Unsafe%"
echo "--------------------------------------------------------------------------------------------"
for dir in "$V3OUT"/*/; do
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

# ===================== Phase 4: AMG Full Pipeline =====================
echo ""
echo "============================================================"
echo "[$(date)] Phase 4: AMG (Activation Matching Guidance)"
echo "============================================================"

AMGBASE="/mnt/home/yhgil99/unlearning/AMG"
AMGOUT="$AMGBASE/outputs"
PYTHON_GEN="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python"
GEN_SCRIPT="$AMGBASE/generate.py"
GEN_BASELINE="$AMGBASE/generate_baseline.py"
RAB="$AMGBASE/prompts/nudity-ring-a-bell.csv"
CKPT="CompVis/stable-diffusion-v1-4"
mkdir -p "$AMGOUT"

FREE_GPUS=($(get_free_gpus))
NUM_GPUS=${#FREE_GPUS[@]}
echo "Free GPUs for AMG: ${FREE_GPUS[*]} ($NUM_GPUS)"

CONFIGS=(
    "baseline|sld|0|999|0.3|"
    "sld_s1|sld|1.0|0.3|0.3|"
    "sld_s3|sld|3.0|0.3|0.3|"
    "sld_s5|sld|5.0|0.3|0.3|"
    "sld_s7|sld|7.0|0.3|0.3|"
    "ashift_s3|anchor_shift|3.0|0.3|0.3|"
    "ashift_s5|anchor_shift|5.0|0.3|0.3|"
    "ashift_s7|anchor_shift|7.0|0.3|0.3|"
    "sld_s3_d05|sld|3.0|0.5|0.3|"
    "sld_s5_d05|sld|5.0|0.5|0.3|"
    "dual_s3|dual|3.0|0.3|0.3|"
    "dual_s5|dual|5.0|0.3|0.3|"
    "sld_s3_sthr02|sld|3.0|0.3|0.2|"
    "sld_s3_sthr05|sld|3.0|0.3|0.5|"
    "sld_s10|sld|10.0|0.3|0.3|"
    "ashift_s3_d05|anchor_shift|3.0|0.5|0.3|"
)

N=${#CONFIGS[@]}
echo "AMG configs: $N"

config_idx=0; batch=0
while [ $config_idx -lt $N ]; do
    batch=$((batch+1)); pids=(); names=(); gpu_idx=0
    while [ $config_idx -lt $N ] && [ $gpu_idx -lt $NUM_GPUS ]; do
        IFS='|' read -r name guide_mode safety_scale det_thr spatial_thr extra <<< "${CONFIGS[$config_idx]}"
        GPU=${FREE_GPUS[$gpu_idx]}
        OUTDIR="$AMGOUT/$name"
        if [ "$name" = "baseline" ]; then
            echo "  [GPU $GPU] AMG: $name (baseline)"
            CUDA_VISIBLE_DEVICES=$GPU $PYTHON_GEN $GEN_BASELINE \
                --ckpt "$CKPT" --prompts "$RAB" --outdir "$OUTDIR" \
                --nsamples 4 --steps 50 --seed 42 > "$AMGOUT/${name}.log" 2>&1 &
        else
            echo "  [GPU $GPU] AMG: $name"
            CUDA_VISIBLE_DEVICES=$GPU $PYTHON_GEN $GEN_SCRIPT \
                --ckpt "$CKPT" --prompts "$RAB" --outdir "$OUTDIR" \
                --nsamples 4 --steps 50 --seed 42 \
                --guide_mode "$guide_mode" --safety_scale "$safety_scale" \
                --det_threshold "$det_thr" --spatial_threshold "$spatial_thr" \
                --det_sticky $extra > "$AMGOUT/${name}.log" 2>&1 &
        fi
        pids+=($!); names+=("$name"); gpu_idx=$((gpu_idx+1)); config_idx=$((config_idx+1))
    done
    echo "  Batch $batch: ${#pids[@]} jobs, waiting..."
    for i in "${!pids[@]}"; do wait ${pids[$i]} || echo "  WARN: ${names[$i]}"; done
    echo "  Batch $batch done!"
done
echo "[$(date)] AMG generation complete!"

# AMG NudeNet
echo ""
echo "[$(date)] AMG NudeNet..."
FREE_GPUS=($(get_free_gpus))
for dir in "$AMGOUT"/*/; do
    name=$(basename "$dir")
    [ -f "$dir/results_nudenet.txt" ] && continue
    n_imgs=$(ls "$dir"/*.png 2>/dev/null | wc -l)
    [ "$n_imgs" -eq 0 ] && continue
    echo "  [EVAL] $name"
    CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} $PYTHON_NN "$NN_SCRIPT" "$dir" 2>/dev/null || true
done

# AMG Qwen3-VL
echo ""
echo "[$(date)] AMG Qwen3-VL..."
FREE_GPUS=($(get_free_gpus))
NUM_GPUS=${#FREE_GPUS[@]}

EVAL_DIRS=()
for dir in "$AMGOUT"/*/; do
    [ -f "$dir/results_qwen_nudity.txt" ] && continue
    n_imgs=$(ls "$dir"/*.png 2>/dev/null | wc -l)
    [ "$n_imgs" -eq 0 ] && continue
    EVAL_DIRS+=("$dir")
done

eval_idx=0
while [ $eval_idx -lt ${#EVAL_DIRS[@]} ]; do
    pids=(); names=(); gpu_idx=0
    while [ $eval_idx -lt ${#EVAL_DIRS[@]} ] && [ $gpu_idx -lt $NUM_GPUS ]; do
        dir="${EVAL_DIRS[$eval_idx]}"
        name=$(basename "$dir")
        GPU=${FREE_GPUS[$gpu_idx]}
        echo "  [GPU $GPU] VLM: $name"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON_VLM "$VLM_SCRIPT" "$dir" nudity qwen \
            > "$AMGOUT/${name}_vlm.log" 2>&1 &
        pids+=($!); names+=("$name"); gpu_idx=$((gpu_idx+1)); eval_idx=$((eval_idx+1))
    done
    for i in "${!pids[@]}"; do wait ${pids[$i]} || echo "  WARN: ${names[$i]}"; done
done

# AMG Results
echo ""
echo "============================================================"
echo "AMG RESULTS"
echo "============================================================"
printf "%-30s %6s %6s %6s %6s %8s %10s\n" "Config" "NotRel" "Safe" "Part" "Full" "SR(%)" "NN_Unsafe%"
echo "--------------------------------------------------------------------------------------------"
for dir in "$AMGOUT"/*/; do
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

echo ""
echo "============================================================"
echo "[$(date)] ALL EXPERIMENTS COMPLETE!"
echo "============================================================"
echo "V3 results: $V3OUT"
echo "AMG results: $AMGOUT"
echo "Full log: $LOG"
