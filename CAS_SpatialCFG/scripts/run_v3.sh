#!/bin/bash
# =============================================================================
# V3 Pipeline: CAS + Spatial CFG (SLD/DAG style)
# Phase 1: Grid search on Ring-A-Bell with 8 GPUs
# Phase 2: NudeNet eval
# Phase 3: Qwen3-VL eval
# Phase 4: Best config → all 4 datasets
# =============================================================================
set -euo pipefail

BASE="/mnt/home/yhgil99/unlearning/CAS_SpatialCFG"
PYTHON_GEN="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python"
PYTHON_NN="/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python"
PYTHON_VLM="/mnt/home/yhgil99/.conda/envs/vlm/bin/python"
GEN_SCRIPT="$BASE/generate.py"
GEN_BASELINE="$BASE/generate_baseline.py"
NN_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/eval_nudenet.py"
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
CKPT="CompVis/stable-diffusion-v1-4"

OUTBASE="$BASE/outputs/v3"
mkdir -p "$OUTBASE"

# Ring-A-Bell for grid search
RAB="$BASE/prompts/nudity-ring-a-bell.csv"

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

FREE_GPUS=($(get_free_gpus))
NUM_GPUS=${#FREE_GPUS[@]}
echo "Free GPUs: ${FREE_GPUS[*]} ($NUM_GPUS total)"

if [ "$NUM_GPUS" -lt 1 ]; then
    echo "ERROR: No free GPUs!"
    exit 1
fi

# ===================== Phase 1: Generation =====================
echo ""
echo "============================================================"
echo "Phase 1: Grid Search on Ring-A-Bell ($NUM_GPUS GPUs)"
echo "============================================================"

# Config format: "name|guide_mode|safety_scale|guide_start_frac|spatial_threshold|cas_threshold|spatial_mode|extra"
CONFIGS=(
    # Baseline (no guidance)
    "baseline|sld|0|0|0.3|999|noise|"
    # SLD mode — different safety scales
    "sld_s1|sld|1.0|0.0|0.3|0.3|noise|"
    "sld_s3|sld|3.0|0.0|0.3|0.3|noise|"
    "sld_s5|sld|5.0|0.0|0.3|0.3|noise|"
    "sld_s7|sld|7.0|0.0|0.3|0.3|noise|"
    # Anchor shift mode
    "ashift_s3|anchor_shift|3.0|0.0|0.3|0.3|noise|"
    "ashift_s5|anchor_shift|5.0|0.0|0.3|0.3|noise|"
    # Cross-attention spatial
    "xattn_sld_s3|sld|3.0|0.0|0.3|0.3|crossattn|"
    "xattn_sld_s5|sld|5.0|0.0|0.3|0.3|crossattn|"
    "xattn_ashift_s3|anchor_shift|3.0|0.0|0.3|0.3|crossattn|"
    "xattn_ashift_s5|anchor_shift|5.0|0.0|0.3|0.3|crossattn|"
    # Late-stage guidance
    "sld_s3_late|sld|3.0|0.3|0.3|0.3|noise|"
    "sld_s5_late|sld|5.0|0.3|0.3|0.3|noise|"
    # DAG adaptive mode
    "dag_s3|dag_adaptive|3.0|0.0|0.3|0.3|noise|"
    "dag_s5|dag_adaptive|5.0|0.0|0.3|0.3|noise|"
    # CAS threshold variations
    "sld_s3_cas05|sld|3.0|0.0|0.3|0.5|noise|"
)

N_CONFIGS=${#CONFIGS[@]}
echo "Total configs: $N_CONFIGS"

# Run in batches of NUM_GPUS
batch_idx=0
config_idx=0

while [ $config_idx -lt $N_CONFIGS ]; do
    batch_idx=$((batch_idx + 1))
    pids=()
    names=()
    gpu_idx=0

    while [ $config_idx -lt $N_CONFIGS ] && [ $gpu_idx -lt $NUM_GPUS ]; do
        IFS='|' read -r name guide_mode safety_scale start_frac spatial_thr cas_thr spatial_mode extra <<< "${CONFIGS[$config_idx]}"
        GPU=${FREE_GPUS[$gpu_idx]}
        OUTDIR="$OUTBASE/$name"

        if [ "$name" = "baseline" ]; then
            # Baseline: no guidance
            echo "[GPU $GPU] GEN: $name (baseline)"
            CUDA_VISIBLE_DEVICES=$GPU $PYTHON_GEN $GEN_BASELINE \
                --ckpt "$CKPT" --prompts "$RAB" --outdir "$OUTDIR" \
                --nsamples 4 --steps 50 --seed 42 \
                > "$OUTBASE/${name}.log" 2>&1 &
        else
            echo "[GPU $GPU] GEN: $name"
            CUDA_VISIBLE_DEVICES=$GPU $PYTHON_GEN $GEN_SCRIPT \
                --ckpt "$CKPT" --prompts "$RAB" --outdir "$OUTDIR" \
                --nsamples 4 --steps 50 --seed 42 \
                --guide_mode "$guide_mode" --safety_scale "$safety_scale" \
                --guide_start_frac "$start_frac" \
                --spatial_threshold "$spatial_thr" --cas_threshold "$cas_thr" \
                --spatial_mode "$spatial_mode" \
                --cas_sticky $extra \
                > "$OUTBASE/${name}.log" 2>&1 &
        fi
        pids+=($!)
        names+=("$name")
        gpu_idx=$((gpu_idx + 1))
        config_idx=$((config_idx + 1))
    done

    echo "  Batch $batch_idx: ${#pids[@]} jobs launched, waiting..."
    for i in "${!pids[@]}"; do
        wait ${pids[$i]} || echo "  WARNING: ${names[$i]} exited with error"
    done
    echo "  Batch $batch_idx done!"
done

echo ""
echo "Phase 1 COMPLETE! All generations done."

# ===================== Phase 2: NudeNet Eval =====================
echo ""
echo "============================================================"
echo "Phase 2: NudeNet Evaluation"
echo "============================================================"

for dir in "$OUTBASE"/*/; do
    name=$(basename "$dir")
    if [ -f "$dir/results_nudenet.txt" ]; then
        echo "  [SKIP] $name (already evaluated)"
        continue
    fi
    # Count images
    n_imgs=$(ls "$dir"/*.png 2>/dev/null | wc -l)
    if [ "$n_imgs" -eq 0 ]; then
        echo "  [SKIP] $name (no images)"
        continue
    fi
    echo "  [EVAL] $name ($n_imgs images)"
    CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} $PYTHON_NN "$NN_SCRIPT" "$dir" 2>/dev/null || true
done

echo "Phase 2 COMPLETE!"

# ===================== Phase 3: Qwen3-VL Eval =====================
echo ""
echo "============================================================"
echo "Phase 3: Qwen3-VL Evaluation (nudity concept)"
echo "============================================================"

# Collect dirs that need VLM eval
EVAL_DIRS=()
for dir in "$OUTBASE"/*/; do
    name=$(basename "$dir")
    if [ -f "$dir/results_qwen_nudity.txt" ]; then
        continue
    fi
    n_imgs=$(ls "$dir"/*.png 2>/dev/null | wc -l)
    if [ "$n_imgs" -eq 0 ]; then
        continue
    fi
    EVAL_DIRS+=("$dir")
done

N_EVAL=${#EVAL_DIRS[@]}
echo "VLM eval needed for $N_EVAL directories"

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

echo "Phase 3 COMPLETE!"

# ===================== Phase 4: Results Summary =====================
echo ""
echo "============================================================"
echo "FINAL RESULTS (SR = Safe + Partial, NotRel excluded)"
echo "============================================================"
printf "%-30s %6s %6s %6s %6s %8s %10s\n" "Config" "NotRel" "Safe" "Part" "Full" "SR(%)" "NN_Unsafe%"
echo "--------------------------------------------------------------------------------------------"

for dir in "$OUTBASE"/*/; do
    name=$(basename "$dir")

    # NudeNet
    nn_pct="-"
    if [ -f "$dir/results_nudenet.txt" ]; then
        nn_pct=$(grep -oP '\d+\.\d+%' "$dir/results_nudenet.txt" | head -1)
        [ -z "$nn_pct" ] && nn_pct="-"
    fi

    # Qwen3-VL
    nr="-"; safe="-"; part="-"; full="-"; sr="-"
    vlm_file="$dir/results_qwen_nudity.txt"
    if [ -f "$vlm_file" ]; then
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
echo "============================================================"
echo "ALL PHASES COMPLETE!"
echo "============================================================"
