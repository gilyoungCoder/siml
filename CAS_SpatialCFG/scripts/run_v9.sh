#!/bin/bash
# ============================================================
# V9: Direct Exemplar HOW
# Phase 1: Generate on Ring-A-Bell + COCO
# Phase 2: NudeNet eval
# Phase 3: Qwen3-VL eval
# Phase 4: Results summary
# ============================================================
set -e

export PYTHONNOUSERSITE=1
PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
VLM_PYTHON="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
BASE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG"
GENERATE="$BASE/generate_v9.py"
EVAL_NN="/mnt/home3/yhgil99/unlearning/vlm/eval_nudenet.py"
EVAL_VLM="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
PROMPTS="$BASE/prompts/nudity-ring-a-bell.csv"
COCO_PROMPTS="$BASE/prompts/coco_30.txt"
OUTBASE="$BASE/outputs/v9"
CONCEPT_DIR="$BASE/exemplars/sd14/concept_directions.pt"

mkdir -p "$OUTBASE"

# Check concept directions exist
if [ ! -f "$CONCEPT_DIR" ]; then
    echo "ERROR: concept_directions.pt not found at $CONCEPT_DIR"
    echo "Run prepare_concept_subspace.py first (or run_v7.sh)"
    exit 1
fi

# Find free GPUs
FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F',' '{gsub(/ MiB/,"",$2); if($2+0 < 1000) print $1}' | tr '\n' ' ')
NUM_GPUS=$(echo $FREE_GPUS | wc -w)
echo "Free GPUs: $FREE_GPUS ($NUM_GPUS total)"

if [ "$NUM_GPUS" -lt 1 ]; then
    echo "ERROR: No free GPUs!"
    exit 1
fi

GPU_ARRAY=($FREE_GPUS)

# ============================================================
# Experiment configs:
# name|guide_mode|safety_scale|cas_threshold|spatial_threshold|sigmoid_alpha|extras
# ============================================================
CONFIGS=(
    # === Core v9: exemplar_hybrid (direct exemplar target/anchor) ===
    "v9_exhyb_ts10_as15|exemplar_hybrid|1.0|0.6|0.3|10|--target_scale 10 --anchor_scale 15"
    "v9_exhyb_ts15_as15|exemplar_hybrid|1.0|0.6|0.3|10|--target_scale 15 --anchor_scale 15"
    "v9_exhyb_ts5_as10|exemplar_hybrid|1.0|0.6|0.3|10|--target_scale 5 --anchor_scale 10"

    # === v9: exemplar_sld (only exemplar target removal) ===
    "v9_exsld_s5|exemplar_sld|5.0|0.6|0.3|10|"
    "v9_exsld_s10|exemplar_sld|10.0|0.6|0.3|10|"

    # === v9: exemplar_contrast (nudity->clothed transform direction) ===
    "v9_contrast_s5|exemplar_contrast|5.0|0.6|0.3|10|"
    "v9_contrast_s10|exemplar_contrast|10.0|0.6|0.3|10|"

    # === v9: exemplar_inpaint (blend with exemplar anchor CFG) ===
    "v9_inpaint_s1|exemplar_inpaint|1.0|0.6|0.3|10|"
)

# ============================================================
# Phase 1: Generate images
# ============================================================
echo "============================================================"
echo "Phase 1: Image Generation (${#CONFIGS[@]} configs)"
echo "============================================================"

PIDS=()
GPU_IDX=0

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME GMODE SSCALE CAS_T SPAT_T SIG_A EXTRAS <<< "$cfg"
    GPU=${GPU_ARRAY[$GPU_IDX]}

    OUTDIR="$OUTBASE/$NAME"
    if [ -d "$OUTDIR" ] && [ "$(ls "$OUTDIR"/*.png 2>/dev/null | wc -l)" -ge 316 ]; then
        echo "SKIP $NAME (already have 316+ images)"
    else
        echo "Starting $NAME on GPU $GPU..."
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$GENERATE" \
            --prompts "$PROMPTS" \
            --outdir "$OUTDIR" \
            --concept_dir_path "$CONCEPT_DIR" \
            --guide_mode "$GMODE" \
            --safety_scale $SSCALE \
            --cas_threshold $CAS_T \
            --spatial_threshold $SPAT_T \
            --sigmoid_alpha $SIG_A \
            $EXTRAS \
            > "$OUTBASE/${NAME}.log" 2>&1 &
        PIDS+=($!)
        GPU_IDX=$(( (GPU_IDX + 1) % NUM_GPUS ))
    fi
done

# COCO FP check
COCO_DIR="$OUTBASE/COCO_v9_exhyb_ts10_as15"
if [ -d "$COCO_DIR" ] && [ "$(ls "$COCO_DIR"/*.png 2>/dev/null | wc -l)" -ge 120 ]; then
    echo "SKIP COCO (already done)"
else
    GPU=${GPU_ARRAY[$GPU_IDX]}
    echo "Starting COCO FP check on GPU $GPU..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$GENERATE" \
        --prompts "$COCO_PROMPTS" \
        --outdir "$COCO_DIR" \
        --concept_dir_path "$CONCEPT_DIR" \
        --guide_mode exemplar_hybrid \
        --safety_scale 1.0 \
        --cas_threshold 0.6 \
        --target_scale 10 --anchor_scale 15 \
        > "$OUTBASE/COCO_v9.log" 2>&1 &
    PIDS+=($!)
fi

# Wait for all generation jobs
echo ""
echo "Waiting for ${#PIDS[@]} generation jobs..."
for pid in "${PIDS[@]}"; do
    wait $pid || echo "WARNING: Job $pid failed"
done
echo "Phase 1 complete!"

# ============================================================
# Phase 2: NudeNet Evaluation
# ============================================================
echo ""
echo "============================================================"
echo "Phase 2: NudeNet Evaluation"
echo "============================================================"

PIDS=()
GPU_IDX=0

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME _ _ _ _ _ _ <<< "$cfg"
    OUTDIR="$OUTBASE/$NAME"
    if [ -f "$OUTDIR/results_nudenet.txt" ]; then
        echo "SKIP NudeNet $NAME (already done)"
    elif [ -d "$OUTDIR" ]; then
        GPU=${GPU_ARRAY[$GPU_IDX]}
        echo "NudeNet: $NAME on GPU $GPU"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$EVAL_NN" "$OUTDIR" \
            > "$OUTBASE/nn_${NAME}.log" 2>&1 &
        PIDS+=($!)
        GPU_IDX=$(( (GPU_IDX + 1) % NUM_GPUS ))
    fi
done

if [ -d "$COCO_DIR" ] && [ ! -f "$COCO_DIR/results_nudenet.txt" ]; then
    GPU=${GPU_ARRAY[$GPU_IDX]}
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$EVAL_NN" "$COCO_DIR" \
        > "$OUTBASE/nn_COCO.log" 2>&1 &
    PIDS+=($!)
fi

for pid in "${PIDS[@]}"; do
    wait $pid || echo "WARNING: Job $pid failed"
done
echo "Phase 2 complete!"

# ============================================================
# Phase 3: Qwen3-VL Evaluation
# ============================================================
echo ""
echo "============================================================"
echo "Phase 3: Qwen3-VL Evaluation"
echo "============================================================"

PIDS=()
GPU_IDX=0

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME _ _ _ _ _ _ <<< "$cfg"
    OUTDIR="$OUTBASE/$NAME"
    if [ -f "$OUTDIR/results_qwen_nudity.txt" ]; then
        echo "SKIP Qwen $NAME (already done)"
    elif [ -d "$OUTDIR" ]; then
        GPU=${GPU_ARRAY[$GPU_IDX]}
        echo "Qwen3-VL: $NAME on GPU $GPU"
        CUDA_VISIBLE_DEVICES=$GPU $VLM_PYTHON "$EVAL_VLM" "$OUTDIR" nudity qwen \
            > "$OUTBASE/qwen_${NAME}.log" 2>&1 &
        PIDS+=($!)
        GPU_IDX=$(( (GPU_IDX + 1) % NUM_GPUS ))
    fi
done

if [ -d "$COCO_DIR" ] && [ ! -f "$COCO_DIR/results_qwen_nudity.txt" ]; then
    GPU=${GPU_ARRAY[$GPU_IDX]}
    CUDA_VISIBLE_DEVICES=$GPU $VLM_PYTHON "$EVAL_VLM" "$COCO_DIR" nudity qwen \
        > "$OUTBASE/qwen_COCO.log" 2>&1 &
    PIDS+=($!)
fi

for pid in "${PIDS[@]}"; do
    wait $pid || echo "WARNING: Job $pid failed"
done
echo "Phase 3 complete!"

# ============================================================
# Phase 4: Results Summary
# ============================================================
echo ""
echo "============================================================"
echo "Phase 4: Results Summary"
echo "============================================================"

printf "%-30s | %10s | %10s | %10s\n" "Experiment" "NudeNet%" "Qwen SR%" "Qwen Full%"
printf "%s\n" "$(printf '%.0s-' {1..70})"

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME _ _ _ _ _ _ <<< "$cfg"
    OUTDIR="$OUTBASE/$NAME"

    NN_RATE="-"
    if [ -f "$OUTDIR/results_nudenet.txt" ]; then
        NN_RATE=$(grep -oP 'Unsafe Rate: \K[\d.]+' "$OUTDIR/results_nudenet.txt" 2>/dev/null || echo "-")
    fi

    QWEN_SR="-"
    QWEN_FULL="-"
    if [ -f "$OUTDIR/results_qwen_nudity.txt" ]; then
        SAFE=$(grep -oP 'Safe: \K\d+' "$OUTDIR/results_qwen_nudity.txt" 2>/dev/null || echo "0")
        PARTIAL=$(grep -oP 'Partial: \K\d+' "$OUTDIR/results_qwen_nudity.txt" 2>/dev/null || echo "0")
        NOTREL=$(grep -oP 'NotRel: \K\d+' "$OUTDIR/results_qwen_nudity.txt" 2>/dev/null || echo "0")
        FULL=$(grep -oP 'Full: \K\d+' "$OUTDIR/results_qwen_nudity.txt" 2>/dev/null || echo "0")
        TOTAL=$((SAFE + PARTIAL + NOTREL + FULL))
        if [ "$TOTAL" -gt 0 ]; then
            QWEN_SR=$(echo "scale=1; ($SAFE + $PARTIAL) * 100 / $TOTAL" | bc 2>/dev/null || echo "-")
            QWEN_FULL=$(echo "scale=1; $FULL * 100 / $TOTAL" | bc 2>/dev/null || echo "-")
        fi
    fi

    printf "%-30s | %10s | %10s | %10s\n" "$NAME" "$NN_RATE" "$QWEN_SR" "$QWEN_FULL"
done

COCO_NN="-"
if [ -f "$COCO_DIR/results_nudenet.txt" ]; then
    COCO_NN=$(grep -oP 'Unsafe Rate: \K[\d.]+' "$COCO_DIR/results_nudenet.txt" 2>/dev/null || echo "-")
fi
printf "%-30s | %10s | %10s | %10s\n" "COCO_v9 (FP)" "$COCO_NN" "-" "-"

echo ""
echo "All results saved to: $OUTBASE/"
echo "Done!"
