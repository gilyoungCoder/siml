#!/bin/bash
# Quick diagnostic: test a few ringabell prompts with different configs
# to analyze spatial mask ratios, guidance strengths, and color artifacts.

cd /mnt/home/yhgil99/unlearning/z0_clf_guidance
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd

PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
CLASSIFIER_CKPT="./work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth"
HARMFUL_STATS_PATH="./harmful_stats.pt"
SD_CKPT="CompVis/stable-diffusion-v1-4"
DIAG_DIR="./diagnostic_test"

mkdir -p "$DIAG_DIR"

# Test configs: vary clip ratio and guidance scale
# Only first 3 prompts to keep it fast
CONFIGS=(
    # name | gs | clip | always | sticky | mon_thr | bs | sp_start | sp_end
    "baseline_gs15_cl0.3|15|0.3|false|false|0.1|0.0|0.2|0.3"
    "gs30_cl0.3|30|0.3|false|false|0.1|0.0|0.2|0.3"
    "gs30_cl1.0|30|1.0|false|false|0.1|0.0|0.2|0.3"
    "gs30_cl2.0|30|2.0|false|false|0.1|0.0|0.2|0.3"
    "gs50_cl1.0|50|1.0|false|false|0.1|0.0|0.2|0.3"
    "gs100_cl1.0|100|1.0|false|false|0.1|0.0|0.2|0.3"
    "always_gs30_cl1.0|30|1.0|true|false|0.0|0.0|0.2|0.3"
    "sticky_gs30_cl1.0|30|1.0|false|true|0.1|0.0|0.2|0.3"
)

echo "=============================================="
echo "DIAGNOSTIC TEST — ${#CONFIGS[@]} configs × 3 prompts"
echo "=============================================="

PIDS=()
GPU=0
for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME GS CL ALWAYS STICKY MON BS SP_S SP_E <<< "$cfg"
    OUT="${DIAG_DIR}/${NAME}"

    CMD="CUDA_VISIBLE_DEVICES=$GPU python generate_monitoring.py \
        --ckpt_path $SD_CKPT \
        --prompt_file $PROMPT_FILE \
        --output_dir $OUT \
        --classifier_ckpt $CLASSIFIER_CKPT \
        --harmful_stats_path $HARMFUL_STATS_PATH \
        --monitoring_mode classifier \
        --monitoring_threshold $MON \
        --guidance_scale $GS \
        --base_guidance_scale $BS \
        --grad_clip_ratio $CL \
        --spatial_threshold_start $SP_S \
        --spatial_threshold_end $SP_E \
        --seed 42 \
        --end_idx 3 \
        --debug"

    if [ "$ALWAYS" = "true" ]; then
        CMD="$CMD --always_guide"
    fi
    if [ "$STICKY" = "true" ]; then
        CMD="$CMD --sticky_trigger"
    fi

    echo "[GPU $GPU] Running: $NAME"
    eval $CMD > "${DIAG_DIR}/${NAME}.log" 2>&1 &
    PIDS+=($!)
    GPU=$(( (GPU + 1) % 8 ))
done

echo "Waiting for ${#PIDS[@]} experiments..."
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo ""
echo "=============================================="
echo "DIAGNOSTIC RESULTS"
echo "=============================================="

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME GS CL ALWAYS STICKY MON BS SP_S SP_E <<< "$cfg"
    LOG="${DIAG_DIR}/${NAME}.log"
    echo ""
    echo "--- ${NAME} ---"
    if [ -f "$LOG" ]; then
        # Show guided step counts
        grep "Guided:" "$LOG" | head -3
        echo ""
        # Show sample of debug lines (mask ratios, guidance norms)
        grep "\[guide\]" "$LOG" | head -10
    else
        echo "  (no log)"
    fi
done

echo ""
echo "Images saved in: $DIAG_DIR/*/  (check visually for artifacts)"
echo "Done!"
