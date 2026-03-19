#!/bin/bash
# Diagnostic 3: High clip ratio with few guided steps (mon0.1)
# Hypothesis: fewer guided steps (5-9) × strong per-step guidance
# may suppress nudity without catastrophic artifacts
cd /mnt/home/yhgil99/unlearning/z0_clf_guidance
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd

PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
CLASSIFIER_CKPT="./work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth"
HARMFUL_STATS_PATH="./harmful_stats.pt"
SD_CKPT="CompVis/stable-diffusion-v1-4"
DIAG_DIR="./diagnostic_test3"
mkdir -p "$DIAG_DIR"

# Key configs: mon0.1 with high clip ratios
# Also: mon0.1+sticky (more steps ~35-40 but still not 50)
# Also: always with very low clip as safety baseline
CONFIGS=(
    # name | gs | clip | always | sticky | mon_thr | bs | sp_start | sp_end
    # --- mon0.1: ~5-9 guided steps, high per-step guidance ---
    "mon0.1_gs30_cl1.0|30|1.0|false|false|0.1|0.0|0.2|0.3"
    "mon0.1_gs30_cl2.0|30|2.0|false|false|0.1|0.0|0.2|0.3"
    "mon0.1_gs30_cl5.0|30|5.0|false|false|0.1|0.0|0.2|0.3"
    "mon0.1_gs50_cl1.0|50|1.0|false|false|0.1|0.0|0.2|0.3"
    "mon0.1_gs50_cl2.0|50|2.0|false|false|0.1|0.0|0.2|0.3"
    "mon0.1_gs100_cl1.0|100|1.0|false|false|0.1|0.0|0.2|0.3"
    "mon0.1_gs100_cl2.0|100|2.0|false|false|0.1|0.0|0.2|0.3"
    "mon0.1_gs30_noclip|30|0|false|false|0.1|0.0|0.2|0.3"
    # --- mon0.1+sticky: ~35-40 guided steps, high clip ---
    "sticky_gs30_cl0.5|30|0.5|false|true|0.1|0.0|0.2|0.3"
    "sticky_gs30_cl1.0|30|1.0|false|true|0.1|0.0|0.2|0.3"
    "sticky_gs30_cl2.0|30|2.0|false|true|0.1|0.0|0.2|0.3"
    "sticky_gs50_cl0.5|50|0.5|false|true|0.1|0.0|0.2|0.3"
    "sticky_gs50_cl1.0|50|1.0|false|true|0.1|0.0|0.2|0.3"
    # --- base_guidance_scale=1.0 (adds non-spatial guidance) ---
    "always_gs30_cl0.3_bs1|30|0.3|true|false|0.0|1.0|0.2|0.3"
    "always_gs50_cl0.3_bs1|50|0.3|true|false|0.0|1.0|0.2|0.3"
    "always_gs30_cl0.5_bs1|30|0.5|true|false|0.0|1.0|0.2|0.3"
)

echo "=============================================="
echo "DIAGNOSTIC 3 — ${#CONFIGS[@]} configs × 5 prompts"
echo "Focus: high clip + few steps, sticky, base_guidance"
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
        --end_idx 5 \
        --debug"

    if [ "$ALWAYS" = "true" ]; then CMD="$CMD --always_guide"; fi
    if [ "$STICKY" = "true" ]; then CMD="$CMD --sticky_trigger"; fi

    echo "[GPU $GPU] Running: $NAME"
    eval $CMD > "${DIAG_DIR}/${NAME}.log" 2>&1 &
    PIDS+=($!)
    GPU=$(( (GPU + 1) % 8 ))

    # Wait if 16 jobs (2 per GPU)
    if [ ${#PIDS[@]} -ge 16 ]; then
        for pid in "${PIDS[@]}"; do wait $pid; done
        PIDS=()
    fi
done

for pid in "${PIDS[@]}"; do wait $pid; done

echo ""
echo "=============================================="
echo "DIAGNOSTIC 3 RESULTS"
echo "=============================================="

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME GS CL ALWAYS STICKY MON BS SP_S SP_E <<< "$cfg"
    LOG="${DIAG_DIR}/${NAME}.log"
    echo ""
    echo "--- ${NAME} ---"
    if [ -f "$LOG" ]; then
        grep "Guided:" "$LOG" | head -5
        echo ""
        grep "\[guide\]" "$LOG" | head -5
    fi
done

echo ""
echo "Images in: $DIAG_DIR/"
echo "Done!"
