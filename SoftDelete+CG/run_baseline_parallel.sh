#!/bin/bash
# Baseline (gradcam) rerun - 8 configs on 8 GPUs in parallel
# Configs 0,1 already done. This runs configs 2-9.
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd

CLASSIFIER_CKPT="work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_DIR="gradcam_stats/nudity_4class"
PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
VLM_PYTHON="/mnt/home/yhgil99/.conda/envs/vlm/bin/python"
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
OUTPUT_BASE="scg_outputs/baseline_rerun"

# Remaining 8 configs (configs 2-9): GS|BS|SP_S|SP_E|GPU
declare -a JOBS=(
    "17.5|1.0|0.1|0.3|0"
    "12.5|1.0|0.2|0.4|1"
    "12.5|1.0|0.1|0.4|2"
    "15.0|1.0|0.2|0.4|3"
    "17.5|1.0|0.1|0.4|4"
    "15.0|1.0|0.1|0.3|5"
    "17.5|1.0|0.2|0.3|6"
    "12.5|1.0|0.1|0.3|7"
)

run_job() {
    local GS=$1 BS=$2 SP_S=$3 SP_E=$4 GPU=$5
    local EXP="gradcam_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}"
    local OUT="${OUTPUT_BASE}/ringabell/${EXP}"
    local LOG="/tmp/baseline_${EXP}.log"

    echo "[GPU $GPU] Starting $EXP" | tee "$LOG"

    if [ -f "${OUT}/categories_qwen3_vl_nudity.json" ]; then
        echo "[GPU $GPU] SKIP (done): $EXP" | tee -a "$LOG"
        return
    fi

    # Generate
    if [ ! -f "${OUT}/generation_stats.json" ]; then
        CUDA_VISIBLE_DEVICES=$GPU python generate_unified_monitoring.py \
            --ckpt_path CompVis/stable-diffusion-v1-4 \
            --prompt_file "$PROMPT_FILE" \
            --output_dir "$OUT" \
            --classifier_ckpt "$CLASSIFIER_CKPT" \
            --gradcam_stats_dir "$GRADCAM_STATS_DIR" \
            --monitoring_mode gradcam \
            --monitoring_threshold 0.05 \
            --spatial_mode gradcam \
            --guidance_scale $GS --base_guidance_scale $BS \
            --spatial_threshold_start $SP_S --spatial_threshold_end $SP_E \
            --spatial_threshold_strategy cosine \
            --num_inference_steps 50 --cfg_scale 7.5 --seed 42 --nsamples 1 \
            >> "$LOG" 2>&1
    fi

    # VLM eval
    CUDA_VISIBLE_DEVICES=$GPU $VLM_PYTHON "$VLM_SCRIPT" "$OUT" nudity qwen >> "$LOG" 2>&1

    echo "[GPU $GPU] DONE: $EXP" | tee -a "$LOG"
}

# Launch all 8 jobs in parallel
for job in "${JOBS[@]}"; do
    IFS='|' read -r GS BS SP_S SP_E GPU <<< "$job"
    run_job "$GS" "$BS" "$SP_S" "$SP_E" "$GPU" &
done

echo "All 8 baseline jobs launched. Waiting..."
wait
echo "=== ALL BASELINE DONE ==="

# Print results summary
echo ""
echo "=== RESULTS SUMMARY ==="
for job in "${JOBS[@]}"; do
    IFS='|' read -r GS BS SP_S SP_E GPU <<< "$job"
    EXP="gradcam_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}"
    OUT="${OUTPUT_BASE}/ringabell/${EXP}"
    if [ -f "${OUT}/results_qwen3_vl_nudity.txt" ]; then
        SR=$(grep "SR " "${OUT}/results_qwen3_vl_nudity.txt")
        echo "$EXP: $SR"
    else
        echo "$EXP: NO RESULTS"
    fi
done
