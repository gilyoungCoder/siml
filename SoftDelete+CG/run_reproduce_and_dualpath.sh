#!/bin/bash
# Step 1: Reproduce original 87.3% baseline (gradcam)
# Step 2: Run dual_path with same config
# Uses 8 GPUs: GPU 0-4 for baseline top 5 configs, GPU 5-7 + overflow for dual_path
#
# Usage: bash run_reproduce_and_dualpath.sh
# Run on any server with 8 GPUs and shared NAS

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd

CLASSIFIER_CKPT="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/nudity_4class"
PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
VLM_PYTHON="/mnt/home/yhgil99/.conda/envs/vlm/bin/python"
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"

# Best config (originally 87.3%)
GS=17.5
BS=1.0
SP_S=0.2
SP_E=0.4

BASELINE_OUT="scg_outputs/reproduce_87/ringabell/gradcam_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}"
DUALPATH_OUT="scg_outputs/reproduce_87/ringabell/dp_gradcam_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}_ds14_z0t0.0005"

run_baseline() {
    local GPU=$1
    local OUT=$2
    local LOG="/tmp/reproduce_baseline_$(basename $OUT).log"

    echo "[GPU $GPU] Baseline: $(basename $OUT)" | tee "$LOG"

    if [ -f "${OUT}/categories_qwen3_vl_nudity.json" ]; then
        echo "[GPU $GPU] SKIP (done)" | tee -a "$LOG"
        return
    fi

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

    CUDA_VISIBLE_DEVICES=$GPU $VLM_PYTHON "$VLM_SCRIPT" "$OUT" nudity qwen >> "$LOG" 2>&1
    echo "[GPU $GPU] DONE baseline" | tee -a "$LOG"
}

run_dualpath() {
    local GPU=$1
    local OUT=$2
    local LOG="/tmp/reproduce_dualpath_$(basename $OUT).log"

    echo "[GPU $GPU] Dual-path: $(basename $OUT)" | tee "$LOG"

    if [ -f "${OUT}/categories_qwen3_vl_nudity.json" ]; then
        echo "[GPU $GPU] SKIP (done)" | tee -a "$LOG"
        return
    fi

    if [ ! -f "${OUT}/generation_stats.json" ]; then
        CUDA_VISIBLE_DEVICES=$GPU python generate_unified_monitoring.py \
            --ckpt_path CompVis/stable-diffusion-v1-4 \
            --prompt_file "$PROMPT_FILE" \
            --output_dir "$OUT" \
            --classifier_ckpt "$CLASSIFIER_CKPT" \
            --gradcam_stats_dir "$GRADCAM_STATS_DIR" \
            --monitoring_mode dual_path \
            --z0_decision_step 14 \
            --monitoring_threshold 0.0005 \
            --cdf_threshold 0.05 \
            --spatial_mode gradcam \
            --guidance_scale $GS --base_guidance_scale $BS \
            --spatial_threshold_start $SP_S --spatial_threshold_end $SP_E \
            --spatial_threshold_strategy cosine \
            --num_inference_steps 50 --cfg_scale 7.5 --seed 42 --nsamples 1 \
            >> "$LOG" 2>&1
    fi

    CUDA_VISIBLE_DEVICES=$GPU $VLM_PYTHON "$VLM_SCRIPT" "$OUT" nudity qwen >> "$LOG" 2>&1
    echo "[GPU $GPU] DONE dual_path" | tee -a "$LOG"
}

# Launch both in parallel: baseline on GPU 0, dual_path on GPU 1
run_baseline 0 "$BASELINE_OUT" &
run_dualpath 1 "$DUALPATH_OUT" &

echo "Launched baseline (GPU 0) and dual_path (GPU 1). Waiting..."
wait

echo ""
echo "=== RESULTS ==="
echo "--- Baseline (gradcam) ---"
cat "${BASELINE_OUT}/results_qwen3_vl_nudity.txt" 2>/dev/null || echo "NO RESULTS"
echo ""
echo "--- Dual-path (ds14, z0t0.0005) ---"
cat "${DUALPATH_OUT}/results_qwen3_vl_nudity.txt" 2>/dev/null || echo "NO RESULTS"
echo ""
echo "=== DONE ==="
