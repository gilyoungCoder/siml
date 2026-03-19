#!/bin/bash
# ============================================================================
# Grid Search: Monitoring Threshold for Sample-Level Monitoring + Spatial CG
# ============================================================================

export CUDA_VISIBLE_DEVICES=0
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# === Configuration ===
CLASSIFIER_CKPT="./work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_DIR="./gradcam_stats/nudity_4class"
SD_MODEL="CompVis/stable-diffusion-v1-4"

# Prompt file (nudity/sexual prompts)
PROMPT_FILE="./prompts/sexual_50.txt"

# Base output directory
BASE_OUTPUT_DIR="./scg_outputs/grid_search_monitoring"

# Number of samples per prompt
NSAMPLES=1

# Fixed parameters
CFG_SCALE=7.5
NUM_STEPS=50
GUIDANCE_SCALE=10.0
BASE_GUIDANCE_SCALE=2.0
SPATIAL_THR_START=0.5
SPATIAL_THR_END=0.1

# ============================================================================
# Grid Search Parameters
# ============================================================================

# Monitoring thresholds to test (P(harm) threshold)
MONITORING_THRESHOLDS=(0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Optional: Also vary spatial thresholds
# SPATIAL_THR_STARTS=(0.3 0.5 0.7)
# SPATIAL_THR_ENDS=(0.1 0.2 0.3)

echo "=============================================="
echo "GRID SEARCH: Monitoring Threshold"
echo "=============================================="
echo "Thresholds to test: ${MONITORING_THRESHOLDS[*]}"
echo "Output base: ${BASE_OUTPUT_DIR}"
echo "=============================================="
echo ""

# Create summary file
SUMMARY_FILE="${BASE_OUTPUT_DIR}/grid_search_summary.txt"
mkdir -p "${BASE_OUTPUT_DIR}"
echo "Grid Search Summary - $(date)" > "${SUMMARY_FILE}"
echo "=============================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Run grid search
for MON_THR in "${MONITORING_THRESHOLDS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Running with monitoring_threshold=${MON_THR}"
    echo "=============================================="

    OUTPUT_DIR="${BASE_OUTPUT_DIR}/mon_thr_${MON_THR}"

    python generate_nudity_4class_sample_level_monitoring.py \
        --ckpt_path "${SD_MODEL}" \
        --prompt_file "${PROMPT_FILE}" \
        --output_dir "${OUTPUT_DIR}" \
        --nsamples ${NSAMPLES} \
        --cfg_scale ${CFG_SCALE} \
        --num_inference_steps ${NUM_STEPS} \
        --classifier_ckpt "${CLASSIFIER_CKPT}" \
        --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
        --monitoring_threshold ${MON_THR} \
        --guidance_scale ${GUIDANCE_SCALE} \
        --base_guidance_scale ${BASE_GUIDANCE_SCALE} \
        --spatial_threshold_start ${SPATIAL_THR_START} \
        --spatial_threshold_end ${SPATIAL_THR_END} \
        --seed 1234

    # Extract stats from generation_stats.json and append to summary
    if [ -f "${OUTPUT_DIR}/generation_stats.json" ]; then
        echo "monitoring_threshold=${MON_THR}" >> "${SUMMARY_FILE}"
        python -c "
import json
with open('${OUTPUT_DIR}/generation_stats.json') as f:
    d = json.load(f)
o = d['overall']
print(f\"  avg_guided_steps: {o['avg_guided_steps']:.1f}\")
print(f\"  avg_guidance_ratio: {o['avg_guidance_ratio']*100:.1f}%\")
print(f\"  no_guidance: {o['no_guidance_count']}\")
print(f\"  light: {o['light_guidance_count']}\")
print(f\"  medium: {o['medium_guidance_count']}\")
print(f\"  heavy: {o['heavy_guidance_count']}\")
" >> "${SUMMARY_FILE}"
        echo "" >> "${SUMMARY_FILE}"
    fi
done

echo ""
echo "=============================================="
echo "GRID SEARCH COMPLETE!"
echo "=============================================="
echo "Results saved to: ${BASE_OUTPUT_DIR}"
echo "Summary: ${SUMMARY_FILE}"
echo ""
echo "Next steps:"
echo "  1. Run NudeNet evaluation on each output directory"
echo "  2. Compare ASR across different thresholds"
echo "  3. Check FID/CLIP scores"
echo "=============================================="

# Print summary
echo ""
echo "=== SUMMARY ==="
cat "${SUMMARY_FILE}"
