#!/bin/bash
# ============================================================================
# Grid Search for Nudity 4-Class with SAFE-HARM RESTRICTED GRADIENT
#
# Applies Restricted Gradient between safe and harm gradients:
#   delta_safe = g_safe - proj(g_safe onto g_harm)
#   delta_harm = g_harm - proj(g_harm onto g_safe)
#   Final: delta_safe - harmful_scale * delta_harm
#
# Based on best config from 4class_always (noskip)
# ============================================================================

export CUDA_VISIBLE_DEVICES=7

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# ============================================================================
# Fixed parameters
# ============================================================================
SD_MODEL="CompVis/stable-diffusion-v1-4"
CLASSIFIER_CKPT="./work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
PROMPT_FILE="./prompts/sexual_50.txt"
GRADCAM_STATS_DIR="./gradcam_stats/nudity_4class"

BASE_OUTPUT_DIR="./scg_outputs/grid_search_results/nudity_4class_safe_harm_restricted"

NSAMPLES=1
CFG_SCALE=7.5
NUM_INFERENCE_STEPS=50
SEED=1234
GUIDANCE_START_STEP=0
GUIDANCE_END_STEP=50

# ============================================================================
# Grid search parameters (6 configs - based on best noskip config)
# ============================================================================
GUIDANCE_SCALES=(10.0 12.5)
SPATIAL_THRESHOLDS=(
    "0.5 0.1"   # Best from noskip
)
HARMFUL_SCALES=(0.5 1.0 1.5)
BASE_GUIDANCE_SCALES=(2.0)
THRESHOLD_STRATEGIES=("cosine_anneal")

# ============================================================================
# Run grid search
# ============================================================================
mkdir -p "${BASE_OUTPUT_DIR}"

# Count total configurations
total_configs=0
for gs in "${GUIDANCE_SCALES[@]}"; do
    for thr in "${SPATIAL_THRESHOLDS[@]}"; do
        for hs in "${HARMFUL_SCALES[@]}"; do
            for bgs in "${BASE_GUIDANCE_SCALES[@]}"; do
                for strategy in "${THRESHOLD_STRATEGIES[@]}"; do
                    ((total_configs++))
                done
            done
        done
    done
done

echo "=============================================="
echo "GRID SEARCH - Nudity 4-Class SAFE-HARM RESTRICTED"
echo "=============================================="
echo "Total configurations: ${total_configs}"
echo "Output directory: ${BASE_OUTPUT_DIR}"
echo "=============================================="

config_idx=0
for gs in "${GUIDANCE_SCALES[@]}"; do
    for thr in "${SPATIAL_THRESHOLDS[@]}"; do
        read -r thr_start thr_end <<< "${thr}"

        for hs in "${HARMFUL_SCALES[@]}"; do
            for bgs in "${BASE_GUIDANCE_SCALES[@]}"; do
                for strategy in "${THRESHOLD_STRATEGIES[@]}"; do
                    ((config_idx++))

                    config_name="gs${gs}_thr${thr_start}-${thr_end}_hs${hs}_bgs${bgs}_${strategy}"
                    output_dir="${BASE_OUTPUT_DIR}/${config_name}"

                    echo ""
                    echo "[${config_idx}/${total_configs}] ${config_name}"

                    python generate_nudity_4class_spatial_cg_always_safe_harm_restricted.py \
                        "${SD_MODEL}" \
                        --prompt_file "${PROMPT_FILE}" \
                        --output_dir "${output_dir}" \
                        --classifier_ckpt "${CLASSIFIER_CKPT}" \
                        --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
                        --nsamples ${NSAMPLES} \
                        --cfg_scale ${CFG_SCALE} \
                        --num_inference_steps ${NUM_INFERENCE_STEPS} \
                        --seed ${SEED} \
                        --guidance_scale ${gs} \
                        --spatial_threshold_start ${thr_start} \
                        --spatial_threshold_end ${thr_end} \
                        --threshold_strategy "${strategy}" \
                        --use_bidirectional \
                        --harmful_scale ${hs} \
                        --base_guidance_scale ${bgs} \
                        --guidance_start_step ${GUIDANCE_START_STEP} \
                        --guidance_end_step ${GUIDANCE_END_STEP}

                    echo "Completed: ${config_name}"
                done
            done
        done
    done
done

echo ""
echo "=============================================="
echo "GRID SEARCH COMPLETE!"
echo "=============================================="
