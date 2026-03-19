#!/bin/bash
# ============================================================================
# Grid Search for Nudity 4-Class Selective Spatial Classifier Guidance
#
# Searches over:
#   - guidance_scale
#   - spatial_threshold_start / spatial_threshold_end
#   - harmful_scale
#   - threshold_strategy
#
# IMPORTANT: Run compute_gradcam_statistics_4class.sh FIRST!
#
# ============================================================================

export CUDA_VISIBLE_DEVICES=0

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# ============================================================================
# Fixed parameters
# ============================================================================
SD_MODEL="CompVis/stable-diffusion-v1-4"
CLASSIFIER_CKPT="./work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
PROMPT_FILE="./prompts/sexual_50.txt"
GRADCAM_STATS_DIR="./gradcam_stats/nudity_4class"

BASE_OUTPUT_DIR="./scg_outputs/grid_search_results/nudity_4class"

NSAMPLES=1
CFG_SCALE=7.5
NUM_INFERENCE_STEPS=50
SEED=1234
BASE_GUIDANCE_SCALE=0.0
GUIDANCE_START_STEP=0
GUIDANCE_END_STEP=50

# ============================================================================
# Grid search parameters
# ============================================================================
GUIDANCE_SCALES=(7.5 10.0 12.5)
SPATIAL_THRESHOLDS=(
    "0.7 0.3"   # Moderate: start high, end low
    "0.7 0.5"   # Moderate-conservative
    "0.5 0.3"   # Lower range
    "0.5 0.1"   # Aggressive
    "0.3 0.5"   # Inverse: start low, end high
    "0.3 0.7"   # Inverse aggressive
    "0.5 0.5"   # Constant moderate
    "0.3 0.3"   # Constant low
)  # "start end" pairs
HARMFUL_SCALES=(0.5 1.0 1.5)
BASE_GUIDANCE_SCALES=(0.0 1.0 2.0)
THRESHOLD_STRATEGIES=("linear_decrease" "cosine_anneal")

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
echo "GRID SEARCH - Nudity 4-Class Selective Spatial CG"
echo "=============================================="
echo "Total configurations: ${total_configs}"
echo "Prompt file: ${PROMPT_FILE}"
echo "Output directory: ${BASE_OUTPUT_DIR}"
echo ""
echo "Grid parameters:"
echo "  guidance_scale: ${GUIDANCE_SCALES[*]}"
echo "  spatial_threshold: ${SPATIAL_THRESHOLDS[*]}"
echo "  harmful_scale: ${HARMFUL_SCALES[*]}"
echo "  base_guidance_scale: ${BASE_GUIDANCE_SCALES[*]}"
echo "  threshold_strategy: ${THRESHOLD_STRATEGIES[*]}"
echo "=============================================="
echo ""

config_idx=0
for gs in "${GUIDANCE_SCALES[@]}"; do
    for thr in "${SPATIAL_THRESHOLDS[@]}"; do
        # Parse threshold pair
        read -r thr_start thr_end <<< "${thr}"

        for hs in "${HARMFUL_SCALES[@]}"; do
            for bgs in "${BASE_GUIDANCE_SCALES[@]}"; do
                for strategy in "${THRESHOLD_STRATEGIES[@]}"; do
                    ((config_idx++))

                    # Create config name
                    config_name="gs${gs}_thr${thr_start}-${thr_end}_hs${hs}_bgs${bgs}_${strategy}"
                    output_dir="${BASE_OUTPUT_DIR}/${config_name}"

                    echo ""
                    echo "=============================================="
                    echo "[${config_idx}/${total_configs}] ${config_name}"
                    echo "=============================================="
                    echo "  guidance_scale: ${gs}"
                    echo "  spatial_threshold: ${thr_start} -> ${thr_end}"
                    echo "  harmful_scale: ${hs}"
                    echo "  base_guidance_scale: ${bgs}"
                    echo "  threshold_strategy: ${strategy}"
                    echo "  output_dir: ${output_dir}"
                    echo ""

                    python generate_nudity_4class_spatial_cg.py \
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

                    echo ""
                    echo "Completed: ${config_name}"
                    echo ""

                done
            done
        done
    done
done

echo ""
echo "=============================================="
echo "GRID SEARCH COMPLETE!"
echo "=============================================="
echo "Results saved to: ${BASE_OUTPUT_DIR}"
echo "Total configurations: ${total_configs}"
echo "=============================================="
