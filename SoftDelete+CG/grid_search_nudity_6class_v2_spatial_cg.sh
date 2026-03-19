#!/bin/bash
# ============================================================================
# Grid Search: Nudity 6-Class v2 Spatial Classifier Guidance
#
# v2 Classes:
#   0: benign (no people)
#   1: safe_clothed (casual clothes) - common safe target
#   2: harm_nude (completely naked + topless merged)
#   3: harm_lingerie
#   4: harm_swimwear
#   5: harm_color (color artifacts) - NEW
#
# Guidance: all harm (2-5) -> safe_clothed (1)
# ============================================================================

export CUDA_VISIBLE_DEVICES=1

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

# Paths
MODEL_PATH="CompVis/stable-diffusion-v1-4"
STEP=22700
CLASSIFIER_PATH="./work_dirs/nudity_6class_v2/checkpoint/step_${STEP}/classifier.pth"
PROMPT_FILE="./prompts/sexual_50.txt"
OUTPUT_BASE_DIR="./scg_outputs/grid_search_nudity_6class_v2_step${STEP}"

# GradCAM stats directory
GRADCAM_STATS_DIR="./gradcam_stats/nudity_6class_v2_step${STEP}"

# Grid search parameters
GUIDANCE_SCALES=(10.0 15.0 20.0)
SPATIAL_THR_STARTS=(0.3 0.5 0.7)
SPATIAL_THR_ENDS=(0.3 0.5)
HARMFUL_SCALES=(0.5 1.0 1.5)
BASE_GUIDANCE_SCALES=(0.0 1.0 2.0)
THR_STRATEGIES=("linear_decrease" "cosine_anneal")

echo "=============================================="
echo "Grid Search: Nudity 6-Class v2 Spatial CG"
echo "=============================================="
echo "Classifier: ${CLASSIFIER_PATH}"
echo "Output base: ${OUTPUT_BASE_DIR}"
echo ""
echo "Grid parameters:"
echo "  GUIDANCE_SCALES: ${GUIDANCE_SCALES[*]}"
echo "  SPATIAL_THR_STARTS: ${SPATIAL_THR_STARTS[*]}"
echo "  SPATIAL_THR_ENDS: ${SPATIAL_THR_ENDS[*]}"
echo "  HARMFUL_SCALES: ${HARMFUL_SCALES[*]}"
echo "  BASE_GUIDANCE_SCALES: ${BASE_GUIDANCE_SCALES[*]}"
echo "  THR_STRATEGIES: ${THR_STRATEGIES[*]}"
echo "=============================================="
echo ""

# Check classifier exists
if [ ! -f "${CLASSIFIER_PATH}" ]; then
    echo "ERROR: Classifier not found: ${CLASSIFIER_PATH}"
    exit 1
fi

# Check GradCAM stats directory exists
if [ ! -d "${GRADCAM_STATS_DIR}" ]; then
    echo "WARNING: GradCAM stats directory not found: ${GRADCAM_STATS_DIR}"
    echo "Run compute_gradcam_stats_nudity_6class_v2.sh first!"
    echo "Continuing without GradCAM stats (will use per-image normalization)..."
fi

# Run grid search
count=0
total=$((${#GUIDANCE_SCALES[@]} * ${#SPATIAL_THR_STARTS[@]} * ${#SPATIAL_THR_ENDS[@]} * ${#HARMFUL_SCALES[@]} * ${#BASE_GUIDANCE_SCALES[@]} * ${#THR_STRATEGIES[@]}))

for gs in "${GUIDANCE_SCALES[@]}"; do
    for thr_start in "${SPATIAL_THR_STARTS[@]}"; do
        for thr_end in "${SPATIAL_THR_ENDS[@]}"; do
            for hs in "${HARMFUL_SCALES[@]}"; do
                for bgs in "${BASE_GUIDANCE_SCALES[@]}"; do
                    for strategy in "${THR_STRATEGIES[@]}"; do
                        count=$((count + 1))

                        # Create output directory name
                        OUTPUT_DIR="${OUTPUT_BASE_DIR}/gs${gs}_thr${thr_start}-${thr_end}_hs${hs}_bgs${bgs}_${strategy}"

                        # Skip if already exists
                        if [ -d "${OUTPUT_DIR}" ] && [ "$(ls -A ${OUTPUT_DIR} 2>/dev/null | grep -v visualizations | head -1)" ]; then
                            echo "[${count}/${total}] SKIP (exists): ${OUTPUT_DIR}"
                            continue
                        fi

                        echo ""
                        echo "=============================================="
                        echo "[${count}/${total}] Running configuration:"
                        echo "  guidance_scale: ${gs}"
                        echo "  spatial_threshold: ${thr_start} -> ${thr_end}"
                        echo "  harmful_scale: ${hs}"
                        echo "  base_guidance_scale: ${bgs}"
                        echo "  strategy: ${strategy}"
                        echo "  output: ${OUTPUT_DIR}"
                        echo "=============================================="

                        python generate_nudity_6class_spatial_cg.py \
                            "${MODEL_PATH}" \
                            --prompt_file "${PROMPT_FILE}" \
                            --output_dir "${OUTPUT_DIR}" \
                            --classifier_ckpt "${CLASSIFIER_PATH}" \
                            --num_classes 6 \
                            --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
                            --nsamples 1 \
                            --cfg_scale 7.5 \
                            --num_inference_steps 50 \
                            --seed 1234 \
                            --guidance_scale ${gs} \
                            --spatial_threshold_start ${thr_start} \
                            --spatial_threshold_end ${thr_end} \
                            --threshold_strategy ${strategy} \
                            --use_bidirectional \
                            --harmful_scale ${hs} \
                            --base_guidance_scale ${bgs} \
                            --guidance_start_step 0 \
                            --guidance_end_step 50

                        echo "[${count}/${total}] Completed: ${OUTPUT_DIR}"
                    done
                done
            done
        done
    done
done

echo ""
echo "=============================================="
echo "Grid search complete!"
echo "Results saved to: ${OUTPUT_BASE_DIR}"
echo "=============================================="
