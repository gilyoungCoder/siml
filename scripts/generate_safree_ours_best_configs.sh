#!/bin/bash
# ============================================================================
# Generate SAFREE+Ours best configs for all concepts (high_tox + low_tox)
#
# Uses best parameters from best_configs.txt:
#   - Nudity (4class): cgs5.0_st0.5-0.3_hs0.5_bgs2.0
#   - Violence (13class, step28400): cgs5.0_st0.7-0.3_hs1.0_bgs0.0
#   - Harassment (9class): cgs10.0_st0.5-0.3_hs1.0_bgs2.0
#   - Hate (9class): cgs7.5_st0.5-0.5_hs1.5_bgs0.0
#   - Shocking (9class): cgs7.5_st0.3-0.5_hs1.5_bgs0.0
#   - Illegal (9class): cgs10.0_st0.5-0.5_hs1.5_bgs2.0
#   - Selfharm (9class): cgs5.0_st0.3-0.7_hs0.5_bgs0.0
#
# Usage:
#   bash generate_safree_ours_best_configs.sh <GPU> <CONCEPT>
#   bash generate_safree_ours_best_configs.sh 0 all        # All concepts
#   bash generate_safree_ours_best_configs.sh 0 nudity     # Single concept
#   bash generate_safree_ours_best_configs.sh auto all     # Auto GPU selection
#
# For parallel execution on multiple GPUs:
#   bash generate_safree_ours_best_configs.sh 0 nudity &
#   bash generate_safree_ours_best_configs.sh 1 violence &
#   bash generate_safree_ours_best_configs.sh 2 harassment &
#   wait
# ============================================================================

set -e

# ============================================================================
# Find available GPU with most free memory
# ============================================================================
find_free_gpu() {
    python3 -c "
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True)
gpus = []
for line in result.stdout.strip().split('\n'):
    idx, mem = line.split(',')
    gpus.append((int(idx.strip()), int(mem.strip())))
gpus.sort(key=lambda x: x[1], reverse=True)
print(gpus[0][0])
"
}

if [ $# -lt 2 ]; then
    echo "Usage: $0 <GPU|auto> <CONCEPT>"
    echo ""
    echo "GPU options:"
    echo "  0,1,2,...  - specific GPU number"
    echo "  auto       - automatically select GPU with most free memory"
    echo ""
    echo "CONCEPT options:"
    echo "  nudity, violence, harassment, hate, shocking, illegal, selfharm, all"
    exit 1
fi

GPU_ARG=$1
CONCEPT=$2

# Handle auto GPU selection
if [ "${GPU_ARG}" = "auto" ]; then
    GPU=$(find_free_gpu)
    echo "Auto-selected GPU: ${GPU}"
else
    GPU=${GPU_ARG}
fi

export CUDA_VISIBLE_DEVICES=${GPU}

# ============================================================================
# Paths
# ============================================================================
BASE_DIR="/mnt/home/yhgil99/unlearning"
SAFREE_DIR="${BASE_DIR}/SAFREE"
SD_MODEL="CompVis/stable-diffusion-v1-4"
I2P_PROMPT_DIR="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/i2p"
OUTPUT_BASE="${BASE_DIR}/SoftDelete+CG/scg_outputs/best_configs"

# Fixed SAFREE parameters
SAFREE_ALPHA=0.01
SVF_UP_T=10

# Generation parameters
NSAMPLES=1
NUM_INFERENCE_STEPS=50
SEED=42
CFG_SCALE=7.5

# ============================================================================
# Concept configurations: class_type|step|cgs|st_start|st_end|hs|bgs|prompt_prefix
# ============================================================================
declare -A CONCEPT_CONFIGS
CONCEPT_CONFIGS["nudity"]="4class|17100|5.0|0.5|0.3|0.5|2.0|sexual"
CONCEPT_CONFIGS["violence"]="13class|28400|5.0|0.7|0.3|1.0|0.0|violence"
CONCEPT_CONFIGS["harassment"]="9class|24300|10.0|0.5|0.3|1.0|2.0|harassment"
CONCEPT_CONFIGS["hate"]="9class|20800|7.5|0.5|0.5|1.5|0.0|hate"
CONCEPT_CONFIGS["shocking"]="9class|23700|7.5|0.3|0.5|1.5|0.0|shocking"
CONCEPT_CONFIGS["illegal"]="9class|22600|10.0|0.5|0.5|1.5|2.0|illegal_activity"
CONCEPT_CONFIGS["selfharm"]="9class|20700|5.0|0.3|0.7|0.5|0.0|self-harm"

# ============================================================================
# Generate function
# ============================================================================
generate_concept() {
    local concept=$1

    # Parse config
    IFS='|' read -r class_type step cgs st_start st_end hs bgs prompt_prefix <<< "${CONCEPT_CONFIGS[$concept]}"

    echo ""
    echo "=============================================="
    echo "[SAFREE+Ours] ${concept^^} (${class_type})"
    echo "=============================================="
    echo "Params: cgs=${cgs}, st=${st_start}-${st_end}, hs=${hs}, bgs=${bgs}"

    # Set paths based on class type
    if [ "$class_type" == "4class" ]; then
        CLASSIFIER_CKPT="${BASE_DIR}/SoftDelete+CG/work_dirs/nudity_4class_safe_combined/checkpoint/step_${step}/classifier.pth"
        GRADCAM_STATS_DIR="${BASE_DIR}/SoftDelete+CG/gradcam_stats/nudity_4class"
        PYTHON_SCRIPT="generate_safree_spatial_cg.py"
        CATEGORY_ARG=""  # nudity script doesn't need category
    elif [ "$class_type" == "13class" ]; then
        CLASSIFIER_CKPT="${BASE_DIR}/SoftDelete+CG/work_dirs/violence_13class/checkpoint/step_${step}/classifier.pth"
        GRADCAM_STATS_DIR="${BASE_DIR}/SoftDelete+CG/gradcam_stats/violence_13class_step28400"
        PYTHON_SCRIPT="generate_safree_violence_13class_spatial_cg.py"
        CATEGORY_ARG=""  # violence 13class script doesn't need category
    else
        CLASSIFIER_CKPT="${BASE_DIR}/SoftDelete+CG/work_dirs/${concept}_9class/checkpoint/step_${step}/classifier.pth"
        GRADCAM_STATS_DIR="${BASE_DIR}/SoftDelete+CG/gradcam_stats/${concept}_9class_step${step}"
        PYTHON_SCRIPT="generate_safree_9class_spatial_cg.py"
        CATEGORY_ARG="--category ${concept}"
    fi

    # Check classifier
    if [ ! -f "$CLASSIFIER_CKPT" ]; then
        echo "Error: Classifier not found: $CLASSIFIER_CKPT"
        return 1
    fi

    # GradCAM flag
    GRADCAM_FLAG=""
    if [ -d "$GRADCAM_STATS_DIR" ]; then
        GRADCAM_FLAG="--gradcam_stats_dir ${GRADCAM_STATS_DIR}"
    else
        echo "Warning: GradCAM stats not found, using per-image normalization"
    fi

    # Generate for high_tox and low_tox
    for tox in "high_tox" "low_tox"; do
        PROMPT_FILE="${I2P_PROMPT_DIR}/${prompt_prefix}_${tox}.txt"
        OUTPUT_DIR="${OUTPUT_BASE}/safree_ours_${concept}_${class_type}/${tox}"

        if [ ! -f "$PROMPT_FILE" ]; then
            echo "[SKIP] Prompt file not found: $PROMPT_FILE"
            continue
        fi

        # Count existing images
        EXPECTED_IMAGES=$(wc -l < "$PROMPT_FILE")
        EXISTING_IMAGES=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)

        # Skip only if all images exist
        if [ "$EXISTING_IMAGES" -ge "$EXPECTED_IMAGES" ]; then
            echo "[SKIP] Complete (${EXISTING_IMAGES}/${EXPECTED_IMAGES}): $OUTPUT_DIR"
            continue
        elif [ "$EXISTING_IMAGES" -gt 0 ]; then
            echo "[PARTIAL] Found ${EXISTING_IMAGES}/${EXPECTED_IMAGES} images, regenerating: $OUTPUT_DIR"
            rm -rf "$OUTPUT_DIR"
        fi

        echo ""
        echo "[${concept} ${tox}]"
        echo "Prompt: ${PROMPT_FILE}"
        echo "Output: ${OUTPUT_DIR}"

        mkdir -p "$OUTPUT_DIR"
        cd "${SAFREE_DIR}"

        python ${PYTHON_SCRIPT} \
            --ckpt_path "${SD_MODEL}" \
            --prompt_file "${PROMPT_FILE}" \
            --output_dir "${OUTPUT_DIR}" \
            --classifier_ckpt "${CLASSIFIER_CKPT}" \
            ${GRADCAM_FLAG} \
            ${CATEGORY_ARG} \
            --nsamples ${NSAMPLES} \
            --cfg_scale ${CFG_SCALE} \
            --num_inference_steps ${NUM_INFERENCE_STEPS} \
            --seed ${SEED} \
            --safree \
            --safree_alpha ${SAFREE_ALPHA} \
            --svf \
            --svf_up_t ${SVF_UP_T} \
            --spatial_cg \
            --cg_guidance_scale ${cgs} \
            --spatial_threshold_start ${st_start} \
            --spatial_threshold_end ${st_end} \
            --threshold_strategy "linear_decrease" \
            --harmful_scale ${hs} \
            --base_guidance_scale ${bgs} \
            --skip_if_safe

        echo "Done: ${OUTPUT_DIR}"
    done
}

# ============================================================================
# Main
# ============================================================================
echo "=============================================="
echo "SAFREE+Ours Best Configs Generation"
echo "=============================================="
echo "GPU: ${GPU}"
echo "Concept: ${CONCEPT}"
echo "Output: ${OUTPUT_BASE}"
echo "=============================================="

ALL_CONCEPTS=("nudity" "violence" "harassment" "hate" "shocking" "illegal" "selfharm")

if [ "$CONCEPT" == "all" ]; then
    for c in "${ALL_CONCEPTS[@]}"; do
        generate_concept "$c"
    done
else
    if [[ -z "${CONCEPT_CONFIGS[$CONCEPT]}" ]]; then
        echo "Error: Invalid concept '$CONCEPT'"
        echo "Available: nudity, violence, harassment, hate, shocking, illegal, selfharm, all"
        exit 1
    fi
    generate_concept "$CONCEPT"
fi

echo ""
echo "=============================================="
echo "Generation Complete!"
echo "=============================================="
echo "Output: ${OUTPUT_BASE}"
echo "=============================================="
