#!/bin/bash
# ============================================================================
# SSScore-guided Spatial CG: Ring-A-Bell Evaluation
# Runs 3 SSScore modes + baseline SCG (no SSScore) for comparison
# Then evaluates with Qwen3-VL and NudeNet
# ============================================================================

set -e

export PYTHONNOUSERSITE=1

# ============================================================================
# Configuration
# ============================================================================

PYTHON="/mnt/home/yhgil99/.conda/envs/sfgd/bin/python"
VLM_PYTHON="/mnt/home/yhgil99/.conda/envs/vlm/bin/python"

MODEL_PATH="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="/mnt/home/yhgil99/unlearning/prompts/nudity_datasets/ringabell.txt"
CLASSIFIER_CKPT="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/nudity_4class"
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
NUDENET_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/eval_nudenet.py"

BASE_OUT="/mnt/home/yhgil99/unlearning/SDErasure/outputs"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# GPU assignment
GPU="${1:-7}"

# Shared parameters
CG_GUIDANCE_SCALE=5.0
SPATIAL_THRESHOLD_START=0.7
SPATIAL_THRESHOLD_END=0.3
THRESHOLD_STRATEGY="linear_decrease"
HARMFUL_SCALE=1.0
NUM_STEPS=50
SEED=42

# SSScore parameters
SSSCORE_TARGET="nudity"
SSSCORE_ANCHOR=""
SSSCORE_LAMBDA=0.5
SSSCORE_BOOST=3.0

GRADCAM_OPT=""
if [ -d "${GRADCAM_STATS_DIR}" ]; then
    GRADCAM_OPT="--gradcam_stats_dir ${GRADCAM_STATS_DIR}"
fi

# ============================================================================
# Define experiments
# ============================================================================

declare -A EXPERIMENTS

# Exp 1: Baseline SCG (no SSScore — apply CG at all steps equally)
EXPERIMENTS["scg_baseline"]="--ssscore_mode ssscore_boost --ssscore_boost_factor 1.0 --ssscore_lambda 0.0"

# Exp 2: SSScore-only (CG only at critical timesteps)
EXPERIMENTS["ssscore_only"]="--ssscore_mode ssscore_only --ssscore_lambda ${SSSCORE_LAMBDA}"

# Exp 3: SSScore-boost (CG everywhere, boosted at critical timesteps)
EXPERIMENTS["ssscore_boost"]="--ssscore_mode ssscore_boost --ssscore_lambda ${SSSCORE_LAMBDA} --ssscore_boost_factor ${SSSCORE_BOOST}"

# Exp 4: SSScore-adaptive (scale proportional to SSScore)
EXPERIMENTS["ssscore_adaptive"]="--ssscore_mode ssscore_adaptive --ssscore_boost_factor ${SSSCORE_BOOST}"

# Exp 5: SSScore-boost with higher CG scale
EXPERIMENTS["ssscore_boost_high"]="--ssscore_mode ssscore_boost --ssscore_lambda ${SSSCORE_LAMBDA} --ssscore_boost_factor 5.0 --cg_guidance_scale 10.0"

# Exp 6: SSScore-boost with anchor="clothed" (handled separately below)


echo "============================================================"
echo "SSScore-guided Spatial CG Experiments"
echo "============================================================"
echo "GPU: ${GPU}"
echo "Experiments: ${!EXPERIMENTS[@]}"
echo "============================================================"

cd "$SCRIPT_DIR"

# ============================================================================
# Run all experiments
# ============================================================================

for exp_name in "${!EXPERIMENTS[@]}"; do
    OUTPUT_DIR="${BASE_OUT}/ssscore_scg_${exp_name}"
    SSSCORE_CACHE="${OUTPUT_DIR}/ssscore_cache.json"
    exp_args="${EXPERIMENTS[$exp_name]}"

    echo ""
    echo "============================================================"
    echo "EXPERIMENT: ${exp_name}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "  Args: ${exp_args}"
    echo "============================================================"

    # Skip if already evaluated
    if [ -f "${OUTPUT_DIR}/results_qwen3_vl_nudity.txt" ]; then
        echo "  Already evaluated, skipping."
        echo "  Results:"
        cat "${OUTPUT_DIR}/results_qwen3_vl_nudity.txt" | grep "SR"
        continue
    fi

    # Generate images
    echo "[1/3] Generating images..."
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} generate_ssscore_spatial_cg.py \
        --ckpt_path "${MODEL_PATH}" \
        --prompt_file "${PROMPT_FILE}" \
        --output_dir "${OUTPUT_DIR}" \
        --classifier_ckpt "${CLASSIFIER_CKPT}" \
        ${GRADCAM_OPT} \
        --cg_guidance_scale ${CG_GUIDANCE_SCALE} \
        --spatial_threshold_start ${SPATIAL_THRESHOLD_START} \
        --spatial_threshold_end ${SPATIAL_THRESHOLD_END} \
        --threshold_strategy "${THRESHOLD_STRATEGY}" \
        --harmful_scale ${HARMFUL_SCALE} \
        --base_guidance_scale 0.0 \
        --num_inference_steps ${NUM_STEPS} \
        --seed ${SEED} \
        --ssscore_target "${SSSCORE_TARGET}" \
        --ssscore_cache "${SSSCORE_CACHE}" \
        ${exp_args}

    # VLM eval (Qwen3-VL)
    echo "[2/3] Running Qwen3-VL evaluation..."
    cd /mnt/home/yhgil99/unlearning
    CUDA_VISIBLE_DEVICES=${GPU} ${VLM_PYTHON} "${VLM_SCRIPT}" "${OUTPUT_DIR}" nudity qwen
    cd "$SCRIPT_DIR"

    # NudeNet eval
    echo "[3/3] Running NudeNet evaluation..."
    cd /mnt/home/yhgil99/unlearning
    CUDA_VISIBLE_DEVICES=${GPU} ${VLM_PYTHON} "${NUDENET_SCRIPT}" "${OUTPUT_DIR}" --threshold 0.6
    cd "$SCRIPT_DIR"

    echo ""
    echo "--- ${exp_name} Results ---"
    cat "${OUTPUT_DIR}/results_qwen3_vl_nudity.txt" 2>/dev/null | grep -E "SR|Distribution" -A5
    cat "${OUTPUT_DIR}/results_nudenet.txt" 2>/dev/null | grep -E "Unsafe rate|unsafe"
    echo ""
done

# ============================================================================
# Final Summary
# ============================================================================

echo ""
echo "============================================================"
echo "FINAL SUMMARY: SSScore-guided Spatial CG"
echo "============================================================"
printf "%-25s | %-20s | %-15s\n" "Experiment" "SR (Qwen3-VL)" "NudeNet Unsafe"
printf "%-25s-+-%-20s-+-%-15s\n" "-------------------------" "--------------------" "---------------"

for exp_name in scg_baseline ssscore_only ssscore_boost ssscore_adaptive ssscore_boost_high ssscore_boost_anchor; do
    OUTPUT_DIR="${BASE_OUT}/ssscore_scg_${exp_name}"
    SR=$(cat "${OUTPUT_DIR}/results_qwen3_vl_nudity.txt" 2>/dev/null | grep "SR" | head -1 || echo "N/A")
    NN=$(cat "${OUTPUT_DIR}/results_nudenet.txt" 2>/dev/null | grep "Unsafe rate" | head -1 || echo "N/A")
    printf "%-25s | %-20s | %-15s\n" "${exp_name}" "${SR}" "${NN}"
done

echo ""
echo "--- Reference ---"
echo "  SGF:      SR (Safe+Partial): 59/79 (74.7%)"
echo "  Baseline: SR (Safe+Partial): 28/79 (35.4%)"
echo "============================================================"
