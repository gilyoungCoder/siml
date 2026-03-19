#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CAS + Spatial CFG: Training-Free Safe Generation
# =============================================================================
# WHEN: CAS (Concept Alignment Score) - cosine(d_prompt, d_target) > threshold
# WHERE: Spatial CFG - target vs anchor noise direction difference
# HOW: Anchor shift / Target negate / Dual guidance
# =============================================================================

# GPU Configuration (GPU 7 is reserved, do NOT use it)
export CUDA_VISIBLE_DEVICES=${GPU:-0}

# Model
CKPT_PATH="CompVis/stable-diffusion-v1-4"

# --- Choose prompt set ---
# For harmful prompts (I2P sexual subset):
PROMPT_FILE="./prompts/sexual.csv"
# For benign prompts (COCO, to verify no false positives):
# PROMPT_FILE="./prompts/coco_30.txt"

# --- Output ---
EXPERIMENT_NAME="${1:-default}"
OUTPUT_DIR="./scg_outputs/cas_spatial_cfg/${EXPERIMENT_NAME}"

# --- Generation ---
NSAMPLES=4
CFG_SCALE=7.5
NUM_INFERENCE_STEPS=50
SEED=42

# =============================================================================
# CAS (WHEN) Parameters
# =============================================================================
CAS_THRESHOLD=0.3        # Trigger when cos(d_prompt, d_target) > this
                          # Harmful prompts: ~0.6-0.9, Safe prompts: ~0.1-0.3
CAS_STICKY="--cas_sticky" # Once triggered, keep guidance on
CAS_WINDOW_START=1000     # Check CAS from this timestep
CAS_WINDOW_END=0          # Check CAS until this timestep (0 = all steps)

# =============================================================================
# Spatial CFG (WHERE) Parameters
# =============================================================================
# Methods: diff_norm, target_strength, cosine_diff, target_projection
SPATIAL_METHOD="target_strength"
SPATIAL_THRESHOLD=0.3     # Threshold for per-pixel guidance activation
SPATIAL_SCALE_HIGH=5.0    # Guidance scale in detected unsafe regions
SPATIAL_SCALE_LOW=0.0     # Guidance scale in safe regions
MASK_BLUR_SIGMA=1.0       # Gaussian smoothing of spatial mask

# =============================================================================
# Safety Guidance (HOW) Parameters
# =============================================================================
# Modes: anchor_shift, target_negate, dual
GUIDANCE_MODE="anchor_shift"
SAFETY_SCALE=1.0          # Global guidance multiplier
GUIDANCE_SCHEDULE="constant"  # constant, linear_decay, cosine
WARMUP_STEPS=3            # Ramp up guidance in first N guided steps

# =============================================================================
# Target / Anchor Concepts
# =============================================================================
TARGET_CONCEPTS="nudity" "nude person" "nsfw person" "naked body"
ANCHOR_CONCEPTS="clothed person" "person wearing clothes" "fully dressed person"

# =============================================================================
# Run
# =============================================================================
mkdir -p "${OUTPUT_DIR}"
LOG="${OUTPUT_DIR}/run.log"

echo "============================================================"
echo "CAS + Spatial CFG: Training-Free Safe Generation"
echo "============================================================"
echo "  GPU: ${CUDA_VISIBLE_DEVICES}"
echo "  Experiment: ${EXPERIMENT_NAME}"
echo "  Prompt file: ${PROMPT_FILE}"
echo "  Output: ${OUTPUT_DIR}"
echo "  CAS threshold: ${CAS_THRESHOLD}"
echo "  Spatial method: ${SPATIAL_METHOD}"
echo "  Guidance mode: ${GUIDANCE_MODE}"
echo "============================================================"

python generate_cas_spatial_cfg.py \
    --ckpt_path "${CKPT_PATH}" \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --nsamples ${NSAMPLES} \
    --cfg_scale ${CFG_SCALE} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --seed ${SEED} \
    --cas_threshold ${CAS_THRESHOLD} \
    ${CAS_STICKY} \
    --cas_window_start ${CAS_WINDOW_START} \
    --cas_window_end ${CAS_WINDOW_END} \
    --target_concepts ${TARGET_CONCEPTS} \
    --anchor_concepts ${ANCHOR_CONCEPTS} \
    --spatial_method ${SPATIAL_METHOD} \
    --spatial_threshold ${SPATIAL_THRESHOLD} \
    --spatial_scale_high ${SPATIAL_SCALE_HIGH} \
    --spatial_scale_low ${SPATIAL_SCALE_LOW} \
    --mask_blur_sigma ${MASK_BLUR_SIGMA} \
    --adaptive_area_scale \
    --guidance_mode ${GUIDANCE_MODE} \
    --safety_scale ${SAFETY_SCALE} \
    --guidance_schedule ${GUIDANCE_SCHEDULE} \
    --warmup_steps ${WARMUP_STEPS} \
    --save_spatial_maps \
    --debug \
    2>&1 | tee "${LOG}"

echo ""
echo "Done! Results saved to: ${OUTPUT_DIR}"
echo "Stats: ${OUTPUT_DIR}/generation_stats.json"
