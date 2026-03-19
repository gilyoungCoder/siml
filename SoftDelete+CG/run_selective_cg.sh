#!/usr/bin/env bash
set -euo pipefail

# =========================
# Selective Classifier Guidance for Machine Unlearning
#
# Key Innovation:
#   - Apply guidance ONLY when harmful content detected (selective)
#   - Spatially-aware guidance using Grad-CAM
#   - Minimal intervention on benign prompts → Better GENEVAL scores
#
# Benefits vs Previous Methods:
#   1. generate_classifier_masked.py: Applies masking at ALL steps
#      → Can degrade benign prompt quality
#
#   2. This approach: Selective intervention
#      → Harmful detected: Apply spatially-masked guidance
#      → Safe content: No intervention (vanilla diffusion)
# =========================

# GPU Configuration
export CUDA_VISIBLE_DEVICES=5

# =========================
# Model & I/O Configuration
# =========================
CKPT_PATH="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="./prompts/sexual_50.txt"
OUTPUT_DIR="./scg_outputs/sexual/selective_cg_v20,1,0+CG"

# =========================
# Generation Parameters
# =========================
NSAMPLES=1
CFG_SCALE=7.5
NUM_INFERENCE_STEPS=50
SEED=1234

# =========================
# Selective Classifier Guidance ⭐
# =========================
SELECTIVE_GUIDANCE=true

# Classifier checkpoint
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
# CLASSIFIER_CKPT="./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth"
# Detection threshold
# - For logits: 0.0 (logit > 0 means harmful)
# - For probabilities: 0.5 (prob > 0.5 means harmful)
# - Higher = more conservative (less guidance applied)
# - Lower = more aggressive (more guidance applied)
HARMFUL_THRESHOLD=0

# Class configuration
HARMFUL_CLASS=2  # 2 = nude people
SAFE_CLASS=1     # 1 = clothed people (guidance target)

# =========================
# Spatial Masking (Grad-CAM)
# =========================
# Strategy 1: Fixed threshold
SPATIAL_THRESHOLD=0
USE_PERCENTILE=false

# Strategy 2: Percentile-based (recommended)
# USE_PERCENTILE=true
# SPATIAL_PERCENTILE=0.3  # Mask top 30% of attention regions

# =========================
# Guidance Parameters
# =========================
GUIDANCE_SCALE=20

# Bidirectional Guidance (NEW!)
# - Pull toward safe class (clothed)
# - Push away from harmful class (nude)
USE_BIDIRECTIONAL=true
HARMFUL_SCALE=1.0  # Relative scale for harmful repulsion (1.0 = equal weight)

GUIDANCE_START_STEP=0
GUIDANCE_END_STEP=50

# Grad-CAM layer (bottleneck recommended for semantic features)
GRADCAM_LAYER="encoder_model.middle_block.2"

# =========================
# GENERAL Classifier Guidance ⭐ (NEW: Always-on CG)
# =========================
# Difference from Selective CG:
# - Selective CG: Only applies when harmful detected + spatial masking
# - General CG: Always applies to entire latent (global guidance)
# - Can be combined for maximum effect!

GENERAL_CG=true                      # Enable general (always-on) CG
GENERAL_CG_SCALE=10                  # Gradient scale for general CG
GENERAL_CG_HARMFUL_SCALE=1.0         # Harmful repulsion scale
GENERAL_CG_USE_BIDIRECTIONAL=true    # Bidirectional (pull to safe + push from harmful)
GENERAL_CG_START_STEP=0              # Start step
GENERAL_CG_END_STEP=50               # End step

# Combined Effect:
# 1. General CG: Global guidance on all steps
# 2. Selective CG: Additional spatial guidance when harmful detected
# → Double protection!

# =========================
# Debug & Visualization
# =========================
DEBUG=true
SAVE_VISUALIZATIONS=true

# =========================
# Run Generation
# =========================
mkdir -p "${OUTPUT_DIR}"
LOG="./logs/run_selective_cg_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG}")"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║            SELECTIVE CLASSIFIER GUIDANCE FOR MACHINE UNLEARNING                ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[CONFIG] Model: ${CKPT_PATH}"
echo "[CONFIG] Prompts: ${PROMPT_FILE}"
echo "[CONFIG] Output: ${OUTPUT_DIR}"
echo "[CONFIG] Log: ${LOG}"
echo ""
echo "┌─ Selective Guidance Configuration ────────────────────────────────────────────┐"
echo "│  Enabled: ${SELECTIVE_GUIDANCE}"
echo "│  "
echo "│  Detection:"
echo "│    Classifier: ${CLASSIFIER_CKPT}"
echo "│    Harmful threshold: ${HARMFUL_THRESHOLD}"
echo "│    Harmful class: ${HARMFUL_CLASS} (nude)"
echo "│    Safe class: ${SAFE_CLASS} (clothed)"
echo "│  "
echo "│  Spatial Masking (Grad-CAM):"
if [ "${USE_PERCENTILE}" = true ]; then
echo "│    Strategy: Percentile-based"
echo "│    Percentile: ${SPATIAL_PERCENTILE} (top ${SPATIAL_PERCENTILE}% regions)"
else
echo "│    Strategy: Fixed threshold"
echo "│    Threshold: ${SPATIAL_THRESHOLD}"
fi
echo "│    Grad-CAM layer: ${GRADCAM_LAYER}"
echo "│  "
echo "│  Guidance:"
echo "│    Mode: $([ "${USE_BIDIRECTIONAL}" = true ] && echo "Bidirectional (pull+push)" || echo "Unidirectional (pull only)")"
echo "│    Guidance scale: ${GUIDANCE_SCALE}"
if [ "${USE_BIDIRECTIONAL}" = true ]; then
echo "│    Harmful scale: ${HARMFUL_SCALE} (repulsion strength)"
fi
echo "│    Active steps: ${GUIDANCE_START_STEP} → ${GUIDANCE_END_STEP}"
echo "│  "
echo "│  How it works:"
echo "│    1. Each denoising step: Monitor latent with classifier"
echo "│    2. If harmful_score > threshold:"
echo "│         → Compute Grad-CAM heatmap for harmful regions"
echo "│         → Apply classifier gradient toward safe class"
echo "│         → Mask gradient to harmful regions only"
echo "│    3. If harmful_score <= threshold:"
echo "│         → Skip guidance (vanilla diffusion)"
echo "│  "
echo "│  Benefits:"
echo "│    ✓ Benign prompts: Minimal intervention → Better GENEVAL"
echo "│    ✓ Harmful prompts: Targeted suppression"
echo "│    ✓ Spatial precision via Grad-CAM"
echo "│    ✓ Reduced computational cost (selective application)"
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─ GENERAL Classifier Guidance ⭐ (Always-on) ──────────────────────────────────┐"
echo "│  Enabled: ${GENERAL_CG}"
if [ "${GENERAL_CG}" = true ]; then
echo "│  Mode: $([ "${GENERAL_CG_USE_BIDIRECTIONAL}" = true ] && echo "Bidirectional (pull+push)" || echo "Unidirectional (pull only)")"
echo "│  Scale: ${GENERAL_CG_SCALE}"
if [ "${GENERAL_CG_USE_BIDIRECTIONAL}" = true ]; then
echo "│  Harmful repulsion scale: ${GENERAL_CG_HARMFUL_SCALE}"
fi
echo "│  Active steps: ${GENERAL_CG_START_STEP} → ${GENERAL_CG_END_STEP}"
echo "│  "
echo "│  Difference from Selective CG:"
echo "│    - Selective: Applies only when harmful detected (conditional)"
echo "│    - General: Always applies to all steps (unconditional)"
echo "│    → Can be combined for double protection!"
fi
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─ Generation Parameters ───────────────────────────────────────────────────────┐"
echo "│  Inference steps: ${NUM_INFERENCE_STEPS}"
echo "│  CFG scale: ${CFG_SCALE}"
echo "│  Samples per prompt: ${NSAMPLES}"
echo "│  Seed: ${SEED}"
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─ Debug & Visualization ───────────────────────────────────────────────────────┐"
echo "│  Debug mode: ${DEBUG}"
echo "│  Save visualizations: ${SAVE_VISUALIZATIONS}"
if [ "${SAVE_VISUALIZATIONS}" = true ]; then
echo "│    → Will save Grad-CAM and guidance statistics"
echo "│    → Location: ${OUTPUT_DIR}/visualizations/"
fi
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""

# Build command arguments
ARGS=(
    "${CKPT_PATH}"
    --prompt_file "${PROMPT_FILE}"
    --output_dir "${OUTPUT_DIR}"
    --nsamples "${NSAMPLES}"
    --cfg_scale "${CFG_SCALE}"
    --num_inference_steps "${NUM_INFERENCE_STEPS}"
    --seed "${SEED}"
)

# Add selective guidance arguments
if [ "${SELECTIVE_GUIDANCE}" = true ]; then
    ARGS+=(
        --selective_guidance
        --classifier_ckpt "${CLASSIFIER_CKPT}"
        --harmful_threshold "${HARMFUL_THRESHOLD}"
        --harmful_class "${HARMFUL_CLASS}"
        --safe_class "${SAFE_CLASS}"
        --spatial_threshold "${SPATIAL_THRESHOLD}"
        --guidance_scale "${GUIDANCE_SCALE}"
        --harmful_scale "${HARMFUL_SCALE}"
        --guidance_start_step "${GUIDANCE_START_STEP}"
        --guidance_end_step "${GUIDANCE_END_STEP}"
        --gradcam_layer "${GRADCAM_LAYER}"
    )

    if [ "${USE_BIDIRECTIONAL}" = true ]; then
        ARGS+=(--use_bidirectional)
    fi

    if [ "${USE_PERCENTILE}" = true ]; then
        ARGS+=(
            --use_percentile
            --spatial_percentile "${SPATIAL_PERCENTILE}"
        )
    fi
fi

# Add GENERAL CG arguments
if [ "${GENERAL_CG}" = true ]; then
    ARGS+=(
        --general_cg
        --general_cg_scale "${GENERAL_CG_SCALE}"
        --general_cg_harmful_scale "${GENERAL_CG_HARMFUL_SCALE}"
        --general_cg_start_step "${GENERAL_CG_START_STEP}"
        --general_cg_end_step "${GENERAL_CG_END_STEP}"
    )

    if [ "${GENERAL_CG_USE_BIDIRECTIONAL}" = true ]; then
        ARGS+=(--general_cg_use_bidirectional)
    fi
fi

# Add debug flags
if [ "${DEBUG}" = true ]; then
    ARGS+=(--debug)
fi

if [ "${SAVE_VISUALIZATIONS}" = true ]; then
    ARGS+=(--save_visualizations)
fi

echo "[INFO] Starting generation with SELECTIVE CLASSIFIER GUIDANCE..."
echo "[INFO] Process will run in background. Monitor with: tail -f \"${LOG}\""
echo ""

# Run in background and save log
nohup python generate_selective_cg.py "${ARGS[@]}" > "${LOG}" 2>&1 &
PID=$!

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║  Process started with PID: ${PID}"
echo "║"
echo "║  Monitor logs:    tail -f \"${LOG}\""
echo "║  Stop process:    kill ${PID}"
echo "║  Check progress:  ls -lh \"${OUTPUT_DIR}\""
echo "║"
echo "║  Expected outputs:"
echo "║    - Generated images: ${OUTPUT_DIR}/*.png"
echo "║    - Visualizations:   ${OUTPUT_DIR}/visualizations/*.png"
echo "║    - Statistics in log file"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[TIP] Compare with baselines:"
echo "  1. Vanilla SD:        Set SELECTIVE_GUIDANCE=false"
echo "  2. Always-on masking: Use generate_classifier_masked.sh"
echo "  3. This (Selective):  Current configuration"
echo ""
