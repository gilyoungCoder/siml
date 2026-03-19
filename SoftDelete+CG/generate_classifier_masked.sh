#!/usr/bin/env bash
set -euo pipefail

# =========================
# Classifier-Guided Output Masking for Machine Unlearning
# Novel approach: Suppress cross-attention output at nude spatial positions
# =========================

# GPU Configuration
export CUDA_VISIBLE_DEVICES=7

# Model & I/O
CKPT_PATH="CompVis/stable-diffusion-v1-4"
PROMPT_FILE="./prompts/violence_50.txt"
OUTPUT_DIR="./outputs/violence/classifier_masked_adversarial_49+CG_percentile0.5+CG_hard"

# Generation Parameters
NSAMPLES=1
CFG_SCALE=5.0
NUM_INFERENCE_STEPS=50
SEED=1234

# =========================
# Harmful Concept Suppression (Attention Score Manipulation)
# =========================
HARM_SUPPRESS=false
HARM_CONCEPTS_FILE="./configs/harm_concepts.txt"
BASE_TAU=0.15                    # Base threshold for attention suppression
HARM_GAMMA_START=60              # Initial suppression strength
HARM_GAMMA_END=0.5               # Final suppression strength

# =========================
# Classifier-Guided Output Masking ⭐ (NEW!)
# =========================
VALUE_MASKING=true               # Enable classifier-guided output masking
MASK_STRATEGY="hard"      # hard, soft, or adversarial
MASK_THRESHOLD=0.5               # Threshold for binary mask (hard/soft)
MASK_STRENGTH=0.5                # Strength for soft masking (0-1)
MASK_START_STEP=0                # Start masking from this step
MASK_END_STEP=49                 # End masking at this step (stop before final detail refinement)

# Alternative: Percentile-based masking
USE_PERCENTILE=true             # Use percentile instead of fixed threshold
MASK_PERCENTILE=0.3              # Mask top 30% attention regions

# Masking Strategies:
# - "hard": Binary mask (0 or 1) based on threshold
#   - nude regions → 0 (completely suppressed)
#   - safe regions → 1 (unmodified)
#
# - "soft": Weighted mask based on heatmap intensity
#   - output = output * (1 - mask_strength * nude_mask)
#   - Gradual suppression proportional to nude confidence
#
# - "adversarial": Can go negative (experimental)
#   - Actively pushes away from nude features

# =========================
# OLD Classifier Guidance (GuidanceModel wrapper - can be combined)
# =========================
CLASSIFIER_GUIDANCE=false
CLASSIFIER_CONFIG="./configs/models/time_dependent_discriminator.yaml"
# CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
CLASSIFIER_CKPT="./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth"
GUIDANCE_SCALE=15
GUIDANCE_START_STEP=1
TARGET_CLASS=1  # 1 = clothed people

# =========================
# GENERAL Classifier Guidance ⭐ (NEW: Direct gradient-based approach)
# =========================
GENERAL_CG=true                  # Enable general classifier guidance
GENERAL_CG_SCALE=10              # Gradient scale for general CG
GENERAL_CG_SAFE_CLASS=1          # 1 = clothed people (safe target)
GENERAL_CG_HARMFUL_CLASS=2       # 2 = nude/violent people (to avoid)
GENERAL_CG_USE_BIDIRECTIONAL=true  # Enable bidirectional guidance (pull to safe + push from harmful)
GENERAL_CG_HARMFUL_SCALE=1.0     # Harmful repulsion scale (relative to GENERAL_CG_SCALE)
GENERAL_CG_START_STEP=0          # Step to start general CG
GENERAL_CG_END_STEP=50           # Step to end general CG

# Benefits of General CG:
# - Simpler and more direct than GuidanceModel wrapper
# - Bidirectional guidance: pull toward safe + push from harmful
# - More flexible and easier to tune
# - Can be combined with value masking

# =========================
# DEBUG Options
# =========================
DEBUG=true              # Enable general debug mode
DEBUG_PROMPTS=true      # Show per-token analysis
DEBUG_STEPS=true        # Show per-step masking statistics

# =========================
# Run Generation
# =========================
mkdir -p "${OUTPUT_DIR}"
LOG="./logs/run_classifier_masked_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG}")"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║         MACHINE UNLEARNING - CLASSIFIER-GUIDED OUTPUT MASKING ⭐              ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[CONFIG] Model: ${CKPT_PATH}"
echo "[CONFIG] Prompts: ${PROMPT_FILE}"
echo "[CONFIG] Output: ${OUTPUT_DIR}"
echo "[CONFIG] Log: ${LOG}"
echo ""
echo "┌─ Harmful Concept Suppression (Attention Scores) ─────────────────────────────┐"
echo "│  Enabled: ${HARM_SUPPRESS}"
echo "│  Concepts file: ${HARM_CONCEPTS_FILE}"
echo "│  Base τ: ${BASE_TAU}"
echo "│  Gamma schedule: ${HARM_GAMMA_START} → ${HARM_GAMMA_END}"
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─ Classifier-Guided Output Masking ⭐ (NEW!) ──────────────────────────────────┐"
echo "│  Enabled: ${VALUE_MASKING}"
echo "│  Strategy: ${MASK_STRATEGY}"
if [ "${USE_PERCENTILE}" = true ]; then
    echo "│  Masking: top ${MASK_PERCENTILE}% (percentile-based)"
else
    echo "│  Threshold: ${MASK_THRESHOLD}"
fi
echo "│  Strength: ${MASK_STRENGTH} (for soft masking)"
echo "│  Active steps: ${MASK_START_STEP} → ${MASK_END_STEP}"
echo "│  "
echo "│  How it works:"
echo "│    1. Use Grad-CAM to identify nude regions in latent space"
echo "│    2. Create spatial mask at different resolutions (64×64, 32×32, 16×16, 8×8)"
echo "│    3. Suppress cross-attention OUTPUT at nude positions"
echo "│    4. Prevents text semantics from being injected into nude regions"
echo "│  "
echo "│  Why OUTPUT masking?"
echo "│    - Query: latent spatial positions [B, 4096, dim]"
echo "│    - Key/Value: text tokens [B, 77, dim]"
echo "│    - Output: attention result injected back to latent [B, 4096, dim]"
echo "│    → Masking output directly controls what gets injected where!"
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─ OLD Classifier Guidance (GuidanceModel) ─────────────────────────────────────┐"
echo "│  Enabled: ${CLASSIFIER_GUIDANCE}"
echo "│  Checkpoint: ${CLASSIFIER_CKPT}"
echo "│  Scale: ${GUIDANCE_SCALE}"
echo "│  Target class: ${TARGET_CLASS} (clothed people)"
echo "│  Start step: ${GUIDANCE_START_STEP}"
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─ GENERAL Classifier Guidance ⭐ (Direct Gradient) ────────────────────────────┐"
echo "│  Enabled: ${GENERAL_CG}"
if [ "${GENERAL_CG}" = true ]; then
echo "│  Mode: $([ "${GENERAL_CG_USE_BIDIRECTIONAL}" = true ] && echo "Bidirectional (pull+push)" || echo "Unidirectional (pull only)")"
echo "│  Scale: ${GENERAL_CG_SCALE}"
echo "│  Safe class: ${GENERAL_CG_SAFE_CLASS} (clothed people)"
echo "│  Harmful class: ${GENERAL_CG_HARMFUL_CLASS} (nude/violent)"
if [ "${GENERAL_CG_USE_BIDIRECTIONAL}" = true ]; then
echo "│  Harmful repulsion scale: ${GENERAL_CG_HARMFUL_SCALE}"
fi
echo "│  Active steps: ${GENERAL_CG_START_STEP} → ${GENERAL_CG_END_STEP}"
fi
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─ Debug Options ───────────────────────────────────────────────────────────────┐"
echo "│  General debug: ${DEBUG}"
echo "│  Per-prompt token analysis: ${DEBUG_PROMPTS}"
echo "│  Per-step masking statistics: ${DEBUG_STEPS}"
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

# Add harmful concept suppression arguments
if [ "${HARM_SUPPRESS}" = true ]; then
    ARGS+=(
        --harm_suppress
        --harm_concepts_file "${HARM_CONCEPTS_FILE}"
        --base_tau "${BASE_TAU}"
        --harm_gamma_start "${HARM_GAMMA_START}"
        --harm_gamma_end "${HARM_GAMMA_END}"
    )
fi

# Add classifier-guided value masking arguments
if [ "${VALUE_MASKING}" = true ]; then
    ARGS+=(
        --value_masking
        --mask_strategy "${MASK_STRATEGY}"
        --mask_threshold "${MASK_THRESHOLD}"
        --mask_strength "${MASK_STRENGTH}"
        --mask_start_step "${MASK_START_STEP}"
        --mask_end_step "${MASK_END_STEP}"
    )

    if [ "${USE_PERCENTILE}" = true ]; then
        ARGS+=(
            --use_percentile
            --mask_percentile "${MASK_PERCENTILE}"
        )
    fi
fi

# Add OLD classifier guidance arguments (GuidanceModel)
if [ "${CLASSIFIER_GUIDANCE}" = true ]; then
    ARGS+=(
        --classifier_guidance
        --classifier_config "${CLASSIFIER_CONFIG}"
        --classifier_ckpt "${CLASSIFIER_CKPT}"
        --guidance_scale "${GUIDANCE_SCALE}"
        --guidance_start_step "${GUIDANCE_START_STEP}"
        --target_class "${TARGET_CLASS}"
    )
fi

# Add GENERAL classifier guidance arguments (Direct gradient-based)
if [ "${GENERAL_CG}" = true ]; then
    ARGS+=(
        --general_cg
        --general_cg_scale "${GENERAL_CG_SCALE}"
        --general_cg_safe_class "${GENERAL_CG_SAFE_CLASS}"
        --general_cg_harmful_class "${GENERAL_CG_HARMFUL_CLASS}"
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

if [ "${DEBUG_PROMPTS}" = true ]; then
    ARGS+=(--debug_prompts)
fi

if [ "${DEBUG_STEPS}" = true ]; then
    ARGS+=(--debug_steps)
fi

echo "[INFO] Starting generation with CLASSIFIER-GUIDED OUTPUT MASKING..."
echo "[INFO] Process will run in background. Monitor with: tail -f \"${LOG}\""
echo ""

# Run in background and save log
nohup python generate_classifier_masked.py "${ARGS[@]}" > "${LOG}" 2>&1 &
PID=$!

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║  Process started with PID: ${PID}"
echo "║"
echo "║  Monitor logs:  tail -f \"${LOG}\""
echo "║  Stop process:  kill ${PID}"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
