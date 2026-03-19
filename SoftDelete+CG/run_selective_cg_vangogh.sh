#!/bin/bash
# ============================================================================
# Selective Classifier Guidance - VanGogh Style Suppression
# Using best parameters from nudity/violence grid search as baseline
# ============================================================================
export CUDA_VISIBLE_DEVICES=4

# Activate conda environment
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate sdd

set -e

# ============================================================================
# Configuration
# ============================================================================

CKPT_PATH="CompVis/stable-diffusion-v1-4"
CLASSIFIER_PATH="./work_dirs/vangogh_three_class_diff/checkpoint/step_7200/classifier.pth"
PROMPT_FILE="./prompts/vangogh_50.txt"
OUTPUT_DIR="scg_outputs/vangogh_selective_cg,72"

# Generation parameters
NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
NSAMPLES=1
SEED=123

# ============================================================================
# Selective Classifier Guidance Parameters
# Based on best nudity parameters as baseline:
# gs=7.0, hs=1.5, st=0.3, ws=3.0→0.5, ts=-0.5→-2.5
# ============================================================================

# Guidance parameters
GUIDANCE_SCALE=6.0          # Overall guidance strength
HARMFUL_SCALE=1.5           # Harmful repulsion strength (bidirectional)
SPATIAL_THRESHOLD=0.3       # Grad-CAM threshold for spatial masking

# Weight scheduling (timestep-dependent guidance strength)
WEIGHT_START=3.0            # Strong guidance at early steps
WEIGHT_END=0.5              # Weak guidance at late steps

# Adaptive threshold scheduling (timestep-dependent detection threshold)
THRESHOLD_START=-0.5        # More sensitive at early steps
THRESHOLD_END=-2.5          # Very sensitive at late steps

# Class indices for VanGogh classifier (3-class)
# Class 0: Not-VanGogh (irrelevant)
# Class 1: Safe (Non-VanGogh style)
# Class 2: VanGogh style (harmful to suppress)
SAFE_CLASS=1
HARMFUL_CLASS=2

# Guidance application range
GUIDANCE_START_STEP=0
GUIDANCE_END_STEP=50

# ============================================================================
# Create output directory
# ============================================================================
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "🎨 VanGogh Style Suppression - Selective CG"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Classifier: step_11100"
echo "  Prompts: vangogh_50.txt (50 prompts)"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Parameters (based on best nudity config):"
echo "  Guidance Scale: $GUIDANCE_SCALE"
echo "  Harmful Scale: $HARMFUL_SCALE"
echo "  Spatial Threshold: $SPATIAL_THRESHOLD"
echo "  Weight Schedule: $WEIGHT_START → $WEIGHT_END"
echo "  Threshold Schedule: $THRESHOLD_START → $THRESHOLD_END"
echo ""
echo "Starting generation..."
echo "============================================"
echo ""

# ============================================================================
# Run Selective Classifier Guidance
# ============================================================================

python generate_selective_cg.py \
    "$CKPT_PATH" \
    --prompt_file "$PROMPT_FILE" \
    --selective_guidance \
    --classifier_ckpt "$CLASSIFIER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --safe_class $SAFE_CLASS \
    --harmful_class $HARMFUL_CLASS \
    --harmful_threshold $THRESHOLD_START \
    --spatial_threshold $SPATIAL_THRESHOLD \
    --guidance_scale $GUIDANCE_SCALE \
    --harmful_scale $HARMFUL_SCALE \
    --use_bidirectional \
    --use_adaptive_threshold \
    --threshold_strategy linear_decrease \
    --threshold_start_value $THRESHOLD_START \
    --threshold_end_value $THRESHOLD_END \
    --use_weight_scheduling \
    --weight_strategy linear_decrease \
    --weight_start_value $WEIGHT_START \
    --weight_end_value $WEIGHT_END \
    --guidance_start_step $GUIDANCE_START_STEP \
    --guidance_end_step $GUIDANCE_END_STEP \
    --normalize_gradient \
    --gradient_norm_type l2 \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --cfg_scale $CFG_SCALE \
    --nsamples $NSAMPLES \
    --seed $SEED \
    --save_visualizations

echo ""
echo "============================================"
echo "✅ VanGogh Suppression Complete!"
echo "============================================"
echo ""
echo "Results saved to:"
echo "  $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - 50 images (1 per prompt)"
echo "  - Grad-CAM visualizations"
echo "  - Generation statistics"
echo ""
echo "Check results with:"
echo "  ls -la $OUTPUT_DIR"
echo "============================================"
