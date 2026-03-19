#!/bin/bash
# ============================================================================
# Multi-Concept Erasure - Nudity + Violence Simultaneous Removal
# ============================================================================
export CUDA_VISIBLE_DEVICES=3

# Activate conda environment
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate sdd

set -e

# ============================================================================
# Configuration
# ============================================================================

CKPT_PATH="CompVis/stable-diffusion-v1-4"

# Classifier paths
NUDITY_CLASSIFIER="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
VIOLENCE_CLASSIFIER="./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth"

# Prompt files
NUDITY_PROMPTS="./prompts/violence_50.txt"
VIOLENCE_PROMPTS="./prompts/violence_50.txt"
# MIXED_PROMPTS="./prompts/mixed_concepts.txt"  # If you have mixed prompts

# Output directory
OUTPUT_DIR="scg_outputs/multi_concept_m2/violence"

# Generation parameters
NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
NSAMPLES=1
SEED=123

# ============================================================================
# Best parameters from grid search
# ============================================================================

# Nudity parameters (from grid search best: gs7.0_hs1.5_st0.3_ws3.0-0.5_ts-0.5--2.5)
# Safe+Partial: 48/50 (96%) - Safe: 39 (78%), Partial: 9 (18%), Full: 0 (0%), NotPeople: 2 (4%)
NUDITY_GUIDANCE_SCALE=7.0
NUDITY_HARMFUL_SCALE=1.5
NUDITY_SPATIAL_THRESHOLD=0.3
NUDITY_HARMFUL_THRESHOLD=-0.5
NUDITY_WEIGHT_START=3.0
NUDITY_WEIGHT_END=0.5
NUDITY_THRESHOLD_START=-0.5
NUDITY_THRESHOLD_END=-2.5

# Violence parameters (from grid search best: gs7.0_hs1.5_st0.4_ws3.0-0.5_ts0.0--1.5)
# Safe+Partial: 33/50 (66%) - Safe: 22 (44%), Partial: 11 (22%), Full: 4 (8%), NotRelevant: 13 (26%)
VIOLENCE_GUIDANCE_SCALE=7.0
VIOLENCE_HARMFUL_SCALE=1.5
VIOLENCE_SPATIAL_THRESHOLD=0.4
VIOLENCE_HARMFUL_THRESHOLD=-0.5
VIOLENCE_WEIGHT_START=3.0
VIOLENCE_WEIGHT_END=0.5
VIOLENCE_THRESHOLD_START=0.0
VIOLENCE_THRESHOLD_END=-1.5

# Combination strategy
COMBINATION_STRATEGY="orthogonal_projection"  # Options: additive, direction_based, orthogonal_projection, max, weighted

# ============================================================================
# Verify files
# ============================================================================

echo "============================================"
echo "🔍 Multi-Concept Erasure Setup"
echo "============================================"

if [ ! -f "$NUDITY_CLASSIFIER" ]; then
    echo "❌ Error: Nudity classifier not found at $NUDITY_CLASSIFIER"
    exit 1
fi
echo "✓ Nudity classifier: $NUDITY_CLASSIFIER"

if [ ! -f "$VIOLENCE_CLASSIFIER" ]; then
    echo "❌ Error: Violence classifier not found at $VIOLENCE_CLASSIFIER"
    exit 1
fi
echo "✓ Violence classifier: $VIOLENCE_CLASSIFIER"

# ============================================================================
# Experiment 1: Nudity prompts with multi-concept erasure
# ============================================================================

if [ -f "$NUDITY_PROMPTS" ]; then
    echo ""
    echo "============================================"
    echo "Experiment 1: Nudity prompts"
    echo "============================================"
    echo "Prompts: $NUDITY_PROMPTS"
    echo "Output: ${OUTPUT_DIR}/nudity_prompts"
    echo ""

    python generate_multi_concept_cg.py \
        "$CKPT_PATH" \
        --prompt_file "$NUDITY_PROMPTS" \
        --output_dir "${OUTPUT_DIR}/nudity_prompts" \
        --nsamples $NSAMPLES \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --cfg_scale $CFG_SCALE \
        --seed $SEED \
        \
        --nudity_enabled \
        --nudity_classifier "$NUDITY_CLASSIFIER" \
        --nudity_harmful_threshold $NUDITY_HARMFUL_THRESHOLD \
        --nudity_spatial_threshold $NUDITY_SPATIAL_THRESHOLD \
        --nudity_guidance_scale $NUDITY_GUIDANCE_SCALE \
        --nudity_harmful_scale $NUDITY_HARMFUL_SCALE \
        --nudity_threshold_start $NUDITY_THRESHOLD_START \
        --nudity_threshold_end $NUDITY_THRESHOLD_END \
        --nudity_weight_start $NUDITY_WEIGHT_START \
        --nudity_weight_end $NUDITY_WEIGHT_END \
        \
        --violence_enabled \
        --violence_classifier "$VIOLENCE_CLASSIFIER" \
        --violence_harmful_threshold $VIOLENCE_HARMFUL_THRESHOLD \
        --violence_spatial_threshold $VIOLENCE_SPATIAL_THRESHOLD \
        --violence_guidance_scale $VIOLENCE_GUIDANCE_SCALE \
        --violence_harmful_scale $VIOLENCE_HARMFUL_SCALE \
        --violence_threshold_start $VIOLENCE_THRESHOLD_START \
        --violence_threshold_end $VIOLENCE_THRESHOLD_END \
        --violence_weight_start $VIOLENCE_WEIGHT_START \
        --violence_weight_end $VIOLENCE_WEIGHT_END \
        \
        --use_bidirectional \
        --use_adaptive_threshold \
        --threshold_strategy "cosine_anneal" \
        --use_weight_scheduling \
        --weight_strategy "cosine_anneal" \
        --use_heatmap_weighted_guidance \
        --combination_strategy "$COMBINATION_STRATEGY" \
        --save_visualizations

    echo ""
    echo "✅ Experiment 1 complete!"
    echo ""
fi

# ============================================================================
# Experiment 2: Violence prompts with multi-concept erasure
# ============================================================================

if [ -f "$VIOLENCE_PROMPTS" ]; then
    echo ""
    echo "============================================"
    echo "Experiment 2: Violence prompts"
    echo "============================================"
    echo "Prompts: $VIOLENCE_PROMPTS"
    echo "Output: ${OUTPUT_DIR}/violence_prompts"
    echo ""

    python generate_multi_concept_cg.py \
        "$CKPT_PATH" \
        --prompt_file "$VIOLENCE_PROMPTS" \
        --output_dir "${OUTPUT_DIR}/violence_prompts" \
        --nsamples $NSAMPLES \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --cfg_scale $CFG_SCALE \
        --seed $SEED \
        \
        --nudity_enabled \
        --nudity_classifier "$NUDITY_CLASSIFIER" \
        --nudity_harmful_threshold $NUDITY_HARMFUL_THRESHOLD \
        --nudity_spatial_threshold $NUDITY_SPATIAL_THRESHOLD \
        --nudity_guidance_scale $NUDITY_GUIDANCE_SCALE \
        --nudity_harmful_scale $NUDITY_HARMFUL_SCALE \
        --nudity_threshold_start $NUDITY_THRESHOLD_START \
        --nudity_threshold_end $NUDITY_THRESHOLD_END \
        --nudity_weight_start $NUDITY_WEIGHT_START \
        --nudity_weight_end $NUDITY_WEIGHT_END \
        \
        --violence_enabled \
        --violence_classifier "$VIOLENCE_CLASSIFIER" \
        --violence_harmful_threshold $VIOLENCE_HARMFUL_THRESHOLD \
        --violence_spatial_threshold $VIOLENCE_SPATIAL_THRESHOLD \
        --violence_guidance_scale $VIOLENCE_GUIDANCE_SCALE \
        --violence_harmful_scale $VIOLENCE_HARMFUL_SCALE \
        --violence_threshold_start $VIOLENCE_THRESHOLD_START \
        --violence_threshold_end $VIOLENCE_THRESHOLD_END \
        --violence_weight_start $VIOLENCE_WEIGHT_START \
        --violence_weight_end $VIOLENCE_WEIGHT_END \
        \
        --use_bidirectional \
        --use_adaptive_threshold \
        --threshold_strategy "cosine_anneal" \
        --use_weight_scheduling \
        --weight_strategy "cosine_anneal" \
        --use_heatmap_weighted_guidance \
        --combination_strategy "$COMBINATION_STRATEGY" \
        --save_visualizations

    echo ""
    echo "✅ Experiment 2 complete!"
    echo ""
fi

# ============================================================================
# Experiment 3: Mixed prompts (if available)
# ============================================================================

if [ -f "$MIXED_PROMPTS" ]; then
    echo ""
    echo "============================================"
    echo "Experiment 3: Mixed concept prompts"
    echo "============================================"
    echo "Prompts: $MIXED_PROMPTS"
    echo "Output: ${OUTPUT_DIR}/mixed_prompts"
    echo ""

    python generate_multi_concept_cg.py \
        "$CKPT_PATH" \
        --prompt_file "$MIXED_PROMPTS" \
        --output_dir "${OUTPUT_DIR}/mixed_prompts" \
        --nsamples $NSAMPLES \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --cfg_scale $CFG_SCALE \
        --seed $SEED \
        \
        --nudity_enabled \
        --nudity_classifier "$NUDITY_CLASSIFIER" \
        --nudity_harmful_threshold $NUDITY_HARMFUL_THRESHOLD \
        --nudity_spatial_threshold $NUDITY_SPATIAL_THRESHOLD \
        --nudity_guidance_scale $NUDITY_GUIDANCE_SCALE \
        --nudity_harmful_scale $NUDITY_HARMFUL_SCALE \
        --nudity_threshold_start $NUDITY_THRESHOLD_START \
        --nudity_threshold_end $NUDITY_THRESHOLD_END \
        --nudity_weight_start $NUDITY_WEIGHT_START \
        --nudity_weight_end $NUDITY_WEIGHT_END \
        \
        --violence_enabled \
        --violence_classifier "$VIOLENCE_CLASSIFIER" \
        --violence_harmful_threshold $VIOLENCE_HARMFUL_THRESHOLD \
        --violence_spatial_threshold $VIOLENCE_SPATIAL_THRESHOLD \
        --violence_guidance_scale $VIOLENCE_GUIDANCE_SCALE \
        --violence_harmful_scale $VIOLENCE_HARMFUL_SCALE \
        --violence_threshold_start $VIOLENCE_THRESHOLD_START \
        --violence_threshold_end $VIOLENCE_THRESHOLD_END \
        --violence_weight_start $VIOLENCE_WEIGHT_START \
        --violence_weight_end $VIOLENCE_WEIGHT_END \
        \
        --use_bidirectional \
        --use_adaptive_threshold \
        --threshold_strategy "cosine_anneal" \
        --use_weight_scheduling \
        --weight_strategy "cosine_anneal" \
        --use_heatmap_weighted_guidance \
        --combination_strategy "$COMBINATION_STRATEGY" \
        --save_visualizations

    echo ""
    echo "✅ Experiment 3 complete!"
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================"
echo "🎉 Multi-Concept Erasure Complete!"
echo "============================================"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Experiments:"
[ -f "$NUDITY_PROMPTS" ] && echo "  ✓ Nudity prompts → ${OUTPUT_DIR}/nudity_prompts"
[ -f "$VIOLENCE_PROMPTS" ] && echo "  ✓ Violence prompts → ${OUTPUT_DIR}/violence_prompts"
[ -f "$MIXED_PROMPTS" ] && echo "  ✓ Mixed prompts → ${OUTPUT_DIR}/mixed_prompts"
echo ""
echo "Configuration:"
echo "  Nudity guidance: scale=${NUDITY_GUIDANCE_SCALE}, harmful_scale=${NUDITY_HARMFUL_SCALE}"
echo "  Violence guidance: scale=${VIOLENCE_GUIDANCE_SCALE}, harmful_scale=${VIOLENCE_HARMFUL_SCALE}"
echo "  Combination strategy: ${COMBINATION_STRATEGY}"
echo "============================================"
