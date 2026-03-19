#!/bin/bash
# ============================================================================
# Test Gradient Conflict Detection
# ============================================================================
export CUDA_VISIBLE_DEVICES=6

# Activate conda environment
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate sdd

set -e

echo "============================================"
echo "🔬 Testing Gradient Conflict Detection"
echo "============================================"
echo ""

# Test with a single mixed prompt
python generate_multi_concept_cg.py \
    CompVis/stable-diffusion-v1-4 \
    --prompt_file prompts/mixed_concepts.txt \
    --output_dir scg_outputs/test_conflict \
    --nudity_classifier ./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth \
    --violence_classifier ./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth \
    --nudity_enabled \
    --violence_enabled \
    --use_bidirectional \
    --use_adaptive_threshold \
    --use_weight_scheduling \
    --nudity_guidance_scale 7.0 \
    --nudity_harmful_scale 1.5 \
    --nudity_spatial_threshold 0.3 \
    --nudity_weight_start 3.0 \
    --nudity_weight_end 0.5 \
    --nudity_threshold_start -0.5 \
    --nudity_threshold_end -2.5 \
    --violence_guidance_scale 7.0 \
    --violence_harmful_scale 1.5 \
    --violence_spatial_threshold 0.4 \
    --violence_weight_start 3.0 \
    --violence_weight_end 0.5 \
    --violence_threshold_start 0.0 \
    --violence_threshold_end -1.5 \
    --combination_strategy additive \
    --num_inference_steps 50 \
    --cfg_scale 7.5 \
    --nsamples 1 \
    --seed 123

echo ""
echo "============================================"
echo "✅ Test Complete!"
echo "============================================"
echo "Check the output at: scg_outputs/test_conflict"
echo "Conflict analysis: scg_outputs/test_conflict/gradient_conflicts.csv"
echo "============================================"
