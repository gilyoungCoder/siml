#!/bin/bash
# ============================================================================
# Full Adaptive CG - 3가지 Adaptive 메커니즘 모두 활성화
#   1. Adaptive Threshold by Timestep (0.0 → -2.0, logit-based)
#   2. Adaptive Guidance Scale by Timestep (8.0 → 1.0)
#   3. Heatmap-Weighted Spatial Guidance (pixel-wise adaptive)
# ============================================================================
export CUDA_VISIBLE_DEVICES=7

set -e

# ============================================================================
# 설정 (필요에 따라 수정)
# ============================================================================

CKPT_PATH="CompVis/stable-diffusion-v1-4"

# Concept 선택 (주석 처리/해제로 변경)
# Nudity
# CLASSIFIER_PATH="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
# PROMPT_FILE="./prompts/sexual_50.txt"
# OUTPUT_DIR="scg_outputs/full_adaptive/nudity"

# Violence (사용하려면 위 3줄 주석처리하고 아래 3줄 주석 해제)
CLASSIFIER_PATH="./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth"
PROMPT_FILE="./prompts/violence_50.txt"
OUTPUT_DIR="scg_outputs/full_adaptive/violence"

# 생성 파라미터
NUM_INFERENCE_STEPS=50
CFG_SCALE=7.5
NSAMPLES=1
SEED=123

# ============================================================================
# Full Adaptive 설정
# ============================================================================

echo "============================================"
echo "🚀 Full Adaptive CG - Image Generation"
echo "============================================"
echo "Classifier: $CLASSIFIER_PATH"
echo "Prompts: $PROMPT_FILE"
echo "Output: $OUTPUT_DIR"
echo "Samples per prompt: $NSAMPLES"
echo ""
echo "Adaptive Features:"
echo "  ✅ Harmful Threshold: 0.0 → -2.0 (Logit, Cosine Anneal)"
echo "  ✅ Guidance Scale: 8.0 → 1.0 (Cosine Anneal)"
echo "  ✅ Heatmap-Weighted: Pixel-wise adaptive"
echo "============================================"
echo ""

# Classifier 확인
if [ ! -f "$CLASSIFIER_PATH" ]; then
    echo "❌ Error: Classifier not found at $CLASSIFIER_PATH"
    exit 1
fi

# Prompt 파일 확인
if [ ! -f "$PROMPT_FILE" ]; then
    echo "❌ Error: Prompt file not found at $PROMPT_FILE"
    exit 1
fi

# 실행
python generate_selective_cg.py \
    "$CKPT_PATH" \
    --prompt_file "$PROMPT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --nsamples $NSAMPLES \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --cfg_scale $CFG_SCALE \
    --seed $SEED \
    \
    --selective_guidance \
    --classifier_ckpt "$CLASSIFIER_PATH" \
    --harmful_threshold -0.5 \
    --spatial_threshold 0.3 \
    --guidance_scale 15.0 \
    --harmful_scale 1 \
    --use_bidirectional \
    \
    --use_adaptive_threshold \
    --threshold_strategy "cosine_anneal" \
    --threshold_start_value 0.0 \
    --threshold_end_value -2.0 \
    \
    --use_heatmap_weighted_guidance \
    \
    --use_weight_scheduling \
    --weight_strategy "cosine_anneal" \
    --weight_start_value 3 \
    --weight_end_value 1 \
    \
    --normalize_gradient \
    --gradient_norm_type "l2" \
    --save_visualizations

echo ""
echo "============================================"
echo "✅ Generation Complete!"
echo "============================================"
echo "Output directory: $OUTPUT_DIR"
echo "Total images: $(ls $OUTPUT_DIR/*.png 2>/dev/null | wc -l)"
echo "Visualizations: $OUTPUT_DIR/visualizations/"
echo "============================================"
