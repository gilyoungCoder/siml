#!/bin/bash
# ============================================================================
# SDErasure Full Evaluation Pipeline
# 1. Generate Ring-A-Bell images with fine-tuned model
# 2. Evaluate with Qwen3-VL (nudity classification)
# 3. Compare with baseline
# ============================================================================

set -e

PYTHON="PYTHONNOUSERSITE=1 /mnt/home/yhgil99/.conda/envs/sfgd/bin/python"
VLM_PYTHON="PYTHONNOUSERSITE=1 /mnt/home/yhgil99/.conda/envs/vlm/bin/python"

MODEL_ID="CompVis/stable-diffusion-v1-4"
RINGABELL_PROMPTS="/mnt/home/yhgil99/unlearning/prompts/nudity_datasets/ringabell.txt"
VLM_EVAL_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
BASE_OUT="/mnt/home/yhgil99/unlearning/SDErasure/outputs"

# ---- Configuration ----
UNET_DIR="${1:-$BASE_OUT/sderasure_nudity/unet}"
OUTPUT_NAME="${2:-sderasure_nudity}"
GEN_GPU="${3:-2}"
EVAL_GPU="${4:-3}"

GEN_DIR="$BASE_OUT/${OUTPUT_NAME}/ringabell_images"
BASELINE_DIR="$BASE_OUT/baseline_sd14/ringabell_images"

echo "============================================================"
echo "SDErasure Evaluation Pipeline"
echo "  UNet dir   : $UNET_DIR"
echo "  Output name: $OUTPUT_NAME"
echo "  Gen GPU    : $GEN_GPU"
echo "  Eval GPU   : $EVAL_GPU"
echo "============================================================"

# ---- Step 1: Generate Ring-A-Bell images ----
if [ ! -d "$GEN_DIR" ] || [ $(ls "$GEN_DIR"/*.png 2>/dev/null | wc -l) -lt 79 ]; then
    echo ""
    echo "[Step 1] Generating Ring-A-Bell images with fine-tuned model..."
    CUDA_VISIBLE_DEVICES=$GEN_GPU eval $PYTHON generate_from_prompts.py \
        --model_id "$MODEL_ID" \
        --unet_dir "$UNET_DIR" \
        --prompt_file "$RINGABELL_PROMPTS" \
        --output_dir "$GEN_DIR" \
        --seed 42 \
        --num_inference_steps 50 \
        --guidance_scale 7.5
    echo "[Step 1] Done: $(ls "$GEN_DIR"/*.png | wc -l) images generated"
else
    echo "[Step 1] Skipped: $GEN_DIR already has $(ls "$GEN_DIR"/*.png | wc -l) images"
fi

# ---- Step 1b: Generate baseline (if missing) ----
if [ ! -d "$BASELINE_DIR" ] || [ $(ls "$BASELINE_DIR"/*.png 2>/dev/null | wc -l) -lt 79 ]; then
    echo ""
    echo "[Step 1b] Generating baseline images (original SD v1.4)..."
    CUDA_VISIBLE_DEVICES=$GEN_GPU eval $PYTHON generate_from_prompts.py \
        --model_id "$MODEL_ID" \
        --prompt_file "$RINGABELL_PROMPTS" \
        --output_dir "$BASELINE_DIR" \
        --seed 42 \
        --num_inference_steps 50 \
        --guidance_scale 7.5
    echo "[Step 1b] Done: $(ls "$BASELINE_DIR"/*.png | wc -l) baseline images"
else
    echo "[Step 1b] Skipped: baseline already has $(ls "$BASELINE_DIR"/*.png | wc -l) images"
fi

# ---- Step 2: VLM Evaluation ----
echo ""
echo "[Step 2] Running Qwen3-VL nudity evaluation..."

# Evaluate SDErasure
echo "  Evaluating SDErasure model..."
CUDA_VISIBLE_DEVICES=$EVAL_GPU eval $VLM_PYTHON "$VLM_EVAL_SCRIPT" "$GEN_DIR" nudity qwen

# Evaluate Baseline
echo "  Evaluating baseline model..."
CUDA_VISIBLE_DEVICES=$EVAL_GPU eval $VLM_PYTHON "$VLM_EVAL_SCRIPT" "$BASELINE_DIR" nudity qwen

# ---- Step 3: Summary ----
echo ""
echo "============================================================"
echo "RESULTS COMPARISON"
echo "============================================================"

echo ""
echo "--- SDErasure ($OUTPUT_NAME) ---"
cat "$GEN_DIR/results_qwen3_vl_nudity.txt" 2>/dev/null || echo "  (results not yet available)"

echo ""
echo "--- Baseline (SD v1.4) ---"
cat "$BASELINE_DIR/results_qwen3_vl_nudity.txt" 2>/dev/null || echo "  (results not yet available)"

echo ""
echo "============================================================"
echo "Evaluation complete!"
echo "  SDErasure results: $GEN_DIR/results_qwen3_vl_nudity.txt"
echo "  Baseline results : $BASELINE_DIR/results_qwen3_vl_nudity.txt"
echo "============================================================"
