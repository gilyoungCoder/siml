#!/usr/bin/env bash
set -euo pipefail

# =========================
# Quick Test Script for Selective CG
# Purpose: Verify implementation works before full experiments
# =========================

export CUDA_VISIBLE_DEVICES=6

# Create test prompt file
TEST_PROMPT_FILE="/tmp/test_prompts_selective_cg.txt"
cat > "${TEST_PROMPT_FILE}" << 'EOF'
a person walking in the park
portrait of a woman
EOF

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                    SELECTIVE CG - QUICK TEST                                   ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[TEST] Running quick test with 2 prompts, 5 steps"
echo ""

# Test configuration
CKPT_PATH="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="./test_output_selective_cg"
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"

# Quick test: Only 5 steps
python generate_selective_cg.py \
    "${CKPT_PATH}" \
    --prompt_file "${TEST_PROMPT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --nsamples 1 \
    --cfg_scale 7.5 \
    --num_inference_steps 5 \
    --seed 1234 \
    --selective_guidance \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --harmful_threshold 0.5 \
    --harmful_class 2 \
    --safe_class 1 \
    --spatial_threshold 0.5 \
    --guidance_scale 5.0 \
    --guidance_start_step 0 \
    --guidance_end_step 5 \
    --debug \
    --save_visualizations

EXIT_CODE=$?

echo ""
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "╔════════════════════════════════════════════════════════════════════════════════╗"
    echo "║  ✅ TEST PASSED                                                                 ║"
    echo "║                                                                                ║"
    echo "║  Implementation verified! You can now run full experiments:                   ║"
    echo "║    ./run_selective_cg.sh              # Standard harmful prompts              ║"
    echo "║    ./run_selective_cg_benign.sh       # Benign prompts test                   ║"
    echo "║                                                                                ║"
    echo "║  Test outputs: ${OUTPUT_DIR}                                                   ║"
    echo "╚════════════════════════════════════════════════════════════════════════════════╝"
else
    echo "╔════════════════════════════════════════════════════════════════════════════════╗"
    echo "║  ❌ TEST FAILED                                                                 ║"
    echo "║                                                                                ║"
    echo "║  Exit code: ${EXIT_CODE}                                                       ║"
    echo "║  Check error messages above for debugging                                     ║"
    echo "╚════════════════════════════════════════════════════════════════════════════════╝"
fi
echo ""

# Cleanup
rm -f "${TEST_PROMPT_FILE}"

exit ${EXIT_CODE}
