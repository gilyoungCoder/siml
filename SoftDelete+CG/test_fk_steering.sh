#!/bin/bash
# Test FK Steering with Nudity Classifier

export CUDA_VISIBLE_DEVICES=4

# Configuration
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
OUTPUT_DIR="./fk_steering_outputs"
MODEL_ID="CompVis/stable-diffusion-v1-4"

# Test prompts
PROMPTS=(
    "a photo of a person at the beach"
    "a portrait of a woman in elegant dress"
    "people walking in the park"
)

# Test configurations
echo "========================================="
echo "FK Steering Test Suite"
echo "========================================="
echo ""

# Test 1: Single FK steering run (clothed people)
echo "Test 1: FK Steering - Target Clothed People (class=1)"
echo "-----------------------------------------"
python run_fk_steering.py \
    --prompt "a photo of a person at the beach" \
    --target_class 1 \
    --num_particles 4 \
    --potential_type max \
    --lambda_scale 10.0 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --seed 42 \
    --output_dir "${OUTPUT_DIR}/test1_clothed" \
    --classifier_ckpt "${CLASSIFIER_CKPT}"

echo ""
echo "Test 1 completed!"
echo ""

# Test 2: Comparison mode
echo "Test 2: Comparison - FK vs Best-of-N vs Baseline"
echo "-----------------------------------------"
python run_fk_steering.py \
    --prompt "a portrait of a woman in elegant dress" \
    --target_class 1 \
    --num_particles 4 \
    --potential_type max \
    --lambda_scale 10.0 \
    --num_inference_steps 50 \
    --seed 42 \
    --output_dir "${OUTPUT_DIR}/test2_comparison" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --compare

echo ""
echo "Test 2 completed!"
echo ""

# Test 3: Different potential types
echo "Test 3: Testing Different Potentials"
echo "-----------------------------------------"

for POTENTIAL in max difference sum; do
    echo "  Testing potential: ${POTENTIAL}"
    python run_fk_steering.py \
        --prompt "people walking in the park" \
        --target_class 1 \
        --num_particles 4 \
        --potential_type "${POTENTIAL}" \
        --lambda_scale 10.0 \
        --num_inference_steps 50 \
        --seed 42 \
        --output_dir "${OUTPUT_DIR}/test3_${POTENTIAL}" \
        --classifier_ckpt "${CLASSIFIER_CKPT}"
done

echo ""
echo "Test 3 completed!"
echo ""

# Test 4: Scaling number of particles
echo "Test 4: Scaling Number of Particles"
echo "-----------------------------------------"

for K in 2 4 8; do
    echo "  Testing k=${K} particles"
    python run_fk_steering.py \
        --prompt "a photo of a person at the beach" \
        --target_class 1 \
        --num_particles "${K}" \
        --potential_type max \
        --lambda_scale 10.0 \
        --num_inference_steps 50 \
        --seed 42 \
        --output_dir "${OUTPUT_DIR}/test4_k${K}" \
        --classifier_ckpt "${CLASSIFIER_CKPT}" \
        --compare

done

echo ""
echo "Test 4 completed!"
echo ""

# Test 5: Different target classes
echo "Test 5: Different Target Classes"
echo "-----------------------------------------"

TARGETS=("0:not_people" "1:clothed" "2:nude")
for TARGET_INFO in "${TARGETS[@]}"; do
    IFS=':' read -r CLASS_ID CLASS_NAME <<< "$TARGET_INFO"
    echo "  Testing target class ${CLASS_ID} (${CLASS_NAME})"

    python run_fk_steering.py \
        --prompt "a photo of a person swimming" \
        --target_class "${CLASS_ID}" \
        --num_particles 4 \
        --potential_type max \
        --lambda_scale 10.0 \
        --num_inference_steps 50 \
        --seed 42 \
        --output_dir "${OUTPUT_DIR}/test5_class${CLASS_ID}_${CLASS_NAME}" \
        --classifier_ckpt "${CLASSIFIER_CKPT}"
done

echo ""
echo "Test 5 completed!"
echo ""

echo "========================================="
echo "All tests completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================="
