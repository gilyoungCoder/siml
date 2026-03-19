#!/bin/bash
# Compare Different Unlearning Methods

set -euo pipefail

export CUDA_VISIBLE_DEVICES=4

PROMPT_FILE="./prompts/sexual_50.txt"
BASE_OUTPUT_DIR="./comparison_outputs"
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║               COMPARISON: Different Unlearning Approaches                      ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 1. Baseline (no unlearning)
echo "1️⃣  Baseline (No Unlearning)"
echo "   Generating images without any nudity removal..."
echo ""

python unlearn_nudity_fk.py \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${BASE_OUTPUT_DIR}/baseline" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --num_particles 1 \
    --lambda_scale 0.0 \
    --nsamples 1 \
    --seed 42

echo ""
echo "   ✓ Baseline complete"
echo ""

# 2. Best-of-N (k=4)
echo "2️⃣  Best-of-N (k=4)"
echo "   Generate 4 samples, pick best according to classifier..."
echo ""

python unlearn_nudity_fk.py \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${BASE_OUTPUT_DIR}/best_of_4" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --num_particles 4 \
    --lambda_scale 0.0 \
    --resampling_interval 1000 \
    --nsamples 1 \
    --seed 42

echo ""
echo "   ✓ Best-of-N complete"
echo ""

# 3. FK Steering (k=2, weak)
echo "3️⃣  FK Steering (k=2, λ=5.0)"
echo "   Weak steering with 2 particles..."
echo ""

python unlearn_nudity_fk.py \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${BASE_OUTPUT_DIR}/fk_k2_lambda5" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --num_particles 2 \
    --lambda_scale 5.0 \
    --potential_type max \
    --resampling_interval 10 \
    --nsamples 1 \
    --seed 42 \
    --save_all_particles

echo ""
echo "   ✓ FK Steering (k=2) complete"
echo ""

# 4. FK Steering (k=4, medium)
echo "4️⃣  FK Steering (k=4, λ=10.0)"
echo "   Medium steering with 4 particles..."
echo ""

python unlearn_nudity_fk.py \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${BASE_OUTPUT_DIR}/fk_k4_lambda10" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --num_particles 4 \
    --lambda_scale 10.0 \
    --potential_type max \
    --resampling_interval 10 \
    --nsamples 1 \
    --seed 42 \
    --save_all_particles

echo ""
echo "   ✓ FK Steering (k=4) complete"
echo ""

# 5. FK Steering (k=4, strong)
echo "5️⃣  FK Steering (k=4, λ=15.0)"
echo "   Strong steering with 4 particles..."
echo ""

python unlearn_nudity_fk.py \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${BASE_OUTPUT_DIR}/fk_k4_lambda15" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --num_particles 4 \
    --lambda_scale 15.0 \
    --potential_type max \
    --resampling_interval 10 \
    --nsamples 1 \
    --seed 42 \
    --save_all_particles

echo ""
echo "   ✓ FK Steering (k=4, strong) complete"
echo ""

# 6. FK Steering (k=8, strong)
echo "6️⃣  FK Steering (k=8, λ=15.0)"
echo "   Strong steering with 8 particles..."
echo ""

python unlearn_nudity_fk.py \
    --prompt_file "${PROMPT_FILE}" \
    --output_dir "${BASE_OUTPUT_DIR}/fk_k8_lambda15" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --num_particles 8 \
    --lambda_scale 15.0 \
    --potential_type max \
    --resampling_interval 10 \
    --nsamples 1 \
    --seed 42 \
    --save_all_particles

echo ""
echo "   ✓ FK Steering (k=8) complete"
echo ""

# Analyze results
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                            RESULTS SUMMARY                                     ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""

python << 'EOF'
import json
import os
import numpy as np

base_dir = "./comparison_outputs"
methods = [
    ("baseline", "Baseline (no unlearning)"),
    ("best_of_4", "Best-of-N (k=4)"),
    ("fk_k2_lambda5", "FK Steering (k=2, λ=5)"),
    ("fk_k4_lambda10", "FK Steering (k=4, λ=10)"),
    ("fk_k4_lambda15", "FK Steering (k=4, λ=15)"),
    ("fk_k8_lambda15", "FK Steering (k=8, λ=15)"),
]

print(f"{'Method':<35} {'Mean Reward':<15} {'Std Reward':<15}")
print("="*65)

for method_dir, method_name in methods:
    results_path = os.path.join(base_dir, method_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

        all_rewards = []
        for prompt_result in results['prompts']:
            for sample_result in prompt_result['samples']:
                all_rewards.append(sample_result['best_reward'])

        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)

        print(f"{method_name:<35} {mean_reward:<15.4f} {std_reward:<15.4f}")
    else:
        print(f"{method_name:<35} {'N/A':<15} {'N/A':<15}")

print("="*65)
print("\nHigher reward = More clothed people (better unlearning)")
print(f"\nAll results saved to: {base_dir}/")
EOF

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║  ✓ Comparison complete!"
echo "║"
echo "║  Results: ${BASE_OUTPUT_DIR}/"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
