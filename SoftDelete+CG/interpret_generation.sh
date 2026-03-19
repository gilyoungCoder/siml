#!/usr/bin/env bash
set -euo pipefail

# =========================
# Classifier Interpretation - Generation Mode
# =========================
# 이미지 생성 과정을 step-by-step으로 분석
# 언제 nude concept이 발현하는지 추적

# GPU Configuration
export CUDA_VISIBLE_DEVICES=4

# =========================
# Input Configuration
# =========================
# 생성할 프롬프트
PROMPT="a person at the beach"
# PROMPT="portrait of a person"
# PROMPT="nude body"

# =========================
# Model Configuration
# =========================
CLASSIFIER_CKPT="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"
SD_MODEL="CompVis/stable-diffusion-v1-4"

# =========================
# Generation Parameters
# =========================
NUM_STEPS=50             # Denoising steps (기본 50)
TARGET_CLASS=2           # 2 = nude people

# =========================
# Output Configuration
# =========================
# 프롬프트에서 파일명으로 안전한 이름 생성
SAFE_PROMPT=$(echo "${PROMPT}" | tr ' ' '_' | tr -cd '[:alnum:]_-' | cut -c1-30)
OUTPUT_DIR="./interpretation_results/generation_${SAFE_PROMPT}"

# =========================
# Run Analysis
# =========================
mkdir -p "${OUTPUT_DIR}"
LOG="./logs/interpret_gen_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG}")"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║           CLASSIFIER INTERPRETATION - GENERATION MODE                          ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[CONFIG] Prompt: \"${PROMPT}\""
echo "[CONFIG] Classifier: ${CLASSIFIER_CKPT}"
echo "[CONFIG] Generation steps: ${NUM_STEPS}"
echo "[CONFIG] Target Class: ${TARGET_CLASS} (nude)"
echo "[CONFIG] Output: ${OUTPUT_DIR}"
echo "[CONFIG] Log: ${LOG}"
echo ""

echo "┌─ Generation Analysis ─────────────────────────────────────────────────────────┐"
echo "│  This mode will:"
echo "│    1. Generate an image from the prompt"
echo "│    2. Capture classifier attention at each denoising step"
echo "│    3. Create visualizations every 10 steps"
echo "│    4. Generate heatmap evolution GIF"
echo "│    5. Plot probability trajectory over time"
echo "│"
echo "│  Outputs:"
echo "│    - final_image.png: Generated image"
echo "│    - step_XXX.png: Heatmap at step XXX (every 10 steps)"
echo "│    - heatmap_evolution.gif: Animation of attention changes"
echo "│    - probability_evolution.png: Class probability over steps"
echo "│    - prediction_trajectory.json: Numerical data"
echo "│"
echo "│  Estimated time: ~${NUM_STEPS} seconds + ~30s processing"
echo "└───────────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "[INFO] Starting generation and analysis..."
echo ""

# Run interpretation
python interpret_classifier.py \
    --mode generation \
    --prompt "${PROMPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --classifier_ckpt "${CLASSIFIER_CKPT}" \
    --sd_model "${SD_MODEL}" \
    --num_steps "${NUM_STEPS}" \
    --target_class "${TARGET_CLASS}" \
    --device cuda \
    2>&1 | tee "${LOG}"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║  Generation analysis complete!"
echo "║"
echo "║  Results saved to: ${OUTPUT_DIR}"
echo "║"
echo "║  Key files:"
echo "║    - final_image.png: Generated image"
echo "║    - heatmap_evolution.gif: Attention animation"
echo "║    - probability_evolution.png: Probability trajectory"
echo "║    - prediction_trajectory.json: Detailed step data"
echo "║"
echo "║  Critical step analysis:"
echo "║"

# Find critical steps where nude probability increases sharply
python << 'EOF'
import json
from pathlib import Path
import sys

trajectory_path = Path(sys.argv[1]) / "prediction_trajectory.json"

if not trajectory_path.exists():
    print("║    Trajectory file not found.")
    sys.exit(0)

with open(trajectory_path) as f:
    data = json.load(f)

nude_probs = [(d['step'], d['probs'][2]) for d in data]

# Find largest increases
increases = []
for i in range(1, len(nude_probs)):
    step, prob = nude_probs[i]
    prev_step, prev_prob = nude_probs[i-1]
    increase = prob - prev_prob
    increases.append((step, prev_prob, prob, increase))

# Sort by increase
increases.sort(key=lambda x: x[3], reverse=True)

print(f"║    Top 5 steps with largest nude probability increase:")
for step, prev_prob, prob, increase in increases[:5]:
    if increase > 0:
        print(f"║      Step {step:2d}: {prev_prob:.3f} → {prob:.3f}  (Δ +{increase:.3f})")

print(f"║")
print(f"║    Final nude probability: {nude_probs[-1][1]:.3f}")

EOF "${OUTPUT_DIR}"

echo "║"
echo "║  View animation: eog ${OUTPUT_DIR}/heatmap_evolution.gif"
echo "║  View graph:     eog ${OUTPUT_DIR}/probability_evolution.png"
echo "║  View final:     eog ${OUTPUT_DIR}/final_image.png"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
