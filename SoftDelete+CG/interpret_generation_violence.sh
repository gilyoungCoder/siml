#!/usr/bin/env bash
set -euo pipefail

# =========================
# Classifier Interpretation - Generation Mode (VIOLENCE)
# =========================
# 이미지 생성 과정을 step-by-step으로 분석
# 언제 violence concept이 발현하는지 추적

# GPU Configuration
export CUDA_VISIBLE_DEVICES=7

# =========================
# Input Configuration
# =========================
# 생성할 프롬프트 (violence related)
PROMPT="people fighting with knife"
# PROMPT="violent scene with people"
# PROMPT="people in combat"
# PROMPT="a person holding a weapon"

# =========================
# Model Configuration
# =========================
CLASSIFIER_CKPT="./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth"
SD_MODEL="CompVis/stable-diffusion-v1-4"

# =========================
# Generation Parameters
# =========================
NUM_STEPS=50             # Denoising steps (기본 50)
TARGET_CLASS=2           # 2 = violent people (harmful)

# =========================
# Output Configuration
# =========================
# 프롬프트에서 파일명으로 안전한 이름 생성
SAFE_PROMPT=$(echo "${PROMPT}" | tr ' ' '_' | tr -cd '[:alnum:]_-' | cut -c1-30)
OUTPUT_DIR="./interpretation_results/violence_generation_${SAFE_PROMPT}"

# =========================
# Run Analysis
# =========================
mkdir -p "${OUTPUT_DIR}"
LOG="./logs/interpret_violence_gen_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG}")"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║      VIOLENCE CLASSIFIER INTERPRETATION - GENERATION MODE                      ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[CONFIG] Prompt: \"${PROMPT}\""
echo "[CONFIG] Classifier: ${CLASSIFIER_CKPT}"
echo "[CONFIG] Generation steps: ${NUM_STEPS}"
echo "[CONFIG] Target Class: ${TARGET_CLASS} (0=not people, 1=peaceful, 2=violent)"
echo "[CONFIG] Output: ${OUTPUT_DIR}"
echo "[CONFIG] Log: ${LOG}"
echo ""

echo "┌─ Generation Analysis ─────────────────────────────────────────────────────────┐"
echo "│  This mode will:"
echo "│    1. Generate an image from the prompt step-by-step"
echo "│    2. Track violence classifier attention at each denoising step"
echo "│    3. Create Grad-CAM visualizations every 10 steps"
echo "│    4. Generate heatmap evolution GIF showing violence detection"
echo "│    5. Plot violence probability trajectory over time"
echo "│"
echo "│  Key insights:"
echo "│    - When does violence concept emerge during generation?"
echo "│    - Which denoising steps are critical for violence?"
echo "│    - How does classifier attention evolve over time?"
echo "│"
echo "│  Outputs:"
echo "│    - final_image.png: Generated image"
echo "│    - step_XXX.png: Grad-CAM heatmap at step XXX (every 10 steps)"
echo "│    - heatmap_evolution.gif: Animation of violence attention changes"
echo "│    - probability_evolution.png: Violence probability over steps"
echo "│    - prediction_trajectory.json: Numerical data for all steps"
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
echo "║    - heatmap_evolution.gif: Violence attention animation"
echo "║    - probability_evolution.png: Violence probability trajectory"
echo "║    - prediction_trajectory.json: Detailed step-by-step data"
echo "║"
echo "║  Critical step analysis:"
echo "║"

# Find critical steps where violence probability increases sharply
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

violent_probs = [(d['step'], d['probs'][2]) for d in data]

# Find largest increases
increases = []
for i in range(1, len(violent_probs)):
    step, prob = violent_probs[i]
    prev_step, prev_prob = violent_probs[i-1]
    increase = prob - prev_prob
    increases.append((step, prev_prob, prob, increase))

# Sort by increase
increases.sort(key=lambda x: x[3], reverse=True)

print(f"║    Top 5 steps with largest violence probability increase:")
for step, prev_prob, prob, increase in increases[:5]:
    if increase > 0:
        print(f"║      Step {step:2d}: {prev_prob:.3f} → {prob:.3f}  (Δ +{increase:.3f})")

print(f"║")
print(f"║    Final violence probability: {violent_probs[-1][1]:.3f}")
print(f"║")

# Find when violence probability first exceeds threshold
threshold = 0.5
for step, prob in violent_probs:
    if prob > threshold:
        print(f"║    Violence first detected (>{threshold}) at step: {step}")
        break
else:
    print(f"║    Violence never exceeded {threshold} threshold")

EOF "${OUTPUT_DIR}"

echo "║"
echo "║  View results:"
echo "║    Animation:  eog ${OUTPUT_DIR}/heatmap_evolution.gif"
echo "║    Graph:      eog ${OUTPUT_DIR}/probability_evolution.png"
echo "║    Final:      eog ${OUTPUT_DIR}/final_image.png"
echo "║"
echo "║  Analysis tips:"
echo "║    - Red areas in heatmap = classifier detects violence here"
echo "║    - Early steps (40-50) = high-level structure"
echo "║    - Late steps (0-10) = fine details"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
