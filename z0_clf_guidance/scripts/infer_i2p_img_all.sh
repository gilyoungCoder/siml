#!/usr/bin/env bash
# ============================================================================
# [INFERENCE] Image-space z0 classifier guidance — ALL 7 I2P concepts
# Prompts: *_high_tox.txt (50 each), 9-class ResNet18 on decoded x0_hat
# Guidance: safe_minus_harm => max log p(safe) - log p(harm)
#   safe = [2,4,6,8],  harm = [1,3,5,7]
# ============================================================================
set -euo pipefail

cd /mnt/home/yhgil99/unlearning/z0_clf_guidance
mkdir -p logs

CKPT_PATH="CompVis/stable-diffusion-v1-4"
PROMPT_DIR="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/i2p"

declare -A CKPTS PROMPTS GPUS

CKPTS[violence]="./work_dirs/z0_img_violence_9class/checkpoint/step_23500/classifier.pth"
CKPTS[harassment]="./work_dirs/z0_img_harassment_9class/checkpoint/step_20900/classifier.pth"
CKPTS[hate]="./work_dirs/z0_img_hate_9class/checkpoint/step_20100/classifier.pth"
CKPTS[illegal]="./work_dirs/z0_img_illegal_9class/checkpoint/step_12500/classifier.pth"
CKPTS[selfharm]="./work_dirs/z0_img_selfharm_9class/checkpoint/step_19700/classifier.pth"
CKPTS[shocking]="./work_dirs/z0_img_shocking_9class/checkpoint/step_22400/classifier.pth"
CKPTS[sexual]="./work_dirs/z0_img_sexual_9class/checkpoint/step_24800/classifier.pth"

PROMPTS[violence]="${PROMPT_DIR}/violence_high_tox.txt"
PROMPTS[harassment]="${PROMPT_DIR}/harassment_high_tox.txt"
PROMPTS[hate]="${PROMPT_DIR}/hate_high_tox.txt"
PROMPTS[illegal]="${PROMPT_DIR}/illegal_activity_high_tox.txt"
PROMPTS[selfharm]="${PROMPT_DIR}/self-harm_high_tox.txt"
PROMPTS[shocking]="${PROMPT_DIR}/shocking_high_tox.txt"
PROMPTS[sexual]="${PROMPT_DIR}/sexual_high_tox.txt"

GPUS[violence]=0
GPUS[harassment]=1
GPUS[hate]=2
GPUS[illegal]=3
GPUS[selfharm]=4
GPUS[shocking]=5
GPUS[sexual]=6

echo "Launching all I2P image-space inference jobs (high_tox, 50 prompts each)..."
echo ""

for concept in violence harassment hate illegal selfharm shocking sexual; do
    gpu="${GPUS[$concept]}"
    ckpt="${CKPTS[$concept]}"
    prompt_file="${PROMPTS[$concept]}"
    outdir="./output_img/i2p_${concept}_img_safe_minus_harm"

    CUDA_VISIBLE_DEVICES=${gpu} nohup python generate.py "${CKPT_PATH}" \
      --prompt_file "${prompt_file}" \
      --output_dir "${outdir}" \
      --nsamples 1 \
      --cfg_scale 7.5 \
      --num_inference_steps 50 \
      --seed 1234 \
      --classifier_ckpt "${ckpt}" \
      --architecture resnet18 \
      --num_classes 9 \
      --space image \
      --guidance_scale 5.0 \
      --guidance_start_step 1 \
      --guidance_mode safe_minus_harm \
      --safe_classes 2 4 6 8 \
      --harm_classes 1 3 5 7 \
      > "logs/infer_i2p_img_${concept}.log" 2>&1 &

    echo "[GPU ${gpu}] ${concept} (img, 9-class, 50 prompts) started. PID=$!"
done

echo ""
echo "=============================================="
echo "All 7 I2P image-space inference jobs launched!"
echo "  tail -f logs/infer_i2p_img_*.log"
echo "=============================================="
