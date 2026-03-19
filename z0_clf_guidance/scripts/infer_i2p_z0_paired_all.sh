#!/usr/bin/env bash
# ============================================================================
# [INFERENCE] Latent z0-space PAIRED classifier guidance — ALL 7 I2P concepts
# Prompts: *_high_tox.txt (50 each), 9-class ResNet18 on z0_hat (4ch)
# Guidance: paired => argmax(harm) -> guide toward corresponding safe pair
#   harm = [1,3,5,7],  safe = [2,4,6,8]  (paired: 1->2, 3->4, 5->6, 7->8)
# Guidance scales: 10, 12.5, 15
# ============================================================================
set -euo pipefail

cd /mnt/home/yhgil99/unlearning/z0_clf_guidance
mkdir -p logs

CKPT_PATH="CompVis/stable-diffusion-v1-4"
PROMPT_DIR="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/i2p"

declare -A CKPTS PROMPTS GPUS

CKPTS[violence]="./work_dirs/z0_violence_9class/checkpoint/step_16000/classifier.pth"
CKPTS[harassment]="./work_dirs/z0_harassment_9class/checkpoint/step_20400/classifier.pth"
CKPTS[hate]="./work_dirs/z0_hate_9class/checkpoint/step_24300/classifier.pth"
CKPTS[illegal]="./work_dirs/z0_illegal_9class/checkpoint/step_13200/classifier.pth"
CKPTS[selfharm]="./work_dirs/z0_selfharm_9class/checkpoint/step_23500/classifier.pth"
CKPTS[shocking]="./work_dirs/z0_shocking_9class/checkpoint/step_22700/classifier.pth"
CKPTS[sexual]="./work_dirs/z0_sexual_9class/checkpoint/step_20400/classifier.pth"

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

SCALES="10.0 12.5 15.0"

for scale in ${SCALES}; do
    # Tag for directory/log names (e.g. "10.0" -> "s10")
    tag=$(echo "${scale}" | sed 's/\.0$//' | sed 's/\./_/')
    echo "=============================================="
    echo "[z0 paired] guidance_scale=${scale}  (tag=s${tag})"
    echo "=============================================="

    for concept in violence harassment hate illegal selfharm shocking sexual; do
        gpu="${GPUS[$concept]}"
        ckpt="${CKPTS[$concept]}"
        prompt_file="${PROMPTS[$concept]}"
        outdir="./output_img/i2p_${concept}_z0_paired_s${tag}"

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
          --space latent \
          --guidance_scale "${scale}" \
          --guidance_start_step 1 \
          --guidance_mode paired \
          --safe_classes 2 4 6 8 \
          --harm_classes 1 3 5 7 \
          > "logs/infer_i2p_z0_paired_s${tag}_${concept}.log" 2>&1 &

        echo "  [GPU ${gpu}] ${concept} started. PID=$!"
    done

    echo "Waiting for scale=${scale} to finish..."
    wait
    echo "Done with scale=${scale}."
    echo ""
done

echo "=============================================="
echo "All z0 paired inference complete! (scales: ${SCALES})"
echo "=============================================="
