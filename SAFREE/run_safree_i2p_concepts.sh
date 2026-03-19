#!/usr/bin/env bash
set -Eeuo pipefail

# ===============================
# SAFREE Inference for I2P Concepts
# 6 concepts: harassment, hate, illegal activity, self-harm, sexual, shocking, violence
# ===============================

# GPU selection (modify as needed)
export CUDA_VISIBLE_DEVICES=0

# Paths
PROMPT_DIR="/mnt/home/yhgil99/guided2-safe-diffusion/prompts/i2p/"
OUTDIR="./safree_outputs/i2p"

# Model
MODEL_ID="CompVis/stable-diffusion-v1-4"

# Generation settings
SEED=42
STEPS=50
GUIDANCE=7.5
NUM_IMAGES=1

# SAFREE settings
SF_ALPHA=0.01
RE_ATTN_T="-1,4"
UP_T=10
FREEU_HYP="1.0-1.0-0.9-0.2"

echo "=============================================="
echo "SAFREE I2P Concepts Inference"
echo "=============================================="
echo "Prompt Dir: ${PROMPT_DIR}"
echo "Output Dir: ${OUTDIR}"
echo "Model: ${MODEL_ID}"
echo "Seed: ${SEED}"
echo "=============================================="

python gen_safree_i2p_concepts.py \
    --prompt_dir "${PROMPT_DIR}" \
    --outdir "${OUTDIR}" \
    --model_id "${MODEL_ID}" \
    --seed ${SEED} \
    --steps ${STEPS} \
    --guidance ${GUIDANCE} \
    --num_images ${NUM_IMAGES} \
    --sf_alpha ${SF_ALPHA} \
    --re_attn_t="-1,4" \
    --up_t ${UP_T} \
    --freeu_hyp "${FREEU_HYP}" \
    --safree \
    --lra \
    --svf \
    --concepts "harassment" "hate" "illegal activity" "self-harm" "sexual" "shocking" "violence"

echo "=============================================="
echo "DONE!"
echo "=============================================="
