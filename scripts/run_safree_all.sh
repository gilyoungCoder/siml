#!/bin/bash
# ============================================================================
# SAFREE generation for all datasets using gen_safree_single.py
# Usage: bash run_safree_all.sh <GPU>
# ============================================================================

set -e

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=${GPU}

BASE_DIR="/mnt/home/yhgil99/unlearning"
cd "${BASE_DIR}/SAFREE"

echo "=============================================="
echo "SAFREE Generation - GPU ${GPU}"
echo "=============================================="

# Common params
MODEL="CompVis/stable-diffusion-v1-4"
SEED=42
STEPS=50
CFG=7.5

# ============================================================================
# Nudity datasets
# ============================================================================

# I2P (142 prompts)
echo "[1/4] I2P (142 prompts)..."
python gen_safree_single.py \
    --txt "${BASE_DIR}/prompts/nudity_datasets/nudity.txt" \
    --save-dir "${BASE_DIR}/outputs/nudity_datasets/i2p/safree" \
    --model_id "${MODEL}" --seed ${SEED} --num_inference_steps ${STEPS} --guidance_scale ${CFG} \
    --category "nudity" --safree -svf -lra --sf_alpha 0.01 --re_attn_t="-1,4" --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2"
mv "${BASE_DIR}/outputs/nudity_datasets/i2p/safree/generated/"* "${BASE_DIR}/outputs/nudity_datasets/i2p/safree/" 2>/dev/null || true

# MMA (1000 prompts)
echo "[2/4] MMA (1000 prompts)..."
python gen_safree_single.py \
    --txt "${BASE_DIR}/prompts/nudity_datasets/mma.txt" \
    --save-dir "${BASE_DIR}/outputs/nudity_datasets/mma/safree" \
    --model_id "${MODEL}" --seed ${SEED} --num_inference_steps ${STEPS} --guidance_scale ${CFG} \
    --category "nudity" --safree -svf -lra --sf_alpha 0.01 --re_attn_t="-1,4" --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2"
mv "${BASE_DIR}/outputs/nudity_datasets/mma/safree/generated/"* "${BASE_DIR}/outputs/nudity_datasets/mma/safree/" 2>/dev/null || true

# Ring-a-Bell (79 prompts)
echo "[3/4] Ring-a-Bell (79 prompts)..."
python gen_safree_single.py \
    --txt "${BASE_DIR}/prompts/nudity_datasets/ringabell.txt" \
    --save-dir "${BASE_DIR}/outputs/nudity_datasets/ringabell/safree" \
    --model_id "${MODEL}" --seed ${SEED} --num_inference_steps ${STEPS} --guidance_scale ${CFG} \
    --category "nudity" --safree -svf -lra --sf_alpha 0.01 --re_attn_t="-1,4" --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2"
mv "${BASE_DIR}/outputs/nudity_datasets/ringabell/safree/generated/"* "${BASE_DIR}/outputs/nudity_datasets/ringabell/safree/" 2>/dev/null || true

# ============================================================================
# COCO (10k prompts)
# ============================================================================
echo "[4/4] COCO (10k prompts)..."
python gen_safree_single.py \
    --txt "${BASE_DIR}/prompts/coco/coco_10k.txt" \
    --save-dir "${BASE_DIR}/outputs/coco/safree" \
    --model_id "${MODEL}" --seed ${SEED} --num_inference_steps ${STEPS} --guidance_scale ${CFG} \
    --category "nudity" --safree -svf -lra --sf_alpha 0.01 --re_attn_t="-1,4" --up_t 10 --freeu_hyp "1.0-1.0-0.9-0.2"
mv "${BASE_DIR}/outputs/coco/safree/generated/"* "${BASE_DIR}/outputs/coco/safree/" 2>/dev/null || true

echo "=============================================="
echo "DONE!"
echo "=============================================="
