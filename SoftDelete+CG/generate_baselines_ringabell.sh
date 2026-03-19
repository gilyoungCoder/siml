#!/bin/bash
# ============================================================================
# Generate Baselines for Ring-A-Bell Dataset
#
# Baselines:
#   1. SD Baseline (vanilla SD v1.4)
#   2. SAFREE Baseline
#
# Usage:
#   ./generate_baselines_ringabell.sh <GPU1> <GPU2>
#   ./generate_baselines_ringabell.sh 0 1
# ============================================================================

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <GPU1> <GPU2>"
    echo "  GPU1: for SD baseline"
    echo "  GPU2: for SAFREE baseline"
    echo "Example: $0 0 1"
    exit 1
fi

GPU1="$1"
GPU2="$2"

# ============================================
# Configuration
# ============================================
PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
OUTPUT_BASE="/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/baselines_ringabell"

SD_MODEL="CompVis/stable-diffusion-v1-4"
NUM_STEPS=50
SEED=42
CFG_SCALE=7.5

mkdir -p "${OUTPUT_BASE}"
LOG_DIR="${OUTPUT_BASE}/logs"
mkdir -p "${LOG_DIR}"

echo "=============================================="
echo "Generate Baselines for Ring-A-Bell (Parallel)"
echo "=============================================="
echo "GPU1 (SD Baseline): ${GPU1}"
echo "GPU2 (SAFREE): ${GPU2}"
echo "Prompt file: ${PROMPT_FILE}"
echo "Output: ${OUTPUT_BASE}"
echo "Logs: ${LOG_DIR}"
echo "=============================================="
echo ""

# ============================================
# 1. SD Baseline (Vanilla SD v1.4) - Background
# ============================================
echo "[1/2] Starting SD Baseline on GPU ${GPU1}..."
SD_OUTPUT="${OUTPUT_BASE}/sd_baseline"
SD_LOG="${LOG_DIR}/sd_baseline.log"

(
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

CUDA_VISIBLE_DEVICES=${GPU1} python -c "
import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path
import csv

# Load prompts
prompts = []
with open('${PROMPT_FILE}', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if 'sensitive prompt' in row:
            prompts.append(row['sensitive prompt'].strip())
        elif 'prompt' in row:
            prompts.append(row['prompt'].strip())

print(f'Loaded {len(prompts)} prompts')

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    '${SD_MODEL}',
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to('cuda')

# Generate
output_dir = Path('${SD_OUTPUT}')
output_dir.mkdir(parents=True, exist_ok=True)

generator = torch.Generator(device='cuda').manual_seed(${SEED})

for i, prompt in enumerate(prompts):
    print(f'[{i+1}/{len(prompts)}] {prompt[:50]}...')

    image = pipe(
        prompt,
        num_inference_steps=${NUM_STEPS},
        guidance_scale=${CFG_SCALE},
        generator=generator
    ).images[0]

    image.save(output_dir / f'{i:06d}.png')

    # Reset generator for next prompt with different seed
    generator = torch.Generator(device='cuda').manual_seed(${SEED})

print(f'Saved {len(prompts)} images to ${SD_OUTPUT}')
"

echo "[1/2] SD Baseline done!"
) > "${SD_LOG}" 2>&1 &
SD_PID=$!

# ============================================
# 2. SAFREE Baseline - Background
# ============================================
echo "[2/2] Starting SAFREE Baseline on GPU ${GPU2}..."
SAFREE_OUTPUT="${OUTPUT_BASE}/safree"
SAFREE_LOG="${LOG_DIR}/safree.log"

(
cd /mnt/home/yhgil99/unlearning/SAFREE

# Pure SAFREE baseline using gen_safree_i2p_concepts.py
# concept=sexual for nudity erasing
CUDA_VISIBLE_DEVICES=${GPU2} python gen_safree_i2p_concepts.py \
    --prompt_file "${PROMPT_FILE}" \
    --concepts "sexual" \
    --model_id "CompVis/stable-diffusion-v1-4" \
    --outdir "${SAFREE_OUTPUT}" \
    --num_images 1 \
    --steps ${NUM_STEPS} \
    --guidance ${CFG_SCALE} \
    --seed ${SEED} \
    --safree \
    --svf \
    --up_t 10 \
    --device "cuda:0" \
    --no_concept_subdir

echo "[2/2] SAFREE Baseline done!"
) > "${SAFREE_LOG}" 2>&1 &
SAFREE_PID=$!

# ============================================
# Wait for both processes
# ============================================
echo ""
echo "Both baselines running in parallel..."
echo "  SD Baseline PID: ${SD_PID} (GPU ${GPU1}) -> ${SD_LOG}"
echo "  SAFREE PID: ${SAFREE_PID} (GPU ${GPU2}) -> ${SAFREE_LOG}"
echo ""
echo "Monitor with: tail -f ${LOG_DIR}/*.log"
echo ""
echo "Waiting for completion..."

wait ${SD_PID}
SD_STATUS=$?
echo "SD Baseline finished with status: ${SD_STATUS}"

wait ${SAFREE_PID}
SAFREE_STATUS=$?
echo "SAFREE Baseline finished with status: ${SAFREE_STATUS}"

# Show errors if any
if [ ${SD_STATUS} -ne 0 ]; then
    echo ""
    echo "[ERROR] SD Baseline failed! Last 20 lines of log:"
    tail -20 "${SD_LOG}"
fi

if [ ${SAFREE_STATUS} -ne 0 ]; then
    echo ""
    echo "[ERROR] SAFREE Baseline failed! Last 20 lines of log:"
    tail -20 "${SAFREE_LOG}"
fi

echo ""
echo "=============================================="
echo "All Baselines Generated!"
echo "=============================================="
echo "SD Baseline: ${SD_OUTPUT}"
echo "SAFREE Baseline: ${SAFREE_OUTPUT}"
echo ""
echo "Next: Run VLM evaluation"
echo "  cd /mnt/home/yhgil99/unlearning && ./vlm/eval_baselines.sh <GPU>"
echo "=============================================="
