#!/bin/bash
# ============================================================================
# Generate images for COCO 10k prompts across multiple GPUs
#
# Supports: sd_baseline, sdd, esd
# Each method gets its own subdirectory under final_coco/
#
# Usage:
#   ./scripts/generate_coco.sh --methods "sd_baseline sdd esd" --gpus 0,1,2,3,4,5,6,7
#   ./scripts/generate_coco.sh --methods "sdd esd" --gpus 0,1
# ============================================================================

set -e

GPU_LIST="0,1,2,3,4,5,6,7"
METHODS_ARG="sd_baseline sdd esd"

while [ $# -gt 0 ]; do
    case "$1" in
        --gpus)    GPU_LIST="$2"; shift 2 ;;
        --methods) METHODS_ARG="$2"; shift 2 ;;
        *)         echo "Unknown: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}
read -ra METHODS <<< "$METHODS_ARG"

BASE_DIR="/mnt/home/yhgil99/unlearning"
PROMPT_CSV="${BASE_DIR}/SAFREE/datasets/coco_30k_10k.csv"
OUTPUT_BASE="${BASE_DIR}/SoftDelete+CG/scg_outputs/final_coco"
SD_MODEL="CompVis/stable-diffusion-v1-4"
LOG_DIR="${BASE_DIR}/SoftDelete+CG/scg_outputs/logs_coco"

SDD_CKPT="/mnt/home/yhgil99/guided2-safe-diffusion/Continual2/sdd_2026-01-29_17-05-34"
ESD_CKPT="/mnt/home/yhgil99/guided2-safe-diffusion/Continual2/esd_2026-01-29_17-05-34"

TOTAL_PROMPTS=$(($(wc -l < "$PROMPT_CSV") - 1))

mkdir -p "$LOG_DIR"

echo "=============================================="
echo "COCO 10k Generation"
echo "=============================================="
echo "Methods: ${METHODS[*]}"
echo "GPUs: ${GPUS[*]} (${NUM_GPUS})"
echo "Total prompts: ${TOTAL_PROMPTS}"
echo "=============================================="
echo ""

# Run methods sequentially, each method uses all GPUs in parallel
for method in "${METHODS[@]}"; do
    output_dir="${OUTPUT_BASE}/${method}"
    mkdir -p "$output_dir"

    # Determine unet_path arg
    UNET_ARG=""
    case "$method" in
        sd_baseline) ;;
        sdd) UNET_ARG="--unet_path ${SDD_CKPT}/step=001000" ;;
        esd) UNET_ARG="--unet_path ${ESD_CKPT}/step=001000" ;;
        *)   echo "Unknown method: $method"; exit 1 ;;
    esac

    CHUNK=$((TOTAL_PROMPTS / NUM_GPUS))
    PIDS=()

    echo "--- ${method} ---"
    for i in "${!GPUS[@]}"; do
        gpu="${GPUS[$i]}"
        start=$((i * CHUNK))
        if [ $i -eq $((NUM_GPUS - 1)) ]; then
            end=$TOTAL_PROMPTS
        else
            end=$(((i + 1) * CHUNK))
        fi

        log="${LOG_DIR}/${method}_gpu${gpu}.log"
        echo "  [GPU ${gpu}] prompts ${start}-${end}"

        CUDA_VISIBLE_DEVICES=$gpu python /mnt/home/yhgil99/guided2-safe-diffusion/generate.py \
            --pretrained_model_name_or_path "${SD_MODEL}" \
            ${UNET_ARG} \
            --image_dir "${output_dir}" \
            --prompt_path "${PROMPT_CSV}" \
            --num_images_per_prompt 1 \
            --num_inference_steps 50 \
            --seed 42 \
            --start $start \
            --end $end \
            --device "cuda:0" \
            > "$log" 2>&1 &

        PIDS+=($!)
    done

    # Wait for this method to finish before starting next
    # (avoid loading multiple models simultaneously)
    FAILED=0
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" 2>/dev/null
        status=$?
        if [ $status -ne 0 ]; then
            echo "  [FAIL] GPU ${GPUS[$i]} (exit ${status})"
            FAILED=$((FAILED + 1))
        else
            echo "  [DONE] GPU ${GPUS[$i]}"
        fi
    done

    if [ $FAILED -eq 0 ]; then
        echo "  ${method} COMPLETE"
    else
        echo "  ${method}: ${FAILED}/${NUM_GPUS} failed"
    fi
    echo ""
done

echo "=============================================="
echo "ALL COMPLETE: ${OUTPUT_BASE}"
echo "=============================================="
