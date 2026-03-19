#!/bin/bash
# =============================================================================
# Run std_sgf + ASCG combined pipeline with multiple configurations
# =============================================================================
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
export PYTHONNOUSERSITE=1

PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
CLASSIFIER_CKPT="work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth"
GRADCAM_STATS_DIR="gradcam_stats/nudity_4class_ringabell"
SGF_REF_PATH="/mnt/home/yhgil99/unlearning/SGF/nudity_sdv1/caches/sd_sgf/i2p_sexual/repellency_proj_ref.pt"
OUTPUT_BASE="scg_outputs/std_sgf_ascg"
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"

# Configurations to test:
# 1. std_sgf + ASCG (sigma schedule, w[1000,800])
# 2. std_sgf + ASCG (sigma schedule, w[1000,600])  - wider window
# 3. std_sgf + ASCG (sigma schedule, w[1000,400])  - widest window
# 4. ASCG only (no SGF, sigma schedule, w[1000,800]) - ablation
# 5. SGF only (no ASCG, sigma schedule, w[1000,800]) - ablation
# 6. std_sgf + ASCG (constant schedule, w[1000,800]) - schedule ablation

declare -a CONFIGS=(
    # format: "name|start_t|end_t|schedule|ascg_scale|base_scale|sgf_scale|extra_flags"
    "sgf_ascg_sigma_w1000-800|1000|800|sigma|10.0|2.0|0.03|"
    "sgf_ascg_sigma_w1000-600|1000|600|sigma|10.0|2.0|0.03|"
    "sgf_ascg_sigma_w1000-400|1000|400|sigma|10.0|2.0|0.03|"
    "ascg_only_sigma_w1000-800|1000|800|sigma|10.0|2.0|0.0|--no_sgf"
    "sgf_only_sigma_w1000-800|1000|800|sigma|0.0|0.0|0.03|--no_ascg"
    "sgf_ascg_const_w1000-800|1000|800|constant|10.0|2.0|0.03|"
    "sgf_ascg_sigma_w1000-800_gs20|1000|800|sigma|20.0|3.0|0.03|"
    "sgf_ascg_sigma_w1000-600_gs20|1000|600|sigma|20.0|3.0|0.03|"
)

GPU_LIST=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPU_LIST[@]}

run_config() {
    local CONFIG="$1"
    local GPU_IDX="$2"

    IFS='|' read -r NAME START_T END_T SCHED ASCG_SC BASE_SC SGF_SC EXTRA <<< "$CONFIG"
    local GPU=${GPU_LIST[$GPU_IDX]}
    local OUT_DIR="${OUTPUT_BASE}/${NAME}"
    local LOG_FILE="${OUTPUT_BASE}/logs/${NAME}.log"

    mkdir -p "${OUTPUT_BASE}/logs"

    # Skip if already done
    if [ -f "${OUT_DIR}/categories_qwen3_vl_nudity.json" ]; then
        echo "[GPU ${GPU}] SKIP (eval done): ${NAME}"
        return 0
    fi

    echo "[GPU ${GPU}] START: ${NAME}"

    # Step 1: Generate
    if [ ! -f "${OUT_DIR}/generation_stats.json" ]; then
        CUDA_VISIBLE_DEVICES=${GPU} python generate_std_sgf_ascg.py \
            --ckpt_path CompVis/stable-diffusion-v1-4 \
            --prompt_file "${PROMPT_FILE}" \
            --output_dir "${OUT_DIR}" \
            --classifier_ckpt "${CLASSIFIER_CKPT}" \
            --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
            --sgf_ref_path "${SGF_REF_PATH}" \
            --guidance_start_t ${START_T} \
            --guidance_end_t ${END_T} \
            --guidance_schedule "${SCHED}" \
            --ascg_scale ${ASCG_SC} \
            --ascg_base_scale ${BASE_SC} \
            --sgf_scale ${SGF_SC} \
            ${EXTRA} \
            >> "${LOG_FILE}" 2>&1

        if [ $? -ne 0 ]; then
            echo "[GPU ${GPU}] FAILED generation: ${NAME}"
            return 1
        fi
    fi

    # Step 2: VLM eval
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate vlm
    CUDA_VISIBLE_DEVICES=${GPU} python "${VLM_SCRIPT}" "${OUT_DIR}" nudity qwen \
        >> "${LOG_FILE}" 2>&1
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
    export PYTHONNOUSERSITE=1

    echo "[GPU ${GPU}] DONE: ${NAME}"
}

# Launch all configs in parallel (one per GPU)
PIDS=()
for i in "${!CONFIGS[@]}"; do
    GPU_IDX=$((i % NUM_GPUS))
    run_config "${CONFIGS[$i]}" "$GPU_IDX" &
    PIDS+=($!)
done

echo "Launched ${#CONFIGS[@]} configs on ${NUM_GPUS} GPUs"
echo "PIDs: ${PIDS[@]}"

# Wait for all to finish
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo ""
echo "============================================"
echo "ALL CONFIGS COMPLETE!"
echo "============================================"
echo "Results in: ${OUTPUT_BASE}/"
