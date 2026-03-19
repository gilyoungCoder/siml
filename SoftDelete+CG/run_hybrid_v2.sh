#!/bin/bash
# =============================================================================
# Run HYBRID std_sgf + ASCG v2: SGF in x0-space + ASCG in z_t-space
# =============================================================================
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
export PYTHONNOUSERSITE=1

PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
CLASSIFIER_CKPT="work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth"
GRADCAM_STATS_DIR="gradcam_stats/nudity_4class_ringabell"
SGF_REF_PATH="/mnt/home/yhgil99/unlearning/SGF/nudity_sdv1/caches/sd_sgf/i2p_sexual/repellency_proj_ref.pt"
OUTPUT_BASE="scg_outputs/hybrid_v2"
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"

declare -a CONFIGS=(
    # name|start_t|end_t|schedule|ascg_scale|base_scale|sgf_scale|extra
    # Main configs (SGF + ASCG hybrid)
    "hybrid_sigma_w1000-800_gs10_bs2|1000|800|sigma|10.0|2.0|0.03|"
    "hybrid_sigma_w1000-600_gs10_bs2|1000|600|sigma|10.0|2.0|0.03|"
    "hybrid_sigma_w1000-400_gs10_bs2|1000|400|sigma|10.0|2.0|0.03|"
    "hybrid_sigma_w1000-800_gs20_bs3|1000|800|sigma|20.0|3.0|0.03|"
    "hybrid_sigma_w1000-600_gs20_bs3|1000|600|sigma|20.0|3.0|0.03|"
    "hybrid_sigma_w1000-400_gs20_bs3|1000|400|sigma|20.0|3.0|0.03|"
    # Ablation: ASCG-only (no SGF, same as v1 but with sigma schedule)
    "ascg_only_sigma_w1000-400_gs20_bs3|1000|400|sigma|20.0|3.0|0.0|--no_sgf"
    # Ablation: constant schedule (not sigma)
    "hybrid_const_w1000-800_gs20_bs3|1000|800|constant|20.0|3.0|0.03|"
)

GPU_LIST=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPU_LIST[@]}

mkdir -p "${OUTPUT_BASE}/logs"

run_config() {
    local CONFIG="$1"
    local GPU_IDX="$2"

    IFS='|' read -r NAME START_T END_T SCHED ASCG_SC BASE_SC SGF_SC EXTRA <<< "$CONFIG"
    local GPU=${GPU_LIST[$GPU_IDX]}
    local OUT_DIR="${OUTPUT_BASE}/${NAME}"
    local LOG_FILE="${OUTPUT_BASE}/logs/${NAME}.log"

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

# Launch all configs in parallel
PIDS=()
for i in "${!CONFIGS[@]}"; do
    GPU_IDX=$((i % NUM_GPUS))
    run_config "${CONFIGS[$i]}" "$GPU_IDX" &
    PIDS+=($!)
done

echo "Launched ${#CONFIGS[@]} configs on ${NUM_GPUS} GPUs"

# Wait for all
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo ""
echo "============================================"
echo "ALL HYBRID v2 CONFIGS COMPLETE!"
echo "============================================"
