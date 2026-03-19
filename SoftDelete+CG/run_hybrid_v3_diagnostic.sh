#!/bin/bash
# =============================================================================
# Diagnostic: Test ascg_only with v1-matched spatial thresholds
# + tiny SGF scale to see if minimal SGF helps
# =============================================================================
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
export PYTHONNOUSERSITE=1

PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
CLASSIFIER_CKPT="work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth"
GRADCAM_STATS_DIR="gradcam_stats/nudity_4class_ringabell"
SGF_REF_PATH="/mnt/home/yhgil99/unlearning/SGF/nudity_sdv1/caches/sd_sgf/i2p_sexual/repellency_proj_ref.pt"
OUTPUT_BASE="scg_outputs/hybrid_v3_diag"
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"

declare -a CONFIGS=(
    # name|start_t|end_t|schedule|ascg_scale|base_scale|sgf_scale|sp_start|sp_end|sp_strategy|extra
    # 1. ascg_only with v1's exact spatial params (linear schedule, sp 0.2-0.4)
    "ascg_v1match_linear_w1000-400_gs20_bs3|1000|400|linear|20.0|3.0|0.0|0.2|0.4|linear|--no_sgf"
    # 2. ascg_only with sigma schedule but v1's spatial params
    "ascg_sigma_w1000-400_gs20_bs3_sp02-04|1000|400|sigma|20.0|3.0|0.0|0.2|0.4|linear|--no_sgf"
    # 3. hybrid with TINY sgf scale (0.003 instead of 0.03) + v1 spatial
    "hybrid_sigma_w1000-400_gs20_bs3_sgf003|1000|400|sigma|20.0|3.0|0.003|0.2|0.4|linear|"
    # 4. hybrid with very tiny sgf (0.001) + v1 spatial
    "hybrid_sigma_w1000-400_gs20_bs3_sgf001|1000|400|sigma|20.0|3.0|0.001|0.2|0.4|linear|"
    # 5. hybrid narrow window w1000-800 tiny sgf + v1 spatial
    "hybrid_sigma_w1000-800_gs20_bs3_sgf003|1000|800|sigma|20.0|3.0|0.003|0.2|0.4|linear|"
    # 6. SGF in z_t space test (no re-noising, just add MMD grad to latents directly)
    # -- this requires code change, skip for now
)

GPU_LIST=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPU_LIST[@]}

mkdir -p "${OUTPUT_BASE}/logs"

run_config() {
    local CONFIG="$1"
    local GPU_IDX="$2"

    IFS='|' read -r NAME START_T END_T SCHED ASCG_SC BASE_SC SGF_SC SP_START SP_END SP_STRAT EXTRA <<< "$CONFIG"
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
            --spatial_threshold_start ${SP_START} \
            --spatial_threshold_end ${SP_END} \
            --spatial_threshold_strategy "${SP_STRAT}" \
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
echo "ALL DIAGNOSTIC CONFIGS COMPLETE!"
echo "============================================"
