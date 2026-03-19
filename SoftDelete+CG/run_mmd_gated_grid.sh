#!/bin/bash
# =============================================================================
# Grid Search: MMD-Gated ASCG
# SGF decides WHEN (MMD distance gating), ASCG decides WHERE (spatial gradient)
# 8 GPUs parallel
# =============================================================================
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
export PYTHONNOUSERSITE=1

PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
CLASSIFIER_CKPT="work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth"
GRADCAM_STATS_DIR="gradcam_stats/nudity_4class_ringabell"
SGF_REF_PATH="/mnt/home/yhgil99/unlearning/SGF/nudity_sdv1/caches/sd_sgf/i2p_sexual/repellency_proj_ref.pt"
OUTPUT_BASE="scg_outputs/mmd_gated_grid"
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"

# Grid search configs
# name|metric|threshold|window_start|window_end|sticky|decision_step|ascg_scale|base_scale|sp_start|sp_end|sp_strategy|schedule
declare -a CONFIGS=(
    # === kernel_sim metric: higher = more similar to unsafe ===
    # Sweep thresholds with linear schedule, proven ASCG params
    "ksim_t0.05_w1000-400_sticky|kernel_sim|0.05|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "ksim_t0.10_w1000-400_sticky|kernel_sim|0.10|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "ksim_t0.15_w1000-400_sticky|kernel_sim|0.15|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "ksim_t0.20_w1000-400_sticky|kernel_sim|0.20|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "ksim_t0.25_w1000-400_sticky|kernel_sim|0.25|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "ksim_t0.30_w1000-400_sticky|kernel_sim|0.30|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    # Non-sticky (re-check every step)
    "ksim_t0.10_w1000-400_nonsticky|kernel_sim|0.10|1000|400|||20.0|3.0|0.2|0.3|cosine|linear"
    "ksim_t0.20_w1000-400_nonsticky|kernel_sim|0.20|1000|400|||20.0|3.0|0.2|0.3|cosine|linear"
    # Decision at step 5 (early decision, stick with it)
    "ksim_t0.10_w1000-400_dec5|kernel_sim|0.10|1000|400||5|20.0|3.0|0.2|0.3|cosine|linear"
    "ksim_t0.15_w1000-400_dec5|kernel_sim|0.15|1000|400||5|20.0|3.0|0.2|0.3|cosine|linear"
    "ksim_t0.20_w1000-400_dec5|kernel_sim|0.20|1000|400||5|20.0|3.0|0.2|0.3|cosine|linear"
    # === min_dist metric: lower = closer to unsafe ===
    "mindist_t50_w1000-400_sticky|min_dist|50.0|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "mindist_t100_w1000-400_sticky|min_dist|100.0|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "mindist_t150_w1000-400_sticky|min_dist|150.0|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "mindist_t200_w1000-400_sticky|min_dist|200.0|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    # === ASCG-only baseline (no MMD gating, same seed) for fair comparison ===
    "ascg_baseline_seed42|kernel_sim|999.0|1000|400|||20.0|3.0|0.2|0.3|cosine|linear"
)

GPU_LIST=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPU_LIST[@]}

mkdir -p "${OUTPUT_BASE}/logs"

run_config() {
    local CONFIG="$1"
    local GPU_IDX="$2"

    IFS='|' read -r NAME METRIC THRESH W_START W_END STICKY DEC_STEP ASCG_SC BASE_SC SP_START SP_END SP_STRAT SCHED <<< "$CONFIG"
    local GPU=${GPU_LIST[$GPU_IDX]}
    local OUT_DIR="${OUTPUT_BASE}/${NAME}"
    local LOG_FILE="${OUTPUT_BASE}/logs/${NAME}.log"

    # Skip if VLM eval already done
    if [ -f "${OUT_DIR}/categories_qwen3_vl_nudity.json" ]; then
        echo "[GPU ${GPU}] SKIP (eval done): ${NAME}"
        return 0
    fi

    echo "[GPU ${GPU}] START: ${NAME}"

    # Build extra args
    local EXTRA_ARGS=""
    if [ -n "$STICKY" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} ${STICKY}"
    fi
    if [ -n "$DEC_STEP" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --mmd_decision_step ${DEC_STEP}"
    fi

    # Step 1: Generate
    if [ ! -f "${OUT_DIR}/generation_stats.json" ]; then
        CUDA_VISIBLE_DEVICES=${GPU} python generate_mmd_gated_ascg.py \
            --ckpt_path CompVis/stable-diffusion-v1-4 \
            --prompt_file "${PROMPT_FILE}" \
            --output_dir "${OUT_DIR}" \
            --classifier_ckpt "${CLASSIFIER_CKPT}" \
            --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
            --sgf_ref_path "${SGF_REF_PATH}" \
            --mmd_metric "${METRIC}" \
            --mmd_threshold ${THRESH} \
            --mmd_window_start ${W_START} \
            --mmd_window_end ${W_END} \
            --ascg_scale ${ASCG_SC} \
            --ascg_base_scale ${BASE_SC} \
            --spatial_threshold_start ${SP_START} \
            --spatial_threshold_end ${SP_END} \
            --spatial_threshold_strategy "${SP_STRAT}" \
            --guidance_schedule "${SCHED}" \
            ${EXTRA_ARGS} \
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

# Launch configs round-robin across GPUs
PIDS=()
for i in "${!CONFIGS[@]}"; do
    GPU_IDX=$((i % NUM_GPUS))
    run_config "${CONFIGS[$i]}" "$GPU_IDX" &
    PIDS+=($!)
    # Stagger launches slightly to avoid loading model simultaneously
    if [ $((i % NUM_GPUS)) -eq $((NUM_GPUS - 1)) ]; then
        # Wait for current batch before starting next
        for pid in "${PIDS[@]}"; do
            wait $pid
        done
        PIDS=()
    fi
done

# Wait for remaining
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo ""
echo "============================================"
echo "ALL MMD-GATED GRID SEARCH COMPLETE!"
echo "============================================"

# Quick summary
echo ""
echo "Results summary:"
for dir in ${OUTPUT_BASE}/*/; do
    name=$(basename "$dir")
    json="${dir}/categories_qwen3_vl_nudity.json"
    stats="${dir}/generation_stats.json"
    if [ -f "$json" ]; then
        python3 -c "
import json
d = json.load(open('$json'))
cats = [v.get('category','') for v in d.values()]
t = len(cats)
safe = cats.count('Safe')
partial = cats.count('Partial')
full = cats.count('Full')
nr = cats.count('NotRel')
sr = safe + partial
# Also get trigger rate
s = json.load(open('$stats'))
no_guid = s['overall'].get('no_guidance_count', 0)
triggered = s['overall'].get('triggered_count', t)
print(f'  {\"$name\":<50s} SR={100*sr/t:5.1f}% (S={safe} P={partial} F={full} NR={nr}) triggered={triggered}/{t}')
" 2>/dev/null
    fi
done
