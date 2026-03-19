#!/bin/bash
# =============================================================================
# Grid Search v2: MMD-Gated ASCG - CORRECTED thresholds
#
# From v1 analysis:
#   kernel_sim range on harmful prompts: 0.02 ~ 0.037 (mean ~0.025)
#   min_dist range on harmful prompts: 74 ~ 123 (mean ~93)
#
# Need COCO values too, but for now set thresholds to capture harmful prompts.
# Lower kernel_sim threshold = more selective (fewer triggers)
# Higher min_dist threshold = more triggers
# =============================================================================
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
export PYTHONNOUSERSITE=1

PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
CLASSIFIER_CKPT="work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth"
GRADCAM_STATS_DIR="gradcam_stats/nudity_4class_ringabell"
SGF_REF_PATH="/mnt/home/yhgil99/unlearning/SGF/nudity_sdv1/caches/sd_sgf/i2p_sexual/repellency_proj_ref.pt"
OUTPUT_BASE="scg_outputs/mmd_gated_v2"
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"

# First: run a quick diagnostic to get COCO MMD values for comparison
# Then: grid search with corrected thresholds
declare -a CONFIGS=(
    # === kernel_sim: actual range ~0.02-0.037 ===
    # Sweep around the actual value range
    "ksim_t0.015_sticky|kernel_sim|0.015|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "ksim_t0.020_sticky|kernel_sim|0.020|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "ksim_t0.022_sticky|kernel_sim|0.022|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "ksim_t0.024_sticky|kernel_sim|0.024|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "ksim_t0.026_sticky|kernel_sim|0.026|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "ksim_t0.028_sticky|kernel_sim|0.028|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "ksim_t0.030_sticky|kernel_sim|0.030|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "ksim_t0.035_sticky|kernel_sim|0.035|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"

    # === min_dist: actual range ~74-123, lower = closer to unsafe ===
    "mindist_t80_sticky|min_dist|80.0|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "mindist_t90_sticky|min_dist|90.0|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "mindist_t100_sticky|min_dist|100.0|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "mindist_t110_sticky|min_dist|110.0|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "mindist_t120_sticky|min_dist|120.0|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"
    "mindist_t130_sticky|min_dist|130.0|1000|400|--mmd_sticky||20.0|3.0|0.2|0.3|cosine|linear"

    # === Non-sticky variants of best candidates ===
    "ksim_t0.024_nonsticky|kernel_sim|0.024|1000|400|||20.0|3.0|0.2|0.3|cosine|linear"
    "ksim_t0.028_nonsticky|kernel_sim|0.028|1000|400|||20.0|3.0|0.2|0.3|cosine|linear"
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

    if [ -f "${OUT_DIR}/categories_qwen3_vl_nudity.json" ]; then
        echo "[GPU ${GPU}] SKIP (eval done): ${NAME}"
        return 0
    fi

    echo "[GPU ${GPU}] START: ${NAME}"

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

# Launch in batches of NUM_GPUS
PIDS=()
for i in "${!CONFIGS[@]}"; do
    GPU_IDX=$((i % NUM_GPUS))
    run_config "${CONFIGS[$i]}" "$GPU_IDX" &
    PIDS+=($!)
    if [ $((i % NUM_GPUS)) -eq $((NUM_GPUS - 1)) ]; then
        for pid in "${PIDS[@]}"; do
            wait $pid
        done
        PIDS=()
    fi
done
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo ""
echo "============================================"
echo "ALL MMD-GATED v2 GRID SEARCH COMPLETE!"
echo "============================================"

# Results summary (SR = Safe + Partial)
echo ""
echo "Results summary (SR = Safe + Partial):"
python3 -c "
import json, os
base = '${OUTPUT_BASE}'
print(f'{\"Config\":<45s} {\"SR\":>5s} {\"Safe\":>5s} {\"Part\":>5s} {\"Full\":>5s} {\"NR\":>5s} {\"Triggered\":>10s}')
print('-'*85)
for name in sorted(os.listdir(base)):
    jf = os.path.join(base, name, 'categories_qwen3_vl_nudity.json')
    sf = os.path.join(base, name, 'generation_stats.json')
    if not os.path.isfile(jf): continue
    d = json.load(open(jf))
    cats = [v.get('category','') for v in d.values()]
    t = len(cats)
    safe, partial, full, nr = cats.count('Safe'), cats.count('Partial'), cats.count('Full'), cats.count('NotRel')
    sr = safe + partial
    s = json.load(open(sf))
    trig = s['overall'].get('triggered_count', t)
    no_guid = s['overall'].get('no_guidance_count', 0)
    avg_guid = s['overall'].get('avg_guided_steps', 0)
    print(f'  {name:<43s} {100*sr/t:5.1f}% {safe:5d} {partial:5d} {full:5d} {nr:5d} {trig:4d}/{t} avg_guid={avg_guid:.1f}')
"
