#!/bin/bash
# ============================================================================
# Grid Search: Concept Alignment (SDErasure-inspired online "when" detection)
#
# cos(d_prompt, d_target) where:
#   d_prompt = ε(prompt) - ε(∅) (from CFG, free)
#   d_target = ε(target) - ε(∅) (1 extra UNet forward)
#
# Harmful prompts: avg ~0.55-0.71, Safe prompts: avg ~0.0-0.15
# Test thresholds: 0.15 - 0.5
# ============================================================================

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
export PYTHONNOUSERSITE=1

# Parse args
DRY_RUN=true
USE_NOHUP=false
NUM_GPUS=8
DATASET="ringabell"

while [[ $# -gt 0 ]]; do
    case $1 in
        --run) DRY_RUN=false; shift;;
        --nohup) USE_NOHUP=true; shift;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        --dataset) DATASET="$2"; shift 2;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

if [ "$DATASET" = "ringabell" ]; then
    PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
    END_IDX=-1
elif [ "$DATASET" = "coco" ]; then
    PROMPT_FILE="/mnt/home/yhgil99/unlearning/prompts/coco/coco_10k.txt"
    END_IDX=200  # Only 200 COCO for quick quality check
fi

if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="./scg_outputs/concept_align_grid_${DATASET}_${TIMESTAMP}.log"
    echo "Running in background... Log: $LOG_FILE"
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    nohup bash "$SCRIPT_PATH" --run --num-gpus "$NUM_GPUS" --dataset "$DATASET" > "$LOG_FILE" 2>&1 &
    echo "PID: $!"; echo "Monitor: tail -f $LOG_FILE"; exit 0
fi

CLASSIFIER_CKPT="work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth"
GRADCAM_STATS_DIR="gradcam_stats/nudity_4class_ringabell"
SD_CKPT="CompVis/stable-diffusion-v1-4"
GEN_SCRIPT="generate_unified_monitoring.py"
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
OUTPUT_BASE="./scg_outputs/concept_align_grid/${DATASET}"

NUM_STEPS=50; SEED=42; NSAMPLES=1; CFG_SCALE=7.5
THRESHOLD_STRATEGY="cosine"
HARMFUL_KW="nude naked breast topless bare undress nsfw"

# Best spatial CG config from previous experiments
GS=10.0; BS=3.0

declare -a ALL_COMBOS=()

# ============================================
# 1. online_ssscore (non-sticky): threshold sweep
#    Tests if per-step concept alignment alone works
# ============================================
for thr in 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
    for sp in "0.2 0.4" "0.3 0.5"; do
        SP_S=$(echo $sp | awk '{print $1}')
        SP_E=$(echo $sp | awk '{print $2}')
        ALL_COMBOS+=("online_ssscore|${thr}|${GS}|${BS}|${SP_S}|${SP_E}|nudity")
    done
done
# 8 x 2 = 16

# ============================================
# 2. online_ssscore_sticky: threshold sweep
#    Once concept alignment triggers, guide all remaining
# ============================================
for thr in 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
    for sp in "0.2 0.4" "0.3 0.5"; do
        SP_S=$(echo $sp | awk '{print $1}')
        SP_E=$(echo $sp | awk '{print $2}')
        ALL_COMBOS+=("online_ssscore_sticky|${thr}|${GS}|${BS}|${SP_S}|${SP_E}|nudity")
    done
done
# 7 x 2 = 14

# ============================================
# 3. Multi-word target concepts
#    Maybe "nude naked breast nudity" as target is more discriminative
# ============================================
for thr in 0.2 0.3 0.4; do
    ALL_COMBOS+=("online_ssscore_sticky|${thr}|${GS}|${BS}|0.2|0.4|nude naked breast nudity nsfw")
done
# 3

# ============================================
# 4. Higher guidance scale with concept alignment
# ============================================
for thr in 0.25 0.35 0.45; do
    ALL_COMBOS+=("online_ssscore_sticky|${thr}|20.0|${BS}|0.2|0.4|nudity")
done
# 3

# Total: 16 + 14 + 3 + 3 = 36

TOTAL=${#ALL_COMBOS[@]}
echo "=============================================="
echo "CONCEPT ALIGNMENT GRID SEARCH"
echo "Dataset: $DATASET"
echo "Total: $TOTAL experiments on $NUM_GPUS GPUs"
echo "=============================================="

if [ "$DRY_RUN" = true ]; then
    for (( ci=0; ci<TOTAL; ci++ )); do
        combo="${ALL_COMBOS[$ci]}"
        IFS='|' read -r MMODE THR GS_VAL BS_VAL SP_S SP_E TARGET <<< "$combo"
        echo "  [$ci] ${MMODE}_thr${THR}_gs${GS_VAL}_sp${SP_S}-${SP_E}_t-${TARGET// /_}"
    done
    echo "Total: $TOTAL. Add --run to execute."
    exit 0
fi

# GPU setup
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=(); for ((i=0; i<NUM_GPUS; i++)); do GPU_LIST+=($i); done
fi
declare -A GPU_PIDS
for g in "${GPU_LIST[@]}"; do GPU_PIDS[$g]=0; done

wait_for_gpu() {
    while true; do
        for g in "${GPU_LIST[@]}"; do
            pid=${GPU_PIDS[$g]}
            if [ "$pid" -eq 0 ] || ! kill -0 "$pid" 2>/dev/null; then
                GPU_PIDS[$g]=0; echo "$g"; return
            fi
        done
        sleep 5
    done
}

run_experiment() {
    local GPU_ID=$1 MMODE=$2 THR=$3 GS_VAL=$4 BS_VAL=$5 SP_S=$6 SP_E=$7 TARGET=$8
    local TARGET_SLUG="${TARGET// /_}"
    local EXP_NAME="${MMODE}_thr${THR}_gs${GS_VAL}_bs${BS_VAL}_sp${SP_S}-${SP_E}_t-${TARGET_SLUG}"
    local OUT_DIR="${OUTPUT_BASE}/${EXP_NAME}"
    local LOG_DIR="${OUTPUT_BASE}/logs"
    mkdir -p "$LOG_DIR"

    if [ -f "${OUT_DIR}/results_qwen3_vl_nudity.txt" ]; then
        echo "[GPU $GPU_ID] SKIP (done): $EXP_NAME"; return
    fi

    local IMG_COUNT=0
    [ -d "${OUT_DIR}" ] && IMG_COUNT=$(ls "${OUT_DIR}/"*.png 2>/dev/null | wc -l)

    echo "[GPU $GPU_ID] START: $EXP_NAME"

    (
        declare -a CMD_ARGS=(
            python3 "$GEN_SCRIPT"
            --ckpt_path "$SD_CKPT" --prompt_file "$PROMPT_FILE"
            --output_dir "${OUT_DIR}"
            --classifier_ckpt "$CLASSIFIER_CKPT" --gradcam_stats_dir "$GRADCAM_STATS_DIR"
            --monitoring_mode "$MMODE" --monitoring_threshold "$THR"
            --guidance_scale "$GS_VAL" --base_guidance_scale "$BS_VAL"
            --spatial_threshold_start "$SP_S" --spatial_threshold_end "$SP_E"
            --spatial_threshold_strategy "$THRESHOLD_STRATEGY"
            --spatial_mode gradcam
            --num_inference_steps "$NUM_STEPS" --nsamples "$NSAMPLES"
            --seed "$SEED" --cfg_scale "$CFG_SCALE"
            --start_idx 0 --end_idx "$END_IDX"
            --harmful_keywords $HARMFUL_KW
            --ssscore_target "$TARGET"
        )

        if [ "$IMG_COUNT" -lt 79 ] || [ "$DATASET" = "coco" ]; then
            CUDA_VISIBLE_DEVICES=$GPU_ID "${CMD_ARGS[@]}" 2>&1
        fi

        # VLM eval (only for ringabell)
        if [ "$DATASET" = "ringabell" ] && [ ! -f "${OUT_DIR}/results_qwen3_vl_nudity.txt" ]; then
            eval "$(conda shell.bash hook 2>/dev/null)" && conda activate vlm
            export PYTHONNOUSERSITE=1; export PYTHONPATH=""
            CUDA_VISIBLE_DEVICES=$GPU_ID /mnt/home/yhgil99/.conda/envs/vlm/bin/python \
                "$VLM_SCRIPT" "${OUT_DIR}" nudity qwen 2>&1
        fi
    ) > "${LOG_DIR}/${EXP_NAME}.log" 2>&1 &

    GPU_PIDS[$GPU_ID]=$!
}

# Main loop
DONE=0
for (( ci=0; ci<TOTAL; ci++ )); do
    combo="${ALL_COMBOS[$ci]}"
    IFS='|' read -r MMODE THR GS_VAL BS_VAL SP_S SP_E TARGET <<< "$combo"
    GPU=$(wait_for_gpu)
    DONE=$((DONE + 1))
    echo "Progress: $DONE/$TOTAL [combo $ci]"
    run_experiment "$GPU" "$MMODE" "$THR" "$GS_VAL" "$BS_VAL" "$SP_S" "$SP_E" "$TARGET"
done

echo "Waiting for remaining jobs..."
wait

echo ""
echo "=== TOP RESULTS ==="
for d in ${OUTPUT_BASE}/*/; do
    f="${d}results_qwen3_vl_nudity.txt"
    if [ -f "$f" ]; then
        sr=$(grep "SR " "$f" | grep -oP '[\d.]+%' | head -1)
        echo "SR=${sr} $(basename $d)"
    fi
done | sort -t= -k2 -rn | head -25
echo "Done!"
