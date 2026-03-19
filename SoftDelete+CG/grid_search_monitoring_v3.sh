#!/bin/bash
# ============================================================================
# Grid Search v3: Fine-tune best monitoring methods
#
# 1. grad_norm_sticky: fine-grained thresholds around sweet spot (0.3-0.8)
# 2. noise_div_free_sticky: corrected thresholds (0.5-15)
# 3. noise_div_sticky: corrected thresholds (0.5-15)
# ============================================================================

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd
export PYTHONNOUSERSITE=1

# Parse args
DRY_RUN=true
USE_NOHUP=false
NUM_GPUS=8

while [[ $# -gt 0 ]]; do
    case $1 in
        --run) DRY_RUN=false; shift;;
        --nohup) USE_NOHUP=true; shift;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

PROMPT_FILE="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
END_IDX=-1

if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="./scg_outputs/monitoring_v3_grid_${TIMESTAMP}.log"
    echo "Running in background... Log: $LOG_FILE"
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    nohup bash "$SCRIPT_PATH" --run --num-gpus "$NUM_GPUS" > "$LOG_FILE" 2>&1 &
    echo "PID: $!"; echo "Monitor: tail -f $LOG_FILE"; exit 0
fi

CLASSIFIER_CKPT="work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth"
GRADCAM_STATS_DIR="gradcam_stats/nudity_4class_ringabell"
SD_CKPT="CompVis/stable-diffusion-v1-4"
GEN_SCRIPT="generate_unified_monitoring.py"
VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
OUTPUT_BASE="./scg_outputs/monitoring_v3_grid/ringabell"

NUM_STEPS=50; SEED=42; NSAMPLES=1; CFG_SCALE=7.5
THRESHOLD_STRATEGY="cosine"
HARMFUL_KW="nude naked breast topless bare undress nsfw"

GS=10.0; BS=3.0
SP_PAIRS=("0.2 0.4" "0.3 0.5")

declare -a ALL_COMBOS=()

# ============================================
# 1. grad_norm_sticky fine-tune (thr 0.2-0.8 step 0.1)
# ============================================
for thr in 0.2 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 0.8; do
    for sp in "${SP_PAIRS[@]}"; do
        SP_S=$(echo $sp | awk '{print $1}')
        SP_E=$(echo $sp | awk '{print $2}')
        ALL_COMBOS+=("grad_norm_sticky|${thr}|${GS}|${BS}|${SP_S}|${SP_E}|none")
    done
done
# 10 × 2 = 20

# ============================================
# 2. noise_div_free_sticky with correct thresholds
# ============================================
for thr in 0.5 1.0 2.0 5.0 8.0 10.0 15.0 20.0; do
    for sp in "${SP_PAIRS[@]}"; do
        SP_S=$(echo $sp | awk '{print $1}')
        SP_E=$(echo $sp | awk '{print $2}')
        ALL_COMBOS+=("noise_div_free_sticky|${thr}|${GS}|${BS}|${SP_S}|${SP_E}|none")
    done
done
# 8 × 2 = 16

# ============================================
# 3. noise_div_sticky with correct thresholds
# ============================================
for thr in 0.5 1.0 2.0 5.0 10.0 15.0; do
    for sp in "${SP_PAIRS[@]}"; do
        SP_S=$(echo $sp | awk '{print $1}')
        SP_E=$(echo $sp | awk '{print $2}')
        ALL_COMBOS+=("noise_div_sticky|${thr}|${GS}|${BS}|${SP_S}|${SP_E}|noise_div_safe")
    done
done
# 6 × 2 = 12

# ============================================
# 4. grad_norm_sticky with higher gs=20
# ============================================
for thr in 0.3 0.5 0.7; do
    for sp in "${SP_PAIRS[@]}"; do
        SP_S=$(echo $sp | awk '{print $1}')
        SP_E=$(echo $sp | awk '{print $2}')
        ALL_COMBOS+=("grad_norm_sticky|${thr}|20.0|${BS}|${SP_S}|${SP_E}|none")
    done
done
# 3 × 2 = 6

# Total: 20 + 16 + 12 + 6 = 54

TOTAL=${#ALL_COMBOS[@]}
echo "=============================================="
echo "MONITORING V3 GRID SEARCH (FINE-TUNE)"
echo "Total: $TOTAL experiments on $NUM_GPUS GPUs"
echo "=============================================="

if [ "$DRY_RUN" = true ]; then
    for (( ci=0; ci<TOTAL; ci++ )); do
        combo="${ALL_COMBOS[$ci]}"
        IFS='|' read -r MMODE THR GS_VAL BS_VAL SP_S SP_E EXTRA <<< "$combo"
        echo "  [$ci] ${MMODE}_thr${THR}_gs${GS_VAL}_bs${BS_VAL}_sp${SP_S}-${SP_E}"
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
    local GPU_ID=$1 MMODE=$2 THR=$3 GS_VAL=$4 BS_VAL=$5 SP_S=$6 SP_E=$7 EXTRA=$8
    local EXP_NAME="${MMODE}_thr${THR}_gs${GS_VAL}_bs${BS_VAL}_sp${SP_S}-${SP_E}"
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
        SAFE_PROMPT="a person wearing clothes"
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
        )

        case "$EXTRA" in
            noise_div_safe) CMD_ARGS+=(--noise_div_safe_prompt "$SAFE_PROMPT") ;;
        esac

        if [ "$IMG_COUNT" -lt 79 ]; then
            CUDA_VISIBLE_DEVICES=$GPU_ID "${CMD_ARGS[@]}" 2>&1
        fi

        if [ ! -f "${OUT_DIR}/results_qwen3_vl_nudity.txt" ]; then
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
    IFS='|' read -r MMODE THR GS_VAL BS_VAL SP_S SP_E EXTRA <<< "$combo"
    GPU=$(wait_for_gpu)
    DONE=$((DONE + 1))
    echo "Progress: $DONE/$TOTAL [combo $ci]"
    run_experiment "$GPU" "$MMODE" "$THR" "$GS_VAL" "$BS_VAL" "$SP_S" "$SP_E" "$EXTRA"
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
