#!/bin/bash
# ============================================================================
# SAFREE+Ours with text-based early exit on all datasets
# Config: safree_ours (mon0.2_gs5_bs2.0_sp0.7-0.3) + text_exit=0.50
# Text exit skips CG only; SAFREE still runs normally
# ============================================================================

cd /mnt/home/yhgil99/unlearning

SCRIPT="SAFREE/generate_safree_monitoring.py"
CLASSIFIER_CKPT="SoftDelete+CG/work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS="SoftDelete+CG/gradcam_stats/nudity_4class"

# safree_ours config
MON=0.2
GS=5.0
BS=2.0
SP_S=0.7
SP_E=0.3
TEXT_THR=0.50

NUM_STEPS=50
SEED=42
CFG_SCALE=7.5

USE_NOHUP=false
NUM_GPUS=8
DATASET="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --nohup) USE_NOHUP=true; shift;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        --text-thr) TEXT_THR="$2"; shift 2;;
        --dataset) DATASET="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="scripts/logs/safree_ours_text_exit_${TIMESTAMP}.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "Running in background... Log: $LOG_FILE"
    nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$0" --num-gpus "$NUM_GPUS" --text-thr "$TEXT_THR" --dataset "$DATASET" > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

# Datasets
declare -A PROMPTS
declare -A OUTPUT_DIRS
PROMPTS[coco]="SAFREE/datasets/coco_30k_10k.csv"
PROMPTS[ringabell]="SAFREE/datasets/nudity-ring-a-bell.csv"
PROMPTS[unlearndiff]="SAFREE/datasets/unlearn_diff_nudity.csv"
PROMPTS[mma]="SAFREE/datasets/mma-diffusion-nsfw-adv-prompts.csv"

OUTPUT_DIRS[coco]="SoftDelete+CG/scg_outputs/final_coco/safree_ours_text_exit"
OUTPUT_DIRS[ringabell]="SoftDelete+CG/scg_outputs/final_ringabell/safree_ours_text_exit"
OUTPUT_DIRS[unlearndiff]="SoftDelete+CG/scg_outputs/final_unlearndiff/safree_ours_text_exit"
OUTPUT_DIRS[mma]="SoftDelete+CG/scg_outputs/final_mma/safree_ours_text_exit"

if [ "$DATASET" = "all" ]; then
    DS_LIST=(ringabell unlearndiff mma coco)
else
    DS_LIST=($DATASET)
fi

# GPU setup
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_LIST+=($i); done
fi
NUM_GPUS=${#GPU_LIST[@]}

echo "=============================================="
echo "SAFREE+Ours + Text Exit"
echo "Config: mon${MON}_gs${GS}_bs${BS}_sp${SP_S}-${SP_E}"
echo "Text exit threshold: ${TEXT_THR}"
echo "Datasets: ${DS_LIST[*]}"
echo "GPUs: ${GPU_LIST[*]}"
echo "=============================================="

for DS in "${DS_LIST[@]}"; do
    PROMPT_FILE="${PROMPTS[$DS]}"
    OUTPUT_DIR="${OUTPUT_DIRS[$DS]}"
    mkdir -p "$OUTPUT_DIR/logs"

    # Count prompts
    if [[ "${PROMPT_FILE}" == *.csv ]]; then
        TOTAL_PROMPTS=$(($(wc -l < "${PROMPT_FILE}") - 1))
    else
        TOTAL_PROMPTS=$(wc -l < "${PROMPT_FILE}")
    fi

    CHUNK=$((TOTAL_PROMPTS / NUM_GPUS))
    REMAINDER=$((TOTAL_PROMPTS % NUM_GPUS))

    echo ""
    echo "=== Dataset: ${DS} (${TOTAL_PROMPTS} prompts) ==="
    echo "Output: ${OUTPUT_DIR}"

    PIDS=()
    START=0

    for ((g=0; g<NUM_GPUS; g++)); do
        GPU=${GPU_LIST[$g]}

        if [ $g -lt $REMAINDER ]; then
            END=$((START + CHUNK + 1))
        else
            END=$((START + CHUNK))
        fi

        [ $START -ge $TOTAL_PROMPTS ] && break
        [ $END -gt $TOTAL_PROMPTS ] && END=$TOTAL_PROMPTS

        echo "  [GPU $GPU] prompts ${START}-${END}"

        CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT" \
            --ckpt_path "CompVis/stable-diffusion-v1-4" \
            --prompt_file "$PROMPT_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --classifier_ckpt "$CLASSIFIER_CKPT" \
            --gradcam_stats_dir "$GRADCAM_STATS" \
            --monitoring_threshold $MON \
            --guidance_scale $GS \
            --base_guidance_scale $BS \
            --spatial_threshold_start $SP_S \
            --spatial_threshold_end $SP_E \
            --spatial_threshold_strategy cosine_anneal \
            --safree --svf --svf_up_t 10 \
            --category nudity \
            --text_exit_threshold $TEXT_THR \
            --num_inference_steps $NUM_STEPS \
            --cfg_scale $CFG_SCALE \
            --seed $SEED \
            --nsamples 1 \
            --start_idx $START \
            --end_idx $END \
            > "${OUTPUT_DIR}/logs/gpu${GPU}.log" 2>&1 &
        PIDS+=($!)

        START=$END
    done

    # Wait for current dataset
    FAIL=0
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || FAIL=$((FAIL + 1))
    done

    echo "  ${DS} done. Images: $(ls "$OUTPUT_DIR"/*.png 2>/dev/null | wc -l)"
    [ $FAIL -gt 0 ] && echo "  WARNING: ${FAIL} jobs failed"
done

echo ""
echo "=== ALL DONE ==="
