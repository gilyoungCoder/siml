#!/bin/bash
# ============================================================================
# SAFREE+Ours v2 COCO generation (mon0.2_gs5_bs1.0_sp0.3-0.3)
# 8 GPUs, split COCO into 8 parts
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash SAFREE/run_safree_ours_v2_coco.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash SAFREE/run_safree_ours_v2_coco.sh --nohup
# ============================================================================

cd /mnt/home/yhgil99/unlearning

SCRIPT="SAFREE/generate_safree_monitoring.py"
OUTPUT_DIR="SoftDelete+CG/scg_outputs/final_coco/safree_ours_v2"
PROMPT_FILE="SAFREE/datasets/coco_30k_10k.csv"

# Nohup handling
if [[ "$1" == "--nohup" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG="${OUTPUT_DIR}/logs/safree_ours_v2_coco_${TIMESTAMP}.log"
    mkdir -p "$(dirname "$LOG")"
    echo "Running in background... Log: $LOG"
    SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$SCRIPT_PATH" > "$LOG" 2>&1 &
    echo "PID: $!"
    exit 0
fi

# GPU setup
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=(0 1 2 3 4 5 6 7)
fi
NUM_GPUS=${#GPU_LIST[@]}

# Total prompts
TOTAL=$(python3 -c "
import csv
with open('$PROMPT_FILE') as f:
    print(sum(1 for _ in csv.DictReader(f)))
")
CHUNK=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))

echo "=============================================="
echo "SAFREE+Ours v2 COCO Generation"
echo "Config: mon0.2_gs5_bs1.0_sp0.3-0.3"
echo "GPUs: ${GPU_LIST[*]} (${NUM_GPUS})"
echo "Total prompts: $TOTAL, chunk size: $CHUNK"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

mkdir -p "${OUTPUT_DIR}/logs"

PIDS=()
for i in "${!GPU_LIST[@]}"; do
    GPU=${GPU_LIST[$i]}
    START=$((i * CHUNK))
    END=$(( (i + 1) * CHUNK ))
    if [ $END -gt $TOTAL ]; then END=$TOTAL; fi
    if [ $START -ge $TOTAL ]; then break; fi

    echo "[GPU $GPU] Prompts [$START:$END]"
    CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT" \
        --ckpt_path CompVis/stable-diffusion-v1-4 \
        --prompt_file "$PROMPT_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --classifier_ckpt SoftDelete+CG/work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth \
        --gradcam_stats_dir SoftDelete+CG/gradcam_stats/nudity_4class \
        --monitoring_threshold 0.2 \
        --guidance_scale 5.0 \
        --base_guidance_scale 1.0 \
        --spatial_threshold_start 0.3 \
        --spatial_threshold_end 0.3 \
        --spatial_threshold_strategy cosine_anneal \
        --num_inference_steps 50 \
        --cfg_scale 7.5 \
        --seed 42 \
        --nsamples 1 \
        --safree --safree_alpha 0.01 \
        --svf --svf_up_t 10 \
        --start_idx $START \
        --end_idx $END \
        > "${OUTPUT_DIR}/logs/part${i}_gpu${GPU}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Total jobs: ${#PIDS[@]}"
echo "--- Waiting ---"

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || FAIL=$((FAIL + 1))
done

echo ""
echo "=== Done ==="
[ $FAIL -gt 0 ] && echo "WARNING: ${FAIL} jobs failed"

N_IMAGES=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.png" | wc -l)
echo "Generated images: $N_IMAGES / $TOTAL"
