#!/bin/bash
# ============================================================================
# Monitoring threshold=0.1 grid search
# GS: 12.5, 15 × BS: 2, 3 = 4 combos
# Datasets: ringabell, unlearndiff, mma + COCO
#
# Usage (2 servers):
#   Server 1 (8 GPUs, safety 12 jobs):
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash SoftDelete+CG/run_mon01_grid.sh --part safety --nohup
#
#   Server 2 (8 GPUs, COCO 4 configs × 2 GPUs each = 8 jobs):
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash SoftDelete+CG/run_mon01_grid.sh --part coco --nohup
# ============================================================================

cd /mnt/home/yhgil99/unlearning

MON_SCRIPT="SoftDelete+CG/generate_nudity_4class_sample_level_monitoring.py"
CLASSIFIER_CKPT="SoftDelete+CG/work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_DIR="SoftDelete+CG/gradcam_stats/nudity_4class"
SD_CKPT="CompVis/stable-diffusion-v1-4"

COCO_CSV="SAFREE/datasets/coco_30k_10k.csv"

OUTPUT_BASE="SoftDelete+CG/scg_outputs/mon01_grid"
NUM_STEPS=50
SEED=42
CFG_SCALE=7.5

MON_THRESH=0.1
SP_START=0.2
SP_END=0.3

declare -A DATASET_PROMPTS
DATASET_PROMPTS[ringabell]="SAFREE/datasets/nudity-ring-a-bell.csv"
DATASET_PROMPTS[unlearndiff]="SAFREE/datasets/unlearn_diff_nudity.csv"
DATASET_PROMPTS[mma]="SAFREE/datasets/mma-diffusion-nsfw-adv-prompts.csv"

GS_LIST=(12.5 15)
BS_LIST=(0 1)

# Parse args
USE_NOHUP=false
PART="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --nohup) USE_NOHUP=true; shift;;
        --part) PART="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

SCRIPT_PATH="/mnt/home/yhgil99/unlearning/SoftDelete+CG/run_mon01_grid.sh"
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${OUTPUT_BASE}/logs/mon01_grid_${PART}_${TIMESTAMP}.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "Running in background... Log: $LOG_FILE"
    nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$SCRIPT_PATH" --part "$PART" > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

# GPU setup
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=()
    for ((i=0; i<8; i++)); do GPU_LIST+=($i); done
fi
NUM_GPUS=${#GPU_LIST[@]}

mkdir -p "${OUTPUT_BASE}/logs"

echo "=============================================="
echo "Mon=0.1 Grid Search [part=$PART]"
echo "GS: ${GS_LIST[*]}, BS: ${BS_LIST[*]}"
echo "GPUs: ${GPU_LIST[*]} (${NUM_GPUS})"
echo "=============================================="

GPU_IDX=0
PIDS=()

# ---------- Safety datasets (12 jobs → 8 GPUs, round-robin) ----------
if [ "$PART" = "safety" ] || [ "$PART" = "all" ]; then
    echo ""
    echo "--- Safety datasets (ringabell, unlearndiff, mma) ---"
    for gs in "${GS_LIST[@]}"; do
        for bs in "${BS_LIST[@]}"; do
            CONFIG="mon0.1_gs${gs}_bs${bs}_sp0.2-0.3"
            for ds in ringabell unlearndiff mma; do
                PROMPT_FILE="${DATASET_PROMPTS[$ds]}"
                OUTDIR="${OUTPUT_BASE}/${ds}/${CONFIG}"
                LOG="${OUTPUT_BASE}/logs/${ds}_${CONFIG}.log"

                if [ -f "${OUTDIR}/generation_stats.json" ]; then
                    echo "[SKIP] ${ds}/${CONFIG}"
                    continue
                fi

                GPU=${GPU_LIST[$((GPU_IDX % NUM_GPUS))]}
                GPU_IDX=$((GPU_IDX + 1))
                mkdir -p "$OUTDIR"

                echo "[GPU $GPU] ${ds}/${CONFIG}"
                CUDA_VISIBLE_DEVICES=$GPU python "$MON_SCRIPT" \
                    --ckpt_path "$SD_CKPT" \
                    --prompt_file "$PROMPT_FILE" \
                    --output_dir "$OUTDIR" \
                    --classifier_ckpt "$CLASSIFIER_CKPT" \
                    --gradcam_stats_dir "$GRADCAM_STATS_DIR" \
                    --monitoring_threshold $MON_THRESH \
                    --guidance_scale $gs \
                    --base_guidance_scale $bs \
                    --spatial_threshold_start $SP_START \
                    --spatial_threshold_end $SP_END \
                    --spatial_threshold_strategy cosine_anneal \
                    --num_inference_steps $NUM_STEPS \
                    --cfg_scale $CFG_SCALE \
                    --seed $SEED \
                    --nsamples 1 \
                    > "$LOG" 2>&1 &
                PIDS+=($!)
            done
        done
    done
fi

# ---------- COCO (4 configs × 2 GPUs each = 8 jobs) ----------
if [ "$PART" = "coco" ] || [ "$PART" = "all" ]; then
    echo ""
    echo "--- COCO dataset (4 configs × 2 GPUs each) ---"

    TOTAL_COCO=$(python3 -c "
import csv
with open('$COCO_CSV') as f:
    print(sum(1 for _ in csv.DictReader(f)))
")
    HALF=$(( (TOTAL_COCO + 1) / 2 ))
    echo "COCO prompts: $TOTAL_COCO, split at: $HALF"

    for gs in "${GS_LIST[@]}"; do
        for bs in "${BS_LIST[@]}"; do
            CONFIG="mon0.1_gs${gs}_bs${bs}_sp0.2-0.3"
            OUTDIR="${OUTPUT_BASE}/coco/${CONFIG}"

            if [ -f "${OUTDIR}/generation_stats.json" ]; then
                echo "[SKIP] coco/${CONFIG}"
                continue
            fi

            mkdir -p "$OUTDIR"

            # GPU A: first half
            GPU_A=${GPU_LIST[$((GPU_IDX % NUM_GPUS))]}
            GPU_IDX=$((GPU_IDX + 1))
            echo "[GPU $GPU_A] coco/${CONFIG} [0:${HALF}]"
            CUDA_VISIBLE_DEVICES=$GPU_A python "$MON_SCRIPT" \
                --ckpt_path "$SD_CKPT" \
                --prompt_file "$COCO_CSV" \
                --output_dir "$OUTDIR" \
                --classifier_ckpt "$CLASSIFIER_CKPT" \
                --gradcam_stats_dir "$GRADCAM_STATS_DIR" \
                --monitoring_threshold $MON_THRESH \
                --guidance_scale $gs \
                --base_guidance_scale $bs \
                --spatial_threshold_start $SP_START \
                --spatial_threshold_end $SP_END \
                --spatial_threshold_strategy cosine_anneal \
                --num_inference_steps $NUM_STEPS \
                --cfg_scale $CFG_SCALE \
                --seed $SEED \
                --nsamples 1 \
                --start_idx 0 \
                --end_idx $HALF \
                > "${OUTPUT_BASE}/logs/coco_${CONFIG}_partA.log" 2>&1 &
            PIDS+=($!)

            # GPU B: second half
            GPU_B=${GPU_LIST[$((GPU_IDX % NUM_GPUS))]}
            GPU_IDX=$((GPU_IDX + 1))
            echo "[GPU $GPU_B] coco/${CONFIG} [${HALF}:${TOTAL_COCO}]"
            CUDA_VISIBLE_DEVICES=$GPU_B python "$MON_SCRIPT" \
                --ckpt_path "$SD_CKPT" \
                --prompt_file "$COCO_CSV" \
                --output_dir "$OUTDIR" \
                --classifier_ckpt "$CLASSIFIER_CKPT" \
                --gradcam_stats_dir "$GRADCAM_STATS_DIR" \
                --monitoring_threshold $MON_THRESH \
                --guidance_scale $gs \
                --base_guidance_scale $bs \
                --spatial_threshold_start $SP_START \
                --spatial_threshold_end $SP_END \
                --spatial_threshold_strategy cosine_anneal \
                --num_inference_steps $NUM_STEPS \
                --cfg_scale $CFG_SCALE \
                --seed $SEED \
                --nsamples 1 \
                --start_idx $HALF \
                --end_idx $TOTAL_COCO \
                > "${OUTPUT_BASE}/logs/coco_${CONFIG}_partB.log" 2>&1 &
            PIDS+=($!)
        done
    done
fi

echo ""
echo "Total jobs launched: ${#PIDS[@]}"
echo "--- Waiting ---"

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || FAIL=$((FAIL + 1))
done

echo ""
echo "=== Done ==="
[ $FAIL -gt 0 ] && echo "WARNING: ${FAIL} jobs failed"

echo ""
echo "Results:"
for gs in "${GS_LIST[@]}"; do
    for bs in "${BS_LIST[@]}"; do
        CONFIG="mon0.1_gs${gs}_bs${bs}_sp0.2-0.3"
        for ds in ringabell unlearndiff mma coco; do
            DIR="${OUTPUT_BASE}/${ds}/${CONFIG}"
            N=$(ls "$DIR"/*.png 2>/dev/null | wc -l)
            echo "  ${ds}/${CONFIG}: ${N} images"
        done
    done
done
