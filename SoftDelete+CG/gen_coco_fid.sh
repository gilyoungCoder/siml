#!/bin/bash
# ============================================================================
# COCO 10k FID Generation — 3 methods, each on a separate server (8 GPUs)
# All scripts now support --start_idx / --end_idx for multi-GPU splitting.
#
# Usage:
#   Server 1 (Ours):        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash SoftDelete+CG/gen_coco_fid.sh --method ours --nohup
#   Server 2 (SAFREE+Ours): CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash SoftDelete+CG/gen_coco_fid.sh --method safree_ours --nohup
#   Server 3 (SAFREE):      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash SoftDelete+CG/gen_coco_fid.sh --method safree --nohup
# ============================================================================

cd /mnt/home/yhgil99/unlearning

COCO_CSV="SAFREE/datasets/coco_30k_10k.csv"
OUTPUT_BASE="SoftDelete+CG/scg_outputs/final_coco"
NUM_STEPS=50
SEED=42
CFG_SCALE=7.5

# Ours config
OURS_SCRIPT="SoftDelete+CG/generate_nudity_4class_sample_level_monitoring.py"
CLASSIFIER_CKPT="SoftDelete+CG/work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_DIR="SoftDelete+CG/gradcam_stats/nudity_4class"

# SAFREE+Ours config
SAFREE_MON_SCRIPT="SAFREE/generate_safree_monitoring.py"

# SAFREE baseline config
SAFREE_SCRIPT="SAFREE/gen_safree_i2p_concepts.py"

# Parse args
METHOD=""
USE_NOHUP=false
NUM_GPUS=8

while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2;;
        --nohup) USE_NOHUP=true; shift;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [ -z "$METHOD" ]; then
    echo "Usage: bash gen_coco_fid.sh --method {ours|safree_ours|safree} [--nohup] [--num-gpus N]"
    exit 1
fi

# Nohup wrapper
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${OUTPUT_BASE}/logs/coco_${METHOD}_${TIMESTAMP}.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "Running in background... Log: $LOG_FILE"
    nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$SCRIPT_PATH" --method "$METHOD" --num-gpus "$NUM_GPUS" > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    exit 0
fi

# GPU setup
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_LIST+=($i); done
fi
NUM_GPUS=${#GPU_LIST[@]}

# Count prompts (excluding header)
TOTAL_PROMPTS=$(tail -n +2 "$COCO_CSV" | wc -l)
CHUNK_SIZE=$(( (TOTAL_PROMPTS + NUM_GPUS - 1) / NUM_GPUS ))

echo "=============================================="
echo "COCO FID Generation: ${METHOD}"
echo "Total prompts: ${TOTAL_PROMPTS}"
echo "GPUs: ${GPU_LIST[*]} (${NUM_GPUS})"
echo "Chunk size: ~${CHUNK_SIZE} per GPU"
echo "=============================================="

mkdir -p "${OUTPUT_BASE}/logs"
PIDS=()

case "$METHOD" in
    ours)
        OUTDIR="${OUTPUT_BASE}/ours"
        mkdir -p "$OUTDIR"
        for ((i=0; i<NUM_GPUS; i++)); do
            START=$((i * CHUNK_SIZE))
            END=$(( (i + 1) * CHUNK_SIZE ))
            GPU=${GPU_LIST[$i]}
            echo "[GPU $GPU] Ours: prompts ${START}-${END}"
            CUDA_VISIBLE_DEVICES=$GPU python "$OURS_SCRIPT" \
                --ckpt_path "CompVis/stable-diffusion-v1-4" \
                --prompt_file "$COCO_CSV" \
                --output_dir "$OUTDIR" \
                --classifier_ckpt "$CLASSIFIER_CKPT" \
                --gradcam_stats_dir "$GRADCAM_STATS_DIR" \
                --monitoring_threshold 0.05 \
                --guidance_scale 12.5 \
                --base_guidance_scale 2.0 \
                --spatial_threshold_start 0.2 \
                --spatial_threshold_end 0.3 \
                --spatial_threshold_strategy cosine_anneal \
                --num_inference_steps $NUM_STEPS \
                --cfg_scale $CFG_SCALE \
                --seed $SEED \
                --nsamples 1 \
                --start_idx $START \
                --end_idx $END \
                > "${OUTPUT_BASE}/logs/ours_gpu${GPU}.log" 2>&1 &
            PIDS+=($!)
        done
        ;;

    safree_ours)
        OUTDIR="${OUTPUT_BASE}/safree_ours"
        mkdir -p "$OUTDIR"
        for ((i=0; i<NUM_GPUS; i++)); do
            START=$((i * CHUNK_SIZE))
            END=$(( (i + 1) * CHUNK_SIZE ))
            GPU=${GPU_LIST[$i]}
            echo "[GPU $GPU] SAFREE+Ours: prompts ${START}-${END}"
            CUDA_VISIBLE_DEVICES=$GPU python "$SAFREE_MON_SCRIPT" \
                --ckpt_path "CompVis/stable-diffusion-v1-4" \
                --prompt_file "$COCO_CSV" \
                --output_dir "$OUTDIR" \
                --classifier_ckpt "$CLASSIFIER_CKPT" \
                --gradcam_stats_dir "$GRADCAM_STATS_DIR" \
                --monitoring_threshold 0.2 \
                --guidance_scale 5 \
                --base_guidance_scale 2.0 \
                --spatial_threshold_start 0.7 \
                --spatial_threshold_end 0.3 \
                --spatial_threshold_strategy cosine_anneal \
                --safree --svf --svf_up_t 10 \
                --num_inference_steps $NUM_STEPS \
                --cfg_scale $CFG_SCALE \
                --seed $SEED \
                --nsamples 1 \
                --start_idx $START \
                --end_idx $END \
                > "${OUTPUT_BASE}/logs/safree_ours_gpu${GPU}.log" 2>&1 &
            PIDS+=($!)
        done
        ;;

    safree)
        OUTDIR="${OUTPUT_BASE}/safree"
        mkdir -p "$OUTDIR"
        for ((i=0; i<NUM_GPUS; i++)); do
            START=$((i * CHUNK_SIZE))
            END=$(( (i + 1) * CHUNK_SIZE ))
            GPU=${GPU_LIST[$i]}
            echo "[GPU $GPU] SAFREE: prompts ${START}-${END}"
            CUDA_VISIBLE_DEVICES=$GPU python "$SAFREE_SCRIPT" \
                --prompt_file "$COCO_CSV" \
                --concepts "sexual" \
                --model_id "CompVis/stable-diffusion-v1-4" \
                --outdir "$OUTDIR" \
                --num_images 1 \
                --steps $NUM_STEPS \
                --guidance $CFG_SCALE \
                --seed $SEED \
                --safree --svf \
                --up_t 10 \
                --no_concept_subdir \
                --start_idx $START \
                --end_idx $END \
                > "${OUTPUT_BASE}/logs/safree_gpu${GPU}.log" 2>&1 &
            PIDS+=($!)
        done
        ;;

    *)
        echo "Unknown method: $METHOD (use: ours, safree_ours, safree)"
        exit 1
        ;;
esac

echo ""
echo "--- Waiting for all ${#PIDS[@]} jobs ---"
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || FAIL=$((FAIL + 1))
done

echo ""
echo "=== COCO FID generation complete: ${METHOD} ==="
echo "Output: ${OUTDIR}"
[ $FAIL -gt 0 ] && echo "WARNING: ${FAIL} jobs failed"
echo "Total images: $(ls "$OUTDIR"/*.png 2>/dev/null | wc -l)"
