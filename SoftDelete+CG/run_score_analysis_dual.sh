#!/bin/bash
# ============================================================================
# Run dual score analysis (z0 + GradCAM CDF) on Ringabell + COCO
#
# Ringabell: 79 prompts (harmful)
# COCO: 50 prompts (benign)
# Total: 129 prompts → split across 4 servers
#
# Site A: RB 0-19  (20 prompts)
# Site B: RB 20-39 (20 prompts)
# Site C: RB 40-78 (39 prompts) — ringabell remainder
# Site D: COCO 0-49 (50 prompts)
#
# Usage:
#   bash run_score_analysis_dual.sh --site A --run     # server 1
#   bash run_score_analysis_dual.sh --site B --run     # server 2
#   bash run_score_analysis_dual.sh --site C --run     # server 3
#   bash run_score_analysis_dual.sh --site D --run     # server 4
#   bash run_score_analysis_dual.sh --dry-run           # preview all
# ============================================================================

cd /mnt/home/yhgil99/unlearning/SoftDelete+CG

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate sdd

# ============================================
# Parse Arguments
# ============================================
DRY_RUN=true
USE_NOHUP=false
SITE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --run) DRY_RUN=false; shift;;
        --dry-run) DRY_RUN=true; shift;;
        --nohup) USE_NOHUP=true; shift;;
        --site) SITE="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

# Nohup handling
if [ "$USE_NOHUP" = true ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="./scg_outputs/score_analysis_dual"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/run_site${SITE}_${TIMESTAMP}.log"
    echo "Running in background..."
    echo "Log: $LOG_FILE"
    SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    nohup bash "$SCRIPT_PATH" --run --site "$SITE" > "$LOG_FILE" 2>&1 &
    echo "PID: $!"
    echo "Monitor: tail -f $LOG_FILE"
    exit 0
fi

# ============================================
# CONFIGURATION
# ============================================
CLASSIFIER_CKPT="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth"
GRADCAM_STATS_DIR="/mnt/home/yhgil99/unlearning/SoftDelete+CG/gradcam_stats/nudity_4class"
SD_CKPT="CompVis/stable-diffusion-v1-4"

RB_PROMPT="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv"
COCO_PROMPT="/mnt/home/yhgil99/unlearning/SAFREE/datasets/coco_30k.csv"

OUTPUT_BASE="scg_outputs/score_analysis_dual"

# ============================================
# Define jobs per site
# ============================================
declare -a JOBS=()

case "$SITE" in
    A)
        JOBS+=("ringabell|${RB_PROMPT}|0|20|${OUTPUT_BASE}/ringabell_scores_A.json")
        ;;
    B)
        JOBS+=("ringabell|${RB_PROMPT}|20|40|${OUTPUT_BASE}/ringabell_scores_B.json")
        ;;
    C)
        JOBS+=("ringabell|${RB_PROMPT}|40|79|${OUTPUT_BASE}/ringabell_scores_C.json")
        ;;
    D)
        JOBS+=("coco|${COCO_PROMPT}|0|50|${OUTPUT_BASE}/coco_scores.json")
        ;;
    "")
        # Run all
        JOBS+=("ringabell|${RB_PROMPT}|0|79|${OUTPUT_BASE}/ringabell_scores.json")
        JOBS+=("coco|${COCO_PROMPT}|0|50|${OUTPUT_BASE}/coco_scores.json")
        ;;
    *)
        echo "Unknown site: $SITE (use A/B/C/D)"
        exit 1
        ;;
esac

echo "=============================================="
echo "DUAL SCORE ANALYSIS (z0 + GradCAM CDF)"
echo "=============================================="
echo "Site: ${SITE:-ALL}"
echo "Jobs: ${#JOBS[@]}"
echo ""

for job in "${JOBS[@]}"; do
    IFS='|' read -r NAME PROMPT START END OUTPUT <<< "$job"
    echo "  ${NAME}: prompts ${START}-$((END-1)) → ${OUTPUT}"
done
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Add --run to execute."
    echo ""
    echo "Commands that would run:"
    for job in "${JOBS[@]}"; do
        IFS='|' read -r NAME PROMPT START END OUTPUT <<< "$job"
        echo "  CUDA_VISIBLE_DEVICES=0 python score_analysis_dual.py \\"
        echo "      --prompt_file ${PROMPT} \\"
        echo "      --output_path ${OUTPUT} \\"
        echo "      --classifier_ckpt ${CLASSIFIER_CKPT} \\"
        echo "      --gradcam_stats_dir ${GRADCAM_STATS_DIR} \\"
        echo "      --start_idx ${START} --end_idx ${END}"
        echo ""
    done
    exit 0
fi

# ============================================
# RUN
# ============================================
for job in "${JOBS[@]}"; do
    IFS='|' read -r NAME PROMPT START END OUTPUT <<< "$job"

    # Skip if already done
    if [ -f "${OUTPUT}" ]; then
        echo "[SKIP] ${OUTPUT} already exists"
        continue
    fi

    echo "[START] ${NAME}: prompts ${START}-$((END-1))"

    CUDA_VISIBLE_DEVICES=0 python score_analysis_dual.py \
        --ckpt_path "${SD_CKPT}" \
        --prompt_file "${PROMPT}" \
        --output_path "${OUTPUT}" \
        --classifier_ckpt "${CLASSIFIER_CKPT}" \
        --gradcam_stats_dir "${GRADCAM_STATS_DIR}" \
        --start_idx ${START} \
        --end_idx ${END}

    echo "[DONE] ${NAME} → ${OUTPUT}"
done

echo ""
echo "=============================================="
echo "SCORE ANALYSIS COMPLETE"
echo "=============================================="
echo "Output: ${OUTPUT_BASE}/"
echo ""
echo "Next: merge results (if split) and analyze:"
echo "  python analyze_dual_scores.py \\"
echo "      --ringabell ${OUTPUT_BASE}/ringabell_scores.json \\"
echo "      --coco ${OUTPUT_BASE}/coco_scores.json"
