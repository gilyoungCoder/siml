#!/usr/bin/env bash
# =============================================================================
# Qwen3-VL Evaluation for v17/v18/v19 (8 GPUs parallel)
# =============================================================================
# Evaluates all v17/v18/v19 output folders that don't have qwen3_vl results yet.
#
# Usage:
#   bash scripts/nohup_qwen3_eval_v17_v18_v19.sh          # launch
#   bash scripts/nohup_qwen3_eval_v17_v18_v19.sh --status  # check progress
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VLM_DIR="${REPO_ROOT}/vlm"
OUTPUT_BASE="${REPO_ROOT}/CAS_SpatialCFG/outputs"
LOG_DIR="${REPO_ROOT}/scripts/logs/qwen3_eval"
VLM_PYTHON="conda run -n vlm --no-capture-output python"
NUM_GPUS=8

if [[ "${1:-}" == "--status" ]]; then
    echo "=== Qwen3-VL Eval Status ==="
    for g in $(seq 0 $((NUM_GPUS - 1))); do
        log="${LOG_DIR}/gpu_${g}.log"
        if [[ -f "$log" ]]; then
            done=$(grep -c "DONE:" "$log" 2>/dev/null || echo 0)
            total=$(grep -c "EVAL:" "$log" 2>/dev/null || echo 0)
            last=$(tail -1 "$log" 2>/dev/null)
            echo "  GPU $g: ${done} done | ${last}"
        fi
    done
    echo ""
    # Count total qwen3_vl files
    for ver in v17 v18 v19; do
        has=$(find "${OUTPUT_BASE}/${ver}" -name "categories_qwen3_vl_nudity.json" 2>/dev/null | wc -l)
        need=$(find "${OUTPUT_BASE}/${ver}" -maxdepth 2 -name "*.png" -printf '%h\n' 2>/dev/null | sort -u | wc -l)
        echo "  ${ver}: ${has}/${need} evaluated"
    done
    exit 0
fi

# Collect all folders needing evaluation
echo "Collecting folders to evaluate..."
FOLDERS=()
for ver in v17 v18 v19; do
    for d in "${OUTPUT_BASE}/${ver}"/*/; do
        [[ -d "$d" ]] || continue
        # Skip if already has qwen3_vl result
        if [[ -f "${d}categories_qwen3_vl_nudity.json" ]]; then
            continue
        fi
        # Check it has images
        img_count=$(find "$d" -maxdepth 1 -name "*.png" | head -1)
        if [[ -n "$img_count" ]]; then
            FOLDERS+=("$d")
        fi
    done
done

TOTAL=${#FOLDERS[@]}
echo "Total folders to evaluate: ${TOTAL}"

if [[ "$TOTAL" -eq 0 ]]; then
    echo "Nothing to evaluate!"
    exit 0
fi

# Distribute round-robin across GPUs
mkdir -p "$LOG_DIR"

for g in $(seq 0 $((NUM_GPUS - 1))); do
    script="${LOG_DIR}/gpu_${g}.sh"
    cat > "$script" << 'HEADER'
#!/usr/bin/env bash
set -euo pipefail
HEADER

    idx=0
    job_count=0
    for folder in "${FOLDERS[@]}"; do
        if (( idx % NUM_GPUS == g )); then
            cat >> "$script" << EVALCMD
echo "[\$(date '+%H:%M:%S')] EVAL: ${folder}"
cd ${VLM_DIR} && CUDA_VISIBLE_DEVICES=${g} ${VLM_PYTHON} opensource_vlm_i2p_all.py "${folder}" nudity qwen 2>&1 | tail -3
echo "[\$(date '+%H:%M:%S')] DONE: ${folder}"

EVALCMD
            job_count=$((job_count + 1))
        fi
        idx=$((idx + 1))
    done

    echo "echo 'GPU ${g} ALL COMPLETE (${job_count} folders)'" >> "$script"
    chmod +x "$script"
    echo "  GPU ${g}: ${job_count} folders"
done

# Launch all with nohup
echo ""
echo "Launching ${TOTAL} evaluations across ${NUM_GPUS} GPUs..."
for g in $(seq 0 $((NUM_GPUS - 1))); do
    script="${LOG_DIR}/gpu_${g}.sh"
    log="${LOG_DIR}/gpu_${g}.log"
    nohup bash "$script" > "$log" 2>&1 &
    echo "  GPU ${g}: PID $! -> ${log}"
done

echo ""
echo "All launched! Monitor with:"
echo "  bash $0 --status"
echo "  tail -f ${LOG_DIR}/gpu_*.log"
