#!/usr/bin/env bash
# worker.sh <host> <gpu>
# Runs on the target host. Reads its assigned manifest CSV and processes jobs sequentially.
# Each job: idempotency check → gen → eval_v2 → eval_v3
#
# Usage (called via launch_all.sh):
#   bash scripts/launch_0420/worker.sh siml-01 0

set -euo pipefail

HOST=${1:?Usage: worker.sh <host> <gpu>}
GPU=${2:?Usage: worker.sh <host> <gpu>}

REPO=/mnt/home3/yhgil99/unlearning
PYTHON_GEN=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PYTHON_VLM=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
MANIFEST=${REPO}/scripts/launch_0420/manifests/worker_${HOST}_g${GPU}.csv
LOG_DIR=${REPO}/logs/launch_0420
LOG_FILE=${LOG_DIR}/worker_${HOST}_g${GPU}.log

mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================"
echo "Worker ${HOST} GPU ${GPU} started at $(date)"
echo "Manifest: ${MANIFEST}"
echo "PID: $$"
echo "================================================================"

if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: Manifest not found: ${MANIFEST}"
    exit 1
fi

# ─── Pre-launch GPU check ─────────────────────────────────────────────────
MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" 2>/dev/null | tr -d ' ')
echo "[GPU CHECK] GPU ${GPU} memory used: ${MEM_USED} MiB"
if [[ -n "${MEM_USED}" ]] && [[ "${MEM_USED}" -gt 1000 ]]; then
    # Check if the process belongs to us (yhgil99) — if so, allow (our own baseline may be running)
    FOREIGN=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "${GPU}" 2>/dev/null | while read -r line; do
        pid=$(echo "$line" | awk -F',' '{print $1}' | tr -d ' ')
        owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
        if [[ "$owner" != "yhgil99" ]]; then
            echo "$pid ($owner)"
        fi
    done)
    if [[ -n "${FOREIGN}" ]]; then
        echo "WARN: GPU ${GPU} has foreign processes: ${FOREIGN} — skipping this worker"
        exit 0
    else
        echo "[GPU CHECK] GPU ${GPU} in use by yhgil99 — proceeding (idempotency will skip finished jobs)"
    fi
fi

# ─── Helper: count images in a directory ─────────────────────────────────
count_images() {
    local dir="$1"
    find "${dir}" -maxdepth 1 -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l
}

# ─── Helper: run VLM eval (skip if result already exists) ────────────────
run_eval() {
    local out_dir="$1"
    local eval_concept="$2"
    local script="$3"   # v2 or v3
    local result_file="${out_dir}/results_qwen3_vl_${eval_concept}.txt"

    # For v3, the file is the same naming convention
    if [[ "${script}" == "v3" ]]; then
        result_file="${out_dir}/results_qwen3_vl_${eval_concept}.txt"
        # v3 overwrites with same name pattern — check if v3 categories json exists
        local cat_file="${out_dir}/categories_qwen3_vl_${eval_concept}.json"
        # v3 uses same output files as v2; check if it's already done via a sentinel
        result_file="${out_dir}/.eval_v3_${eval_concept}.done"
    fi

    if [[ "${script}" == "v2" ]]; then
        local cat_file="${out_dir}/categories_qwen3_vl_${eval_concept}.json"
        if [[ -f "${cat_file}" ]] && [[ -s "${cat_file}" ]]; then
            echo "  [SKIP eval_v2] ${eval_concept} already done"
            return 0
        fi
        echo "  [EVAL v2] ${eval_concept} in ${out_dir}"
        cd "${REPO}/vlm" && CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON_VLM} \
            opensource_vlm_i2p_all_v2.py "${out_dir}" "${eval_concept}" qwen \
            && echo "  [EVAL v2 OK]" || echo "  [EVAL v2 FAILED]"
    else
        # v3 — use sentinel file since it overwrites v2 files
        local sentinel="${out_dir}/.eval_v3_${eval_concept}.done"
        if [[ -f "${sentinel}" ]]; then
            echo "  [SKIP eval_v3] ${eval_concept} already done"
            return 0
        fi
        echo "  [EVAL v3] ${eval_concept} in ${out_dir}"
        cd "${REPO}/vlm" && CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON_VLM} \
            opensource_vlm_i2p_all_v3.py "${out_dir}" "${eval_concept}" qwen \
            && touch "${sentinel}" && echo "  [EVAL v3 OK]" || echo "  [EVAL v3 FAILED]"
    fi
}

# ─── Main job loop ────────────────────────────────────────────────────────
TOTAL=0
DONE=0
SKIPPED=0
FAILED=0

# Read CSV (skip header)
while IFS=',' read -r phase backbone dataset config_id gen_cmd eval_concept output_dir; do
    # Skip CSV header
    [[ "${phase}" == "phase" ]] && continue

    TOTAL=$((TOTAL + 1))
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "[JOB ${TOTAL}] phase=${phase} backbone=${backbone} dataset=${dataset}"
    echo "  config_id=${config_id}"
    echo "  output_dir=${output_dir}"
    echo "  eval_concept=${eval_concept}"
    echo "  time=$(date)"

    # ── Idempotency check: skip generation if dir exists with enough images ──
    # Determine expected image count
    case "${dataset}" in
        rab)           EXPECTED=79 ;;
        mja_*)         EXPECTED=100 ;;
        *)             EXPECTED=1 ;;
    esac

    if [[ -d "${output_dir}" ]]; then
        IMG_COUNT=$(count_images "${output_dir}")
        if [[ "${IMG_COUNT}" -ge "${EXPECTED}" ]]; then
            echo "  [SKIP gen] ${output_dir} has ${IMG_COUNT}/${EXPECTED} images"
            SKIPPED=$((SKIPPED + 1))
            # Still run evals in case they were missed
            run_eval "${output_dir}" "${eval_concept}" "v2"
            run_eval "${output_dir}" "${eval_concept}" "v3"
            DONE=$((DONE + 1))
            continue
        else
            # Check if another process is currently generating (lock file)
            if [[ -f "${output_dir}/.lock" ]]; then
                LOCK_PID=$(cat "${output_dir}/.lock" 2>/dev/null)
                if kill -0 "${LOCK_PID}" 2>/dev/null; then
                    echo "  [WAIT] Another process (PID ${LOCK_PID}) is generating — skipping to next job"
                    SKIPPED=$((SKIPPED + 1))
                    continue
                else
                    echo "  [STALE LOCK] Removing stale lock file"
                    rm -f "${output_dir}/.lock"
                fi
            fi
            echo "  [RESUME] ${output_dir} has ${IMG_COUNT}/${EXPECTED} images — re-running"
        fi
    fi

    # ── Write lock file ──
    mkdir -p "${output_dir}"
    echo $$ > "${output_dir}/.lock"

    # ── Run generation ──
    echo "  [GEN] Running: ${gen_cmd}"
    GEN_START=$(date +%s)
    if eval "${gen_cmd}"; then
        GEN_END=$(date +%s)
        GEN_ELAPSED=$((GEN_END - GEN_START))
        echo "  [GEN OK] elapsed=${GEN_ELAPSED}s"
        rm -f "${output_dir}/.lock"

        # ── Run evals ──
        run_eval "${output_dir}" "${eval_concept}" "v2"
        run_eval "${output_dir}" "${eval_concept}" "v3"
        DONE=$((DONE + 1))
    else
        GEN_END=$(date +%s)
        GEN_ELAPSED=$((GEN_END - GEN_START))
        echo "  [GEN FAILED] elapsed=${GEN_ELAPSED}s exit_code=$?"
        rm -f "${output_dir}/.lock"
        FAILED=$((FAILED + 1))
    fi

done < "${MANIFEST}"

echo ""
echo "================================================================"
echo "Worker ${HOST} GPU ${GPU} FINISHED at $(date)"
echo "Total=${TOTAL} Done=${DONE} Skipped=${SKIPPED} Failed=${FAILED}"
echo "================================================================"
