#!/usr/bin/env bash
# launch_all.sh — Orchestrator. Run on siml-09 (or any host with SSH access to all).
# Performs pre-checks, then launches worker.sh via nohup on each (host, gpu).
#
# Usage: bash scripts/launch_0420/launch_all.sh

set -uo pipefail

REPO=/mnt/home3/yhgil99/unlearning
LOG_ROOT=${REPO}/logs/launch_0420
SCRIPT_DIR=${REPO}/scripts/launch_0420
MANIFEST_DIR=${SCRIPT_DIR}/manifests

mkdir -p "${LOG_ROOT}"

LAUNCH_LOG="${LOG_ROOT}/launch_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

echo "================================================================"
echo "launch_all.sh started at $(date)"
echo "================================================================"

# ─── All (host, gpu) pairs ───────────────────────────────────────────────
declare -a WORKERS=(
    "siml-01 0" "siml-01 1" "siml-01 2" "siml-01 3"
    "siml-01 4" "siml-01 5" "siml-01 6" "siml-01 7"
    "siml-06 4" "siml-06 5" "siml-06 6" "siml-06 7"
    "siml-08 4" "siml-08 5"
    "siml-09 0"
)

ABORT=0

# ─── Step 1: Pre-check GPU memory ────────────────────────────────────────
echo ""
echo "=== Step 1: GPU pre-checks ==="
for worker in "${WORKERS[@]}"; do
    HOST=$(echo $worker | awk '{print $1}')
    GPU=$(echo $worker | awk '{print $2}')

    MEM=$(ssh -o ConnectTimeout=10 "${HOST}" \
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${GPU} 2>/dev/null | tr -d ' '" 2>/dev/null)

    if [[ -z "${MEM}" ]]; then
        echo "  WARN: ${HOST} GPU ${GPU} — could not query (SSH failed or GPU not found)"
        continue
    fi

    if [[ "${MEM}" -gt 1000 ]]; then
        # Check if it's yhgil99's own processes
        FOREIGN=$(ssh -o ConnectTimeout=10 "${HOST}" "
            nvidia-smi --query-compute-apps=pid --format=csv,noheader -i ${GPU} 2>/dev/null | while read pid; do
                pid=\$(echo \$pid | tr -d ' ')
                owner=\$(ps -o user= -p \$pid 2>/dev/null | tr -d ' ')
                if [[ \"\$owner\" != \"yhgil99\" ]]; then echo \"\$pid (\$owner)\"; fi
            done
        " 2>/dev/null)

        if [[ -n "${FOREIGN}" ]]; then
            echo "  ABORT: ${HOST} GPU ${GPU} — FOREIGN process using ${MEM}MB: ${FOREIGN}"
            ABORT=1
        else
            echo "  OK(busy): ${HOST} GPU ${GPU} — ${MEM}MB (yhgil99's own process — idempotency will handle)"
        fi
    else
        echo "  FREE: ${HOST} GPU ${GPU} — ${MEM}MB"
    fi
done

if [[ "${ABORT}" -eq 1 ]]; then
    echo ""
    echo "ABORT: One or more GPUs have foreign processes. Fix before launching."
    exit 1
fi

# ─── Step 2: Verify dataset prompt files ─────────────────────────────────
echo ""
echo "=== Step 2: Dataset prompt file checks ==="
DATASETS=(
    "CAS_SpatialCFG/prompts/mja_sexual.txt:100"
    "CAS_SpatialCFG/prompts/mja_violent.txt:100"
    "CAS_SpatialCFG/prompts/mja_disturbing.txt:100"
    "CAS_SpatialCFG/prompts/mja_illegal.txt:100"
    "CAS_SpatialCFG/prompts/nudity-ring-a-bell.csv:79"
)

ssh -o ConnectTimeout=10 siml-01 "
cd ${REPO}
for spec in $(printf "'%s' " "${DATASETS[@]}"); do
    f=\$(echo \$spec | cut -d: -f1)
    expected=\$(echo \$spec | cut -d: -f2)
    if [[ ! -f \"\$f\" ]]; then
        echo \"MISSING: \$f\"
    else
        # Count prompts (subtract 1 for CSV header)
        if [[ \"\$f\" == *.csv ]]; then
            n=\$((\$(wc -l < \"\$f\") - 1))
        else
            n=\$(wc -l < \"\$f\")
        fi
        echo \"OK: \$f (\$n prompts, expected \$expected)\"
    fi
done
" 2>/dev/null

# ─── Step 3: Verify concept packs ────────────────────────────────────────
echo ""
echo "=== Step 3: Concept pack checks ==="
ssh -o ConnectTimeout=10 siml-01 "
cd ${REPO}
for concept in sexual violent disturbing illegal; do
    f=CAS_SpatialCFG/exemplars/concepts_v2/\${concept}/clip_grouped.pt
    if [[ -f \"\$f\" ]]; then echo \"OK: \$f\"
    else echo \"MISSING: \$f\"; fi
done
" 2>/dev/null

# ─── Step 4: Build manifests ─────────────────────────────────────────────
echo ""
echo "=== Step 4: Building job manifests ==="
ssh -o ConnectTimeout=10 siml-01 "
    cd ${REPO} && \
    /mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10 \
        scripts/launch_0420/build_job_manifests.py
" 2>/dev/null

# Verify manifests were created
MANIFEST_COUNT=$(ssh -o ConnectTimeout=10 siml-01 "ls ${MANIFEST_DIR}/*.csv 2>/dev/null | wc -l" 2>/dev/null)
echo "  Created ${MANIFEST_COUNT} manifest files"
if [[ "${MANIFEST_COUNT}" -lt 15 ]]; then
    echo "ABORT: Expected 15 manifests, got ${MANIFEST_COUNT}"
    exit 1
fi

# ─── Step 5: Smoke tests ─────────────────────────────────────────────────
echo ""
echo "=== Step 5: Smoke tests (1 prompt, 14 steps) ==="

SMOKE_DIR=/tmp/smoke_launch_0420
PYTHON_GEN=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10

# SD1.4 smoke test on siml-01 GPU 5 (free)
echo "  [SMOKE SD1.4] siml-01 GPU 5..."
SMOKE_OK_SD14=0
ssh -o ConnectTimeout=30 siml-01 "
    cd ${REPO} && \
    CUDA_VISIBLE_DEVICES=5 ${PYTHON_GEN} \
        CAS_SpatialCFG/generate_v27.py \
        --prompts CAS_SpatialCFG/prompts/mja_sexual.txt \
        --outdir /tmp/smoke_sd14 \
        --probe_mode both --cas_threshold 0.6 \
        --safety_scale 1.0 --attn_threshold 0.1 \
        --how_mode anchor_inpaint \
        --family_guidance \
        --family_config CAS_SpatialCFG/exemplars/concepts_v2/sexual/clip_grouped.pt \
        --end_idx 1 --steps 14 2>&1 | tail -5
" 2>/dev/null && SMOKE_OK_SD14=1 || true
echo "  SD1.4 smoke: $([ $SMOKE_OK_SD14 -eq 1 ] && echo PASS || echo FAIL)"

# SD3 smoke test on siml-06 GPU 4
echo "  [SMOKE SD3] siml-06 GPU 4..."
SMOKE_OK_SD3=0
ssh -o ConnectTimeout=60 siml-06 "
    cd ${REPO} && \
    CUDA_VISIBLE_DEVICES=4 ${PYTHON_GEN} \
        scripts/sd3/generate_sd3_safegen.py \
        --prompts CAS_SpatialCFG/prompts/mja_sexual.txt \
        --outdir /tmp/smoke_sd3 \
        --device cuda:0 --no_cpu_offload \
        --probe_mode both --cas_threshold 0.6 \
        --safety_scale 1.0 --attn_threshold 0.1 \
        --how_mode anchor_inpaint \
        --family_guidance \
        --family_config CAS_SpatialCFG/exemplars/concepts_v2/sexual/clip_grouped.pt \
        --end_idx 1 --steps 14 2>&1 | tail -5
" 2>/dev/null && SMOKE_OK_SD3=1 || true
echo "  SD3 smoke: $([ $SMOKE_OK_SD3 -eq 1 ] && echo PASS || echo FAIL)"

# FLUX1 smoke test on siml-09 GPU 0
echo "  [SMOKE FLUX1] siml-09 GPU 0..."
SMOKE_OK_FLUX1=0
ssh -o ConnectTimeout=120 siml-09 "
    cd ${REPO} && \
    CUDA_VISIBLE_DEVICES=0 ${PYTHON_GEN} \
        CAS_SpatialCFG/generate_flux1_v1.py \
        --prompts CAS_SpatialCFG/prompts/mja_sexual.txt \
        --outdir /tmp/smoke_flux1 \
        --height 1024 --width 1024 \
        --device cuda:0 \
        --probe_mode both --cas_threshold 0.6 \
        --safety_scale 1.5 --attn_threshold 0.1 \
        --how_mode anchor_inpaint \
        --family_guidance \
        --family_config CAS_SpatialCFG/exemplars/concepts_v2/sexual/clip_grouped.pt \
        --end_idx 1 --steps 14 2>&1 | tail -5
" 2>/dev/null && SMOKE_OK_FLUX1=1 || true
echo "  FLUX1 smoke: $([ $SMOKE_OK_FLUX1 -eq 1 ] && echo PASS || echo FAIL)"

if [[ $SMOKE_OK_SD14 -eq 0 ]] || [[ $SMOKE_OK_SD3 -eq 0 ]] || [[ $SMOKE_OK_FLUX1 -eq 0 ]]; then
    echo ""
    echo "ABORT: Smoke test(s) failed. Fix before launching."
    exit 1
fi

# ─── Step 6: Launch all workers ──────────────────────────────────────────
echo ""
echo "=== Step 6: Launching 15 workers ==="

declare -A PIDS

for worker in "${WORKERS[@]}"; do
    HOST=$(echo $worker | awk '{print $1}')
    GPU=$(echo $worker | awk '{print $2}')
    WLOG="${LOG_ROOT}/worker_${HOST}_g${GPU}.out"

    echo -n "  Launching ${HOST} GPU ${GPU}... "
    PID=$(ssh -o ConnectTimeout=10 "${HOST}" "
        cd ${REPO} && \
        nohup bash scripts/launch_0420/worker.sh ${HOST} ${GPU} \
            > ${WLOG} 2>&1 & echo \$!
    " 2>/dev/null)

    if [[ -n "${PID}" ]]; then
        echo "PID=${PID}"
        PIDS["${HOST}_${GPU}"]="${PID}"
    else
        echo "FAILED to launch"
    fi
done

# ─── Summary ─────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "LAUNCH SUMMARY"
echo "================================================================"
echo "Workers launched:"
for worker in "${WORKERS[@]}"; do
    HOST=$(echo $worker | awk '{print $1}')
    GPU=$(echo $worker | awk '{print $2}')
    KEY="${HOST}_${GPU}"
    PID="${PIDS[$KEY]:-FAILED}"
    MANIFEST="${MANIFEST_DIR}/worker_${HOST}_g${GPU}.csv"
    NJOBS=$(ssh -o ConnectTimeout=5 "${HOST}" "wc -l < ${MANIFEST} 2>/dev/null || echo 0" 2>/dev/null)
    NJOBS=$((${NJOBS:-0} - 1))  # subtract header
    echo "  ${HOST} GPU ${GPU}: PID=${PID} jobs=${NJOBS}"
done

echo ""
echo "Status check: bash ${SCRIPT_DIR}/status_all.sh"
echo "Logs: ${LOG_ROOT}/"
echo "================================================================"
