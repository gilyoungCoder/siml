#!/usr/bin/env bash
# MJA exemplar image generation — siml-05, GPUs 0-7
# 4 concepts (sexual / violent / disturbing / illegal) x 3 backbones (sd14 / sd3 / flux1)
# = 12 jobs bin-packed across 8 GPUs.
#
# Launch (from repo root):
#   bash scripts/mja_exemplars/run_all_siml05.sh
#
# Monitoring:
#   tail -f logs/mja_exemplars/*.log
#   nvidia-smi

set -euo pipefail

HOST=$(hostname | tr '[:upper:]' '[:lower:]')
if [[ "$HOST" != *siml-05* && "$HOST" != *siml05* ]]; then
    echo "[WARN] hostname is '$HOST', expected siml-05. Continuing anyway (override with FORCE=1)."
    if [[ "${FORCE:-0}" != "1" ]]; then
        echo "       export FORCE=1 to bypass this check."
        exit 1
    fi
fi

REPO_ROOT="/mnt/home3/yhgil99/unlearning"
PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
GEN="${REPO_ROOT}/SafeGen/safegen/gen_mja_exemplar_images.py"
PACK_ROOT="${REPO_ROOT}/SafeGen/configs/concept_packs"
OUT_ROOT="${REPO_ROOT}/CAS_SpatialCFG/exemplars/mja_v1"
LOG_ROOT="${REPO_ROOT}/logs/mja_exemplars"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

# Allow flux bf16 via env if desired; otherwise fp16 default
FLUX_DTYPE="${FLUX_DTYPE:-bf16}"

# run_job <gpu_id> <concept> <backbone> [extra args...]
run_job() {
    local gpu="$1"; local concept="$2"; local backbone="$3"; shift 3
    local log="${LOG_ROOT}/${concept}_${backbone}_gpu${gpu}.log"
    echo "[GPU ${gpu}] ${concept}/${backbone} -> ${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${GEN}" \
        --concept "${concept}" \
        --backbone "${backbone}" \
        --concept_pack "${PACK_ROOT}/mja_${concept}" \
        --output_root "${OUT_ROOT}" \
        "$@" \
        >"${log}" 2>&1
}

# run_chain <gpu_id> "<concept1>:<backbone1>" "<concept2>:<backbone2>" ...
# Sequentially runs all jobs pinned to the same GPU.
run_chain() {
    local gpu="$1"; shift
    for spec in "$@"; do
        local concept="${spec%%:*}"
        local backbone="${spec##*:}"
        local extra=()
        if [[ "${backbone}" == "flux1" ]]; then
            extra+=(--dtype "${FLUX_DTYPE}")
        fi
        run_job "${gpu}" "${concept}" "${backbone}" "${extra[@]}"
    done
}

echo "==============================================================="
echo " MJA exemplar generation on $(hostname)"
echo " GPUs: 0 1 2 3 4 5 6 7"
echo " Output root: ${OUT_ROOT}"
echo " Log root:    ${LOG_ROOT}"
echo "==============================================================="

# Bin-pack: FLUX dominates (slow), SD1.4 lightest.
#   GPU0  flux/sexual
#   GPU1  flux/violent
#   GPU2  flux/disturbing
#   GPU3  flux/illegal
#   GPU4  sd3/sexual    -> sd3/violent
#   GPU5  sd3/disturbing -> sd3/illegal
#   GPU6  sd14/sexual   -> sd14/violent
#   GPU7  sd14/disturbing -> sd14/illegal

run_chain 0 "sexual:flux1" &
PID_GPU0=$!
run_chain 1 "violent:flux1" &
PID_GPU1=$!
run_chain 2 "disturbing:flux1" &
PID_GPU2=$!
run_chain 3 "illegal:flux1" &
PID_GPU3=$!
run_chain 4 "sexual:sd3" "violent:sd3" &
PID_GPU4=$!
run_chain 5 "disturbing:sd3" "illegal:sd3" &
PID_GPU5=$!
run_chain 6 "sexual:sd14" "violent:sd14" &
PID_GPU6=$!
run_chain 7 "disturbing:sd14" "illegal:sd14" &
PID_GPU7=$!

echo "PIDs: 0=$PID_GPU0 1=$PID_GPU1 2=$PID_GPU2 3=$PID_GPU3 4=$PID_GPU4 5=$PID_GPU5 6=$PID_GPU6 7=$PID_GPU7"

# Wait for all; exit non-zero if any failed.
FAIL=0
for pid in "$PID_GPU0" "$PID_GPU1" "$PID_GPU2" "$PID_GPU3" \
           "$PID_GPU4" "$PID_GPU5" "$PID_GPU6" "$PID_GPU7"; do
    if ! wait "$pid"; then
        FAIL=1
        echo "[ERROR] worker pid=$pid exited non-zero"
    fi
done

if [[ "$FAIL" -eq 0 ]]; then
    echo "[DONE] all 12 jobs finished OK"
else
    echo "[DONE-WITH-ERRORS] see logs in ${LOG_ROOT}"
fi
exit "$FAIL"
