#!/usr/bin/env bash
# gpu_policy.sh — GPU allowlist and safety check function
# Source this file in all dispatch scripts.
#
# Allowed GPU sets (NEVER use GPUs outside these lists):
#   siml-01: 0 1 2 3 4 5 6 7  (all yhgil99)
#   siml-06: 4 5 6 7           (GPUs 0,1,2,3 are USED BY OTHERS — FORBIDDEN)
#   siml-08: 4 5               (GPU 0=giung2, GPU 6,7=others — FORBIDDEN)
#   siml-09: 0                  (H100, yhgil99 only)

SIML01_GPUS=(0 1 2 3 4 5 6 7)
SIML06_GPUS=(4 5 6 7)
SIML08_GPUS=(4 5)
SIML09_GPUS=(0)

PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
REPO=/mnt/home3/yhgil99/unlearning

# check_gpu_free <host> <gpu_id>
# Returns 0 if used_memory < 1000 MB (GPU is free), 1 if busy (abort).
check_gpu_free() {
    local host="$1"
    local gpu_id="$2"
    local used_mem
    used_mem=$(ssh "$host" \
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=${gpu_id}" \
        2>/dev/null | tr -d ' ')
    if [[ -z "$used_mem" ]]; then
        echo "ERROR: could not query ${host} GPU ${gpu_id}" >&2
        return 1
    fi
    if (( used_mem > 1000 )); then
        echo "ABORT GPU ${gpu_id} on ${host}: used_memory=${used_mem} MB (> 1000 MB threshold)" >&2
        return 1
    fi
    echo "OK: ${host} GPU ${gpu_id} is free (${used_mem} MB)"
    return 0
}
