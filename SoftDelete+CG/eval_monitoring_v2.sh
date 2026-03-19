#!/bin/bash
# Quick VLM eval for monitoring_v2_grid experiments (images already generated)
cd /mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs/monitoring_v2_grid/ringabell

eval "$(conda shell.bash hook 2>/dev/null)" && conda activate vlm
export PYTHONNOUSERSITE=1

VLM_SCRIPT="/mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
NUM_GPUS=${1:-8}

# Build list of dirs needing eval
declare -a NEED_EVAL=()
for d in */; do
    [ "$d" = "logs/" ] && continue
    cnt=$(ls "${d}"*.png 2>/dev/null | wc -l)
    if [ "$cnt" -gt 0 ] && [ ! -f "${d}results_qwen3_vl_nudity.txt" ]; then
        NEED_EVAL+=("$d")
    fi
done

TOTAL=${#NEED_EVAL[@]}
echo "Need VLM eval: $TOTAL experiments"
echo "GPUs: $NUM_GPUS"

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=()
    for ((i=0; i<NUM_GPUS; i++)); do GPU_LIST+=($i); done
fi

declare -A GPU_PIDS
for g in "${GPU_LIST[@]}"; do GPU_PIDS[$g]=0; done

wait_for_gpu() {
    while true; do
        for g in "${GPU_LIST[@]}"; do
            pid=${GPU_PIDS[$g]}
            if [ "$pid" -eq 0 ] || ! kill -0 "$pid" 2>/dev/null; then
                GPU_PIDS[$g]=0
                echo "$g"
                return
            fi
        done
        sleep 3
    done
}

DONE=0
for d in "${NEED_EVAL[@]}"; do
    GPU=$(wait_for_gpu)
    DONE=$((DONE + 1))
    dname="${d%/}"
    echo "[$DONE/$TOTAL] GPU $GPU: $dname"

    (
        eval "$(conda shell.bash hook 2>/dev/null)" && conda activate vlm
        export PYTHONNOUSERSITE=1
        export PYTHONPATH=""
        CUDA_VISIBLE_DEVICES=$GPU /mnt/home/yhgil99/.conda/envs/vlm/bin/python "$VLM_SCRIPT" "$d" nudity qwen 2>&1
    ) > "logs/vlm_${dname}.log" 2>&1 &

    GPU_PIDS[$GPU]=$!
done

echo "Waiting for remaining..."
wait

echo ""
echo "=== RESULTS ==="
for d in */; do
    f="${d}results_qwen3_vl_nudity.txt"
    if [ -f "$f" ]; then
        sr=$(grep "SR " "$f" | grep -oP '[\d.]+%' | head -1)
        echo "SR=${sr} ${d%/}"
    fi
done | sort -t= -k2 -rn | head -30

echo ""
echo "Done!"
