#!/bin/bash
PY=/mnt/home/yhgil99/.conda/envs/sdd_copy/bin/python
EVAL=/mnt/home/yhgil99/unlearning/vlm/eval_nudenet.py
DIRS=()
for base in /mnt/home/yhgil99/unlearning/CAS_SpatialCFG/outputs/v3 \
            /mnt/home/yhgil99/unlearning/CAS_SpatialCFG/outputs/v4 \
            /mnt/home/yhgil99/unlearning/CAS_SpatialCFG/outputs/v5 \
            /mnt/home/yhgil99/unlearning/AMG/outputs; do
    for d in "$base"/*/; do
        ls "$d"/*.png &>/dev/null && DIRS+=("$d")
    done
done
echo "Total: ${#DIRS[@]} | GPU 1-7 | $(date)"
idx=0; batch=1
while [ $idx -lt ${#DIRS[@]} ]; do
    pids=()
    for gpu in 1 2 3 4 5 6 7; do
        if [ $idx -lt ${#DIRS[@]} ]; then
            d="${DIRS[$idx]}"
            savepath="${d}results_nudenet_06.txt"
            CUDA_VISIBLE_DEVICES=$gpu $PY $EVAL "$d" --threshold 0.6 --save_path "$savepath" >/dev/null 2>&1 &
            pids+=($!)
            idx=$((idx+1))
        fi
    done
    for p in "${pids[@]}"; do wait $p; done
    echo "Batch $batch ($idx/${#DIRS[@]})"
    batch=$((batch+1))
done
echo "DONE $(date)"
