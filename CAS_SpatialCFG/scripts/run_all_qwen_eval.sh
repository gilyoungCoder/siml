#!/bin/bash
# ============================================================
# Run Qwen3-VL evaluation on ALL completed experiments
# Chains sequentially per GPU (waits for generation if needed)
# ============================================================
set -e

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/mnt/home3/yhgil99/.conda/envs/vlm/lib:$LD_LIBRARY_PATH
VLM_PYTHON="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
EVAL_VLM="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
BASE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG"

cd /tmp  # Avoid numpy source dir issues

echo "============================================"
echo "Qwen3-VL Evaluation Started: $(date)"
echo "============================================"

run_qwen() {
    local DIR=$1
    local GPU=$2
    local NAME=$(basename "$DIR")

    if [ ! -d "$DIR" ]; then return; fi

    # Wait for generation to complete (at least 100 images for COCO, 316 for Ring-A-Bell)
    local MIN_IMGS=100
    echo "$NAME" | grep -qi "coco" || MIN_IMGS=316

    local WAIT=0
    while [ "$(ls "$DIR"/*.png 2>/dev/null | wc -l)" -lt $MIN_IMGS ]; do
        if [ $WAIT -eq 0 ]; then
            echo "[GPU $GPU] Waiting for $NAME generation to complete..."
        fi
        WAIT=1
        sleep 120
    done

    if [ -f "$DIR/results_qwen_nudity.txt" ]; then
        echo "[GPU $GPU] SKIP Qwen $NAME (already done)"
        return
    fi

    echo "[GPU $GPU] Qwen3-VL: $NAME ($(ls "$DIR"/*.png 2>/dev/null | wc -l) images)"
    CUDA_VISIBLE_DEVICES=$GPU $VLM_PYTHON "$EVAL_VLM" "$DIR" nudity qwen 2>&1 | tail -3
    echo "[GPU $GPU] Done: $NAME"
}

# GPU 0: v6 experiments
(
    run_qwen "$BASE/outputs/v6/v6_crossattn_ts10_as15" 0
    run_qwen "$BASE/outputs/v6/v6_crossattn_ts15_as15" 0
    run_qwen "$BASE/outputs/v6/COCO_v6_crossattn" 0
) > "$BASE/outputs/qwen_eval_gpu0.log" 2>&1 &

# GPU 1: v7 experiments (part 1)
(
    run_qwen "$BASE/outputs/v7/v7_hyb_ts10_as15" 1
    run_qwen "$BASE/outputs/v7/v7_hyb_ts15_as15" 1
    run_qwen "$BASE/outputs/v7/v7_hyb_ts10_as10" 1
) > "$BASE/outputs/qwen_eval_gpu1.log" 2>&1 &

# GPU 2: v7 experiments (part 2)
(
    run_qwen "$BASE/outputs/v7/v7_sld_s10" 2
    run_qwen "$BASE/outputs/v7/v7_ainp_s10" 2
    run_qwen "$BASE/outputs/v7/COCO_v7_hyb_ts10_as15" 2
) > "$BASE/outputs/qwen_eval_gpu2.log" 2>&1 &

# GPU 3: v8 experiments (part 1)
(
    run_qwen "$BASE/outputs/v8/v8_proj_s5" 3
    run_qwen "$BASE/outputs/v8/v8_proj_s10" 3
    run_qwen "$BASE/outputs/v8/v8_proj_s15" 3
) > "$BASE/outputs/qwen_eval_gpu3.log" 2>&1 &

# GPU 4: v8 experiments (part 2)
(
    run_qwen "$BASE/outputs/v8/v8_dual_ts10_as15" 4
    run_qwen "$BASE/outputs/v8/v8_dual_ts15_as15" 4
    run_qwen "$BASE/outputs/v8/v8_projp_s1" 4
    run_qwen "$BASE/outputs/v8/v8_projp_s15" 4
    run_qwen "$BASE/outputs/v8/v8_hyb_ts10_as15" 4
    run_qwen "$BASE/outputs/v8/COCO_v8_proj_s10" 4
) > "$BASE/outputs/qwen_eval_gpu4.log" 2>&1 &

# GPU 5: v9 experiments (part 1)
(
    run_qwen "$BASE/outputs/v9/v9_exhyb_ts10_as15" 5
    run_qwen "$BASE/outputs/v9/v9_exhyb_ts15_as15" 5
    run_qwen "$BASE/outputs/v9/v9_exhyb_ts5_as10" 5
) > "$BASE/outputs/qwen_eval_gpu5.log" 2>&1 &

# GPU 6: v9 experiments (part 2)
(
    run_qwen "$BASE/outputs/v9/v9_exsld_s5" 6
    run_qwen "$BASE/outputs/v9/v9_exsld_s10" 6
    run_qwen "$BASE/outputs/v9/v9_contrast_s5" 6
    run_qwen "$BASE/outputs/v9/v9_contrast_s10" 6
) > "$BASE/outputs/qwen_eval_gpu6.log" 2>&1 &

# GPU 7: v9 remaining
(
    run_qwen "$BASE/outputs/v9/v9_inpaint_s1" 7
    run_qwen "$BASE/outputs/v9/COCO_v9_exhyb" 7
) > "$BASE/outputs/qwen_eval_gpu7.log" 2>&1 &

wait
echo "============================================"
echo "ALL QWEN EVALUATION COMPLETE: $(date)"
echo "============================================"

# Print final summary
echo ""
echo "=== FULL RESULTS ==="
printf "%-35s | %8s | %8s | %8s\n" "Experiment" "NN%" "SR%" "Full%"
printf "%s\n" "$(printf '%.0s-' {1..70})"

for ver in v6 v7 v8 v9; do
    for d in "$BASE"/outputs/$ver/*/; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        [[ "$name" == *debug* ]] && continue
        [[ "$name" == *maps* ]] && continue

        nn="-"
        [ -f "$d/results_nudenet.txt" ] && nn=$(grep -oP 'Unsafe Rate: \K[\d.]+' "$d/results_nudenet.txt" 2>/dev/null || echo "-")

        sr="-"; full="-"
        if [ -f "$d/results_qwen_nudity.txt" ]; then
            s=$(grep -oP 'Safe: \K\d+' "$d/results_qwen_nudity.txt" 2>/dev/null || echo 0)
            p=$(grep -oP 'Partial: \K\d+' "$d/results_qwen_nudity.txt" 2>/dev/null || echo 0)
            nr=$(grep -oP 'NotRel: \K\d+' "$d/results_qwen_nudity.txt" 2>/dev/null || echo 0)
            f=$(grep -oP 'Full: \K\d+' "$d/results_qwen_nudity.txt" 2>/dev/null || echo 0)
            t=$((s+p+nr+f))
            if [ "$t" -gt 0 ]; then
                sr=$(echo "scale=1; ($s+$p)*100/$t" | bc 2>/dev/null || echo "-")
                full=$(echo "scale=1; $f*100/$t" | bc 2>/dev/null || echo "-")
            fi
        fi

        printf "%-35s | %8s | %8s | %8s\n" "$name" "$nn" "$sr" "$full"
    done
done
