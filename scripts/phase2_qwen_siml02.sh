#!/bin/bash
# Phase 2-B: Qwen VLM evaluation on siml-02
set -e
cd /mnt/home3/yhgil99/unlearning

eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
conda activate vlm

OUT="CAS_SpatialCFG/outputs/v2_experiments"
BASELINE="CAS_SpatialCFG/outputs/baselines_v2"

echo "=== Qwen VLM Evaluation START $(date) ==="

eval_qwen() {
    local dir=$1 concept=$2 gpu=$3
    local rfile="$dir/results_qwen_${concept}.txt"
    [ -f "$rfile" ] && echo "[SKIP] $rfile" && return 0
    [ "$(ls $dir/*.png 2>/dev/null | wc -l)" -eq 0 ] && return 0
    echo "[GPU$gpu] Qwen $concept: $(basename $(dirname $dir))/$(basename $dir)"
    CUDA_VISIBLE_DEVICES=$gpu python3 vlm/opensource_vlm_i2p_all.py "$dir" "$concept" qwen 2>&1 | tail -3
    echo "[DONE GPU$gpu] $(basename $dir)"
}

# Concept mapping
declare -A QMAP
QMAP[sexual]="nudity"
QMAP[violent]="violence"
QMAP[disturbing]="shocking"
QMAP[illegal]="illegal"
QMAP[harassment]="harassment"
QMAP[hate]="hate"
QMAP[selfharm]="self_harm"

# ── Baselines first (GPU 0) ──
echo "--- Baselines ---"
for name in rab mma p4dn unlearndiff mja_sexual; do
    [ -d "$BASELINE/$name" ] && eval_qwen "$BASELINE/$name" "nudity" 0
done
for name in mja_violent; do
    [ -d "$BASELINE/$name" ] && eval_qwen "$BASELINE/$name" "violence" 0
done
for name in mja_disturbing; do
    [ -d "$BASELINE/$name" ] && eval_qwen "$BASELINE/$name" "shocking" 0
done
for name in mja_illegal; do
    [ -d "$BASELINE/$name" ] && eval_qwen "$BASELINE/$name" "illegal" 0
done

# ── Experiments: one concept per GPU ──
echo "--- Experiments ---"

# GPU 0: sexual experiments
(
for d in $OUT/sexual/*/; do
    [ -d "$d" ] && eval_qwen "$d" "nudity" 0
done
echo "[GPU0 QWEN COMPLETE]"
) &

# GPU 2: violent experiments
(
for d in $OUT/violent/*/; do
    [ -d "$d" ] && eval_qwen "$d" "violence" 2
done
echo "[GPU2 QWEN COMPLETE]"
) &

# GPU 3: disturbing experiments
(
for d in $OUT/disturbing/*/; do
    [ -d "$d" ] && eval_qwen "$d" "shocking" 3
done
echo "[GPU3 QWEN COMPLETE]"
) &

# GPU 4: illegal experiments
(
for d in $OUT/illegal/*/; do
    [ -d "$d" ] && eval_qwen "$d" "illegal" 4
done
echo "[GPU4 QWEN COMPLETE]"
) &

# GPU 5: harassment + hate experiments
(
for d in $OUT/harassment/*/; do
    [ -d "$d" ] && eval_qwen "$d" "harassment" 5
done
for d in $OUT/hate/*/; do
    [ -d "$d" ] && eval_qwen "$d" "hate" 5
done
echo "[GPU5 QWEN COMPLETE]"
) &

# GPU 6: selfharm experiments
(
for d in $OUT/selfharm/*/; do
    [ -d "$d" ] && eval_qwen "$d" "self_harm" 6
done
echo "[GPU6 QWEN COMPLETE]"
) &

wait

echo ""
echo "=== Qwen VLM Evaluation COMPLETE $(date) ==="

# Summary
echo ""
echo "=== QWEN RESULTS SUMMARY ==="
for concept_dir in $OUT/*/; do
    concept=$(basename "$concept_dir")
    qc="${QMAP[$concept]:-$concept}"
    echo "--- $concept ($qc) ---"
    for d in $concept_dir/*/; do
        rfile="$d/results_qwen_${qc}.txt"
        [ -f "$rfile" ] || continue
        sr=$(grep "SR " "$rfile" 2>/dev/null | head -1)
        echo "  $(basename $d): $sr"
    done
done
