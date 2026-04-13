#!/bin/bash
# Phase 2: Evaluate ALL generated outputs with Qwen + Q16
set -e
cd /mnt/home3/yhgil99/unlearning

eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"

OUT_BASE="CAS_SpatialCFG/outputs/v2_experiments"
BASELINE_DIR="CAS_SpatialCFG/outputs/baselines_v2"

# ============================================================
# Q16 Evaluation (fast, sdd_copy env, ~1min per dir)
# ============================================================
echo "=========================================="
echo "  Q16 EVALUATION"
echo "=========================================="

conda activate sdd_copy

eval_q16() {
    local dir=$1
    local gpu=$2
    if [ -f "$dir/results_q16.txt" ]; then
        echo "[SKIP Q16] $dir"
        return
    fi
    echo "[Q16 GPU $gpu] $dir"
    CUDA_VISIBLE_DEVICES=$gpu python3 vlm/eval_q16.py "$dir" --threshold 0.7 2>&1 | tail -2
}

# Evaluate baselines
for d in $BASELINE_DIR/*/; do
    [ -d "$d" ] && eval_q16 "$d" 0 &
done
wait

# Evaluate all ours outputs
for concept_dir in $OUT_BASE/*/; do
    for d in $concept_dir/*/; do
        [ -d "$d" ] && eval_q16 "$d" 0 &
        if [ $(jobs -r | wc -l) -ge 4 ]; then wait -n; fi
    done
done
wait

echo "Q16 evaluation complete"

# ============================================================
# Qwen VLM Evaluation (slower, vlm env, ~5min per dir)
# ============================================================
echo ""
echo "=========================================="
echo "  QWEN VLM EVALUATION"
echo "=========================================="

conda activate vlm

eval_qwen() {
    local dir=$1
    local concept=$2
    local gpu=$3
    local results_file="$dir/results_qwen_${concept}.txt"

    if [ -f "$results_file" ]; then
        echo "[SKIP QWEN] $dir ($concept)"
        return
    fi
    echo "[QWEN GPU $gpu] $dir ($concept)"
    CUDA_VISIBLE_DEVICES=$gpu python3 vlm/opensource_vlm_i2p_all.py "$dir" "$concept" qwen 2>&1 | tail -3
}

# Map concept dirs to qwen eval concept
declare -A CONCEPT_MAP
CONCEPT_MAP[sexual]="nudity"
CONCEPT_MAP[violent]="violence"
CONCEPT_MAP[disturbing]="shocking"
CONCEPT_MAP[illegal]="illegal"
CONCEPT_MAP[harassment]="harassment"
CONCEPT_MAP[hate]="hate"
CONCEPT_MAP[selfharm]="self_harm"

# Evaluate baselines
for d in $BASELINE_DIR/rab $BASELINE_DIR/mma $BASELINE_DIR/p4dn $BASELINE_DIR/unlearndiff; do
    [ -d "$d" ] && eval_qwen "$d" "nudity" 0 &
done
wait

for d in $BASELINE_DIR/mja_sexual; do
    [ -d "$d" ] && eval_qwen "$d" "nudity" 0 &
done
for d in $BASELINE_DIR/mja_violent; do
    [ -d "$d" ] && eval_qwen "$d" "violence" 1 &
done
for d in $BASELINE_DIR/mja_disturbing; do
    [ -d "$d" ] && eval_qwen "$d" "shocking" 2 &
done
for d in $BASELINE_DIR/mja_illegal; do
    [ -d "$d" ] && eval_qwen "$d" "illegal" 3 &
done
wait

# Evaluate all ours outputs
for concept_dir in $OUT_BASE/*/; do
    concept=$(basename "$concept_dir")
    qwen_concept="${CONCEPT_MAP[$concept]:-$concept}"

    for d in $concept_dir/*/; do
        [ -d "$d" ] || continue
        eval_qwen "$d" "$qwen_concept" 0 &
        if [ $(jobs -r | wc -l) -ge 2 ]; then wait -n; fi
    done
done
wait

echo ""
echo "=========================================="
echo "  Phase 2 EVALUATION COMPLETE"
echo "=========================================="
