#!/bin/bash
# Run SHGD ablation study
# Tests each component independently to measure contribution
# Usage: bash run_ablation.sh [gpu_id]

PYTHON="PYTHONNOUSERSITE=1 /mnt/home/yhgil99/.conda/envs/sfgd/bin/python"

GPU=${1:-"0"}
PROMPT_FILE="../rab_grid_search/data/ringabell_full.txt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="outputs/ablation_${TIMESTAMP}"

cd "$(dirname "$0")"

echo "=============================================="
echo "SHGD Ablation Study"
echo "=============================================="

# 1. Baseline: Standard SD (no safety)
echo "[1/7] Standard SD (baseline)..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON generate.py \
    --config configs/default.yaml \
    --prompt_file "$PROMPT_FILE" \
    --save_dir "$BASE_DIR/baseline_std" \
    --anchor_guidance_scale 0.0 \
    --guide_start_frac 0.0 \
    --guide_end_frac 0.0 \
    --device cuda:0

# 2. Negative guidance only (no anchor, no heal)
echo "[2/7] Negative guidance only..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON generate.py \
    --config configs/default.yaml \
    --prompt_file "$PROMPT_FILE" \
    --save_dir "$BASE_DIR/neg_guidance_only" \
    --anchor_guidance_scale 3.0 \
    --guide_start_frac 0.8 \
    --guide_end_frac 0.0 \
    --consistency_threshold 0.0 \
    --device cuda:0

# 3. Dual-Anchor CFG only (no heal)
echo "[3/7] Dual-Anchor CFG only..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON generate.py \
    --config configs/default.yaml \
    --prompt_file "$PROMPT_FILE" \
    --save_dir "$BASE_DIR/dual_anchor_only" \
    --anchor_guidance_scale 3.0 \
    --guide_start_frac 0.8 \
    --guide_end_frac 0.0 \
    --consistency_threshold 0.0 \
    --device cuda:0

# 4. Full SHGD (Dual-Anchor + Heal, fixed schedule)
echo "[4/7] SHGD (fixed schedule)..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON generate.py \
    --config configs/default.yaml \
    --prompt_file "$PROMPT_FILE" \
    --save_dir "$BASE_DIR/shgd_fixed" \
    --anchor_guidance_scale 3.0 \
    --guide_start_frac 0.8 \
    --guide_end_frac 0.4 \
    --heal_strength 0.3 \
    --consistency_threshold 0.0 \
    --device cuda:0

# 5. Full SHGD (Dual-Anchor + Heal + Self-Consistency)
echo "[5/7] SHGD (adaptive)..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON generate.py \
    --config configs/default.yaml \
    --prompt_file "$PROMPT_FILE" \
    --save_dir "$BASE_DIR/shgd_adaptive" \
    --anchor_guidance_scale 3.0 \
    --guide_start_frac 0.8 \
    --guide_end_frac 0.4 \
    --heal_strength 0.3 \
    --consistency_threshold 0.85 \
    --device cuda:0

# 6. SHGD with micro-healing
echo "[6/7] SHGD (micro-heal)..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON generate.py \
    --config configs/micro_heal.yaml \
    --prompt_file "$PROMPT_FILE" \
    --save_dir "$BASE_DIR/shgd_micro_heal" \
    --device cuda:0

# 7. SHGD aggressive
echo "[7/7] SHGD (aggressive)..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON generate.py \
    --config configs/aggressive.yaml \
    --prompt_file "$PROMPT_FILE" \
    --save_dir "$BASE_DIR/shgd_aggressive" \
    --device cuda:0

echo ""
echo "=============================================="
echo "Ablation complete. Running evaluation..."
echo "=============================================="

# Evaluate all
for exp_dir in $BASE_DIR/*/; do
    exp_name=$(basename "$exp_dir")
    echo "Evaluating: $exp_name"
    $PYTHON evaluate.py \
        --image_dir "${exp_dir}all" \
        --eval_type nudenet \
        --output "${exp_dir}eval_nudenet.json" 2>/dev/null
done

# Print summary
echo ""
echo "=============================================="
echo "Ablation Results Summary"
echo "=============================================="
printf "%-25s %10s %10s %10s\n" "Experiment" "Safe" "Unsafe" "Rate"
echo "--------------------------------------------------------------"
for exp_dir in $BASE_DIR/*/; do
    exp_name=$(basename "$exp_dir")
    if [ -f "${exp_dir}eval_nudenet.json" ]; then
        safe=$(python -c "import json; d=json.load(open('${exp_dir}eval_nudenet.json')); print(d.get('safe', 'N/A'))" 2>/dev/null)
        unsafe=$(python -c "import json; d=json.load(open('${exp_dir}eval_nudenet.json')); print(d.get('unsafe', 'N/A'))" 2>/dev/null)
        rate=$(python -c "import json; d=json.load(open('${exp_dir}eval_nudenet.json')); print(f\"{d.get('safety_rate', 0):.2%}\")" 2>/dev/null)
        printf "%-25s %10s %10s %10s\n" "$exp_name" "$safe" "$unsafe" "$rate"
    fi
done

echo ""
echo "Full results in: $BASE_DIR"
