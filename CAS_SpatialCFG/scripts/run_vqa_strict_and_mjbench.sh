#!/bin/bash
# VQAScore Strict Alignment (anchor_strict) + MJ-Bench Safety Evaluation
# Uses all 8 GPUs in parallel

PYTHON_SDD="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
PYTHON_VLM="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
VQA_SCRIPT="/mnt/home3/yhgil99/unlearning/vlm/eval_vqascore_alignment.py"
MJ_SCRIPT="/mnt/home3/yhgil99/unlearning/vlm/eval_mjbench_safety.py"
PROMPTS_RB="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/anchor_strict_ringabell.csv"
PROMPTS_UD="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/anchor_strict_unlearndiff.csv"
OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs"

echo "=========================================="
echo "VQA Strict Alignment + MJ-Bench Evaluation"
echo "Started: $(date)"
echo "=========================================="

# ============================================================
# Phase 1: VQAScore Alignment on v3-v12 best configs (ringabell)
# ============================================================
echo ""
echo "=== Phase 1: VQA Alignment on v3-v12 best configs (ringabell) ==="

# GPU 0: baseline + v4
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 $PYTHON_SDD $VQA_SCRIPT \
    $OUT/v3/baseline --prompts $PROMPTS_RB --prompt_type all &

CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 $PYTHON_SDD $VQA_SCRIPT \
    $OUT/v4/sld_s10 --prompts $PROMPTS_RB --prompt_type all &

# GPU 1: v6 + v7
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 $PYTHON_SDD $VQA_SCRIPT \
    $OUT/v6/v6_crossattn_ts20_as15 --prompts $PROMPTS_RB --prompt_type all &

CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 $PYTHON_SDD $VQA_SCRIPT \
    $OUT/v7/v7_hyb_ts15_as15 --prompts $PROMPTS_RB --prompt_type all &

# GPU 2: v10 + v11 + v12
CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 $PYTHON_SDD $VQA_SCRIPT \
    $OUT/v10/v10_proj_ts2_as1 --prompts $PROMPTS_RB --prompt_type all &

CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 $PYTHON_SDD $VQA_SCRIPT \
    $OUT/v11/v11_proj_K4_eta03 --prompts $PROMPTS_RB --prompt_type all &

CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 $PYTHON_SDD $VQA_SCRIPT \
    $OUT/v12/v12_xattn_proj_ts2_as1 --prompts $PROMPTS_RB --prompt_type all &

wait
echo "Phase 1 done: $(date)"

# ============================================================
# Phase 2: VQAScore Alignment on v13 ringabell configs
# ============================================================
echo ""
echo "=== Phase 2: VQA Alignment on v13 ringabell configs ==="

V13_RB_DIRS=(
    ringabell_clip_hyb_ss10_st03
    ringabell_clip_hyb_ss15_st03
    ringabell_clip_hyb_ss20_st03
    ringabell_clip_hyb_ss15_st02
    ringabell_clip_hyb_ss10_st02_a15
    ringabell_clip_hyb_ss15_st03_a15
    ringabell_clip_hybproj_ss10_st03
    ringabell_clip_hybproj_ss15_st03
    ringabell_clip_hybproj_ss20_st03
    ringabell_clip_hybproj_ss10_st04
    ringabell_clip_hybproj_ss10_st05_a15
    ringabell_clip_hybproj_ss15_st04
    ringabell_clip_proj_ss10_st03
    ringabell_clip_proj_ss15_st03
    ringabell_clip_sld_ss30_st03
    ringabell_clip_sld_ss30_st04
    ringabell_clip_sld_ss50_st03
    ringabell_both_hyb_ss15_st03
    ringabell_text_hyb_ss15_st03
)

# Distribute across GPUs 0-7
GPU_IDX=0
for dir in "${V13_RB_DIRS[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU_IDX PYTHONNOUSERSITE=1 $PYTHON_SDD $VQA_SCRIPT \
        $OUT/v13/$dir --prompts $PROMPTS_RB --prompt_type all &
    GPU_IDX=$(( (GPU_IDX + 1) % 8 ))
done

wait
echo "Phase 2 done: $(date)"

# ============================================================
# Phase 3: VQAScore Alignment on v13 unlearndiff configs
# ============================================================
echo ""
echo "=== Phase 3: VQA Alignment on v13 unlearndiff configs ==="

CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 $PYTHON_SDD $VQA_SCRIPT \
    $OUT/v13/unlearndiff_clip_hybproj_ss10_st03 --prompts $PROMPTS_UD --prompt_type all &

CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 $PYTHON_SDD $VQA_SCRIPT \
    $OUT/v13/unlearndiff_clip_proj_ss15_st03 --prompts $PROMPTS_UD --prompt_type all &

wait
echo "Phase 3 done: $(date)"

# ============================================================
# Phase 4: MJ-Bench Safety Evaluation with Qwen3-VL
# ============================================================
echo ""
echo "=== Phase 4: MJ-Bench Safety Evaluation ==="

CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 $PYTHON_VLM $MJ_SCRIPT \
    --subset all --output_dir /mnt/home3/yhgil99/unlearning/vlm/mjbench_results &

wait
echo "Phase 4 done: $(date)"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "=== VQA Alignment Summary (anchor_strict) ==="
echo "=========================================="
echo ""
echo "--- v3-v12 best configs (ringabell) ---"
for dir in baseline v4/sld_s10 v6/v6_crossattn_ts20_as15 v7/v7_hyb_ts15_as15 v10/v10_proj_ts2_as1 v11/v11_proj_K4_eta03 v12/v12_xattn_proj_ts2_as1; do
    version=$(echo $dir | cut -d'/' -f1)
    f="$OUT/$dir/results_vqascore_alignment.txt"
    if [ -f "$f" ]; then
        echo "[$version] $(tail -3 "$f" | head -3)"
    fi
done

echo ""
echo "--- v13 ringabell configs ---"
for dir in "${V13_RB_DIRS[@]}"; do
    f="$OUT/v13/$dir/results_vqascore_alignment.txt"
    if [ -f "$f" ]; then
        echo "[$dir] $(tail -3 "$f" | head -3)"
    fi
done

echo ""
echo "--- v13 unlearndiff configs ---"
for dir in unlearndiff_clip_hybproj_ss10_st03 unlearndiff_clip_proj_ss15_st03; do
    f="$OUT/v13/$dir/results_vqascore_alignment.txt"
    if [ -f "$f" ]; then
        echo "[$dir] $(tail -3 "$f" | head -3)"
    fi
done

echo ""
echo "--- MJ-Bench Results ---"
if [ -f "/mnt/home3/yhgil99/unlearning/vlm/mjbench_results/mjbench_safety_all.txt" ]; then
    cat /mnt/home3/yhgil99/unlearning/vlm/mjbench_results/mjbench_safety_all.txt
fi

echo ""
echo "All done: $(date)"
