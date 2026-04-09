#!/bin/bash
# =============================================================================
# v13: Run best configs on ALL datasets (MMA, P4DN, UnlearnDiff, COCO)
# Uses 8 GPUs in parallel
# =============================================================================

set -e

PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
PYTHON_VLM=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL_NN="/mnt/home3/yhgil99/unlearning/vlm/eval_nudenet.py"
EVAL_QWEN="/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py"
WORKDIR="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG"
CLIP_EMB="exemplars/sd14/clip_exemplar_embeddings.pt"
OUTBASE="outputs/v13"

cd "$WORKDIR"

# Best configs from grid search
# 1. clip_proj_ss15_st03: projection, ss=1.5, st=0.3, sa=10 (NudeNet 8.33%)
# 2. clip_hybproj_ss10_st03: hybrid_proj, ss=1.0, st=0.3, sa=10 (NudeNet 26.67%, SR 90%)
# 3. clip_hybproj_ss10_st05_a15: hybrid_proj, ss=1.0, st=0.5, sa=15 (NudeNet 15%)

# Datasets
declare -A DATASETS
DATASETS[ringabell]="prompts/ringabell_anchor_subset.csv"
DATASETS[mma]="prompts/mma-diffusion-nsfw-adv-prompts.csv"
DATASETS[p4dn]="prompts/p4dn_16_prompt.csv"
DATASETS[unlearndiff]="prompts/unlearn_diff_nudity.csv"
DATASETS[coco]="prompts/coco_250.txt"

# MMA: limit to 200 prompts
MMA_END=200

launch_gen() {
    local GPU=$1 DATASET=$2 PROMPTS=$3 GUIDE=$4 SS=$5 ST=$6 SA=$7 LABEL=$8 END_IDX=$9
    local OUTDIR="${OUTBASE}/${DATASET}_${LABEL}"

    if [ -f "${OUTDIR}/stats.json" ]; then
        echo "SKIP gen: ${DATASET}_${LABEL}"
        return
    fi

    echo ">>> GPU $GPU: gen ${DATASET}_${LABEL}"

    local EXTRA_ARGS=""
    if [ -n "$END_IDX" ] && [ "$END_IDX" -gt 0 ]; then
        EXTRA_ARGS="--end_idx $END_IDX"
    fi

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON generate_v13.py \
        --prompts "$PROMPTS" \
        --outdir "$OUTDIR" \
        --clip_embeddings "$CLIP_EMB" \
        --probe_source clip_exemplar \
        --guide_mode "$GUIDE" \
        --safety_scale "$SS" \
        --spatial_threshold "$ST" \
        --sigmoid_alpha "$SA" \
        --cas_threshold 0.6 \
        --nsamples 4 --steps 50 --seed 42 \
        $EXTRA_ARGS

    echo ">>> GPU $GPU: NudeNet ${DATASET}_${LABEL}"
    $PYTHON "$EVAL_NN" "$OUTDIR"
}

echo "=============================================="
echo "v13: All Datasets — 8 GPU Parallel"
echo "=============================================="

# =======================================
# PHASE 1: Generate + NudeNet (8 GPUs)
# =======================================
echo ""
echo ">>> PHASE 1: Generation + NudeNet"

# Best config 1: projection ss=1.5 st=0.3
launch_gen 0 mma "${DATASETS[mma]}" projection 1.5 0.3 10 clip_proj_ss15_st03 $MMA_END &
launch_gen 1 p4dn "${DATASETS[p4dn]}" projection 1.5 0.3 10 clip_proj_ss15_st03 0 &
launch_gen 2 unlearndiff "${DATASETS[unlearndiff]}" projection 1.5 0.3 10 clip_proj_ss15_st03 0 &
launch_gen 3 coco "${DATASETS[coco]}" projection 1.5 0.3 10 clip_proj_ss15_st03 0 &

# Best config 2: hybrid_proj ss=1.0 st=0.3
launch_gen 4 mma "${DATASETS[mma]}" hybrid_proj 1.0 0.3 10 clip_hybproj_ss10_st03 $MMA_END &
launch_gen 5 p4dn "${DATASETS[p4dn]}" hybrid_proj 1.0 0.3 10 clip_hybproj_ss10_st03 0 &
launch_gen 6 unlearndiff "${DATASETS[unlearndiff]}" hybrid_proj 1.0 0.3 10 clip_hybproj_ss10_st03 0 &
launch_gen 7 coco "${DATASETS[coco]}" hybrid_proj 1.0 0.3 10 clip_hybproj_ss10_st03 0 &

wait
echo ">>> PHASE 1 DONE"

# =======================================
# PHASE 2: Qwen3-VL evaluation (8 GPUs)
# =======================================
echo ""
echo ">>> PHASE 2: Qwen3-VL evaluation"

GPU=0
for LABEL in clip_proj_ss15_st03 clip_hybproj_ss10_st03; do
    for DATASET in mma p4dn unlearndiff coco; do
        OUTDIR="${OUTBASE}/${DATASET}_${LABEL}"
        if [ -f "${OUTDIR}/results_qwen3_vl_nudity.txt" ]; then
            echo "SKIP Qwen: ${DATASET}_${LABEL}"
            continue
        fi
        if [ ! -d "$OUTDIR" ]; then
            echo "SKIP Qwen: ${DATASET}_${LABEL} (no dir)"
            continue
        fi
        echo ">>> GPU $GPU: Qwen ${DATASET}_${LABEL}"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON_VLM "$EVAL_QWEN" "$OUTDIR" nudity qwen &
        GPU=$(( (GPU + 1) % 8 ))
    done
done

wait
echo ">>> PHASE 2 DONE"

# =======================================
# PHASE 3: Summary
# =======================================
echo ""
echo "=============================================="
echo ">>> FINAL RESULTS SUMMARY"
echo "=============================================="

printf "%-45s | %8s | %8s | %8s\n" "Config" "NudeNet%" "Qwen Full" "Qwen SR%"
printf "%-45s-+-%8s-+-%8s-+-%8s\n" "$(printf '%0.s-' {1..45})" "--------" "--------" "--------"

for d in ${OUTBASE}/*/; do
    label=$(basename "$d")
    nn="N/A"
    full="N/A"
    sr="N/A"

    if [ -f "${d}results_nudenet.txt" ]; then
        nn=$(grep "Unsafe Rate:" "${d}results_nudenet.txt" | grep -oP '[\d.]+(?=%)' | head -1)
    fi
    if [ -f "${d}results_qwen3_vl_nudity.txt" ]; then
        full=$(grep "Full:" "${d}results_qwen3_vl_nudity.txt" | grep -oP '[\d.]+(?=%)' | head -1)
        sr=$(grep "SR " "${d}results_qwen3_vl_nudity.txt" | grep -oP '[\d.]+(?=%)' | head -1)
    fi

    printf "%-45s | %7s%% | %7s%% | %7s%%\n" "$label" "$nn" "$full" "$sr"
done

echo ""
echo ">>> ALL DONE!"
