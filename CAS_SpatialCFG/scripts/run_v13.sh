#!/bin/bash
# =============================================================================
# v13: Exemplar Cross-Attention Probe — Full Pipeline
# Step 1: Prepare CLIP exemplar embeddings
# Step 2: Grid search over key hyperparameters on Ring-A-Bell
# Step 3: Evaluate with NudeNet + Qwen3-VL
# Step 4: Best config → run on all datasets + COCO
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../scripts/lib/repo_env.sh
source "${SCRIPT_DIR}/../../scripts/lib/repo_env.sh"

# --- Config ---
GPU=${1:-0}
PYTHON="${UNLEARNING_SDD_COPY_PYTHON}"
PYTHON_VLM="${UNLEARNING_VLM_PYTHON}"
EVAL_NUDENET="${UNLEARNING_REPO_ROOT}/vlm/eval_nudenet.py"
EVAL_QWEN="${UNLEARNING_REPO_ROOT}/vlm/opensource_vlm_i2p_all.py"
WORKDIR="${UNLEARNING_REPO_ROOT}/CAS_SpatialCFG"

cd "$WORKDIR"

export CUDA_VISIBLE_DEVICES=$GPU
export XFORMERS_DISABLED=1

echo "=============================================="
echo "v13: Exemplar Cross-Attention Probe Pipeline"
echo "GPU: $GPU"
echo "=============================================="

# =============================================================================
# STEP 1: Prepare CLIP exemplar embeddings (if not exists)
# =============================================================================
CLIP_EMB="exemplars/sd14/clip_exemplar_embeddings.pt"
if [ ! -f "$CLIP_EMB" ]; then
    echo ""
    echo ">>> STEP 1: Preparing CLIP exemplar embeddings..."
    $PYTHON prepare_clip_exemplar.py \
        --exemplar_dir exemplars/sd14/exemplar_images \
        --output "$CLIP_EMB" \
        --projection simple \
        --n_tokens 4
    echo ">>> CLIP embeddings saved to $CLIP_EMB"
else
    echo ">>> STEP 1: CLIP embeddings already exist at $CLIP_EMB, skipping."
fi

# =============================================================================
# STEP 2: Grid search on Ring-A-Bell
# =============================================================================
echo ""
echo ">>> STEP 2: Grid search on Ring-A-Bell..."

RINGABELL="prompts/ringabell_anchor_subset.csv"
OUTBASE="outputs/v13"

# Grid search parameters:
# - probe_source: clip_exemplar vs text (v6 baseline) vs both
# - guide_mode: hybrid
# - safety_scale: 1.0, 1.5, 2.0
# - spatial_threshold: 0.2, 0.3
# - sigmoid_alpha: 10, 15

declare -a CONFIGS=(
    # probe_source | guide_mode | safety_scale | spatial_threshold | sigmoid_alpha | label
    "clip_exemplar|hybrid|1.0|0.3|10|clip_hyb_ss10_st03"
    "clip_exemplar|hybrid|1.5|0.3|10|clip_hyb_ss15_st03"
    "clip_exemplar|hybrid|2.0|0.3|10|clip_hyb_ss20_st03"
    "clip_exemplar|hybrid|1.5|0.2|10|clip_hyb_ss15_st02"
    "clip_exemplar|hybrid|1.5|0.3|15|clip_hyb_ss15_st03_a15"
    "clip_exemplar|hybrid|1.0|0.2|15|clip_hyb_ss10_st02_a15"
    "text|hybrid|1.5|0.3|10|text_hyb_ss15_st03"
    "both|hybrid|1.5|0.3|10|both_hyb_ss15_st03"
    "clip_exemplar|sld|3.0|0.3|10|clip_sld_ss30_st03"
    "clip_exemplar|hybrid_proj|1.0|0.3|10|clip_hybproj_ss10_st03"
)

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r PROBE GUIDE SS ST SA LABEL <<< "$cfg"
    OUTDIR="${OUTBASE}/ringabell_${LABEL}"

    if [ -f "${OUTDIR}/stats.json" ]; then
        echo "  SKIP $LABEL (already exists)"
        continue
    fi

    echo ""
    echo "  >>> Running: $LABEL"
    echo "      probe=$PROBE guide=$GUIDE ss=$SS st=$ST sa=$SA"

    $PYTHON generate_v13.py \
        --prompts "$RINGABELL" \
        --outdir "$OUTDIR" \
        --clip_embeddings "$CLIP_EMB" \
        --probe_source "$PROBE" \
        --guide_mode "$GUIDE" \
        --safety_scale "$SS" \
        --spatial_threshold "$ST" \
        --sigmoid_alpha "$SA" \
        --cas_threshold 0.6 \
        --nsamples 4 \
        --steps 50 \
        --seed 42

    echo "  >>> NudeNet eval: $LABEL"
    $PYTHON "$EVAL_NUDENET" "$OUTDIR"
done

echo ""
echo ">>> STEP 2 DONE: Grid search complete."
echo ""

# =============================================================================
# STEP 3: Qwen3-VL evaluation on all Ring-A-Bell configs
# =============================================================================
echo ">>> STEP 3: Qwen3-VL evaluation..."

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r PROBE GUIDE SS ST SA LABEL <<< "$cfg"
    OUTDIR="${OUTBASE}/ringabell_${LABEL}"

    if unlearning_find_qwen_result_txt "${OUTDIR}" >/dev/null 2>&1; then
        echo "  SKIP Qwen: $LABEL (already exists)"
        continue
    fi

    if [ ! -d "$OUTDIR" ]; then
        echo "  SKIP Qwen: $LABEL (output dir missing)"
        continue
    fi

    echo "  >>> Qwen eval: $LABEL"
    $PYTHON_VLM "$EVAL_QWEN" "$OUTDIR" nudity qwen
done

echo ""
echo ">>> STEP 3 DONE: Qwen evaluation complete."

# =============================================================================
# STEP 4: Summary — print all results
# =============================================================================
echo ""
echo "=============================================="
echo ">>> STEP 4: Results Summary (Ring-A-Bell)"
echo "=============================================="
echo ""
printf "%-40s | %10s | %10s | %10s\n" "Config" "NudeNet%" "Qwen Full%" "Qwen SR%"
printf "%-40s-+-%10s-+-%10s-+-%10s\n" "$(printf '%0.s-' {1..40})" "----------" "----------" "----------"

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r PROBE GUIDE SS ST SA LABEL <<< "$cfg"
    OUTDIR="${OUTBASE}/ringabell_${LABEL}"

    # NudeNet
    NN_RESULT="$(unlearning_nudenet_percent "${OUTDIR}" 2>/dev/null | tr -d '%' || echo N/A)"

    # Qwen
    QWEN_FULL="$(unlearning_qwen_percent_value "${OUTDIR}" Full || echo N/A)"
    QWEN_SR="$(unlearning_qwen_percent_value "${OUTDIR}" SR || echo N/A)"

    printf "%-40s | %10s | %10s | %10s\n" "$LABEL" "${NN_RESULT:-N/A}" "${QWEN_FULL:-N/A}" "${QWEN_SR:-N/A}"
done

echo ""
echo "=============================================="
echo ">>> Pipeline complete!"
echo ">>> Check outputs in: ${OUTBASE}/"
echo "=============================================="
