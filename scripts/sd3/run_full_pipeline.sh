#!/bin/bash
# ============================================================================
# SD3 Full Pipeline: Generate → Evaluate → Collect Results
# Runs ONE experiment end-to-end on a single GPU
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_full_pipeline.sh <method> <dataset> <concept>
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0 bash run_full_pipeline.sh baseline rab sexual
#   CUDA_VISIBLE_DEVICES=1 bash run_full_pipeline.sh safree mja_violent violence
#   CUDA_VISIBLE_DEVICES=2 bash run_full_pipeline.sh safegen mja_sexual sexual
# ============================================================================

set -e

METHOD=${1:?Usage: run_full_pipeline.sh <method> <dataset> <concept>}
DATASET=${2:?Usage: run_full_pipeline.sh <method> <dataset> <concept>}
CONCEPT=${3:-sexual}

PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
PYTHON_VLM="/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10"
SD="/mnt/home3/yhgil99/unlearning/scripts/sd3"
VLM="/mnt/home3/yhgil99/unlearning/vlm"
OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/sd3/${METHOD}/${DATASET}"
LOG="/mnt/home3/yhgil99/unlearning/logs/sd3"
MJA="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts"
P="/mnt/home3/yhgil99/unlearning/SAFREE/datasets"

STEPS=28; CFG=7.0; RES=1024; SEED=42

mkdir -p "$LOG" "$OUT"

# ── Dataset → Prompt file mapping ──
declare -A PROMPT_MAP=(
    ["rab"]="$P/nudity-ring-a-bell.csv"
    ["unlearndiff"]="$P/unlearn_diff_nudity.csv"
    ["p4dn"]="$P/p4dn_16_prompt.csv"
    ["mma"]="$P/mma-diffusion-nsfw-adv-prompts.csv"
    ["mja_sexual"]="$MJA/mja_sexual.txt"
    ["mja_violent"]="$MJA/mja_violent.txt"
    ["mja_disturbing"]="$MJA/mja_disturbing.txt"
    ["mja_illegal"]="$MJA/mja_illegal.txt"
    ["coco250"]="$P/coco_30k_10k.csv"
)

# ── Concept → SafeGen target/anchor ──
declare -A TARGET_CONCEPTS=(
    ["sexual"]="nudity nude_person naked_body"
    ["violence"]="violence weapon blood gore"
    ["shocking"]="gore disturbing grotesque horrific"
    ["illegal"]="illegal crime drugs theft"
    ["none"]="nudity nude_person naked_body"
)
declare -A ANCHOR_CONCEPTS=(
    ["sexual"]="clothed_person person_wearing_clothes"
    ["violence"]="peaceful_scene nature_landscape"
    ["shocking"]="serene_landscape beautiful_art"
    ["illegal"]="legal_activity normal_daily_life"
    ["none"]="clothed_person person_wearing_clothes"
)

# ── Concept → SAFREE concept keyword ──
declare -A SAFREE_CONCEPT=(
    ["sexual"]="sexual"
    ["violence"]="violence"
    ["shocking"]="shocking"
    ["illegal"]="illegal"
    ["none"]="none"
)

# ── Concept → Qwen eval category ──
declare -A EVAL_CATEGORY=(
    ["sexual"]="nudity"
    ["violence"]="violence"
    ["shocking"]="shocking"
    ["illegal"]="illegal"
    ["none"]="nudity"
)

PROMPT="${PROMPT_MAP[$DATASET]}"
if [ -z "$PROMPT" ]; then
    echo "[ERROR] Unknown dataset: $DATASET"
    exit 1
fi

EXTRA_ARGS=""
if [ "$DATASET" = "coco250" ]; then
    EXTRA_ARGS="--end 250"
fi

echo "=============================================="
echo "SD3 Full Pipeline"
echo "  Method:  $METHOD"
echo "  Dataset: $DATASET"
echo "  Concept: $CONCEPT"
echo "  Prompts: $PROMPT"
echo "  Output:  $OUT"
echo "=============================================="
echo ""

# ============================================
# STEP 1: GENERATE
# ============================================
echo "$(date) [STEP 1] Generating images ..."

if [ "$METHOD" = "baseline" ]; then
    $PYTHON $SD/generate_sd3_baseline.py \
        --prompts "$PROMPT" --outdir "$OUT" \
        --steps $STEPS --cfg_scale $CFG --resolution $RES --seed $SEED \
        $EXTRA_ARGS 2>&1 | tee "$LOG/pipeline_${METHOD}_${DATASET}_gen.log"

elif [ "$METHOD" = "safree" ]; then
    $PYTHON $SD/generate_sd3_safree.py \
        --prompts "$PROMPT" --outdir "$OUT" \
        --concept "${SAFREE_CONCEPT[$CONCEPT]}" \
        --steps $STEPS --cfg_scale $CFG --resolution $RES --seed $SEED \
        $EXTRA_ARGS 2>&1 | tee "$LOG/pipeline_${METHOD}_${DATASET}_gen.log"

elif [ "$METHOD" = "safegen" ]; then
    # Convert underscored concepts to quoted args
    TARGS=""
    for t in ${TARGET_CONCEPTS[$CONCEPT]}; do
        TARGS="$TARGS \"$(echo $t | tr '_' ' ')\""
    done
    AANCH=""
    for a in ${ANCHOR_CONCEPTS[$CONCEPT]}; do
        AANCH="$AANCH \"$(echo $a | tr '_' ' ')\""
    done

    eval $PYTHON $SD/generate_sd3_safegen.py \
        --prompts "$PROMPT" --outdir "$OUT" \
        --target_concepts $TARGS \
        --anchor_concepts $AANCH \
        --cas_threshold 0.4 --how_mode anchor_inpaint --safety_scale 1.0 \
        --probe_mode text \
        --steps $STEPS --cfg_scale $CFG --resolution $RES --seed $SEED \
        $EXTRA_ARGS 2>&1 | tee "$LOG/pipeline_${METHOD}_${DATASET}_gen.log"
fi

NUM_IMAGES=$(ls "$OUT"/*.png 2>/dev/null | wc -l)
echo ""
echo "$(date) [STEP 1 DONE] Generated $NUM_IMAGES images"
echo ""

# ============================================
# STEP 2: EVALUATE (Qwen3-VL)
# ============================================
echo "$(date) [STEP 2] Evaluating with Qwen3-VL ..."
EVAL_CAT="${EVAL_CATEGORY[$CONCEPT]}"

if [ -f "$VLM/opensource_vlm_i2p_all.py" ] && [ -x "$PYTHON_VLM" ]; then
    $PYTHON_VLM $VLM/opensource_vlm_i2p_all.py "$OUT" "$EVAL_CAT" qwen \
        2>&1 | tee "$LOG/pipeline_${METHOD}_${DATASET}_eval.log"
    echo "$(date) [STEP 2 DONE] Qwen3-VL evaluation complete"
else
    echo "[SKIP] Qwen3-VL not available (vlm env or script missing)"
fi

echo ""

# ============================================
# STEP 3: EVALUATE (NudeNet — nudity only)
# ============================================
if [ "$CONCEPT" = "sexual" ] || [ "$CONCEPT" = "none" ]; then
    echo "$(date) [STEP 3] Evaluating with NudeNet ..."
    if [ -f "$VLM/eval_nudenet.py" ]; then
        $PYTHON $VLM/eval_nudenet.py "$OUT" \
            2>&1 | tee "$LOG/pipeline_${METHOD}_${DATASET}_nudenet.log"
        echo "$(date) [STEP 3 DONE] NudeNet evaluation complete"
    else
        echo "[SKIP] NudeNet eval script not found"
    fi
fi

echo ""
echo "=============================================="
echo "$(date) PIPELINE COMPLETE: ${METHOD}/${DATASET}"
echo "  Images: $OUT"
echo "  Results: $OUT/results_qwen_*.txt"
echo "=============================================="
