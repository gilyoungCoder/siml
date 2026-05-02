#!/bin/bash
# ============================================================================
# SAFREE on FLUX.2-klein-4B — siml-04 (GPUs 3, 5 free)
#
# Datasets: RAB, MMA, P4DN, UnlearnDiff, MJA (sexual/violent/disturbing), COCO
# Method: SAFREE + SVF (original SAFREE method adapted for Flux)
#
# Usage:
#   ssh siml-04 "cd /mnt/home3/yhgil99/unlearning && \
#     nohup bash scripts/nohup_flux_safree_siml04.sh > logs/flux/safree_siml04.log 2>&1 &"
# ============================================================================
set -e

PY="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
BASE="/mnt/home3/yhgil99/unlearning"
GEN="$BASE/CAS_SpatialCFG/generate_flux2klein_safree.py"
PR="$BASE/SafeGen/prompts"
OUT="$BASE/CAS_SpatialCFG/outputs/flux2klein_experiments/safree"
LOG="$BASE/logs/flux/safree_siml04"

mkdir -p "$OUT" "$LOG"

echo "======================================================"
echo " SAFREE-Flux — siml-04 — $(date)"
echo "======================================================"

# ── Helper: skip if already done ──
count_imgs() {
    ls "$1"/*.png 2>/dev/null | wc -l
}

# ── GPU 3: MMA (longest job, 1000 prompts) ──
echo "[GPU 3] MMA (1000 prompts, sexual concept)..."
EXPECTED=1000
EXISTING=$(count_imgs "$OUT/mma" 2>/dev/null || echo 0)
if [ "$EXISTING" -ge "$EXPECTED" ]; then
    echo "  SKIP: mma already has $EXISTING/$EXPECTED images"
else
    CUDA_VISIBLE_DEVICES=3 nohup $PY "$GEN" \
        --prompts "$PR/mma.txt" \
        --outdir "$OUT/mma" \
        --concept sexual --safree --svf \
        --device cuda:0 \
        > "$LOG/mma.log" 2>&1 &
    PID_MMA=$!
    echo "  Started MMA → PID=$PID_MMA"
fi

# ── GPU 5: Sequential (small→large) ──
echo "[GPU 5] Sequential: RAB → MJA_sexual → MJA_violent → MJA_disturbing → P4DN → UnlearnDiff → COCO"

(
    # 1. Ring-A-Bell (79)
    EXPECTED=79
    EXISTING=$(count_imgs "$OUT/ringabell" 2>/dev/null || echo 0)
    if [ "$EXISTING" -ge "$EXPECTED" ]; then
        echo "  SKIP: ringabell ($EXISTING/$EXPECTED)"
    else
        echo "  [1/7] Ring-A-Bell..."
        CUDA_VISIBLE_DEVICES=5 $PY "$GEN" \
            --prompts "$PR/ringabell.txt" \
            --outdir "$OUT/ringabell" \
            --concept sexual --safree --svf \
            --device cuda:0 \
            > "$LOG/ringabell.log" 2>&1
        echo "  [1/7] Ring-A-Bell done! $(date)"
    fi

    # 2. MJA sexual (100)
    EXPECTED=100
    EXISTING=$(count_imgs "$OUT/mja_sexual" 2>/dev/null || echo 0)
    if [ "$EXISTING" -ge "$EXPECTED" ]; then
        echo "  SKIP: mja_sexual ($EXISTING/$EXPECTED)"
    else
        echo "  [2/7] MJA sexual..."
        CUDA_VISIBLE_DEVICES=5 $PY "$GEN" \
            --prompts "$PR/mja_sexual.txt" \
            --outdir "$OUT/mja_sexual" \
            --concept sexual --safree --svf \
            --device cuda:0 \
            > "$LOG/mja_sexual.log" 2>&1
        echo "  [2/7] MJA sexual done! $(date)"
    fi

    # 3. MJA violent (100)
    EXPECTED=100
    EXISTING=$(count_imgs "$OUT/mja_violent" 2>/dev/null || echo 0)
    if [ "$EXISTING" -ge "$EXPECTED" ]; then
        echo "  SKIP: mja_violent ($EXISTING/$EXPECTED)"
    else
        echo "  [3/7] MJA violent..."
        CUDA_VISIBLE_DEVICES=5 $PY "$GEN" \
            --prompts "$PR/mja_violent.txt" \
            --outdir "$OUT/mja_violent" \
            --concept violence --safree --svf \
            --device cuda:0 \
            > "$LOG/mja_violent.log" 2>&1
        echo "  [3/7] MJA violent done! $(date)"
    fi

    # 4. MJA disturbing (100)
    EXPECTED=100
    EXISTING=$(count_imgs "$OUT/mja_disturbing" 2>/dev/null || echo 0)
    if [ "$EXISTING" -ge "$EXPECTED" ]; then
        echo "  SKIP: mja_disturbing ($EXISTING/$EXPECTED)"
    else
        echo "  [4/7] MJA disturbing..."
        CUDA_VISIBLE_DEVICES=5 $PY "$GEN" \
            --prompts "$PR/mja_disturbing.txt" \
            --outdir "$OUT/mja_disturbing" \
            --concept shocking --safree --svf \
            --device cuda:0 \
            > "$LOG/mja_disturbing.log" 2>&1
        echo "  [4/7] MJA disturbing done! $(date)"
    fi

    # 5. P4DN (151)
    EXPECTED=151
    EXISTING=$(count_imgs "$OUT/p4dn" 2>/dev/null || echo 0)
    if [ "$EXISTING" -ge "$EXPECTED" ]; then
        echo "  SKIP: p4dn ($EXISTING/$EXPECTED)"
    else
        echo "  [5/7] P4DN..."
        CUDA_VISIBLE_DEVICES=5 $PY "$GEN" \
            --prompts "$PR/p4dn.txt" \
            --outdir "$OUT/p4dn" \
            --concept sexual --safree --svf \
            --device cuda:0 \
            > "$LOG/p4dn.log" 2>&1
        echo "  [5/7] P4DN done! $(date)"
    fi

    # 6. UnlearnDiff (142)
    EXPECTED=142
    EXISTING=$(count_imgs "$OUT/unlearndiff" 2>/dev/null || echo 0)
    if [ "$EXISTING" -ge "$EXPECTED" ]; then
        echo "  SKIP: unlearndiff ($EXISTING/$EXPECTED)"
    else
        echo "  [6/7] UnlearnDiff..."
        CUDA_VISIBLE_DEVICES=5 $PY "$GEN" \
            --prompts "$PR/unlearndiff.txt" \
            --outdir "$OUT/unlearndiff" \
            --concept sexual --safree --svf \
            --device cuda:0 \
            > "$LOG/unlearndiff.log" 2>&1
        echo "  [6/7] UnlearnDiff done! $(date)"
    fi

    # 7. COCO (250, benign — concept=none, no SAFREE)
    EXPECTED=250
    EXISTING=$(count_imgs "$OUT/coco_250" 2>/dev/null || echo 0)
    if [ "$EXISTING" -ge "$EXPECTED" ]; then
        echo "  SKIP: coco_250 ($EXISTING/$EXPECTED)"
    else
        echo "  [7/7] COCO (benign, concept=none)..."
        CUDA_VISIBLE_DEVICES=5 $PY "$GEN" \
            --prompts "$PR/coco_250.txt" \
            --outdir "$OUT/coco_250" \
            --concept none \
            --device cuda:0 \
            > "$LOG/coco_250.log" 2>&1
        echo "  [7/7] COCO done! $(date)"
    fi

    echo "[GPU 5] All sequential jobs done! $(date)"
) &
PID_SEQ=$!

echo "Waiting for all jobs..."
wait $PID_MMA $PID_SEQ 2>/dev/null
echo ""
echo "======================================================"
echo " SAFREE-Flux ALL DONE — $(date)"
echo "======================================================"
echo "Results: $OUT"
for d in "$OUT"/*/; do
    [ -d "$d" ] && echo "  $(basename $d): $(count_imgs "$d") imgs"
done
