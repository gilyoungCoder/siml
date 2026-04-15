#!/bin/bash
# ============================================================================
# SD3 ALL Experiments — Sequential Pipeline (Generate + Evaluate)
# For large VRAM GPUs (98GB) — no CPU offload needed!
#
# Usage: nohup bash run_all_sequential.sh > logs/sd3/all_sequential.log 2>&1 &
# ============================================================================

set -e

PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
SD="/mnt/home3/yhgil99/unlearning/scripts/sd3"
OUT_BASE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/sd3"
LOG="/mnt/home3/yhgil99/unlearning/logs/sd3"
MJA="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts"
P="/mnt/home3/yhgil99/unlearning/SAFREE/datasets"

STEPS=28; CFG=7.0; RES=1024; SEED=42

mkdir -p "$LOG"

# ── Helper: skip if already done ──
count_images() {
    ls "$1"/*.png 2>/dev/null | wc -l
}

# ── BASELINE ──
run_baseline() {
    local NAME=$1 PROMPTS=$2 EXTRA=$3
    local OUTDIR="$OUT_BASE/baseline/$NAME"
    local EXPECTED=$4

    EXISTING=$(count_images "$OUTDIR" 2>/dev/null || echo 0)
    if [ "$EXISTING" -ge "$EXPECTED" ]; then
        echo "$(date) [SKIP] baseline/$NAME — already $EXISTING images"
        return
    fi

    echo "$(date) [START] baseline/$NAME ($EXISTING/$EXPECTED existing)"
    $PYTHON $SD/generate_sd3_baseline.py \
        --prompts "$PROMPTS" --outdir "$OUTDIR" \
        --steps $STEPS --cfg_scale $CFG --resolution $RES --seed $SEED \
        --no_cpu_offload $EXTRA \
        2>&1 | tail -5
    echo "$(date) [DONE] baseline/$NAME — $(count_images "$OUTDIR") images"
    echo ""
}

# ── SAFREE ──
run_safree() {
    local NAME=$1 PROMPTS=$2 CONCEPT=$3 EXTRA=$4
    local OUTDIR="$OUT_BASE/safree/$NAME"
    local EXPECTED=$5

    EXISTING=$(count_images "$OUTDIR" 2>/dev/null || echo 0)
    if [ "$EXISTING" -ge "$EXPECTED" ]; then
        echo "$(date) [SKIP] safree/$NAME — already $EXISTING images"
        return
    fi

    echo "$(date) [START] safree/$NAME ($EXISTING/$EXPECTED existing)"
    $PYTHON $SD/generate_sd3_safree.py \
        --prompts "$PROMPTS" --outdir "$OUTDIR" \
        --concept "$CONCEPT" \
        --steps $STEPS --cfg_scale $CFG --resolution $RES --seed $SEED \
        --no_cpu_offload $EXTRA \
        2>&1 | tail -5
    echo "$(date) [DONE] safree/$NAME — $(count_images "$OUTDIR") images"
    echo ""
}

# ── SAFEGEN (ours) ──
run_safegen() {
    local NAME=$1 PROMPTS=$2 EXTRA=$3
    local TARGET="$4" ANCHOR="$5"
    local OUTDIR="$OUT_BASE/safegen/$NAME"
    local EXPECTED=$6

    EXISTING=$(count_images "$OUTDIR" 2>/dev/null || echo 0)
    if [ "$EXISTING" -ge "$EXPECTED" ]; then
        echo "$(date) [SKIP] safegen/$NAME — already $EXISTING images"
        return
    fi

    echo "$(date) [START] safegen/$NAME ($EXISTING/$EXPECTED existing)"
    eval $PYTHON $SD/generate_sd3_safegen.py \
        --prompts "$PROMPTS" --outdir "$OUTDIR" \
        --target_concepts $TARGET \
        --anchor_concepts $ANCHOR \
        --cas_threshold 0.4 --how_mode anchor_inpaint --safety_scale 1.0 \
        --probe_mode text \
        --steps $STEPS --cfg_scale $CFG --resolution $RES --seed $SEED \
        --no_cpu_offload $EXTRA \
        2>&1 | tail -5
    echo "$(date) [DONE] safegen/$NAME — $(count_images "$OUTDIR") images"
    echo ""
}

# ============================================================================
echo "=============================================="
echo "$(date) SD3 ALL EXPERIMENTS START"
echo "=============================================="
echo ""

# ── Phase 1: Baselines (9 datasets) ──
echo "===== PHASE 1: BASELINES ====="
run_baseline "rab"             "$P/nudity-ring-a-bell.csv"                "" 79
run_baseline "unlearndiff"     "$P/unlearn_diff_nudity.csv"               "" 142
run_baseline "p4dn"            "$P/p4dn_16_prompt.csv"                    "" 151
run_baseline "mma"             "$P/mma-diffusion-nsfw-adv-prompts.csv"    "" 1000
run_baseline "mja_sexual"      "$MJA/mja_sexual.txt"                      "" 100
run_baseline "mja_violent"     "$MJA/mja_violent.txt"                     "" 100
run_baseline "mja_disturbing"  "$MJA/mja_disturbing.txt"                  "" 100
run_baseline "mja_illegal"     "$MJA/mja_illegal.txt"                     "" 100
run_baseline "coco250"         "$P/coco_30k_10k.csv"                      "--end 250" 250

# ── Phase 2: SAFREE (9 datasets) ──
echo "===== PHASE 2: SAFREE ====="
run_safree "rab"             "$P/nudity-ring-a-bell.csv"              "sexual"   "" 79
run_safree "unlearndiff"     "$P/unlearn_diff_nudity.csv"             "sexual"   "" 142
run_safree "p4dn"            "$P/p4dn_16_prompt.csv"                  "sexual"   "" 151
run_safree "mma"             "$P/mma-diffusion-nsfw-adv-prompts.csv"  "sexual"   "" 1000
run_safree "mja_sexual"      "$MJA/mja_sexual.txt"                    "sexual"   "" 100
run_safree "mja_violent"     "$MJA/mja_violent.txt"                   "violence" "" 100
run_safree "mja_disturbing"  "$MJA/mja_disturbing.txt"                "shocking" "" 100
run_safree "mja_illegal"     "$MJA/mja_illegal.txt"                   "illegal"  "" 100
run_safree "coco250"         "$P/coco_30k_10k.csv"                    "none"     "--end 250" 250

# ── Phase 3: SafeGen / Ours (9 datasets) ──
echo "===== PHASE 3: SAFEGEN (OURS) ====="
NT='"nudity" "nude person" "naked body"'
NA='"clothed person" "person wearing clothes"'
VT='"violence" "weapon" "blood" "gore"'
VA='"peaceful scene" "nature landscape"'
DT='"gore" "disturbing" "grotesque" "horrific"'
DA='"serene landscape" "beautiful art"'
IT='"illegal" "crime" "drugs" "theft"'
IA='"legal activity" "normal daily life"'

run_safegen "rab"             "$P/nudity-ring-a-bell.csv"              "" "$NT" "$NA" 79
run_safegen "unlearndiff"     "$P/unlearn_diff_nudity.csv"             "" "$NT" "$NA" 142
run_safegen "p4dn"            "$P/p4dn_16_prompt.csv"                  "" "$NT" "$NA" 151
run_safegen "mma"             "$P/mma-diffusion-nsfw-adv-prompts.csv"  "" "$NT" "$NA" 1000
run_safegen "mja_sexual"      "$MJA/mja_sexual.txt"                    "" "$NT" "$NA" 100
run_safegen "mja_violent"     "$MJA/mja_violent.txt"                   "" "$VT" "$VA" 100
run_safegen "mja_disturbing"  "$MJA/mja_disturbing.txt"                "" "$DT" "$DA" 100
run_safegen "mja_illegal"     "$MJA/mja_illegal.txt"                   "" "$IT" "$IA" 100
run_safegen "coco250"         "$P/coco_30k_10k.csv"                    "--end 250" "$NT" "$NA" 250

echo ""
echo "=============================================="
echo "$(date) ALL 27 EXPERIMENTS COMPLETE!"
echo "=============================================="
echo ""
echo "Next: Run evaluation with Qwen3-VL"
echo "  bash $SD/run_eval_all.sh"
