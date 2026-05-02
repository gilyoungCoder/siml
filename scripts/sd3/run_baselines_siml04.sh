#!/bin/bash
# ============================================================================
# Baseline + SAFREE — ALL datasets sequential on siml-04 (8x 24GB, GPU per job)
# Runs baseline on GPU0, safree on GPU1 — both sequential within their GPU
# Usage: ssh siml-04 "nohup bash /path/run_baselines_siml04.sh > log 2>&1 &"
# ============================================================================

PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
SD="/mnt/home3/yhgil99/unlearning/scripts/sd3"
OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/sd3"
LOG="/mnt/home3/yhgil99/unlearning/logs/sd3"
MJA="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts"
P="/mnt/home3/yhgil99/unlearning/SAFREE/datasets"
COMMON="--steps 28 --cfg_scale 7.0 --resolution 1024 --seed 42"

mkdir -p "$LOG"

count_images() { ls "$1"/*.png 2>/dev/null | wc -l; }

run_baseline() {
    local GPU=$1 NAME=$2 PROMPTS=$3 EXPECTED=$4 EXTRA=$5
    local OUTDIR="$OUT/baseline/$NAME"
    local EXISTING=$(count_images "$OUTDIR" 2>/dev/null || echo 0)
    if [ "$EXISTING" -ge "$EXPECTED" ]; then
        echo "$(date) [SKIP] baseline/$NAME — $EXISTING/$EXPECTED done"
        return
    fi
    echo "$(date) [GPU$GPU] baseline/$NAME ($EXISTING/$EXPECTED)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $SD/generate_sd3_baseline.py \
        --prompts "$PROMPTS" --outdir "$OUTDIR" $COMMON $EXTRA
    echo "$(date) [DONE] baseline/$NAME — $(count_images "$OUTDIR")"
}

run_safree() {
    local GPU=$1 NAME=$2 PROMPTS=$3 CONCEPT=$4 EXPECTED=$5 EXTRA=$6
    local OUTDIR="$OUT/safree/$NAME"
    local EXISTING=$(count_images "$OUTDIR" 2>/dev/null || echo 0)
    if [ "$EXISTING" -ge "$EXPECTED" ]; then
        echo "$(date) [SKIP] safree/$NAME — $EXISTING/$EXPECTED done"
        return
    fi
    echo "$(date) [GPU$GPU] safree/$NAME ($EXISTING/$EXPECTED)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $SD/generate_sd3_safree.py \
        --prompts "$PROMPTS" --outdir "$OUTDIR" --concept "$CONCEPT" $COMMON $EXTRA
    echo "$(date) [DONE] safree/$NAME — $(count_images "$OUTDIR")"
}

echo "$(date) ===== BASELINES + SAFREE on siml-04 ====="

# Run baseline (GPU0) and safree (GPU1) in parallel, each sequential internally
(
    echo "--- BASELINE QUEUE (GPU0) ---"
    run_baseline 0 "rab"             "$P/nudity-ring-a-bell.csv"              79
    run_baseline 0 "unlearndiff"     "$P/unlearn_diff_nudity.csv"             142
    run_baseline 0 "p4dn"            "$P/p4dn_16_prompt.csv"                  151
    run_baseline 0 "mja_sexual"      "$MJA/mja_sexual.txt"                    100
    run_baseline 0 "mja_violent"     "$MJA/mja_violent.txt"                   100
    run_baseline 0 "mja_disturbing"  "$MJA/mja_disturbing.txt"                100
    run_baseline 0 "mja_illegal"     "$MJA/mja_illegal.txt"                   100
    run_baseline 0 "coco250"         "$P/coco_30k_10k.csv"                    250 "--end 250"
    run_baseline 0 "mma"             "$P/mma-diffusion-nsfw-adv-prompts.csv"  1000
    echo "$(date) --- BASELINE QUEUE DONE ---"
) &
PID_BL=$!

(
    echo "--- SAFREE QUEUE (GPU1) ---"
    run_safree 1 "rab"             "$P/nudity-ring-a-bell.csv"              "sexual"   79
    run_safree 1 "unlearndiff"     "$P/unlearn_diff_nudity.csv"             "sexual"   142
    run_safree 1 "p4dn"            "$P/p4dn_16_prompt.csv"                  "sexual"   151
    run_safree 1 "mja_sexual"      "$MJA/mja_sexual.txt"                    "sexual"   100
    run_safree 1 "mja_violent"     "$MJA/mja_violent.txt"                   "violence" 100
    run_safree 1 "mja_disturbing"  "$MJA/mja_disturbing.txt"                "shocking" 100
    run_safree 1 "mja_illegal"     "$MJA/mja_illegal.txt"                   "illegal"  100
    run_safree 1 "coco250"         "$P/coco_30k_10k.csv"                    "none"     250 "--end 250"
    run_safree 1 "mma"             "$P/mma-diffusion-nsfw-adv-prompts.csv"  "sexual"   1000
    echo "$(date) --- SAFREE QUEUE DONE ---"
) &
PID_SF=$!

echo "Baseline PID: $PID_BL (GPU0)"
echo "SAFREE PID: $PID_SF (GPU1)"

wait $PID_BL $PID_SF

echo "$(date) ===== ALL BASELINES + SAFREE COMPLETE ====="
