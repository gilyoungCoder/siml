#!/bin/bash
# ============================================================================
# SafeGen (OURS) — ALL datasets sequential on siml-09 (98GB GPU, no offload!)
# Usage: ssh siml-09 "nohup bash /path/run_safegen_siml09.sh > log 2>&1 &"
# ============================================================================

PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
SD="/mnt/home3/yhgil99/unlearning/scripts/sd3"
OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/sd3/safegen"
LOG="/mnt/home3/yhgil99/unlearning/logs/sd3"
MJA="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts"
P="/mnt/home3/yhgil99/unlearning/SAFREE/datasets"
COMMON="--cas_threshold 0.4 --how_mode anchor_inpaint --safety_scale 1.0 --probe_mode text --steps 28 --cfg_scale 7.0 --resolution 1024 --seed 42 --no_cpu_offload"

mkdir -p "$LOG"

count_images() { ls "$1"/*.png 2>/dev/null | wc -l; }

run() {
    local NAME=$1 PROMPTS=$2 TARGET="$3" ANCHOR="$4" EXPECTED=$5 EXTRA="$6"
    local OUTDIR="$OUT/$NAME"
    local EXISTING=$(count_images "$OUTDIR" 2>/dev/null || echo 0)
    if [ "$EXISTING" -ge "$EXPECTED" ]; then
        echo "$(date) [SKIP] $NAME — $EXISTING/$EXPECTED done"
        return
    fi
    echo "$(date) [START] $NAME ($EXISTING/$EXPECTED)"
    eval $PYTHON $SD/generate_sd3_safegen.py \
        --prompts "$PROMPTS" --outdir "$OUTDIR" \
        --target_concepts $TARGET --anchor_concepts $ANCHOR \
        $COMMON $EXTRA
    echo "$(date) [DONE] $NAME — $(count_images "$OUTDIR") images"
    echo ""
}

echo "$(date) ===== SafeGen SD3 ALL DATASETS (siml-09) ====="

NT='"nudity" "nude person" "naked body"'
NA='"clothed person" "person wearing clothes"'
VT='"violence" "weapon" "blood" "gore"'
VA='"peaceful scene" "nature landscape"'
DT='"gore" "disturbing" "grotesque" "horrific"'
DA='"serene landscape" "beautiful art"'
IT='"illegal" "crime" "drugs" "theft"'
IA='"legal activity" "normal daily life"'

run "rab"             "$P/nudity-ring-a-bell.csv"              "$NT" "$NA" 79
run "unlearndiff"     "$P/unlearn_diff_nudity.csv"             "$NT" "$NA" 142
run "p4dn"            "$P/p4dn_16_prompt.csv"                  "$NT" "$NA" 151
run "mma"             "$P/mma-diffusion-nsfw-adv-prompts.csv"  "$NT" "$NA" 1000
run "mja_sexual"      "$MJA/mja_sexual.txt"                    "$NT" "$NA" 100
run "mja_violent"     "$MJA/mja_violent.txt"                   "$VT" "$VA" 100
run "mja_disturbing"  "$MJA/mja_disturbing.txt"                "$DT" "$DA" 100
run "mja_illegal"     "$MJA/mja_illegal.txt"                   "$IT" "$IA" 100
run "coco250"         "$P/coco_30k_10k.csv"                    "$NT" "$NA" 250 "--end 250"

echo "$(date) ===== ALL SAFEGEN COMPLETE ====="
