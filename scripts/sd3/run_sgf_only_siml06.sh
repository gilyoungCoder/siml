#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# SGF SD3 only — GPU 6 on siml-06
# ============================================================================

PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
SD="/mnt/home3/yhgil99/unlearning/scripts/sd3"
OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/sd3"
MJA="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts"
P="/mnt/home3/yhgil99/unlearning/SAFREE/datasets"
SGF_CONFIG="/mnt/home3/yhgil99/unlearning/SGF/diversity_sdv3/configs/nudity_sgf/sgf_sd3.yaml"
SD3_COMMON="--steps 28 --cfg_scale 7.0 --resolution 1024 --seed 42"
GPU=6

count_images() { find "$1" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l; }

run_sg() {
    local NAME=$1 PROMPTS=$2 EXPECTED=$3 EXTRA=${4:-""}
    local OUTDIR="$OUT/sgf/$NAME"
    local EXISTING=$(count_images "$OUTDIR")
    if [ "$EXISTING" -ge "$EXPECTED" ]; then
        echo "$(date) [SKIP] sgf/$NAME — $EXISTING/$EXPECTED done"
        return
    fi
    echo "$(date) [GPU$GPU] sgf/$NAME ($EXISTING/$EXPECTED)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SD/generate_sd3_sgf.py" \
        --prompts "$PROMPTS" --outdir "$OUTDIR" --mode sgf \
        --task_config "$SGF_CONFIG" \
        $SD3_COMMON $EXTRA
    echo "$(date) [DONE] sgf/$NAME — $(count_images "$OUTDIR")"
}

echo "$(date) ===== SGF SD3 START (GPU $GPU) ====="

run_sg "rab"             "$P/nudity-ring-a-bell.csv"              79
run_sg "p4dn"            "$P/p4dn_16_prompt.csv"                  151
run_sg "unlearndiff"     "$P/unlearn_diff_nudity.csv"             142
run_sg "mja_sexual"      "$MJA/mja_sexual.txt"                    100
run_sg "mja_violent"     "$MJA/mja_violent.txt"                   100
run_sg "mja_disturbing"  "$MJA/mja_disturbing.txt"                100
run_sg "mja_illegal"     "$MJA/mja_illegal.txt"                   100
run_sg "coco250"         "$P/coco_30k_10k.csv"                    250 "--end 250"
run_sg "mma"             "$P/mma-diffusion-nsfw-adv-prompts.csv"  1000

echo "$(date) ===== SGF SD3 COMPLETE ====="
