#!/bin/bash
# ============================================================================
# Sequential queue: SafeGen remaining experiments
# Runs on a single GPU — launch one per GPU for parallelism
# Usage: CUDA_VISIBLE_DEVICES=4 bash queue_safegen_remaining.sh
# ============================================================================

PYTHON="/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10"
SD="/mnt/home3/yhgil99/unlearning/scripts/sd3"
OUT="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/sd3/safegen"
LOG="/mnt/home3/yhgil99/unlearning/logs/sd3"
MJA="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts"
P="/mnt/home3/yhgil99/unlearning/SAFREE/datasets"

COMMON="--cas_threshold 0.4 --how_mode anchor_inpaint --safety_scale 1.0 --probe_mode text --steps 28 --cfg_scale 7.0 --resolution 1024 --seed 42"

run_safegen() {
    local NAME=$1 PROMPTS=$2 TARGET="$3" ANCHOR="$4"
    echo "$(date) [START] safegen/$NAME"
    $PYTHON $SD/generate_sd3_safegen.py \
        --prompts "$PROMPTS" --outdir "$OUT/$NAME" \
        --target_concepts $TARGET \
        --anchor_concepts $ANCHOR \
        $COMMON 2>&1 | tee "$LOG/safegen_${NAME}_queue.log"
    echo "$(date) [DONE] safegen/$NAME"
    echo ""
}

# ── Queue: SafeGen missing experiments ──
run_safegen "unlearndiff" "$P/unlearn_diff_nudity.csv" \
    '"nudity" "nude person" "naked body"' '"clothed person" "person wearing clothes"'

run_safegen "p4dn" "$P/p4dn_16_prompt.csv" \
    '"nudity" "nude person" "naked body"' '"clothed person" "person wearing clothes"'

run_safegen "mja_illegal" "$MJA/mja_illegal.txt" \
    '"illegal" "crime" "drugs" "theft"' '"legal activity" "normal daily life"'

run_safegen "mma" "$P/mma-diffusion-nsfw-adv-prompts.csv" \
    '"nudity" "nude person" "naked body"' '"clothed person" "person wearing clothes"'

run_safegen "coco250" "$P/coco_30k_10k.csv --end 250" \
    '"nudity" "nude person" "naked body"' '"clothed person" "person wearing clothes"'

echo "$(date) === ALL SAFEGEN QUEUE COMPLETE ==="
