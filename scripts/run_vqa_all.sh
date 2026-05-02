#!/bin/bash
# VQAScore evaluation: baseline vs ours across all datasets
set -e
cd /mnt/home3/yhgil99/unlearning

eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
conda activate sdd_copy

BL="CAS_SpatialCFG/outputs/baselines_v2"
OURS="CAS_SpatialCFG/outputs/v2_experiments"

eval_vqa() {
    local dir=$1 prompts=$2 gpu=$3
    [ -f "$dir/results_vqascore.txt" ] && echo "[SKIP VQA] $dir" && return 0
    [ "$(ls $dir/*.png 2>/dev/null | wc -l)" -eq 0 ] && return 0
    echo "[VQA GPU$gpu] $(basename $(dirname $dir))/$(basename $dir)"
    CUDA_VISIBLE_DEVICES=$gpu python3 vlm/eval_vqascore.py \
        "$dir" --prompts "$prompts" 2>&1 | tail -2
}

echo "=== VQA START $(date) ==="

# GPU 0: Baselines nudity
(
eval_vqa "$BL/rab" "CAS_SpatialCFG/prompts/ringabell.txt" 0
eval_vqa "$BL/mma" "CAS_SpatialCFG/prompts/mma.txt" 0
eval_vqa "$BL/p4dn" "CAS_SpatialCFG/prompts/p4dn.txt" 0
eval_vqa "$BL/unlearndiff" "CAS_SpatialCFG/prompts/unlearndiff.txt" 0
echo "[GPU0 DONE]"
) &

# GPU 1: Baselines MJA
(
eval_vqa "$BL/mja_sexual" "CAS_SpatialCFG/prompts/mja_sexual.txt" 1
eval_vqa "$BL/mja_violent" "CAS_SpatialCFG/prompts/mja_violent.txt" 1
eval_vqa "$BL/mja_disturbing" "CAS_SpatialCFG/prompts/mja_disturbing.txt" 1
eval_vqa "$BL/mja_illegal" "CAS_SpatialCFG/prompts/mja_illegal.txt" 1
echo "[GPU1 DONE]"
) &

# GPU 2: Ours RAB best configs
(
for d in rab_text_anchor_inpaint_fam_cas0.4_ss1.2 rab_both_anchor_inpaint_single_cas0.4_ss1.2 rab_image_anchor_inpaint_fam_cas0.4_ss1.2; do
    eval_vqa "$OURS/sexual/$d" "CAS_SpatialCFG/prompts/ringabell.txt" 2
done
echo "[GPU2 DONE]"
) &

# GPU 3: Ours MMA + UDiff + P4DN best
(
eval_vqa "$OURS/sexual/mma_both_anchor_inpaint_fam_cas0.6_ss1.2" "CAS_SpatialCFG/prompts/mma.txt" 3
eval_vqa "$OURS/sexual/udiff_both_anchor_inpaint_single_cas0.6_ss1.2" "CAS_SpatialCFG/prompts/unlearndiff.txt" 3
eval_vqa "$OURS/sexual/p4dn_both_anchor_inpaint_single_cas0.6_ss1.2" "CAS_SpatialCFG/prompts/p4dn.txt" 3
echo "[GPU3 DONE]"
) &

# GPU 4: Ours MJA sexual + violent best
(
eval_vqa "$OURS/sexual/mja_both_anchor_inpaint_fam_cas0.6_ss1.2" "CAS_SpatialCFG/prompts/mja_sexual.txt" 4
eval_vqa "$OURS/violent/mja_both_anchor_inpaint_single_cas0.4_ss1.5" "CAS_SpatialCFG/prompts/mja_violent.txt" 4
echo "[GPU4 DONE]"
) &

# GPU 5: Ours MJA disturbing + illegal best
(
eval_vqa "$OURS/disturbing/mja_both_anchor_inpaint_single_cas0.6_ss1.0" "CAS_SpatialCFG/prompts/mja_disturbing.txt" 5
eval_vqa "$OURS/illegal/mja_image_anchor_inpaint_fam_cas0.5_ss1.0" "CAS_SpatialCFG/prompts/mja_illegal.txt" 5
echo "[GPU5 DONE]"
) &

# GPU 6: Ours I2P concepts best
(
eval_vqa "$OURS/violent/i2p_both_anchor_inpaint_single_cas0.4_ss1.5" "SAFREE/datasets/i2p_categories/i2p_violence.csv" 6
eval_vqa "$OURS/harassment/i2p_both_anchor_inpaint_fam_cas0.4_ss1.2" "SAFREE/datasets/i2p_categories/i2p_harassment.csv" 6
eval_vqa "$OURS/hate/i2p_both_anchor_inpaint_fam_cas0.4_ss1.0" "SAFREE/datasets/i2p_categories/i2p_hate.csv" 6
echo "[GPU6 DONE]"
) &

# GPU 7: Ours I2P concepts best (cont)
(
eval_vqa "$OURS/disturbing/i2p_both_anchor_inpaint_single_cas0.5_ss1.0" "SAFREE/datasets/i2p_categories/i2p_shocking.csv" 7
eval_vqa "$OURS/selfharm/i2p_both_anchor_inpaint_single_cas0.4_ss1.0" "SAFREE/datasets/i2p_categories/i2p_self-harm.csv" 7
eval_vqa "$OURS/illegal/i2p_both_anchor_inpaint_fam_cas0.5_ss1.0" "SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv" 7
eval_vqa "$OURS/sexual/i2p_both_anchor_inpaint_single_cas0.6_ss1.2" "SAFREE/datasets/i2p_categories/i2p_sexual.csv" 7
echo "[GPU7 DONE]"
) &

wait
echo "=== VQA COMPLETE $(date) ==="

# Summary
echo ""
echo "=== VQA RESULTS ==="
for d in $BL/*/; do
    [ -f "$d/results_vqascore.txt" ] && echo "BL $(basename $d): $(grep 'Mean VQA' $d/results_vqascore.txt 2>/dev/null | head -1)"
done
for concept_dir in $OURS/*/; do
    concept=$(basename $concept_dir)
    for d in $concept_dir/*/; do
        [ -f "$d/results_vqascore.txt" ] && echo "$concept $(basename $d): $(grep 'Mean VQA' $d/results_vqascore.txt 2>/dev/null | head -1)"
    done
done
