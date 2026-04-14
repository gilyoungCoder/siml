#!/bin/bash
set -e
SAFREE_CUSTOM="/mnt/home3/yhgil99/unlearning/SAFREE"
SAVE_BASE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/safree_reproduction"
I2P_DIR="/mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories"
MJA_DIR="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts"
PYTHON="/mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10"
cd $SAFREE_CUSTOM

run_concept() {
    local gpu=$1 prompt_file=$2 concept=$3 save_name=$4
    local outdir="$SAVE_BASE/$save_name"
    mkdir -p "$outdir"
    echo "[GPU$gpu] $save_name ($concept)"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -s gen_safree_i2p_concepts.py \
        --prompt_file "$prompt_file" --concepts "$concept" \
        --outdir "$outdir" --no_concept_subdir \
        --model_id CompVis/stable-diffusion-v1-4 \
        --safree --svf --lra --device cuda:0 2>&1 | tail -3
    echo "[DONE GPU$gpu] $save_name ($(find $outdir -name '*.png' 2>/dev/null | wc -l) imgs)"
}

echo "=== SAFREE Concepts START $(date) ==="
run_concept 2 "$I2P_DIR/i2p_violence.csv" violence i2p_violence &
run_concept 3 "$I2P_DIR/i2p_harassment.csv" harassment i2p_harassment &
(run_concept 4 "$I2P_DIR/i2p_hate.csv" hate i2p_hate && run_concept 4 "$I2P_DIR/i2p_self-harm.csv" self-harm i2p_selfharm) &
run_concept 5 "$I2P_DIR/i2p_shocking.csv" shocking i2p_shocking &
run_concept 6 "$I2P_DIR/i2p_illegal_activity.csv" "illegal activity" i2p_illegal &
(run_concept 7 "$MJA_DIR/mja_sexual.txt" nudity mja_sexual && run_concept 7 "$MJA_DIR/mja_violent.txt" violence mja_violent && run_concept 7 "$MJA_DIR/mja_disturbing.txt" shocking mja_disturbing && run_concept 7 "$MJA_DIR/mja_illegal.txt" "illegal activity" mja_illegal) &
wait
echo "=== SAFREE Concepts COMPLETE $(date) ==="
