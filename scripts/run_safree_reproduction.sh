#!/bin/bash
# SAFREE reproduction: I2P 7 concepts + MJA 4 concepts + nudity 4 datasets
# Uses official SAFREE repo (jaehong31/SAFREE)
set -e

SAFREE_DIR="/mnt/home3/yhgil99/unlearning/unlearning-baselines/SAFREE_github"
SAVE_BASE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/safree_reproduction"
DATA_DIR="/mnt/home3/yhgil99/unlearning/SAFREE/datasets"

cd $SAFREE_DIR

PYTHON="/mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10"
echo "Using: $PYTHON"
$PYTHON -s -c "import albumentations; import clip; print('deps OK')"

# Ensure pretrained files exist
mkdir -p pretrained
[ -f pretrained/nudenet_classifier_model.onnx ] || cp ~/.NudeNet/classifier_model.onnx pretrained/nudenet_classifier_model.onnx 2>/dev/null || true
[ -f pretrained/Q16_prompts.p ] || python3 -c "
import urllib.request, os
os.makedirs('pretrained', exist_ok=True)
urllib.request.urlretrieve('https://github.com/ml-research/Q16/raw/main/data/ViT-L-14/prompts.p', 'pretrained/Q16_prompts.p')
print('Downloaded Q16 prompts')
" 2>/dev/null || true

CONFIG="./configs/sd_config.json"
MODEL="CompVis/stable-diffusion-v1-4"

run_safree() {
    local gpu=$1 data=$2 category=$3 save_name=$4
    local save_dir="${SAVE_BASE}/${save_name}"

    if [ -d "$save_dir" ] && [ "$(find $save_dir -name '*.png' 2>/dev/null | wc -l)" -gt 5 ]; then
        echo "[SKIP] $save_name already has images"
        return
    fi

    mkdir -p "$save_dir"
    echo "[GPU$gpu] SAFREE: $save_name ($category)"

    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -s generate_safree.py \
        --config $CONFIG \
        --data "$data" \
        --nudenet-path ./pretrained/nudenet_classifier_model.onnx \
        --num-samples 1 \
        --erase-id std \
        --model_id $MODEL \
        --category "$category" \
        --save-dir "$save_dir" \
        --safree -svf -lra \
        2>&1 | tail -5

    echo "[DONE GPU$gpu] $save_name ($(find $save_dir -name '*.png' 2>/dev/null | wc -l) imgs)"
}

echo "=== SAFREE Reproduction START $(date) ==="

# ── Nudity datasets (GPU 0-3) ──
run_safree 0 "${DATA_DIR}/i2p_categories/i2p_sexual.csv" "nudity" "i2p_sexual" &
run_safree 1 "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/ringabell.csv" "nudity" "rab" &
run_safree 2 "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/mma.txt" "nudity" "mma" &
run_safree 3 "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/unlearndiff.txt" "nudity" "unlearndiff" &
wait

# ── I2P concepts (GPU 0-6) ──
run_safree 0 "${DATA_DIR}/i2p_categories/i2p_violence.csv" "violence" "i2p_violence" &
run_safree 1 "${DATA_DIR}/i2p_categories/i2p_harassment.csv" "harassment" "i2p_harassment" &
run_safree 2 "${DATA_DIR}/i2p_categories/i2p_hate.csv" "hate" "i2p_hate" &
run_safree 3 "${DATA_DIR}/i2p_categories/i2p_shocking.csv" "shocking" "i2p_shocking" &
run_safree 4 "${DATA_DIR}/i2p_categories/i2p_illegal_activity.csv" "illegal activity" "i2p_illegal" &
run_safree 5 "${DATA_DIR}/i2p_categories/i2p_self-harm.csv" "self-harm" "i2p_selfharm" &
wait

# ── MJA datasets (GPU 0-3) ──
run_safree 0 "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/mja_sexual.txt" "nudity" "mja_sexual" &
run_safree 1 "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/mja_violent.txt" "violence" "mja_violent" &
run_safree 2 "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/mja_disturbing.txt" "shocking" "mja_disturbing" &
run_safree 3 "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/mja_illegal.txt" "illegal activity" "mja_illegal" &
wait

echo "=== SAFREE Reproduction COMPLETE $(date) ==="
