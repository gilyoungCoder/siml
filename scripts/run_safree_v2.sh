#!/bin/bash
# SAFREE reproduction v2: nudity via official, others via gen_safree_i2p_concepts.py
set -e

SAFREE_OFFICIAL="/mnt/home3/yhgil99/unlearning/unlearning-baselines/SAFREE_github"
SAFREE_CUSTOM="/mnt/home3/yhgil99/unlearning/SAFREE"
SAVE_BASE="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/safree_reproduction"
I2P_DIR="/mnt/home3/yhgil99/unlearning/SAFREE/datasets/i2p_categories"
MJA_DIR="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts"
PYTHON="/mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10"

mkdir -p $SAVE_BASE

# Ensure pretrained
mkdir -p $SAFREE_OFFICIAL/pretrained
[ -f $SAFREE_OFFICIAL/pretrained/nudenet_classifier_model.onnx ] || cp ~/.NudeNet/classifier_model.onnx $SAFREE_OFFICIAL/pretrained/ 2>/dev/null || true

echo "=== SAFREE v2 START $(date) ==="

# Pre-create output subdirs (SAFREE saves to safe/unsafe/all)
for d in i2p_sexual rab mma unlearndiff i2p_violence i2p_harassment i2p_hate i2p_shocking i2p_illegal i2p_selfharm mja_sexual mja_violent mja_disturbing mja_illegal; do
    mkdir -p "$SAVE_BASE/$d/safe" "$SAVE_BASE/$d/unsafe" "$SAVE_BASE/$d/all"
done

# ── GPU 0: Nudity I2P (official, 931 prompts) ──
(
cd $SAFREE_OFFICIAL
CUDA_VISIBLE_DEVICES=0 $PYTHON -s generate_safree.py \
    --config ./configs/sd_config.json \
    --data "$I2P_DIR/i2p_sexual.csv" \
    --nudenet-path ./pretrained/nudenet_classifier_model.onnx \
    --num-samples 1 --erase-id std \
    --model_id CompVis/stable-diffusion-v1-4 \
    --category nudity \
    --save-dir "$SAVE_BASE/i2p_sexual" \
    --safree -svf -lra 2>&1 | tail -3
echo "[DONE GPU0] i2p_sexual"
) &

# ── GPU 1: Nudity RAB (official) ──
(
cd $SAFREE_OFFICIAL
CUDA_VISIBLE_DEVICES=1 $PYTHON -s generate_safree.py \
    --config ./configs/sd_config.json \
    --data "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/ringabell.csv" \
    --nudenet-path ./pretrained/nudenet_classifier_model.onnx \
    --num-samples 1 --erase-id std \
    --model_id CompVis/stable-diffusion-v1-4 \
    --category nudity \
    --save-dir "$SAVE_BASE/rab" \
    --safree -svf -lra 2>&1 | tail -3
echo "[DONE GPU1] rab"
) &

# ── GPU 2: Violence I2P (custom, 756 prompts) ──
(
cd $SAFREE_CUSTOM
CUDA_VISIBLE_DEVICES=2 $PYTHON -s gen_safree_i2p_concepts.py \
    --prompts "$I2P_DIR/i2p_violence.csv" \
    --concepts violence \
    --outdir "$SAVE_BASE/i2p_violence" \
    --no_concept_subdir \
    --model_id CompVis/stable-diffusion-v1-4 \
    --safree --svf --lra 2>&1 | tail -3
echo "[DONE GPU2] i2p_violence"
) &

# ── GPU 3: Harassment I2P (custom, 824 prompts) ──
(
cd $SAFREE_CUSTOM
CUDA_VISIBLE_DEVICES=3 $PYTHON -s gen_safree_i2p_concepts.py \
    --prompts "$I2P_DIR/i2p_harassment.csv" \
    --concepts harassment \
    --outdir "$SAVE_BASE/i2p_harassment" \
    --no_concept_subdir \
    --model_id CompVis/stable-diffusion-v1-4 \
    --safree --svf --lra 2>&1 | tail -3
echo "[DONE GPU3] i2p_harassment"
) &

# ── GPU 4: Hate I2P (custom, 231 prompts) ──
(
cd $SAFREE_CUSTOM
CUDA_VISIBLE_DEVICES=4 $PYTHON -s gen_safree_i2p_concepts.py \
    --prompts "$I2P_DIR/i2p_hate.csv" \
    --concepts hate \
    --outdir "$SAVE_BASE/i2p_hate" \
    --no_concept_subdir \
    --model_id CompVis/stable-diffusion-v1-4 \
    --safree --svf --lra 2>&1 | tail -3
echo "[DONE GPU4] i2p_hate"
) &

# ── GPU 5: Shocking I2P (custom, 856 prompts) ──
(
cd $SAFREE_CUSTOM
CUDA_VISIBLE_DEVICES=5 $PYTHON -s gen_safree_i2p_concepts.py \
    --prompts "$I2P_DIR/i2p_shocking.csv" \
    --concepts shocking \
    --outdir "$SAVE_BASE/i2p_shocking" \
    --no_concept_subdir \
    --model_id CompVis/stable-diffusion-v1-4 \
    --safree --svf --lra 2>&1 | tail -3
echo "[DONE GPU5] i2p_shocking"
) &

# ── GPU 6: Illegal + Self-harm I2P (custom, sequential) ──
(
cd $SAFREE_CUSTOM
CUDA_VISIBLE_DEVICES=6 $PYTHON -s gen_safree_i2p_concepts.py \
    --prompts "$I2P_DIR/i2p_illegal_activity.csv" \
    --concepts "illegal activity" \
    --outdir "$SAVE_BASE/i2p_illegal" \
    --no_concept_subdir \
    --model_id CompVis/stable-diffusion-v1-4 \
    --safree --svf --lra 2>&1 | tail -3
echo "[DONE GPU6] i2p_illegal"

CUDA_VISIBLE_DEVICES=6 $PYTHON -s gen_safree_i2p_concepts.py \
    --prompts "$I2P_DIR/i2p_self-harm.csv" \
    --concepts self-harm \
    --outdir "$SAVE_BASE/i2p_selfharm" \
    --no_concept_subdir \
    --model_id CompVis/stable-diffusion-v1-4 \
    --safree --svf --lra 2>&1 | tail -3
echo "[DONE GPU6] i2p_selfharm"
) &

# ── GPU 7: MJA 4 datasets (custom, sequential, 100 each) ──
(
cd $SAFREE_CUSTOM
for ds_concept in "mja_sexual.txt nudity" "mja_violent.txt violence" "mja_disturbing.txt shocking" "mja_illegal.txt illegal activity"; do
    set -- $ds_concept
    ds=$1; shift; concept="$*"
    name=$(echo $ds | sed 's/.txt//')
    CUDA_VISIBLE_DEVICES=7 $PYTHON -s gen_safree_i2p_concepts.py \
        --prompts "$MJA_DIR/$ds" \
        --concepts "$concept" \
        --outdir "$SAVE_BASE/$name" \
        --no_concept_subdir \
        --model_id CompVis/stable-diffusion-v1-4 \
        --safree --svf --lra 2>&1 | tail -3
    echo "[DONE GPU7] $name"
done
) &

wait
echo "=== SAFREE v2 COMPLETE $(date) ==="
