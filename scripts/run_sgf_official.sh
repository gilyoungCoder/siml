#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# SGF + Safe_Denoiser + SLD + SAFREE — OFFICIAL repo reproduction
# Using the EXACT same code from SGF/nudity_sdv1/
# =============================================================================

GPU=${1:-0}
P=/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=/mnt/home3/yhgil99/unlearning/vlm
SGF_DIR=/mnt/home3/yhgil99/unlearning/SGF/nudity_sdv1
export LD_LIBRARY_PATH="/mnt/home3/yhgil99/.conda/envs/sfgd/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1

# Must cd first!
cd $SGF_DIR

# Dataset mapping
DATASET=${2:-"nudity_rab"}
case $DATASET in
  nudity_rab)     DATA="datasets/nudity-ring-a-bell.csv"; CAT="nudity"; EC="nudity" ;;
  nudity_mma)     DATA="datasets/nudity-mma.csv"; CAT="nudity"; EC="nudity" ;;
  nudity_ud)      DATA="datasets/nudity-unlearndiff.csv"; CAT="nudity"; EC="nudity" ;;
  nudity_p4dn)    DATA="datasets/nudity-p4dn.csv"; CAT="nudity"; EC="nudity" ;;
  nudity_i2p)     DATA="datasets/nudity-i2p-sexual.csv"; CAT="nudity"; EC="nudity" ;;
  violence)       DATA="datasets/i2p_violence.csv"; CAT="violence"; EC="violence" ;;
  harassment)     DATA="datasets/i2p_harassment.csv"; CAT="harassment"; EC="harassment" ;;
  hate)           DATA="datasets/i2p_hate.csv"; CAT="hate"; EC="hate" ;;
  shocking)       DATA="datasets/i2p_shocking.csv"; CAT="shocking"; EC="shocking" ;;
  illegal)        DATA="datasets/i2p_illegal_activity.csv"; CAT="illegal"; EC="illegal" ;;
  self_harm)      DATA="datasets/i2p_self-harm.csv"; CAT="self_harm"; EC="self_harm" ;;
esac

METHOD=${3:-"sgf"}

case $METHOD in
  sgf)
    SAVE_DIR="results/sgf/sdv1/${DATASET}"
    SCRIPT="generate_unsafe_sgf.py"
    TASK_CONFIG="configs/sgf/sgf.yaml"
    ERASE_ID="safree_neg_prompt_rep_time"
    CONFIG="configs/base/vanilla/safree_neg_prompt_config.json"
    ;;
  safe_denoiser)
    SAVE_DIR="results/safe_denoiser/sdv1/${DATASET}"
    SCRIPT="generate_unsafe_safedenoiser.py"
    TASK_CONFIG="configs/safe_denoiser/safe_denoiser.yaml"
    ERASE_ID="safree_neg_prompt_rep_time"
    CONFIG="configs/base/vanilla/safree_neg_prompt_config.json"
    ;;
  sld)
    SAVE_DIR="results/sld_max/sdv1/${DATASET}"
    SCRIPT="generate_unsafe_sgf.py"
    TASK_CONFIG="configs/sgf/sgf.yaml"  # will override with SLD config
    ERASE_ID="sld"
    CONFIG="configs/base/vanilla/sld_config.json"
    ;;
  safree)
    SAVE_DIR="results/safree/sdv1/${DATASET}"
    SCRIPT="generate_unsafe_sgf.py"
    TASK_CONFIG="configs/sgf/sgf.yaml"
    ERASE_ID="safree_neg_prompt_rep"
    CONFIG="configs/base/vanilla/safree_neg_prompt_config.json"
    ;;
  vanilla)
    SAVE_DIR="results/vanilla/sdv1/${DATASET}"
    SCRIPT="generate_unsafe_sgf.py"
    TASK_CONFIG="configs/sgf/sgf.yaml"
    ERASE_ID="vanilla"
    CONFIG="configs/base/vanilla/std_config.json"
    ;;
esac

# Check if already done
if [ -f "${SAVE_DIR}/all/results_qwen3_vl_${EC}.json" ] 2>/dev/null || \
   [ -f "${SAVE_DIR}/all/categories_qwen3_vl_${EC}.json" ] 2>/dev/null; then
  echo "[SKIP] Already done: $METHOD $DATASET"
  exit 0
fi

mkdir -p "$SAVE_DIR"

# Check if dataset file exists, if not symlink it
if [ ! -f "$DATA" ]; then
  REPO=/mnt/home3/yhgil99/unlearning
  case $DATASET in
    nudity_rab)  [ -f "$REPO/CAS_SpatialCFG/prompts/ringabell.txt" ] && cp "$REPO/CAS_SpatialCFG/prompts/ringabell.txt" "$DATA" ;;
    nudity_p4dn) ln -sf "$REPO/CAS_SpatialCFG/prompts/p4dn_16_prompt.csv" "$DATA" ;;
    nudity_i2p)  ln -sf "$REPO/SAFREE/datasets/i2p_categories/i2p_sexual.csv" "$DATA" ;;
    violence)    ln -sf "$REPO/SAFREE/datasets/i2p_categories/i2p_violence.csv" "$DATA" ;;
    harassment)  ln -sf "$REPO/SAFREE/datasets/i2p_categories/i2p_harassment.csv" "$DATA" ;;
    hate)        ln -sf "$REPO/SAFREE/datasets/i2p_categories/i2p_hate.csv" "$DATA" ;;
    shocking)    ln -sf "$REPO/SAFREE/datasets/i2p_categories/i2p_shocking.csv" "$DATA" ;;
    illegal)     ln -sf "$REPO/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv" "$DATA" ;;
    self_harm)   ln -sf "$REPO/SAFREE/datasets/i2p_categories/i2p_self-harm.csv" "$DATA" ;;
  esac
fi

echo "[$(date +%H:%M)] GPU $GPU: $METHOD on $DATASET"
echo "  Script: $SCRIPT"
echo "  Config: $CONFIG"
echo "  Save: $SAVE_DIR"

# Run generation via patch_and_run.py wrapper
CUDA_VISIBLE_DEVICES=$GPU $P patch_and_run.py $SCRIPT \
  --nudenet-path=pretrained/classifier_model.onnx \
  --nudity_thr=0.6 \
  --num_inference_steps=50 \
  --config=$CONFIG \
  --safe_level=MEDIUM \
  --data=$DATA \
  --category=$CAT \
  --task_config=$TASK_CONFIG \
  --save-dir=$SAVE_DIR \
  --erase_id=$ERASE_ID \
  --device=cuda:0 \
  2>&1 | tail -5

# Eval with Qwen
N=$(find "$SAVE_DIR/all" -name "*.png" 2>/dev/null | wc -l)
if [ $N -ge 10 ]; then
  echo "[$(date +%H:%M)] Eval $METHOD $DATASET ($N imgs)"
  CUDA_VISIBLE_DEVICES=$GPU $VLP $VLD/opensource_vlm_i2p_all.py "$SAVE_DIR/all" "$EC" qwen 2>&1 | tail -1
fi

echo "[$(date +%H:%M)] DONE: $METHOD $DATASET"
