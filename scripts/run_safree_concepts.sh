#!/usr/bin/env bash
set -euo pipefail

GPU=${1:-0}
CONCEPT=${2:-nudity}

export PYTHONNOUSERSITE=1
P=/mnt/home3/yhgil99/.conda/envs/safree/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=/mnt/home3/yhgil99/unlearning/vlm
REPO=/mnt/home3/yhgil99/unlearning
I2P=$REPO/SAFREE/datasets/i2p_categories
SAFREE_DIR=$REPO/SAFREE
OUTBASE=$REPO/unlearning-baselines/outputs/safree_official

case $CONCEPT in
  nudity_rab)  PF=$REPO/CAS_SpatialCFG/prompts/ringabell.txt; CAT=nudity; EC=nudity ;;
  nudity_p4dn) PF=$REPO/CAS_SpatialCFG/prompts/p4dn_16_prompt.csv; CAT=nudity; EC=nudity ;;
  nudity_ud)   PF=$REPO/CAS_SpatialCFG/prompts/unlearn_diff_nudity.csv; CAT=nudity; EC=nudity ;;
  nudity_mma)  PF=$REPO/CAS_SpatialCFG/prompts/mma-diffusion-nsfw-adv-prompts.csv; CAT=nudity; EC=nudity ;;
  nudity_i2p)  PF=$I2P/i2p_sexual.csv; CAT=nudity; EC=nudity ;;
  violence)    PF=$I2P/i2p_violence.csv; CAT=violence; EC=violence ;;
  harassment)  PF=$I2P/i2p_harassment.csv; CAT=harassment; EC=harassment ;;
  hate)        PF=$I2P/i2p_hate.csv; CAT=hate; EC=hate ;;
  shocking)    PF=$I2P/i2p_shocking.csv; CAT=shocking; EC=shocking ;;
  illegal)     PF=$I2P/i2p_illegal_activity.csv; CAT=illegal; EC=illegal ;;
  self_harm)   PF=$I2P/i2p_self-harm.csv; CAT=self_harm; EC=self_harm ;;
esac

ODIR=$OUTBASE/$CONCEPT

# Skip if already evaluated
if [ -f "$ODIR/categories_qwen3_vl_${EC}.json" ] 2>/dev/null; then
  echo "[SKIP] Already done: SAFREE $CONCEPT"
  exit 0
fi

mkdir -p "$ODIR" "$ODIR/all" "$ODIR/safe" "$ODIR/unsafe"
N=$(find "$ODIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)

if [ $N -lt 10 ]; then
  echo "[$(date +%H:%M)] GPU $GPU: SAFREE $CONCEPT (cat=$CAT)"
  cd $SAFREE_DIR
  CUDA_VISIBLE_DEVICES=$GPU $P generate_safree.py \
    --data "$PF" \
    --config configs/sd_config.json \
    --save-dir "$ODIR" \
    --model_id CompVis/stable-diffusion-v1-4 \
    --category "$CAT" \
    --nudenet-path pretrained/nudenet_classifier_model.onnx \
    --num-samples 1 \
    --device cuda:0 \
    --safree -svf -lra \
    --sf_alpha 0.01 \
    --re_attn_t="-1,4" \
    --up_t 10 2>&1 | tail -5
fi

# Find generated images (SAFREE puts them in all/ or safe/)
IMGDIR="$ODIR"
[ -d "$ODIR/all" ] && IMGDIR="$ODIR/all"

N=$(find "$IMGDIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
if [ $N -ge 10 ]; then
  echo "[$(date +%H:%M)] Eval SAFREE $CONCEPT ($N imgs)"
  CUDA_VISIBLE_DEVICES=$GPU $VLP $VLD/opensource_vlm_i2p_all.py "$IMGDIR" "$EC" qwen 2>&1 | tail -1
fi

echo "[$(date +%H:%M)] DONE: SAFREE $CONCEPT"
