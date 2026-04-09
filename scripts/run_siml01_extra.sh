#!/usr/bin/env bash
set -euo pipefail

P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=/mnt/home3/yhgil99/unlearning/vlm
REPO=/mnt/home3/yhgil99/unlearning
I2P=$REPO/SAFREE/datasets/i2p_categories
OUTBASE=$REPO/unlearning-baselines/outputs
TRAIN=$REPO/SDErasure/train_sderasure.py
GEN=$REPO/SDErasure/generate_from_prompts.py
RECE_GEN=$REPO/scripts/gen_rece.py
RECE_UNSAFE=$REPO/unlearning-baselines/RECE/ckpts/unsafe_ep1.pt

eval_qwen() {
  local gpu=$1 dir=$2 concept=$3
  [ -f "${dir}/categories_qwen3_vl_${concept}.json" ] && return
  local n=$(find "$dir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  [ $n -lt 10 ] && return
  echo "[$(date +%H:%M)] Eval $(basename $dir) / $concept ($n imgs)"
  CUDA_VISIBLE_DEVICES=$gpu $VLP $VLD/opensource_vlm_i2p_all.py "$dir" "$concept" qwen 2>&1 | tail -1
}

# =============================================================================
# GPU 0: RECE unsafe_ep1.pt — generate ALL concepts + eval
# (unsafe checkpoint may erase general unsafe content)
# =============================================================================
(
echo "=== GPU 0: RECE unsafe on all concepts ==="
for concept in violence harassment hate shocking illegal_activity self_harm; do
  case $concept in
    violence)          PF=$I2P/i2p_violence.csv; EC=violence ;;
    harassment)        PF=$I2P/i2p_harassment.csv; EC=harassment ;;
    hate)              PF=$I2P/i2p_hate.csv; EC=hate ;;
    shocking)          PF=$I2P/i2p_shocking.csv; EC=shocking ;;
    illegal_activity)  PF=$I2P/i2p_illegal_activity.csv; EC=illegal ;;
    self_harm)         PF=$I2P/i2p_self-harm.csv; EC=self_harm ;;
  esac
  ODIR=$OUTBASE/rece_unsafe/$concept
  mkdir -p "$ODIR"
  N=$(find "$ODIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  if [ $N -lt 50 ]; then
    echo "[$(date +%H:%M)] GPU 0: RECE-unsafe $concept"
    CUDA_VISIBLE_DEVICES=0 $P $RECE_GEN --ckpt "$RECE_UNSAFE" --prompts "$PF" --outdir "$ODIR"
  fi
  eval_qwen 0 "$ODIR" "$EC"
done
echo "GPU0 RECE-unsafe DONE — $(date)"
) &

# =============================================================================
# GPU 2: SDErasure train+gen+eval — violence + harassment
# =============================================================================
(
echo "=== GPU 2: SDErasure violence + harassment ==="
for concept in violence harassment; do
  case $concept in
    violence)
      TARGET="violence gore bloodshed fighting"
      ANCHOR="peaceful scene nature landscape"
      RETAIN="a photo of a person;a beautiful landscape;a city street"
      PF=$I2P/i2p_violence.csv; EC=violence ;;
    harassment)
      TARGET="harassment bullying threatening intimidation"
      ANCHOR="friendly conversation cooperation respect"
      RETAIN="a photo of a person;a beautiful landscape;a city street"
      PF=$I2P/i2p_harassment.csv; EC=harassment ;;
  esac

  UNET_DIR=$REPO/SDErasure/outputs/sderasure_${concept}/unet
  ODIR=$OUTBASE/sderasure/$concept

  # Train if no checkpoint
  if [ ! -d "$UNET_DIR" ]; then
    echo "[$(date +%H:%M)] GPU 2: Training SDErasure $concept"
    CUDA_VISIBLE_DEVICES=2 $P $TRAIN \
      --target_concept "$TARGET" \
      --anchor_concept "$ANCHOR" \
      --retain_concepts "a photo of a person" "a beautiful landscape" "a city street" \
      --output_dir $REPO/SDErasure/outputs/sderasure_${concept} \
      --num_steps 500 --lr 1e-5
  fi

  # Generate
  if [ -d "$UNET_DIR" ]; then
    mkdir -p "$ODIR"
    N=$(find "$ODIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
    if [ $N -lt 50 ]; then
      echo "[$(date +%H:%M)] GPU 2: Generating SDErasure $concept"
      CUDA_VISIBLE_DEVICES=2 $P $GEN \
        --model_id CompVis/stable-diffusion-v1-4 \
        --unet_dir "$UNET_DIR" \
        --prompt_file "$PF" \
        --output_dir "$ODIR" \
        --seed 42 --num_inference_steps 50 --guidance_scale 7.5
    fi
    eval_qwen 2 "$ODIR" "$EC"
  else
    echo "WARNING: SDErasure $concept UNet not found after training"
  fi
done
echo "GPU2 SDErasure DONE — $(date)"
) &

# =============================================================================
# GPU 3: SDErasure train+gen+eval — hate + shocking + illegal + self_harm
# =============================================================================
(
echo "=== GPU 3: SDErasure hate + shocking + illegal + self_harm ==="
for concept in hate shocking illegal_activity self_harm; do
  case $concept in
    hate)
      TARGET="hate speech discrimination racist symbol"
      ANCHOR="diversity harmony cultural respect equality"
      PF=$I2P/i2p_hate.csv; EC=hate ;;
    shocking)
      TARGET="grotesque body horror disturbing imagery gore"
      ANCHOR="serene scene beautiful art harmonious composition"
      PF=$I2P/i2p_shocking.csv; EC=shocking ;;
    illegal_activity)
      TARGET="drugs crime illegal weapons theft"
      ANCHOR="legal activity professional work social norm"
      PF=$I2P/i2p_illegal_activity.csv; EC=illegal ;;
    self_harm)
      TARGET="self harm suicide self injury cutting"
      ANCHOR="healing meditation healthy activity mental wellness"
      PF=$I2P/i2p_self-harm.csv; EC=self_harm ;;
  esac

  UNET_DIR=$REPO/SDErasure/outputs/sderasure_${concept}/unet
  ODIR=$OUTBASE/sderasure/$concept

  if [ ! -d "$UNET_DIR" ]; then
    echo "[$(date +%H:%M)] GPU 3: Training SDErasure $concept"
    CUDA_VISIBLE_DEVICES=3 $P $TRAIN \
      --target_concept "$TARGET" \
      --anchor_concept "$ANCHOR" \
      --retain_concepts "a photo of a person" "a beautiful landscape" "a city street" \
      --output_dir $REPO/SDErasure/outputs/sderasure_${concept} \
      --num_steps 500 --lr 1e-5
  fi

  if [ -d "$UNET_DIR" ]; then
    mkdir -p "$ODIR"
    N=$(find "$ODIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
    if [ $N -lt 50 ]; then
      echo "[$(date +%H:%M)] GPU 3: Generating SDErasure $concept"
      CUDA_VISIBLE_DEVICES=3 $P $GEN \
        --model_id CompVis/stable-diffusion-v1-4 \
        --unet_dir "$UNET_DIR" \
        --prompt_file "$PF" \
        --output_dir "$ODIR" \
        --seed 42 --num_inference_steps 50 --guidance_scale 7.5
    fi
    eval_qwen 3 "$ODIR" "$EC"
  fi
done
echo "GPU3 SDErasure DONE — $(date)"
) &

echo "=== siml-01 extra: GPU 0(RECE-unsafe) 2(SDErasure v+h) 3(SDErasure h+s+i+sh) ==="
wait
echo "=== ALL DONE — $(date) ==="
