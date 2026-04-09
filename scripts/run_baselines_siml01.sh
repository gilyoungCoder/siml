#!/usr/bin/env bash
set -euo pipefail

P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=/mnt/home3/yhgil99/unlearning/vlm
REPO=/mnt/home3/yhgil99/unlearning
OUTBASE=$REPO/unlearning-baselines/outputs
I2P=$REPO/SAFREE/datasets/i2p_categories

eval_qwen() {
  local gpu=$1 dir=$2 concept=$3
  [ -f "${dir}/categories_qwen3_vl_${concept}.json" ] && return
  local n=$(find "$dir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  [ $n -lt 10 ] && return
  echo "[$(date +%H:%M)] Eval $(basename $dir) / $concept ($n imgs)"
  CUDA_VISIBLE_DEVICES=$gpu $VLP $VLD/opensource_vlm_i2p_all.py "$dir" "$concept" qwen 2>&1 | tail -1
}

# GPU 5: RECE nudity
(
CKPT=$REPO/unlearning-baselines/RECE/ckpts/nudity_ep2.pt
for ds in nudity_ringabell nudity_unlearndiff nudity_mma; do
  case $ds in
    nudity_ringabell)  PF=$REPO/CAS_SpatialCFG/prompts/ringabell.txt ;;
    nudity_unlearndiff) PF=$REPO/CAS_SpatialCFG/prompts/unlearn_diff_nudity.csv ;;
    nudity_mma)        PF=$REPO/CAS_SpatialCFG/prompts/mma-diffusion-nsfw-adv-prompts.csv ;;
  esac
  ODIR=$OUTBASE/rece/$ds
  mkdir -p "$ODIR"
  N=$(find "$ODIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  if [ $N -lt 50 ]; then
    echo "[$(date +%H:%M)] GPU 5: RECE $ds"
    CUDA_VISIBLE_DEVICES=5 $P $REPO/scripts/gen_rece.py --ckpt "$CKPT" --prompts "$PF" --outdir "$ODIR"
  fi
  eval_qwen 5 "$ODIR" nudity
done
echo "GPU5 DONE — $(date)"
) &

# GPU 6: SDErasure nudity
(
UNET=$REPO/SDErasure/outputs/sderasure_nudity/unet
for ds in nudity_ringabell nudity_unlearndiff nudity_mma; do
  case $ds in
    nudity_ringabell)  PF=$REPO/CAS_SpatialCFG/prompts/ringabell.txt ;;
    nudity_unlearndiff) PF=$REPO/CAS_SpatialCFG/prompts/unlearn_diff_nudity.csv ;;
    nudity_mma)        PF=$REPO/CAS_SpatialCFG/prompts/mma-diffusion-nsfw-adv-prompts.csv ;;
  esac
  ODIR=$OUTBASE/sderasure/$ds
  mkdir -p "$ODIR"
  N=$(find "$ODIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  if [ $N -lt 50 ]; then
    echo "[$(date +%H:%M)] GPU 6: SDErasure $ds"
    CUDA_VISIBLE_DEVICES=6 $P $REPO/SDErasure/generate_from_prompts.py \
      --model_id CompVis/stable-diffusion-v1-4 \
      --unet_dir "$UNET" \
      --prompt_file "$PF" \
      --output_dir "$ODIR" \
      --seed 42 --num_inference_steps 50 --guidance_scale 7.5
  fi
  eval_qwen 6 "$ODIR" nudity
done
echo "GPU6 DONE — $(date)"
) &

# GPU 7: SLD-Max all non-nudity concepts
(
for concept in violence harassment hate shocking illegal_activity self_harm; do
  case $concept in
    violence)          PF=$I2P/i2p_violence.csv; EC=violence ;;
    harassment)        PF=$I2P/i2p_harassment.csv; EC=harassment ;;
    hate)              PF=$I2P/i2p_hate.csv; EC=hate ;;
    shocking)          PF=$I2P/i2p_shocking.csv; EC=shocking ;;
    illegal_activity)  PF=$I2P/i2p_illegal_activity.csv; EC=illegal ;;
    self_harm)         PF=$I2P/i2p_self-harm.csv; EC=self_harm ;;
  esac
  ODIR=$OUTBASE/sld_max/$concept
  mkdir -p "$ODIR"
  N=$(find "$ODIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  if [ $N -lt 50 ]; then
    echo "[$(date +%H:%M)] GPU 7: SLD-Max $concept"
    CUDA_VISIBLE_DEVICES=7 $P $REPO/scripts/gen_sld_max.py --prompts "$PF" --outdir "$ODIR"
  fi
  eval_qwen 7 "$ODIR" "$EC"
done
echo "GPU7 DONE — $(date)"
) &

echo "=== siml-01 baselines launched: GPU 5(RECE) 6(SDErasure) 7(SLD-Max) ==="
wait
echo "=== ALL DONE — $(date) ==="
