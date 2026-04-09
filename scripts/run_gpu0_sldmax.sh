#!/usr/bin/env bash
set -euo pipefail
P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=/mnt/home3/yhgil99/unlearning/vlm
REPO=/mnt/home3/yhgil99/unlearning
OUT=$REPO/unlearning-baselines/outputs/sld_max

for ds in nudity_ringabell nudity_unlearndiff nudity_mma nudity_i2p; do
  case $ds in
    nudity_ringabell)  PF=$REPO/CAS_SpatialCFG/prompts/ringabell.txt ;;
    nudity_unlearndiff) PF=$REPO/CAS_SpatialCFG/prompts/unlearn_diff_nudity.csv ;;
    nudity_mma)        PF=$REPO/CAS_SpatialCFG/prompts/mma-diffusion-nsfw-adv-prompts.csv ;;
    nudity_i2p)        PF=$REPO/SAFREE/datasets/i2p_categories/i2p_sexual.csv ;;
  esac
  ODIR=$OUT/$ds
  mkdir -p "$ODIR"
  N=$(find "$ODIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  if [ $N -lt 50 ]; then
    echo "[$(date +%H:%M)] GPU 0: SLD-Max $ds"
    CUDA_VISIBLE_DEVICES=0 $P $REPO/scripts/gen_sld_max.py --prompts "$PF" --outdir "$ODIR"
  fi
  [ ! -f "$ODIR/categories_qwen3_vl_nudity.json" ] && \
    CUDA_VISIBLE_DEVICES=0 $VLP $VLD/opensource_vlm_i2p_all.py "$ODIR" nudity qwen 2>&1 | tail -1
done
echo "GPU0 SLD-Max DONE — $(date)"
