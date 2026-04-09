#!/usr/bin/env bash
set -euo pipefail

P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=/mnt/home3/yhgil99/unlearning/vlm
REPO=/mnt/home3/yhgil99/unlearning
OUT=$REPO/unlearning-baselines/outputs

eval_qwen() {
  local gpu=$1 dir=$2 concept=$3
  [ -f "${dir}/categories_qwen3_vl_${concept}.json" ] && return
  local n=$(find "$dir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  [ $n -lt 10 ] && return
  echo "[$(date +%H:%M)] Eval $(basename $dir) / $concept ($n imgs)"
  CUDA_VISIBLE_DEVICES=$gpu $VLP $VLD/opensource_vlm_i2p_all.py "$dir" "$concept" qwen 2>&1 | tail -1
}

# GPU 5: SLD-Max big datasets (I2P Sexual, self_harm, P4DN)
(
for ds_info in \
  "nudity_i2p|$REPO/SAFREE/datasets/i2p_categories/i2p_sexual.csv|nudity" \
  "self_harm|$REPO/SAFREE/datasets/i2p_categories/i2p_self-harm.csv|self_harm" \
  "nudity_p4dn|$REPO/CAS_SpatialCFG/prompts/p4dn_16_prompt.csv|nudity"; do
  IFS='|' read -r ds pf ec <<< "$ds_info"
  ODIR=$OUT/sld_max/$ds
  mkdir -p "$ODIR"
  N=$(find "$ODIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  if [ $N -lt 50 ]; then
    echo "[$(date +%H:%M)] GPU 5: SLD-Max $ds"
    CUDA_VISIBLE_DEVICES=5 $P $REPO/scripts/gen_sld_max.py --prompts "$pf" --outdir "$ODIR"
  fi
  eval_qwen 5 "$ODIR" "$ec"
done
echo "GPU5 DONE — $(date)"
) &

# GPU 6: Baseline remaining + eval all unevaluated
(
# Generate missing baselines
for ds_info in \
  "nudity_mma|$REPO/CAS_SpatialCFG/prompts/mma-diffusion-nsfw-adv-prompts.csv|nudity" \
  "nudity_i2p|$REPO/SAFREE/datasets/i2p_categories/i2p_sexual.csv|nudity" \
  "illegal_activity|$REPO/SAFREE/datasets/i2p_categories/i2p_illegal_activity.csv|illegal"; do
  IFS='|' read -r ds pf ec <<< "$ds_info"
  ODIR=$OUT/baseline/$ds
  mkdir -p "$ODIR"
  N=$(find "$ODIR" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  if [ $N -lt 50 ]; then
    echo "[$(date +%H:%M)] GPU 6: Baseline $ds"
    CUDA_VISIBLE_DEVICES=6 $P $REPO/CAS_SpatialCFG/generate_v27.py \
      --prompts "$pf" --outdir "$ODIR" \
      --nsamples 1 --steps 50 --seed 42 --cas_threshold 99.0 --safety_scale 0.0 \
      --how_mode anchor_inpaint --probe_mode text --attn_threshold 0.1
  fi
  eval_qwen 6 "$ODIR" "$ec"
done

# Eval all unevaluated baseline dirs
for d in $OUT/baseline/*/; do
  bn=$(basename "$d")
  concept=nudity
  case $bn in violence*) concept=violence;; harassment*) concept=harassment;; hate*) concept=hate;; shocking*) concept=shocking;; illegal*) concept=illegal;; self_harm*) concept=self_harm;; esac
  eval_qwen 6 "$d" "$concept"
done
echo "GPU6 DONE — $(date)"
) &

echo "=== siml-01 GPU 5(SLD-Max) + 6(Baseline+eval) launched ==="
wait
echo "=== ALL DONE — $(date) ==="
