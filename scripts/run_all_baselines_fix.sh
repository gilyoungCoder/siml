#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# FULL BASELINE RE-RUN — SDErasure v12 + all datasets + P4DN
# =============================================================================

P=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
VLP=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
VLD=/mnt/home3/yhgil99/unlearning/vlm
REPO=/mnt/home3/yhgil99/unlearning
OUTBASE=$REPO/unlearning-baselines/outputs
I2P=$REPO/SAFREE/datasets/i2p_categories
GEN=$REPO/SDErasure/generate_from_prompts.py
RECE_GEN=$REPO/scripts/gen_rece.py
SLD_GEN=$REPO/scripts/gen_sld_max.py

# SDErasure v12 (CORRECT checkpoint!)
SDE_NUDITY_UNET=$REPO/SDErasure/outputs/sderasure_nudity_v12/unet
# Concept-specific SDErasure UNets
SDE_VIOLENCE_UNET=$REPO/SDErasure/outputs/sderasure_violence/unet
SDE_HARASSMENT_UNET=$REPO/SDErasure/outputs/sderasure_harassment/unet
SDE_HATE_UNET=$REPO/SDErasure/outputs/sderasure_hate/unet
SDE_SHOCKING_UNET=$REPO/SDErasure/outputs/sderasure_shocking/unet
SDE_ILLEGAL_UNET=$REPO/SDErasure/outputs/sderasure_illegal_activity/unet

# RECE checkpoints
RECE_NUDITY=$REPO/unlearning-baselines/RECE/ckpts/nudity_ep2.pt
RECE_UNSAFE=$REPO/unlearning-baselines/RECE/ckpts/unsafe_ep1.pt

# All prompt files
declare -A PROMPTS=(
  [nudity_ringabell]="$REPO/CAS_SpatialCFG/prompts/ringabell.txt"
  [nudity_p4dn]="$REPO/CAS_SpatialCFG/prompts/p4dn_16_prompt.csv"
  [nudity_unlearndiff]="$REPO/CAS_SpatialCFG/prompts/unlearn_diff_nudity.csv"
  [nudity_mma]="$REPO/CAS_SpatialCFG/prompts/mma-diffusion-nsfw-adv-prompts.csv"
  [nudity_i2p]="$I2P/i2p_sexual.csv"
  [violence]="$I2P/i2p_violence.csv"
  [harassment]="$I2P/i2p_harassment.csv"
  [hate]="$I2P/i2p_hate.csv"
  [shocking]="$I2P/i2p_shocking.csv"
  [illegal_activity]="$I2P/i2p_illegal_activity.csv"
  [self_harm]="$I2P/i2p_self-harm.csv"
)

declare -A EVAL_CONCEPT=(
  [nudity_ringabell]=nudity [nudity_p4dn]=nudity [nudity_unlearndiff]=nudity
  [nudity_mma]=nudity [nudity_i2p]=nudity
  [violence]=violence [harassment]=harassment [hate]=hate
  [shocking]=shocking [illegal_activity]=illegal [self_harm]=self_harm
)

eval_qwen() {
  local gpu=$1 dir=$2 concept=$3
  [ -f "${dir}/categories_qwen3_vl_${concept}.json" ] && return
  local n=$(find "$dir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  [ $n -lt 10 ] && return
  echo "[$(date +%H:%M)] Eval $(basename $dir) / $concept ($n imgs)"
  CUDA_VISIBLE_DEVICES=$gpu $VLP $VLD/opensource_vlm_i2p_all.py "$dir" "$concept" qwen 2>&1 | tail -1
}

gen_sderasure() {
  local gpu=$1 unet=$2 dataset=$3 pf=$4
  local odir=$OUTBASE/sderasure_v12/$dataset
  local n=$(find "$odir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  [ $n -ge 50 ] && return
  mkdir -p "$odir"
  echo "[$(date +%H:%M)] GPU $gpu: SDErasure $dataset"
  CUDA_VISIBLE_DEVICES=$gpu $P $GEN \
    --model_id CompVis/stable-diffusion-v1-4 \
    --unet_dir "$unet" --prompt_file "$pf" --output_dir "$odir" \
    --seed 42 --num_inference_steps 50 --guidance_scale 7.5
}

gen_rece() {
  local gpu=$1 ckpt=$2 dataset=$3 pf=$4
  local odir=$OUTBASE/rece/$dataset
  local n=$(find "$odir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  [ $n -ge 50 ] && return
  mkdir -p "$odir"
  echo "[$(date +%H:%M)] GPU $gpu: RECE $dataset"
  CUDA_VISIBLE_DEVICES=$gpu $P $RECE_GEN --ckpt "$ckpt" --prompts "$pf" --outdir "$odir"
}

gen_sld() {
  local gpu=$1 dataset=$2 pf=$3
  local odir=$OUTBASE/sld_max/$dataset
  local n=$(find "$odir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  [ $n -ge 50 ] && return
  mkdir -p "$odir"
  echo "[$(date +%H:%M)] GPU $gpu: SLD-Max $dataset"
  CUDA_VISIBLE_DEVICES=$gpu $P $SLD_GEN --prompts "$pf" --outdir "$odir"
}

gen_baseline() {
  local gpu=$1 dataset=$2 pf=$3
  local odir=$OUTBASE/baseline/$dataset
  local n=$(find "$odir" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
  [ $n -ge 50 ] && return
  mkdir -p "$odir"
  echo "[$(date +%H:%M)] GPU $gpu: Baseline $dataset"
  CUDA_VISIBLE_DEVICES=$gpu $P $REPO/CAS_SpatialCFG/generate_v27.py \
    --prompts "$pf" --outdir "$odir" \
    --nsamples 1 --steps 50 --seed 42 --cas_threshold 99.0 --safety_scale 0.0 \
    --how_mode anchor_inpaint --probe_mode text --attn_threshold 0.1
}

# =============================================================================
# GPU 0: SDErasure v12 — ALL nudity datasets (ringabell, p4dn, unlearndiff, mma)
# =============================================================================
(
echo "=== GPU 0: SDErasure v12 nudity ==="
for ds in nudity_ringabell nudity_p4dn nudity_unlearndiff nudity_mma; do
  gen_sderasure 0 "$SDE_NUDITY_UNET" "$ds" "${PROMPTS[$ds]}"
  eval_qwen 0 "$OUTBASE/sderasure_v12/$ds" nudity
done
echo "GPU0 DONE — $(date)"
) &

# =============================================================================
# GPU 1: SDErasure concepts (violence, harassment, hate, shocking)
# =============================================================================
(
echo "=== GPU 1: SDErasure concepts ==="
for ds in violence harassment hate shocking; do
  case $ds in
    violence) UNET=$SDE_VIOLENCE_UNET ;;
    harassment) UNET=$SDE_HARASSMENT_UNET ;;
    hate) UNET=$SDE_HATE_UNET ;;
    shocking) UNET=$SDE_SHOCKING_UNET ;;
  esac
  gen_sderasure 1 "$UNET" "$ds" "${PROMPTS[$ds]}"
  eval_qwen 1 "$OUTBASE/sderasure_v12/$ds" "${EVAL_CONCEPT[$ds]}"
done
echo "GPU1 DONE — $(date)"
) &

# =============================================================================
# GPU 2: RECE — P4DN + missing datasets
# =============================================================================
(
echo "=== GPU 2: RECE nudity + concepts ==="
for ds in nudity_p4dn nudity_i2p; do
  gen_rece 2 "$RECE_NUDITY" "$ds" "${PROMPTS[$ds]}"
  eval_qwen 2 "$OUTBASE/rece/$ds" nudity
done
# RECE unsafe for remaining concepts
for ds in illegal_activity self_harm; do
  gen_rece 2 "$RECE_UNSAFE" "$ds" "${PROMPTS[$ds]}"
  eval_qwen 2 "$OUTBASE/rece/$ds" "${EVAL_CONCEPT[$ds]}"
done
echo "GPU2 DONE — $(date)"
) &

# =============================================================================
# GPU 3: SLD-Max — P4DN + missing
# =============================================================================
(
echo "=== GPU 3: SLD-Max P4DN + missing ==="
for ds in nudity_p4dn illegal_activity self_harm; do
  gen_sld 3 "$ds" "${PROMPTS[$ds]}"
  eval_qwen 3 "$OUTBASE/sld_max/$ds" "${EVAL_CONCEPT[$ds]}"
done
echo "GPU3 DONE — $(date)"
) &

# =============================================================================
# GPU 5: Baseline — P4DN + shocking + self_harm + missing
# =============================================================================
(
echo "=== GPU 5: Baseline missing datasets ==="
for ds in nudity_p4dn nudity_unlearndiff nudity_i2p shocking self_harm; do
  gen_baseline 5 "$ds" "${PROMPTS[$ds]}"
  eval_qwen 5 "$OUTBASE/baseline/$ds" "${EVAL_CONCEPT[$ds]}"
done
echo "GPU5 DONE — $(date)"
) &

# =============================================================================
# GPU 6: Eval remaining unevaluated dirs + RECE-unsafe eval
# =============================================================================
(
echo "=== GPU 6: Remaining evals ==="
# RECE unsafe illegal + SDErasure illegal
for d in $OUTBASE/rece_unsafe/illegal_activity $OUTBASE/sderasure/illegal_activity $OUTBASE/sld_max/illegal_activity; do
  [ -d "$d" ] && eval_qwen 6 "$d" illegal
done
# COCO FID
echo "[$(date +%H:%M)] GPU 6: COCO FID v27 hyb"
$P $REPO/scripts/run_coco_fid_v27.sh 6 2>&1 | tail -5
echo "GPU6 DONE — $(date)"
) &

echo "=============================================="
echo "  FULL BASELINE RE-RUN — $(date)"
echo "  GPU 0: SDErasure v12 nudity (4 datasets)"
echo "  GPU 1: SDErasure concepts (4 concepts)"
echo "  GPU 2: RECE P4DN + concepts"
echo "  GPU 3: SLD-Max P4DN + missing"
echo "  GPU 5: Baseline P4DN + missing"
echo "  GPU 6: Remaining evals + COCO FID"
echo "=============================================="

wait
echo "ALL DONE — $(date)"
