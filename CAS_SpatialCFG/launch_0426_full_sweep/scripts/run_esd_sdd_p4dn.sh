#!/bin/bash
# Generate p4dn images using ESD and SDD checkpoints, then v5 eval.
# Usage: bash run_esd_sdd_p4dn.sh <GPU>
set -uo pipefail
GPU=${1:-0}
export PYTHONNOUSERSITE=1
ROOT=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/launch_0426_full_sweep
PY_GEN=/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10
PY_EVAL=/mnt/home3/yhgil99/.conda/envs/vlm/bin/python3.10
EVAL=/mnt/home3/yhgil99/unlearning/vlm/opensource_vlm_i2p_all_v5.py
GEN_REPO=/mnt/home3/yhgil99/guided2-safe-diffusion
PROMPTS=/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/prompts/p4dn.txt
OUTBASE=$ROOT/outputs/phase_esd_sdd_p4dn
LOG=$ROOT/logs/esd_sdd_p4dn_g${GPU}_$(date +%m%d_%H%M).log
mkdir -p $OUTBASE $ROOT/logs
echo "[$(date)] start GPU=$GPU" > $LOG

# Method → checkpoint
declare -A CKPT=(
  [esd]=/mnt/home3/yhgil99/guided2-safe-diffusion/Continual2/esd_2026-01-29_17-05-34
  [sdd]=/mnt/home3/yhgil99/guided2-safe-diffusion/Continual2/sdd_2026-01-29_17-05-34
)

for METHOD in esd sdd; do
  OUT=$OUTBASE/$METHOD
  mkdir -p $OUT
  EXISTING=$(ls $OUT/*.png $OUT/*.jpg 2>/dev/null | wc -l)
  if [ "$EXISTING" -ge 150 ]; then
    echo "[$(date +%H:%M:%S)] [$METHOD] SKIP_GEN ($EXISTING/150 imgs)" | tee -a $LOG
  else
    echo "[$(date +%H:%M:%S)] [$METHOD] GEN start ckpt=${CKPT[$METHOD]}" | tee -a $LOG
    cd $GEN_REPO
    CUDA_VISIBLE_DEVICES=$GPU $PY_GEN generate.py \
      --pretrained_model_name_or_path "${CKPT[$METHOD]}" \
      --image_dir "$OUT" \
      --prompt_path "$PROMPTS" \
      --num_images_per_prompt 1 \
      --num_inference_steps 50 \
      --use_fp16 \
      --seed 42 \
      --device "cuda:0" >> $LOG 2>&1
    final=$(ls $OUT/*.png $OUT/*.jpg 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] [$METHOD] GEN done imgs=$final" | tee -a $LOG
  fi
done

# Eval (rubric=nudity for p4dn)
for METHOD in esd sdd; do
  OUT=$OUTBASE/$METHOD
  J=$OUT/categories_qwen3_vl_nudity_v5.json
  if [ -f "$J" ]; then
    echo "[$(date +%H:%M:%S)] [$METHOD] SKIP_EVAL (json exists)" | tee -a $LOG
  else
    echo "[$(date +%H:%M:%S)] [$METHOD] EVAL start" | tee -a $LOG
    cd /mnt/home3/yhgil99/unlearning/vlm
    CUDA_VISIBLE_DEVICES=$GPU $PY_EVAL $EVAL "$OUT" nudity qwen >> $LOG 2>&1
    sr=$(grep -oP "SR \(Safe\+Partial\): \d+/\d+ \(\K[0-9.]+" $OUT/results_qwen3_vl_nudity_v5.txt 2>/dev/null | head -1)
    echo "[$(date +%H:%M:%S)] [$METHOD] EVAL done SR=${sr}%" | tee -a $LOG
  fi
done

echo "[$(date)] all done" | tee -a $LOG
