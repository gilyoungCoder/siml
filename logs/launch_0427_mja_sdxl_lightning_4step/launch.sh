#!/usr/bin/env bash
set -euo pipefail
BASE=/mnt/home3/yhgil99/unlearning
G=/mnt/home3/yhgil99/guided2-safe-diffusion
OUT=$BASE/CAS_SpatialCFG/outputs/launch_0427_mja_sdxl_lightning_4step
LOGD=$BASE/logs/launch_0427_mja_sdxl_lightning_4step
PY=python3
MODEL=ByteDance/SDXL-Lightning
export MAX_INFER_BATCH_SIZE=4
run_one() {
  local gpu=$1 concept=$2 prompt_dir=$3
  local odir=$OUT/$concept
  mkdir -p "$odir"
  echo "[$(date -Is)] launch concept=$concept gpu=$gpu out=$odir" | tee -a "$LOGD/launch.log"
  (cd "$G" && CUDA_VISIBLE_DEVICES=$gpu nohup $PY generateLight_bakup.py \
    --pretrained_model_name_or_path "$MODEL" \
    --pipeline_type lightning \
    --image_dir "$odir" \
    --prompt_path "$BASE/CAS_SpatialCFG/outputs/baselines_v2/$prompt_dir/prompts.txt" \
    --device cuda:0 \
    --use_fp16 \
    --num_images_per_prompt 5 \
    --num_inference_steps 4 \
    > "$LOGD/${concept}_gpu${gpu}.log" 2>&1 & echo $! > "$LOGD/${concept}_gpu${gpu}.pid")
}
run_one 1 mja_violent mja_violent
run_one 2 mja_illegal mja_illegal
run_one 3 mja_disturbing mja_disturbing
