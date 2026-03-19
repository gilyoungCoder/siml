#!/bin/bash
# Safe_Denoiser - Ring-A-Bell nudity erasing experiment

set -e
cd /mnt/home/yhgil99/unlearning/SGF/nudity_sdv1

SAVE_DIR="results/safe_denoiser/sdv1/nudity_rab"
PYTHON="PYTHONNOUSERSITE=1 /mnt/home/yhgil99/.conda/envs/sfgd/bin/python"

eval $PYTHON generate_unsafe_safedenoiser.py \
  --nudenet-path=pretrained/classifier_model.onnx \
  --nudity_thr=0.6 \
  --num_inference_steps=50 \
  --config=configs/base/vanilla/safree_neg_prompt_config.json \
  --safe_level=MEDIUM \
  --data=datasets/nudity-ring-a-bell.csv \
  --category=nudity \
  --task_config=configs/safe_denoiser/safe_denoiser.yaml \
  --save-dir=${SAVE_DIR} \
  --erase_id=safree_neg_prompt_rep_time \
  --device=cuda:1

echo "Generation done. Running VLM evaluation..."

PYTHONNOUSERSITE=1 /mnt/home/yhgil99/.conda/envs/vlm/bin/python \
  /mnt/home/yhgil99/unlearning/vlm/opensource_vlm_i2p_all.py \
  ${SAVE_DIR}/all \
  nudity \
  qwen

echo "All done! Results at ${SAVE_DIR}"
