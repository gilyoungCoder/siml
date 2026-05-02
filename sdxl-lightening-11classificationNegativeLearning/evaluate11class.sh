#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4

python evaluate11class.py \
  --not_people_data_path  /mnt/home/yhgil99/dataset/sdxlLight/imagenet \
  --classes10_dir         /mnt/home/yhgil99/dataset/sdxlLight/10class \
  --model_path            ./work_dirs/11cls1024/classifier_final.pth \
  --batch_size            2 \

echo "▶ 11-class evaluation started. Logs → eval11.log"
