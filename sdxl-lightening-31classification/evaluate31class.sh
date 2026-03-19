#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

python evaluate31class.py \
  --not_people_data_path  /mnt/home/yhgil99/dataset/sdxlLight/imagenet \
  --classes30_dir         /mnt/home/yhgil99/dataset/sdxlLight/30class \
  --model_path            work_dirs/31cls1024_v2/checkpoint/step_00200/classifier.pth \
  --batch_size            16