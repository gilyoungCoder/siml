export CUDA_VISIBLE_DEVICES=6

python evaluate_classifier.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --classifier_ckpt ./work_dirs/sdxl1024/classifier_final.pth \
  --not_people_dir    /mnt/home/yhgil99/dataset/sdxlLight/imagenet \
  --fully_clothed_dir /mnt/home/yhgil99/dataset/sdxlLight/3class/fullyclothed \
  --partial_nude_dir  /mnt/home/yhgil99/dataset/sdxlLight/3class/partial_nude \
  --full_nude_dir     /mnt/home/yhgil99/dataset/sdxlLight/3class/Wnudity \
  --batch_size 8 \
  --seed 42
