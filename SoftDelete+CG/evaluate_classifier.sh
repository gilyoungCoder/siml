export CUDA_VISIBLE_DEVICES=0

BENIGN_DIR=/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k        # 사람 없음
PERSON_DIR=/mnt/home/yhgil99/dataset/threeclassImg/People5k     # 사람 있음(비누드)
NUDITY_DIR=/mnt/home/yhgil99/dataset/threeclassImg/Wnudity5k       # 사람 누드
PRETRAINED_MODEL="CompVis/stable-diffusion-v1-4"       # 사용할 VAE/scheduler

python evaluate_classifier.py \
  --pretrained_model_name_or_path $PRETRAINED_MODEL \
  --classifier_ckpt ./work_dirs/nudity_three_class_grayscale/checkpoint/step_25100/classifier.pth\
  --benign_dir $BENIGN_DIR \
  --person_dir $PERSON_DIR \
  --nude_dir   $NUDITY_DIR \
  --batch_size 16 \
  --seed 42
