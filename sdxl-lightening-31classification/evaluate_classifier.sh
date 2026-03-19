export CUDA_VISIBLE_DEVICES=1

BENIGN_DIR=/mnt/home/yhgil99/dataset/threeclassImg/imagenet5k        # 사람 없음
PERSON_DIR=/mnt/home/yhgil99/dataset/threeclassImg/People5k     # 사람 있음(비누드)
NUDITY_DIR=/mnt/home/yhgil99/dataset/threeclassImg/Wnudity5k       # 사람 누드

python evaluateDetal.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --classifier_ckpt /mnt/home/yhgil99/unlearning/three_classificaiton_new/work_dirs/nudity_three_class_Imagenet_Filtered/checkpoint/step_21800/classifier.pth\
  --benign_dir $BENIGN_DIR \
  --person_dir $PERSON_DIR \
  --nude_dir   $NUDITY_DIR \
  --batch_size 4 \
  --seed 42
