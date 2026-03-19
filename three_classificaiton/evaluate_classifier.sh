export CUDA_VISIBLE_DEVICES=4

python evaluate_classifier.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --classifier_ckpt /mnt/home/yhgil99/unlearning/three_classificaiton/work_dirs/nudity_three_class/checkpoint/step_13600/classifier.pth \
  --benign_dir /mnt/home/yhgil99/dataset/threeclass/not_people \
  --person_dir /mnt/home/yhgil99/dataset/threeclass/people \
  --nude_dir   /mnt/home/yhgil99/dataset/threeclass/nudity \
  --batch_size 4 \
  --seed 42
