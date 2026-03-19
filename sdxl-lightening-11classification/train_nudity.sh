#!/bin/bash
# output directory 이름과 기타 변수들 설정
export CUDA_VISIBLE_DEVICES=0
export TORCH_XLA_DISABLE_FLASH_ATTENTION=1

output_dir=monet_classifier_chekpoint
benign_data_path=/mnt/home/yhgil99/dataset/benign_data
image_resolution=512
pretrained_model_mode=stablediffusion   # geodiffusion 또는 stablediffusion
nudity_data_path=/mnt/home/yhgil99/dataset/target_monet    # nudity 이미지가 저장된 폴더 (target dataset)
seen_dataset_ratio=0.12
learning_rate=1.0e-4
noise_std=1.0

# 데이터셋 구성에 따라 배치 사이즈와 체크포인트 저장 주기를 설정 (여기서는 new_coco_stuff 사용)
train_batch_size=4
save_ckpt_freq=100

# pretrained_model 설정
if [ $pretrained_model_mode == "stablediffusion" ]; then
    pretrained_model=runwayml/stable-diffusion-v1-5
# elif [ $pretrained_model_mode == "geodiffusion" ] && [ $image_resolution == 512 ]; then
#     pretrained_model=KaiChen1998/geodiffusion-coco-stuff-512x512
# elif [ $pretrained_model_mode == "geodiffusion" ] && [ $image_resolution == 256 ]; then
#     pretrained_model=KaiChen1998/geodiffusion-coco-stuff-256x256
fi

# 실행할 training script 선택 (GPU와 TPU에 따라 스크립트를 다르게 호출)
command_line="python train_nudity_classifier.py "

# 필요한 인자들을 command_line에 추가합니다.
command_line+="--benign_data_path $benign_data_path "
command_line+="--nudity_data_path $nudity_data_path "
command_line+="--output_dir work_dirs/$output_dir "
command_line+="--report_to wandb "
command_line+="--use_wandb "
# command_line+="--mixed_precision fp16 "
command_line+="--train_batch_size $train_batch_size "
command_line+="--pretrained_model_name_or_path $pretrained_model "
command_line+="--save_ckpt_freq $save_ckpt_freq "
command_line+="--learning_rate $learning_rate "
command_line+="--noise_std $noise_std "

echo "Executing: $command_line"

nohup $command_line > train_classifier_monet_chkpoint.log 2>&1 &

exit 0
