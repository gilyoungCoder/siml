#!/bin/bash

# For the missing person dataset
# output_dir=missing_person_512_512_240925_disc_seen_unseen_vaal_after_training_cycle_4
# dataset_config_name=configs/data/missing_person_512x512.py
# dataset_config_type=missing_person # missing_person, new_coco_stuff
# image_resolution=512
# pretrained_model_mode=geodiffusion #geodiffusion, stablediffusion
# blip_finetue=true
# lora_mode=false
# time_dependent=false
# adversarial_scale=1

# For the new coco stuff dataset
# output_dir=250115_disc_vaal_mscoco_balanced_time_dependent_cycle_2_epoch_100_with_acq
# output_dir=250117_disc_vaal_mscoco_balanced_time_dependent_cycle_1_epoch_100_seen_0.04_coreset_selected
output_dir=250122_disc_vaal_missing_person_balanceed_time_depndent_cycle_1_epoch_100_seen_0.1_coreset_selected
dataset_config_type=missing_person # missing_person, new_coco_stuff
image_resolution=512
pretrained_model_mode=geodiffusion #geodiffusion, stablediffusion
blip_finetue=true
lora_mode=false
time_dependent=true
adversarial_scale=1
seen_dataset_ratio=0.2  # 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4
oversample_from_unseen=false
for_guidance=false
# training_epoch=100
training_epoch=50

model_yaml=/home/djfelrl11/geodiffusion/configs/models/time_dependent_vaal.yaml
# model_yaml=/home/djfelrl11/geodiffusion/configs/models/time_dependent_vaal_resolution_ablation/time_dependent_vaal_256.yaml

manual_seen_file_name=""
manual_unseen_file_name=""

learning_rate=1.0e-4
cycle=0 # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# resume=false
# resume_ckpt=/home/djfelrl11/geodiffusion/work_dirs/241015_disc_vaal_mscoco_balanced_time_independent_cycle_3/checkpoint/iter_60001
resume=false
resume_ckpt=/home/djfelrl11/geodiffusion/work_dirs/250121_disc_vaal_mscoco_balanced_time_dependent_cycle_1_epoch_100_seen_0.04_coreset_selected/checkpoint/iter_7350



if [ $dataset_config_type == "missing_person" ] && [ $image_resolution == 256 ]; then
	dataset_config_name=configs/data/missing_person_256x256.py
	train_batch_size=16
	save_ckpt_freq=500
elif [ $dataset_config_type == "missing_person" ] && [ $image_resolution == 512 ]; then
	dataset_config_name=configs/data/missing_person_512x512.py
	train_batch_size=16
	save_ckpt_freq=250
elif [ $dataset_config_type == "new_coco_stuff" ] && [ $image_resolution == 256 ]; then
	dataset_config_name=configs/data/new_coco_stuff_256x256.py
	train_batch_size=16
	save_ckpt_freq=5000
elif [ $dataset_config_type == "new_coco_stuff" ] && [ $image_resolution == 512 ]; then
	dataset_config_name=configs/data/new_coco_stuff_512x512.py
	train_batch_size=16
	save_ckpt_freq=5000
fi

if [ $pretrained_model_mode == "stablediffusion" ]; then
	pretrained_model=runwayml/stable-diffusion-v1-5
elif [ $pretrained_model_mode == "geodiffusion" ] && [ $image_resolution == 512 ]; then
	pretrained_model=KaiChen1998/geodiffusion-coco-stuff-512x512
elif [ $pretrained_model_mode == "geodiffusion" ] && [ $image_resolution == 256 ]; then
	pretrained_model=KaiChen1998/geodiffusion-coco-stuff-256x256
fi

command_line="bash "

if [[ $PJRT_DEVICE == "TPU" ]]; then
	echo "The device is set as TPU: get into the TPU mode."
	# command_line+="tools/tpu/dist_train_discriminator.sh "
	command_line+="tools/tpu/dist_train_discriminator_vaal.sh "
elif which nvidia-smi > /dev/null 2>&1; then
	echo "The device is set as GPU: get into the GPU mode."
	# command_line+="tools/gpu/dist_train.sh "
	command_line+="tools/gpu/dist_train_discriminator_vaal.sh "
else
	echo "unknown device"
	exit 1
fi

if [[ $for_guidance == true ]]; then
	output_dir=${output_dir}_for_guidance
fi

command_line+="--dataset_config_name $dataset_config_name "
command_line+="--train_text_encoder_params added_embedding "
command_line+="--output_dir work_dirs/$output_dir "
command_line+="--report_to wandb "
command_line+="--train_batch_size $train_batch_size "
command_line+="--pretrained_model_name_or_path $pretrained_model "
command_line+="--save_ckpt_freq $save_ckpt_freq "
command_line+="--adversarial_scale $adversarial_scale "

if [[ $blip_finetue == false ]]; then
	command_line+="--no_blip_finetue "
fi

if [[ $lora_mode == true ]]; then
	command_line+="--lora_mode "
	command_line+="--lora_rank 8 "
fi

if [[ $time_dependent == true ]]; then
	command_line+="--time_dependent "
fi

if [[ $noisy_mode == true ]]; then
	command_line+="--noisy_mode "
fi

if [[ $resume == true ]]; then
	command_line+="--resume "
	command_line+="--resume_ckpt $resume_ckpt "
fi

if [[ $oversample_from_unseen == true ]]; then
	command_line+="--oversample_from_unseen "
fi

if [[ $for_guidance == true ]]; then
	command_line+="--for_guidance "
fi

if [[ $manual_seen_file_name != "" ]]; then
	command_line+="--manual_seen_file_name $manual_seen_file_name "
fi

if [[ $manual_unseen_file_name != "" ]]; then
	command_line+="--manual_unseen_file_name $manual_unseen_file_name "
fi

command_line+="--model_yaml $model_yaml "

command_line+="--cycle $cycle "
command_line+="--seen_dataset_ratio $seen_dataset_ratio "
command_line+="--learning_rate $learning_rate "
command_line+="--num_train_epochs $training_epoch "


echo $command_line
eval $command_line


exit 0