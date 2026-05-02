#!/bin/bash

output_dir=241027_disc_vaal_mscoco_balanced_time_independent_seen_unseen
dataset_config_name=configs/data/missing_person_512x512.py
dataset_config_type=new_coco_stuff # missing_person, new_coco_stuff
image_resolution=512
pretrained_model_mode=geodiffusion #geodiffusion, stablediffusion
blip_finetue=true
lora_mode=false
time_dependent=true
adversarial_scale=1
seen_dataset_ratio=0.1  # 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4
cycle=0 # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# ckpt_path=/home/djfelrl11/geodiffusion/work_dirs/241107_disc_vaal_mscoco_balanced_time_independent_cycle_1_start_20_alpha_0.5_lr_1.0e_4_entropy/checkpoint/iter_44340
ckpt_path=/home/djfelrl11/geodiffusion/work_dirs/250108_disc_vaal_mscoco_balanced_time_dependent_cycle_2_seed_1/checkpoint/iter_65520
transductive=false

oversample_from_unseen=false


if [ $dataset_config_type == "missing_person" ] && [ $image_resolution == 256 ]; then
	dataset_config_name=configs/data/missing_person_256x256.py
	train_batch_size=16
	save_ckpt_freq=500
elif [ $dataset_config_type == "missing_person" ] && [ $image_resolution == 512 ]; then
	dataset_config_name=configs/data/missing_person_512x512.py
	train_batch_size=16
	save_ckpt_freq=250
elif [ $dataset_config_type == "new_coco_stuff" ]; then
	dataset_config_name=configs/data/new_coco_stuff_512x512.py
	train_batch_size=64
	# train_batch_size=16
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
	command_line+="tools/tpu/dist_calculate_vaal_value.sh "
elif which nvidia-smi > /dev/null 2>&1; then
	echo "The device is set as GPU: get into the GPU mode."
	# command_line+="tools/gpu/dist_train.sh "
	command_line+="tools/gpu/dist_train_discriminator_vaal.sh "
else
	echo "unknown device"
	exit 1
fi

command_line+="--dataset_config_name $dataset_config_name "
command_line+="--train_text_encoder_params added_embedding "
command_line+="--output_dir work_dirs/$output_dir "
command_line+="--report_to wandb "
command_line+="--train_batch_size $train_batch_size "
command_line+="--pretrained_model_name_or_path $pretrained_model "
command_line+="--save_ckpt_freq $save_ckpt_freq "
command_line+="--adversarial_scale $adversarial_scale "
command_line+="--ckpt_path $ckpt_path "

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

if [[ $transductive == true ]]; then
	command_line+="--transductive "
fi

if [[ $oversample_from_unseen == true ]]; then
	command_line+="--oversample_from_unseen "
fi

command_line+="--cycle $cycle "
command_line+="--seen_dataset_ratio $seen_dataset_ratio "



echo $command_line
eval $command_line


exit 0