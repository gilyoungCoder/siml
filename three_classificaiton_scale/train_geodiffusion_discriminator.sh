#!/bin/bash
output_dir=241123_naive_disc_mscoco_balanced_time_independent_cycle_1_initial_seen_0.1
dataset_config_name=configs/data/missing_person_512x512.py
dataset_config_type=new_coco_stuff # missing_person, new_coco_stuff
image_resolution=512
pretrained_model_mode=geodiffusion #geodiffusion, stablediffusion
blip_finetue=true
lora_mode=false
seen_dataset_ratio=0.1
learning_rate=1.0e-4


if [ $dataset_config_type == "missing_person" ] && [ $image_resolution == 256 ]; then
	dataset_config_name=configs/data/missing_person_256x256.py
	train_batch_size=16
	save_ckpt_freq=500
elif [ $dataset_config_type == "missing_person" ] && [ $image_resolution == 512 ]; then
	dataset_config_name=configs/data/missing_person_512x512.py
	train_batch_size=16
	save_ckpt_freq=1000
elif [ $dataset_config_type == "new_coco_stuff" ]; then
	dataset_config_name=configs/data/new_coco_stuff_512x512.py
	train_batch_size=32
	save_ckpt_freq=5000
fi

if [ $pretrained_model_mode == "stablediffusion" ]; then
	pretrained_model=runwayml/stable-diffusion-v1-5
elif [ $pretrained_model_mode == "geodiffusion" ] && [ $image_resolution == 512 ]; then
	pretrained_model=KaiChen1998/geodiffusion-coco-stuff-512x512
elif [ $pretrained_model_mode == "geodiffusion" ] && [ $image_resolution == 256 ]; then
	pretrained_model=KaiChen1998/geodiffusion-coco-stuff-256x256
fi


# "
command_line="bash "

if [[ $PJRT_DEVICE == "TPU" ]]; then
	echo "The device is set as TPU: get into the TPU mode."
	command_line+="tools/tpu/dist_train_discriminator.sh "
elif which nvidia-smi > /dev/null 2>&1; then
	echo "The device is set as GPU: get into the GPU mode."
	command_line+="tools/gpu/dist_train.sh "
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
command_line+="--seen_dataset_ratio $seen_dataset_ratio "
command_line+="--learning_rate $learning_rate "


if [[ $blip_finetue == false ]]; then
	command_line+="--no_blip_finetue "
fi

if [[ $lora_mode == true ]]; then
	command_line+="--lora_mode "
	command_line+="--lora_rank 8 "
fi

echo $command_line
eval $command_line


exit 0