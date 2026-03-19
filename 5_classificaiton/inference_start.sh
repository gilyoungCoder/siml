#!/bin/bash

# Mode for called by other scripts
called=${CALLED:-false}

# Mode selection
geodiffusion_mode=coco_stuff # missing_person, new_coco_stuff
lora_mode=false
freedom=true
freedom_model_type=time_dependent_vaal

cuda_visible_devices=0
cycle=1
geodiffusion_scale_list=(4.0)
freedom_scale_list=(10.0 0.0)
# freedom_scale_list=(10.0)
# freedom_scale_list=(3.0 5.0 7.0)

for freedom_scale in ${freedom_scale_list[@]}; do
for geodiffusion_scale in ${geodiffusion_scale_list[@]}; do

output_dir=output_img/250224_ms_coco_guide_coreset_gap_retinanet_cycle_1_disc_scale_${freedom_scale}
manual_generation_file_list="/home/djfelrl11/cycle_${cycle}_generation_list.yaml"
guide_start=0
img2img=false
strength=0.6
grad_only_bbox=false
grad_bbox_reverse=false
seed_list=(0 1 2) # (0)



# generated_bbox_dir=/home/djfelrl11/data_random_bbox/annotations
generated_bbox_dir=""

if [[ $geodiffusion_mode == "missing_person" ]]; then
	dataset_config_name=configs/data/missing_person_512x512.py
	checkpoint_dir=work_dirs/missing_person_512x512_240629/checkpoint/iter_2820
elif [[ $geodiffusion_mode == "coco_stuff" ]]; then
	dataset_config_name=configs/data/new_coco_stuff_512x512.py
	checkpoint_dir=KaiChen1998/geodiffusion-coco-stuff-512x512
fi


case $freedom_model_type in
	"classifier")
		freedom_model_args_file=configs/models/classifier.yaml
		freedom_model_ckpt=work_dirs/missing_person_512_512_240910_clf/checkpoint/iter_1620/classifier.pth
		;;
	"discriminator")
		freedom_model_args_file=configs/models/discriminator.yaml
		freedom_model_ckpt=work_dirs/missing_person_512_512_240910_disc/checkpoint/iter_2760/discriminator.pth
		;;
	"yolo")
		freedom_model_args_file=configs/models/yolo.yaml
		freedom_model_ckpt="/home/djfelrl11/0926_coco_bayesian/weights/best.pt"
		;;
	"time_dependent_discriminator")
		freedom_model_args_file=configs/models/time_dependent_discriminator.yaml
		freedom_model_ckpt=work_dirs/missing_person_512_512_240916_disc/checkpoint/iter_2760/discriminator.pth
		;;
	"vaal")
		freedom_model_args_file=configs/models/vaal.yaml
		freedom_model_ckpt=/home/djfelrl11/geodiffusion/work_dirs/241027_disc_vaal_mscoco_balanced_time_independent_seen_unseen/checkpoint/iter_44340
		;;
	"time_dependent_vaal")
		freedom_model_args_file=/home/djfelrl11/geodiffusion/configs/models/time_dependent_vaal.yaml
		freedom_model_ckpt=/home/djfelrl11/geodiffusion/work_dirs/250223_disc_vaal_mscoco_balanced_time_dependent_cycle_1_epoch_100_seen_0.04_coreset_selected_retinanet/checkpoint/iter_7350
		;;
	"augmented_discriminator")
		freedom_model_args_file=configs/models/augmented_discriminator_no_pretrained_layer.yaml
		freedom_model_ckpt=/home/djfelrl11/geodiffusion/work_dirs/241223_mscoco_balanced_augmented_discriminator_no_pretrained_layer_no_label/checkpoint/iter_1/discriminator_head.pth
		
		;;
	"obj_detection_discriminator")
		freedom_model_args_file=configs/models/object_detection_discriminator.yaml
		freedom_model_ckpt=/home/djfelrl11/geodiffusion/work_dirs/250109_mscoco_balanced_obj_detection_disc/checkpoint/iter_15001/discriminator_head.pth
		;;

	*)
		echo "Unknown model type: $freedom_model_type"
		exit 1
		;;
esac

if [[ $called != "false" ]]; then
	output_dir=${OUTPUT_DIR}_disc_scale_${freedom_scale}
	freedom_model_args_file=${FREEDOM_MODEL_ARGS_FILE:-$freedom_model_args_file}
	freedom_model_ckpt=${FREEDOM_MODEL_CKPT:-$freedom_model_ckpt}
	cycle=${CYCLE:-$cycle}
	manual_generation_file_list="/home/djfelrl11/cycle_${cycle}_generation_list.yaml"
	echo "Called by other scripts"
fi


for seed in ${seed_list[@]}; do
command_line="CUDA_VISIBLE_DEVICES=$cuda_visible_devices bash tools/gpu/dist_test.sh $checkpoint_dir --dataset_config_name $dataset_config_name "

	if [[ $lora_mode == true ]]; then
		command_line+="--lora_mode "
		command_line+="--lora_model runwayml/stable-diffusion-v1-5 "
	fi

	if [[ $freedom == true ]]; then
		command_line+="--freedom "
		command_line+="--freedom_model_type $freedom_model_type "
		command_line+="--freedom_model_ckpt $freedom_model_ckpt "
		command_line+="--freedom_model_args_file $freedom_model_args_file "
		command_line+="--freedom_scale $freedom_scale "
		command_line+="--freedom_bald_iteration 10 "
	fi

	if [[ $generated_bbox_dir != "" ]]; then
		command_line+="--generated_bbox_dir $generated_bbox_dir "
	fi

	if [[ $geodiffusion_mode == "missing_person" ]]; then
		command_line+="--trained_text_encoder "
	fi

	if [[ $manual_generation_file_list != "" ]]; then
		command_line+="--manual_generation_file_list $manual_generation_file_list "
	fi

	if [[ $guide_start != "" ]]; then
		command_line+="--guide_start $guide_start "
	fi

	if [[ $img2img == true ]]; then
		command_line+="--img2img "
		command_line+="--strength $strength "
	fi

	if [[ $seed != "" ]]; then
		command_line+="--seed $seed "
	fi

	if [[ $grad_only_bbox == true ]]; then
		command_line+="--grad_only_bbox "
	fi

	if [[ $grad_bbox_reverse == true ]]; then
		command_line+="--grad_bbox_reverse "
	fi

	if [[ $geodiffusion_scale != "" ]]; then
		command_line+="--geodiffusion_scale $geodiffusion_scale "
	fi

	
	command_line+="--output_dir ${output_dir}_${seed} "
	command_line+="--cycle 1 "

	eval $command_line
done
done

python compress_and_send.py $output_dir
done