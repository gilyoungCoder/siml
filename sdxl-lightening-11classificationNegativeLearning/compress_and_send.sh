#!/bin/bash

file=inference_start.sh
content=$(cat $file)

content_parse=()
IFS=$'\n'
readarray -t content_parse <<< "$content"

for val in "${content_parse[@]}"; do
    if [[ $val == *"="* ]]; then
        IFS="="
        read -ra val_parse <<< "$val"
        if [[ ${val_parse[0]} == "dataset_config_name" ]]; then
            dataset_config_name=${val_parse[1]}
        elif [[ ${val_parse[0]} == "checkpoint_dir" ]]; then
            checkpoint_dir=${val_parse[1]}
        elif [[ ${val_parse[0]} == "lora_mode" ]]; then
            lora_mode=${val_parse[1]}
        elif [[ ${val_parse[0]} == "freedom" ]]; then
            freedom=${val_parse[1]}
        elif [[ ${val_parse[0]} == "freedom_model_type" ]]; then
            freedom_model_type=${val_parse[1]}
        elif [[ ${val_parse[0]} == "freedom_scale" ]]; then
            freedom_scale=${val_parse[1]}
        elif [[ ${val_parse[0]} == "output_dir" ]]; then
            output_dir=${val_parse[1]}
        elif [[ ${val_parse[0]} == "cuda_visible_devices" ]]; then
            cuda_visible_devices=${val_parse[1]}
        elif [[ ${val_parse[0]} == "generated_bbox_dir" ]]; then
            generated_bbox_dir=${val_parse[1]}
        fi
    fi
    # IFS=' '
done

# target_compress_dir="$checkpoint_dir/$output_dir"
target_compress_dir=$output_dir
if [ ! -d $target_compress_dir ]; then
    target_compress_dir="$output_dir"
    if [ ! -d $target_compress_dir ]; then
        echo "The target directory does not exist: $target_compress_dir"
        exit 1
    fi
fi

echo $target_compress_dir

# IFS=/
readarray -d "/" -t output_dir_parse <<< "$output_dir"

# echo ${output_dir_parse[-1]}
compression_file_name=${output_dir_parse[-1]}

# remove line switch 
compression_file_name=${compression_file_name//[$'\t\r\n']}
compression_file_name_ext="$compression_file_name.tar.gz"
echo $compression_file_name_ext

tar -czvf $compression_file_name_ext $target_compress_dir

# scp -i ~/.ssh/google_compute_engine -p 2022 $compression_file_name_ext  jeongjun@137.68.191.45:~

