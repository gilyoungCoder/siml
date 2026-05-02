CUDA_VISIBLE_DEVICES=4 nohup python train_11class.py \
  --not_people_data_path  /mnt/home/yhgil99/dataset/sdxlLight/imagenet \
  --classes10_dir        /mnt/home/yhgil99/dataset/sdxlLight/10class \
  --output_dir           work_dirs/11cls1024_version2 \
  --train_batch_size     4 \
  --num_train_epochs     3 \
  --use_wandb \
  --wandb_project        nudity11 \
  --wandb_run_name       run1 \
  > train11.log 2>&1 &