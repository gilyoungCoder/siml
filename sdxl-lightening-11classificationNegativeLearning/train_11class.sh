CUDA_VISIBLE_DEVICES=2 nohup python train_11class.py \
  --not_people_data_path  /mnt/home/yhgil99/dataset/sdxlLight/imagenet \
  --classes10_dir        /mnt/home/yhgil99/dataset/sdxlLight/10class \
  --output_dir           work_dirs/11cls1024NLL\
  --train_batch_size     4 \
  --num_train_epochs     8 \
  --use_wandb \
  --wandb_project        nudity11NL \
  --wandb_run_name       run1 \
  > train11.log 2>&1 &