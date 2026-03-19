CUDA_VISIBLE_DEVICES=1 nohup python train_31class.py \
  --not_people_data_path  /mnt/home/yhgil99/dataset/sdxlLight/imagenet \
  --classes30_dir        /mnt/home/yhgil99/dataset/sdxlLight/30class \
  --output_dir           work_dirs/31cls1024_v2 \
  --train_batch_size     8 \
  --num_train_epochs     10 \
  --use_wandb \
  --wandb_project        nudity31 \
  --wandb_run_name       after2000 \
  > train31.log 2>&1 &
