export CUDA_VISIBLE_DEVICES=6

# A안: <clothed> (allow 데이터로 학습, 사람 유지 목적이면 --use_person_templates on)
nohup python train_ti_sd14.py \
  --train_data_dir /mnt/home/yhgil99/dataset/softDelete/allowlist \
  --placeholder_token "<clothed>" \
  --initializer_token "clothes" \
  --num_vectors 4 \
  --use_person_templates \
  --max_train_steps 6000 \
  --output_dir work_dirs/ti_clothed

# # B안: <nudity> (harm 데이터로 학습, 부정 컨셉)
# accelerate launch train_ti_sd14.py \
#   --train_data_dir /path/to/harm_data \
#   --placeholder_token "<nudity>" \
#   --initializer_token "nude" \
#   --num_vectors 4 \
#   --max_train_steps 6000 \
#   --output_dir work_dirs/ti_nudity
