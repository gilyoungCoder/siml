PY_ARGS=${@:1}
PORT=${PORT:-29501}

accelerate launch --tpu --main_training_function main \
train_geodiffusion_augmented_discriminator.py \
    --pretrained_model_name_or_path KaiChen1998/geodiffusion-coco-stuff-512x512 \
    --prompt_version v1 --num_bucket_per_side 256 256 --bucket_sincos_embed --train_text_encoder \
    --foreground_loss_mode constant --foreground_loss_weight 2.0 --foreground_loss_norm \
    --seed 0 --train_batch_size 4 --gradient_accumulation_steps 1 \
    --num_train_epochs 60 --learning_rate 1.5e-4 --max_grad_norm 1 \
    --lr_text_layer_decay 0.95 --lr_text_ratio 0.75 --lr_scheduler cosine --lr_warmup_steps 3000 \
    --dataset_config_name configs/data/missing_person_256x256.py \
    --uncond_prob 0.1 \
    --save_ckpt_freq 500 \
    ${PY_ARGS}
