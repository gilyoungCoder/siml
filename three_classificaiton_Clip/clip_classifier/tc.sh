export CUDA_VISIBLE_DEVICES=6

nohup python train_L2C.py \
    --cifar_root /mnt/home/yhgil99/dataset/clip/cifar-10-batches-py/cifar10_test_subset1 \
    --vae_repo runwayml/stable-diffusion-v1-5 \
    --clip_model openai/clip-vit-large-patch14 \
    > nohup_latent2clip.log 2>&1 &