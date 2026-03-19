import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from torchvision import transforms
from tqdm.auto import tqdm
import yaml
import wandb
from geo_models.classifier.classifier import create_classifier, load_discriminator


from mmengine.config import Config
# benign 데이터셋은 기존 COCO 기반 데이터셋을 사용하고, target(누드) 데이터셋은 별도 폴더에서 로드합니다.
from geo_utils.data.new_new_coco_stuff import NewNewCocoStuffDataset
from geo_models.embed import SplitEmbedding
# 여기서는 nudity classifier를 사용합니다.
from geo_utils.guidance_utils import GuidanceModel  # guidance_utils.py 파일 내 GuidanceModel 클래스
# from nudity_classifier import NudityClassifierGradientModel

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for nudity classifier (binary) with DDPM time-step noise injection for classifier guidance.")
    # 모델 관련 인자
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="Path or identifier for the pre-trained model (for VAE and scheduler).")
    parser.add_argument("--prompt_version", type=str, default="v1", help="Text prompt version.")
    parser.add_argument("--num_bucket_per_side", type=int, nargs="+", default=None, help="Bucket number along each side.")
    parser.add_argument("--bucket_sincos_embed", action="store_true", help="Use 2D sine-cosine embedding for bucket locations.")
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder.")
    parser.add_argument("--train_text_encoder_params", type=str, nargs="+", default=["token_embedding", "position", "encoder", "final_layer_norm"],
                        help="Parameters of text encoder to train.")
    # 데이터 관련 인자
    parser.add_argument("--dataset_config_name", type=str, default=None,
                        help="Config file (mmengine format) for benign dataset (non-nudity images).")
    parser.add_argument("--benign_data_path", type=str, default=None, required=True)
    parser.add_argument("--image_column", type=str, default="image", help="Image column name in benign dataset.")
    parser.add_argument("--caption_column", type=str, default="text", help="Caption column name in benign dataset.")
    # 추가: target(누드) 데이터셋 경로
    parser.add_argument("--nudity_data_path", type=str, required=True, help="Directory containing nudity images (target).")
    # 출력 및 기타 인자
    parser.add_argument("--output_dir", type=str, default="nudity_classifier_output", help="Directory to save checkpoints.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory for downloaded models/datasets.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    # 최적화 관련 인자
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size per device.")
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total training steps; if not provided, computed from epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Steps to accumulate gradients before update.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale learning rate by GPUs, accumulation steps, etc.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="Learning rate scheduler type.")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Warmup steps for learning rate scheduler.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit Adam optimizer.")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA model.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 for Adam.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 for Adam.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    # HF 및 logging 인자
    parser.add_argument("--push_to_hub", action="store_true", help="Push the model to Hugging Face Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="Token for HF Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Repository name for HF Hub.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="TensorBoard log directory.")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="no", help="Mixed precision mode.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Logging platform.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank.")
    # wandb 인자
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb logging.")
    parser.add_argument("--wandb_project", type=str, default="nudity_classifier_project", help="WandB project name.")
    parser.add_argument("--wandb_run_name", type=str, default="run_01", help="WandB run name.")
    # 추가: DDPM forward process 시 노이즈 주입 관련 인자 (노이즈 스탠다드 편차)
    parser.add_argument("--noise_std", type=float, default=1.0, help="Standard deviation for DDPM noise injection.")
    parser.add_argument("--save_ckpt_freq", type=int, default=100, help="Checkpoint 저장 주기 (steps).")

    
    args = parser.parse_args()
    return args

def collate_fn_non_foreground_loss_mode(examples):
    # 각 샘플이 튜플이나 리스트인 경우 첫번째 요소가 이미지 텐서라면,
    # ex[0]로 접근합니다.
    if isinstance(examples[0], dict):
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        input_ids = torch.stack([ex["input_ids"] for ex in examples]).long()
    elif isinstance(examples[0], (list, tuple)):
        pixel_values = torch.stack([ex[0] for ex in examples])
        # input_ids가 없다면, 더미 텐서 생성 (여기서는 모두 0)
        input_ids = torch.zeros(len(examples), dtype=torch.long)
    else:
        raise ValueError("Unexpected sample type in collate_fn_non_foreground_loss_mode.")
    return {"pixel_values": pixel_values, "input_ids": input_ids, "bbox_mask": None}


def main(models=None):  

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # wandb 초기화
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )
    logger.info("Device: %s", accelerator.device)
    logger.info("Processes: %s", accelerator.num_processes)

    if args.seed is not None:
        seed = args.seed + accelerator.process_index
        set_seed(seed)
        logger.info("Set random seed to: %s", seed)

    # HF Hub repository (필요 시)
    if accelerator.is_local_main_process and args.push_to_hub:
        from huggingface_hub import Repository
        repo_name = args.hub_model_id if args.hub_model_id else f"{whoami(HfFolder.get_token())['name']}/{Path(args.output_dir).name}"
        repo = Repository(args.output_dir, clone_from=repo_name)
        with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
            gitignore.write("step_*\n")
            gitignore.write("epoch_*\n")

    # VAE 로드 (모델 고정)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)

    # diffusion scheduler 로드 (DDPM)
    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    # # benign 데이터셋: dataset_config_name에 따른 benign 데이터 (COCO 기반)
    # benign_cfg = Config.fromfile(args.dataset_config_name)
    # benign_data_path = os.path.join(benign_cfg.data_root, benign_cfg.data.train.img_prefix)
    # benign_files = sorted(os.listdir(benign_data_path))
    # total_benign = len(benign_files)
    # # benign 데이터는 전체의 일정 비율만 사용 (target label 0)
    # benign_indices = random.sample(range(total_benign), int(total_benign * 0.1))  # 예: 10% 사용
    # 기존 Config.fromfile() 대신 args.benign_data_path 사용
    benign_data_path = args.benign_data_path  # command-line 인자로 전달받은 benign 데이터셋 폴더 경로
    benign_files = sorted(os.listdir(benign_data_path))
    total_benign = len(benign_files)
    # 만약 전체 데이터를 사용할 경우 아래 코드를 주석처리하거나, 원하는 비율만 사용하고 싶다면 그대로 사용
    benign_indices = random.sample(range(total_benign), int(total_benign * 1))  # 예: 전체의 100% 사용


    benign_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    benign_dataset = NewNewCocoStuffDataset(benign_data_path, indices=benign_indices, transform=benign_transform)

    # target 데이터셋: nudity 이미지가 저장된 폴더, 모든 이미지를 사용 (target label 1)
    target_data_path = args.nudity_data_path
    target_files = sorted(os.listdir(target_data_path))
    target_indices = list(range(len(target_files)))  # 전체 사용

    target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    target_dataset = NewNewCocoStuffDataset(target_data_path, indices=target_indices, transform=target_transform)

    benign_dataloader = torch.utils.data.DataLoader(benign_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True, num_workers=4 * accelerator.num_processes, collate_fn=collate_fn_non_foreground_loss_mode)
    target_dataloader = torch.utils.data.DataLoader(target_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True, num_workers=4 * accelerator.num_processes, collate_fn=collate_fn_non_foreground_loss_mode)

    # 검증셋 (각 10%)
    val_benign_idx = random.sample(range(len(benign_dataset)), int(len(benign_dataset) * 0.1))
    val_target_idx = random.sample(range(len(target_dataset)), int(len(target_dataset) * 0.1))
    val_benign_dataset = torch.utils.data.Subset(benign_dataset, val_benign_idx)
    val_target_dataset = torch.utils.data.Subset(target_dataset, val_target_idx)
    val_benign_dataloader = torch.utils.data.DataLoader(val_benign_dataset, shuffle=False, batch_size=args.train_batch_size, drop_last=False, num_workers=4 * accelerator.num_processes, collate_fn=collate_fn_non_foreground_loss_mode)
    val_target_dataloader = torch.utils.data.DataLoader(val_target_dataset, shuffle=False, batch_size=args.train_batch_size, drop_last=False, num_workers=4 * accelerator.num_processes, collate_fn=collate_fn_non_foreground_loss_mode)

    logger.info("Benign dataset size: %d", len(benign_dataset))
    logger.info("Target dataset size: %d", len(target_dataset))

    # collate function (binary classifier: bbox_mask 사용하지 않음)
    collate_fn = collate_fn_non_foreground_loss_mode

    # 학습 스텝 수 계산
    num_update_steps_per_epoch = math.ceil(min(len(benign_dataloader), len(target_dataloader)) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # nudity classifier 모델 로드 (binary output)
    model_config = {}  # 필요시 추가 설정 가능 (예: layer depth, channel 수 등)
    # GuidanceModel 생성 (모델 타입은 "nudity_classifier")
    from diffusers import StableDiffusionPipeline

    # StableDiffusionPipeline에서 VAE와 scheduler를 추출하는 예시:
    pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path)
    vae = pipe.vae
    scheduler = pipe.scheduler
    model_config = {
        "model_type": "nudity_classifier",
        "model_args": {
        # NudityClassifierGradientModel에 필요한 인자들, 예를 들어:
        # "num_filters": 64,
        # "num_layers": 4,
        # "dropout": 0.1,
        }
    }

    # guidance_model = GuidanceModel(pipe, model_config, None, device=accelerator.device)
    # # training에 사용할 classifier는 guidance_model.gradient_model입니다.
    # classifier_model = guidance_model
    # classifier_model = load_discriminator(ckpt_path=None, condition=None, eval=False, channel=4)
    clf_path="/mnt/home/yhgil99/guided3-safe-diffusion/geodiffusion-features-code_refactor/work_dirs/vangoh_classifier_checkpoint/checkpoint/iter_4001/nudity_classifier.pth"  # 자유 모델 체크포인트
    classifier_model = load_discriminator(ckpt_path=clf_path, condition=None, eval=False, channel=4)


    # classifier_model = NudityClassifierGradientModel(model_config, device=accelerator.device)

    # optimizer 설정
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Install bitsandbytes with `pip install bitsandbytes` to use 8-bit Adam.")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        classifier_model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Accelerator로 wrap
    classifier_model, optimizer, benign_dataloader, target_dataloader = accelerator.prepare(
        classifier_model, optimizer, benign_dataloader, target_dataloader
    )

    # 무한 dataloader 함수
    def infinite_dataloader(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    benign_dataloader = infinite_dataloader(benign_dataloader)
    target_dataloader = infinite_dataloader(target_dataloader)

    val_benign_dataloader, val_target_dataloader = accelerator.prepare(val_benign_dataloader, val_target_dataloader)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    logger.info("Weight dtype: %s", weight_dtype)

    vae.to(device=accelerator.device)
    vae.to(dtype=weight_dtype)

    if accelerator.is_local_main_process:
        saved_args = copy.copy(args)
        if args.num_bucket_per_side is not None:
            saved_args.num_bucket_per_side = ' '.join([str(x) for x in args.num_bucket_per_side])
        saved_args.train_text_encoder_params = ' '.join(args.train_text_encoder_params)
        accelerator.init_trackers("nudity_classifier_finetune", config=vars(saved_args))
        if args.use_wandb:
            # wandb.config.update(vars(args))
            wandb.config.update(vars(args), allow_val_change=True)


    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info("Total examples (per epoch estimate): %d", min(len(benign_dataset), len(target_dataset)))
    logger.info("Num Epochs: %d", args.num_train_epochs)
    logger.info("Batch size per device: %d", args.train_batch_size)
    logger.info("Total train batch size: %d", total_batch_size)
    logger.info("Gradient Accumulation steps: %d", args.gradient_accumulation_steps)
    logger.info("Total optimization steps: %d", args.max_train_steps)

    # Progress bar
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # 검증 단계 (여기서는 dummy timestep=0 사용)
    def validate(val_benign_dl, val_target_dl):
        classifier_model.eval()
        benign_loss = 0.0
        target_loss = 0.0
        benign_acc = 0.0
        target_acc = 0.0
        benign_total = 0
        target_total = 0
        with torch.no_grad():
            for batch in tqdm(val_benign_dl, desc="Val benign", disable=not accelerator.is_local_main_process):
                imgs = batch["pixel_values"].to(weight_dtype)
                lat = vae.encode(imgs).latent_dist.sample() * 0.18215
                # 고정 timestep 0로 평가 (또는 평균 timestep 등)
                ts = torch.zeros(lat.shape[0], device=lat.device, dtype=torch.long)
                pred = classifier_model(lat, ts)
                pred = torch.sigmoid(pred)
                target = torch.zeros_like(pred)
                loss = loss_function(pred, target)
                benign_loss += loss.item()
                benign_acc += ((pred < 0.5) == (target < 0.5)).sum().item()
                benign_total += len(pred)
            for batch in tqdm(val_target_dl, desc="Val target", disable=not accelerator.is_local_main_process):
                imgs = batch["pixel_values"].to(weight_dtype)
                lat = vae.encode(imgs).latent_dist.sample() * 0.18215
                ts = torch.zeros(lat.shape[0], device=lat.device, dtype=torch.long)
                pred = classifier_model(lat, ts)
                pred = torch.sigmoid(pred)
                target = torch.ones_like(pred)
                loss = loss_function(pred, target)
                target_loss += loss.item()
                target_acc += ((pred > 0.5) == (target > 0.5)).sum().item()
                target_total += len(pred)
        benign_loss /= benign_total
        benign_acc /= benign_total
        target_loss /= target_total
        target_acc /= target_total
        metrics = {"val_benign_loss": benign_loss, "val_target_loss": target_loss,
                   "val_benign_acc": benign_acc, "val_target_acc": target_acc, "val_loss": (benign_loss+target_loss)/2, "val_acc": (benign_acc+target_acc)/2 }
        accelerator.log(metrics, step=global_step)
        if args.use_wandb:
            wandb.log(metrics, step=global_step)
        logger.info("Validation benign loss: %.4f, acc: %.4f", benign_loss, benign_acc)
        logger.info("Validation target loss: %.4f, acc: %.4f", target_loss, target_acc)
        classifier_model.train()
        
    # 학습 loop 시작
    # 여기서 DDPM forward process를 모사하여, 각 배치마다 랜덤 timestep을 샘플하고,
    # 노이즈를 주입하여 noisy latent를 생성한 후 classifier를 forward합니다.
    while global_step < args.max_train_steps:
        train_loss = 0.0

        benign_batch = next(benign_dataloader)
        target_batch = next(target_dataloader)
        # benign와 target 배치를 하나로 결합
        benign_imgs = benign_batch["pixel_values"].to(weight_dtype)
        target_imgs = target_batch["pixel_values"].to(weight_dtype)
        all_imgs = torch.cat([benign_imgs, target_imgs], dim=0)

        # VAE 인코딩: clean latent
        latents = vae.encode(all_imgs).latent_dist.sample()
        latents = latents * 0.18215

        bsz = latents.shape[0]
        # 각 샘플에 대해 랜덤 time step (0 ~ T-1) 샘플링
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        # 각 샘플에 대해 noise 생성
        noise = torch.randn_like(latents)
        # alphas_cumprod 값 추출 (배치 차원 확장)
        alpha_cumprod = noise_scheduler.alphas_cumprod.to(latents.device)
        alpha_bar = alpha_cumprod[timesteps].view(bsz, *([1] * (latents.ndim - 1)))
        # DDPM forward process: noisy latent = sqrt(alpha_bar)*clean_latent + sqrt(1-alpha_bar)*noise
        noisy_latents = torch.sqrt(alpha_bar) * latents + torch.sqrt(1 - alpha_bar) * noise

        # classifier forward: noisy_latents와 해당 timestep을 입력 (타입 캐스팅 유지)
        pred = classifier_model(noisy_latents, timesteps)
        pred = torch.sigmoid(pred)
        # 라벨 할당: benign 부분(앞쪽)은 0, target 부분(뒷쪽)은 1
        labels = torch.zeros_like(pred)
        labels[len(benign_imgs):] = 1

        loss = loss_function(pred, labels)
        loss = loss.mean()

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1
        progress_bar.update(1)
        if global_step % 100 == 1:
            accelerator.wait_for_everyone()
            accelerator.print("Checkpoint save at step", global_step)
            if accelerator.is_local_main_process:
                classifier_model_unwrapped = accelerator.unwrap_model(classifier_model)
                classifier_model_cpu = copy.deepcopy(classifier_model_unwrapped).to("cpu")
                ckpt_dir = os.path.join(args.output_dir, 'checkpoint', f'iter_{global_step}')
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(classifier_model_cpu.state_dict(), os.path.join(ckpt_dir, "nudity_classifier.pth"))
            accelerator.wait_for_everyone()
            validate(val_benign_dataloader, val_target_dataloader)
        logs = {"step_loss": loss.detach().item(), "step": global_step}
        accelerator.log(logs, step=global_step)
        if args.use_wandb:
            wandb.log(logs, step=global_step)
        progress_bar.set_postfix(**logs)
        train_loss = 0.0

    
    validate(val_benign_dataloader, val_target_dataloader)

    # Save final checkpoint
    classifier_model_unwrapped = accelerator.unwrap_model(classifier_model)
    classifier_model_cpu = copy.deepcopy(classifier_model_unwrapped).to("cpu")
    final_ckpt_dir = os.path.join(args.output_dir, 'checkpoint', f'iter_{global_step}')
    os.makedirs(final_ckpt_dir, exist_ok=True)
    torch.save(classifier_model_cpu.state_dict(), os.path.join(final_ckpt_dir, "nudity_classifier.pth"))

    if args.push_to_hub:
        repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()

if __name__ == "__main__":
    # BCELoss for binary classification
    loss_function = nn.BCELoss()
    main()
