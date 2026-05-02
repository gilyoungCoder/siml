#!/usr/bin/env python
# train5class.py

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from geo_models.classifier.classifier import load_discriminator

from huggingface_hub import HfFolder, whoami, Repository
import wandb
from tqdm.auto import tqdm

logger = get_logger(__name__)


class FiveClassDataset(Dataset):
    """
    5-class dataset:
      0: Nonpeople
      1: Clothed
      2: Revealing (Non-nude)
      3: Full Nudity
    """
    def __init__(
        self,
        nonpeople_dir: str,
        clothed_dir: str,
        revealing_dir: str,
        full_nudity_dir: str,
        transform=None,
    ):
        self.paths = []
        self.labels = []
        dirs = [
            (nonpeople_dir,    0),
            (clothed_dir,      1),
            (revealing_dir,    2),
            (full_nudity_dir,  3),
        ]
        for d, label in dirs:
            if not os.path.isdir(d):
                raise ValueError(f"데이터 디렉터리를 찾을 수 없습니다: {d}")
            for fname in sorted(os.listdir(d)):
                if fname.lower().endswith(".png"):
                    self.paths.append(os.path.join(d, fname))
                    self.labels.append(label)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "label": self.labels[idx]}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 5-class classifier of nudity/person levels with DDPM noise injection"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="사전 학습된 VAE/스케줄러 경로 (예: runwayml/stable-diffusion-v1-5)"
    )
    # “root 대신” 클래스별 디렉터리 경로를 각각 받도록 수정
    parser.add_argument("--nonpeople_dir",       type=str, required=True, help="Nonpeople 클래스 폴더 경로")
    parser.add_argument("--clothed_dir",         type=str, required=True, help="Clothed 클래스 폴더 경로")
    parser.add_argument("--revealing_dir",       type=str, required=True, help="Revealing (Non-nude) 클래스 경로")
    parser.add_argument("--full_nudity_dir",     type=str, required=True, help="Full Nudity 클래스 폴더 경로")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="five_class_output",
        help="모델 및 체크포인트가 저장될 디렉터리"
    )
    parser.add_argument("--seed", type=int, default=None, help="랜덤 시드 (선택 사항)")
    parser.add_argument("--train_batch_size", type=int, default=16, help="학습 배치 사이즈")
    parser.add_argument("--num_train_epochs", type=int, default=30, help="학습 epoch 수")
    parser.add_argument("--max_train_steps", type=int, default=None, help="최대 학습 스텝 (epoch 대신)")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="그래디언트 누적 스텝 수"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="학습률")
    parser.add_argument("--use_wandb", action="store_true", help="WandB 로깅 사용 여부")
    parser.add_argument("--wandb_project", type=str, default="five_class_project", help="WandB 프로젝트 이름")
    parser.add_argument("--wandb_run_name", type=str, default="five_class_run", help="WandB run 이름")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        default="no",
        help="혼합 정밀도 사용 (no | fp16 | bf16)"
    )
    parser.add_argument("--report_to", type=str, default="tensorboard", help="로그 리포팅 서비스")
    parser.add_argument("--push_to_hub", action="store_true", help="최종 모델을 HuggingFace Hub에 업로드할지")
    parser.add_argument("--hub_token", type=str, default=None, help="HuggingFace Hub 토큰")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Hub에 업로드할 모델 ID")
    parser.add_argument("--save_ckpt_freq", type=int, default=100, help="몇 스텝마다 체크포인트를 저장할지")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )
    device = accelerator.device
    logger.info(f"Using device: {device}")

    if args.seed is not None:
        seed = args.seed + accelerator.process_index
        set_seed(seed)
        logger.info(f"Set random seed to: {seed}")

    # HuggingFace Hub 설정 (push_to_hub 옵션이 켜졌을 때)
    if accelerator.is_local_main_process and args.push_to_hub:
        repo_name = args.hub_model_id or f"{whoami(HfFolder.get_token())['name']}/{Path(args.output_dir).name}"
        repo = Repository(args.output_dir, clone_from=repo_name)
        with open(os.path.join(args.output_dir, ".gitignore"), "w") as f:
            f.write("checkpoint/**\n")

    # ------------------------------------------------------------
    # 1) VAE & Scheduler 로드
    # ------------------------------------------------------------
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    vae.requires_grad_(False)
    vae.to(device)
    scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # ------------------------------------------------------------
    # 2) 데이터셋 & 데이터로더 준비
    # ------------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    full_dataset = FiveClassDataset(
        nonpeople_dir=args.nonpeople_dir,
        clothed_dir=args.clothed_dir,
        revealing_dir=args.revealing_dir,
        full_nudity_dir=args.full_nudity_dir,
        transform=transform,
    )
    total = len(full_dataset)
    print(f"Total images: {total}")

    # 전체의 10%를 검증용으로 분리
    val_size = int(0.1 * total)
    train_size = total - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ------------------------------------------------------------
    # 3) 분류기 (5-way) 불러오기
    # ------------------------------------------------------------
    classifier = load_discriminator(
        ckpt_path=None,   # 사전 학습된 체크포인트 사용 안 함
        condition=None,
        eval=False,
        channel=4,        # VAE 인코딩 출력 채널 수
        num_classes=4,    # 5개 클래스로 설정
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate)

    classifier, optimizer, train_loader, val_loader = accelerator.prepare(
        classifier, optimizer, train_loader, val_loader
    )

    # ------------------------------------------------------------
    # 4) 학습 루프
    # ------------------------------------------------------------
    global_step = 0
    max_steps = args.max_train_steps or args.num_train_epochs * len(train_loader)
    progress = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)

    classifier.train()
    while global_step < max_steps:
        for batch in train_loader:
            imgs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            # (1) VAE 인코딩
            latents = vae.encode(imgs).latent_dist.sample() * 0.18215

            # (2) DDPM 노이즈 주입
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (bsz,), device=device
            )
            noise = torch.randn_like(latents)
            alpha_bar = scheduler.alphas_cumprod.to(device)[timesteps].view(bsz, 1, 1, 1)
            noisy_latents = torch.sqrt(alpha_bar) * latents + torch.sqrt(1 - alpha_bar) * noise

            # (3) 분류기 순전파
            norm_ts = timesteps / scheduler.num_train_timesteps
            logits = classifier(noisy_latents, norm_ts)  # [B, 5]
            loss = loss_fn(logits, labels)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            progress.update(1)
            progress.set_postfix(step=global_step, loss=loss.item())

            # (4) 중간 검증 & 체크포인트 저장
            if global_step % args.save_ckpt_freq == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    ckpt_dir = os.path.join(args.output_dir, "checkpoint", f"step_{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(
                        accelerator.unwrap_model(classifier).state_dict(),
                        os.path.join(ckpt_dir, "classifier.pth")
                    )
                # validation
                classifier.eval()
                val_loss, val_acc, total_val = 0.0, 0.0, 0
                with torch.no_grad():
                    for vb in val_loader:
                        vimgs, vlabels = vb["pixel_values"].to(device), vb["label"].to(device)
                        vlat = vae.encode(vimgs).latent_dist.sample() * 0.18215
                        vtimesteps = torch.randint(
                            0, scheduler.num_train_timesteps, (vlat.shape[0],), device=device
                        )
                        vnoise = torch.randn_like(vlat)
                        valpha_bar = scheduler.alphas_cumprod.to(device)[vtimesteps].view(
                            vlat.shape[0], 1, 1, 1
                        )
                        noisy_vlat = torch.sqrt(valpha_bar) * vlat + torch.sqrt(1 - valpha_bar) * vnoise

                        vnorm_ts = vtimesteps / scheduler.num_train_timesteps
                        vlogits = classifier(noisy_vlat, vnorm_ts)
                        vloss = loss_fn(vlogits, vlabels)
                        preds = vlogits.argmax(dim=-1)

                        val_loss += vloss.item() * vlat.shape[0]
                        val_acc += (preds == vlabels).sum().item()
                        total_val += vlat.shape[0]

                val_loss /= total_val
                val_acc /= total_val
                logger.info(f"[Validation] Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                if args.use_wandb:
                    wandb.log({"val_loss": val_loss, "val_acc": val_acc}, step=global_step)
                classifier.train()

            if global_step >= max_steps:
                break

    # ------------------------------------------------------------
    # 5) 최종 모델 저장 및 HuggingFace Hub 업로드 (선택 사항)
    # ------------------------------------------------------------
    if accelerator.is_local_main_process:
        final_path = os.path.join(args.output_dir, "classifier_final.pth")
        torch.save(accelerator.unwrap_model(classifier).state_dict(), final_path)
        if args.push_to_hub:
            repo.push_to_hub(commit_message="Final 5-class model")

    accelerator.end_training()


if __name__ == "__main__":
    main()
