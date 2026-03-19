#!/usr/bin/env python3
# train_4class.py

"""
4-class nudity classifier for SDXL-Lightning
(0: not-people | 1: fully-clothed | 2: partial-nude | 3: full-nude)
DDPM-noise-injected latent training (1000-step).
"""

import argparse, os
from pathlib import Path

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from geo_models.classifier.classifier import load_discriminator
from huggingface_hub import HfFolder, Repository, whoami
import wandb
from tqdm.auto import tqdm

logger = get_logger(__name__)

class FourClassDataset(Dataset):
    """0: not_people | 1: fully_clothed | 2: partial_nude | 3: full_nude"""
    def __init__(self, not_people_dir, fully_clothed_dir,
                 partial_nude_dir, full_nude_dir, transform=None):
        samples = []
        for root, label in [
            (not_people_dir,    0),
            (fully_clothed_dir, 1),
            (partial_nude_dir,  2),
            (full_nude_dir,     3),
        ]:
            if not os.path.isdir(root):
                raise ValueError(f"데이터 디렉터리를 찾을 수 없습니다: {root}")
            for fname in sorted(os.listdir(root)):
                if fname.lower().endswith((".png",".jpg",".jpeg",".webp")):
                    samples.append((os.path.join(root, fname), label))
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "label": label}


def get_args():
    p = argparse.ArgumentParser("4-class nudity classifier")
    # data
    p.add_argument("--not_people_data_path",    required=True)
    p.add_argument("--fully_clothed_data_path", required=True)
    p.add_argument("--partial_nude_data_path",  required=True)
    p.add_argument("--full_nude_data_path",     required=True)
    # output / model
    p.add_argument("--output_dir",    default="clf4_out")
    p.add_argument("--sdxl_base",     default="stabilityai/stable-diffusion-xl-base-1.0")
    # training
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--learning_rate",    type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=int, default=30)
    p.add_argument("--max_train_steps",  type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # misc
    p.add_argument("--seed",            type=int, default=None)
    p.add_argument("--mixed_precision", choices=["no","fp16","bf16"], default="no")
    # logging
    p.add_argument("--use_wandb",       action="store_true")
    p.add_argument("--wandb_project",   default="nudity4")
    p.add_argument("--wandb_run_name",  default="run1")
    # hub
    p.add_argument("--push_to_hub",     action="store_true")
    p.add_argument("--hub_model_id",    type=str, default=None)
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # WandB 초기화
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
        wandb.watch_called = False

    # Accelerator (+WandB 로깅)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.use_wandb else None
    )
    device = accelerator.device
    logger.info(f"Using device: {device}")

    # Seed 설정
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Hub 설정 (옵션)
    if accelerator.is_local_main_process and args.push_to_hub:
        repo_name = args.hub_model_id or \
            f"{whoami(HfFolder.get_token())['name']}/{Path(args.output_dir).name}"
        repo = Repository(args.output_dir, clone_from=repo_name)
        (Path(args.output_dir)/".gitignore").write_text("checkpoint/**\n")

    # VAE 및 1000-step DDPM Scheduler 로드
    vae = AutoencoderKL.from_pretrained(args.sdxl_base, subfolder="vae").to(device)
    vae.requires_grad_(False)
    scheduler = DDPMScheduler.from_pretrained(
        args.sdxl_base, subfolder="scheduler",
        num_train_timesteps=1000, clip_sample=False
    )
    alphas = scheduler.alphas_cumprod.to(device)
    sqrt_a  = alphas.sqrt().view(1000,1,1,1)
    sqrt_1a = (1 - alphas).sqrt().view(1000,1,1,1)

    # 데이터셋 & Dataloader
    tfm = transforms.Compose([
        transforms.Resize((1024,1024)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = FourClassDataset(
        args.not_people_data_path,
        args.fully_clothed_data_path,
        args.partial_nude_data_path,
        args.full_nude_data_path,
        transform=tfm
    )
    val_len   = int(0.1 * len(ds))
    train_len = len(ds) - val_len
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed or 42)
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.train_batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.train_batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # 분류기 모델 & 옵티마이저
    clf = load_discriminator(
        ckpt_path=None, condition=None,
        eval=False, channel=4, num_classes=4
    ).to(device)
    if args.mixed_precision=="fp16":
        clf = clf.half()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(clf.parameters(), lr=args.learning_rate)

    # Accelerator 준비
    clf, optimizer, train_loader, val_loader = accelerator.prepare(
        clf, optimizer, train_loader, val_loader
    )

    # 학습 설정
    max_steps = args.max_train_steps or (args.num_train_epochs * len(train_loader))
    pbar = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)
    step = 0
    best_val_acc = 0.0

    clf.train()
    while step < max_steps:
        for batch in train_loader:
            imgs   = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            # latent 추출 및 노이즈 주입
            lat = vae.encode(imgs).latent_dist.sample() * 0.18215
            bsz  = lat.size(0)
            ts   = torch.randint(0, 1000, (bsz,), device=device)
            noise = torch.randn_like(lat)
            noisy = sqrt_a[ts] * lat + sqrt_1a[ts] * noise

            # forward & backward
            logits = clf(noisy, ts.float()/1000.0)
            loss   = loss_fn(logits, labels)
            accelerator.backward(loss)
            optimizer.step(); optimizer.zero_grad()

            # 스텝 카운트 및 진행바
            step += 1
            pbar.update(1)
            pbar.set_postfix(step=step, loss=loss.item())

            # 1) 학습 손실 로깅
            if args.use_wandb:
                accelerator.log({"train_loss": loss.item()}, step=step)

            # 2) 주기적 체크포인트 (예: 500 스텝마다)
            if step % 500 == 0 and accelerator.is_local_main_process:
                torch.save(
                    accelerator.unwrap_model(clf).state_dict(),
                    Path(args.output_dir)/f"step_{step}.pth"
                )

            # 3) 검증 및 성능 향상 체크포인트
            if step % 100 == 0:
                accelerator.wait_for_everyone()
                clf.eval()
                tot = corr = vloss = 0.0
                with torch.no_grad():
                    for vb in val_loader:
                        vimg   = vb["pixel_values"].to(device)
                        vlbl   = vb["label"].to(device)
                        vlat   = vae.encode(vimg).latent_dist.sample() * 0.18215
                        vts    = torch.randint(0,1000,(vlat.size(0),),device=device)
                        vnoisy = sqrt_a[vts]*vlat + sqrt_1a[vts]*torch.randn_like(vlat)
                        vout   = clf(vnoisy, vts.float()/1000.0)
                        batch_sz = vlat.size(0)
                        vloss   += loss_fn(vout, vlbl).item() * batch_sz
                        corr    += (vout.argmax(-1)==vlbl).sum().item()
                        tot     += batch_sz
                val_loss = vloss / tot
                val_acc  = corr   / tot
                logger.info(f"[VAL @ {step}] loss={val_loss:.4f} acc={val_acc:.4f}")

                if args.use_wandb:
                    wandb.log({"val_loss": val_loss, "val_acc": val_acc}, step=step)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    ckpt_dir = Path(args.output_dir)/"checkpoint"/f"step_{step:05d}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        accelerator.unwrap_model(clf).state_dict(),
                        ckpt_dir/"classifier.pth"
                    )
                clf.train()

            if step >= max_steps:
                break

        accelerator.wait_for_everyone()

    # 최종 모델 저장
    if accelerator.is_local_main_process:
        final_ckpt = Path(args.output_dir)/"classifier_final.pth"
        torch.save(accelerator.unwrap_model(clf).state_dict(), final_ckpt)
        if args.push_to_hub:
            repo.push_to_hub("final checkpoint")

    accelerator.end_training()


if __name__ == "__main__":
    main()
