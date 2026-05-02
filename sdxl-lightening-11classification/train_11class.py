#!/usr/bin/env python3
# train_11class.py – SDXL-Lightning (4-step DDIM) 11-class classifier

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
from tqdm.auto import tqdm
import wandb

logger = get_logger(__name__)

CLS_10 = [
    "fully_clothed", "casual_wear", "summer_casual", "athletic_wear",
    "one_piece_swimwear", "bikini_swimwear", "lingerie",
    "topless_with_jeans", "implied_nude", "artistic_full_nude",
]

class ElevenClsDS(Dataset):
    def __init__(self, not_people_dir, classes10_dir, transform=None):
        samples = []
        p0 = Path(not_people_dir)
        if not p0.is_dir(): raise ValueError(f"{p0} not found")
        for f in sorted(p0.iterdir()):
            if f.suffix.lower() in (".png",".jpg",".jpeg",".webp"):
                samples.append((str(f), 0))
        root = Path(classes10_dir)
        for idx, cls in enumerate(CLS_10, start=1):
            d = root/cls
            if not d.is_dir(): raise ValueError(f"{d} not found")
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in (".png",".jpg",".jpeg",".webp"):
                    samples.append((str(f), idx))
        self.samples = samples
        self.transform = transform

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, y = self.samples[i]
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        return {"pixel_values": img, "label": y}


def get_args():
    p = argparse.ArgumentParser("11-class nudity classifier")
    # data
    p.add_argument("--not_people_data_path", required=True)
    p.add_argument("--classes10_dir", required=True)
    # output & model
    p.add_argument("--output_dir", default="clf11_out")
    p.add_argument("--sdxl_base", default="stabilityai/stable-diffusion-xl-base-1.0")
    # training
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=int, default=30)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--mixed_precision", choices=["no","fp16","bf16"], default="no")
    # logging
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="nudity11")
    p.add_argument("--wandb_run_name", default="run1")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
        wandb.watch_called = False  # 중복 호출 방지

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.use_wandb else None
    )
    device = accelerator.device

    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # VAE & 4-step DDPM scheduler
    vae = AutoencoderKL.from_pretrained(args.sdxl_base, subfolder="vae").to(device)
    vae.requires_grad_(False)
    scheduler = DDPMScheduler.from_pretrained(
        args.sdxl_base, subfolder="scheduler",
        num_train_timesteps=1000, clip_sample=False
    )
    alphas = scheduler.alphas_cumprod.to(device)
    sqrt_a   = alphas.sqrt().view(1000,1,1,1)
    sqrt_1_a = (1 - alphas).sqrt().view(1000,1,1,1)

    # DataLoader
    tfm = transforms.Compose([
        transforms.Resize((1024,1024)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = ElevenClsDS(args.not_people_data_path, args.classes10_dir, tfm)
    val_len = int(0.1 * len(ds))
    train_len = len(ds) - val_len
    tr_ds, va_ds = torch.utils.data.random_split(ds, [train_len, val_len],
                                                 generator=torch.Generator().manual_seed(args.seed or 42))
    tr_loader = DataLoader(tr_ds, batch_size=args.train_batch_size,
                           shuffle=True, num_workers=4)
    va_loader = DataLoader(va_ds, batch_size=args.train_batch_size,
                           shuffle=False, num_workers=4)

    # Classifier & optimizer
    clf = load_discriminator(
        ckpt_path=None, condition=None,
        eval=False, channel=4, num_classes=11
    ).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.learning_rate)
    ce  = nn.CrossEntropyLoss()

    clf, opt, tr_loader, va_loader = accelerator.prepare(
        clf, opt, tr_loader, va_loader
    )

    max_steps = args.max_train_steps or args.num_train_epochs * len(tr_loader)
    pbar = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)
    step = 0
    best_val = 0.0

    clf.train()
    while step < max_steps:
        for batch in tr_loader:
            imgs, labels = batch["pixel_values"], batch["label"]
            lat = vae.encode(imgs).latent_dist.sample() * 0.18215
            bs = lat.size(0)
            ts = torch.randint(0, 1000, (bs,), device=device)
            noise = torch.randn_like(lat)
            noisy = sqrt_a[ts] * lat + sqrt_1_a[ts] * noise

            out = clf(noisy, ts.float()/1000)
            loss = ce(out, labels.to(device))
            accelerator.backward(loss)
            opt.step(); opt.zero_grad()

            step += 1
            pbar.update(1)
            pbar.set_postfix(step=step, loss=loss.item())

            # 1) train loss 로깅
            if args.use_wandb and accelerator.is_local_main_process:
                accelerator.log({"train_loss": loss.item()}, step=step)

            # 2) 중간 저장 (예: 500 스텝마다)
            if step % 500 == 0 and accelerator.is_local_main_process:
                torch.save(
                    accelerator.unwrap_model(clf).state_dict(),
                    Path(args.output_dir)/f"step_{step}.pth"
                )

            # 3) Validation & 성능 기반 저장
            if step % 100 == 0:
                accelerator.wait_for_everyone()
                clf.eval()
                tot=acc=vloss=0.0
                with torch.no_grad():
                    for vb in va_loader:
                        vlat = vae.encode(vb["pixel_values"]).latent_dist.sample() * 0.18215
                        vbs = vlat.size(0)
                        vts = torch.randint(0,1000,(vbs,),device=device)
                        vnoisy = sqrt_a[vts]*vlat + sqrt_1_a[vts]*torch.randn_like(vlat)
                        vo = clf(vnoisy, vts.float()/1000)
                        vloss += ce(vo, vb["label"].to(device)).item() * vbs
                        acc   += (vo.argmax(-1)==vb["label"].to(device)).sum().item()
                        tot   += vbs
                val_loss = vloss / tot
                val_acc  = acc   / tot
                logger.info(f"[VAL @ {step}] loss={val_loss:.4f}, acc={val_acc:.4f}")

                # WandB에 validation 메트릭 로깅
                if args.use_wandb and accelerator.is_local_main_process:
                    accelerator.log({"val_loss": val_loss, "val_acc": val_acc}, step=step)

                # 성능 향상 시 체크포인트 저장
                if val_acc > best_val:
                    best_val = val_acc
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

    # 최종 저장
    if accelerator.is_local_main_process:
        torch.save(
            accelerator.unwrap_model(clf).state_dict(),
            Path(args.output_dir)/"classifier_final.pth"
        )

if __name__=="__main__":
    main()
