#!/usr/bin/env python3
# train_11class_nl.py  — 11-class nudity classifier with Negative Learning

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "label": y}


# --------------------------------------------------------------------------------------------------------------------
# Negative Learning via NLLLoss
# --------------------------------------------------------------------------------------------------------------------
nll_layer = nn.NLLLoss()

def nl_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute NL loss = -log(1 - p_comp) averaged over batch,
    where comp is a random complementary label per sample.
    """
    B, C = logits.shape
    # sample complementary labels
    with torch.no_grad():
        comp = torch.randint(0, C-1, (B,), device=logits.device)
        comp = torch.where(comp >= labels, comp + 1, comp)
    # log(1 - p)
    logp = torch.log(1.0 - F.softmax(logits, dim=-1) + 1e-8)
    return nll_layer(logp, comp)


# --------------------------------------------------------------------------------------------------------------------
# Initialize classifier to uniform (softmax = [1/11,...,1/11])
# --------------------------------------------------------------------------------------------------------------------
def init_uniform(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def get_args():
    p = argparse.ArgumentParser("11-class nudity classifier (NL)")
    p.add_argument("--not_people_data_path", required=True)
    p.add_argument("--classes10_dir",          required=True)
    p.add_argument("--output_dir", default="clf11_nl_out")
    p.add_argument("--sdxl_base", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--learning_rate",    type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=int, default=30)
    p.add_argument("--max_train_steps",  type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--mixed_precision", choices=["no","fp16","bf16"], default="no")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="nudity11_nl")
    p.add_argument("--wandb_run_name", default="run_nl")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    accel = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.use_wandb else None
    )
    device = accel.device
    logger.info(f"Device: {device}")

    if args.seed is not None:
        set_seed(args.seed + accel.process_index)

    # VAE & DDPM scheduler
    vae = AutoencoderKL.from_pretrained(args.sdxl_base, subfolder="vae").to(device)
    vae.requires_grad_(False)
    sched = DDPMScheduler.from_pretrained(
        args.sdxl_base, subfolder="scheduler",
        num_train_timesteps=1000, clip_sample=False
    )
    alphas = sched.alphas_cumprod.to(device)
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
    tr_ds, va_ds = torch.utils.data.random_split(
        ds, [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed or 42)
    )
    tr_loader = DataLoader(tr_ds, batch_size=args.train_batch_size,
                           shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=args.train_batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)

    # Classifier
    clf = load_discriminator(
        ckpt_path=None, condition=None,
        eval=False, channel=4, num_classes=11
    ).to(device)
    clf.apply(init_uniform)  # uniform initial outputs

    opt = torch.optim.AdamW(clf.parameters(), lr=args.learning_rate)

    clf, opt, tr_loader, va_loader = accel.prepare(clf, opt, tr_loader, va_loader)

    max_steps = args.max_train_steps or args.num_train_epochs * len(tr_loader)
    pbar = tqdm(range(max_steps), disable=not accel.is_local_main_process)
    step = 0
    best_val_acc = 0.0

    clf.train()
    while step < max_steps:
        for batch in tr_loader:
            imgs, labels = batch["pixel_values"].to(device), batch["label"].to(device)

            lat = vae.encode(imgs).latent_dist.sample() * 0.18215
            bs = lat.size(0)
            ts = torch.randint(0, 1000, (bs,), device=device)
            noise = torch.randn_like(lat)
            noisy = sqrt_a[ts] * lat + sqrt_1_a[ts] * noise

            logits = clf(noisy, ts.float()/1000)
            loss   = nl_loss(logits, labels)

            accel.backward(loss)
            opt.step(); opt.zero_grad()

            step += 1
            pbar.update(1)
            pbar.set_postfix(step=step, nl_loss=loss.item())
            if args.use_wandb and accel.is_local_main_process:
                wandb.log({"train_nl_loss": loss.item()}, step=step)

            if step % 100 == 0:
                accel.wait_for_everyone()
                clf.eval()
                total_nl = total = correct = 0.0
                with torch.no_grad():
                    for vb in va_loader:
                        vimgs, vlabels = vb["pixel_values"].to(device), vb["label"].to(device)
                        vlat = vae.encode(vimgs).latent_dist.sample() * 0.18215
                        vbs  = vlat.size(0)
                        vts  = torch.randint(0,1000,(vbs,),device=device)
                        vnoisy = sqrt_a[vts]*vlat + sqrt_1_a[vts]*torch.randn_like(vlat)

                        vlogits = clf(vnoisy, vts.float()/1000)
                        # compute validation NL loss
                        batch_nl = nl_loss(vlogits, vlabels).item() * vbs
                        total_nl += batch_nl
                        # accuracy
                        preds = vlogits.argmax(dim=-1)
                        correct += (preds == vlabels).sum().item()
                        total   += vbs

                avg_nl = total_nl / total
                val_acc = correct   / total
                logger.info(f"[VAL @ {step}] NL_loss={avg_nl:.4f}, acc={val_acc:.4f}")
                if args.use_wandb and accel.is_local_main_process:
                    wandb.log({"val_nl_loss": avg_nl, "val_acc": val_acc}, step=step)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    ckpt_dir = Path(args.output_dir)/"checkpoint"/f"step_{step:05d}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(accel.unwrap_model(clf).state_dict(), ckpt_dir/"classifier.pth")

                clf.train()

            if step >= max_steps:
                break
        accel.wait_for_everyone()

    if accel.is_local_main_process:
        torch.save(accel.unwrap_model(clf).state_dict(),
                   Path(args.output_dir)/"classifier_final.pth")

if __name__ == "__main__":
    main()
