import argparse
import logging
import math
import os
import random
import copy
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from geo_models.classifier.classifier import create_classifier, discriminator_defaults, load_discriminator

from huggingface_hub import HfFolder, whoami, Repository
import yaml
import wandb
from tqdm.auto import tqdm

logger = get_logger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(d: str) -> List[str]:
    files = []
    if d is None:
        return files
    for fname in sorted(os.listdir(d)):
        p = os.path.join(d, fname)
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMG_EXTS:
            files.append(p)
    return files

class TwoClassDataset(Dataset):
    """
    2-class dataset:
      0: allow (non-nude, e.g., benign/person/fully-clothed)  <-- allow_data_path
      1: harm  (nude)                                          <-- harm_data_path
    """
    def __init__(self, allow_dir: str, harm_dir: str, transform=None):
        self.paths = []
        self.labels = []

        # allow -> 0
        for p in list_images(allow_dir):
            self.paths.append(p)
            self.labels.append(0)

        # harm -> 1
        for p in list_images(harm_dir):
            self.paths.append(p)
            self.labels.append(1)

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return {"pixel_values": img, "label": label}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 2-class allow/harm (non-nude/nude) classifier with DDPM noise injection"
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--allow_data_path", type=str, required=True, help="root dir for allow (non-nude) images")
    parser.add_argument("--harm_data_path", type=str, required=True, help="root dir for harm (nude) images")
    parser.add_argument("--output_dir", type=str, default="two_class_output")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="two_class_project")
    parser.add_argument("--wandb_run_name", type=str, default="run1")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--save_ckpt_freq", type=int, default=100)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )
    device = accelerator.device
    logger.info(f"Using device: {device}")

    # Seed
    if args.seed is not None:
        seed = args.seed + accelerator.process_index
        set_seed(seed)
        logger.info(f"Set random seed to: {seed}")

    # Hub setup
    if accelerator.is_local_main_process and args.push_to_hub:
        user = whoami(HfFolder.get_token())["name"]
        repo_name = args.hub_model_id or f"{user}/{Path(args.output_dir).name}"
        repo = Repository(args.output_dir, clone_from=repo_name)
        with open(os.path.join(args.output_dir, ".gitignore"), "w") as f:
            f.write("checkpoint/**\n")

    # VAE & scheduler
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False).to(device)

    scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # Dataset & Dataloader
    full_dataset = TwoClassDataset(
        args.allow_data_path,
        args.harm_data_path,
        transform=transform,
    )
    total = len(full_dataset)
    logger.info(f"Total images: {total}")
    print(f"total len: {total}")

    val_size = max(1, int(0.1 * total))
    train_size = total - val_size
    # 동일 시드로 재현 가능한 split
    gen = torch.Generator().manual_seed(args.seed if args.seed is not None else 42)
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.train_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model: 2-way classifier (channel=4, num_classes=2)
    # disc_args = discriminator_defaults(); disc_args["in_channels"]=4; disc_args["out_channels"]=2
    # classifier = create_classifier(**disc_args).to(device)
    classifier = load_discriminator(ckpt_path=None, condition=None, eval=False, channel=4, num_classes=2).to(device)

    # Optim & Loss
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate)

    # Prepare with accelerator
    classifier, optimizer, train_loader, val_loader = accelerator.prepare(
        classifier, optimizer, train_loader, val_loader
    )

    # Training loop
    global_step = 0
    max_steps = args.max_train_steps or args.num_train_epochs * len(train_loader)
    progress = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)

    classifier.train()
    while global_step < max_steps:
        for batch in train_loader:
            imgs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)  # 0=allow, 1=harm

            # VAE encode -> latents
            latents = vae.encode(imgs).latent_dist.sample() * 0.18215

            # random timestep & noise injection
            bsz = latents.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=device)
            noise = torch.randn_like(latents)
            alpha_cumprod = scheduler.alphas_cumprod.to(device)
            alpha_bar = alpha_cumprod[timesteps].view(bsz, *([1]*(latents.ndim-1)))
            noisy_latents = torch.sqrt(alpha_bar) * latents + torch.sqrt(1 - alpha_bar) * noise

            # Classifier forward
            norm_ts = timesteps / scheduler.num_train_timesteps
            logits = classifier(noisy_latents, norm_ts)  # [B,2]
            loss = loss_fn(logits, labels)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            progress.update(1)
            progress.set_postfix(step=global_step, loss=loss.item())

            # checkpoint & validate
            if global_step % args.save_ckpt_freq == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    ckpt_dir = os.path.join(args.output_dir, "checkpoint", f"step_{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(accelerator.unwrap_model(classifier).state_dict(),
                               os.path.join(ckpt_dir, "classifier.pth"))
                # validation
                classifier.eval()
                val_loss, val_acc = 0.0, 0.0
                total_val = 0
                with torch.no_grad():
                    for vb in val_loader:
                        vimgs   = vb["pixel_values"].to(device)
                        vlabels = vb["label"].to(device)

                        vlat = vae.encode(vimgs).latent_dist.sample() * 0.18215

                        bsz = vlat.shape[0]
                        vtimesteps = torch.randint(
                            0, scheduler.num_train_timesteps, (bsz,),
                            device=device, dtype=torch.long,
                        )
                        vnoise = torch.randn_like(vlat)
                        alpha_cumprod = scheduler.alphas_cumprod.to(device)
                        alpha_bar = alpha_cumprod[vtimesteps].view(bsz, *([1]*(vlat.ndim-1)))
                        noisy_vlat = torch.sqrt(alpha_bar) * vlat + torch.sqrt(1 - alpha_bar) * vnoise

                        vnorm_ts = vtimesteps / scheduler.num_train_timesteps
                        vlogits = classifier(noisy_vlat, vnorm_ts)   # [B,2]
                        vloss = loss_fn(vlogits, vlabels)
                        preds = vlogits.argmax(dim=-1)

                        val_loss += vloss.item() * bsz
                        val_acc  += (preds == vlabels).sum().item()
                        total_val += bsz
                val_loss /= max(1, total_val)
                val_acc  /= max(1, total_val)
                logger.info(f"[Validation] Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                if args.use_wandb:
                    wandb.log({"val_loss": val_loss, "val_acc": val_acc}, step=global_step)
                classifier.train()

            if global_step >= max_steps:
                break

    # final save
    if accelerator.is_local_main_process:
        final_path = os.path.join(args.output_dir, "classifier_final.pth")
        torch.save(accelerator.unwrap_model(classifier).state_dict(), final_path)
        if args.push_to_hub:
            repo.push_to_hub(commit_message="Final model")

    accelerator.end_training()

if __name__ == "__main__":
    main()
