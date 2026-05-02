#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train an IMAGE-SPACE z0 classifier for SD1.4 guidance.

Pipeline (each iteration):
  image -> VAE.encode() -> z0
  -> add noise at random t -> zt
  -> SD1.4 UNet(zt, t, uncond_emb) [frozen] -> noise_pred
  -> z0_hat = Tweedie(zt, noise_pred, alpha_bar) [detach]
  -> VAE.decode(z0_hat) -> x0_hat (3x512x512) [detach]
  -> ImageClassifier(x0_hat) -> logits -> CE loss -> optimize
"""

import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm.auto import tqdm

from models.image_classifier import build_image_classifier
from utils.dataset import ThreeClassFolderDataset
from utils.denoise_utils import get_alpha_bar, predict_z0, inject_noise

logger = get_logger(__name__)

VAE_SCALE = 0.18215


def decode_latent_to_image(vae, z0_hat):
    """Decode latent z0_hat to image space via VAE decoder."""
    x0_hat = vae.decode(z0_hat / VAE_SCALE, return_dict=False)[0]
    # Clamp to [-1, 1] range (VAE output can exceed this)
    x0_hat = x0_hat.clamp(-1, 1)
    return x0_hat


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train image-space z0 classifier for SD1.4 guidance"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument("--benign_data_path", type=str, required=True)
    parser.add_argument("--person_data_path", type=str, nargs="+", required=True)
    parser.add_argument("--nudity_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="work_dirs/z0_img_classifier")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--architecture", type=str, default="resnet18",
                        choices=["resnet18", "vit_b"],
                        help="Classifier architecture: resnet18 or vit_b (ViT-B/16)")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--save_ckpt_freq", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--balance_classes", action="store_true", default=True)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="z0_img_classifier")
    parser.add_argument("--wandb_run_name", type=str, default="run1")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   config=vars(args))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device

    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # ================================================================
    # Frozen: VAE, UNet, text encoder
    # ================================================================
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    unet.requires_grad_(False)
    unet.eval()
    unet.to(device)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    text_encoder.to(device)

    uncond_tokens = tokenizer(
        [""], padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        uncond_emb = text_encoder(**uncond_tokens).last_hidden_state

    scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # ================================================================
    # Dataset
    # ================================================================
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = ThreeClassFolderDataset(
        args.benign_data_path, args.person_data_path, args.nudity_data_path,
        transform=transform, balance=args.balance_classes, seed=args.seed,
    )

    total = len(dataset)
    val_size = int(0.1 * total)
    train_size = total - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.train_batch_size, shuffle=False, num_workers=4
    )

    # ================================================================
    # Trainable classifier (image space, 3ch)
    # ================================================================
    classifier = build_image_classifier(
        architecture=args.architecture,
        num_classes=args.num_classes,
        pretrained_backbone=True,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate)

    classifier, optimizer, train_loader, val_loader = accelerator.prepare(
        classifier, optimizer, train_loader, val_loader
    )

    # ================================================================
    # Training loop
    # ================================================================
    global_step = 0
    max_steps = args.max_train_steps or args.num_train_epochs * len(train_loader)
    progress = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)

    classifier.train()
    while global_step < max_steps:
        for batch in train_loader:
            imgs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            bsz = imgs.shape[0]

            with torch.no_grad():
                # VAE encode -> z0
                z0 = vae.encode(imgs).latent_dist.sample() * VAE_SCALE

                # Noise injection -> zt
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (bsz,), device=device
                )
                noise = torch.randn_like(z0)
                alpha_bar = get_alpha_bar(scheduler, timesteps, device)
                zt = inject_noise(z0, noise, alpha_bar)

                # UNet denoise -> noise_pred
                uncond_batch = uncond_emb.expand(bsz, -1, -1)
                noise_pred = unet(
                    zt, timesteps, encoder_hidden_states=uncond_batch
                ).sample

                # Tweedie -> z0_hat
                z0_hat = predict_z0(zt, noise_pred, alpha_bar)

                # VAE decode -> x0_hat (image space)
                x0_hat = decode_latent_to_image(vae, z0_hat)

            # Classify decoded image
            logits = classifier(x0_hat)
            loss = loss_fn(logits, labels)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            progress.update(1)

            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()
            progress.set_postfix(
                step=global_step, loss=f"{loss.item():.4f}", acc=f"{acc:.3f}"
            )

            if args.use_wandb and global_step % 10 == 0:
                import wandb
                wandb.log({
                    "train_loss": loss.item(), "train_acc": acc,
                }, step=global_step)

            # Checkpoint & validation
            if global_step % args.save_ckpt_freq == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    ckpt_dir = os.path.join(
                        args.output_dir, "checkpoint", f"step_{global_step}"
                    )
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(
                        accelerator.unwrap_model(classifier).state_dict(),
                        os.path.join(ckpt_dir, "classifier.pth"),
                    )

                classifier.eval()
                val_loss_sum, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for vb in val_loader:
                        vimgs = vb["pixel_values"].to(device)
                        vlabels = vb["label"].to(device)
                        vbsz = vimgs.shape[0]

                        vz0 = vae.encode(vimgs).latent_dist.sample() * VAE_SCALE
                        vt = torch.randint(
                            0, scheduler.config.num_train_timesteps,
                            (vbsz,), device=device,
                        )
                        vnoise = torch.randn_like(vz0)
                        valpha = get_alpha_bar(scheduler, vt, device)
                        vzt = inject_noise(vz0, vnoise, valpha)

                        vuncond = uncond_emb.expand(vbsz, -1, -1)
                        vpred = unet(
                            vzt, vt, encoder_hidden_states=vuncond
                        ).sample
                        vz0_hat = predict_z0(vzt, vpred, valpha)
                        vx0_hat = decode_latent_to_image(vae, vz0_hat)

                        vlogits = classifier(vx0_hat)
                        vloss = loss_fn(vlogits, vlabels)
                        vpreds = vlogits.argmax(dim=-1)

                        val_loss_sum += vloss.item() * vbsz
                        val_correct += (vpreds == vlabels).sum().item()
                        val_total += vbsz

                val_loss = val_loss_sum / max(val_total, 1)
                val_acc = val_correct / max(val_total, 1)
                logger.info(
                    f"[Val] step={global_step}, loss={val_loss:.4f}, acc={val_acc:.4f}"
                )
                if args.use_wandb:
                    import wandb
                    wandb.log({
                        "val_loss": val_loss, "val_acc": val_acc,
                    }, step=global_step)
                classifier.train()

            if global_step >= max_steps:
                break

    if accelerator.is_local_main_process:
        torch.save(
            accelerator.unwrap_model(classifier).state_dict(),
            os.path.join(args.output_dir, "classifier_final.pth"),
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
