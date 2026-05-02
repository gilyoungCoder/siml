#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a z0-space classifier for REPA/SiT latent classifier guidance.

Pipeline (each iteration):
  image -> VAE.encode() -> x0 (clean latent, B,4,32,32)
  -> sample random t ~ U(0,1), inject noise via flow matching
  -> x_t = (1-t)*x0 + t*noise
  -> SiT(x_t, t, y=1000) [frozen, no_grad] -> v_pred
  -> x0_hat = x_t - t * v_pred [detach]
  -> Classifier(x0_hat) -> logits -> CE loss -> optimize classifier

Key differences from SD version (z0_clf_guidance/train.py):
  - Flow matching instead of DDPM (continuous t in [0,1])
  - x0 prediction: x0 = x_t - t*v (vs Tweedie formula)
  - SiT model instead of UNet (no text encoder needed, class-conditional)
  - Latent size: 32x32 from 256px images (vs 64x64 from 512px)
"""

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
from tqdm.auto import tqdm

# REPA model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repa_src"))
from models.sit import SiT_models
from utils import download_model

# Local
from classifiers.latent_classifier import LatentResNet18Classifier
from clf_utils.dataset import ThreeClassFolderDataset
from clf_utils.denoise_utils_flow import predict_x0_from_velocity, inject_noise_flow

logger = get_logger(__name__)

VAE_SCALE = 0.18215
NULL_CLASS = 1000  # SiT unconditional class


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train z0-space classifier for REPA/SiT guidance"
    )
    # Dataset paths
    parser.add_argument("--benign_data_path", type=str, required=True,
                        help="Directory of non-people (class 0) images")
    parser.add_argument("--person_data_path", type=str, nargs="+", required=True,
                        help="Directory(ies) of clothed people (class 1) images")
    parser.add_argument("--nudity_data_path", type=str, required=True,
                        help="Directory of nude (class 2) images")
    # Training
    parser.add_argument("--output_dir", type=str, default="work_dirs/repa_z0_classifier")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--save_ckpt_freq", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--balance_classes", action="store_true", default=True)
    # SiT model
    parser.add_argument("--sit_model", type=str, default="SiT-XL/2")
    parser.add_argument("--sit_ckpt", type=str, default=None,
                        help="Path to SiT checkpoint. None = auto-download pretrained")
    parser.add_argument("--encoder_depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--vae_type", type=str, default="ema", choices=["ema", "mse"])
    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="repa_clf_guidance")
    parser.add_argument("--wandb_run_name", type=str, default="z0_resnet18")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name,
            config=vars(args),
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device

    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # ================================================================
    # Frozen components: VAE, SiT
    # ================================================================
    latent_size = args.resolution // 8  # 32 for 256px

    # VAE
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{args.vae_type}"
    )
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device)

    # SiT model (frozen, for one-step denoising)
    block_kwargs = {"fused_attn": False, "qk_norm": False}
    sit_model = SiT_models[args.sit_model](
        input_size=latent_size,
        num_classes=1000,
        use_cfg=True,
        z_dims=[768],
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    ).to(device)

    if args.sit_ckpt is None:
        logger.info("Downloading pretrained SiT-XL/2 checkpoint...")
        state_dict = download_model("last.pt")
    else:
        state_dict = torch.load(args.sit_ckpt, map_location=device)
        if "ema" in state_dict:
            state_dict = state_dict["ema"]

    sit_model.load_state_dict(state_dict)
    sit_model.requires_grad_(False)
    sit_model.eval()
    logger.info(f"SiT model loaded: {sum(p.numel() for p in sit_model.parameters()):,} params")

    # ================================================================
    # Dataset (256x256 for REPA)
    # ================================================================
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.resolution, args.resolution)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = ThreeClassFolderDataset(
        args.benign_data_path,
        args.person_data_path,
        args.nudity_data_path,
        transform=transform,
        balance=args.balance_classes,
        seed=args.seed,
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
    # Trainable classifier
    # ================================================================
    classifier = LatentResNet18Classifier(
        num_classes=args.num_classes
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
    progress = tqdm(
        range(max_steps), disable=not accelerator.is_local_main_process
    )

    classifier.train()
    while global_step < max_steps:
        for batch in train_loader:
            imgs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            bsz = imgs.shape[0]

            # Step 1: VAE encode -> x0 (clean latent)
            with torch.no_grad():
                x0 = vae.encode(imgs).latent_dist.sample() * VAE_SCALE
                # x0 shape: (B, 4, 32, 32) for 256px images

            # Step 2: Flow matching noise injection
            # Sample random t ~ U(0, 1)
            t = torch.rand(bsz, device=device)
            noise = torch.randn_like(x0)
            t_4d = t.view(-1, 1, 1, 1)
            x_t = inject_noise_flow(x0, noise, t_4d)  # (B, 4, 32, 32)

            # Step 3: SiT one-step denoising (frozen, unconditional)
            with torch.no_grad():
                y_null = torch.full((bsz,), NULL_CLASS, device=device, dtype=torch.long)
                # SiT forward: returns (velocity, projected_features)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    v_pred, _ = sit_model(x_t.float(), t.float(), y=y_null)
                v_pred = v_pred.float()

            # Step 4: One-step x0 prediction (detached)
            x0_hat = predict_x0_from_velocity(x_t, v_pred, t).detach()

            # Step 5: Classify x0_hat
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
                    "train_loss": loss.item(),
                    "train_acc": acc,
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

                # Validation
                classifier.eval()
                val_loss_sum, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for vb in val_loader:
                        vimgs = vb["pixel_values"].to(device)
                        vlabels = vb["label"].to(device)
                        vbsz = vimgs.shape[0]

                        vx0 = vae.encode(vimgs).latent_dist.sample() * VAE_SCALE
                        vt = torch.rand(vbsz, device=device)
                        vnoise = torch.randn_like(vx0)
                        vt_4d = vt.view(-1, 1, 1, 1)
                        vx_t = inject_noise_flow(vx0, vnoise, vt_4d)

                        vy_null = torch.full((vbsz,), NULL_CLASS, device=device, dtype=torch.long)
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            vv_pred, _ = sit_model(vx_t.float(), vt.float(), y=vy_null)
                        vv_pred = vv_pred.float()
                        vx0_hat = predict_x0_from_velocity(vx_t, vv_pred, vt).detach()

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
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    }, step=global_step)
                classifier.train()

            if global_step >= max_steps:
                break

    # Save final model
    if accelerator.is_local_main_process:
        final_path = os.path.join(args.output_dir, "classifier_final.pth")
        torch.save(
            accelerator.unwrap_model(classifier).state_dict(), final_path
        )
        logger.info(f"Final model saved to {final_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
