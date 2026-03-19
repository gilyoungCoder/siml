#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a z0-space classifier for RAE/DiTDH latent classifier guidance.

Pipeline (each iteration):
  image -> RAE.encode() -> x0 (clean DINOv2 latent, B,768,16,16)
  -> sample random t ~ U(0,1), optionally apply time_dist_shift
  -> x_t = (1-t)*x0 + t*noise
  -> DiTDH(x_t, t, y=1000) [frozen, no_grad] -> v_pred (tensor, NOT tuple!)
  -> x0_hat = x_t - t * v_pred [detach]
  -> Classifier(x0_hat) -> logits -> CE loss -> optimize classifier

Key differences from REPA version (repa_clf_guidance/train_z0.py):
  - RAE DINOv2 encoder instead of VAE (latent: [768,16,16] vs [4,32,32])
  - DiTDH-XL instead of SiT-XL/2 (returns velocity tensor directly, NOT tuple)
  - time_dist_shift for time distribution: t' = shift*t/(1+(shift-1)*t)
  - DINOv2LatentClassifier (MLP) instead of ResNet18
  - Images in [0,1] range (RAE handles normalization internally)
"""

import argparse
import logging
import math
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm

# RAE model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rae_src"))
from stage1 import RAE
from stage2.models.DDT import DiTwDDTHead

# Local
from classifiers.latent_classifier import DINOv2LatentClassifier
from clf_utils.dataset import ThreeClassFolderDataset
from clf_utils.denoise_utils_flow import predict_x0_from_velocity, inject_noise_flow

logger = get_logger(__name__)

NULL_CLASS = 1000  # DiTDH unconditional class


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train z0-space classifier for RAE/DiTDH guidance"
    )
    # Dataset paths
    parser.add_argument("--benign_data_path", type=str, required=True,
                        help="Directory of non-people (class 0) images")
    parser.add_argument("--person_data_path", type=str, nargs="+", required=True,
                        help="Directory(ies) of clothed people (class 1) images")
    parser.add_argument("--nudity_data_path", type=str, required=True,
                        help="Directory of nude (class 2) images")
    # Training
    parser.add_argument("--output_dir", type=str, default="work_dirs/rae_z0_classifier")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_ckpt_freq", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--balance_classes", action="store_true", default=True)
    # RAE encoder
    parser.add_argument("--decoder_config_path", type=str,
                        default="rae_src/configs/decoder/ViTXL",
                        help="Path to RAE decoder config (HuggingFace format)")
    parser.add_argument("--stat_path", type=str, required=True,
                        help="Path to DINOv2 latent normalization stats (stat.pt)")
    # DiTDH model
    parser.add_argument("--ditdh_ckpt", type=str, required=True,
                        help="Path to DiTDH-XL checkpoint")
    # Time distribution shift
    parser.add_argument("--use_time_shift", action="store_true", default=True,
                        help="Apply time_dist_shift to sampled timesteps")
    parser.add_argument("--time_shift_dim", type=int, default=196608,
                        help="shift_dim for time_dist_shift (768*16*16=196608)")
    parser.add_argument("--time_shift_base", type=int, default=4096,
                        help="shift_base for time_dist_shift")
    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="rae_clf_guidance")
    parser.add_argument("--wandb_run_name", type=str, default="z0_dinov2_mlp")
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
    # Time distribution shift
    # ================================================================
    if args.use_time_shift:
        time_dist_shift = math.sqrt(args.time_shift_dim / args.time_shift_base)
        logger.info(f"time_dist_shift = {time_dist_shift:.4f}")
    else:
        time_dist_shift = 1.0

    # ================================================================
    # Frozen components: RAE encoder, DiTDH
    # ================================================================

    # RAE encoder (DINOv2 + normalization stats)
    # decoder_config_path is needed for RAE constructor but decoder won't be used
    rae = RAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path='facebook/dinov2-with-registers-base',
        encoder_input_size=224,
        encoder_params={
            'dinov2_path': 'facebook/dinov2-with-registers-base',
            'normalize': True,
        },
        decoder_config_path=args.decoder_config_path,
        pretrained_decoder_path=None,  # no decoder weights needed for training
        noise_tau=0.,
        reshape_to_2d=True,
        normalization_stat_path=args.stat_path,
    )
    rae.requires_grad_(False)
    rae.eval()
    rae.to(device)
    logger.info("RAE encoder loaded (DINOv2-base with registers)")

    # DiTDH-XL model (frozen, for one-step denoising)
    ditdh = DiTwDDTHead(
        input_size=16,
        patch_size=1,
        in_channels=768,
        hidden_size=[1152, 2048],
        depth=[28, 2],
        num_heads=[16, 16],
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_pos_embed=True,
    ).to(device)

    state_dict = torch.load(args.ditdh_ckpt, map_location="cpu")
    if "ema" in state_dict:
        state_dict = state_dict["ema"]
    ditdh.load_state_dict(state_dict, strict=True)
    ditdh.requires_grad_(False)
    ditdh.eval()
    logger.info(f"DiTDH-XL loaded: {sum(p.numel() for p in ditdh.parameters()):,} params")

    # ================================================================
    # Dataset (images in [0,1] range — RAE handles normalization)
    # ================================================================
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(0.5),
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
    classifier = DINOv2LatentClassifier(
        in_channels=768,
        num_classes=args.num_classes,
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

            # Step 1: RAE encode -> x0 (normalized DINOv2 latent)
            with torch.no_grad():
                x0 = rae.encode(imgs)
                # x0 shape: (B, 768, 16, 16)

            # Step 2: Flow matching noise injection
            t = torch.rand(bsz, device=device)
            # Apply time_dist_shift: t' = shift*t/(1+(shift-1)*t)
            if time_dist_shift > 1.0:
                t = time_dist_shift * t / (1 + (time_dist_shift - 1) * t)
            noise = torch.randn_like(x0)
            t_4d = t.view(-1, 1, 1, 1)
            x_t = inject_noise_flow(x0, noise, t_4d)  # (B, 768, 16, 16)

            # Step 3: DiTDH one-step denoising (frozen, unconditional)
            # NOTE: DiTDH returns velocity tensor directly (NOT tuple like SiT!)
            with torch.no_grad():
                y_null = torch.full((bsz,), NULL_CLASS, device=device, dtype=torch.long)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    v_pred = ditdh(x_t.float(), t.float(), y=y_null)
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

                        vx0 = rae.encode(vimgs)
                        vt = torch.rand(vbsz, device=device)
                        if time_dist_shift > 1.0:
                            vt = time_dist_shift * vt / (1 + (time_dist_shift - 1) * vt)
                        vnoise = torch.randn_like(vx0)
                        vt_4d = vt.view(-1, 1, 1, 1)
                        vx_t = inject_noise_flow(vx0, vnoise, vt_4d)

                        vy_null = torch.full((vbsz,), NULL_CLASS, device=device, dtype=torch.long)
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            vv_pred = ditdh(vx_t.float(), vt.float(), y=vy_null)
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
