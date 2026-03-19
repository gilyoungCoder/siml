#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a z0-space 9-class I2P classifier for SD1.4 guidance.

9-class structure:
  0: benign (imagenet)
  1: harm0, 2: safe0
  3: harm1, 4: safe1
  5: harm2, 6: safe2
  7: harm3, 8: safe3

Pipeline (each iteration):
  image -> VAE.encode() -> z0
  -> add noise at random t -> zt
  -> SD1.4 UNet(zt, t, uncond_emb) [frozen] -> noise_pred
  -> z0_hat = Tweedie(zt, noise_pred, alpha_bar) [detach]
  -> Classifier(z0_hat) -> logits -> CE loss -> optimize
"""

import argparse
import logging
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm.auto import tqdm
import yaml

from models.latent_classifier import LatentResNet18Classifier
from utils.denoise_utils import get_alpha_bar, predict_z0, inject_noise

logger = get_logger(__name__)

# ================================================================
# 9-class structure
# ================================================================
CLASS_NAMES = {
    0: "benign",
    1: "harm0", 2: "safe0",
    3: "harm1", 4: "safe1",
    5: "harm2", 6: "safe2",
    7: "harm3", 8: "safe3",
}

HARM_TO_SAFE = {1: 2, 3: 4, 5: 6, 7: 8}
HARM_CLASSES = [1, 3, 5, 7]
SAFE_CLASSES = [2, 4, 6, 8]
BENIGN_CLASS = 0
NUM_CLASSES = 9


# ================================================================
# Dataset
# ================================================================
class I2P9ClassDataset(Dataset):
    """
    9-class I2P dataset:
      0: benign (imagenet)
      1-8: harm/safe pairs (harm0, safe0, harm1, safe1, ...)

    Balances classes by downsampling to the smallest class.
    """
    EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

    def __init__(self, class_dirs, transform=None, balance_classes=True, seed=42):
        self.transform = transform
        class_paths = {}
        for class_id, dir_path in class_dirs.items():
            if os.path.exists(dir_path):
                paths = sorted([
                    os.path.join(dir_path, f) for f in os.listdir(dir_path)
                    if os.path.splitext(f)[1].lower() in self.EXTENSIONS
                ])
                class_paths[class_id] = paths
            else:
                print(f"  Warning: not found class {class_id} ({CLASS_NAMES.get(class_id, '?')}): {dir_path}")
                class_paths[class_id] = []

        print("\n[Dataset] Original distribution:")
        for cid in sorted(class_paths):
            print(f"  Class {cid} ({CLASS_NAMES.get(cid, '?')}): {len(class_paths[cid])}")

        if balance_classes:
            non_empty = [len(p) for p in class_paths.values() if p]
            if not non_empty:
                raise ValueError("All class directories are empty!")
            min_size = min(non_empty)
            rng = random.Random(seed)
            for cid in class_paths:
                if len(class_paths[cid]) > min_size:
                    class_paths[cid] = rng.sample(class_paths[cid], min_size)
            total = sum(len(p) for p in class_paths.values())
            print(f"[Dataset] Balanced to {min_size}/class, total={total}\n")

        self.paths = []
        self.labels = []
        for cid, paths in class_paths.items():
            for p in paths:
                self.paths.append(p)
                self.labels.append(cid)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "label": self.labels[idx]}


# ================================================================
# Arguments
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train z0-space 9-class I2P classifier"
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--benign_dir", type=str, required=True,
                        help="Directory for benign images (imagenet)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Base dir with harm0/ safe0/ harm1/ safe1/ ... subdirs")
    parser.add_argument("--concept_name", type=str, required=True,
                        help="Concept name (harassment, hate, illegal, ...)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=25000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_ckpt_freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--balance_classes", action="store_true", default=True)
    parser.add_argument("--no_balance_classes", action="store_false",
                        dest="balance_classes")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str,
                        default="z0_i2p_9class_classifier")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="wandb")
    return parser.parse_args()


# ================================================================
# Main
# ================================================================
def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = f"./work_dirs/z0_{args.concept_name}_9class"
    if args.wandb_run_name is None:
        args.wandb_run_name = f"z0_{args.concept_name}_9class"

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[Z0] Training 9-class classifier for: {args.concept_name.upper()}")
    print(f"{'='*60}\n")

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
    # Frozen components
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

    # Unconditional text embedding
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

    data_dir = Path(args.data_dir)
    class_dirs = {
        0: args.benign_dir,
        1: str(data_dir / "harm0"), 2: str(data_dir / "safe0"),
        3: str(data_dir / "harm1"), 4: str(data_dir / "safe1"),
        5: str(data_dir / "harm2"), 6: str(data_dir / "safe2"),
        7: str(data_dir / "harm3"), 8: str(data_dir / "safe3"),
    }

    full_dataset = I2P9ClassDataset(
        class_dirs=class_dirs,
        transform=transform,
        balance_classes=args.balance_classes,
        seed=args.seed,
    )

    total = len(full_dataset)
    val_size = int(0.1 * total)
    train_size = total - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.train_batch_size, shuffle=False, num_workers=4
    )

    # ================================================================
    # Classifier (ResNet18, 9 classes, no timestep)
    # ================================================================
    classifier = LatentResNet18Classifier(num_classes=NUM_CLASSES).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate)

    classifier, optimizer, train_loader, val_loader = accelerator.prepare(
        classifier, optimizer, train_loader, val_loader
    )

    # ================================================================
    # Training loop
    # ================================================================
    global_step = 0
    max_steps = args.max_train_steps
    progress = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)

    classifier.train()
    while global_step < max_steps:
        for batch in train_loader:
            imgs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            bsz = imgs.shape[0]

            # VAE encode -> z0
            with torch.no_grad():
                z0 = vae.encode(imgs).latent_dist.sample() * 0.18215

            # Noise injection -> zt
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (bsz,), device=device
            )
            noise = torch.randn_like(z0)
            alpha_bar = get_alpha_bar(scheduler, timesteps, device)
            zt = inject_noise(z0, noise, alpha_bar)

            # UNet one-step denoising (frozen)
            with torch.no_grad():
                uncond_batch = uncond_emb.expand(bsz, -1, -1)
                noise_pred = unet(
                    zt, timesteps, encoder_hidden_states=uncond_batch
                ).sample

            # Tweedie -> z0_hat (detached)
            z0_hat = predict_z0(zt, noise_pred, alpha_bar).detach()

            # Classify
            logits = classifier(z0_hat)
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
                    with open(os.path.join(ckpt_dir, "class_mapping.yaml"), "w") as f:
                        yaml.dump({
                            "concept_name": args.concept_name,
                            "class_names": CLASS_NAMES,
                            "harm_to_safe": HARM_TO_SAFE,
                            "harm_classes": HARM_CLASSES,
                            "safe_classes": SAFE_CLASSES,
                            "benign_class": BENIGN_CLASS,
                        }, f)

                # Validation
                classifier.eval()
                val_loss_sum, val_correct, val_total = 0.0, 0, 0
                class_correct = {i: 0 for i in range(NUM_CLASSES)}
                class_total = {i: 0 for i in range(NUM_CLASSES)}

                with torch.no_grad():
                    for vb in val_loader:
                        vimgs = vb["pixel_values"].to(device)
                        vlabels = vb["label"].to(device)
                        vbsz = vimgs.shape[0]

                        vz0 = vae.encode(vimgs).latent_dist.sample() * 0.18215
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
                        vz0_hat = predict_z0(vzt, vpred, valpha).detach()

                        vlogits = classifier(vz0_hat)
                        vloss = loss_fn(vlogits, vlabels)
                        vpreds = vlogits.argmax(dim=-1)

                        val_loss_sum += vloss.item() * vbsz
                        val_correct += (vpreds == vlabels).sum().item()
                        val_total += vbsz

                        for i in range(vbsz):
                            lbl = vlabels[i].item()
                            class_total[lbl] += 1
                            if vpreds[i] == vlabels[i]:
                                class_correct[lbl] += 1

                val_loss = val_loss_sum / max(val_total, 1)
                val_acc = val_correct / max(val_total, 1)

                logger.info(f"[Val] step={global_step}, loss={val_loss:.4f}, acc={val_acc:.4f}")
                for cid in range(NUM_CLASSES):
                    if class_total[cid] > 0:
                        cacc = class_correct[cid] / class_total[cid]
                        logger.info(f"  {CLASS_NAMES[cid]}: {cacc:.4f} ({class_correct[cid]}/{class_total[cid]})")

                if args.use_wandb:
                    import wandb
                    log_dict = {"val_loss": val_loss, "val_acc": val_acc}
                    for cid in range(NUM_CLASSES):
                        if class_total[cid] > 0:
                            log_dict[f"val_acc_{CLASS_NAMES[cid]}"] = (
                                class_correct[cid] / class_total[cid]
                            )
                    wandb.log(log_dict, step=global_step)

                classifier.train()

            if global_step >= max_steps:
                break

    # Final save
    if accelerator.is_local_main_process:
        torch.save(
            accelerator.unwrap_model(classifier).state_dict(),
            os.path.join(args.output_dir, "classifier_final.pth"),
        )
        with open(os.path.join(args.output_dir, "class_mapping.yaml"), "w") as f:
            yaml.dump({
                "concept_name": args.concept_name,
                "class_names": CLASS_NAMES,
                "harm_to_safe": HARM_TO_SAFE,
                "harm_classes": HARM_CLASSES,
                "safe_classes": SAFE_CLASSES,
                "benign_class": BENIGN_CLASS,
            }, f)

    accelerator.end_training()
    print(f"\n{'='*60}")
    print(f"[Z0] Complete: {args.concept_name.upper()}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
