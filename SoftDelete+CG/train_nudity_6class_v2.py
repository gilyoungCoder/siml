import argparse
import logging
import math
import os
import random
import copy
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from geo_models.classifier.classifier import create_classifier, discriminator_defaults
from geo_models.classifier.classifier import create_classifier, load_discriminator

from huggingface_hub import HfFolder, whoami, Repository
import yaml
import wandb
from tqdm.auto import tqdm

logger = get_logger(__name__)

# Class mapping for 6-class nudity classifier v2
# 0: benign (no people, landscapes, objects)
# 1: safe_clothed (casual clothes - common safe target)
# 2: harm_nude (completely naked + topless with jeans - MERGED)
# 3: harm_lingerie (underwear/lingerie)
# 4: harm_swimwear (revealing bikini)
# 5: harm_color (color artifacts)

CLASS_NAMES = {
    0: "benign",
    1: "safe_clothed",
    2: "harm_nude",       # merged: completely naked + topless
    3: "harm_lingerie",
    4: "harm_swimwear",
    5: "harm_color",
}

# All harm classes map to safe_clothed
HARM_TO_SAFE = {
    2: 1,  # harm_nude -> safe_clothed
    3: 1,  # harm_lingerie -> safe_clothed
    4: 1,  # harm_swimwear -> safe_clothed
    5: 1,  # harm_color -> safe_clothed
}

HARM_CLASSES = [2, 3, 4, 5]
SAFE_CLASS = 1
BENIGN_CLASS = 0


class Nudity6ClassDatasetV2(Dataset):
    """
    6-class nudity dataset v2:
      0: benign (no people)
      1: safe_clothed (casual clothes)
      2: harm_nude (completely naked + topless with jeans - MERGED)
      3: harm_lingerie (underwear/lingerie)
      4: harm_swimwear (revealing bikini)
      5: harm_color (color artifacts)

    Balances classes by downsampling to the size of the smallest class.
    """
    def __init__(self, class_dirs: Dict[int, List[str]], transform=None, balance_classes=True, seed=None):
        """
        Args:
            class_dirs: Dict mapping class_id -> list of directory paths
                        e.g., {0: ["/path/to/benign"], 2: ["/path/to/naked", "/path/to/topless"], ...}
                        Each class can have multiple directories (for merging)
        """
        self.class_dirs = class_dirs
        self.transform = transform

        # Collect paths for each class
        class_paths = {}
        for class_id, dir_list in class_dirs.items():
            all_paths = []
            for dir_path in dir_list:
                if os.path.exists(dir_path):
                    paths = [os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path))
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                    all_paths.extend(paths)
                else:
                    print(f"Warning: Directory not found for class {class_id} ({CLASS_NAMES.get(class_id, 'unknown')}): {dir_path}")
            class_paths[class_id] = all_paths

        # Print original class distribution
        print(f"\n[Dataset] Original class distribution:")
        for class_id in sorted(class_paths.keys()):
            print(f"  Class {class_id} ({CLASS_NAMES.get(class_id, 'unknown')}): {len(class_paths[class_id])} samples")

        if balance_classes:
            # Find minimum class size (excluding empty classes)
            non_empty_sizes = [len(paths) for paths in class_paths.values() if len(paths) > 0]
            if not non_empty_sizes:
                raise ValueError("All class directories are empty!")
            min_size = min(non_empty_sizes)
            print(f"\n[Dataset] Balancing classes to minimum size: {min_size}")

            # Set random seed for reproducibility
            if seed is not None:
                random.seed(seed)

            # Sample from each class
            for class_id in class_paths:
                if len(class_paths[class_id]) > min_size:
                    class_paths[class_id] = random.sample(class_paths[class_id], min_size)

            print(f"[Dataset] Balanced class distribution:")
            for class_id in sorted(class_paths.keys()):
                print(f"  Class {class_id} ({CLASS_NAMES.get(class_id, 'unknown')}): {len(class_paths[class_id])} samples")
            total = sum(len(paths) for paths in class_paths.values())
            print(f"[Dataset] Total samples after balancing: {total}\n")

        # Combine all paths and labels
        self.paths = []
        self.labels = []
        for class_id, paths in class_paths.items():
            for path in paths:
                self.paths.append(path)
                self.labels.append(class_id)

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
        description="Train a 6-class nudity classifier v2 (merged nude + color artifacts)"
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)

    # Class directories
    parser.add_argument("--benign_dir", type=str, required=True, help="Directory for benign images (class 0)")
    parser.add_argument("--clothed_dir", type=str, required=True, help="Directory for clothed person (class 1)")

    # harm_nude: merged from two directories
    parser.add_argument("--full_nude_dir", type=str, required=True, help="Directory for full nude (merged into class 2)")
    parser.add_argument("--topless_dir", type=str, required=True, help="Directory for topless (merged into class 2)")

    parser.add_argument("--lingerie_dir", type=str, required=True, help="Directory for lingerie (class 3)")
    parser.add_argument("--swimwear_dir", type=str, required=True, help="Directory for swimwear (class 4)")
    parser.add_argument("--color_artifacts_dir", type=str, required=True, help="Directory for color artifacts (class 5)")

    parser.add_argument("--output_dir", type=str, default="nudity_6class_v2_output")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="nudity_6class_v2_project")
    parser.add_argument("--wandb_run_name", type=str, default="run1")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--save_ckpt_freq", type=int, default=100)
    parser.add_argument("--balance_classes", action="store_true", default=True,
                        help="Balance classes by downsampling to minimum class size")
    parser.add_argument("--no_balance_classes", action="store_false", dest="balance_classes",
                        help="Do not balance classes")
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
        repo_name = args.hub_model_id or f"{whoami(HfFolder.get_token())['name']}/{Path(args.output_dir).name}"
        repo = Repository(args.output_dir, clone_from=repo_name)
        with open(os.path.join(args.output_dir, ".gitignore"), "w") as f:
            f.write("checkpoint/**\n")

    # VAE & scheduler
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.to(device)

    scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # Build class directories dict
    # Key difference: class 2 has TWO directories (merged)
    class_dirs = {
        0: [args.benign_dir],
        1: [args.clothed_dir],
        2: [args.full_nude_dir, args.topless_dir],  # MERGED: completely naked + topless
        3: [args.lingerie_dir],
        4: [args.swimwear_dir],
        5: [args.color_artifacts_dir],  # NEW: color artifacts
    }

    # Dataset & Dataloader
    full_dataset = Nudity6ClassDatasetV2(
        class_dirs=class_dirs,
        transform=transform,
        balance_classes=args.balance_classes,
        seed=args.seed,
    )

    # Split train / val 90% / 10%
    total = len(full_dataset)
    print(f"total len: {total}")
    val_size = int(0.1 * total)
    train_size = total - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.train_batch_size, shuffle=False, num_workers=4)

    # Model: 6-way classifier
    classifier = load_discriminator(ckpt_path=None, condition=None, eval=False, channel=4, num_classes=6).to(device)

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
            labels = batch["label"].to(device)

            # VAE encode
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
            logits = classifier(noisy_latents, norm_ts)  # shape [B,6]
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

                    # Save class mapping for reference
                    with open(os.path.join(ckpt_dir, "class_mapping.yaml"), "w") as f:
                        yaml.dump({
                            "class_names": CLASS_NAMES,
                            "harm_to_safe": HARM_TO_SAFE,
                            "harm_classes": HARM_CLASSES,
                            "safe_class": SAFE_CLASS,
                            "benign_class": BENIGN_CLASS,
                        }, f)

                # validation
                classifier.eval()
                val_loss, val_acc = 0.0, 0.0
                total_val = 0

                # Per-class accuracy tracking
                class_correct = {i: 0 for i in range(6)}
                class_total = {i: 0 for i in range(6)}

                with torch.no_grad():
                    for vb in val_loader:
                        vimgs = vb["pixel_values"].to(device)
                        vlabels = vb["label"].to(device)

                        # VAE encoding
                        vlat = vae.encode(vimgs).latent_dist.sample() * 0.18215

                        bsz = vlat.shape[0]
                        vtimesteps = torch.randint(
                            0,
                            scheduler.num_train_timesteps,
                            (bsz,),
                            device=device,
                            dtype=torch.long,
                        )

                        # DDPM forward noise injection
                        noise = torch.randn_like(vlat)
                        alpha_cumprod = scheduler.alphas_cumprod.to(device)
                        alpha_bar = alpha_cumprod[vtimesteps].view(bsz, *([1]*(vlat.ndim-1)))
                        noisy_vlat = torch.sqrt(alpha_bar) * vlat + torch.sqrt(1 - alpha_bar) * noise

                        # classifier forward
                        vnorm_ts = vtimesteps / scheduler.num_train_timesteps
                        vlogits = classifier(noisy_vlat, vnorm_ts)
                        vloss = loss_fn(vlogits, vlabels)
                        preds = vlogits.argmax(dim=-1)

                        val_loss += vloss.item() * bsz
                        val_acc += (preds == vlabels).sum().item()
                        total_val += bsz

                        # Per-class accuracy
                        for i in range(bsz):
                            label = vlabels[i].item()
                            class_total[label] += 1
                            if preds[i] == vlabels[i]:
                                class_correct[label] += 1

                val_loss /= total_val
                val_acc /= total_val

                # Log per-class accuracy
                logger.info(f"[Validation] Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                logger.info("[Per-class Accuracy]")
                for class_id in range(6):
                    if class_total[class_id] > 0:
                        class_acc = class_correct[class_id] / class_total[class_id]
                        logger.info(f"  Class {class_id} ({CLASS_NAMES[class_id]}): {class_acc:.4f} ({class_correct[class_id]}/{class_total[class_id]})")

                if args.use_wandb:
                    log_dict = {"val_loss": val_loss, "val_acc": val_acc}
                    for class_id in range(6):
                        if class_total[class_id] > 0:
                            log_dict[f"val_acc_{CLASS_NAMES[class_id]}"] = class_correct[class_id] / class_total[class_id]
                    wandb.log(log_dict, step=global_step)

                classifier.train()

            if global_step >= max_steps:
                break

    # final save
    if accelerator.is_local_main_process:
        final_path = os.path.join(args.output_dir, "classifier_final.pth")
        torch.save(accelerator.unwrap_model(classifier).state_dict(), final_path)

        # Save class mapping
        with open(os.path.join(args.output_dir, "class_mapping.yaml"), "w") as f:
            yaml.dump({
                "class_names": CLASS_NAMES,
                "harm_to_safe": HARM_TO_SAFE,
                "harm_classes": HARM_CLASSES,
                "safe_class": SAFE_CLASS,
                "benign_class": BENIGN_CLASS,
            }, f)

        if args.push_to_hub:
            repo.push_to_hub(commit_message="Final model")

    accelerator.end_training()

if __name__ == "__main__":
    main()
