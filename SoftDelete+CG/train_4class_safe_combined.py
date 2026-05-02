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


class FourClassDataset(Dataset):
    """
    4-class dataset:
      0: benign (no people)
      1: safe_clothed (person with clothes - combined from safe + safe_failure)
      2: harm_nude (nudity)
      3: harm_color (normal images but with color artifacts/distortions)

    Balances classes by downsampling to the size of the smallest class.
    person_dirs and harm_color_dirs can be single directory (str) or multiple directories (list).
    """
    def __init__(
        self,
        benign_dir: str,
        person_dirs: List[str],
        nude_dir: str,
        harm_color_dirs: List[str],
        transform=None,
        balance_classes: bool = True,
        seed: int = None
    ):
        # Collect all paths and labels per class
        benign_paths = self._collect_images(benign_dir)
        nude_paths = self._collect_images(nude_dir)

        # Handle single or multiple person directories
        if isinstance(person_dirs, str):
            person_dirs = [person_dirs]

        # Handle single or multiple harm_color directories
        if isinstance(harm_color_dirs, str):
            harm_color_dirs = [harm_color_dirs]

        # Collect paths from each person directory
        person_paths_per_dir = []
        for pdir in person_dirs:
            paths = self._collect_images(pdir)
            person_paths_per_dir.append(paths)
            print(f"  Person dir '{pdir}': {len(paths)} samples")

        # Collect paths from each harm_color directory
        harm_color_paths_per_dir = []
        for hdir in harm_color_dirs:
            paths = self._collect_images(hdir)
            harm_color_paths_per_dir.append(paths)
            print(f"  Harm_color dir '{hdir}': {len(paths)} samples")

        # Print original class distribution
        total_person = sum(len(p) for p in person_paths_per_dir)
        total_harm_color = sum(len(p) for p in harm_color_paths_per_dir)
        print(f"\n[Dataset] Original class distribution:")
        print(f"  Class 0 (Benign):     {len(benign_paths)} samples")
        print(f"  Class 1 (Person):     {total_person} samples (from {len(person_dirs)} directories)")
        print(f"  Class 2 (Nude):       {len(nude_paths)} samples")
        print(f"  Class 3 (Harm_color): {total_harm_color} samples (from {len(harm_color_dirs)} directories)")

        if balance_classes:
            # Find minimum class size
            min_size = min(len(benign_paths), total_person, len(nude_paths), total_harm_color)
            print(f"\n[Dataset] Balancing classes to minimum size: {min_size}")

            # Set random seed for reproducibility
            if seed is not None:
                random.seed(seed)

            # Randomly sample from benign and nude
            benign_paths = random.sample(benign_paths, min_size)
            nude_paths = random.sample(nude_paths, min_size)

            # For person: sample equally from each directory
            person_paths = self._sample_from_multiple_dirs(person_paths_per_dir, min_size, "Person")

            # For harm_color: sample equally from each directory
            harm_color_paths = self._sample_from_multiple_dirs(harm_color_paths_per_dir, min_size, "Harm_color")

            print(f"[Dataset] Balanced class distribution:")
            print(f"  Class 0 (Benign):     {len(benign_paths)} samples")
            print(f"  Class 1 (Person):     {len(person_paths)} samples")
            print(f"  Class 2 (Nude):       {len(nude_paths)} samples")
            print(f"  Class 3 (Harm_color): {len(harm_color_paths)} samples")
            print(f"[Dataset] Total samples after balancing: {min_size * 4}\n")
        else:
            # No balancing: just combine all paths
            person_paths = []
            for paths in person_paths_per_dir:
                person_paths.extend(paths)

            harm_color_paths = []
            for paths in harm_color_paths_per_dir:
                harm_color_paths.extend(paths)

        # Combine all paths and labels
        self.paths = []
        self.labels = []

        for path in benign_paths:
            self.paths.append(path)
            self.labels.append(0)

        for path in person_paths:
            self.paths.append(path)
            self.labels.append(1)

        for path in nude_paths:
            self.paths.append(path)
            self.labels.append(2)

        for path in harm_color_paths:
            self.paths.append(path)
            self.labels.append(3)

        self.transform = transform

    def _collect_images(self, directory: str) -> List[str]:
        """Collect image paths from directory."""
        valid_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        paths = []
        for f in sorted(os.listdir(directory)):
            if os.path.splitext(f)[1].lower() in valid_exts:
                paths.append(os.path.join(directory, f))
        return paths

    def _sample_from_multiple_dirs(self, paths_per_dir: List[List[str]], target_size: int, class_name: str) -> List[str]:
        """Sample equally from multiple directories."""
        num_dirs = len(paths_per_dir)
        samples_per_dir = target_size // num_dirs
        remainder = target_size % num_dirs

        sampled_paths = []
        for i, paths in enumerate(paths_per_dir):
            # Add one extra sample to first 'remainder' directories
            n_samples = samples_per_dir + (1 if i < remainder else 0)
            n_samples = min(n_samples, len(paths))  # Don't exceed available
            sampled = random.sample(paths, n_samples)
            sampled_paths.extend(sampled)
            print(f"    {class_name} Dir {i+1}: sampled {n_samples} from {len(paths)} available")

        # If we still need more samples (due to small directories), sample more from larger ones
        if len(sampled_paths) < target_size:
            all_remaining = []
            for paths in paths_per_dir:
                all_remaining.extend([p for p in paths if p not in sampled_paths])
            needed = target_size - len(sampled_paths)
            if len(all_remaining) >= needed:
                sampled_paths.extend(random.sample(all_remaining, needed))

        return sampled_paths

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
        description="Train a 4-class classifier: benign/person/nude/harm_color with DDPM noise injection"
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--benign_data_path", type=str, required=True,
                        help="Directory for benign (no people) images")
    parser.add_argument("--person_data_path", type=str, nargs='+', required=True,
                        help="One or more directories for person (clothed) images. If multiple, samples equally from each.")
    parser.add_argument("--nudity_data_path", type=str, required=True,
                        help="Directory for nudity images")
    parser.add_argument("--harm_color_data_path", type=str, nargs='+', required=True,
                        help="One or more directories for harm_color (color artifact) images. If multiple, samples equally from each.")
    parser.add_argument("--output_dir", type=str, default="four_class_output")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="four_class_project")
    parser.add_argument("--wandb_run_name", type=str, default="run1")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--save_ckpt_freq", type=int, default=100)
    parser.add_argument("--balance_classes", action="store_true", default=True,
                        help="Balance classes by downsampling to minimum class size")
    parser.add_argument("--no_balance_classes", action="store_false", dest="balance_classes",
                        help="Do not balance classes")
    parser.add_argument("--grayscale", action="store_true", default=False,
                        help="Convert all images to grayscale (3-channel gray)")
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
    transform_list = [
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(0.5),
    ]

    # Grayscale option: convert all images to grayscale (3-channel)
    if args.grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=3))
        logger.info("Grayscale mode enabled: all images will be converted to grayscale")

    transform_list.append(transforms.Normalize([0.5]*3, [0.5]*3))
    transform = transforms.Compose(transform_list)

    # Dataset & Dataloader
    full_dataset = FourClassDataset(
        benign_dir=args.benign_data_path,
        person_dirs=args.person_data_path,
        nude_dir=args.nudity_data_path,
        harm_color_dirs=args.harm_color_data_path,
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

    # Model: 4-way classifier
    classifier = load_discriminator(ckpt_path=None, condition=None, eval=False, channel=4, num_classes=4).to(device)

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
            logits = classifier(noisy_latents, norm_ts)  # shape [B, 4]
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
                # Per-class accuracy tracking
                class_correct = [0, 0, 0, 0]
                class_total = [0, 0, 0, 0]

                with torch.no_grad():
                    for vb in val_loader:
                        vimgs = vb["pixel_values"].to(device)
                        vlabels = vb["label"].to(device)

                        # VAE encoding
                        vlat = vae.encode(vimgs).latent_dist.sample() * 0.18215

                        bsz = vlat.shape[0]
                        # Random timestep sampling
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

                        # Classifier forward
                        vnorm_ts = vtimesteps / scheduler.num_train_timesteps
                        vlogits = classifier(noisy_vlat, vnorm_ts)
                        vloss = loss_fn(vlogits, vlabels)
                        preds = vlogits.argmax(dim=-1)

                        val_loss += vloss.item() * bsz
                        val_acc += (preds == vlabels).sum().item()
                        total_val += bsz

                        # Per-class accuracy
                        for c in range(4):
                            mask = vlabels == c
                            class_total[c] += mask.sum().item()
                            class_correct[c] += ((preds == vlabels) & mask).sum().item()

                val_loss /= total_val
                val_acc /= total_val

                # Log per-class accuracy
                class_names = ["Benign", "Person", "Nude", "Harm_color"]
                per_class_acc = {}
                for c in range(4):
                    if class_total[c] > 0:
                        acc = class_correct[c] / class_total[c]
                        per_class_acc[f"val_acc_{class_names[c]}"] = acc
                        logger.info(f"  Class {c} ({class_names[c]}): {acc:.4f} ({class_correct[c]}/{class_total[c]})")

                logger.info(f"[Validation] Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

                if args.use_wandb:
                    log_dict = {"val_loss": val_loss, "val_acc": val_acc}
                    log_dict.update(per_class_acc)
                    wandb.log(log_dict, step=global_step)

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
