"""
JTT (Just Train Twice) Training for 3-class Nudity Classifier

Based on: "Just Train Twice: Improving Group Robustness without Training Group Information"
(Liu et al., ICML 2021)

Dataset Structure:
- harm: nude images (majority group - ERM succeeds)
- harm_failure: nude images that ERM misclassifies (minority group - e.g., dark skin tones)
- safe: safe images (majority group - ERM succeeds)
- safe_failure: safe images that ERM misclassifies (minority group - e.g., light skin tones)

JTT upweights the failure cases (minority groups) to improve worst-group accuracy.
"""

import argparse
import logging
import math
import os
import random
import copy
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
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


class JTTDataset(Dataset):
    """
    JTT Dataset for binary classification (harm vs safe) with failure case upweighting.

    Groups:
      - harm (majority): nude images that ERM classifies correctly
      - harm_failure (minority): nude images that ERM misclassifies
      - safe (majority): safe images that ERM classifies correctly
      - safe_failure (minority): safe images that ERM misclassifies

    Labels:
      - 0: safe (non-nude)
      - 1: harm (nude)
    """
    def __init__(
        self,
        harm_dir: str,
        harm_failure_dir: str,
        safe_dir: str,
        safe_failure_dir: str,
        transform=None,
        upweight_factor: float = 1.0,
        seed: int = None,
        balance_base_classes: bool = True,
    ):
        """
        Args:
            harm_dir: Directory containing majority nude images
            harm_failure_dir: Directory containing minority nude images (ERM failures)
            safe_dir: Directory containing majority safe images
            safe_failure_dir: Directory containing minority safe images (ERM failures)
            transform: Image transforms
            upweight_factor: How many times to upsample failure cases (lambda_up in JTT paper)
            seed: Random seed for reproducibility
            balance_base_classes: Whether to balance harm vs safe base classes
        """
        if seed is not None:
            random.seed(seed)

        # Collect paths for each group
        harm_paths = self._get_image_paths(harm_dir)
        harm_failure_paths = self._get_image_paths(harm_failure_dir)
        safe_paths = self._get_image_paths(safe_dir)
        safe_failure_paths = self._get_image_paths(safe_failure_dir)

        print(f"\n{'='*60}")
        print(f"JTT Dataset Configuration")
        print(f"{'='*60}")
        print(f"[Original Distribution]")
        print(f"  harm (majority):         {len(harm_paths):>5} samples")
        print(f"  harm_failure (minority): {len(harm_failure_paths):>5} samples")
        print(f"  safe (majority):         {len(safe_paths):>5} samples")
        print(f"  safe_failure (minority): {len(safe_failure_paths):>5} samples")
        print(f"  Upweight factor (λ_up):  {upweight_factor}")

        # Balance base classes if requested
        if balance_base_classes:
            # Balance majority groups
            min_majority = min(len(harm_paths), len(safe_paths))
            harm_paths = random.sample(harm_paths, min_majority)
            safe_paths = random.sample(safe_paths, min_majority)

            # Balance minority groups (failure cases)
            min_minority = min(len(harm_failure_paths), len(safe_failure_paths))
            harm_failure_paths = random.sample(harm_failure_paths, min_minority)
            safe_failure_paths = random.sample(safe_failure_paths, min_minority)

            print(f"\n[After Balancing]")
            print(f"  harm (majority):         {len(harm_paths):>5} samples")
            print(f"  harm_failure (minority): {len(harm_failure_paths):>5} samples")
            print(f"  safe (majority):         {len(safe_paths):>5} samples")
            print(f"  safe_failure (minority): {len(safe_failure_paths):>5} samples")

        # Build dataset with upweighting
        self.paths = []
        self.labels = []
        self.groups = []  # For tracking: 0=harm, 1=harm_failure, 2=safe, 3=safe_failure
        self.weights = []

        # Add majority samples (weight = 1)
        for path in harm_paths:
            self.paths.append(path)
            self.labels.append(1)  # harm = 1
            self.groups.append(0)  # majority harm
            self.weights.append(1.0)

        for path in safe_paths:
            self.paths.append(path)
            self.labels.append(0)  # safe = 0
            self.groups.append(2)  # majority safe
            self.weights.append(1.0)

        # Add minority samples (weight = upweight_factor)
        # JTT upsamples by repeating samples, but we use weighted sampling
        for path in harm_failure_paths:
            self.paths.append(path)
            self.labels.append(1)  # harm = 1
            self.groups.append(1)  # minority harm (failure)
            self.weights.append(upweight_factor)

        for path in safe_failure_paths:
            self.paths.append(path)
            self.labels.append(0)  # safe = 0
            self.groups.append(3)  # minority safe (failure)
            self.weights.append(upweight_factor)

        self.transform = transform

        # Calculate effective samples after upweighting
        effective_harm = len(harm_paths) + len(harm_failure_paths) * upweight_factor
        effective_safe = len(safe_paths) + len(safe_failure_paths) * upweight_factor

        print(f"\n[Effective Distribution after Upweighting]")
        print(f"  harm total:  {effective_harm:>7.1f} effective samples")
        print(f"    - majority:  {len(harm_paths):>5} x 1.0 = {len(harm_paths):.1f}")
        print(f"    - minority:  {len(harm_failure_paths):>5} x {upweight_factor} = {len(harm_failure_paths) * upweight_factor:.1f}")
        print(f"  safe total:  {effective_safe:>7.1f} effective samples")
        print(f"    - majority:  {len(safe_paths):>5} x 1.0 = {len(safe_paths):.1f}")
        print(f"    - minority:  {len(safe_failure_paths):>5} x {upweight_factor} = {len(safe_failure_paths) * upweight_factor:.1f}")
        print(f"\n  Total samples: {len(self.paths)}")
        print(f"  Total effective: {effective_harm + effective_safe:.1f}")
        print(f"{'='*60}\n")

    def _get_image_paths(self, directory: str) -> List[str]:
        """Get all image paths from a directory."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        paths = []
        for f in sorted(os.listdir(directory)):
            if os.path.splitext(f)[1].lower() in valid_extensions:
                paths.append(os.path.join(directory, f))
        return paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {
            "pixel_values": img,
            "label": self.labels[idx],
            "group": self.groups[idx],
            "weight": self.weights[idx],
        }

    def get_sampler_weights(self):
        """Return weights for WeightedRandomSampler."""
        return self.weights


class JTTDatasetThreeClass(Dataset):
    """
    JTT Dataset for 3-class classification (benign/person/nude) with failure case upweighting.

    This version maintains compatibility with the original 3-class setup while adding
    JTT upweighting for the nudity class failures.
    """
    def __init__(
        self,
        benign_dir: str,
        person_dir: str,
        harm_dir: str,
        harm_failure_dir: str,
        safe_dir: str,
        safe_failure_dir: str,
        transform=None,
        upweight_factor: float = 1.0,
        seed: int = None,
        balance_classes: bool = True,
    ):
        if seed is not None:
            random.seed(seed)

        # Collect paths
        benign_paths = self._get_image_paths(benign_dir)
        person_paths = self._get_image_paths(person_dir)
        harm_paths = self._get_image_paths(harm_dir)
        harm_failure_paths = self._get_image_paths(harm_failure_dir)
        safe_paths = self._get_image_paths(safe_dir)
        safe_failure_paths = self._get_image_paths(safe_failure_dir)

        print(f"\n{'='*60}")
        print(f"JTT 3-Class Dataset Configuration")
        print(f"{'='*60}")
        print(f"[Original Distribution]")
        print(f"  Class 0 (Benign):        {len(benign_paths):>5} samples")
        print(f"  Class 1 (Person/Safe):   {len(safe_paths) + len(safe_failure_paths):>5} samples")
        print(f"    - majority:            {len(safe_paths):>5}")
        print(f"    - failure (minority):  {len(safe_failure_paths):>5}")
        print(f"  Class 2 (Nude/Harm):     {len(harm_paths) + len(harm_failure_paths):>5} samples")
        print(f"    - majority:            {len(harm_paths):>5}")
        print(f"    - failure (minority):  {len(harm_failure_paths):>5}")
        print(f"  Upweight factor (λ_up):  {upweight_factor}")

        # Build dataset
        self.paths = []
        self.labels = []
        self.groups = []
        self.weights = []

        # Class 0: Benign (no failure cases)
        for path in benign_paths:
            self.paths.append(path)
            self.labels.append(0)
            self.groups.append(0)
            self.weights.append(1.0)

        # Class 1: Person/Safe
        for path in safe_paths:
            self.paths.append(path)
            self.labels.append(1)
            self.groups.append(1)  # majority
            self.weights.append(1.0)

        for path in safe_failure_paths:
            self.paths.append(path)
            self.labels.append(1)
            self.groups.append(2)  # minority (failure)
            self.weights.append(upweight_factor)

        # Class 2: Nude/Harm
        for path in harm_paths:
            self.paths.append(path)
            self.labels.append(2)
            self.groups.append(3)  # majority
            self.weights.append(1.0)

        for path in harm_failure_paths:
            self.paths.append(path)
            self.labels.append(2)
            self.groups.append(4)  # minority (failure)
            self.weights.append(upweight_factor)

        self.transform = transform

        print(f"\n[Dataset Summary]")
        print(f"  Total samples: {len(self.paths)}")
        print(f"  Failure cases upweighted by: {upweight_factor}x")
        print(f"{'='*60}\n")

    def _get_image_paths(self, directory: str) -> List[str]:
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        paths = []
        if not os.path.exists(directory):
            return paths
        for f in sorted(os.listdir(directory)):
            if os.path.splitext(f)[1].lower() in valid_extensions:
                paths.append(os.path.join(directory, f))
        return paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {
            "pixel_values": img,
            "label": self.labels[idx],
            "group": self.groups[idx],
            "weight": self.weights[idx],
        }

    def get_sampler_weights(self):
        return self.weights


def parse_args():
    parser = argparse.ArgumentParser(
        description="JTT Training for nudity classifier - upweights failure cases to improve worst-group accuracy"
    )
    # Model
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="Path to pretrained SD model for VAE")

    # Data paths
    parser.add_argument("--harm_dir", type=str, required=True,
                        help="Directory containing majority nude images")
    parser.add_argument("--harm_failure_dir", type=str, required=True,
                        help="Directory containing minority nude images (ERM failures)")
    parser.add_argument("--safe_dir", type=str, required=True,
                        help="Directory containing majority safe images")
    parser.add_argument("--safe_failure_dir", type=str, required=True,
                        help="Directory containing minority safe images (ERM failures)")

    # JTT specific
    parser.add_argument("--upweight_factor", type=float, default=20.0,
                        help="How many times to upweight failure cases (lambda_up in JTT)")
    parser.add_argument("--use_weighted_sampler", action="store_true", default=True,
                        help="Use WeightedRandomSampler instead of loss weighting")

    # Training
    parser.add_argument("--output_dir", type=str, default="jtt_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="L2 regularization (important for JTT)")

    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="jtt_nudity_classifier")
    parser.add_argument("--wandb_run_name", type=str, default="jtt_run")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--report_to", type=str, default="tensorboard")

    # Checkpointing
    parser.add_argument("--save_ckpt_freq", type=int, default=100)
    parser.add_argument("--balance_classes", action="store_true", default=True)

    return parser.parse_args()


def evaluate_by_group(classifier, val_loader, vae, scheduler, loss_fn, device):
    """
    Evaluate model performance by group to compute worst-group accuracy.

    Groups:
      0: harm majority
      1: harm failure (minority)
      2: safe majority
      3: safe failure (minority)
    """
    classifier.eval()

    group_correct = {0: 0, 1: 0, 2: 0, 3: 0}
    group_total = {0: 0, 1: 0, 2: 0, 3: 0}
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            groups = batch["group"]

            # VAE encode
            latents = vae.encode(imgs).latent_dist.sample() * 0.18215

            bsz = latents.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=device)
            noise = torch.randn_like(latents)
            alpha_cumprod = scheduler.alphas_cumprod.to(device)
            alpha_bar = alpha_cumprod[timesteps].view(bsz, *([1]*(latents.ndim-1)))
            noisy_latents = torch.sqrt(alpha_bar) * latents + torch.sqrt(1 - alpha_bar) * noise

            norm_ts = timesteps / scheduler.num_train_timesteps
            logits = classifier(noisy_latents, norm_ts)
            loss = loss_fn(logits, labels)

            preds = logits.argmax(dim=-1)

            total_loss += loss.item() * bsz
            total_samples += bsz

            # Track per-group accuracy
            for i in range(bsz):
                g = groups[i].item()
                group_total[g] += 1
                if preds[i] == labels[i]:
                    group_correct[g] += 1

    # Compute accuracies
    avg_loss = total_loss / total_samples
    avg_acc = sum(group_correct.values()) / sum(group_total.values())

    group_acc = {}
    for g in group_total:
        if group_total[g] > 0:
            group_acc[g] = group_correct[g] / group_total[g]
        else:
            group_acc[g] = 0.0

    # Worst-group accuracy (minimum across groups with samples)
    valid_accs = [acc for g, acc in group_acc.items() if group_total[g] > 0]
    worst_group_acc = min(valid_accs) if valid_accs else 0.0

    return {
        "avg_loss": avg_loss,
        "avg_acc": avg_acc,
        "worst_group_acc": worst_group_acc,
        "group_acc": group_acc,
        "group_total": group_total,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

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

    # JTT Dataset
    full_dataset = JTTDataset(
        harm_dir=args.harm_dir,
        harm_failure_dir=args.harm_failure_dir,
        safe_dir=args.safe_dir,
        safe_failure_dir=args.safe_failure_dir,
        transform=transform,
        upweight_factor=args.upweight_factor,
        seed=args.seed,
        balance_base_classes=args.balance_classes,
    )

    # Split train / val (90% / 10%)
    total = len(full_dataset)
    val_size = int(0.1 * total)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(args.seed) if args.seed else None
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    # Create weighted sampler for training
    if args.use_weighted_sampler:
        # Get weights for training samples only
        train_indices = train_ds.indices
        train_weights = [full_dataset.weights[i] for i in train_indices]
        sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.train_batch_size,
            sampler=sampler,
            num_workers=4,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=4,
        )

    val_loader = DataLoader(val_ds, batch_size=args.train_batch_size, shuffle=False, num_workers=4)

    # Model: binary classifier (harm vs safe)
    classifier = load_discriminator(
        ckpt_path=None,
        condition=None,
        eval=False,
        channel=4,
        num_classes=2  # Binary: safe(0) vs harm(1)
    ).to(device)

    # Optimizer with weight decay (important for JTT)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Prepare with accelerator
    classifier, optimizer, train_loader, val_loader = accelerator.prepare(
        classifier, optimizer, train_loader, val_loader
    )

    # Training loop
    global_step = 0
    max_steps = args.max_train_steps or args.num_train_epochs * len(train_loader)
    progress = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)

    best_worst_group_acc = 0.0

    classifier.train()
    while global_step < max_steps:
        for batch in train_loader:
            imgs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            # VAE encode
            latents = vae.encode(imgs).latent_dist.sample() * 0.18215

            # Random timestep & noise injection
            bsz = latents.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=device)
            noise = torch.randn_like(latents)
            alpha_cumprod = scheduler.alphas_cumprod.to(device)
            alpha_bar = alpha_cumprod[timesteps].view(bsz, *([1]*(latents.ndim-1)))
            noisy_latents = torch.sqrt(alpha_bar) * latents + torch.sqrt(1 - alpha_bar) * noise

            # Classifier forward
            norm_ts = timesteps / scheduler.num_train_timesteps
            logits = classifier(noisy_latents, norm_ts)
            loss = loss_fn(logits, labels)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            progress.update(1)
            progress.set_postfix(step=global_step, loss=loss.item())

            # Checkpoint & validate
            if global_step % args.save_ckpt_freq == 0:
                accelerator.wait_for_everyone()

                # Validation with group-wise metrics
                eval_results = evaluate_by_group(
                    accelerator.unwrap_model(classifier),
                    val_loader, vae, scheduler, loss_fn, device
                )

                logger.info(f"\n[Step {global_step}] Validation Results:")
                logger.info(f"  Average Loss: {eval_results['avg_loss']:.4f}")
                logger.info(f"  Average Acc:  {eval_results['avg_acc']:.4f}")
                logger.info(f"  Worst-Group Acc: {eval_results['worst_group_acc']:.4f}")
                logger.info(f"  Group Accuracies:")
                logger.info(f"    - harm (majority):    {eval_results['group_acc'].get(0, 0):.4f} ({eval_results['group_total'].get(0, 0)} samples)")
                logger.info(f"    - harm_failure:       {eval_results['group_acc'].get(1, 0):.4f} ({eval_results['group_total'].get(1, 0)} samples)")
                logger.info(f"    - safe (majority):    {eval_results['group_acc'].get(2, 0):.4f} ({eval_results['group_total'].get(2, 0)} samples)")
                logger.info(f"    - safe_failure:       {eval_results['group_acc'].get(3, 0):.4f} ({eval_results['group_total'].get(3, 0)} samples)")

                if args.use_wandb:
                    wandb.log({
                        "val_loss": eval_results['avg_loss'],
                        "val_acc": eval_results['avg_acc'],
                        "worst_group_acc": eval_results['worst_group_acc'],
                        "harm_majority_acc": eval_results['group_acc'].get(0, 0),
                        "harm_failure_acc": eval_results['group_acc'].get(1, 0),
                        "safe_majority_acc": eval_results['group_acc'].get(2, 0),
                        "safe_failure_acc": eval_results['group_acc'].get(3, 0),
                    }, step=global_step)

                # Save checkpoint
                if accelerator.is_local_main_process:
                    ckpt_dir = os.path.join(args.output_dir, "checkpoint", f"step_{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(
                        accelerator.unwrap_model(classifier).state_dict(),
                        os.path.join(ckpt_dir, "classifier.pth")
                    )

                    # Save best model based on worst-group accuracy
                    if eval_results['worst_group_acc'] > best_worst_group_acc:
                        best_worst_group_acc = eval_results['worst_group_acc']
                        best_dir = os.path.join(args.output_dir, "best_checkpoint")
                        os.makedirs(best_dir, exist_ok=True)
                        torch.save(
                            accelerator.unwrap_model(classifier).state_dict(),
                            os.path.join(best_dir, "classifier.pth")
                        )
                        logger.info(f"  New best worst-group accuracy: {best_worst_group_acc:.4f}")

                classifier.train()

            if global_step >= max_steps:
                break

    # Final save
    if accelerator.is_local_main_process:
        final_path = os.path.join(args.output_dir, "classifier_final.pth")
        torch.save(accelerator.unwrap_model(classifier).state_dict(), final_path)
        logger.info(f"Final model saved to {final_path}")
        logger.info(f"Best worst-group accuracy: {best_worst_group_acc:.4f}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
