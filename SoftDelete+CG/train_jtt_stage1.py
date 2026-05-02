"""
JTT Stage 1: Identification Model Training

Train an ERM model and save checkpoints at regular intervals.
After training, examine the validation curves to determine the optimal T (early stopping point).
Then use train_jtt_stage2.py with the selected checkpoint.

Usage:
  1. Run this script to train the identification model
  2. Check WandB/logs to find optimal T (before overfitting)
  3. Run train_jtt_stage2.py with --stage1_checkpoint pointing to the selected checkpoint
"""

import argparse
import logging
import os
import random
import json
from pathlib import Path
from typing import List, Dict, Set

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from geo_models.classifier.classifier import load_discriminator

import yaml
import wandb
from tqdm.auto import tqdm

logger = get_logger(__name__)


class JTTDataset(Dataset):
    """
    Dataset for JTT training with group tracking.

    Supports minority ratio control to simulate realistic spurious correlation scenarios.
    In the JTT paper (Waterbirds), minority groups are ~5% of total data.
    """

    def __init__(
        self,
        data_dirs: Dict[str, str],
        transform=None,
        num_classes: int = 3,
        seed: int = None,
        minority_ratio: float = 0.05,  # Minority를 전체의 몇 %로 할지 (논문: ~5%)
    ):
        """
        Args:
            data_dirs: Dictionary of data directories
            transform: Image transforms
            num_classes: 2 or 3
            seed: Random seed
            minority_ratio: Ratio of minority (failure) samples within each class.
                           e.g., 0.05 means 5% failure, 95% majority
                           Set to None or 1.0 to use all available data without sampling
        """
        if seed is not None:
            random.seed(seed)

        self.num_classes = num_classes
        self.transform = transform
        self.minority_ratio = minority_ratio

        self.paths = []
        self.labels = []
        self.group_names = []

        if num_classes == 3:
            # Class 0: Benign (no failure cases)
            if data_dirs.get("benign"):
                benign_paths = self._get_image_paths(data_dirs["benign"])
                for p in benign_paths:
                    self.paths.append(p)
                    self.labels.append(0)
                    self.group_names.append("benign")

            # Class 1: Safe (person, clothed)
            safe_majority = self._get_image_paths(data_dirs.get("safe", "")) if data_dirs.get("safe") else []
            safe_minority = self._get_image_paths(data_dirs.get("safe_failure", "")) if data_dirs.get("safe_failure") else []
            safe_majority, safe_minority = self._balance_majority_minority(safe_majority, safe_minority, "safe")

            for p in safe_majority:
                self.paths.append(p)
                self.labels.append(1)
                self.group_names.append("safe")
            for p in safe_minority:
                self.paths.append(p)
                self.labels.append(1)
                self.group_names.append("safe_failure")

            # Class 2: Harm (nude)
            harm_majority = self._get_image_paths(data_dirs.get("harm", "")) if data_dirs.get("harm") else []
            harm_minority = self._get_image_paths(data_dirs.get("harm_failure", "")) if data_dirs.get("harm_failure") else []
            harm_majority, harm_minority = self._balance_majority_minority(harm_majority, harm_minority, "harm")

            for p in harm_majority:
                self.paths.append(p)
                self.labels.append(2)
                self.group_names.append("harm")
            for p in harm_minority:
                self.paths.append(p)
                self.labels.append(2)
                self.group_names.append("harm_failure")

        else:  # 2-class
            # Class 0: Safe
            safe_majority = self._get_image_paths(data_dirs.get("safe", "")) if data_dirs.get("safe") else []
            safe_minority = self._get_image_paths(data_dirs.get("safe_failure", "")) if data_dirs.get("safe_failure") else []
            safe_majority, safe_minority = self._balance_majority_minority(safe_majority, safe_minority, "safe")

            for p in safe_majority:
                self.paths.append(p)
                self.labels.append(0)
                self.group_names.append("safe")
            for p in safe_minority:
                self.paths.append(p)
                self.labels.append(0)
                self.group_names.append("safe_failure")

            # Class 1: Harm
            harm_majority = self._get_image_paths(data_dirs.get("harm", "")) if data_dirs.get("harm") else []
            harm_minority = self._get_image_paths(data_dirs.get("harm_failure", "")) if data_dirs.get("harm_failure") else []
            harm_majority, harm_minority = self._balance_majority_minority(harm_majority, harm_minority, "harm")

            for p in harm_majority:
                self.paths.append(p)
                self.labels.append(1)
                self.group_names.append("harm")
            for p in harm_minority:
                self.paths.append(p)
                self.labels.append(1)
                self.group_names.append("harm_failure")

        self._print_distribution()

    def _balance_majority_minority(
        self,
        majority_paths: List[str],
        minority_paths: List[str],
        class_name: str
    ) -> tuple:
        """
        Balance majority and minority samples according to minority_ratio.

        If minority_ratio = 0.05:
          - minority should be 5% of class total
          - majority should be 95% of class total

        Formula: minority / (majority + minority) = minority_ratio
                 minority = majority * minority_ratio / (1 - minority_ratio)
        """
        if self.minority_ratio is None or self.minority_ratio >= 1.0:
            # Use all data without balancing
            return majority_paths, minority_paths

        if len(majority_paths) == 0 or len(minority_paths) == 0:
            return majority_paths, minority_paths

        # Calculate target minority count based on majority count
        # minority / (majority + minority) = ratio
        # minority = majority * ratio / (1 - ratio)
        target_minority = int(len(majority_paths) * self.minority_ratio / (1 - self.minority_ratio))

        # Sample minority to target count (or use all if not enough)
        if target_minority < len(minority_paths):
            sampled_minority = random.sample(minority_paths, target_minority)
        else:
            sampled_minority = minority_paths
            # Optionally: could also reduce majority to maintain ratio
            # target_majority = int(len(minority_paths) * (1 - self.minority_ratio) / self.minority_ratio)
            # majority_paths = random.sample(majority_paths, min(target_majority, len(majority_paths)))

        actual_ratio = len(sampled_minority) / (len(majority_paths) + len(sampled_minority))
        print(f"  [{class_name}] Majority: {len(majority_paths)}, Minority: {len(sampled_minority)} "
              f"(target ratio: {self.minority_ratio:.1%}, actual: {actual_ratio:.1%})")

        return majority_paths, sampled_minority

    def _get_image_paths(self, directory: str) -> List[str]:
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        paths = []
        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            return paths
        for f in sorted(os.listdir(directory)):
            if os.path.splitext(f)[1].lower() in valid_extensions:
                paths.append(os.path.join(directory, f))
        return paths

    def _print_distribution(self):
        from collections import Counter
        group_counts = Counter(self.group_names)
        label_counts = Counter(self.labels)

        print(f"\n{'='*60}")
        print(f"JTT Stage 1 Dataset - {self.num_classes}-class")
        print(f"{'='*60}")
        print(f"[Group Distribution]")
        for group, count in sorted(group_counts.items()):
            print(f"  {group}: {count}")
        print(f"\n[Label Distribution]")
        for label, count in sorted(label_counts.items()):
            print(f"  Class {label}: {count}")
        print(f"\n  Total: {len(self.paths)}")
        print(f"{'='*60}\n")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {
            "pixel_values": img,
            "label": self.labels[idx],
            "idx": idx,
            "path": self.paths[idx],
            "group": self.group_names[idx],
        }


def find_error_set(model, data_loader, vae, scheduler, device) -> Dict:
    """Find misclassified samples and analyze by group."""
    model.eval()
    error_indices = set()
    group_errors = {}
    group_totals = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Finding Error Set"):
            imgs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            indices = batch["idx"]
            groups = batch["group"]

            latents = vae.encode(imgs).latent_dist.sample() * 0.18215

            bsz = latents.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=device)
            noise = torch.randn_like(latents)
            alpha_cumprod = scheduler.alphas_cumprod.to(device)
            alpha_bar = alpha_cumprod[timesteps].view(bsz, *([1]*(latents.ndim-1)))
            noisy_latents = torch.sqrt(alpha_bar) * latents + torch.sqrt(1 - alpha_bar) * noise

            norm_ts = timesteps / scheduler.num_train_timesteps
            logits = model(noisy_latents, norm_ts)
            preds = logits.argmax(dim=-1)

            for i in range(bsz):
                group = groups[i]
                group_totals[group] = group_totals.get(group, 0) + 1

                if preds[i] != labels[i]:
                    error_indices.add(indices[i].item())
                    group_errors[group] = group_errors.get(group, 0) + 1

    total_samples = sum(group_totals.values())
    error_rate = len(error_indices) / total_samples if total_samples > 0 else 0

    return {
        "error_indices": error_indices,
        "group_errors": group_errors,
        "group_totals": group_totals,
        "total_errors": len(error_indices),
        "total_samples": total_samples,
        "error_rate": error_rate,
    }


def evaluate_by_group(model, val_loader, vae, scheduler, loss_fn, device) -> Dict:
    """Evaluate and return per-group metrics."""
    model.eval()
    group_correct = {}
    group_total = {}
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            groups = batch["group"]

            latents = vae.encode(imgs).latent_dist.sample() * 0.18215

            bsz = latents.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=device)
            noise = torch.randn_like(latents)
            alpha_cumprod = scheduler.alphas_cumprod.to(device)
            alpha_bar = alpha_cumprod[timesteps].view(bsz, *([1]*(latents.ndim-1)))
            noisy_latents = torch.sqrt(alpha_bar) * latents + torch.sqrt(1 - alpha_bar) * noise

            norm_ts = timesteps / scheduler.num_train_timesteps
            logits = model(noisy_latents, norm_ts)
            loss = loss_fn(logits, labels)
            preds = logits.argmax(dim=-1)

            total_loss += loss.item() * bsz
            total_samples += bsz

            for i in range(bsz):
                g = groups[i]
                group_total[g] = group_total.get(g, 0) + 1
                if preds[i] == labels[i]:
                    group_correct[g] = group_correct.get(g, 0) + 1

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_acc = sum(group_correct.values()) / total_samples if total_samples > 0 else 0

    group_acc = {}
    for g in group_total:
        group_acc[g] = group_correct.get(g, 0) / group_total[g] if group_total[g] > 0 else 0

    valid_accs = [acc for g, acc in group_acc.items() if group_total[g] > 0]
    worst_group_acc = min(valid_accs) if valid_accs else 0.0

    return {
        "avg_loss": avg_loss,
        "avg_acc": avg_acc,
        "worst_group_acc": worst_group_acc,
        "group_acc": group_acc,
        "group_total": group_total,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="JTT Stage 1: Train identification model")

    # Model
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)

    # Data paths
    parser.add_argument("--benign_dir", type=str, default=None)
    parser.add_argument("--harm_dir", type=str, required=True)
    parser.add_argument("--harm_failure_dir", type=str, default=None)
    parser.add_argument("--safe_dir", type=str, required=True)
    parser.add_argument("--safe_failure_dir", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=3, choices=[2, 3])

    # JTT data balancing
    parser.add_argument("--minority_ratio", type=float, default=0.05,
                        help="Minority (failure) ratio within each class. "
                             "Paper uses ~5%%. Set to 1.0 to use all data without sampling.")

    # Training
    parser.add_argument("--output_dir", type=str, default="jtt_stage1_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    # Checkpointing - save frequently to have options for T selection
    parser.add_argument("--save_ckpt_freq", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--analyze_error_set_freq", type=int, default=500,
                        help="Analyze error set every N steps (for monitoring)")

    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="jtt_stage1")
    parser.add_argument("--wandb_run_name", type=str, default="stage1")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--report_to", type=str, default="tensorboard")

    return parser.parse_args()


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

    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

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

    # Dataset
    data_dirs = {
        "benign": args.benign_dir,
        "harm": args.harm_dir,
        "harm_failure": args.harm_failure_dir,
        "safe": args.safe_dir,
        "safe_failure": args.safe_failure_dir,
    }

    dataset = JTTDataset(
        data_dirs=data_dirs,
        transform=transform,
        num_classes=args.num_classes,
        seed=args.seed,
        minority_ratio=args.minority_ratio,
    )

    # Split train/val
    total = len(dataset)
    val_size = int(0.1 * total)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(args.seed) if args.seed else None
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.train_batch_size, shuffle=False, num_workers=4)

    # For error set analysis (full dataset)
    full_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=4)

    # Model
    model = load_discriminator(ckpt_path=None, condition=None, eval=False, channel=4, num_classes=args.num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # Prepare
    model, optimizer, train_loader, val_loader, full_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, full_loader
    )

    # Training loop
    global_step = 0
    max_steps = args.num_train_epochs * len(train_loader)
    progress = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)

    print(f"\n{'#'*60}")
    print(f"# JTT STAGE 1: IDENTIFICATION MODEL TRAINING")
    print(f"# Total epochs: {args.num_train_epochs}")
    print(f"# Checkpoint freq: every {args.save_ckpt_freq} steps")
    print(f"# Error set analysis freq: every {args.analyze_error_set_freq} steps")
    print(f"{'#'*60}\n")

    model.train()
    for epoch in range(1, args.num_train_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            imgs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            latents = vae.encode(imgs).latent_dist.sample() * 0.18215

            bsz = latents.shape[0]
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=device)
            noise = torch.randn_like(latents)
            alpha_cumprod = scheduler.alphas_cumprod.to(device)
            alpha_bar = alpha_cumprod[timesteps].view(bsz, *([1]*(latents.ndim-1)))
            noisy_latents = torch.sqrt(alpha_bar) * latents + torch.sqrt(1 - alpha_bar) * noise

            norm_ts = timesteps / scheduler.num_train_timesteps
            logits = model(noisy_latents, norm_ts)
            loss = loss_fn(logits, labels)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            progress.update(1)
            progress.set_postfix(epoch=epoch, loss=loss.item())

            # Save checkpoint
            if global_step % args.save_ckpt_freq == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"step_{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(
                        accelerator.unwrap_model(model).state_dict(),
                        os.path.join(ckpt_dir, "classifier.pth")
                    )

                # Validation
                val_results = evaluate_by_group(
                    accelerator.unwrap_model(model), val_loader, vae, scheduler, loss_fn, device
                )

                logger.info(f"\n[Step {global_step}] Validation:")
                logger.info(f"  Loss: {val_results['avg_loss']:.4f}, Acc: {val_results['avg_acc']:.4f}")
                logger.info(f"  Worst-Group Acc: {val_results['worst_group_acc']:.4f}")
                for g, acc in sorted(val_results['group_acc'].items()):
                    logger.info(f"    {g}: {acc:.4f}")

                if args.use_wandb:
                    log_dict = {
                        "val_loss": val_results['avg_loss'],
                        "val_acc": val_results['avg_acc'],
                        "worst_group_acc": val_results['worst_group_acc'],
                        "step": global_step,
                        "epoch": epoch,
                    }
                    for g, acc in val_results['group_acc'].items():
                        log_dict[f"val_acc_{g}"] = acc
                    wandb.log(log_dict, step=global_step)

                model.train()

            # Analyze error set periodically
            if global_step % args.analyze_error_set_freq == 0:
                error_info = find_error_set(
                    accelerator.unwrap_model(model), full_loader, vae, scheduler, device
                )

                logger.info(f"\n[Step {global_step}] Error Set Analysis:")
                logger.info(f"  Total errors: {error_info['total_errors']}/{error_info['total_samples']} ({100*error_info['error_rate']:.2f}%)")
                logger.info(f"  Per-group error rates:")
                for g in sorted(error_info['group_totals'].keys()):
                    errors = error_info['group_errors'].get(g, 0)
                    total = error_info['group_totals'][g]
                    rate = 100 * errors / total if total > 0 else 0
                    logger.info(f"    {g}: {errors}/{total} ({rate:.2f}%)")

                # Save error set info
                if accelerator.is_local_main_process:
                    error_set_path = os.path.join(args.output_dir, "checkpoints", f"step_{global_step}", "error_set_info.json")
                    error_data = {
                        "step": global_step,
                        "epoch": epoch,
                        "total_errors": error_info['total_errors'],
                        "total_samples": error_info['total_samples'],
                        "error_rate": error_info['error_rate'],
                        "group_errors": error_info['group_errors'],
                        "group_totals": error_info['group_totals'],
                        "error_indices": list(error_info['error_indices']),
                    }
                    with open(error_set_path, "w") as f:
                        json.dump(error_data, f, indent=2)

                if args.use_wandb:
                    wandb.log({
                        "error_set_size": error_info['total_errors'],
                        "error_rate": error_info['error_rate'],
                    }, step=global_step)

                model.train()

        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"[Epoch {epoch}] Avg Train Loss: {avg_epoch_loss:.4f}")

    # Final save
    if accelerator.is_local_main_process:
        final_path = os.path.join(args.output_dir, "classifier_final.pth")
        torch.save(accelerator.unwrap_model(model).state_dict(), final_path)

        # Final error set analysis
        error_info = find_error_set(
            accelerator.unwrap_model(model), full_loader, vae, scheduler, device
        )

        final_error_path = os.path.join(args.output_dir, "final_error_set.json")
        error_data = {
            "step": global_step,
            "total_errors": error_info['total_errors'],
            "total_samples": error_info['total_samples'],
            "error_rate": error_info['error_rate'],
            "group_errors": error_info['group_errors'],
            "group_totals": error_info['group_totals'],
            "error_indices": list(error_info['error_indices']),
            "error_paths": [dataset.paths[i] for i in error_info['error_indices']],
            "error_groups": [dataset.group_names[i] for i in error_info['error_indices']],
        }
        with open(final_error_path, "w") as f:
            json.dump(error_data, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Stage 1 Training Complete!")
        print(f"{'='*60}")
        print(f"Checkpoints saved in: {args.output_dir}/checkpoints/")
        print(f"Final model: {final_path}")
        print(f"Final error set: {final_error_path}")
        print(f"\nNext steps:")
        print(f"  1. Check WandB/logs to find optimal T (step with good val acc but before overfitting)")
        print(f"  2. Run train_jtt_stage2.py with --stage1_checkpoint <selected_checkpoint>")
        print(f"{'='*60}\n")

    accelerator.end_training()


if __name__ == "__main__":
    main()
