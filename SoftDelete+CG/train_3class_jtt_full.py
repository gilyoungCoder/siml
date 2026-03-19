"""
Full JTT (Just Train Twice) Training Pipeline for Nudity Classifier

Based on: "Just Train Twice: Improving Group Robustness without Training Group Information"
(Liu et al., ICML 2021)

Full Pipeline:
  Stage 1 (Identification):
    - Train ERM model for T epochs
    - Find misclassified samples → Error Set

  Stage 2 (Upweighting):
    - Train new model from scratch
    - Upweight samples in Error Set by λ_up

This prevents the classifier from relying on spurious correlations (e.g., skin tone → nudity)
"""

import argparse
import logging
import math
import os
import random
import copy
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Set

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

import yaml
import wandb
from tqdm.auto import tqdm

logger = get_logger(__name__)


class JTTFullDataset(Dataset):
    """
    Dataset for full JTT pipeline.

    Supports:
      - 3-class classification: benign(0), safe/person(1), harm/nude(2)
      - 2-class classification: safe(0), harm(1)

    Tracks sample paths for Error Set identification.
    """
    def __init__(
        self,
        data_dirs: Dict[str, str],
        transform=None,
        num_classes: int = 3,
        seed: int = None,
        balance_classes: bool = True,
    ):
        """
        Args:
            data_dirs: Dictionary mapping class names to directories
                For 3-class: {"benign": path, "safe": path, "safe_failure": path,
                              "harm": path, "harm_failure": path}
                For 2-class: {"safe": path, "safe_failure": path,
                              "harm": path, "harm_failure": path}
            transform: Image transforms
            num_classes: 2 or 3
            seed: Random seed
            balance_classes: Whether to balance classes
        """
        if seed is not None:
            random.seed(seed)

        self.num_classes = num_classes
        self.transform = transform

        # Collect all image paths
        self.paths = []
        self.labels = []
        self.group_names = []  # For tracking which group each sample belongs to

        if num_classes == 3:
            # Class 0: Benign
            if "benign" in data_dirs and data_dirs["benign"]:
                benign_paths = self._get_image_paths(data_dirs["benign"])
                for p in benign_paths:
                    self.paths.append(p)
                    self.labels.append(0)
                    self.group_names.append("benign")

            # Class 1: Safe (person, clothed)
            safe_paths = []
            if "safe" in data_dirs and data_dirs["safe"]:
                safe_paths.extend(self._get_image_paths(data_dirs["safe"]))
            if "safe_failure" in data_dirs and data_dirs["safe_failure"]:
                safe_failure_paths = self._get_image_paths(data_dirs["safe_failure"])
                for p in safe_failure_paths:
                    self.paths.append(p)
                    self.labels.append(1)
                    self.group_names.append("safe_failure")
            for p in safe_paths:
                self.paths.append(p)
                self.labels.append(1)
                self.group_names.append("safe")

            # Class 2: Harm (nude)
            harm_paths = []
            if "harm" in data_dirs and data_dirs["harm"]:
                harm_paths.extend(self._get_image_paths(data_dirs["harm"]))
            if "harm_failure" in data_dirs and data_dirs["harm_failure"]:
                harm_failure_paths = self._get_image_paths(data_dirs["harm_failure"])
                for p in harm_failure_paths:
                    self.paths.append(p)
                    self.labels.append(2)
                    self.group_names.append("harm_failure")
            for p in harm_paths:
                self.paths.append(p)
                self.labels.append(2)
                self.group_names.append("harm")

        else:  # 2-class
            # Class 0: Safe
            if "safe" in data_dirs and data_dirs["safe"]:
                safe_paths = self._get_image_paths(data_dirs["safe"])
                for p in safe_paths:
                    self.paths.append(p)
                    self.labels.append(0)
                    self.group_names.append("safe")
            if "safe_failure" in data_dirs and data_dirs["safe_failure"]:
                safe_failure_paths = self._get_image_paths(data_dirs["safe_failure"])
                for p in safe_failure_paths:
                    self.paths.append(p)
                    self.labels.append(0)
                    self.group_names.append("safe_failure")

            # Class 1: Harm
            if "harm" in data_dirs and data_dirs["harm"]:
                harm_paths = self._get_image_paths(data_dirs["harm"])
                for p in harm_paths:
                    self.paths.append(p)
                    self.labels.append(1)
                    self.group_names.append("harm")
            if "harm_failure" in data_dirs and data_dirs["harm_failure"]:
                harm_failure_paths = self._get_image_paths(data_dirs["harm_failure"])
                for p in harm_failure_paths:
                    self.paths.append(p)
                    self.labels.append(1)
                    self.group_names.append("harm_failure")

        # Print distribution
        self._print_distribution()

    def _get_image_paths(self, directory: str) -> List[str]:
        """Get all image paths from a directory."""
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
        """Print dataset distribution."""
        from collections import Counter
        group_counts = Counter(self.group_names)
        label_counts = Counter(self.labels)

        print(f"\n{'='*60}")
        print(f"JTT Full Dataset - {self.num_classes}-class")
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
            "idx": idx,  # For tracking
            "path": self.paths[idx],
            "group": self.group_names[idx],
        }


class JTTTrainer:
    """
    Full JTT Training Pipeline.

    Stage 1: Train identification model and find error set
    Stage 2: Train final model with upweighted error set
    """

    def __init__(
        self,
        args,
        accelerator: Accelerator,
        vae: AutoencoderKL,
        scheduler: DDPMScheduler,
        dataset: JTTFullDataset,
        device: torch.device,
    ):
        self.args = args
        self.accelerator = accelerator
        self.vae = vae
        self.scheduler = scheduler
        self.dataset = dataset
        self.device = device

        self.loss_fn = nn.CrossEntropyLoss()
        self.error_set_indices: Set[int] = set()

    def create_model(self) -> nn.Module:
        """Create a new classifier model."""
        return load_discriminator(
            ckpt_path=None,
            condition=None,
            eval=False,
            channel=4,
            num_classes=self.dataset.num_classes,
        ).to(self.device)

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        desc: str = "Training",
    ) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"{desc} Epoch {epoch}", disable=not self.accelerator.is_local_main_process)
        for batch in pbar:
            imgs = batch["pixel_values"].to(self.device)
            labels = batch["label"].to(self.device)

            # VAE encode
            latents = self.vae.encode(imgs).latent_dist.sample() * 0.18215

            # Random timestep & noise
            bsz = latents.shape[0]
            timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bsz,), device=self.device)
            noise = torch.randn_like(latents)
            alpha_cumprod = self.scheduler.alphas_cumprod.to(self.device)
            alpha_bar = alpha_cumprod[timesteps].view(bsz, *([1]*(latents.ndim-1)))
            noisy_latents = torch.sqrt(alpha_bar) * latents + torch.sqrt(1 - alpha_bar) * noise

            # Forward
            norm_ts = timesteps / self.scheduler.num_train_timesteps
            logits = model(noisy_latents, norm_ts)
            loss = self.loss_fn(logits, labels)

            self.accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=loss.item())

        return total_loss / num_batches

    def find_error_set(
        self,
        model: nn.Module,
        data_loader: DataLoader,
    ) -> Set[int]:
        """
        Find samples that the model misclassifies.

        Returns:
            Set of indices for misclassified samples
        """
        model.eval()
        error_indices = set()

        # Track per-group errors for analysis
        group_errors = {}
        group_totals = {}

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Finding Error Set", disable=not self.accelerator.is_local_main_process):
                imgs = batch["pixel_values"].to(self.device)
                labels = batch["label"].to(self.device)
                indices = batch["idx"]
                groups = batch["group"]

                # VAE encode
                latents = self.vae.encode(imgs).latent_dist.sample() * 0.18215

                bsz = latents.shape[0]
                timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bsz,), device=self.device)
                noise = torch.randn_like(latents)
                alpha_cumprod = self.scheduler.alphas_cumprod.to(self.device)
                alpha_bar = alpha_cumprod[timesteps].view(bsz, *([1]*(latents.ndim-1)))
                noisy_latents = torch.sqrt(alpha_bar) * latents + torch.sqrt(1 - alpha_bar) * noise

                norm_ts = timesteps / self.scheduler.num_train_timesteps
                logits = model(noisy_latents, norm_ts)
                preds = logits.argmax(dim=-1)

                # Find misclassified samples
                for i in range(bsz):
                    group = groups[i]
                    group_totals[group] = group_totals.get(group, 0) + 1

                    if preds[i] != labels[i]:
                        error_indices.add(indices[i].item())
                        group_errors[group] = group_errors.get(group, 0) + 1

        # Print error analysis
        print(f"\n{'='*60}")
        print(f"Error Set Analysis")
        print(f"{'='*60}")
        print(f"Total errors: {len(error_indices)} / {sum(group_totals.values())} ({100*len(error_indices)/sum(group_totals.values()):.2f}%)")
        print(f"\n[Per-Group Error Rates]")
        for group in sorted(group_totals.keys()):
            errors = group_errors.get(group, 0)
            total = group_totals[group]
            error_rate = 100 * errors / total if total > 0 else 0
            print(f"  {group}: {errors}/{total} ({error_rate:.2f}%)")
        print(f"{'='*60}\n")

        return error_indices

    def stage1_identification(self) -> Tuple[nn.Module, Set[int]]:
        """
        Stage 1: Train identification model and find error set.

        Returns:
            Tuple of (identification model, error set indices)
        """
        print(f"\n{'#'*60}")
        print(f"# STAGE 1: IDENTIFICATION")
        print(f"# Training ERM model for {self.args.stage1_epochs} epochs")
        print(f"{'#'*60}\n")

        # Create model
        model = self.create_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        # Split data
        total = len(self.dataset)
        val_size = int(0.1 * total)
        train_size = total - val_size

        generator = torch.Generator().manual_seed(self.args.seed) if self.args.seed else None
        train_ds, val_ds = torch.utils.data.random_split(
            self.dataset, [train_size, val_size], generator=generator
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.args.train_batch_size,
            shuffle=False,
            num_workers=4,
        )

        # Prepare with accelerator
        model, optimizer, train_loader, val_loader = self.accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )

        # Train for T epochs
        for epoch in range(1, self.args.stage1_epochs + 1):
            avg_loss = self.train_epoch(model, train_loader, optimizer, epoch, desc="[Stage1]")
            logger.info(f"[Stage 1] Epoch {epoch}/{self.args.stage1_epochs}, Loss: {avg_loss:.4f}")

        # Find error set on TRAINING data (not validation)
        # Need to use full training set for error set identification
        full_loader = DataLoader(
            self.dataset,
            batch_size=self.args.train_batch_size,
            shuffle=False,
            num_workers=4,
        )
        full_loader = self.accelerator.prepare(full_loader)

        error_set = self.find_error_set(self.accelerator.unwrap_model(model), full_loader)

        # Save identification model and error set
        if self.accelerator.is_local_main_process:
            stage1_dir = os.path.join(self.args.output_dir, "stage1")
            os.makedirs(stage1_dir, exist_ok=True)

            # Save model
            torch.save(
                self.accelerator.unwrap_model(model).state_dict(),
                os.path.join(stage1_dir, "identification_model.pth")
            )

            # Save error set
            error_set_data = {
                "indices": list(error_set),
                "paths": [self.dataset.paths[i] for i in error_set],
                "groups": [self.dataset.group_names[i] for i in error_set],
                "total_samples": len(self.dataset),
                "error_rate": len(error_set) / len(self.dataset),
            }
            with open(os.path.join(stage1_dir, "error_set.json"), "w") as f:
                json.dump(error_set_data, f, indent=2)

            logger.info(f"Stage 1 complete. Error set size: {len(error_set)}")

        return model, error_set

    def stage2_upweighting(self, error_set: Set[int]) -> nn.Module:
        """
        Stage 2: Train final model with upweighted error set.

        Args:
            error_set: Set of indices for samples to upweight

        Returns:
            Trained final model
        """
        print(f"\n{'#'*60}")
        print(f"# STAGE 2: UPWEIGHTING")
        print(f"# Upweight factor: {self.args.upweight_factor}")
        print(f"# Error set size: {len(error_set)}")
        print(f"{'#'*60}\n")

        # Create weights for sampling
        weights = []
        for i in range(len(self.dataset)):
            if i in error_set:
                weights.append(self.args.upweight_factor)
            else:
                weights.append(1.0)

        # Calculate effective distribution
        error_weight = len(error_set) * self.args.upweight_factor
        normal_weight = (len(self.dataset) - len(error_set)) * 1.0
        total_weight = error_weight + normal_weight

        print(f"[Effective Distribution]")
        print(f"  Normal samples: {len(self.dataset) - len(error_set)} x 1.0 = {normal_weight:.0f}")
        print(f"  Error samples:  {len(error_set)} x {self.args.upweight_factor} = {error_weight:.0f}")
        print(f"  Error set effective ratio: {100*error_weight/total_weight:.2f}%\n")

        # Split data (same split as stage 1 for consistency)
        total = len(self.dataset)
        val_size = int(0.1 * total)
        train_size = total - val_size

        generator = torch.Generator().manual_seed(self.args.seed) if self.args.seed else None
        train_ds, val_ds = torch.utils.data.random_split(
            self.dataset, [train_size, val_size], generator=generator
        )

        # Create weighted sampler for training
        train_indices = train_ds.indices
        train_weights = [weights[i] for i in train_indices]
        sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.args.train_batch_size,
            shuffle=False,
            num_workers=4,
        )

        # Create NEW model from scratch
        model = self.create_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        # Prepare with accelerator
        model, optimizer, train_loader, val_loader = self.accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )

        # Training loop
        best_worst_group_acc = 0.0
        global_step = 0

        for epoch in range(1, self.args.stage2_epochs + 1):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"[Stage2] Epoch {epoch}", disable=not self.accelerator.is_local_main_process)
            for batch in pbar:
                imgs = batch["pixel_values"].to(self.device)
                labels = batch["label"].to(self.device)

                latents = self.vae.encode(imgs).latent_dist.sample() * 0.18215

                bsz = latents.shape[0]
                timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bsz,), device=self.device)
                noise = torch.randn_like(latents)
                alpha_cumprod = self.scheduler.alphas_cumprod.to(self.device)
                alpha_bar = alpha_cumprod[timesteps].view(bsz, *([1]*(latents.ndim-1)))
                noisy_latents = torch.sqrt(alpha_bar) * latents + torch.sqrt(1 - alpha_bar) * noise

                norm_ts = timesteps / self.scheduler.num_train_timesteps
                logits = model(noisy_latents, norm_ts)
                loss = self.loss_fn(logits, labels)

                self.accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                pbar.set_postfix(loss=loss.item())

                # Save checkpoint
                if global_step % self.args.save_ckpt_freq == 0:
                    if self.accelerator.is_local_main_process:
                        ckpt_dir = os.path.join(self.args.output_dir, "stage2", "checkpoint", f"step_{global_step}")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        torch.save(
                            self.accelerator.unwrap_model(model).state_dict(),
                            os.path.join(ckpt_dir, "classifier.pth")
                        )

            avg_loss = epoch_loss / num_batches

            # Validation
            val_results = self.evaluate_by_group(
                self.accelerator.unwrap_model(model),
                val_loader,
                error_set,
            )

            logger.info(f"\n[Stage 2] Epoch {epoch}/{self.args.stage2_epochs}")
            logger.info(f"  Train Loss: {avg_loss:.4f}")
            logger.info(f"  Val Loss: {val_results['avg_loss']:.4f}")
            logger.info(f"  Val Acc: {val_results['avg_acc']:.4f}")
            logger.info(f"  Worst-Group Acc: {val_results['worst_group_acc']:.4f}")
            logger.info(f"  Per-Group Accuracies:")
            for group, acc in sorted(val_results['group_acc'].items()):
                total = val_results['group_total'].get(group, 0)
                logger.info(f"    {group}: {acc:.4f} ({total} samples)")

            if self.args.use_wandb:
                log_dict = {
                    "stage2/train_loss": avg_loss,
                    "stage2/val_loss": val_results['avg_loss'],
                    "stage2/val_acc": val_results['avg_acc'],
                    "stage2/worst_group_acc": val_results['worst_group_acc'],
                    "stage2/epoch": epoch,
                }
                for group, acc in val_results['group_acc'].items():
                    log_dict[f"stage2/acc_{group}"] = acc
                wandb.log(log_dict)

            # Save best model
            if val_results['worst_group_acc'] > best_worst_group_acc:
                best_worst_group_acc = val_results['worst_group_acc']
                if self.accelerator.is_local_main_process:
                    best_dir = os.path.join(self.args.output_dir, "best_checkpoint")
                    os.makedirs(best_dir, exist_ok=True)
                    torch.save(
                        self.accelerator.unwrap_model(model).state_dict(),
                        os.path.join(best_dir, "classifier.pth")
                    )
                    logger.info(f"  New best worst-group accuracy: {best_worst_group_acc:.4f}")

        # Save final model
        if self.accelerator.is_local_main_process:
            final_path = os.path.join(self.args.output_dir, "classifier_final.pth")
            torch.save(self.accelerator.unwrap_model(model).state_dict(), final_path)
            logger.info(f"\nFinal model saved to {final_path}")
            logger.info(f"Best worst-group accuracy: {best_worst_group_acc:.4f}")

        return model

    def evaluate_by_group(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        error_set: Set[int],
    ) -> Dict:
        """Evaluate model performance by group."""
        model.eval()

        group_correct = {}
        group_total = {}
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["pixel_values"].to(self.device)
                labels = batch["label"].to(self.device)
                groups = batch["group"]

                latents = self.vae.encode(imgs).latent_dist.sample() * 0.18215

                bsz = latents.shape[0]
                timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bsz,), device=self.device)
                noise = torch.randn_like(latents)
                alpha_cumprod = self.scheduler.alphas_cumprod.to(self.device)
                alpha_bar = alpha_cumprod[timesteps].view(bsz, *([1]*(latents.ndim-1)))
                noisy_latents = torch.sqrt(alpha_bar) * latents + torch.sqrt(1 - alpha_bar) * noise

                norm_ts = timesteps / self.scheduler.num_train_timesteps
                logits = model(noisy_latents, norm_ts)
                loss = self.loss_fn(logits, labels)

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
            if group_total[g] > 0:
                group_acc[g] = group_correct.get(g, 0) / group_total[g]
            else:
                group_acc[g] = 0.0

        valid_accs = [acc for g, acc in group_acc.items() if group_total[g] > 0]
        worst_group_acc = min(valid_accs) if valid_accs else 0.0

        return {
            "avg_loss": avg_loss,
            "avg_acc": avg_acc,
            "worst_group_acc": worst_group_acc,
            "group_acc": group_acc,
            "group_total": group_total,
        }

    def run(self):
        """Run the full JTT pipeline."""
        # Stage 1: Identification
        id_model, error_set = self.stage1_identification()

        # Stage 2: Upweighting
        final_model = self.stage2_upweighting(error_set)

        return final_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full JTT Training Pipeline for nudity classifier"
    )
    # Model
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)

    # Data paths
    parser.add_argument("--benign_dir", type=str, default=None,
                        help="Directory for benign images (no person)")
    parser.add_argument("--harm_dir", type=str, required=True,
                        help="Directory for harm/nude images")
    parser.add_argument("--harm_failure_dir", type=str, default=None,
                        help="Directory for harm failure cases (optional)")
    parser.add_argument("--safe_dir", type=str, required=True,
                        help="Directory for safe/person images")
    parser.add_argument("--safe_failure_dir", type=str, default=None,
                        help="Directory for safe failure cases (optional)")

    # Model type
    parser.add_argument("--num_classes", type=int, default=2, choices=[2, 3],
                        help="Number of classes (2: harm/safe, 3: benign/safe/harm)")

    # JTT hyperparameters
    parser.add_argument("--stage1_epochs", type=int, default=10,
                        help="Number of epochs for Stage 1 identification model (T in paper)")
    parser.add_argument("--stage2_epochs", type=int, default=30,
                        help="Number of epochs for Stage 2 final model")
    parser.add_argument("--upweight_factor", type=float, default=20.0,
                        help="Upweight factor for error set samples (lambda_up)")

    # Training
    parser.add_argument("--output_dir", type=str, default="jtt_full_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="jtt_full")
    parser.add_argument("--wandb_run_name", type=str, default="jtt_full_run")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--report_to", type=str, default="tensorboard")

    # Checkpointing
    parser.add_argument("--save_ckpt_freq", type=int, default=200)

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

    # Build data directories dict
    data_dirs = {
        "benign": args.benign_dir,
        "harm": args.harm_dir,
        "harm_failure": args.harm_failure_dir,
        "safe": args.safe_dir,
        "safe_failure": args.safe_failure_dir,
    }

    # Dataset
    dataset = JTTFullDataset(
        data_dirs=data_dirs,
        transform=transform,
        num_classes=args.num_classes,
        seed=args.seed,
    )

    # JTT Trainer
    trainer = JTTTrainer(
        args=args,
        accelerator=accelerator,
        vae=vae,
        scheduler=scheduler,
        dataset=dataset,
        device=device,
    )

    # Run full pipeline
    trainer.run()

    accelerator.end_training()


if __name__ == "__main__":
    main()
