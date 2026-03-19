"""
JTT Stage 2: Upweighted Training

Load a Stage 1 checkpoint, generate error set, and train final model with upweighting.

Usage:
  python train_jtt_stage2.py \
    --stage1_checkpoint path/to/step_XXX/classifier.pth \
    --upweight_factor 20.0 \
    ...

Or use pre-computed error set:
  python train_jtt_stage2.py \
    --error_set_json path/to/error_set_info.json \
    --upweight_factor 20.0 \
    ...
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
    """

    def __init__(
        self,
        data_dirs: Dict[str, str],
        transform=None,
        num_classes: int = 3,
        seed: int = None,
        minority_ratio: float = 0.05,
    ):
        if seed is not None:
            random.seed(seed)

        self.num_classes = num_classes
        self.transform = transform
        self.minority_ratio = minority_ratio

        self.paths = []
        self.labels = []
        self.group_names = []

        if num_classes == 3:
            if data_dirs.get("benign"):
                for p in self._get_image_paths(data_dirs["benign"]):
                    self.paths.append(p)
                    self.labels.append(0)
                    self.group_names.append("benign")

            # Safe class with minority balancing
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

            # Harm class with minority balancing
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

    def _balance_majority_minority(self, majority_paths: List[str], minority_paths: List[str], class_name: str) -> tuple:
        """Balance majority and minority samples according to minority_ratio."""
        if self.minority_ratio is None or self.minority_ratio >= 1.0:
            return majority_paths, minority_paths

        if len(majority_paths) == 0 or len(minority_paths) == 0:
            return majority_paths, minority_paths

        target_minority = int(len(majority_paths) * self.minority_ratio / (1 - self.minority_ratio))

        if target_minority < len(minority_paths):
            sampled_minority = random.sample(minority_paths, target_minority)
        else:
            sampled_minority = minority_paths

        actual_ratio = len(sampled_minority) / (len(majority_paths) + len(sampled_minority))
        print(f"  [{class_name}] Majority: {len(majority_paths)}, Minority: {len(sampled_minority)} "
              f"(target: {self.minority_ratio:.1%}, actual: {actual_ratio:.1%})")

        return majority_paths, sampled_minority

    def _get_image_paths(self, directory: str) -> List[str]:
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        paths = []
        if not os.path.exists(directory):
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
        print(f"JTT Stage 2 Dataset - {self.num_classes}-class")
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


def find_error_set(model, data_loader, vae, scheduler, device) -> Set[int]:
    """Find misclassified sample indices."""
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

    # Print analysis
    print(f"\n{'='*60}")
    print(f"Error Set Analysis")
    print(f"{'='*60}")
    print(f"Total errors: {len(error_indices)} / {sum(group_totals.values())} ({100*len(error_indices)/sum(group_totals.values()):.2f}%)")
    print(f"\n[Per-Group Error Rates]")
    for group in sorted(group_totals.keys()):
        errors = group_errors.get(group, 0)
        total = group_totals[group]
        rate = 100 * errors / total if total > 0 else 0
        print(f"  {group}: {errors}/{total} ({rate:.2f}%)")
    print(f"{'='*60}\n")

    return error_indices


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
    parser = argparse.ArgumentParser(description="JTT Stage 2: Train with upweighted error set")

    # Stage 1 checkpoint OR pre-computed error set
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Path to Stage 1 checkpoint (classifier.pth)")
    parser.add_argument("--error_set_json", type=str, default=None,
                        help="Path to pre-computed error set JSON (alternative to checkpoint)")

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
                        help="Minority ratio within each class. Paper uses ~5%%.")

    # JTT hyperparameters
    parser.add_argument("--upweight_factor", type=float, default=20.0,
                        help="Upweight factor for error set samples (lambda_up)")

    # Training
    parser.add_argument("--output_dir", type=str, default="jtt_stage2_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    # Checkpointing
    parser.add_argument("--save_ckpt_freq", type=int, default=100)

    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="jtt_stage2")
    parser.add_argument("--wandb_run_name", type=str, default="stage2")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--report_to", type=str, default="tensorboard")

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    if args.stage1_checkpoint is None and args.error_set_json is None:
        raise ValueError("Must provide either --stage1_checkpoint or --error_set_json")

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

    # Get error set
    if args.error_set_json:
        # Load pre-computed error set
        print(f"\nLoading error set from: {args.error_set_json}")
        with open(args.error_set_json, "r") as f:
            error_data = json.load(f)
        error_set = set(error_data["error_indices"])
        print(f"Loaded {len(error_set)} error indices")
    else:
        # Generate error set from Stage 1 checkpoint
        print(f"\nGenerating error set from checkpoint: {args.stage1_checkpoint}")

        # Load Stage 1 model
        stage1_model = load_discriminator(
            ckpt_path=None, condition=None, eval=False, channel=4, num_classes=args.num_classes
        ).to(device)
        stage1_model.load_state_dict(torch.load(args.stage1_checkpoint, map_location=device))

        # Create loader for error set detection
        full_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=4)
        full_loader = accelerator.prepare(full_loader)

        # Find error set
        error_set = find_error_set(stage1_model, full_loader, vae, scheduler, device)

        # Save error set
        error_set_path = os.path.join(args.output_dir, "error_set.json")
        error_data = {
            "stage1_checkpoint": args.stage1_checkpoint,
            "error_indices": list(error_set),
            "error_paths": [dataset.paths[i] for i in error_set],
            "error_groups": [dataset.group_names[i] for i in error_set],
            "total_errors": len(error_set),
            "total_samples": len(dataset),
        }
        with open(error_set_path, "w") as f:
            json.dump(error_data, f, indent=2)
        print(f"Error set saved to: {error_set_path}")

        del stage1_model
        torch.cuda.empty_cache()

    # Create weights for sampling
    weights = []
    for i in range(len(dataset)):
        if i in error_set:
            weights.append(args.upweight_factor)
        else:
            weights.append(1.0)

    # Print effective distribution
    error_weight = len(error_set) * args.upweight_factor
    normal_weight = (len(dataset) - len(error_set)) * 1.0
    total_weight = error_weight + normal_weight

    print(f"\n{'='*60}")
    print(f"JTT Stage 2: Upweighted Training")
    print(f"{'='*60}")
    print(f"Upweight factor (λ_up): {args.upweight_factor}")
    print(f"Error set size: {len(error_set)} / {len(dataset)}")
    print(f"\n[Effective Distribution]")
    print(f"  Normal samples: {len(dataset) - len(error_set)} x 1.0 = {normal_weight:.0f}")
    print(f"  Error samples:  {len(error_set)} x {args.upweight_factor} = {error_weight:.0f}")
    print(f"  Error set effective ratio: {100*error_weight/total_weight:.2f}%")
    print(f"{'='*60}\n")

    # Split train/val (same split as Stage 1)
    total = len(dataset)
    val_size = int(0.1 * total)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(args.seed) if args.seed else None
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    # Weighted sampler for training
    train_indices = train_ds.indices
    train_weights = [weights[i] for i in train_indices]
    sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.train_batch_size, shuffle=False, num_workers=4)

    # Create NEW model from scratch
    model = load_discriminator(ckpt_path=None, condition=None, eval=False, channel=4, num_classes=args.num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # Prepare
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Training loop
    global_step = 0
    best_worst_group_acc = 0.0
    max_steps = args.num_train_epochs * len(train_loader)
    progress = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)

    print(f"\n{'#'*60}")
    print(f"# JTT STAGE 2: UPWEIGHTED TRAINING")
    print(f"# Total epochs: {args.num_train_epochs}")
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

            # Checkpoint
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
                    wandb.log(log_dict)

                # Save best
                if val_results['worst_group_acc'] > best_worst_group_acc:
                    best_worst_group_acc = val_results['worst_group_acc']
                    if accelerator.is_local_main_process:
                        best_dir = os.path.join(args.output_dir, "best_checkpoint")
                        os.makedirs(best_dir, exist_ok=True)
                        torch.save(
                            accelerator.unwrap_model(model).state_dict(),
                            os.path.join(best_dir, "classifier.pth")
                        )
                        logger.info(f"  New best worst-group acc: {best_worst_group_acc:.4f}")

                model.train()

        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"[Epoch {epoch}] Avg Train Loss: {avg_epoch_loss:.4f}")

    # Final save
    if accelerator.is_local_main_process:
        final_path = os.path.join(args.output_dir, "classifier_final.pth")
        torch.save(accelerator.unwrap_model(model).state_dict(), final_path)

        print(f"\n{'='*60}")
        print(f"Stage 2 Training Complete!")
        print(f"{'='*60}")
        print(f"Final model: {final_path}")
        print(f"Best checkpoint: {args.output_dir}/best_checkpoint/classifier.pth")
        print(f"Best worst-group accuracy: {best_worst_group_acc:.4f}")
        print(f"{'='*60}\n")

    accelerator.end_training()


if __name__ == "__main__":
    main()
