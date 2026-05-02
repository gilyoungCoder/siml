#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classifier Accuracy Evaluation Script

Evaluates 3-class classifier accuracy on test/validation data.
Outputs per-class accuracy, confusion matrix, and overall metrics.
"""

import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from diffusers import AutoencoderKL, DDPMScheduler
from geo_models.classifier.classifier import load_discriminator


class ThreeClassTestDataset(Dataset):
    """3-class test dataset for evaluation."""

    def __init__(self, benign_dir, person_dir, nude_dir, transform=None, max_samples_per_class=None):
        self.paths = []
        self.labels = []
        self.transform = transform

        # Collect paths for each class
        for label, data_dir in enumerate([benign_dir, person_dir, nude_dir]):
            if not os.path.exists(data_dir):
                print(f"[WARNING] Directory not found: {data_dir}")
                continue

            files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            if max_samples_per_class and len(files) > max_samples_per_class:
                files = files[:max_samples_per_class]

            for f in files:
                self.paths.append(os.path.join(data_dir, f))
                self.labels.append(label)

        print(f"\n[Dataset] Loaded {len(self.paths)} samples")
        for label, name in enumerate(['Benign', 'Person', 'Nude']):
            count = sum(1 for l in self.labels if l == label)
            print(f"  Class {label} ({name}): {count} samples")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "label": self.labels[idx], "path": self.paths[idx]}


def evaluate_classifier(
    classifier,
    vae,
    scheduler,
    dataloader,
    device,
    num_timesteps_to_eval=[0, 100, 250, 500, 750, 999],
):
    """
    Evaluate classifier accuracy at different timesteps.

    Args:
        classifier: 3-class classifier model
        vae: VAE encoder
        scheduler: DDPM scheduler
        dataloader: Test dataloader
        device: Device to use
        num_timesteps_to_eval: List of timesteps to evaluate at

    Returns:
        Dictionary of results
    """
    classifier.eval()
    vae.eval()

    results = {t: {'preds': [], 'labels': [], 'probs': []} for t in num_timesteps_to_eval}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            imgs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            # VAE encode
            latents = vae.encode(imgs).latent_dist.sample() * 0.18215

            for t in num_timesteps_to_eval:
                # Add noise at timestep t
                bsz = latents.shape[0]
                timesteps = torch.full((bsz,), t, device=device, dtype=torch.long)

                if t > 0:
                    noise = torch.randn_like(latents)
                    alpha_cumprod = scheduler.alphas_cumprod.to(device)
                    alpha_bar = alpha_cumprod[timesteps].view(bsz, *([1]*(latents.ndim-1)))
                    noisy_latents = torch.sqrt(alpha_bar) * latents + torch.sqrt(1 - alpha_bar) * noise
                else:
                    noisy_latents = latents

                # Classifier forward
                norm_ts = timesteps.float() / scheduler.config.num_train_timesteps
                logits = classifier(noisy_latents, norm_ts)
                probs = F.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)

                results[t]['preds'].extend(preds.cpu().numpy().tolist())
                results[t]['labels'].extend(labels.cpu().numpy().tolist())
                results[t]['probs'].extend(probs.cpu().numpy().tolist())

    return results


def print_results(results, class_names=['Benign', 'Person', 'Nude']):
    """Print evaluation results."""

    print("\n" + "="*80)
    print("CLASSIFIER EVALUATION RESULTS")
    print("="*80)

    for t, data in sorted(results.items()):
        preds = np.array(data['preds'])
        labels = np.array(data['labels'])

        # Overall accuracy
        accuracy = (preds == labels).mean() * 100

        print(f"\n{'─'*60}")
        print(f"Timestep t={t}")
        print(f"{'─'*60}")
        print(f"Overall Accuracy: {accuracy:.2f}%")

        # Per-class accuracy
        print("\nPer-class Accuracy:")
        for i, name in enumerate(class_names):
            mask = labels == i
            if mask.sum() > 0:
                class_acc = (preds[mask] == labels[mask]).mean() * 100
                print(f"  {name}: {class_acc:.2f}% ({mask.sum()} samples)")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=class_names, digits=3))

        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        print("Confusion Matrix:")
        print(f"{'':>10}", end="")
        for name in class_names:
            print(f"{name:>10}", end="")
        print()
        for i, name in enumerate(class_names):
            print(f"{name:>10}", end="")
            for j in range(len(class_names)):
                print(f"{cm[i,j]:>10}", end="")
            print()

    print("\n" + "="*80)


def save_confusion_matrix_plot(results, output_dir, class_names=['Benign', 'Person', 'Nude']):
    """Save confusion matrix plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with subplots for each timestep
    n_timesteps = len(results)
    fig, axes = plt.subplots(1, n_timesteps, figsize=(5*n_timesteps, 4))
    if n_timesteps == 1:
        axes = [axes]

    for ax, (t, data) in zip(axes, sorted(results.items())):
        preds = np.array(data['preds'])
        labels = np.array(data['labels'])

        cm = confusion_matrix(labels, preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        accuracy = (preds == labels).mean() * 100
        ax.set_title(f't={t}\nAcc: {accuracy:.1f}%')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved confusion matrix plot: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 3-class classifier accuracy")

    # Model paths
    parser.add_argument("--pretrained_model", type=str, default="CompVis/stable-diffusion-v1-4",
                        help="Pretrained SD model for VAE")
    parser.add_argument("--classifier_ckpt", type=str, required=True,
                        help="Classifier checkpoint path")

    # Data paths
    parser.add_argument("--benign_dir", type=str, required=True,
                        help="Directory with benign (no person) images")
    parser.add_argument("--person_dir", type=str, required=True,
                        help="Directory with person (clothed) images")
    parser.add_argument("--nude_dir", type=str, required=True,
                        help="Directory with nude images")

    # Evaluation settings
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples_per_class", type=int, default=None,
                        help="Max samples per class (default: use all)")
    parser.add_argument("--timesteps", type=str, default="0,100,250,500",
                        help="Comma-separated timesteps to evaluate at")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Output directory for plots")

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Using: {device}")

    # Parse timesteps
    timesteps = [int(t.strip()) for t in args.timesteps.split(',')]
    print(f"[Timesteps] Evaluating at: {timesteps}")

    # Load VAE
    print(f"\n[1/3] Loading VAE from {args.pretrained_model}...")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    vae.requires_grad_(False)
    vae.to(device)
    vae.eval()

    # Load scheduler
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    # Load classifier
    print(f"\n[2/3] Loading classifier from {args.classifier_ckpt}...")
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=3
    ).to(device)
    classifier.eval()
    print(f"  Loaded successfully")

    # Create dataset
    print(f"\n[3/3] Loading test data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    dataset = ThreeClassTestDataset(
        benign_dir=args.benign_dir,
        person_dir=args.person_dir,
        nude_dir=args.nude_dir,
        transform=transform,
        max_samples_per_class=args.max_samples_per_class
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Evaluate
    print(f"\n[Evaluating]...")
    results = evaluate_classifier(
        classifier=classifier,
        vae=vae,
        scheduler=scheduler,
        dataloader=dataloader,
        device=device,
        num_timesteps_to_eval=timesteps
    )

    # Print results
    print_results(results)

    # Save plots
    save_confusion_matrix_plot(results, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
