#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute GradCAM Statistics for 4-Class Nudity Classifier

Computes mean/std/min/max/percentiles of RAW GradCAM heatmap values
for each harm class (class 2: harm_nude, class 3: harm_color).

These statistics are used for absolute normalization during inference,
enabling consistent spatial thresholding across different images.

Usage:
  # Compute stats for harm_nude (class 2)
  python compute_gradcam_statistics_4class.py \
    --data_dir /path/to/nudity_images \
    --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
    --classifier_ckpt ./work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth \
    --output_dir ./gradcam_stats/nudity_4class \
    --target_class 2 \
    --num_samples 1000

  # Compute stats for harm_color (class 3)
  python compute_gradcam_statistics_4class.py \
    --data_dir /path/to/color_artifact_images \
    --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
    --classifier_ckpt ./work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth \
    --output_dir ./gradcam_stats/nudity_4class \
    --target_class 3 \
    --num_samples 1000

Output files:
  - gradcam_stats_harm_nude_class2.json
  - gradcam_stats_harm_color_class3.json
"""

import os
import json
import math
import random
import torch
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

from PIL import Image
from torchvision import transforms

from diffusers import AutoencoderKL, DDPMScheduler
from geo_models.classifier.classifier import load_discriminator
from geo_utils.classifier_interpretability import ClassifierGradCAM


# Class name mapping for 4-class classifier
CLASS_NAMES = {
    0: "benign",
    1: "safe_clothed",
    2: "harm_nude",
    3: "harm_color"
}


# -----------------------------
# Streaming stats (Welford) + reservoir for percentiles
# -----------------------------
class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, x: np.ndarray):
        if x.size == 0:
            return
        self.min = min(self.min, float(np.min(x)))
        self.max = max(self.max, float(np.max(x)))

        for v in x.astype(np.float64, copy=False):
            self.n += 1
            delta = v - self.mean
            self.mean += delta / self.n
            delta2 = v - self.mean
            self.M2 += delta * delta2

    @property
    def var(self):
        return self.M2 / self.n if self.n > 0 else 0.0

    @property
    def std(self):
        return math.sqrt(self.var)


class ReservoirSampler:
    def __init__(self, max_samples: int, seed: int = 42):
        self.max_samples = int(max_samples)
        self.samples = []
        self.seen = 0
        random.seed(seed)

    def update(self, x: np.ndarray):
        for v in x:
            self.seen += 1
            if len(self.samples) < self.max_samples:
                self.samples.append(float(v))
            else:
                j = random.randint(1, self.seen)
                if j <= self.max_samples:
                    self.samples[j - 1] = float(v)

    def get_array(self):
        if len(self.samples) == 0:
            return np.array([], dtype=np.float32)
        return np.array(self.samples, dtype=np.float32)


def parse_args():
    p = ArgumentParser(description="Compute GradCAM statistics for 4-class nudity classifier")

    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory containing images for the target class")

    p.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                   help="Stable Diffusion checkpoint path containing subfolders: vae, scheduler")

    p.add_argument("--classifier_ckpt", type=str, required=True,
                   help="4-class classifier checkpoint path")

    p.add_argument("--output_dir", type=str, default="./gradcam_stats/nudity_4class",
                   help="Output directory for statistics JSON files")

    p.add_argument("--target_class", type=int, required=True,
                   help="Target class for GradCAM (e.g., 2=harm_nude, 3=harm_color)")

    p.add_argument("--num_classes", type=int, default=4,
                   help="Number of classes in classifier (3 for 3-class, 4 for 4-class)")

    p.add_argument("--num_samples", type=int, default=1000,
                   help="Number of images to sample for statistics")

    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for VAE encoding")

    p.add_argument("--gradcam_layer", type=str, default="encoder_model.middle_block.2",
                   help="Target layer for Grad-CAM")

    p.add_argument("--device", type=str, default="cuda",
                   help="cuda / cuda:0 / cpu")

    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")

    # timestep sampling controls
    p.add_argument("--t_min", type=int, default=0,
                   help="Minimum timestep (inclusive) for random sampling")
    p.add_argument("--t_max", type=int, default=None,
                   help="Maximum timestep (inclusive) for random sampling")
    p.add_argument("--fixed_t", type=int, default=None,
                   help="If set, use a fixed timestep t for all samples")

    # percentile estimation
    p.add_argument("--max_percentile_samples", type=int, default=2000000,
                   help="Reservoir sample size for percentile estimation")

    # Top-K percentile for focused statistics (only high-activation pixels)
    p.add_argument("--top_percentile", type=float, default=20.0,
                   help="Only use top K%% of pixels for topk_stats (e.g., 20 = top 20%%)")

    return p.parse_args()


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"pixel_values": img, "path": str(path)}


def list_images(data_dir: Path):
    exts = [".png", ".jpg", ".jpeg", ".webp"]
    paths = []
    for ext in exts:
        paths.extend(list(data_dir.glob(f"**/*{ext}")))
    return sorted(paths)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    class_name = CLASS_NAMES.get(args.target_class, f"class{args.target_class}")

    print("=" * 80)
    print(f"COMPUTING GRADCAM STATISTICS FOR CLASS {args.target_class} ({class_name})")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Pretrained SD path: {args.pretrained_model_name_or_path}")
    print(f"Classifier ckpt: {args.classifier_ckpt}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target class: {args.target_class} ({class_name})")
    print(f"Num samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"GradCAM layer: {args.gradcam_layer}")
    print("=" * 80 + "\n")

    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    # Collect images
    data_dir = Path(args.data_dir)
    image_paths = list_images(data_dir)
    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {data_dir}")

    # Sample subset
    if len(image_paths) > args.num_samples:
        np.random.shuffle(image_paths)
        image_paths = image_paths[:args.num_samples]
        print(f"Sampling {len(image_paths)} images")

    dataset = ImageFolderDataset(image_paths, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load VAE & Scheduler
    print("\n[1/4] Loading VAE & Scheduler...")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae = vae.to(device)
    vae.eval()
    vae.requires_grad_(False)

    scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")
    num_T = int(scheduler.num_train_timesteps)
    if args.t_max is None:
        t_max = num_T - 1
    else:
        t_max = min(int(args.t_max), num_T - 1)
    t_min = max(int(args.t_min), 0)

    print(f"  Scheduler timesteps: {num_T}")
    print(f"  t sampling: {'fixed ' + str(args.fixed_t) if args.fixed_t is not None else f'random [{t_min}, {t_max}]'}")
    print("  VAE & Scheduler loaded")

    # Load classifier
    print(f"\n[2/4] Loading {args.num_classes}-class classifier...")
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=args.num_classes
    ).to(device)
    classifier.eval()
    print(f"  Classifier loaded ({args.num_classes} classes)")

    # Init GradCAM
    print("\n[3/4] Initializing GradCAM...")
    gradcam = ClassifierGradCAM(
        classifier_model=classifier,
        target_layer_name=args.gradcam_layer
    )
    print("  GradCAM initialized")

    # Compute stats
    print("\n[4/4] Computing GradCAM heatmaps...")
    stats = RunningStats()  # pixel-level stats (all pixels)
    topk_stats = RunningStats()  # top-K percentile pixels only
    reservoir = ReservoirSampler(max_samples=args.max_percentile_samples, seed=args.seed)
    topk_reservoir = ReservoirSampler(max_samples=args.max_percentile_samples, seed=args.seed)

    # Sample-level stats: mean of each heatmap (1 scalar per latent)
    sample_level_means = []
    topk_threshold_percentile = 100.0 - args.top_percentile  # e.g., 20% top -> threshold at 80th percentile

    alpha_cumprod = scheduler.alphas_cumprod.to(device)

    total_images = 0
    error_images = 0

    for batch in tqdm(loader, desc="Batches"):
        imgs = batch["pixel_values"].to(device)
        bsz = imgs.shape[0]

        # VAE encode
        with torch.no_grad():
            lat = vae.encode(imgs).latent_dist.sample() * 0.18215

        # Sample timesteps
        if args.fixed_t is not None:
            t_int = torch.full((bsz,), int(args.fixed_t), device=device, dtype=torch.long)
            t_int = torch.clamp(t_int, 0, num_T - 1)
        else:
            t_int = torch.randint(t_min, t_max + 1, (bsz,), device=device, dtype=torch.long)

        # Noise injection
        noise = torch.randn_like(lat)
        alpha_bar = alpha_cumprod[t_int].view(bsz, *([1] * (lat.ndim - 1)))
        noisy_lat = torch.sqrt(alpha_bar) * lat + torch.sqrt(1.0 - alpha_bar) * noise

        # Normalized timestep (t / T)
        t_norm = t_int.to(torch.float32) / float(num_T)

        # GradCAM per-sample
        for i in range(bsz):
            try:
                heatmap, info = gradcam.generate_heatmap(
                    latent=noisy_lat[i:i+1],
                    timestep=t_norm[i:i+1],
                    target_class=args.target_class,
                    normalize=False  # RAW values
                )

                vals = heatmap.detach().flatten().cpu().numpy()
                stats.update(vals)
                reservoir.update(vals)

                # Top-K: only pixels above (100-top_percentile) percentile threshold
                threshold = np.percentile(vals, topk_threshold_percentile)
                topk_vals = vals[vals >= threshold]
                topk_stats.update(topk_vals)
                topk_reservoir.update(topk_vals)

                # Sample-level: mean of this heatmap
                heatmap_mean = float(heatmap.mean().item())
                sample_level_means.append(heatmap_mean)

                total_images += 1

            except Exception as e:
                error_images += 1
                continue

    if stats.n == 0:
        raise RuntimeError("No heatmap values collected. Check GradCAM layer name / classifier forward signature.")

    # Percentiles from reservoir
    sample_vals = reservoir.get_array()
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = {f"p{p}": float(np.percentile(sample_vals, p)) for p in percentiles} if sample_vals.size > 0 else {}

    # Top-K percentiles from reservoir
    topk_sample_vals = topk_reservoir.get_array()
    topk_percentile_values = {f"p{p}": float(np.percentile(topk_sample_vals, p)) for p in percentiles} if topk_sample_vals.size > 0 else {}

    # Sample-level statistics
    image_means_arr = np.array(sample_level_means)
    sample_level_stats = {
        "mean": float(np.mean(image_means_arr)) if len(image_means_arr) > 0 else 0.0,
        "std": float(np.std(image_means_arr)) if len(image_means_arr) > 0 else 0.0,
        "min": float(np.min(image_means_arr)) if len(image_means_arr) > 0 else 0.0,
        "max": float(np.max(image_means_arr)) if len(image_means_arr) > 0 else 0.0,
        "percentiles": {f"p{p}": float(np.percentile(image_means_arr, p)) for p in percentiles} if len(image_means_arr) > 0 else {},
        "num_images": len(image_means_arr)
    }

    # Top-K pixel statistics (only high-activation pixels)
    topk_pixel_stats = {
        "mean": float(topk_stats.mean) if topk_stats.n > 0 else 0.0,
        "std": float(topk_stats.std) if topk_stats.n > 0 else 0.0,
        "min": float(topk_stats.min) if topk_stats.n > 0 else 0.0,
        "max": float(topk_stats.max) if topk_stats.n > 0 else 0.0,
        "percentiles": topk_percentile_values,
        "num_values": int(topk_stats.n),
        "top_percentile": float(args.top_percentile),
        "description": f"Statistics from top {args.top_percentile}% pixels of each heatmap"
    }

    # Prepare output
    out = {
        "mean": float(stats.mean),
        "std": float(stats.std),
        "min": float(stats.min),
        "max": float(stats.max),
        "percentiles": percentile_values,
        "topk": topk_pixel_stats,  # NEW: top-K pixel statistics
        "sample_level": sample_level_stats,
        "num_images_processed": int(total_images),
        "num_images_error": int(error_images),
        "num_values_seen": int(stats.n),
        "percentile_sample_size": int(sample_vals.size),
        "classifier_ckpt": args.classifier_ckpt,
        "target_class": int(args.target_class),
        "target_class_name": class_name,
        "gradcam_layer": args.gradcam_layer,
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "data_dir": str(args.data_dir),
        "timestep_sampling": {
            "mode": "fixed" if args.fixed_t is not None else "random",
            "fixed_t": int(args.fixed_t) if args.fixed_t is not None else None,
            "t_min": int(t_min),
            "t_max": int(t_max),
            "num_train_timesteps": int(num_T),
            "classifier_timestep_format": "t_norm = t_int / num_train_timesteps"
        }
    }

    # Save output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"gradcam_stats_{class_name}_class{args.target_class}.json"
    output_path = output_dir / output_filename

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 80)
    print(f"STATISTICS SUMMARY FOR CLASS {args.target_class} ({class_name})")
    print("=" * 80)
    print(f"Images processed: {total_images} (errors: {error_images})")
    print(f"\n[Pixel-level stats]")
    print(f"Total values seen: {stats.n:,}")
    print(f"Mean: {out['mean']:.6f}")
    print(f"Std:  {out['std']:.6f}")
    print(f"Min:  {out['min']:.6f}")
    print(f"Max:  {out['max']:.6f}")
    if percentile_values:
        print("Percentiles (from reservoir sample):")
        for p in percentiles:
            print(f"  {p:>2}th: {percentile_values[f'p{p}']:.6f}")

    print(f"\n[Top-{args.top_percentile}% pixel stats] (high-activation pixels only)")
    print(f"Total values: {topk_stats.n:,}")
    print(f"Mean: {topk_pixel_stats['mean']:.6f}")
    print(f"Std:  {topk_pixel_stats['std']:.6f}")
    print(f"Min:  {topk_pixel_stats['min']:.6f}")
    print(f"Max:  {topk_pixel_stats['max']:.6f}")
    if topk_percentile_values:
        print("Percentiles:")
        for p in percentiles:
            print(f"  {p:>2}th: {topk_percentile_values[f'p{p}']:.6f}")

    print(f"\n[Sample-level stats] (mean of each heatmap)")
    print(f"Mean: {sample_level_stats['mean']:.6f}")
    print(f"Std:  {sample_level_stats['std']:.6f}")
    print(f"Min:  {sample_level_stats['min']:.6f}")
    print(f"Max:  {sample_level_stats['max']:.6f}")
    if sample_level_stats['percentiles']:
        print("Percentiles:")
        for p in percentiles:
            print(f"  {p:>2}th: {sample_level_stats['percentiles'][f'p{p}']:.6f}")

    print(f"\nSaved to: {output_path}")
    print("=" * 80 + "\n")

    # Inference hint
    print("=" * 80)
    print("USAGE IN INFERENCE")
    print("=" * 80)
    print("[All pixels] Normalize RAW heatmap values with:")
    print(f"  z = (value - {out['mean']:.4f}) / {out['std']:.4f}")
    print("Then map z -> [0,1] with Gaussian CDF.")
    print(f"\n[Top-{args.top_percentile}% pixels] For spatial masking (recommended):")
    print(f"  z = (value - {topk_pixel_stats['mean']:.4f}) / {topk_pixel_stats['std']:.4f}")
    print("  Use this for more accurate harmful region detection.")
    print("\n[Sample-level] For monitoring P(harm):")
    print(f"  heatmap_mean = heatmap.mean()")
    print(f"  z = (heatmap_mean - {sample_level_stats['mean']:.4f}) / {sample_level_stats['std']:.4f}")
    print(f"  p_harm = CDF(z)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
