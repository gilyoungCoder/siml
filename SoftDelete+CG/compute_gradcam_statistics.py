#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute GradCAM Statistics from Training Dataset (Classifier-Consistent)

This script computes mean/std/min/max/percentiles of RAW GradCAM heatmap values
on the SAME input distribution used by the classifier during training/inference:

image -> VAE encode -> sample t -> add DDPM noise -> classifier(noisy_lat, t/T) -> GradCAM

Usage:
  python compute_gradcam_statistics.py \
    --data_dir ./data/harmful_images \
    --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
    --classifier_ckpt ./work_dirs/classifier.pth \
    --output_file ./gradcam_stats.json \
    --num_samples 1000

Optional:
  --t_min 0 --t_max 999    (t sampling range)
  --fixed_t 500            (use fixed t instead of random)
  --batch_size 8
  --max_percentile_samples 2000000
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
        # x: 1D numpy array
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
        # x: 1D numpy array
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
    p = ArgumentParser(description="Compute GradCAM statistics on classifier-consistent distribution")

    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory containing images (e.g., harmful GT images)")

    # IMPORTANT: use same checkpoint root as evaluate_classifier.py
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                   help="Stable Diffusion checkpoint path containing subfolders: vae, scheduler")

    p.add_argument("--classifier_ckpt", type=str, required=True,
                   help="Classifier checkpoint path")

    p.add_argument("--output_file", type=str, default="./gradcam_stats.json",
                   help="Output JSON file to save statistics")

    p.add_argument("--num_samples", type=int, default=1000,
                   help="Number of images to sample for statistics")

    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for VAE encoding + noise injection (GradCAM is per-sample inside batch)")

    p.add_argument("--harmful_class", type=int, default=2,
                   help="Target class index for GradCAM (e.g., 2=nude)")

    p.add_argument("--num_classes", type=int, default=3,
                   help="Number of classes in classifier (3 for nudity, 9 for violence)")

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
                   help="Maximum timestep (inclusive) for random sampling; default = num_train_timesteps-1")
    p.add_argument("--fixed_t", type=int, default=None,
                   help="If set, use a fixed timestep t for all samples (overrides random sampling)")

    # percentile estimation
    p.add_argument("--max_percentile_samples", type=int, default=2000000,
                   help="Reservoir sample size for percentile estimation (memory-safe)")

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

    print("=" * 80)
    print("COMPUTING GRADCAM STATISTICS (CLASSIFIER-CONSISTENT)")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Pretrained SD path: {args.pretrained_model_name_or_path}")
    print(f"Classifier ckpt: {args.classifier_ckpt}")
    print(f"Output file: {args.output_file}")
    print(f"Num samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Harmful class: {args.harmful_class}")
    print(f"GradCAM layer: {args.gradcam_layer}")
    print("=" * 80 + "\n")

    # 1) Transform: match evaluate_classifier.py
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    # 2) Collect images
    data_dir = Path(args.data_dir)
    image_paths = list_images(data_dir)
    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {data_dir}")

    # sample subset
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

    # 3) Load VAE + Scheduler (same as evaluate_classifier.py)
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

    # 4) Load classifier
    print("\n[2/4] Loading classifier...")
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=args.num_classes
    ).to(device)
    classifier.eval()
    print(f"  Classifier loaded ({args.num_classes} classes)")

    # 5) Init GradCAM
    print("\n[3/4] Initializing GradCAM...")
    gradcam = ClassifierGradCAM(
        classifier_model=classifier,
        target_layer_name=args.gradcam_layer
    )
    print("  GradCAM initialized")

    # 6) Compute stats
    print("\n[4/4] Computing GradCAM heatmaps...")
    stats = RunningStats()
    reservoir = ReservoirSampler(max_samples=args.max_percentile_samples, seed=args.seed)

    alpha_cumprod = scheduler.alphas_cumprod.to(device)

    total_images = 0
    error_images = 0

    for batch in tqdm(loader, desc="Batches"):
        imgs = batch["pixel_values"].to(device)  # [B,3,512,512]
        bsz = imgs.shape[0]

        # VAE encode (no_grad OK)
        with torch.no_grad():
            lat = vae.encode(imgs).latent_dist.sample() * 0.18215  # [B,4,H/8,W/8]

        # sample timesteps (int for noise injection)
        if args.fixed_t is not None:
            t_int = torch.full((bsz,), int(args.fixed_t), device=device, dtype=torch.long)
            t_int = torch.clamp(t_int, 0, num_T - 1)
        else:
            t_int = torch.randint(t_min, t_max + 1, (bsz,), device=device, dtype=torch.long)

        # noise injection (match evaluate_classifier.py)
        noise = torch.randn_like(lat)
        alpha_bar = alpha_cumprod[t_int].view(bsz, *([1] * (lat.ndim - 1)))  # [B,1,1,1]
        noisy_lat = torch.sqrt(alpha_bar) * lat + torch.sqrt(1.0 - alpha_bar) * noise  # [B,4,h,w]

        # classifier conditioning uses normalized timestep t/T (float)
        t_norm = t_int.to(torch.float32) / float(num_T)  # [B]

        # GradCAM per-sample (GradCAM usually does backward; keep grads enabled here)
        for i in range(bsz):
            try:
                heatmap, info = gradcam.generate_heatmap(
                    latent=noisy_lat[i:i+1],
                    timestep=t_norm[i:i+1],          # IMPORTANT: match classifier forward
                    target_class=args.harmful_class,
                    normalize=False                  # RAW values
                )

                # heatmap: torch.Tensor, collect raw values
                vals = heatmap.detach().flatten().cpu().numpy()
                stats.update(vals)
                reservoir.update(vals)

                total_images += 1

            except Exception as e:
                error_images += 1
                # keep going; optionally print path
                # print(f"Error on {batch['path'][i]}: {e}")
                continue

    if stats.n == 0:
        raise RuntimeError("No heatmap values collected. Check GradCAM layer name / classifier forward signature.")

    # Percentiles from reservoir (approx if reservoir < total values)
    sample_vals = reservoir.get_array()
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = {f"p{p}": float(np.percentile(sample_vals, p)) for p in percentiles} if sample_vals.size > 0 else {}

    # Prepare output
    out = {
        "mean": float(stats.mean),
        "std": float(stats.std),
        "min": float(stats.min),
        "max": float(stats.max),
        "percentiles": percentile_values,
        "num_images_processed": int(total_images),
        "num_images_error": int(error_images),
        "num_values_seen": int(stats.n),
        "percentile_sample_size": int(sample_vals.size),
        "classifier_ckpt": args.classifier_ckpt,
        "harmful_class": int(args.harmful_class),
        "gradcam_layer": args.gradcam_layer,
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "timestep_sampling": {
            "mode": "fixed" if args.fixed_t is not None else "random",
            "fixed_t": int(args.fixed_t) if args.fixed_t is not None else None,
            "t_min": int(t_min),
            "t_max": int(t_max),
            "num_train_timesteps": int(num_T),
            "classifier_timestep_format": "t_norm = t_int / num_train_timesteps"
        }
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 80)
    print("STATISTICS SUMMARY (RAW GradCAM)")
    print("=" * 80)
    print(f"Images processed: {total_images} (errors: {error_images})")
    print(f"Total values seen: {stats.n:,}")
    print(f"Mean: {out['mean']:.6f}")
    print(f"Std:  {out['std']:.6f}")
    print(f"Min:  {out['min']:.6f}")
    print(f"Max:  {out['max']:.6f}")
    if percentile_values:
        print("Percentiles (from reservoir sample):")
        for p in percentiles:
            print(f"  {p:>2}th: {percentile_values[f'p{p}']:.6f}")
    print(f"\n✓ Saved to: {output_path}")
    print("=" * 80 + "\n")

    # Inference hint
    print("=" * 80)
    print("USAGE IN INFERENCE (example)")
    print("=" * 80)
    print("You can normalize RAW heatmap values with:")
    print(f"  z = (value - {out['mean']:.4f}) / {out['std']:.4f}")
    print("Then map z -> [0,1] with a CDF (e.g., Gaussian) or a sigmoid.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
