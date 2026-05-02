#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute sample-level GradCAM stats and add to harmful_stats.pt.

Multi-GPU version: distributes images across GPUs, each computes
per-sample GradCAM means, then merges to get sample_level_mu/sigma.

Usage:
    # Single GPU
    python compute_sample_level_stats.py

    # Multi-GPU worker (called by shell script)
    CUDA_VISIBLE_DEVICES=0 python compute_sample_level_stats.py --gpu_id 0 --num_gpus 8

    # Merge mode
    python compute_sample_level_stats.py --merge
"""

import argparse
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKL
from models.latent_classifier import LatentResNet18Classifier


EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def scan_images(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted([
        os.path.join(directory, f) for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in EXTENSIONS
    ])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_ckpt", type=str,
                        default="./work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth")
    parser.add_argument("--nudity_data_path", type=str,
                        default="/mnt/home/yhgil99/dataset/threeclassImg/Wnudity5k")
    parser.add_argument("--pretrained_model", type=str,
                        default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--harmful_class", type=int, default=2)
    parser.add_argument("--gradcam_layer", type=str, default="layer2")
    parser.add_argument("--stats_path", type=str, default="./harmful_stats.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--tmp_dir", type=str, default="./tmp_sample_stats")
    # Multi-GPU
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    # Merge
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.tmp_dir, exist_ok=True)

    if args.merge:
        # Merge per-GPU results
        all_means = []
        for f in sorted(os.listdir(args.tmp_dir)):
            if f.startswith("sample_means_gpu") and f.endswith(".pt"):
                means = torch.load(os.path.join(args.tmp_dir, f))
                all_means.append(means)
                print(f"  Loaded {f}: {means.shape[0]} samples")

        if not all_means:
            print("No per-GPU results found!")
            return

        all_means = torch.cat(all_means)
        print(f"\nTotal samples: {all_means.shape[0]}")
        print(f"  sample_level_mu:     {all_means.mean().item():.6f}")
        print(f"  sample_level_sigma:  {all_means.std().item():.6f}")
        print(f"  sample_level_median: {all_means.median().item():.6f}")
        print(f"  sample_level_p10:    {all_means.quantile(0.10).item():.6f}")
        print(f"  sample_level_p25:    {all_means.quantile(0.25).item():.6f}")
        print(f"  sample_level_p75:    {all_means.quantile(0.75).item():.6f}")
        print(f"  sample_level_p90:    {all_means.quantile(0.90).item():.6f}")

        # Update harmful_stats.pt
        stats = torch.load(args.stats_path, map_location="cpu")
        stats["sample_level_mu"] = all_means.mean().item()
        stats["sample_level_sigma"] = all_means.std().item()
        stats["sample_level_median"] = all_means.median().item()
        stats["sample_level_p10"] = all_means.quantile(0.10).item()
        stats["sample_level_p25"] = all_means.quantile(0.25).item()
        stats["sample_level_p75"] = all_means.quantile(0.75).item()
        stats["sample_level_p90"] = all_means.quantile(0.90).item()
        torch.save(stats, args.stats_path)
        print(f"\nUpdated: {args.stats_path}")

        # Compare with old pixel-level stats
        print(f"\n  Old pixel-level full_mu:    {stats['gradcam_full_mu']:.6f}")
        print(f"  Old pixel-level full_sigma: {stats['gradcam_full_sigma']:.6f}")
        print(f"  New sample-level mu:        {stats['sample_level_mu']:.6f}")
        print(f"  New sample-level sigma:     {stats['sample_level_sigma']:.6f}")
        return

    # === Worker mode ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load images and split
    all_paths = scan_images(args.nudity_data_path)
    total = len(all_paths)
    chunk = (total + args.num_gpus - 1) // args.num_gpus
    start = args.gpu_id * chunk
    end = min(start + chunk, total)
    paths = all_paths[start:end]

    print(f"GPU {args.gpu_id}: images [{start}:{end}] ({len(paths)}/{total})")

    # Load VAE + classifier
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device)

    classifier = LatentResNet18Classifier(
        num_classes=args.num_classes, pretrained_backbone=False
    ).to(device)
    classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location=device))
    classifier.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    # Compute per-sample GradCAM means
    sample_means = []
    for i in tqdm(range(0, len(paths), args.batch_size), desc=f"GPU{args.gpu_id}"):
        batch_paths = paths[i:i + args.batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(transform(img))
            except Exception as e:
                print(f"  Skip {p}: {e}")
                continue

        if not imgs:
            continue

        batch = torch.stack(imgs).to(device)

        with torch.no_grad():
            z0 = vae.encode(batch).latent_dist.mean * 0.18215

        with torch.enable_grad():
            gradcam = classifier.compute_gradcam(
                z0, target_class=args.harmful_class, layer_name=args.gradcam_layer
            )  # (B, 1, H, W)

        # Per-sample mean
        per_sample = gradcam.mean(dim=[1, 2, 3])  # (B,)
        sample_means.append(per_sample.cpu())

    sample_means = torch.cat(sample_means)
    out_path = os.path.join(args.tmp_dir, f"sample_means_gpu{args.gpu_id}.pt")
    torch.save(sample_means, out_path)
    print(f"GPU {args.gpu_id}: saved {sample_means.shape[0]} samples to {out_path}")
    print(f"  mean={sample_means.mean():.6f}, std={sample_means.std():.6f}")


if __name__ == "__main__":
    main()
