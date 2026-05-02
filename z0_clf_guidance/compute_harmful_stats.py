#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute harmful score statistics for Gaussian CDF-based spatial thresholding.

Runs the z0 classifier on harmful training images to collect:
  1. Harmful class probability distribution (mu, sigma)
  2. GradCAM value distribution (mu, sigma) for spatial CDF thresholding

The GradCAM stats are used for spatial masking:
  Proper GradCAM (intermediate feature maps + upsampling) gives per-pixel
  activation maps. We compute the CDF of these values under the training
  distribution, making the spatial threshold interpretable as a CDF percentile.

Usage:
    python compute_harmful_stats.py \
        --classifier_ckpt ./work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth \
        --nudity_data_path /path/to/nude/images \
        --output_path ./harmful_stats.pt
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
    """Scan directory for image files."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted([
        os.path.join(directory, f) for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in EXTENSIONS
    ])


def collect_stats(image_paths, classifier, vae, transform,
                  device, harmful_class, gradcam_layer, batch_size=16):
    """
    Run classifier on VAE-encoded images, return:
      - per-class logits and probabilities
      - GradCAM values (per-pixel, normalized [0,1] per sample) for CDF stats
    """
    all_logits = []
    all_probs = []
    all_gradcam = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing"):
        batch_paths = image_paths[i:i + batch_size]
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

        # VAE encode -> z0
        with torch.no_grad():
            z0 = vae.encode(batch).latent_dist.mean * 0.18215

        # Classifier (no grad) for probability stats
        with torch.no_grad():
            logits = classifier(z0)
            probs = F.softmax(logits, dim=-1)
            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())

        # Proper GradCAM for spatial CDF stats
        with torch.enable_grad():
            gradcam = classifier.compute_gradcam(
                z0, target_class=harmful_class, layer_name=gradcam_layer
            )  # (B, 1, H, W), normalized [0, 1] per sample
        all_gradcam.append(gradcam.cpu())

    all_logits = torch.cat(all_logits)
    all_probs = torch.cat(all_probs)
    all_gradcam = torch.cat(all_gradcam)  # (N, 1, H, W)

    return all_logits, all_probs, all_gradcam


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute harmful score stats for GradCAM CDF thresholding"
    )
    parser.add_argument("--classifier_ckpt", type=str, required=True,
                        help="Path to trained z0 classifier checkpoint")
    parser.add_argument("--nudity_data_path", type=str, required=True,
                        help="Directory of harmful (nude) training images")
    parser.add_argument("--pretrained_model", type=str,
                        default="CompVis/stable-diffusion-v1-4",
                        help="SD model for VAE encoder")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--harmful_class", type=int, default=2,
                        help="Harmful class index (default: 2=nude)")
    parser.add_argument("--gradcam_layer", type=str, default="layer4",
                        choices=["layer1", "layer2", "layer3", "layer4"],
                        help="Which ResNet layer for GradCAM")
    parser.add_argument("--output_path", type=str, default="./harmful_stats.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process (None=all)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device)

    # Load classifier
    print(f"Loading classifier: {args.classifier_ckpt}")
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

    # Collect stats from harmful (nude) images
    nude_paths = scan_images(args.nudity_data_path)
    if args.max_samples:
        nude_paths = nude_paths[:args.max_samples]
    print(f"\nHarmful images: {len(nude_paths)}")
    print(f"GradCAM layer: {args.gradcam_layer}")

    nude_logits, nude_probs, nude_gradcam = collect_stats(
        nude_paths, classifier, vae, transform, device,
        args.harmful_class, args.gradcam_layer, args.batch_size,
    )

    hc = args.harmful_class

    # GradCAM stats: top 30% of pixel values per image
    # Only the "activated" regions matter for CDF calibration.
    top_frac = 0.3
    all_top_vals = []
    for idx in range(nude_gradcam.shape[0]):
        flat = nude_gradcam[idx].flatten()  # (H*W,)
        k = max(1, int(flat.numel() * top_frac))
        topk_vals, _ = flat.topk(k)
        all_top_vals.append(topk_vals)
    top_vals = torch.cat(all_top_vals)

    # Subsample for quantile computation
    max_quantile_samples = 1_000_000
    if top_vals.numel() > max_quantile_samples:
        perm = torch.randperm(top_vals.numel())[:max_quantile_samples]
        top_sub = top_vals[perm]
    else:
        top_sub = top_vals

    # Sample-level stats: per-image GradCAM mean → distribution across samples
    # This is what monitoring uses: mean(GradCAM(z0)) is one scalar per sample
    sample_means = nude_gradcam.mean(dim=[1, 2, 3])  # (N,)

    # Also compute full-image pixel-level stats for reference
    gc_flat = nude_gradcam.flatten()

    stats = {
        "harmful_class": hc,
        "n_harmful_samples": len(nude_probs),
        "classifier_ckpt": args.classifier_ckpt,
        "gradcam_layer": args.gradcam_layer,
        "top_fraction": top_frac,
        # Harmful class probability stats
        "prob_mu": nude_probs[:, hc].mean().item(),
        "prob_sigma": nude_probs[:, hc].std().item(),
        # Harmful class logit stats
        "logit_mu": nude_logits[:, hc].mean().item(),
        "logit_sigma": nude_logits[:, hc].std().item(),
        # GradCAM stats from TOP 30% pixels (key for spatial CDF)
        "gradcam_mu": top_vals.mean().item(),
        "gradcam_sigma": top_vals.std().item(),
        "gradcam_median": top_sub.median().item(),
        "gradcam_p90": top_sub.quantile(0.9).item(),
        "gradcam_p95": top_sub.quantile(0.95).item(),
        # Sample-level GradCAM stats (for monitoring CDF)
        # Each sample → mean(GradCAM heatmap) → one scalar → distribution
        "sample_level_mu": sample_means.mean().item(),
        "sample_level_sigma": sample_means.std().item(),
        "sample_level_median": sample_means.median().item(),
        "sample_level_p10": sample_means.quantile(0.10).item(),
        "sample_level_p25": sample_means.quantile(0.25).item(),
        "sample_level_p75": sample_means.quantile(0.75).item(),
        "sample_level_p90": sample_means.quantile(0.90).item(),
        # Full-image pixel-level GradCAM stats (reference only)
        "gradcam_full_mu": gc_flat.mean().item(),
        "gradcam_full_sigma": gc_flat.std().item(),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(stats, args.output_path)

    print(f"\n{'=' * 60}")
    print("HARMFUL SCORE STATISTICS (GradCAM-based)")
    print(f"{'=' * 60}")
    print(f"  Classifier:    {args.classifier_ckpt}")
    print(f"  Harmful class: {hc}")
    print(f"  GradCAM layer: {args.gradcam_layer}")
    print(f"  N samples:     {stats['n_harmful_samples']}")
    print(f"  Probability:   mu={stats['prob_mu']:.4f}, sigma={stats['prob_sigma']:.4f}")
    print(f"  Logit:         mu={stats['logit_mu']:.4f}, sigma={stats['logit_sigma']:.4f}")
    print(f"  GradCAM (top {top_frac:.0%} pixels):")
    print(f"    mu={stats['gradcam_mu']:.6f}, sigma={stats['gradcam_sigma']:.6f}")
    print(f"    median={stats['gradcam_median']:.6f}")
    print(f"    p90={stats['gradcam_p90']:.6f}, p95={stats['gradcam_p95']:.6f}")
    print(f"  GradCAM sample-level (for monitoring CDF):")
    print(f"    mu={stats['sample_level_mu']:.6f}, sigma={stats['sample_level_sigma']:.6f}")
    print(f"    median={stats['sample_level_median']:.6f}")
    print(f"    p10={stats['sample_level_p10']:.6f}, p90={stats['sample_level_p90']:.6f}")
    print(f"  GradCAM pixel-level (full image, reference):")
    print(f"    mu={stats['gradcam_full_mu']:.6f}, sigma={stats['gradcam_full_sigma']:.6f}")
    print(f"\n  Saved to: {args.output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
