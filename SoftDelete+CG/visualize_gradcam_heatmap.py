#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GradCAM Heatmap Visualization

Visualize GradCAM heatmaps on images to understand classifier behavior.
Shows:
1. Original image
2. Raw heatmap
3. CDF-normalized heatmap (pixel-level)
4. Histogram of heatmap values

Usage:
    python visualize_gradcam_heatmap.py \
        --image_path /path/to/image.jpg \
        --classifier_ckpt ./work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth \
        --gradcam_stats_dir ./gradcam_stats/nudity_4class \
        --target_class 2 \
        --output_dir ./gradcam_vis
"""

import os
import json
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
from torchvision import transforms
from torch.distributions import Normal

from diffusers import AutoencoderKL, DDPMScheduler
from geo_models.classifier.classifier import load_discriminator
from geo_utils.classifier_interpretability import ClassifierGradCAM


CLASS_NAMES = {0: "benign", 1: "safe_clothed", 2: "harm_nude", 3: "harm_color"}


def load_gradcam_stats(stats_dir: str, target_class: int):
    """Load stats for target class."""
    stats_dir = Path(stats_dir)
    class_name = CLASS_NAMES[target_class]
    fname = f"gradcam_stats_{class_name}_class{target_class}.json"
    path = stats_dir / fname

    if path.exists():
        with open(path) as f:
            d = json.load(f)
        topk = d.get("topk", {})
        sample = d.get("sample_level", {})
        return {
            # All-pixel stats (for reference)
            "pixel_mean": float(d["mean"]),
            "pixel_std": float(d["std"]),
            # Top-K pixel stats (for spatial CDF visualization)
            "topk_mean": float(topk.get("mean", d["mean"])),
            "topk_std": float(topk.get("std", d["std"])),
            # Sample-level stats (for monitoring P(harm))
            "sample_mean": float(sample.get("mean", d["mean"])),
            "sample_std": float(sample.get("std", d["std"])),
        }
    return None


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--image_path", type=str, help="Single image path")
    parser.add_argument("--image_dir", type=str, help="Directory of images")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to visualize from dir")
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--gradcam_stats_dir", type=str, default=None)
    parser.add_argument("--target_class", type=int, default=2, choices=[2, 3])
    parser.add_argument("--output_dir", type=str, default="./gradcam_vis")
    parser.add_argument("--pretrained_model", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--gradcam_layer", type=str, default="encoder_model.middle_block.2")
    parser.add_argument("--timestep", type=int, default=500, help="Timestep for noising (0-999)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def visualize_single_image(
    image_path: str,
    vae,
    scheduler,
    classifier,
    gradcam,
    stats: dict,
    target_class: int,
    timestep: int,
    output_dir: Path,
    device: str
):
    """Visualize GradCAM for a single image."""

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Encode to latent
    with torch.no_grad():
        latent = vae.encode(img_tensor).latent_dist.sample() * 0.18215

    # Add noise at timestep
    noise = torch.randn_like(latent)
    alpha_bar = scheduler.alphas_cumprod[timestep].to(device)
    noisy_latent = torch.sqrt(alpha_bar) * latent + torch.sqrt(1 - alpha_bar) * noise

    # Classifier prediction
    norm_t = torch.tensor([timestep / 1000.0], device=device)
    clf_dtype = next(classifier.parameters()).dtype
    with torch.no_grad():
        logits = classifier(noisy_latent.to(clf_dtype), norm_t)
        probs = F.softmax(logits, dim=1)[0]

    # Generate heatmap
    with torch.enable_grad():
        heatmap_raw, info = gradcam.generate_heatmap(
            noisy_latent.to(clf_dtype), norm_t, target_class, normalize=False
        )

    heatmap_raw = heatmap_raw.detach().cpu().numpy().squeeze()  # (1, 64, 64) -> (64, 64)

    # CDF normalize using topk stats (for spatial visualization)
    heatmap_cdf = None
    if stats:
        z = (heatmap_raw - stats["topk_mean"]) / (stats["topk_std"] + 1e-8)
        normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
        heatmap_cdf = normal.cdf(torch.tensor(z)).numpy()

    # Compute sample-level P(harm)
    heatmap_mean = heatmap_raw.mean()
    if stats:
        z_sample = (heatmap_mean - stats["sample_mean"]) / (stats["sample_std"] + 1e-8)
        p_harm = Normal(torch.tensor(0.0), torch.tensor(1.0)).cdf(torch.tensor(z_sample)).item()
    else:
        p_harm = None

    # Resize heatmaps to original image size for overlay
    img_np = np.array(img)
    img_h, img_w = img_np.shape[:2]

    # Resize raw heatmap to image size
    heatmap_resized = np.array(Image.fromarray(heatmap_raw).resize((img_w, img_h), Image.BILINEAR))

    # Normalize for colormap
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    heatmap_colored = plt.cm.jet(heatmap_norm)[:, :, :3]  # RGB only

    # Create overlay (alpha blend)
    alpha = 0.5
    overlay_raw = (1 - alpha) * (img_np / 255.0) + alpha * heatmap_colored
    overlay_raw = np.clip(overlay_raw, 0, 1)

    # CDF overlay if available
    overlay_cdf = None
    if heatmap_cdf is not None:
        heatmap_cdf_resized = np.array(Image.fromarray(heatmap_cdf.astype(np.float32)).resize((img_w, img_h), Image.BILINEAR))
        heatmap_cdf_colored = plt.cm.jet(heatmap_cdf_resized)[:, :, :3]
        overlay_cdf = (1 - alpha) * (img_np / 255.0) + alpha * heatmap_cdf_colored
        overlay_cdf = np.clip(overlay_cdf, 0, 1)

    # Create visualization (2x4 grid)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: Original, Raw Heatmap, Raw Overlay, CDF Overlay
    # 1. Original image
    ax = axes[0, 0]
    ax.imshow(img)
    ax.set_title("Original Image")
    ax.axis("off")

    # 2. Raw heatmap (64x64)
    ax = axes[0, 1]
    im = ax.imshow(heatmap_raw, cmap="jet")
    ax.set_title(f"Raw Heatmap (64x64)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 3. Raw heatmap overlay
    ax = axes[0, 2]
    ax.imshow(overlay_raw)
    ax.set_title(f"Raw Overlay (alpha={alpha})")
    ax.axis("off")

    # 4. CDF heatmap overlay (using topk stats)
    ax = axes[0, 3]
    if overlay_cdf is not None:
        ax.imshow(overlay_cdf)
        ax.set_title(f"TopK CDF Overlay (alpha={alpha})")
    else:
        ax.imshow(img)
        ax.text(0.5, 0.5, "No stats", ha="center", va="center", transform=ax.transAxes, fontsize=14, color="white")
        ax.set_title("CDF Overlay (N/A)")
    ax.axis("off")

    # Row 2: Classifier probs, CDF heatmap, Histogram, Summary
    # 5. Classifier probabilities
    ax = axes[1, 0]
    classes = list(CLASS_NAMES.values())
    colors = ["green", "blue", "red", "orange"]
    ax.barh(classes, probs.cpu().numpy(), color=colors)
    ax.set_xlim(0, 1)
    ax.set_title(f"Classifier Probs (t={timestep})")
    for i, p in enumerate(probs.cpu().numpy()):
        ax.text(p + 0.02, i, f"{p:.3f}", va="center")

    # 6. CDF-normalized heatmap using topk stats (64x64)
    ax = axes[1, 1]
    if heatmap_cdf is not None:
        im = ax.imshow(heatmap_cdf, cmap="jet", vmin=0, vmax=1)
        ax.set_title(f"TopK CDF Heatmap (64x64)")
        plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        ax.text(0.5, 0.5, "No stats available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("CDF Heatmap (N/A)")
    ax.axis("off")

    # 7. Histogram of raw values
    ax = axes[1, 2]
    ax.hist(heatmap_raw.flatten(), bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(heatmap_mean, color="red", linestyle="--", label=f"mean={heatmap_mean:.4f}")
    if stats:
        ax.axvline(stats["pixel_mean"], color="green", linestyle=":", label=f"stats_mean={stats['pixel_mean']:.4f}")
    ax.set_xlabel("Heatmap Value")
    ax.set_ylabel("Count")
    ax.set_title("Raw Heatmap Distribution")
    ax.legend(fontsize=8)

    # 8. Sample-level info
    ax = axes[1, 3]
    ax.axis("off")
    info_text = f"Image: {Path(image_path).name}\n"
    info_text += f"Timestep: {timestep}\n"
    info_text += f"Target class: {target_class} ({CLASS_NAMES[target_class]})\n\n"
    info_text += f"Heatmap stats:\n"
    info_text += f"  mean: {heatmap_mean:.6f}\n"
    info_text += f"  min:  {heatmap_raw.min():.6f}\n"
    info_text += f"  max:  {heatmap_raw.max():.6f}\n"
    info_text += f"  std:  {heatmap_raw.std():.6f}\n\n"
    if stats:
        info_text += f"Sample-level P(harm):\n"
        info_text += f"  z = ({heatmap_mean:.4f} - {stats['sample_mean']:.4f}) / {stats['sample_std']:.4f}\n"
        info_text += f"  z = {z_sample:.4f}\n"
        info_text += f"  P(harm) = CDF(z) = {p_harm:.4f}\n"
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=10, verticalalignment="top", family="monospace")
    ax.set_title("Summary")

    plt.tight_layout()

    # Save
    output_name = Path(image_path).stem + f"_gradcam_class{target_class}_t{timestep}.png"
    output_path = output_dir / output_name
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")
    return {
        "image": str(image_path),
        "heatmap_mean": float(heatmap_mean),
        "p_harm": float(p_harm) if p_harm is not None else None,
        "probs": probs.cpu().numpy().tolist()
    }


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("GradCAM Heatmap Visualization")
    print("="*60)

    # Load VAE
    print("\n[1/4] Loading VAE...")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae").to(device)
    vae.eval()

    # Load scheduler
    print("[2/4] Loading Scheduler...")
    scheduler = DDPMScheduler.from_config(args.pretrained_model, subfolder="scheduler")

    # Load classifier
    print("[3/4] Loading Classifier...")
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt, condition=None, eval=True, channel=4, num_classes=4
    ).to(device)
    classifier.eval()

    # Init GradCAM
    print("[4/4] Initializing GradCAM...")
    gradcam = ClassifierGradCAM(classifier, args.gradcam_layer)

    # Load stats
    stats = None
    if args.gradcam_stats_dir:
        stats = load_gradcam_stats(args.gradcam_stats_dir, args.target_class)
        if stats:
            print(f"\nLoaded stats for class {args.target_class}:")
            print(f"  [all pixel] mean={stats['pixel_mean']:.4f}, std={stats['pixel_std']:.4f}")
            print(f"  [topk]      mean={stats['topk_mean']:.4f}, std={stats['topk_std']:.4f}")
            print(f"  [sample]    mean={stats['sample_mean']:.4f}, std={stats['sample_std']:.4f}")

    # Collect images
    image_paths = []
    if args.image_path:
        image_paths = [args.image_path]
    elif args.image_dir:
        exts = [".png", ".jpg", ".jpeg", ".webp"]
        for ext in exts:
            image_paths.extend(list(Path(args.image_dir).glob(f"*{ext}")))
        random.shuffle(image_paths)
        image_paths = image_paths[:args.num_images]

    if not image_paths:
        print("No images found!")
        return

    print(f"\nProcessing {len(image_paths)} images...")

    results = []
    for img_path in image_paths:
        result = visualize_single_image(
            str(img_path), vae, scheduler, classifier, gradcam,
            stats, args.target_class, args.timestep, output_dir, device
        )
        results.append(result)

    # Save summary
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! Visualizations saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
