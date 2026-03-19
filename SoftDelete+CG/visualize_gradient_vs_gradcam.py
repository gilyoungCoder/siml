#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare Classifier Gradient vs Grad-CAM spatial distribution

This script visualizes:
1. Classifier gradient magnitude (spatial distribution)
2. Grad-CAM heatmap (spatial distribution)
3. Difference between them

To answer: "Does classifier gradient naturally focus on nude regions,
           or does it spread to background?"
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import os

from geo_models.classifier.classifier import load_discriminator
from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline
from geo_utils.classifier_interpretability import ClassifierGradCAM


def visualize_comparison(
    image_path: str,
    classifier_ckpt: str,
    output_dir: str = "./visualization/gradient_vs_gradcam"
):
    """
    Compare gradient and Grad-CAM for a single image.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading Stable Diffusion pipeline...")
    pipe = CustomStableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        safety_checker=None
    ).to(device)

    print("[INFO] Loading classifier...")
    classifier = load_discriminator(
        ckpt_path=classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=3
    ).to(device)
    classifier.eval()

    # Load and encode image
    print(f"[INFO] Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB").resize((512, 512))
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = (image_np - 0.5) / 0.5  # Normalize to [-1, 1]
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)

    # Encode to latent
    with torch.no_grad():
        latent = pipe.vae.encode(image_tensor).latent_dist.sample()
        latent = latent * pipe.vae.config.scaling_factor

    timestep = torch.tensor([0], device=device, dtype=torch.long)

    # ==========================================
    # Method 1: Classifier Gradient
    # ==========================================
    print("\n[1/2] Computing Classifier Gradient...")
    latent_grad = latent.clone().detach().requires_grad_(True)

    logits = classifier(latent_grad, timestep)
    probs = F.softmax(logits, dim=-1)

    # Gradient for "clothed" class (we want to move toward this)
    target_score = logits[:, 1]  # Clothed
    target_score.backward()

    grad = latent_grad.grad  # [1, 4, 64, 64]
    grad_magnitude = grad.abs().mean(dim=1).squeeze(0)  # [64, 64]

    # Normalize
    grad_magnitude = (grad_magnitude - grad_magnitude.min()) / (grad_magnitude.max() - grad_magnitude.min() + 1e-8)

    print(f"  Gradient magnitude range: [{grad.min().item():.4f}, {grad.max().item():.4f}]")
    print(f"  Gradient std: {grad.std().item():.4f}")

    # ==========================================
    # Method 2: Grad-CAM
    # ==========================================
    print("\n[2/2] Computing Grad-CAM...")
    gradcam = ClassifierGradCAM(
        classifier,
        target_layer_name="encoder_model.middle_block.2"
    )

    heatmap, info = gradcam.generate_heatmap(
        latent,
        timestep,
        target_class=2,  # Nude
        normalize=True
    )
    heatmap = heatmap.squeeze(0).cpu()  # [64, 64]

    print(f"  Heatmap range: [{heatmap.min().item():.4f}, {heatmap.max().item():.4f}]")
    print(f"  Prediction: {['Not People', 'Clothed', 'Nude'][logits.argmax().item()]}")
    print(f"  Probs: Not People={probs[0,0]:.3f}, Clothed={probs[0,1]:.3f}, Nude={probs[0,2]:.3f}")

    # ==========================================
    # Visualization
    # ==========================================
    print("\n[INFO] Creating visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Original image and distributions
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    im1 = axes[0, 1].imshow(grad_magnitude.cpu().numpy(), cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title(f"Classifier Gradient Magnitude\n(toward Clothed class)")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(heatmap.numpy(), cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title(f"Grad-CAM Heatmap\n(Nude class attention)")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2])

    # Row 2: Difference and statistics
    diff = (grad_magnitude.cpu() - heatmap).numpy()
    im3 = axes[1, 0].imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 0].set_title("Difference\n(Gradient - Grad-CAM)")
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0])

    # Histogram comparison
    axes[1, 1].hist(grad_magnitude.cpu().numpy().flatten(), bins=50, alpha=0.5, label='Gradient', color='blue')
    axes[1, 1].hist(heatmap.numpy().flatten(), bins=50, alpha=0.5, label='Grad-CAM', color='red')
    axes[1, 1].set_xlabel("Intensity")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Distribution Comparison")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Spatial correlation
    grad_flat = grad_magnitude.cpu().numpy().flatten()
    gradcam_flat = heatmap.numpy().flatten()

    axes[1, 2].scatter(grad_flat, gradcam_flat, alpha=0.3, s=1)
    axes[1, 2].plot([0, 1], [0, 1], 'r--', label='y=x')
    axes[1, 2].set_xlabel("Gradient Magnitude")
    axes[1, 2].set_ylabel("Grad-CAM Intensity")
    axes[1, 2].set_title(f"Spatial Correlation\n(R² = {np.corrcoef(grad_flat, gradcam_flat)[0,1]**2:.3f})")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, Path(image_path).stem + "_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved visualization to: {output_path}")
    plt.close()

    # ==========================================
    # Quantitative Analysis
    # ==========================================
    print("\n" + "="*80)
    print("QUANTITATIVE COMPARISON")
    print("="*80)

    # 1. Spatial concentration (Gini coefficient)
    def gini_coefficient(x):
        """Higher Gini = more concentrated"""
        x_sorted = np.sort(x.flatten())
        n = len(x_sorted)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1) / n

    gini_grad = gini_coefficient(grad_magnitude.cpu().numpy())
    gini_gradcam = gini_coefficient(heatmap.numpy())

    print(f"\n[Spatial Concentration]")
    print(f"  Gradient Gini:  {gini_grad:.4f}")
    print(f"  Grad-CAM Gini:  {gini_gradcam:.4f}")
    print(f"  → Higher = more concentrated/localized")

    # 2. Top-k overlap
    k = int(64 * 64 * 0.3)  # Top 30% pixels

    grad_topk = grad_magnitude.cpu().numpy().flatten().argsort()[-k:]
    gradcam_topk = heatmap.numpy().flatten().argsort()[-k:]

    overlap = len(set(grad_topk) & set(gradcam_topk)) / k

    print(f"\n[Top-30% Pixel Overlap]")
    print(f"  Overlap ratio:  {overlap:.2%}")
    print(f"  → Higher = more similar spatial targeting")

    # 3. Spatial correlation
    corr = np.corrcoef(grad_flat, gradcam_flat)[0, 1]
    print(f"\n[Spatial Correlation]")
    print(f"  Pearson R:  {corr:.4f}")
    print(f"  R²:         {corr**2:.4f}")
    print(f"  → Higher = more similar distribution")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    if overlap > 0.7 and corr > 0.7:
        print("✓ Gradient and Grad-CAM are SIMILAR")
        print("  → Classifier gradient naturally focuses on nude regions")
        print("  → Output masking may not provide significant advantage")
    elif overlap < 0.5 or corr < 0.5:
        print("✗ Gradient and Grad-CAM are DIFFERENT")
        print("  → Classifier gradient spreads to background/context")
        print("  → Output masking provides more precise spatial control")
    else:
        print("~ Gradient and Grad-CAM are MODERATELY SIMILAR")
        print("  → Some differences exist, output masking may help")

    print("="*80 + "\n")

    return {
        'gini_grad': gini_grad,
        'gini_gradcam': gini_gradcam,
        'overlap': overlap,
        'correlation': corr,
        'r_squared': corr**2
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--classifier_ckpt", type=str,
                       default="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth")
    parser.add_argument("--output_dir", type=str, default="./visualization/gradient_vs_gradcam")

    args = parser.parse_args()

    stats = visualize_comparison(
        args.image,
        args.classifier_ckpt,
        args.output_dir
    )


if __name__ == "__main__":
    main()
