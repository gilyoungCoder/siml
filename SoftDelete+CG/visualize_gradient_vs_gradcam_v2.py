#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IMPROVED: Compare Classifier Gradient vs Grad-CAM (공정한 비교)

개선사항:
1. 같은 target class 사용 (Nude class 2번으로 통일)
2. Gradient를 실제 guidance처럼 계산 (safe class로 향하는 방향)
3. 절대 magnitude 보존 (normalize 전 비교)
4. Thresholding을 통한 "실제 건드리는 영역" 시각화
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


def visualize_improved_comparison(
    image_path: str,
    classifier_ckpt: str,
    output_dir: str = "./visualization/gradient_vs_gradcam_v2",
    threshold_percentile: float = 0.3,  # Top 30% for Gradient
    gradcam_threshold: float = 0.3      # Fixed threshold 0.3 for Grad-CAM
):
    """
    개선된 비교: 공정하고 해석 가능한 visualization
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
    image_np = (image_np - 0.5) / 0.5
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        latent = pipe.vae.encode(image_tensor).latent_dist.sample()
        latent = latent * pipe.vae.config.scaling_factor

    timestep = torch.tensor([0], device=device, dtype=torch.long)

    # Get prediction
    with torch.no_grad():
        logits = classifier(latent, timestep)
        probs = F.softmax(logits, dim=-1)

    print(f"\nPrediction: {['Not People', 'Clothed', 'Nude'][logits.argmax().item()]}")
    print(f"Probs: Not People={probs[0,0]:.3f}, Clothed={probs[0,1]:.3f}, Nude={probs[0,2]:.3f}")

    # ==========================================
    # Method 1: Classifier Gradient (SAFE 방향)
    # ==========================================
    print("\n[1/3] Computing Classifier Gradient (toward SAFE class)...")
    latent_grad = latent.clone().detach().requires_grad_(True)

    logits_grad = classifier(latent_grad, timestep)

    # Safe class (Clothed = 1)로 향하는 gradient
    safe_score = logits_grad[:, 1]  # Clothed
    safe_score.backward()

    grad_safe = latent_grad.grad  # [1, 4, 64, 64]

    # L2 norm across channels (실제 guidance magnitude)
    grad_safe_magnitude = torch.norm(grad_safe, dim=1).squeeze(0)  # [64, 64]

    print(f"  Gradient magnitude: mean={grad_safe_magnitude.mean().item():.4f}, "
          f"max={grad_safe_magnitude.max().item():.4f}")

    # ==========================================
    # Method 2: Classifier Gradient (NUDE 방향) - Grad-CAM과 비교용
    # ==========================================
    print("\n[2/3] Computing Classifier Gradient (toward NUDE class)...")
    latent_grad2 = latent.clone().detach().requires_grad_(True)

    logits_grad2 = classifier(latent_grad2, timestep)

    # Nude class (2)로 향하는 gradient
    nude_score = logits_grad2[:, 2]  # Nude
    nude_score.backward()

    grad_nude = latent_grad2.grad  # [1, 4, 64, 64]

    # L2 norm across channels
    grad_nude_magnitude = torch.norm(grad_nude, dim=1).squeeze(0)  # [64, 64]

    print(f"  Gradient magnitude: mean={grad_nude_magnitude.mean().item():.4f}, "
          f"max={grad_nude_magnitude.max().item():.4f}")

    # ==========================================
    # Method 3: Grad-CAM (NUDE class)
    # ==========================================
    print("\n[3/3] Computing Grad-CAM (NUDE class attention)...")
    gradcam = ClassifierGradCAM(
        classifier,
        target_layer_name="encoder_model.middle_block.2"
    )

    heatmap, info = gradcam.generate_heatmap(
        latent,
        timestep,
        target_class=2,  # Nude
        normalize=False  # 원본 magnitude 유지
    )
    heatmap = heatmap.squeeze(0).cpu()  # [64, 64]

    # ReLU + normalize for visualization
    heatmap_vis = F.relu(heatmap)
    heatmap_vis = heatmap_vis / (heatmap_vis.max() + 1e-8)

    print(f"  Heatmap magnitude: mean={heatmap.mean().item():.4f}, "
          f"max={heatmap.max().item():.4f}")

    # ==========================================
    # Thresholding: "실제 건드리는 영역" 정의
    # ==========================================
    print(f"\n[INFO] Computing top-{threshold_percentile*100:.0f}% regions...")

    def get_top_k_mask(tensor, percentile):
        """Top percentile을 1, 나머지를 0으로"""
        flat = tensor.flatten()
        k = int(flat.numel() * percentile)
        if k == 0:
            k = 1
        threshold = torch.topk(flat, k=k)[0][-1]
        return (tensor >= threshold).float()

    # Masks: Gradient uses top-k, Grad-CAM uses fixed threshold
    mask_grad_safe = get_top_k_mask(grad_safe_magnitude.cpu(), threshold_percentile)
    mask_grad_nude = get_top_k_mask(grad_nude_magnitude.cpu(), threshold_percentile)
    mask_gradcam = (heatmap_vis >= gradcam_threshold).float()  # Fixed threshold for Grad-CAM

    # ==========================================
    # Visualization
    # ==========================================
    print("\n[INFO] Creating visualization...")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Row 1: Original image + 3 methods (magnitude)
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.imshow(image)
    ax00.set_title("Original Image", fontsize=14, fontweight='bold')
    ax00.axis("off")

    # Normalize for visualization
    grad_safe_norm = (grad_safe_magnitude.cpu() - grad_safe_magnitude.cpu().min()) / \
                     (grad_safe_magnitude.cpu().max() - grad_safe_magnitude.cpu().min() + 1e-8)
    grad_nude_norm = (grad_nude_magnitude.cpu() - grad_nude_magnitude.cpu().min()) / \
                     (grad_nude_magnitude.cpu().max() - grad_nude_magnitude.cpu().min() + 1e-8)

    ax01 = fig.add_subplot(gs[0, 1])
    im01 = ax01.imshow(grad_safe_norm.numpy(), cmap='hot', vmin=0, vmax=1)
    ax01.set_title("Gradient -> Safe (Clothed)\n[Guidance Direction]", fontsize=12)
    ax01.axis("off")
    plt.colorbar(im01, ax=ax01, fraction=0.046)

    ax02 = fig.add_subplot(gs[0, 2])
    im02 = ax02.imshow(grad_nude_norm.numpy(), cmap='hot', vmin=0, vmax=1)
    ax02.set_title("Gradient -> Nude\n[Comparison]", fontsize=12)
    ax02.axis("off")
    plt.colorbar(im02, ax=ax02, fraction=0.046)

    ax03 = fig.add_subplot(gs[0, 3])
    im03 = ax03.imshow(heatmap_vis.numpy(), cmap='hot', vmin=0, vmax=1)
    ax03.set_title("Grad-CAM (Nude Attention)\n[Masking Reference]", fontsize=12)
    ax03.axis("off")
    plt.colorbar(im03, ax=ax03, fraction=0.046)

    # Row 2: Top-k masks (binary)
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.imshow(image)
    ax10.set_title("Original (Reference)", fontsize=12)
    ax10.axis("off")

    ax11 = fig.add_subplot(gs[1, 1])
    ax11.imshow(mask_grad_safe.numpy(), cmap='Reds', vmin=0, vmax=1)
    ax11.set_title(f"Top {threshold_percentile*100:.0f}%: Gradient -> Safe\n[Actual Guidance Region]",
                   fontsize=12, fontweight='bold')
    ax11.axis("off")

    ax12 = fig.add_subplot(gs[1, 2])
    ax12.imshow(mask_grad_nude.numpy(), cmap='Reds', vmin=0, vmax=1)
    ax12.set_title(f"Top {threshold_percentile*100:.0f}%: Gradient -> Nude\n[Comparison]", fontsize=12)
    ax12.axis("off")

    ax13 = fig.add_subplot(gs[1, 3])
    ax13.imshow(mask_gradcam.numpy(), cmap='Reds', vmin=0, vmax=1)
    ax13.set_title(f"Threshold {gradcam_threshold:.1f}: Grad-CAM\n[Actual Masking Region]",
                   fontsize=12, fontweight='bold')
    ax13.axis("off")

    # Row 3: Overlap analysis
    ax20 = fig.add_subplot(gs[2, 0])
    # Overlap: Gradient(Safe) vs Grad-CAM
    overlap_safe_gradcam = mask_grad_safe * mask_gradcam
    union_safe_gradcam = torch.clamp(mask_grad_safe + mask_gradcam, 0, 1)

    # Color: Red=Gradient only, Blue=GradCAM only, Purple=Both
    overlap_vis = torch.zeros(64, 64, 3)
    overlap_vis[:, :, 0] = mask_grad_safe  # Red channel
    overlap_vis[:, :, 2] = mask_gradcam    # Blue channel

    ax20.imshow(overlap_vis.numpy())
    overlap_ratio = overlap_safe_gradcam.sum() / (union_safe_gradcam.sum() + 1e-8)
    ax20.set_title(f"Overlap: Gradient(Safe) vs Grad-CAM\nIoU={overlap_ratio:.2%}\n"
                   f"(Red=Grad, Blue=CAM, Purple=Both)",
                   fontsize=11)
    ax20.axis("off")

    # Histogram comparison
    ax21 = fig.add_subplot(gs[2, 1])
    ax21.hist(grad_safe_norm.numpy().flatten(), bins=50, alpha=0.5,
              label='Gradient→Safe', color='blue', density=True)
    ax21.hist(heatmap_vis.numpy().flatten(), bins=50, alpha=0.5,
              label='Grad-CAM', color='red', density=True)
    ax21.set_xlabel("Normalized Intensity", fontsize=10)
    ax21.set_ylabel("Density", fontsize=10)
    ax21.set_title("Distribution Comparison", fontsize=12)
    ax21.legend()
    ax21.grid(True, alpha=0.3)

    # Scatter plot
    ax22 = fig.add_subplot(gs[2, 2])
    grad_flat = grad_safe_norm.numpy().flatten()
    gradcam_flat = heatmap_vis.numpy().flatten()
    ax22.scatter(grad_flat, gradcam_flat, alpha=0.3, s=1)
    ax22.plot([0, 1], [0, 1], 'r--', label='y=x')
    corr = np.corrcoef(grad_flat, gradcam_flat)[0, 1]
    ax22.set_xlabel("Gradient -> Safe", fontsize=10)
    ax22.set_ylabel("Grad-CAM", fontsize=10)
    ax22.set_title(f"Spatial Correlation\nR-squared={corr**2:.3f}", fontsize=12)
    ax22.legend()
    ax22.grid(True, alpha=0.3)

    # Quantitative summary
    ax23 = fig.add_subplot(gs[2, 3])
    ax23.axis('off')

    # Calculate metrics
    overlap_grad_nude_gradcam = (mask_grad_nude * mask_gradcam).sum() / \
                                 (torch.clamp(mask_grad_nude + mask_gradcam, 0, 1).sum() + 1e-8)

    summary_text = f"""
QUANTITATIVE ANALYSIS

1. Overlap (IoU)
   Gradient(Safe) & Grad-CAM: {overlap_ratio:.2%}
   Gradient(Nude) & Grad-CAM: {overlap_grad_nude_gradcam:.2%}

2. Spatial Correlation
   Gradient(Safe) vs Grad-CAM: R^2={corr**2:.3f}

3. Coverage
   Gradient(Safe) Top-{threshold_percentile*100:.0f}%: {mask_grad_safe.sum().item():.0f} px
   Gradient(Nude) Top-{threshold_percentile*100:.0f}%: {mask_grad_nude.sum().item():.0f} px
   Grad-CAM Thresh-{gradcam_threshold:.1f}: {mask_gradcam.sum().item():.0f} px

INTERPRETATION:
{"High overlap -> Similar targeting" if overlap_ratio > 0.5 else "Low overlap -> Different targeting"}
{"High R^2 -> Strongly correlated" if corr**2 > 0.5 else "Low R^2 -> Weakly correlated"}

-> {"Grad-CAM more localized" if mask_gradcam.sum() < mask_grad_safe.sum() else "Gradient more localized"}
    """

    ax23.text(0.05, 0.95, summary_text.strip(),
              transform=ax23.transAxes,
              fontsize=10,
              verticalalignment='top',
              fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Save
    output_path = os.path.join(output_dir, Path(image_path).stem + "_improved_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[INFO] Saved visualization to: {output_path}")
    plt.close()

    # ==========================================
    # Detailed Analysis
    # ==========================================
    print("\n" + "="*80)
    print("DETAILED QUANTITATIVE ANALYSIS")
    print("="*80)

    print(f"\n[1] Top-{threshold_percentile*100:.0f}% Pixel Count:")
    print(f"  Gradient (Safe → Clothed): {mask_grad_safe.sum().item():.0f} pixels")
    print(f"  Gradient (Nude direction):  {mask_grad_nude.sum().item():.0f} pixels")
    print(f"  Grad-CAM (Nude attention):  {mask_gradcam.sum().item():.0f} pixels")

    print(f"\n[2] Overlap (IoU):")
    print(f"  Gradient(Safe) ∩ Grad-CAM: {overlap_ratio:.2%}")
    print(f"  Gradient(Nude) ∩ Grad-CAM: {overlap_grad_nude_gradcam:.2%}")

    print(f"\n[3] Spatial Correlation:")
    print(f"  Pearson R: {corr:.4f}")
    print(f"  R²:        {corr**2:.4f}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if overlap_ratio < 0.3:
        print("✗ Gradient와 Grad-CAM은 매우 다른 영역을 targeting합니다.")
        print("  → Grad-CAM 기반 masking이 더 정밀한 공간적 제어를 제공합니다.")
        print("  → Selective CG 방식의 공간적 masking이 유용합니다.")
    elif overlap_ratio > 0.7:
        print("✓ Gradient와 Grad-CAM이 유사한 영역을 targeting합니다.")
        print("  → Gradient만으로도 충분히 localized될 수 있습니다.")
    else:
        print("~ Gradient와 Grad-CAM이 부분적으로 겹칩니다.")
        print("  → Grad-CAM masking이 일부 개선을 제공할 수 있습니다.")

    print("="*80 + "\n")

    return {
        'overlap_safe_gradcam': overlap_ratio.item(),
        'overlap_nude_gradcam': overlap_grad_nude_gradcam.item(),
        'correlation': corr,
        'r_squared': corr**2,
        'coverage_grad_safe': mask_grad_safe.sum().item(),
        'coverage_grad_nude': mask_grad_nude.sum().item(),
        'coverage_gradcam': mask_gradcam.sum().item()
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--classifier_ckpt", type=str,
                       default="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth")
    parser.add_argument("--output_dir", type=str, default="./visualization/gradient_vs_gradcam_v2")
    parser.add_argument("--threshold_percentile", type=float, default=0.3,
                       help="Top percentile for Gradient masking (default: 0.3 = top 30%)")
    parser.add_argument("--gradcam_threshold", type=float, default=0.3,
                       help="Fixed threshold for Grad-CAM masking (default: 0.3)")

    args = parser.parse_args()

    stats = visualize_improved_comparison(
        args.image,
        args.classifier_ckpt,
        args.output_dir,
        args.threshold_percentile,
        args.gradcam_threshold
    )


if __name__ == "__main__":
    main()
