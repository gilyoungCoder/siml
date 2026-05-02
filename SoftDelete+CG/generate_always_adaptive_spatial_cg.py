#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Always-On Adaptive Spatial Classifier Guidance for Machine Unlearning

Novel Approach:
  - ALWAYS apply spatial classifier guidance (no detection threshold)
  - ONLY adaptive spatial threshold changes over timesteps
  - Simpler: removes harmful detection logic, focuses on spatial masking

Key Differences from Selective CG:
  1. Selective CG:
     - Detects harmful content first (harmful_threshold)
     - Only applies guidance if detected
     - Two thresholds: harmful_threshold + spatial_threshold

  2. Always-On Adaptive Spatial CG (This):
     - NO harmful detection step
     - ALWAYS applies guidance
     - ONE threshold: adaptive spatial_threshold only
     - Step-dependent spatial threshold scheduling

Benefits:
  - Simpler pipeline (fewer hyperparameters)
  - Consistent intervention across all prompts
  - Adaptive spatial control prevents over/under-correction
  - Early steps: broad spatial coverage (high threshold = less masking)
  - Late steps: fine-grained spatial targeting (low threshold = more masking)

Technical Flow:
  1. Each denoising step:
     a. Compute Grad-CAM heatmap for harmful class
     b. Get adaptive spatial threshold based on current step
     c. Create spatial mask using adaptive threshold
     d. Compute classifier gradient toward safe class
     e. Apply gradient masked to harmful regions

  2. No detection logic - always execute guidance

Hyperparameters (Simplified):
  Essential:
    - guidance_scale: Strength of classifier guidance
    - spatial_threshold_start: Initial spatial threshold (early steps)
    - spatial_threshold_end: Final spatial threshold (late steps)
    - threshold_strategy: How to schedule (linear, cosine, etc.)

  Optional:
    - use_bidirectional: Add harmful repulsion (default: True)
    - harmful_scale: Repulsion strength if bidirectional
    - guidance_start_step / guidance_end_step: Active range
"""

import os
import sys
import random
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Dict

from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDIMScheduler

from geo_models.classifier.classifier import load_discriminator
from geo_utils.classifier_interpretability import ClassifierGradCAM


# =========================
# Adaptive Spatial Threshold Scheduler
# =========================
class AdaptiveSpatialThresholdScheduler:
    """
    Schedules spatial threshold across denoising steps.

    Strategy:
    - Early steps (high noise): Higher threshold → less aggressive masking → broader guidance
    - Late steps (low noise): Lower threshold → more aggressive masking → fine-grained guidance
    """

    def __init__(
        self,
        strategy: str = "linear_decrease",
        start_value: float = 0.7,
        end_value: float = 0.3,
        total_steps: int = 50
    ):
        """
        Args:
            strategy: Scheduling strategy
                - "constant": Fixed threshold
                - "linear_decrease": Linear decrease (recommended for spatial)
                - "linear_increase": Linear increase
                - "cosine_anneal": Smooth cosine transition
            start_value: Threshold at step 0
            end_value: Threshold at final step
            total_steps: Total denoising steps
        """
        self.strategy = strategy
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps

    def get_threshold(self, current_step: int) -> float:
        """Get spatial threshold for current step."""
        if self.strategy == "constant":
            return self.start_value

        # Normalize step to [0, 1]
        t = current_step / max(self.total_steps - 1, 1)

        if self.strategy == "linear_decrease":
            return self.start_value - (self.start_value - self.end_value) * t
        elif self.strategy == "linear_increase":
            return self.start_value + (self.end_value - self.start_value) * t
        elif self.strategy == "cosine_anneal":
            return self.end_value + (self.start_value - self.end_value) * 0.5 * (1 + np.cos(np.pi * t))
        else:
            return self.start_value


# =========================
# Spatial Mask Generator
# =========================
class AdaptiveSpatialMaskGenerator:
    """
    Generates spatial masks using Grad-CAM with adaptive thresholds.
    """

    def __init__(
        self,
        classifier_model,
        harmful_class: int = 2,
        gradcam_layer: str = "encoder_model.middle_block.2",
        device: str = "cuda",
        debug: bool = False,
        gradcam_stats: Optional[Dict] = None
    ):
        self.classifier = classifier_model
        self.harmful_class = harmful_class
        self.device = device
        self.debug = debug
        self.classifier_dtype = next(self.classifier.parameters()).dtype

        # GradCAM statistics for absolute normalization
        self.gradcam_stats = gradcam_stats
        if gradcam_stats:
            print(f"✓ Using GradCAM statistics:")
            print(f"  Mean: {gradcam_stats['mean']:.4f}")
            print(f"  Std:  {gradcam_stats['std']:.4f}")

        # Initialize Grad-CAM
        self.gradcam = ClassifierGradCAM(
            classifier_model=classifier_model,
            target_layer_name=gradcam_layer
        )

        # Ensure model on device
        self.classifier = self.classifier.to(device)
        self.classifier.eval()
        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

        # Statistics tracking for visualization
        self.stats = {
            'total_steps': 0,
            'step_history': []  # Store per-step info
        }

    def generate_mask(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        spatial_threshold: float,
        current_step: Optional[int] = None,
        return_heatmap: bool = False
    ) -> tuple:
        """
        Generate spatial mask for current latent.

        Args:
            latent: [B, 4, H, W] latent tensor
            timestep: [B] or scalar timestep
            spatial_threshold: Threshold for masking (0-1)
            current_step: Current denoising step (for tracking)
            return_heatmap: Whether to return heatmap for visualization

        Returns:
            mask: [B, H, W] binary mask (1 = apply guidance, 0 = skip)
            heatmap: [B, H, W] optional Grad-CAM heatmap
        """
        # Ensure timestep is tensor
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)

        B = latent.shape[0]
        if timestep.shape[0] != B:
            timestep = timestep.expand(B)

        # Generate Grad-CAM heatmap
        latent_input = latent.to(dtype=self.classifier_dtype)

        # Generate GradCAM heatmap
        # If we have statistics, use RAW values for absolute normalization
        # Otherwise, use per-image normalization (old behavior)
        use_raw = self.gradcam_stats is not None

        # Normalize timestep to match training (t / num_train_timesteps)
        norm_timestep = timestep.float() / 1000.0

        with torch.enable_grad():
            heatmap, info = self.gradcam.generate_heatmap(
                latent=latent_input,
                timestep=norm_timestep,
                target_class=self.harmful_class,
                normalize=not use_raw  # Don't normalize if using stats
            )
        # heatmap: [B, H, W]

        # Apply statistics-based normalization if available
        if self.gradcam_stats:
            # Z-score normalization: (x - mean) / std
            mean = self.gradcam_stats['mean']
            std = self.gradcam_stats['std']

            # Standardize
            heatmap_standardized = (heatmap - mean) / (std + 1e-8)

            # Apply Gaussian CDF to get probability [0, 1]
            # Using approximation: CDF(z) ≈ 1 / (1 + exp(-1.702 * z))
            # More accurate: use scipy.stats.norm.cdf, but torch version:
            from torch.distributions import Normal
            normal = Normal(torch.tensor(0.0, device=heatmap.device),
                           torch.tensor(1.0, device=heatmap.device))
            heatmap = normal.cdf(heatmap_standardized)
            # Now heatmap is in [0, 1] range with absolute meaning

        # Create binary mask using threshold
        mask = (heatmap >= spatial_threshold).float()

        # Track statistics
        if current_step is not None:
            self.stats['total_steps'] += 1
            mask_ratio = mask.mean().item()
            heatmap_mean = heatmap.mean().item()

            step_info = {
                'step': current_step,
                'spatial_threshold': spatial_threshold,
                'mask_ratio': mask_ratio,
                'heatmap_mean': heatmap_mean,
                'heatmap_max': heatmap.max().item(),
                'heatmap': heatmap.detach().cpu() if return_heatmap else None
            }
            self.stats['step_history'].append(step_info)

        if return_heatmap:
            return mask, heatmap
        else:
            return mask, None

    def get_statistics(self) -> Dict:
        """Return accumulated statistics."""
        return self.stats.copy()

    def reset_statistics(self):
        """Reset statistics for new generation."""
        self.stats = {
            'total_steps': 0,
            'step_history': []
        }


# =========================
# Spatially Masked Guidance
# =========================
class AlwaysOnSpatialGuidance:
    """
    Always applies classifier guidance with spatial masking.
    No detection logic - guidance applied at every step.
    """

    def __init__(
        self,
        classifier_model,
        safe_class: int = 1,
        harmful_class: int = 2,
        device: str = "cuda",
        use_bidirectional: bool = True
    ):
        self.classifier = classifier_model
        self.safe_class = safe_class
        self.harmful_class = harmful_class
        self.device = device
        self.use_bidirectional = use_bidirectional
        self.classifier_dtype = next(self.classifier.parameters()).dtype

        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

    def compute_gradient(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        spatial_mask: torch.Tensor,
        guidance_scale: float = 5.0,
        harmful_scale: float = 1.0,
        base_guidance_scale: float = 2.0
    ) -> torch.Tensor:
        """
        Compute spatially-weighted classifier gradient.

        Strategy:
        - Harmful regions (mask=1): guidance_scale (강하게)
        - Non-harmful regions (mask=0): base_guidance_scale (약하게)

        Args:
            latent: [B, 4, H, W] latent tensor
            timestep: Timestep value
            spatial_mask: [B, H, W] binary mask (1=harmful, 0=safe)
            guidance_scale: Strong gradient for harmful regions
            harmful_scale: Repulsion scale (if bidirectional)
            base_guidance_scale: Weak gradient for non-harmful regions

        Returns:
            weighted_grad: [B, 4, H, W] gradient to add to latent
        """
        with torch.enable_grad():
            latent_input = latent.detach().to(dtype=self.classifier_dtype).requires_grad_(True)

            # Ensure timestep is tensor
            if not isinstance(timestep, torch.Tensor):
                timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
            elif timestep.dim() == 0:
                timestep = timestep.unsqueeze(0).to(latent.device)
            else:
                timestep = timestep.to(latent.device)

            B = latent_input.shape[0]
            if timestep.shape[0] != B:
                timestep = timestep.expand(B).to(latent.device)

            # Normalize timestep to match training (t / num_train_timesteps)
            norm_timestep = timestep.float() / 1000.0

            if self.use_bidirectional:
                # Bidirectional: pull to safe + push from harmful
                latent_for_safe = latent_input.detach().requires_grad_(True)
                logits_safe = self.classifier(latent_for_safe, norm_timestep)
                safe_logit = logits_safe[:, self.safe_class].sum()
                grad_safe = torch.autograd.grad(safe_logit, latent_for_safe)[0]

                latent_for_harmful = latent_input.detach().requires_grad_(True)
                logits_harmful = self.classifier(latent_for_harmful, norm_timestep)
                harmful_logit = logits_harmful[:, self.harmful_class].sum()
                grad_harmful = torch.autograd.grad(harmful_logit, latent_for_harmful)[0]

                grad = grad_safe - harmful_scale * grad_harmful
            else:
                # Unidirectional: pull to safe only
                logits = self.classifier(latent_input, norm_timestep)
                safe_logit = logits[:, self.safe_class].sum()
                grad = torch.autograd.grad(safe_logit, latent_input)[0]

        # Create spatially-weighted guidance:
        # harmful regions: guidance_scale
        # non-harmful regions: base_guidance_scale
        mask_expanded = spatial_mask.unsqueeze(1)  # [B, 1, H, W]

        # Weight map: harmful=guidance_scale, non-harmful=base_guidance_scale
        weight_map = mask_expanded * guidance_scale + (1 - mask_expanded) * base_guidance_scale

        # Apply weighted guidance
        weighted_grad = grad * weight_map

        # Convert back to latent dtype
        weighted_grad = weighted_grad.to(dtype=latent.dtype)

        return weighted_grad.detach()

    def apply_guidance(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        spatial_mask: torch.Tensor,
        guidance_scale: float = 5.0,
        harmful_scale: float = 1.0,
        base_guidance_scale: float = 2.0
    ) -> torch.Tensor:
        """Apply spatially-weighted guidance to latent."""
        weighted_grad = self.compute_gradient(
            latent=latent,
            timestep=timestep,
            spatial_mask=spatial_mask,
            guidance_scale=guidance_scale,
            harmful_scale=harmful_scale,
            base_guidance_scale=base_guidance_scale
        )

        guided_latent = latent + weighted_grad
        return guided_latent


# =========================
# Arguments
# =========================
def parse_args():
    parser = ArgumentParser(description="Always-On Adaptive Spatial CG for Unlearning")

    # Model & Generation
    parser.add_argument("ckpt_path", type=str, help="Path to pretrained SD model")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts")
    parser.add_argument("--output_dir", type=str, default="output_img/always_adaptive_spatial_cg",
                        help="Output directory")

    # Generation parameters
    parser.add_argument("--nsamples", type=int, default=1, help="Samples per prompt")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # Classifier
    parser.add_argument("--classifier_ckpt", type=str,
                        default="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth",
                        help="Classifier checkpoint path")
    parser.add_argument("--harmful_class", type=int, default=2,
                        help="Harmful class index (2 = nude)")
    parser.add_argument("--safe_class", type=int, default=1,
                        help="Safe class for guidance target (1 = clothed)")
    parser.add_argument("--gradcam_layer", type=str,
                        default="encoder_model.middle_block.2",
                        help="Target layer for Grad-CAM")
    parser.add_argument("--gradcam_stats_file", type=str, default=None,
                        help="JSON file with GradCAM statistics from training data (for absolute normalization)")

    # === SIMPLIFIED HYPERPARAMETERS ===

    # 1. Guidance strength
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="[Core] Classifier guidance strength (default: 5.0)")

    # 2. Adaptive spatial threshold
    parser.add_argument("--spatial_threshold_start", type=float, default=0.7,
                        help="[Core] Initial spatial threshold - early steps (default: 0.7, higher = less masking)")
    parser.add_argument("--spatial_threshold_end", type=float, default=0.3,
                        help="[Core] Final spatial threshold - late steps (default: 0.3, lower = more masking)")
    parser.add_argument("--threshold_strategy", type=str, default="linear_decrease",
                        choices=["constant", "linear_decrease", "linear_increase", "cosine_anneal"],
                        help="[Core] Spatial threshold scheduling strategy (default: linear_decrease)")

    # 3. Bidirectional guidance (optional)
    parser.add_argument("--use_bidirectional", action="store_true",
                        help="[Optional] Enable bidirectional guidance (pull to safe + push from harmful)")
    parser.add_argument("--harmful_scale", type=float, default=1.0,
                        help="[Optional] Harmful repulsion scale if bidirectional (default: 1.0)")
    parser.add_argument("--base_guidance_scale", type=float, default=2.0,
                        help="[Optional] Base guidance scale for non-harmful regions (default: 2.0)")

    # 4. Active step range (optional)
    parser.add_argument("--guidance_start_step", type=int, default=0,
                        help="[Optional] Step to start guidance (default: 0)")
    parser.add_argument("--guidance_end_step", type=int, default=50,
                        help="[Optional] Step to end guidance (default: 50)")

    # Debug & Visualization
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Save Grad-CAM and guidance visualizations")

    args = parser.parse_args()
    return args


# =========================
# Utilities
# =========================
def load_prompts(prompt_file: str) -> List[str]:
    """Load prompts from file."""
    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def save_image(image, filepath: Path):
    """Save PIL image to disk."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.resize((512, 512))
    image.save(filepath)
    print(f"  Saved: {filepath}")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def visualize_adaptive_guidance(
    mask_generator: AdaptiveSpatialMaskGenerator,
    output_dir: Path,
    prefix: str = "adaptive_guidance",
    generated_image: Optional[Image.Image] = None
):
    """
    Visualize adaptive spatial guidance statistics.

    Args:
        mask_generator: AdaptiveSpatialMaskGenerator instance
        output_dir: Directory to save visualizations
        prefix: Filename prefix
        generated_image: Final generated PIL Image for overlay visualization
    """
    import matplotlib.pyplot as plt
    import cv2

    stats = mask_generator.get_statistics()
    history = stats.get('step_history', [])

    if not history:
        print("[WARNING] No guidance history to visualize")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    steps = [h['step'] for h in history]
    spatial_thresholds = [h['spatial_threshold'] for h in history]
    mask_ratios = [h['mask_ratio'] for h in history]
    heatmap_means = [h['heatmap_mean'] for h in history]
    heatmap_maxs = [h['heatmap_max'] for h in history]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Adaptive Spatial Threshold
    ax1 = axes[0]
    ax1.plot(steps, spatial_thresholds, marker='o', linewidth=2, markersize=4, color='blue')
    ax1.set_xlabel('Denoising Step', fontsize=12)
    ax1.set_ylabel('Spatial Threshold', fontsize=12)
    ax1.set_title('Adaptive Spatial Threshold Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Plot 2: Mask Ratio (actual masked region)
    ax2 = axes[1]
    ax2.plot(steps, mask_ratios, marker='s', color='orange', linewidth=2, markersize=4, label='Mask Ratio')
    ax2.plot(steps, spatial_thresholds, linestyle='--', color='blue', alpha=0.5, label='Threshold')
    ax2.set_xlabel('Denoising Step', fontsize=12)
    ax2.set_ylabel('Ratio', fontsize=12)
    ax2.set_title('Masked Region Coverage', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Plot 3: Heatmap Statistics
    ax3 = axes[2]
    ax3.plot(steps, heatmap_means, marker='o', color='green', linewidth=2, markersize=4, label='Mean')
    ax3.plot(steps, heatmap_maxs, marker='^', color='red', linewidth=2, markersize=4, label='Max')
    ax3.axhline(y=spatial_thresholds[0], color='blue', linestyle='--', alpha=0.3, label='Initial Threshold')
    ax3.axhline(y=spatial_thresholds[-1], color='purple', linestyle='--', alpha=0.3, label='Final Threshold')
    ax3.set_xlabel('Denoising Step', fontsize=12)
    ax3.set_ylabel('Heatmap Value', fontsize=12)
    ax3.set_title('Grad-CAM Heatmap Statistics', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])

    plt.tight_layout()

    # Save
    save_path = output_dir / f"{prefix}_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: {save_path}")
    plt.close()

    # Save heatmap visualization for selected steps (if available)
    if generated_image is not None:
        # Convert PIL to numpy array
        img_array = np.array(generated_image.resize((512, 512)))

        # Visualize 3 key steps: early, middle, late
        heatmaps_to_visualize = [len(history)//4, len(history)//2, len(history)-1]
        heatmap_saved = False

        for idx in heatmaps_to_visualize:
            if 0 <= idx < len(history) and history[idx]['heatmap'] is not None:
                step_info = history[idx]
                heatmap = step_info['heatmap'].squeeze().numpy()  # [H, W] latent space (e.g., 64x64)

                # Upsample heatmap to image resolution (512x512)
                heatmap_upsampled = cv2.resize(heatmap, (512, 512), interpolation=cv2.INTER_LINEAR)
                threshold = step_info['spatial_threshold']
                mask_upsampled = (heatmap_upsampled >= threshold).astype(float)

                fig, axes = plt.subplots(1, 4, figsize=(20, 4))

                # 1. Original generated image
                axes[0].imshow(img_array)
                axes[0].set_title(f"Generated Image", fontsize=12)
                axes[0].axis('off')

                # 2. Heatmap on image
                axes[1].imshow(img_array)
                im1 = axes[1].imshow(heatmap_upsampled, cmap='hot', alpha=0.5, vmin=0, vmax=1)
                axes[1].set_title(f"Step {step_info['step']}: Heatmap Overlay\n(mean={step_info['heatmap_mean']:.3f})", fontsize=12)
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                # 3. Binary mask on image
                axes[2].imshow(img_array)
                # Create colored mask (red for masked regions)
                mask_colored = np.zeros((512, 512, 4))
                mask_colored[:, :, 0] = 1.0  # Red channel
                mask_colored[:, :, 3] = mask_upsampled * 0.6  # Alpha channel
                axes[2].imshow(mask_colored)
                axes[2].set_title(f"Binary Mask (threshold={threshold:.3f})\n(ratio={step_info['mask_ratio']:.1%})", fontsize=12)
                axes[2].axis('off')

                # 4. Side-by-side comparison
                axes[3].imshow(img_array)
                # Create hot colormap overlay
                import matplotlib.cm as cm
                heatmap_colored = cm.hot(heatmap_upsampled)
                heatmap_colored[:, :, 3] = mask_upsampled * 0.7  # Only show where masked
                axes[3].imshow(heatmap_colored)
                axes[3].set_title(f"Guided Regions\n(red = strong guidance)", fontsize=12)
                axes[3].axis('off')

                plt.tight_layout()
                heatmap_path = output_dir / f"{prefix}_heatmap_step{step_info['step']:02d}.png"
                plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                plt.close()

                if not heatmap_saved:
                    print(f"✓ Saved heatmap visualizations on generated image")
                    heatmap_saved = True
    else:
        print("[WARNING] No generated image provided for heatmap overlay")

    # Print summary
    print("\n" + "="*60)
    print("ADAPTIVE SPATIAL GUIDANCE SUMMARY")
    print("="*60)
    print(f"Total steps:              {stats['total_steps']}")
    print(f"Threshold range:          {spatial_thresholds[0]:.3f} → {spatial_thresholds[-1]:.3f}")
    print(f"Avg mask ratio:           {np.mean(mask_ratios):.1%}")
    print(f"Avg heatmap mean:         {np.mean(heatmap_means):.3f}")
    print(f"Max heatmap value:        {max(heatmap_maxs):.3f}")

    # Analysis: Check if spatial masking is working properly
    print("\n🔍 Spatial Masking Analysis:")
    high_mask_ratio_steps = sum(1 for r in mask_ratios if r > 0.5)
    low_mask_ratio_steps = sum(1 for r in mask_ratios if r < 0.1)
    print(f"  High mask ratio steps (>50%):  {high_mask_ratio_steps}/{stats['total_steps']}")
    print(f"  Low mask ratio steps (<10%):   {low_mask_ratio_steps}/{stats['total_steps']}")

    # Check if heatmap is spatially localized
    if heatmap_means and heatmap_maxs:
        avg_mean = np.mean(heatmap_means)
        avg_max = np.mean(heatmap_maxs)
        localization_ratio = avg_mean / max(avg_max, 0.001)
        print(f"  Localization ratio (mean/max): {localization_ratio:.3f}")
        if localization_ratio < 0.3:
            print(f"  ✓ Heatmap is LOCALIZED (spatial masking likely working)")
        else:
            print(f"  ⚠ Heatmap is UNIFORM (spatial masking may not be effective)")

    print("="*60 + "\n")


# =========================
# Main Generation Function
# =========================
def generate_with_always_on_guidance(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    mask_generator: AdaptiveSpatialMaskGenerator,
    guidance_module: AlwaysOnSpatialGuidance,
    threshold_scheduler: AdaptiveSpatialThresholdScheduler,
    args,
    output_dir: Path
):
    """
    Generate images with always-on adaptive spatial guidance.
    """
    device = pipe.device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("ALWAYS-ON ADAPTIVE SPATIAL CLASSIFIER GUIDANCE")
    print("="*80)
    print(f"Total prompts: {len(prompts)}")
    print(f"Samples per prompt: {args.nsamples}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Spatial threshold: {args.spatial_threshold_start} → {args.spatial_threshold_end}")
    print(f"Threshold strategy: {args.threshold_strategy}")
    print(f"Bidirectional: {args.use_bidirectional}")
    if args.use_bidirectional:
        print(f"Harmful repulsion scale: {args.harmful_scale}")
    print(f"Active steps: {args.guidance_start_step} → {args.guidance_end_step}")
    print("="*80 + "\n")

    total_images = 0

    # Process each prompt
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        print(f"\n[Prompt {prompt_idx+1}/{len(prompts)}] {prompt}")

        for sample_idx in range(args.nsamples):
            # Reset statistics for this generation
            mask_generator.reset_statistics()

            # Define callback for always-on guidance
            def callback_on_step_end(pipe, step, timestep, callback_kwargs):
                latents = callback_kwargs["latents"]

                # Check if guidance should be applied at this step
                if args.guidance_start_step <= step <= args.guidance_end_step:
                    # Get adaptive spatial threshold for current step
                    spatial_threshold = threshold_scheduler.get_threshold(step)

                    # Generate spatial mask (with tracking for visualization)
                    spatial_mask, heatmap = mask_generator.generate_mask(
                        latent=latents,
                        timestep=timestep,
                        spatial_threshold=spatial_threshold,
                        current_step=step,
                        return_heatmap=args.save_visualizations
                    )

                    # Apply spatially-weighted guidance
                    guided_latents = guidance_module.apply_guidance(
                        latent=latents,
                        timestep=timestep,
                        spatial_mask=spatial_mask,
                        guidance_scale=args.guidance_scale,
                        harmful_scale=args.harmful_scale,
                        base_guidance_scale=args.base_guidance_scale
                    )

                    callback_kwargs["latents"] = guided_latents

                    if args.debug:
                        mask_ratio = spatial_mask.mean().item()
                        print(f"  [Step {step}] threshold={spatial_threshold:.3f}, mask_ratio={mask_ratio:.1%}")

                return callback_kwargs

            # Generate image
            with torch.no_grad():
                output = pipe(
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.cfg_scale,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=["latents"]
                )

            image = output.images[0]

            # Save image
            safe_prompt = "".join(c if c.isalnum() or c in [' ', '-', '_'] else '_' for c in prompt)
            safe_prompt = safe_prompt[:50].strip().replace(' ', '_')
            filename = f"{prompt_idx:04d}_{sample_idx:02d}_{safe_prompt}.png"
            save_path = output_dir / filename
            save_image(image, save_path)

            total_images += 1

            # Save visualization
            if args.save_visualizations:
                stats = mask_generator.get_statistics()
                if stats['total_steps'] > 0:
                    viz_dir = output_dir / "visualizations"
                    viz_dir.mkdir(exist_ok=True)
                    visualize_adaptive_guidance(
                        mask_generator=mask_generator,
                        output_dir=viz_dir,
                        prefix=f"{prompt_idx:04d}_{sample_idx:02d}_{safe_prompt}",
                        generated_image=image  # Pass the generated image
                    )

    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"Total images generated: {total_images}")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")


# =========================
# Main
# =========================
def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup accelerator
    accelerator = Accelerator()
    device = accelerator.device

    print("\n" + "="*80)
    print("ALWAYS-ON ADAPTIVE SPATIAL CG - INITIALIZATION")
    print("="*80)
    print(f"Model: {args.ckpt_path}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print("="*80 + "\n")

    # Load GradCAM statistics (if provided)
    gradcam_stats = None
    if args.gradcam_stats_file:
        print(f"[1/6] Loading GradCAM statistics from {args.gradcam_stats_file}...")
        import json
        with open(args.gradcam_stats_file, 'r') as f:
            gradcam_stats = json.load(f)
        print(f"  Statistics loaded:")
        print(f"    Mean: {gradcam_stats['mean']:.6f}")
        print(f"    Std:  {gradcam_stats['std']:.6f}")
        print(f"  Using ABSOLUTE normalization based on training data")
    else:
        print(f"[1/6] No GradCAM statistics file provided")
        print(f"  Using per-image normalization (old behavior)")

    # Load prompts
    print(f"\n[2/6] Loading prompts from {args.prompt_file}...")
    prompts = load_prompts(args.prompt_file)
    print(f"  Loaded {len(prompts)} prompts")

    # Load Stable Diffusion pipeline
    print(f"\n[2/5] Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    print(f"  Pipeline loaded successfully")

    # Load classifier
    print(f"\n[3/5] Loading classifier from {args.classifier_ckpt}...")
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=3  # [not_people, clothed, nude]
    ).to(device)
    classifier.eval()
    print(f"  Classifier loaded (dtype: float32)")

    # Initialize threshold scheduler
    print(f"\n[4/5] Initializing adaptive spatial threshold scheduler...")
    threshold_scheduler = AdaptiveSpatialThresholdScheduler(
        strategy=args.threshold_strategy,
        start_value=args.spatial_threshold_start,
        end_value=args.spatial_threshold_end,
        total_steps=args.num_inference_steps
    )
    print(f"  Strategy: {args.threshold_strategy}")
    print(f"  Threshold range: {args.spatial_threshold_start} → {args.spatial_threshold_end}")

    # Initialize mask generator
    print(f"\n[6/6] Initializing spatial mask generator...")
    mask_generator = AdaptiveSpatialMaskGenerator(
        classifier_model=classifier,
        harmful_class=args.harmful_class,
        gradcam_layer=args.gradcam_layer,
        device=device,
        debug=args.debug,
        gradcam_stats=gradcam_stats  # Pass statistics for absolute normalization
    )
    print(f"  Mask generator initialized")

    # Initialize guidance module
    guidance_module = AlwaysOnSpatialGuidance(
        classifier_model=classifier,
        safe_class=args.safe_class,
        harmful_class=args.harmful_class,
        device=device,
        use_bidirectional=args.use_bidirectional
    )
    guidance_mode = "bidirectional" if args.use_bidirectional else "unidirectional"
    print(f"  Guidance module initialized ({guidance_mode})")

    # Generate images
    print(f"\n✓ All modules ready - Starting generation...")
    output_dir = Path(args.output_dir)

    generate_with_always_on_guidance(
        pipe=pipe,
        prompts=prompts,
        mask_generator=mask_generator,
        guidance_module=guidance_module,
        threshold_scheduler=threshold_scheduler,
        args=args,
        output_dir=output_dir
    )

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
