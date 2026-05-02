#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nudity 4-Class Adaptive Spatial Classifier Guidance with RESTRICTED GRADIENT

4 classes:
  0: benign (no people)
  1: safe_clothed (person with clothes)
  2: harm_nude (nudity)
  3: harm_color (color artifacts/distortions)

Guidance Logic (RESTRICTED GRADIENT version):
  At each step:
  1. Compute classifier logits for current latent
  2. Compute GradCAM for BOTH harm classes (nude=2, color=3)
  3. Check if each harm class exceeds spatial threshold
  4. If BOTH exceed threshold:
     - Apply Restricted Gradient Optimization to remove conflicting directions
     - δ*_nude = g_nude - proj(g_nude onto g_color)
     - δ*_color = g_color - proj(g_color onto g_nude)
     - Combined: δ* = δ*_nude + δ*_color
     - Final: grad = grad_safe - harmful_scale * δ*
  5. If only ONE exceeds threshold:
     - Apply standard single-harm guidance

Key Features:
  - Checks BOTH harm classes for threshold (not just max)
  - Restricted gradient when both are active (removes conflicting directions)
  - Spatial masking: guidance applied only to harmful regions
  - Bidirectional guidance: pull to safe + push from harmful(s)

Based on: generate_nudity_4class_spatial_cg_always.py
"""

import os
import sys
import json
import random
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Dict, Tuple

from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDIMScheduler

from geo_models.classifier.classifier import load_discriminator
from geo_utils.classifier_interpretability import ClassifierGradCAM


# =========================
# Nudity 4-Class Configuration
# =========================
NUDITY_4CLASS_CONFIG = {
    # Class indices
    "benign": 0,
    "safe_clothed": 1,
    "harm_nude": 2,
    "harm_color": 3,

    # Safe classes (no guidance needed)
    "safe_classes": [0, 1],

    # Harm classes (guidance applied)
    "harm_classes": [2, 3],

    # Target safe class for guidance (clothed person)
    "guidance_target_safe": 1,

    # Class names for logging
    "class_names": {
        0: "benign",
        1: "safe_clothed",
        2: "harm_nude",
        3: "harm_color"
    }
}


# =========================
# Adaptive Spatial Threshold Scheduler
# =========================
class AdaptiveSpatialThresholdScheduler:
    """Schedules spatial threshold across denoising steps."""

    def __init__(
        self,
        strategy: str = "linear_decrease",
        start_value: float = 0.7,
        end_value: float = 0.3,
        total_steps: int = 50
    ):
        self.strategy = strategy
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps

    def get_threshold(self, current_step: int) -> float:
        """Get spatial threshold for current step."""
        if self.strategy == "constant":
            return self.start_value

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
# Multi-Harm Class Detector (for Restricted Gradient)
# =========================
class MultiHarmClassDetector:
    """
    Detects BOTH harm classes and checks threshold for each.
    Returns which harm classes exceed the spatial threshold.
    """

    def __init__(
        self,
        classifier_model,
        config: Dict = NUDITY_4CLASS_CONFIG,
        device: str = "cuda"
    ):
        self.classifier = classifier_model
        self.config = config
        self.device = device
        self.classifier_dtype = next(self.classifier.parameters()).dtype

        self.safe_classes = config["safe_classes"]
        self.harm_classes = config["harm_classes"]  # [2, 3]
        self.target_safe = config["guidance_target_safe"]  # 1
        self.class_names = config["class_names"]

    def detect_harm(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor
    ) -> Tuple[Dict[int, float], int, Dict]:
        """
        Detect harm logits for all harm classes.

        Returns:
            harm_logits_dict: {harm_class_idx: logit_value} for all harm classes
            safe_class: Target safe class for guidance
            info: Dict with detection details
        """
        with torch.no_grad():
            latent_input = latent.to(dtype=self.classifier_dtype)

            # Ensure timestep format
            if not isinstance(timestep, torch.Tensor):
                timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
            elif timestep.dim() == 0:
                timestep = timestep.unsqueeze(0)

            B = latent_input.shape[0]
            if timestep.shape[0] != B:
                timestep = timestep.expand(B).to(latent.device)

            # Normalize timestep to match training
            norm_timestep = timestep.float() / 1000.0

            # Get classifier logits
            logits = self.classifier(latent_input, norm_timestep)  # [B, 4]

            # Collect harm logits
            harm_logits_dict = {}
            for harm_cls in self.harm_classes:
                harm_logits_dict[harm_cls] = logits[0, harm_cls].item()

            # Find max class (for logging)
            max_class = logits.argmax(dim=1)[0].item()
            max_harm_class = max(self.harm_classes, key=lambda c: logits[0, c].item())

            safe_class = self.target_safe

            # Prepare info dict
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
            info = {
                "all_logits": logits[0].detach().cpu().numpy(),
                "all_probs": probs,
                "max_class": max_class,
                "max_class_name": self.class_names[max_class],
                "max_harm_class": max_harm_class,
                "max_harm_class_name": self.class_names[max_harm_class],
                "harm_logits": harm_logits_dict,
                "safe_class": safe_class,
                "safe_class_name": self.class_names[safe_class]
            }

        return harm_logits_dict, safe_class, info


# =========================
# GradCAM stats loader (per-class)
# =========================
def load_gradcam_stats_map(stats_dir: str) -> Dict[int, Dict[str, float]]:
    """
    Load per-class GradCAM statistics JSON files:
      - gradcam_stats_harm_nude_class2.json (class 2)
      - gradcam_stats_harm_color_class3.json (class 3)
    Returns: {harm_class: {"mean": float, "std": float}}
    """
    stats_dir = Path(stats_dir)
    mapping = {
        2: "gradcam_stats_harm_nude_class2.json",
        3: "gradcam_stats_harm_color_class3.json",
    }

    stats_map: Dict[int, Dict[str, float]] = {}
    for cls, fname in mapping.items():
        path = stats_dir / fname
        if not path.exists():
            print(f"[Warning] GradCAM stats file not found: {path}")
            continue
        with open(path, "r") as f:
            d = json.load(f)
        if "mean" not in d or "std" not in d:
            raise ValueError(f"[GradCAM stats invalid] {path} must contain keys: mean, std")
        stats_map[cls] = {"mean": float(d["mean"]), "std": float(d["std"])}

    return stats_map


# =========================
# Multi-Harm Spatial Mask Generator (for Restricted Gradient)
# =========================
class MultiHarmSpatialMaskGenerator:
    """
    Generates spatial masks for BOTH harm classes.
    Returns mask + heatmap for each harm class that exceeds threshold.
    """

    def __init__(
        self,
        classifier_model,
        harm_detector: MultiHarmClassDetector,
        gradcam_layer: str = "encoder_model.middle_block.2",
        device: str = "cuda",
        debug: bool = False,
        gradcam_stats_map: Optional[Dict[int, Dict[str, float]]] = None
    ):
        self.classifier = classifier_model
        self.harm_detector = harm_detector
        self.device = device
        self.debug = debug
        self.classifier_dtype = next(self.classifier.parameters()).dtype

        # Per-class GradCAM stats map
        self.gradcam_stats_map = gradcam_stats_map

        if self.gradcam_stats_map is not None:
            print("  Using per-class GradCAM statistics:")
            for cls in sorted(self.gradcam_stats_map.keys()):
                m = self.gradcam_stats_map[cls]["mean"]
                s = self.gradcam_stats_map[cls]["std"]
                name = NUDITY_4CLASS_CONFIG["class_names"][cls]
                print(f"    class {cls} ({name}): mean={m:.4f}, std={s:.4f}")
        else:
            print("  No GradCAM statistics provided -> per-image normalization")

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

        # Statistics tracking
        self.stats = {
            'total_steps': 0,
            'both_harm_steps': 0,
            'single_harm_steps': 0,
            'no_harm_steps': 0,
            'step_history': []
        }

    def _apply_cdf_normalization(self, heatmap: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        """Convert raw heatmap to CDF normalized [0,1] range."""
        z = (heatmap - mean) / (std + 1e-8)
        from torch.distributions import Normal
        normal = Normal(
            torch.tensor(0.0, device=heatmap.device, dtype=heatmap.dtype),
            torch.tensor(1.0, device=heatmap.device, dtype=heatmap.dtype)
        )
        return normal.cdf(z)

    def _generate_heatmap_for_class(
        self,
        latent: torch.Tensor,
        norm_timestep: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        """Generate CDF-normalized heatmap for a specific class."""
        use_abs = (self.gradcam_stats_map is not None) and (target_class in self.gradcam_stats_map)
        gradcam_normalize_flag = not use_abs

        with torch.enable_grad():
            heatmap, _ = self.gradcam.generate_heatmap(
                latent=latent,
                timestep=norm_timestep,
                target_class=target_class,
                normalize=gradcam_normalize_flag
            )

        # Apply class-conditional CDF normalization if available
        if use_abs:
            stats = self.gradcam_stats_map[target_class]
            heatmap = self._apply_cdf_normalization(heatmap, stats["mean"], stats["std"])

        return heatmap

    def generate_masks(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        spatial_threshold: float,
        current_step: Optional[int] = None
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], List[int], int, Dict]:
        """
        Generate spatial masks for ALL harm classes.

        Returns:
            masks_dict: {harm_class: mask} for classes exceeding threshold
            heatmaps_dict: {harm_class: heatmap} for all harm classes
            active_harm_classes: List of harm classes that exceed threshold
            safe_class: Target safe class for guidance
            detection_info: Detection details
        """
        # Ensure timestep is tensor
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)

        B = latent.shape[0]
        if timestep.shape[0] != B:
            timestep = timestep.expand(B)

        # Detect harm logits
        harm_logits_dict, safe_class, detection_info = self.harm_detector.detect_harm(
            latent=latent,
            timestep=timestep
        )

        # Prepare inputs
        latent_input = latent.to(dtype=self.classifier_dtype)
        norm_timestep = timestep.float() / 1000.0

        # Generate heatmaps for BOTH harm classes
        heatmaps_dict = {}
        masks_dict = {}
        active_harm_classes = []

        for harm_class in self.harm_detector.harm_classes:
            heatmap = self._generate_heatmap_for_class(latent_input, norm_timestep, harm_class)
            heatmaps_dict[harm_class] = heatmap

            # Check if exceeds threshold
            mask = (heatmap >= spatial_threshold).float()
            mask_ratio = mask.mean().item()

            if mask_ratio > 0:  # Has some pixels above threshold
                masks_dict[harm_class] = mask
                active_harm_classes.append(harm_class)

        # Update statistics
        self.stats['total_steps'] += 1
        if len(active_harm_classes) == 2:
            self.stats['both_harm_steps'] += 1
        elif len(active_harm_classes) == 1:
            self.stats['single_harm_steps'] += 1
        else:
            self.stats['no_harm_steps'] += 1

        # Build step info
        step_info = {
            'step': current_step,
            'spatial_threshold': spatial_threshold,
            'active_harm_classes': active_harm_classes,
            'num_active': len(active_harm_classes),
            'mask_ratios': {hc: masks_dict[hc].mean().item() for hc in active_harm_classes},
            'heatmap_means': {hc: heatmaps_dict[hc].mean().item() for hc in heatmaps_dict}
        }
        self.stats['step_history'].append(step_info)

        if self.debug:
            active_names = [self.harm_detector.class_names[c] for c in active_harm_classes]
            print(f"  [Step {current_step}] Active: {active_names} - thr={spatial_threshold:.3f}")

        return masks_dict, heatmaps_dict, active_harm_classes, safe_class, detection_info

    def get_statistics(self) -> Dict:
        return self.stats.copy()

    def reset_statistics(self):
        self.stats = {
            'total_steps': 0,
            'both_harm_steps': 0,
            'single_harm_steps': 0,
            'no_harm_steps': 0,
            'step_history': []
        }


# =========================
# Restricted Gradient Guidance (Core Innovation)
# =========================
class RestrictedGradientGuidance:
    """
    Applies Restricted Gradient Optimization when multiple harm classes are active.

    When BOTH harm classes exceed threshold:
      - Compute gradients for each harm class
      - Remove conflicting directions via projection:
        δ*_nude = g_nude - proj(g_nude onto g_color)
        δ*_color = g_color - proj(g_color onto g_nude)
      - Combine: δ* = δ*_nude + δ*_color
      - Final: grad = grad_safe - harmful_scale * δ*

    When only ONE harm class exceeds threshold:
      - Standard single-harm guidance
    """

    def __init__(
        self,
        classifier_model,
        config: Dict = NUDITY_4CLASS_CONFIG,
        device: str = "cuda"
    ):
        self.classifier = classifier_model
        self.config = config
        self.device = device
        self.classifier_dtype = next(self.classifier.parameters()).dtype

        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

    def _compute_class_gradient(
        self,
        latent: torch.Tensor,
        norm_timestep: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        """Compute gradient for a specific class."""
        latent_input = latent.detach().to(dtype=self.classifier_dtype).requires_grad_(True)
        logits = self.classifier(latent_input, norm_timestep)
        class_logit = logits[:, target_class].sum()
        grad = torch.autograd.grad(class_logit, latent_input)[0]
        return grad

    def _project_out(self, g_a: torch.Tensor, g_b: torch.Tensor) -> torch.Tensor:
        """
        Project out g_b's direction from g_a.
        δ*_a = g_a - (g_a^T g_b / ||g_b||^2) * g_b
        """
        # Flatten for dot product
        g_a_flat = g_a.view(-1)
        g_b_flat = g_b.view(-1)

        # Compute projection coefficient
        g_b_norm_sq = torch.dot(g_b_flat, g_b_flat) + 1e-8
        proj_coef = torch.dot(g_a_flat, g_b_flat) / g_b_norm_sq

        # Project out
        delta_a = g_a - proj_coef * g_b

        return delta_a

    def compute_restricted_gradient(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        active_harm_classes: List[int],
        masks_dict: Dict[int, torch.Tensor],
        safe_class: int,
        guidance_scale: float = 5.0,
        harmful_scale: float = 1.0,
        base_guidance_scale: float = 0.0
    ) -> torch.Tensor:
        """
        Compute spatially-masked gradient with restricted optimization.

        If len(active_harm_classes) == 2:
          - Apply restricted gradient (project out conflicting directions)
        If len(active_harm_classes) == 1:
          - Standard single-harm guidance
        If len(active_harm_classes) == 0:
          - Just safe gradient with base_guidance_scale
        """
        with torch.enable_grad():
            # Ensure timestep format
            if not isinstance(timestep, torch.Tensor):
                timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
            elif timestep.dim() == 0:
                timestep = timestep.unsqueeze(0).to(latent.device)
            else:
                timestep = timestep.to(latent.device)

            B = latent.shape[0]
            if timestep.shape[0] != B:
                timestep = timestep.expand(B).to(latent.device)

            norm_timestep = timestep.float() / 1000.0

            # Compute safe gradient
            grad_safe = self._compute_class_gradient(latent, norm_timestep, safe_class)

            if len(active_harm_classes) == 0:
                # No harm detected -> just safe guidance with base scale
                return (base_guidance_scale * grad_safe).to(dtype=latent.dtype).detach()

            elif len(active_harm_classes) == 1:
                # Single harm -> standard guidance
                harm_class = active_harm_classes[0]
                grad_harm = self._compute_class_gradient(latent, norm_timestep, harm_class)

                # grad = grad_safe - harmful_scale * grad_harm
                grad = grad_safe - harmful_scale * grad_harm

                # Apply spatial mask
                mask = masks_dict[harm_class].unsqueeze(1)  # [B, 1, H, W]
                weight_map = mask * guidance_scale + (1 - mask) * base_guidance_scale
                weighted_grad = grad * weight_map

                return weighted_grad.to(dtype=latent.dtype).detach()

            else:
                # BOTH harm classes active -> Restricted Gradient Optimization
                # harm_classes should be [2, 3] for nudity
                harm_class_nude = 2  # harm_nude
                harm_class_color = 3  # harm_color

                grad_nude = self._compute_class_gradient(latent, norm_timestep, harm_class_nude)
                grad_color = self._compute_class_gradient(latent, norm_timestep, harm_class_color)

                # Restricted gradient: remove conflicting directions
                # δ*_nude = g_nude - proj(g_nude onto g_color)
                # δ*_color = g_color - proj(g_color onto g_nude)
                delta_nude = self._project_out(grad_nude, grad_color)
                delta_color = self._project_out(grad_color, grad_nude)

                # Combined harm gradient
                delta_combined = delta_nude + delta_color

                # Final: grad = grad_safe - harmful_scale * delta_combined
                grad = grad_safe - harmful_scale * delta_combined

                # Apply spatial mask (union of both masks)
                mask_nude = masks_dict.get(harm_class_nude, torch.zeros_like(latent[:, 0:1, :, :]))
                mask_color = masks_dict.get(harm_class_color, torch.zeros_like(latent[:, 0:1, :, :]))

                # Union mask: max of both
                combined_mask = torch.max(mask_nude, mask_color).unsqueeze(1) if mask_nude.dim() == 3 else torch.max(mask_nude.unsqueeze(1), mask_color.unsqueeze(1))

                weight_map = combined_mask * guidance_scale + (1 - combined_mask) * base_guidance_scale
                weighted_grad = grad * weight_map

                return weighted_grad.to(dtype=latent.dtype).detach()

    def apply_guidance(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        active_harm_classes: List[int],
        masks_dict: Dict[int, torch.Tensor],
        safe_class: int,
        guidance_scale: float = 5.0,
        harmful_scale: float = 1.0,
        base_guidance_scale: float = 0.0
    ) -> torch.Tensor:
        """Apply restricted gradient guidance to latent."""
        weighted_grad = self.compute_restricted_gradient(
            latent=latent,
            timestep=timestep,
            active_harm_classes=active_harm_classes,
            masks_dict=masks_dict,
            safe_class=safe_class,
            guidance_scale=guidance_scale,
            harmful_scale=harmful_scale,
            base_guidance_scale=base_guidance_scale
        )
        return latent + weighted_grad


# =========================
# Arguments
# =========================
def parse_args():
    parser = ArgumentParser(description="Nudity 4-Class Spatial CG with Restricted Gradient")

    # Model & Generation
    parser.add_argument("ckpt_path", type=str, help="Path to pretrained SD model")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts")
    parser.add_argument("--output_dir", type=str, default="output_img/nudity_4class_restricted",
                        help="Output directory")

    # Generation parameters
    parser.add_argument("--nsamples", type=int, default=1, help="Samples per prompt")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # Classifier
    parser.add_argument("--classifier_ckpt", type=str, required=True,
                        help="Nudity 4-class classifier checkpoint path")
    parser.add_argument("--gradcam_layer", type=str,
                        default="encoder_model.middle_block.2",
                        help="Target layer for Grad-CAM")

    # Per-class GradCAM statistics
    parser.add_argument("--gradcam_stats_dir", type=str, default=None,
                        help="Directory containing per-class GradCAM statistics JSON files")

    # Guidance parameters
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="Classifier guidance strength for harmful regions")
    parser.add_argument("--spatial_threshold_start", type=float, default=0.7,
                        help="Initial spatial threshold (early steps)")
    parser.add_argument("--spatial_threshold_end", type=float, default=0.3,
                        help="Final spatial threshold (late steps)")
    parser.add_argument("--threshold_strategy", type=str, default="linear_decrease",
                        choices=["constant", "linear_decrease", "linear_increase", "cosine_anneal"],
                        help="Spatial threshold scheduling strategy")

    # Harmful gradient scale
    parser.add_argument("--harmful_scale", type=float, default=1.0,
                        help="Scale for harmful gradient repulsion")
    parser.add_argument("--base_guidance_scale", type=float, default=0.0,
                        help="Base guidance scale for non-harmful regions")

    # Active step range
    parser.add_argument("--guidance_start_step", type=int, default=0,
                        help="Step to start guidance")
    parser.add_argument("--guidance_end_step", type=int, default=50,
                        help="Step to end guidance")

    # Debug & Visualization
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Save guidance visualizations")

    args = parser.parse_args()
    return args


# =========================
# Utilities
# =========================
def load_prompts(prompt_file: str) -> List[str]:
    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def save_image(image, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.resize((512, 512))
    image.save(filepath)
    print(f"  Saved: {filepath}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def visualize_restricted_guidance(
    mask_generator: MultiHarmSpatialMaskGenerator,
    output_dir: Path,
    prefix: str = "restricted_guidance",
    generated_image: Optional[Image.Image] = None
):
    """Visualize restricted gradient guidance statistics."""
    import matplotlib.pyplot as plt

    stats = mask_generator.get_statistics()
    history = stats.get('step_history', [])

    if not history:
        print("[WARNING] No guidance history to visualize")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = [h['step'] for h in history]
    num_active = [h['num_active'] for h in history]
    spatial_thresholds = [h['spatial_threshold'] for h in history]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Number of active harm classes
    ax1 = axes[0]
    colors = ['green' if n == 0 else ('orange' if n == 1 else 'red') for n in num_active]
    ax1.scatter(steps, num_active, c=colors, s=50, alpha=0.7)
    ax1.set_xlabel('Denoising Step', fontsize=12)
    ax1.set_ylabel('Num Active Harm Classes', fontsize=12)
    ax1.set_title('Active Harm Classes (Green=0, Orange=1, Red=2 [Restricted])', fontsize=14, fontweight='bold')
    ax1.set_yticks([0, 1, 2])
    ax1.grid(True, alpha=0.3)

    # Plot 2: Spatial threshold
    ax2 = axes[1]
    ax2.plot(steps, spatial_thresholds, marker='o', linewidth=2, markersize=4)
    ax2.set_xlabel('Denoising Step', fontsize=12)
    ax2.set_ylabel('Spatial Threshold', fontsize=12)
    ax2.set_title('Adaptive Spatial Threshold Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Plot 3: Mask ratios per class
    ax3 = axes[2]
    mask_ratios_nude = []
    mask_ratios_color = []
    for h in history:
        mr = h.get('mask_ratios', {})
        mask_ratios_nude.append(mr.get(2, 0))
        mask_ratios_color.append(mr.get(3, 0))

    ax3.plot(steps, mask_ratios_nude, marker='o', linewidth=2, markersize=4, label='harm_nude (2)', color='red')
    ax3.plot(steps, mask_ratios_color, marker='s', linewidth=2, markersize=4, label='harm_color (3)', color='orange')
    ax3.plot(steps, spatial_thresholds, linestyle='--', alpha=0.5, label='Threshold', color='gray')
    ax3.set_xlabel('Denoising Step', fontsize=12)
    ax3.set_ylabel('Mask Ratio', fontsize=12)
    ax3.set_title('Mask Coverage per Harm Class', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])

    plt.tight_layout()
    save_path = output_dir / f"{prefix}_restricted_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization: {save_path}")
    plt.close()

    # Summary
    print("\n" + "="*60)
    print("RESTRICTED GRADIENT GUIDANCE SUMMARY")
    print("="*60)
    print(f"Total steps: {stats['total_steps']}")
    print(f"Both harm active (restricted): {stats['both_harm_steps']} ({stats['both_harm_steps']/max(1,stats['total_steps'])*100:.1f}%)")
    print(f"Single harm active: {stats['single_harm_steps']} ({stats['single_harm_steps']/max(1,stats['total_steps'])*100:.1f}%)")
    print(f"No harm active: {stats['no_harm_steps']} ({stats['no_harm_steps']/max(1,stats['total_steps'])*100:.1f}%)")
    print("="*60 + "\n")


# =========================
# Main Generation Function
# =========================
def generate_with_restricted_guidance(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    mask_generator: MultiHarmSpatialMaskGenerator,
    guidance_module: RestrictedGradientGuidance,
    threshold_scheduler: AdaptiveSpatialThresholdScheduler,
    args,
    output_dir: Path
):
    device = pipe.device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("NUDITY 4-CLASS RESTRICTED GRADIENT GUIDANCE")
    print("="*80)
    print(f"Total prompts: {len(prompts)}")
    print(f"Samples per prompt: {args.nsamples}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Spatial threshold: {args.spatial_threshold_start} -> {args.spatial_threshold_end}")
    print(f"Threshold strategy: {args.threshold_strategy}")
    print(f"Harmful repulsion scale: {args.harmful_scale}")
    print(f"Active steps: {args.guidance_start_step} -> {args.guidance_end_step}")
    print("="*80 + "\n")

    total_images = 0

    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        print(f"\n[Prompt {prompt_idx+1}/{len(prompts)}] {prompt}")

        for sample_idx in range(args.nsamples):
            mask_generator.reset_statistics()

            def callback_on_step_end(pipe, step, timestep, callback_kwargs):
                latents = callback_kwargs["latents"]

                if args.guidance_start_step <= step <= args.guidance_end_step:
                    spatial_threshold = threshold_scheduler.get_threshold(step)

                    # Generate masks for BOTH harm classes
                    masks_dict, heatmaps_dict, active_harm_classes, safe_class, detection_info = \
                        mask_generator.generate_masks(
                            latent=latents,
                            timestep=timestep,
                            spatial_threshold=spatial_threshold,
                            current_step=step
                        )

                    # Apply restricted gradient guidance
                    if len(active_harm_classes) > 0:
                        guided_latents = guidance_module.apply_guidance(
                            latent=latents,
                            timestep=timestep,
                            active_harm_classes=active_harm_classes,
                            masks_dict=masks_dict,
                            safe_class=safe_class,
                            guidance_scale=args.guidance_scale,
                            harmful_scale=args.harmful_scale,
                            base_guidance_scale=args.base_guidance_scale
                        )

                        callback_kwargs["latents"] = guided_latents

                return callback_kwargs

            with torch.no_grad():
                output = pipe(
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.cfg_scale,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=["latents"]
                )

            image = output.images[0]

            safe_prompt = "".join(c if c.isalnum() or c in [' ', '-', '_'] else '_' for c in prompt)
            safe_prompt = safe_prompt[:50].strip().replace(' ', '_')
            filename = f"{prompt_idx:04d}_{sample_idx:02d}_{safe_prompt}.png"
            save_path = output_dir / filename
            save_image(image, save_path)

            total_images += 1

            if args.save_visualizations:
                stats = mask_generator.get_statistics()
                if stats['total_steps'] > 0:
                    viz_dir = output_dir / "visualizations"
                    viz_dir.mkdir(exist_ok=True)
                    visualize_restricted_guidance(
                        mask_generator=mask_generator,
                        output_dir=viz_dir,
                        prefix=f"{prompt_idx:04d}_{sample_idx:02d}_{safe_prompt}",
                        generated_image=image
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
    set_seed(args.seed)

    accelerator = Accelerator()
    device = accelerator.device

    print("\n" + "="*80)
    print("NUDITY 4-CLASS RESTRICTED GRADIENT CG - INITIALIZATION")
    print("="*80)
    print(f"Model: {args.ckpt_path}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print("="*80 + "\n")

    # Load GradCAM statistics (per-class)
    gradcam_stats_map = None
    if args.gradcam_stats_dir is not None:
        print(f"[1/6] Loading per-class GradCAM statistics from: {args.gradcam_stats_dir}")
        gradcam_stats_map = load_gradcam_stats_map(args.gradcam_stats_dir)
    else:
        print(f"[1/6] No GradCAM statistics directory provided -> per-image normalization")

    # Load prompts
    print(f"\n[2/6] Loading prompts from {args.prompt_file}...")
    prompts = load_prompts(args.prompt_file)
    print(f"  Loaded {len(prompts)} prompts")

    # Load pipeline
    print(f"\n[3/6] Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    print(f"  Pipeline loaded")

    # Load classifier
    print(f"\n[4/6] Loading nudity 4-class classifier from {args.classifier_ckpt}...")
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=4  # [benign, safe_clothed, harm_nude, harm_color]
    ).to(device)
    classifier.eval()
    print(f"  Classifier loaded (4 classes)")

    # Initialize multi-harm detector
    print(f"\n[5/6] Initializing multi-harm class detector...")
    harm_detector = MultiHarmClassDetector(
        classifier_model=classifier,
        config=NUDITY_4CLASS_CONFIG,
        device=device
    )
    print(f"  Safe classes: {NUDITY_4CLASS_CONFIG['safe_classes']}")
    print(f"  Harm classes: {NUDITY_4CLASS_CONFIG['harm_classes']}")

    # Initialize threshold scheduler
    threshold_scheduler = AdaptiveSpatialThresholdScheduler(
        strategy=args.threshold_strategy,
        start_value=args.spatial_threshold_start,
        end_value=args.spatial_threshold_end,
        total_steps=args.num_inference_steps
    )
    print(f"  Threshold: {args.spatial_threshold_start} -> {args.spatial_threshold_end} ({args.threshold_strategy})")

    # Initialize mask generator
    print(f"\n[6/6] Initializing multi-harm spatial mask generator...")
    mask_generator = MultiHarmSpatialMaskGenerator(
        classifier_model=classifier,
        harm_detector=harm_detector,
        gradcam_layer=args.gradcam_layer,
        device=device,
        debug=args.debug,
        gradcam_stats_map=gradcam_stats_map
    )
    print(f"  Mask generator initialized (multi-harm)")

    # Initialize restricted gradient guidance module
    guidance_module = RestrictedGradientGuidance(
        classifier_model=classifier,
        config=NUDITY_4CLASS_CONFIG,
        device=device
    )
    print(f"  Guidance module initialized (restricted gradient)")

    # Generate
    generate_with_restricted_guidance(
        pipe=pipe,
        prompts=prompts,
        mask_generator=mask_generator,
        guidance_module=guidance_module,
        threshold_scheduler=threshold_scheduler,
        args=args,
        output_dir=Path(args.output_dir)
    )

    print("\n All done!")


if __name__ == "__main__":
    main()
