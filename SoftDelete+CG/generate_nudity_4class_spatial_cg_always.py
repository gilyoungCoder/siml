#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nudity 4-Class Adaptive Spatial Classifier Guidance (ALWAYS GUIDANCE VERSION)

4 classes:
  0: benign (no people)
  1: safe_clothed (person with clothes)
  2: harm_nude (nudity)
  3: harm_color (color artifacts/distortions)

Guidance Logic (ALWAYS version - no skip):
  At each step:
  1. Compute classifier logits for current latent
  2. ALWAYS select max among harm classes (2, 3) regardless of overall max
  3. Compute GradCAM heatmap for the selected harm class
  4. Create spatial mask using adaptive threshold
  5. Apply bidirectional guidance: safe_grad - harm_grad

Key Features:
  - ALWAYS applies guidance (never skips even if benign/safe is max)
  - Selects dominant harm class among harm classes only
  - Spatial masking: guidance applied only to harmful regions
  - Bidirectional guidance: pull to safe + push from harmful
  - Adaptive spatial threshold scheduling

Difference from original:
  - Original: skips guidance if max class is benign(0) or safe(1)
  - This version: always applies guidance using max harm class
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
# Harm Class Detector (4-class) - ALWAYS GUIDANCE VERSION
# =========================
class HarmClassDetector:
    """
    Always selects max harm class for guidance (no skip).
    Even if max logit is benign/safe, still pick max among harm classes.
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
        self.harm_classes = config["harm_classes"]
        self.target_safe = config["guidance_target_safe"]
        self.class_names = config["class_names"]

    def detect_harm(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor
    ) -> Tuple[bool, Optional[int], int, Dict]:
        """
        Always return harm class (max among harm classes) for guidance.
        Never skips guidance - always applies guidance using max harm class.

        Args:
            latent: [B, 4, H, W] latent tensor
            timestep: Current timestep

        Returns:
            is_harmful: Always True (always apply guidance)
            harm_class: Index of max harm class (among harm classes only)
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

            # Find max class (for logging)
            max_class = logits.argmax(dim=1)[0].item()
            original_is_harmful = max_class in self.harm_classes

            # ALWAYS select max among harm classes (no skip)
            harm_indices = torch.tensor(self.harm_classes, device=logits.device)
            harm_logits = logits[:, harm_indices]  # [B, num_harm_classes]
            max_harm_idx = harm_logits.argmax(dim=1)[0].item()
            harm_class = self.harm_classes[max_harm_idx]

            safe_class = self.target_safe

            # Prepare info dict
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
            info = {
                "all_logits": logits[0].detach().cpu().numpy(),
                "all_probs": probs,
                "max_class": max_class,
                "max_class_name": self.class_names[max_class],
                "max_logit": logits[0, max_class].item(),
                "max_prob": probs[max_class],
                "original_is_harmful": original_is_harmful,  # was max class harmful?
                "is_harmful": True,  # Always True in this version
                "harm_class": harm_class,
                "harm_class_name": self.class_names[harm_class],
                "harm_logit": logits[0, harm_class].item(),
                "safe_class": safe_class,
                "safe_class_name": self.class_names[safe_class]
            }

        # Always return True for is_harmful (always apply guidance)
        return True, harm_class, safe_class, info


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
# Selective Spatial Mask Generator (4-class)
# =========================
class SelectiveSpatialMaskGenerator:
    """
    Generates spatial masks only when harmful content detected.
    Uses GradCAM on the detected harm class.
    """

    def __init__(
        self,
        classifier_model,
        harm_detector: HarmClassDetector,
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
            'harmful_steps': 0,
            'guidance_applied': 0,
            'step_history': [],
            'harm_class_history': []
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

    def generate_mask(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        spatial_threshold: float,
        current_step: Optional[int] = None,
        return_heatmap: bool = False
    ) -> Tuple[bool, Optional[torch.Tensor], Optional[torch.Tensor], Optional[int], int, Dict]:
        """
        Generate spatial mask if harmful content detected.

        Returns:
            should_guide: Whether guidance should be applied
            mask: [B, H, W] binary mask (None if not harmful)
            heatmap: [B, H, W] optional heatmap
            harm_class: Detected harm class (None if safe)
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

        # Detect if harmful
        is_harmful, harm_class, safe_class, detection_info = self.harm_detector.detect_harm(
            latent=latent,
            timestep=timestep
        )

        # Update statistics
        self.stats['total_steps'] += 1
        if is_harmful:
            self.stats['harmful_steps'] += 1

        # If not harmful, skip guidance
        if not is_harmful:
            step_info = {
                'step': current_step,
                'spatial_threshold': spatial_threshold,
                'is_harmful': False,
                'max_class': detection_info['max_class'],
                'max_class_name': detection_info['max_class_name'],
                'mask_ratio': 0.0,
                'heatmap': None
            }
            self.stats['step_history'].append(step_info)
            self.stats['harm_class_history'].append(None)

            if self.debug:
                print(f"  [Step {current_step}] Safe ({detection_info['max_class_name']}) - Skipping guidance")

            return False, None, None, None, safe_class, detection_info

        # Harmful detected -> generate GradCAM mask
        self.stats['guidance_applied'] += 1

        # Prepare inputs
        latent_input = latent.to(dtype=self.classifier_dtype)
        norm_timestep = timestep.float() / 1000.0

        # Decide normalize flag: use raw if we have stats
        use_abs = (self.gradcam_stats_map is not None) and (harm_class in self.gradcam_stats_map)
        gradcam_normalize_flag = not use_abs

        with torch.enable_grad():
            heatmap, info = self.gradcam.generate_heatmap(
                latent=latent_input,
                timestep=norm_timestep,
                target_class=harm_class,
                normalize=gradcam_normalize_flag
            )

        # Apply class-conditional CDF normalization if available
        if use_abs:
            stats = self.gradcam_stats_map[harm_class]
            heatmap = self._apply_cdf_normalization(heatmap, stats["mean"], stats["std"])

        # Create binary mask using threshold
        mask = (heatmap >= spatial_threshold).float()

        # Track statistics
        mask_ratio = mask.mean().item()
        heatmap_mean = heatmap.mean().item()

        step_info = {
            'step': current_step,
            'spatial_threshold': spatial_threshold,
            'is_harmful': True,
            'harm_class': harm_class,
            'harm_class_name': detection_info['harm_class_name'],
            'safe_class': safe_class,
            'mask_ratio': mask_ratio,
            'heatmap_mean': heatmap_mean,
            'heatmap_max': heatmap.max().item(),
            'heatmap': heatmap.detach().cpu() if return_heatmap else None
        }
        self.stats['step_history'].append(step_info)
        self.stats['harm_class_history'].append(harm_class)

        if self.debug:
            print(f"  [Step {current_step}] Harmful ({detection_info['harm_class_name']}) - "
                  f"thr={spatial_threshold:.3f}, mask={mask_ratio:.1%}")

        if return_heatmap:
            return True, mask, heatmap, harm_class, safe_class, detection_info
        else:
            return True, mask, None, harm_class, safe_class, detection_info

    def get_statistics(self) -> Dict:
        return self.stats.copy()

    def reset_statistics(self):
        self.stats = {
            'total_steps': 0,
            'harmful_steps': 0,
            'guidance_applied': 0,
            'step_history': [],
            'harm_class_history': []
        }


# =========================
# Selective Spatial Guidance (4-class)
# =========================
class SelectiveSpatialGuidance:
    """
    Applies bidirectional classifier guidance only to harmful regions.
    Guidance direction: safe_grad - harm_grad (pull to safe, push from harm)
    """

    def __init__(
        self,
        classifier_model,
        config: Dict = NUDITY_4CLASS_CONFIG,
        device: str = "cuda",
        use_bidirectional: bool = True
    ):
        self.classifier = classifier_model
        self.config = config
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
        harm_class: int,
        safe_class: int,
        guidance_scale: float = 5.0,
        harmful_scale: float = 1.0,
        base_guidance_scale: float = 0.0
    ) -> torch.Tensor:
        """
        Compute spatially-masked bidirectional gradient.

        Gradient = safe_grad - harmful_scale * harm_grad
        Applied with spatial weighting:
          - Harmful regions (mask=1): guidance_scale
          - Non-harmful regions (mask=0): base_guidance_scale
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

            # Normalize timestep
            norm_timestep = timestep.float() / 1000.0

            if self.use_bidirectional:
                # Bidirectional: safe_grad - harmful_scale * harm_grad
                latent_for_safe = latent_input.detach().requires_grad_(True)
                logits_safe = self.classifier(latent_for_safe, norm_timestep)
                safe_logit = logits_safe[:, safe_class].sum()
                grad_safe = torch.autograd.grad(safe_logit, latent_for_safe)[0]

                latent_for_harmful = latent_input.detach().requires_grad_(True)
                logits_harmful = self.classifier(latent_for_harmful, norm_timestep)
                harmful_logit = logits_harmful[:, harm_class].sum()
                grad_harmful = torch.autograd.grad(harmful_logit, latent_for_harmful)[0]

                # -(harm_grad - safe_grad) = safe_grad - harm_grad
                grad = grad_safe - harmful_scale * grad_harmful
            else:
                # Unidirectional: pull to safe only
                logits = self.classifier(latent_input, norm_timestep)
                safe_logit = logits[:, safe_class].sum()
                grad = torch.autograd.grad(safe_logit, latent_input)[0]

        # Spatially-weighted guidance
        mask_expanded = spatial_mask.unsqueeze(1)  # [B,1,H,W]
        weight_map = mask_expanded * guidance_scale + (1 - mask_expanded) * base_guidance_scale
        weighted_grad = grad * weight_map
        weighted_grad = weighted_grad.to(dtype=latent.dtype)

        return weighted_grad.detach()

    def apply_guidance(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        spatial_mask: torch.Tensor,
        harm_class: int,
        safe_class: int,
        guidance_scale: float = 5.0,
        harmful_scale: float = 1.0,
        base_guidance_scale: float = 0.0
    ) -> torch.Tensor:
        weighted_grad = self.compute_gradient(
            latent=latent,
            timestep=timestep,
            spatial_mask=spatial_mask,
            harm_class=harm_class,
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
    parser = ArgumentParser(description="Nudity 4-Class Selective Spatial CG")

    # Model & Generation
    parser.add_argument("ckpt_path", type=str, help="Path to pretrained SD model")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts")
    parser.add_argument("--output_dir", type=str, default="output_img/nudity_4class_spatial_cg",
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

    # Bidirectional guidance
    parser.add_argument("--use_bidirectional", action="store_true", default=True,
                        help="Enable bidirectional guidance (pull to safe + push from harmful)")
    parser.add_argument("--no_bidirectional", action="store_false", dest="use_bidirectional",
                        help="Disable bidirectional guidance")
    parser.add_argument("--harmful_scale", type=float, default=1.0,
                        help="Harmful repulsion scale if bidirectional")
    parser.add_argument("--base_guidance_scale", type=float, default=0.0,
                        help="Base guidance scale for non-harmful regions (0 = no guidance outside mask)")

    # Active step range
    parser.add_argument("--guidance_start_step", type=int, default=0,
                        help="Step to start guidance")
    parser.add_argument("--guidance_end_step", type=int, default=50,
                        help="Step to end guidance")

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


def visualize_selective_guidance(
    mask_generator: SelectiveSpatialMaskGenerator,
    output_dir: Path,
    prefix: str = "selective_guidance",
    generated_image: Optional[Image.Image] = None
):
    """Visualize selective spatial guidance statistics."""
    import matplotlib.pyplot as plt

    stats = mask_generator.get_statistics()
    history = stats.get('step_history', [])

    if not history:
        print("[WARNING] No guidance history to visualize")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = [h['step'] for h in history]
    is_harmful_list = [1 if h['is_harmful'] else 0 for h in history]
    spatial_thresholds = [h['spatial_threshold'] for h in history]
    mask_ratios = [h['mask_ratio'] for h in history]
    harm_classes = [h.get('harm_class', None) for h in history]

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))

    # Plot 1: Harmful detection over time
    ax1 = axes[0]
    colors = ['green' if not h else 'red' for h in is_harmful_list]
    ax1.scatter(steps, is_harmful_list, c=colors, s=50, alpha=0.7)
    ax1.set_xlabel('Denoising Step', fontsize=12)
    ax1.set_ylabel('Is Harmful', fontsize=12)
    ax1.set_title('Harmful Content Detection (Red=Harmful, Green=Safe)', fontsize=14, fontweight='bold')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Safe', 'Harmful'])
    ax1.grid(True, alpha=0.3)

    # Plot 2: Spatial threshold
    ax2 = axes[1]
    ax2.plot(steps, spatial_thresholds, marker='o', linewidth=2, markersize=4)
    ax2.set_xlabel('Denoising Step', fontsize=12)
    ax2.set_ylabel('Spatial Threshold', fontsize=12)
    ax2.set_title('Adaptive Spatial Threshold Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Plot 3: Mask ratio
    ax3 = axes[2]
    ax3.plot(steps, mask_ratios, marker='s', linewidth=2, markersize=4, label='Mask Ratio')
    ax3.plot(steps, spatial_thresholds, linestyle='--', alpha=0.5, label='Threshold')
    ax3.set_xlabel('Denoising Step', fontsize=12)
    ax3.set_ylabel('Ratio', fontsize=12)
    ax3.set_title('Masked Region Coverage', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])

    # Plot 4: Detected harm class
    ax4 = axes[3]
    colors_map = {2: 'red', 3: 'orange', None: 'gray'}
    labels_map = {2: 'harm_nude', 3: 'harm_color', None: 'safe'}
    for hc in [2, 3]:
        mask = [1 if h == hc else 0 for h in harm_classes]
        ax4.scatter([s for s, m in zip(steps, mask) if m == 1],
                    [hc for m in mask if m == 1],
                    c=colors_map[hc], label=labels_map[hc], s=50, alpha=0.7)
    ax4.set_xlabel('Denoising Step', fontsize=12)
    ax4.set_ylabel('Harm Class', fontsize=12)
    ax4.set_title('Detected Harm Class Over Time', fontsize=14, fontweight='bold')
    ax4.set_yticks([2, 3])
    ax4.set_yticklabels(['harm_nude(2)', 'harm_color(3)'])
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / f"{prefix}_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization: {save_path}")
    plt.close()

    # Summary
    from collections import Counter
    harm_cnt = Counter([h for h in harm_classes if h is not None])
    print("\n" + "="*60)
    print("SELECTIVE GUIDANCE SUMMARY")
    print("="*60)
    print(f"Total steps: {stats['total_steps']}")
    print(f"Harmful detected: {stats['harmful_steps']} ({stats['harmful_steps']/max(1,stats['total_steps'])*100:.1f}%)")
    print(f"Guidance applied: {stats['guidance_applied']} ({stats['guidance_applied']/max(1,stats['total_steps'])*100:.1f}%)")
    for hc, c in sorted(harm_cnt.items()):
        name = NUDITY_4CLASS_CONFIG['class_names'][hc]
        print(f"  {name}: {c} steps")
    print("="*60 + "\n")


# =========================
# Main Generation Function
# =========================
def generate_with_selective_guidance(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    mask_generator: SelectiveSpatialMaskGenerator,
    guidance_module: SelectiveSpatialGuidance,
    threshold_scheduler: AdaptiveSpatialThresholdScheduler,
    args,
    output_dir: Path
):
    device = pipe.device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("NUDITY 4-CLASS SELECTIVE SPATIAL CLASSIFIER GUIDANCE")
    print("="*80)
    print(f"Total prompts: {len(prompts)}")
    print(f"Samples per prompt: {args.nsamples}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Spatial threshold: {args.spatial_threshold_start} -> {args.spatial_threshold_end}")
    print(f"Threshold strategy: {args.threshold_strategy}")
    print(f"Bidirectional: {args.use_bidirectional}")
    if args.use_bidirectional:
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

                    # Check if harmful and get mask
                    should_guide, spatial_mask, heatmap, harm_class, safe_class, detection_info = \
                        mask_generator.generate_mask(
                            latent=latents,
                            timestep=timestep,
                            spatial_threshold=spatial_threshold,
                            current_step=step,
                            return_heatmap=args.save_visualizations
                        )

                    # Only apply guidance if harmful detected
                    if should_guide and spatial_mask is not None:
                        guided_latents = guidance_module.apply_guidance(
                            latent=latents,
                            timestep=timestep,
                            spatial_mask=spatial_mask,
                            harm_class=harm_class,
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
                    visualize_selective_guidance(
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
    print("NUDITY 4-CLASS SELECTIVE SPATIAL CG - INITIALIZATION")
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

    # Initialize harm detector
    print(f"\n[5/6] Initializing harm class detector...")
    harm_detector = HarmClassDetector(
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
    print(f"\n[6/6] Initializing selective spatial mask generator...")
    mask_generator = SelectiveSpatialMaskGenerator(
        classifier_model=classifier,
        harm_detector=harm_detector,
        gradcam_layer=args.gradcam_layer,
        device=device,
        debug=args.debug,
        gradcam_stats_map=gradcam_stats_map
    )
    print(f"  Mask generator initialized")

    # Initialize guidance module
    guidance_module = SelectiveSpatialGuidance(
        classifier_model=classifier,
        config=NUDITY_4CLASS_CONFIG,
        device=device,
        use_bidirectional=args.use_bidirectional
    )
    guidance_mode = "bidirectional" if args.use_bidirectional else "unidirectional"
    print(f"  Guidance module initialized ({guidance_mode})")

    # Generate
    generate_with_selective_guidance(
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
