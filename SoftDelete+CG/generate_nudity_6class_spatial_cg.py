#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nudity 6-Class Adaptive Spatial Classifier Guidance

Dynamic Harm-Safe Pair Selection:
  - 6 classes: benign(0), safe_clothed(1), harm_full_nude(2), harm_topless(3),
               harm_lingerie(4), harm_swimwear(5)
  - harm classes: 2, 3, 4, 5 (nudity-related)
  - safe class: 1 (clothed)

At each step:
  1. Compute classifier logits for current latent
  2. Find which harm class has highest activation among harm classes
  3. Use that harm class and safe_clothed(1) for guidance
  4. Apply spatial guidance: push from harm, pull to safe

GradCAM absolute normalization (class-conditional):
  - For the selected harm class c*, load (mean_c*, std_c*)
  - Convert raw heatmap h to CDF value: Phi((h - mean_c*) / std_c*)
  - Compare CDF value against spatial threshold to create mask

Harm-Safe Pairs:
  - All harm classes (2,3,4,5) -> safe_clothed (1)
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
# Nudity 6-Class Configuration
# =========================
NUDITY_6CLASS_CONFIG = {
    # Class indices
    "benign": 0,
    "safe_clothed": 1,
    # Harm classes
    "harm_classes": [2, 3, 4, 5],
    # Safe class (all harm -> clothed)
    "safe_class": 1,
    # Mapping from harm to safe (all map to clothed)
    "harm_to_safe": {2: 1, 3: 1, 4: 1, 5: 1},
    # Class names for logging
    "class_names": {
        0: "benign",
        1: "safe_clothed",
        2: "harm_full_nude",
        3: "harm_topless",
        4: "harm_lingerie",
        5: "harm_swimwear"
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
# Dynamic Harm Class Detector
# =========================
class DynamicHarmClassDetector:
    """
    Detects which harm class is most activated for current latent.
    Uses classifier logits to determine dominant harm category.

    If skip_if_safe=True, returns should_skip=True when max logit is benign(0) or safe(1).
    """

    def __init__(
        self,
        classifier_model,
        config: Dict = NUDITY_6CLASS_CONFIG,
        device: str = "cuda",
        skip_if_safe: bool = False
    ):
        self.classifier = classifier_model
        self.config = config
        self.device = device
        self.classifier_dtype = next(self.classifier.parameters()).dtype
        self.skip_if_safe = skip_if_safe

        self.harm_classes = config["harm_classes"]
        self.harm_to_safe = config["harm_to_safe"]
        self.class_names = config["class_names"]
        self.safe_class = config["safe_class"]
        self.benign_class = config.get("benign", 0)

    def detect_dominant_harm(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor
    ) -> Tuple[bool, Optional[int], int, Dict]:
        """
        Detect which harm class is most activated.

        Args:
            latent: [B, 4, H, W] latent tensor
            timestep: Current timestep

        Returns:
            should_skip: Whether to skip guidance (True if safe/benign and skip_if_safe=True)
            harm_class: Index of dominant harm class (None if skipping)
            safe_class: Index of safe class (always 1 for nudity)
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
            logits = self.classifier(latent_input, norm_timestep)  # [B, 6]

            # Find overall max class
            max_class = logits.argmax(dim=1)[0].item()
            is_safe_or_benign = max_class in [self.benign_class, self.safe_class]

            # Check if we should skip
            should_skip = self.skip_if_safe and is_safe_or_benign

            if should_skip:
                # Skip guidance
                info = {
                    "all_logits": logits[0].detach().cpu().numpy(),
                    "max_class": max_class,
                    "max_class_name": self.class_names[max_class],
                    "should_skip": True,
                    "skip_reason": "max_class is safe/benign"
                }
                return True, None, self.safe_class, info

            # Get logits for harm classes only
            harm_indices = torch.tensor(self.harm_classes, device=logits.device)
            harm_logits = logits[:, harm_indices]  # [B, num_harm]

            # Find dominant harm class (max among harm classes)
            max_harm_idx = harm_logits.argmax(dim=1)  # [B]
            dominant_harm = self.harm_classes[max_harm_idx[0].item()]
            paired_safe = self.safe_class

            # Prepare info dict
            info = {
                "all_logits": logits[0].detach().cpu().numpy(),
                "harm_logits": harm_logits[0].detach().cpu().numpy(),
                "max_class": max_class,
                "max_class_name": self.class_names[max_class],
                "is_safe_or_benign": is_safe_or_benign,
                "dominant_harm": dominant_harm,
                "dominant_harm_name": self.class_names[dominant_harm],
                "paired_safe": paired_safe,
                "paired_safe_name": self.class_names[paired_safe],
                "harm_logit_value": logits[0, dominant_harm].item(),
                "safe_logit_value": logits[0, paired_safe].item(),
                "should_skip": False
            }

        return False, dominant_harm, paired_safe, info


# =========================
# GradCAM stats loader
# =========================
def load_gradcam_stats_map(stats_dir: str, harm_classes: List[int], class_names: Dict[int, str]) -> Dict[int, Dict[str, float]]:
    """
    Load per-class GradCAM statistics JSON files.
    Expected files: gradcam_stats_class{N}.json or gradcam_stats_{classname}_class{N}.json
    Returns: {harm_class: {"mean": float, "std": float}}
    """
    stats_dir = Path(stats_dir)
    stats_map: Dict[int, Dict[str, float]] = {}

    for cls in harm_classes:
        class_name = class_names.get(cls, f"class{cls}")
        # Try different naming conventions
        possible_names = [
            f"gradcam_stats_class{cls}.json",
            f"gradcam_stats_class_{cls}.json",
            f"gradcam_stats_{class_name}_class{cls}.json",
            f"gradcam_stats_{class_name.replace('harm_', '')}_class{cls}.json",
        ]

        found = False
        for fname in possible_names:
            path = stats_dir / fname
            if path.exists():
                with open(path, "r") as f:
                    d = json.load(f)
                if "mean" not in d or "std" not in d:
                    raise ValueError(f"[GradCAM stats invalid] {path} must contain keys: mean, std")
                stats_map[cls] = {"mean": float(d["mean"]), "std": float(d["std"])}
                found = True
                break

        if not found:
            print(f"[WARNING] GradCAM stats not found for class {cls} ({class_name}), will use per-image normalization")

    return stats_map


# =========================
# Dynamic Spatial Mask Generator (6-class)
# =========================
class DynamicSpatialMaskGenerator:
    """
    Generates spatial masks with dynamic harm class selection.
    Uses GradCAM on the currently dominant harm class.
    """

    def __init__(
        self,
        classifier_model,
        harm_detector: DynamicHarmClassDetector,
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

        # Class-conditional stats map
        self.gradcam_stats_map = gradcam_stats_map

        if self.gradcam_stats_map is not None and len(self.gradcam_stats_map) > 0:
            print("  Using class-conditional GradCAM statistics:")
            for cls in sorted(self.gradcam_stats_map.keys()):
                m = self.gradcam_stats_map[cls]["mean"]
                s = self.gradcam_stats_map[cls]["std"]
                name = NUDITY_6CLASS_CONFIG["class_names"].get(cls, f"class{cls}")
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
            'step_history': [],
            'harm_class_history': []
        }

    def _apply_cdf_normalization(self, heatmap: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        """
        heatmap: raw GradCAM (no per-image normalization)
        return: cdf( (heatmap-mean)/std ) in [0,1]
        """
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
        Generate spatial mask for current latent with dynamic harm class.

        Returns:
            should_skip: Whether to skip guidance (True if safe/benign and skip_if_safe=True)
            mask: [B, H, W] binary mask (None if skipping)
            heatmap: [B, H, W] optional heatmap (after normalization used for threshold)
            harm_class: Detected dominant harm class (None if skipping)
            safe_class: Paired safe class
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

        # Detect dominant harm class (may return should_skip=True)
        should_skip, harm_class, safe_class, detection_info = self.harm_detector.detect_dominant_harm(
            latent=latent,
            timestep=timestep
        )

        # If should skip, return early
        if should_skip:
            if current_step is not None:
                self.stats['total_steps'] += 1
                step_info = {
                    'step': current_step,
                    'should_skip': True,
                    'skip_reason': detection_info.get('skip_reason', 'safe/benign'),
                    'max_class': detection_info.get('max_class'),
                    'max_class_name': detection_info.get('max_class_name'),
                }
                self.stats['step_history'].append(step_info)
                self.stats['harm_class_history'].append(None)
            return True, None, None, None, safe_class, detection_info

        # Prepare inputs
        latent_input = latent.to(dtype=self.classifier_dtype)
        norm_timestep = timestep.float() / 1000.0

        # Decide normalize flag for gradcam
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
                'harm_class': harm_class,
                'safe_class': safe_class,
                'harm_class_name': detection_info['dominant_harm_name'],
                'safe_class_name': detection_info['paired_safe_name'],
                'heatmap': heatmap.detach().cpu() if return_heatmap else None
            }
            self.stats['step_history'].append(step_info)
            self.stats['harm_class_history'].append(harm_class)

        if return_heatmap:
            return False, mask, heatmap, harm_class, safe_class, detection_info
        else:
            return False, mask, None, harm_class, safe_class, detection_info

    def get_statistics(self) -> Dict:
        return self.stats.copy()

    def reset_statistics(self):
        self.stats = {
            'total_steps': 0,
            'step_history': [],
            'harm_class_history': []
        }


# =========================
# Dynamic Spatial Guidance (6-class)
# =========================
class DynamicSpatialGuidance:
    """
    Applies classifier guidance with dynamic harm-safe pair selection.
    """

    def __init__(
        self,
        classifier_model,
        config: Dict = NUDITY_6CLASS_CONFIG,
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
        base_guidance_scale: float = 2.0
    ) -> torch.Tensor:
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
                # Bidirectional: pull to safe + push from harmful
                latent_for_safe = latent_input.detach().requires_grad_(True)
                logits_safe = self.classifier(latent_for_safe, norm_timestep)
                safe_logit = logits_safe[:, safe_class].sum()
                grad_safe = torch.autograd.grad(safe_logit, latent_for_safe)[0]

                latent_for_harmful = latent_input.detach().requires_grad_(True)
                logits_harmful = self.classifier(latent_for_harmful, norm_timestep)
                harmful_logit = logits_harmful[:, harm_class].sum()
                grad_harmful = torch.autograd.grad(harmful_logit, latent_for_harmful)[0]

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
        base_guidance_scale: float = 2.0
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
    parser = ArgumentParser(description="Nudity 6-Class Adaptive Spatial CG")

    # Model & Generation
    parser.add_argument("ckpt_path", type=str, help="Path to pretrained SD model")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts")
    parser.add_argument("--output_dir", type=str, default="output_img/nudity_6class_spatial_cg",
                        help="Output directory")

    # Generation parameters
    parser.add_argument("--nsamples", type=int, default=1, help="Samples per prompt")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # Classifier
    parser.add_argument("--classifier_ckpt", type=str, required=True,
                        help="Nudity 6-class classifier checkpoint path")
    parser.add_argument("--num_classes", type=int, default=6,
                        help="Number of classes in classifier")
    parser.add_argument("--gradcam_layer", type=str,
                        default="encoder_model.middle_block.2",
                        help="Target layer for Grad-CAM")

    # GradCAM statistics directory
    parser.add_argument("--gradcam_stats_dir", type=str, default=None,
                        help="Directory containing per-class GradCAM statistics JSON files")

    # Guidance parameters
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="Classifier guidance strength")
    parser.add_argument("--spatial_threshold_start", type=float, default=0.7,
                        help="Initial spatial threshold")
    parser.add_argument("--spatial_threshold_end", type=float, default=0.3,
                        help="Final spatial threshold")
    parser.add_argument("--threshold_strategy", type=str, default="linear_decrease",
                        choices=["constant", "linear_decrease", "linear_increase", "cosine_anneal"],
                        help="Spatial threshold scheduling strategy")

    # Bidirectional guidance
    parser.add_argument("--use_bidirectional", action="store_true",
                        help="Enable bidirectional guidance (pull to safe + push from harmful)")
    parser.add_argument("--harmful_scale", type=float, default=1.0,
                        help="Harmful repulsion scale if bidirectional")
    parser.add_argument("--base_guidance_scale", type=float, default=2.0,
                        help="Base guidance scale for non-harmful regions")

    # Active step range
    parser.add_argument("--guidance_start_step", type=int, default=0,
                        help="Step to start guidance")
    parser.add_argument("--guidance_end_step", type=int, default=50,
                        help="Step to end guidance")

    # Skip if safe option
    parser.add_argument("--skip_if_safe", action="store_true",
                        help="Skip guidance if max logit is benign(0) or safe_clothed(1)")

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


def visualize_dynamic_guidance(
    mask_generator: DynamicSpatialMaskGenerator,
    output_dir: Path,
    prefix: str = "dynamic_guidance",
    generated_image: Optional[Image.Image] = None
):
    import matplotlib.pyplot as plt

    stats = mask_generator.get_statistics()
    history = stats.get('step_history', [])

    if not history:
        print("[WARNING] No guidance history to visualize")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = [h['step'] for h in history]
    spatial_thresholds = [h['spatial_threshold'] for h in history]
    mask_ratios = [h['mask_ratio'] for h in history]
    heatmap_means = [h['heatmap_mean'] for h in history]
    harm_classes = [h['harm_class'] for h in history]

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))

    ax1 = axes[0]
    ax1.plot(steps, spatial_thresholds, marker='o', linewidth=2, markersize=4)
    ax1.set_xlabel('Denoising Step', fontsize=12)
    ax1.set_ylabel('Spatial Threshold', fontsize=12)
    ax1.set_title('Adaptive Spatial Threshold Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    ax2 = axes[1]
    ax2.plot(steps, mask_ratios, marker='s', linewidth=2, markersize=4, label='Mask Ratio')
    ax2.plot(steps, spatial_thresholds, linestyle='--', alpha=0.5, label='Threshold')
    ax2.set_xlabel('Denoising Step', fontsize=12)
    ax2.set_ylabel('Ratio', fontsize=12)
    ax2.set_title('Masked Region Coverage', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    ax3 = axes[2]
    colors = {2: 'red', 3: 'orange', 4: 'purple', 5: 'blue'}
    labels = {2: 'full_nude', 3: 'topless', 4: 'lingerie', 5: 'swimwear'}
    for hc in NUDITY_6CLASS_CONFIG['harm_classes']:
        mask = [1 if h == hc else 0 for h in harm_classes]
        ax3.scatter([s for s, m in zip(steps, mask) if m == 1],
                    [hc for m in mask if m == 1],
                    c=colors.get(hc, 'gray'), label=labels.get(hc, f'class{hc}'), s=50, alpha=0.7)
    ax3.set_xlabel('Denoising Step', fontsize=12)
    ax3.set_ylabel('Harm Class', fontsize=12)
    ax3.set_title('Dominant Harm Class Detection Over Time', fontsize=14, fontweight='bold')
    ax3.set_yticks(NUDITY_6CLASS_CONFIG['harm_classes'])
    ax3.set_yticklabels([f'{labels.get(c, c)}({c})' for c in NUDITY_6CLASS_CONFIG['harm_classes']])
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[3]
    ax4.plot(steps, heatmap_means, marker='o', linewidth=2, markersize=4, label='Mean')
    ax4.set_xlabel('Denoising Step', fontsize=12)
    ax4.set_ylabel('Heatmap Value', fontsize=12)
    ax4.set_title('Grad-CAM Heatmap Statistics (post-normalization)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])

    plt.tight_layout()
    save_path = output_dir / f"{prefix}_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization: {save_path}")
    plt.close()

    from collections import Counter
    cnt = Counter(harm_classes)
    print("\n" + "="*60)
    print("DYNAMIC HARM CLASS DETECTION SUMMARY")
    print("="*60)
    print(f"Total steps: {stats['total_steps']}")
    for hc, c in sorted(cnt.items()):
        name = NUDITY_6CLASS_CONFIG['class_names'].get(hc, f'class{hc}')
        print(f"  {name}: {c} steps ({c/len(harm_classes)*100:.1f}%)")
    print("="*60 + "\n")


# =========================
# Main Generation Function
# =========================
def generate_with_dynamic_guidance(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    mask_generator: DynamicSpatialMaskGenerator,
    guidance_module: DynamicSpatialGuidance,
    threshold_scheduler: AdaptiveSpatialThresholdScheduler,
    args,
    output_dir: Path
):
    device = pipe.device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("NUDITY 6-CLASS DYNAMIC SPATIAL CLASSIFIER GUIDANCE")
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

                    should_skip, spatial_mask, heatmap, harm_class, safe_class, detection_info = mask_generator.generate_mask(
                        latent=latents,
                        timestep=timestep,
                        spatial_threshold=spatial_threshold,
                        current_step=step,
                        return_heatmap=args.save_visualizations
                    )

                    # Skip guidance if should_skip=True (safe/benign detected)
                    if should_skip:
                        if args.debug:
                            print(f"  [Step {step}] SKIP - {detection_info.get('max_class_name', 'safe/benign')}")
                        return callback_kwargs

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

                    if args.debug:
                        mask_ratio = spatial_mask.mean().item()
                        print(f"  [Step {step}] thr={spatial_threshold:.3f}, "
                              f"mask={mask_ratio:.1%}, "
                              f"harm={detection_info['dominant_harm_name']}({harm_class}), "
                              f"safe={detection_info['paired_safe_name']}({safe_class})")

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
                    visualize_dynamic_guidance(
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
    print("NUDITY 6-CLASS DYNAMIC SPATIAL CG - INITIALIZATION")
    print("="*80)
    print(f"Model: {args.ckpt_path}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print("="*80 + "\n")

    # Update config based on num_classes if different from default
    if args.num_classes != 6:
        print(f"[INFO] Using {args.num_classes} classes instead of default 6")
        # Adjust harm classes based on num_classes
        # Assuming class 0 is benign, class 1 is safe, rest are harm
        NUDITY_6CLASS_CONFIG["harm_classes"] = list(range(2, args.num_classes))
        NUDITY_6CLASS_CONFIG["harm_to_safe"] = {i: 1 for i in range(2, args.num_classes)}

    # Load GradCAM statistics if provided
    gradcam_stats_map = None
    if args.gradcam_stats_dir is not None:
        print(f"[1/6] Loading per-class GradCAM statistics from dir: {args.gradcam_stats_dir}")
        gradcam_stats_map = load_gradcam_stats_map(
            args.gradcam_stats_dir,
            NUDITY_6CLASS_CONFIG["harm_classes"],
            NUDITY_6CLASS_CONFIG["class_names"]
        )
    else:
        print(f"[1/6] No GradCAM statistics provided -> per-image normalization fallback")

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
    print(f"\n[4/6] Loading nudity {args.num_classes}-class classifier from {args.classifier_ckpt}...")
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=args.num_classes
    ).to(device)
    classifier.eval()
    print(f"  Classifier loaded ({args.num_classes} classes)")

    # Initialize harm detector
    print(f"\n[5/6] Initializing dynamic harm class detector...")
    harm_detector = DynamicHarmClassDetector(
        classifier_model=classifier,
        config=NUDITY_6CLASS_CONFIG,
        device=device,
        skip_if_safe=args.skip_if_safe
    )
    print(f"  Harm classes: {NUDITY_6CLASS_CONFIG['harm_classes']}")
    print(f"  Safe class: {NUDITY_6CLASS_CONFIG['safe_class']}")
    print(f"  Skip if safe: {args.skip_if_safe}")

    # Initialize threshold scheduler
    threshold_scheduler = AdaptiveSpatialThresholdScheduler(
        strategy=args.threshold_strategy,
        start_value=args.spatial_threshold_start,
        end_value=args.spatial_threshold_end,
        total_steps=args.num_inference_steps
    )
    print(f"  Threshold: {args.spatial_threshold_start} -> {args.spatial_threshold_end}")

    # Initialize mask generator
    print(f"\n[6/6] Initializing dynamic spatial mask generator...")
    mask_generator = DynamicSpatialMaskGenerator(
        classifier_model=classifier,
        harm_detector=harm_detector,
        gradcam_layer=args.gradcam_layer,
        device=device,
        debug=args.debug,
        gradcam_stats_map=gradcam_stats_map
    )
    print(f"  Mask generator initialized")

    # Initialize guidance module
    guidance_module = DynamicSpatialGuidance(
        classifier_model=classifier,
        config=NUDITY_6CLASS_CONFIG,
        device=device,
        use_bidirectional=args.use_bidirectional
    )
    guidance_mode = "bidirectional" if args.use_bidirectional else "unidirectional"
    print(f"  Guidance module initialized ({guidance_mode})")

    # Generate
    generate_with_dynamic_guidance(
        pipe=pipe,
        prompts=prompts,
        mask_generator=mask_generator,
        guidance_module=guidance_module,
        threshold_scheduler=threshold_scheduler,
        args=args,
        output_dir=Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
