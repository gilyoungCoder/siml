#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mode-Aware Adaptive Spatial Classifier Guidance for Machine Unlearning

결합 버전: Clustering Centroid 기반 Mode Detection + GradCAM Spatial Guidance

핵심 아이디어:
  1. Clustering으로 현재 latent가 "유해한 mode"에 속하는지 감지
  2. 유해한 mode일 때만 GradCAM spatial guidance 적용
  3. Mode별로 다른 guidance scale 적용 가능

Technical Flow:
  1. 매 denoising step에서:
     a. 현재 latent가 어느 cluster에 속하는지 판단 (ClusterManager)
     b. 해당 cluster의 guidance scale 결정 (mode_scales)
     c. Grad-CAM으로 유해 영역 탐지 (spatial mask)
     d. GradCAM stats 기반 absolute normalization
     e. 영역별 가중치 gradient 적용

  2. Multi-timestep centroids 지원:
     - 각 timestep에 맞는 centroid 사용
     - 더 정확한 mode 감지

Advantages over single methods:
  - Mode detection: 불필요한 guidance 방지 (benign prompt에는 약하게)
  - Spatial guidance: 정확한 영역 타겟팅 (전체 이미지가 아닌 유해 부분만)
  - GradCAM stats: 일관된 threshold 해석
"""

import os
import sys
import random
import json
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
from geo_utils.mode_aware_gradient_model import ClusterManager


# =========================
# Adaptive Spatial Threshold Scheduler
# =========================
class AdaptiveSpatialThresholdScheduler:
    """
    Schedules spatial threshold across denoising steps.
    """

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
# Mode-Aware Spatial Mask Generator
# =========================
class ModeAwareSpatialMaskGenerator:
    """
    Combines clustering-based mode detection with GradCAM spatial masking.
    """

    def __init__(
        self,
        classifier_model,
        cluster_manager: ClusterManager,
        harmful_class: int = 2,
        gradcam_layer: str = "encoder_model.middle_block.2",
        device: str = "cuda",
        debug: bool = False,
        gradcam_stats: Optional[Dict] = None,
        mode_scales: Optional[Dict[int, float]] = None,
        mode_threshold: float = 0.5,  # Distance threshold for mode detection
    ):
        self.classifier = classifier_model
        self.cluster_manager = cluster_manager
        self.harmful_class = harmful_class
        self.device = device
        self.debug = debug
        self.classifier_dtype = next(self.classifier.parameters()).dtype

        # Mode-specific scales (cluster_id -> scale multiplier)
        self.mode_scales = mode_scales or {}
        self.mode_threshold = mode_threshold

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

        # Statistics tracking
        self.stats = {
            'total_steps': 0,
            'step_history': []
        }

    def get_mode_info(self, latent: torch.Tensor, timestep: int) -> Tuple[int, float, float]:
        """
        Detect which mode (cluster) the current latent belongs to.

        Args:
            latent: [B, 4, H, W] latent tensor
            timestep: Current timestep (0-1000)

        Returns:
            cluster_id: Nearest cluster ID
            distance: Distance to nearest cluster
            scale_multiplier: Guidance scale multiplier for this mode
        """
        with torch.no_grad():
            # Switch to appropriate timestep centroids if available
            if hasattr(self.cluster_manager, '_centroids_dict'):
                available_ts = self.cluster_manager._timesteps
                # Find closest timestep
                closest_t = min(available_ts, key=lambda t: abs(t - timestep))
                if closest_t != getattr(self, '_current_centroid_t', None):
                    self.cluster_manager.set_timestep(closest_t)
                    self._current_centroid_t = closest_t

            # Get nearest cluster
            cluster_ids, distances = self.cluster_manager.get_nearest_cluster(latent.float().cpu())
            cluster_id = cluster_ids[0].item()
            distance = distances[0].item()

        # Get mode-specific scale multiplier
        scale_multiplier = self.mode_scales.get(cluster_id, 1.0)

        return cluster_id, distance, scale_multiplier

    def generate_mask(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        spatial_threshold: float,
        current_step: Optional[int] = None,
        return_heatmap: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Generate spatial mask with mode-aware information.

        Returns:
            mask: [B, H, W] binary mask
            heatmap: [B, H, W] optional Grad-CAM heatmap
            mode_info: Dict with cluster_id, distance, scale_multiplier
        """
        # Get mode information
        timestep_val = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
        cluster_id, distance, scale_multiplier = self.get_mode_info(latent, timestep_val)

        mode_info = {
            'cluster_id': cluster_id,
            'distance': distance,
            'scale_multiplier': scale_multiplier
        }

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
        use_raw = self.gradcam_stats is not None

        with torch.enable_grad():
            heatmap, info = self.gradcam.generate_heatmap(
                latent=latent_input,
                timestep=timestep,
                target_class=self.harmful_class,
                normalize=not use_raw
            )

        # Apply statistics-based normalization if available
        if self.gradcam_stats:
            mean = self.gradcam_stats['mean']
            std = self.gradcam_stats['std']
            heatmap_standardized = (heatmap - mean) / (std + 1e-8)

            from torch.distributions import Normal
            normal = Normal(torch.tensor(0.0, device=heatmap.device),
                           torch.tensor(1.0, device=heatmap.device))
            heatmap = normal.cdf(heatmap_standardized)

        # Create binary mask
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
                'cluster_id': cluster_id,
                'cluster_distance': distance,
                'scale_multiplier': scale_multiplier,
                'heatmap': heatmap.detach().cpu() if return_heatmap else None
            }
            self.stats['step_history'].append(step_info)

        if return_heatmap:
            return mask, heatmap, mode_info
        else:
            return mask, None, mode_info

    def get_statistics(self) -> Dict:
        return self.stats.copy()

    def reset_statistics(self):
        self.stats = {
            'total_steps': 0,
            'step_history': []
        }


# =========================
# Mode-Aware Spatial Guidance
# =========================
class ModeAwareSpatialGuidance:
    """
    Applies classifier guidance with mode-aware scaling and spatial masking.
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
        base_guidance_scale: float = 5.0,
        harmful_scale: float = 1.0,
        mode_scale_multiplier: float = 1.0,
        weak_guidance_scale: float = 2.0
    ) -> torch.Tensor:
        """
        Compute spatially-weighted classifier gradient with mode-aware scaling.

        Args:
            latent: [B, 4, H, W] latent tensor
            timestep: Timestep value
            spatial_mask: [B, H, W] binary mask (1=harmful, 0=safe)
            base_guidance_scale: Base gradient scale for harmful regions
            harmful_scale: Repulsion scale (if bidirectional)
            mode_scale_multiplier: Mode-specific multiplier from cluster
            weak_guidance_scale: Weak gradient for non-harmful regions

        Returns:
            weighted_grad: [B, 4, H, W] gradient to add to latent
        """
        # Apply mode-specific scaling
        effective_scale = base_guidance_scale * mode_scale_multiplier

        with torch.enable_grad():
            latent_input = latent.detach().to(dtype=self.classifier_dtype).requires_grad_(True)

            if not isinstance(timestep, torch.Tensor):
                timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
            elif timestep.dim() == 0:
                timestep = timestep.unsqueeze(0).to(latent.device)
            else:
                timestep = timestep.to(latent.device)

            B = latent_input.shape[0]
            if timestep.shape[0] != B:
                timestep = timestep.expand(B).to(latent.device)

            if self.use_bidirectional:
                latent_for_safe = latent_input.detach().requires_grad_(True)
                logits_safe = self.classifier(latent_for_safe, timestep)
                safe_logit = logits_safe[:, self.safe_class].sum()
                grad_safe = torch.autograd.grad(safe_logit, latent_for_safe)[0]

                latent_for_harmful = latent_input.detach().requires_grad_(True)
                logits_harmful = self.classifier(latent_for_harmful, timestep)
                harmful_logit = logits_harmful[:, self.harmful_class].sum()
                grad_harmful = torch.autograd.grad(harmful_logit, latent_for_harmful)[0]

                grad = grad_safe - harmful_scale * grad_harmful
            else:
                logits = self.classifier(latent_input, timestep)
                safe_logit = logits[:, self.safe_class].sum()
                grad = torch.autograd.grad(safe_logit, latent_input)[0]

        # Create spatially-weighted guidance
        mask_expanded = spatial_mask.unsqueeze(1)  # [B, 1, H, W]
        weight_map = mask_expanded * effective_scale + (1 - mask_expanded) * weak_guidance_scale

        weighted_grad = grad * weight_map
        weighted_grad = weighted_grad.to(dtype=latent.dtype)

        return weighted_grad.detach()

    def apply_guidance(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        spatial_mask: torch.Tensor,
        base_guidance_scale: float = 5.0,
        harmful_scale: float = 1.0,
        mode_scale_multiplier: float = 1.0,
        weak_guidance_scale: float = 2.0
    ) -> torch.Tensor:
        """Apply mode-aware spatially-weighted guidance to latent."""
        weighted_grad = self.compute_gradient(
            latent=latent,
            timestep=timestep,
            spatial_mask=spatial_mask,
            base_guidance_scale=base_guidance_scale,
            harmful_scale=harmful_scale,
            mode_scale_multiplier=mode_scale_multiplier,
            weak_guidance_scale=weak_guidance_scale
        )

        guided_latent = latent + weighted_grad
        return guided_latent


# =========================
# Arguments
# =========================
def parse_args():
    parser = ArgumentParser(description="Mode-Aware Adaptive Spatial CG for Unlearning")

    # Model & Generation
    parser.add_argument("ckpt_path", type=str, help="Path to pretrained SD model")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts")
    parser.add_argument("--output_dir", type=str, default="output_img/mode_aware_spatial_cg",
                        help="Output directory")

    # Generation parameters
    parser.add_argument("--nsamples", type=int, default=1, help="Samples per prompt")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # Classifier
    parser.add_argument("--classifier_ckpt", type=str, required=True,
                        help="Classifier checkpoint path")
    parser.add_argument("--harmful_class", type=int, default=2,
                        help="Harmful class index")
    parser.add_argument("--safe_class", type=int, default=1,
                        help="Safe class for guidance target")
    parser.add_argument("--gradcam_layer", type=str,
                        default="encoder_model.middle_block.2",
                        help="Target layer for Grad-CAM")

    # Clustering centroids (MODE-AWARE)
    parser.add_argument("--centroids_path", type=str, required=True,
                        help="Path to cluster centroids (multi-timestep supported)")
    parser.add_argument("--mode_scales_file", type=str, default=None,
                        help="JSON file with per-cluster guidance scales")

    # GradCAM Statistics
    parser.add_argument("--gradcam_stats_file", type=str, default=None,
                        help="JSON file with GradCAM statistics for absolute normalization")

    # Guidance parameters
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="Base classifier guidance strength")
    parser.add_argument("--spatial_threshold_start", type=float, default=0.7,
                        help="Initial spatial threshold")
    parser.add_argument("--spatial_threshold_end", type=float, default=0.3,
                        help="Final spatial threshold")
    parser.add_argument("--threshold_strategy", type=str, default="linear_decrease",
                        choices=["constant", "linear_decrease", "linear_increase", "cosine_anneal"],
                        help="Spatial threshold scheduling strategy")

    # Bidirectional guidance
    parser.add_argument("--use_bidirectional", action="store_true",
                        help="Enable bidirectional guidance")
    parser.add_argument("--harmful_scale", type=float, default=1.0,
                        help="Harmful repulsion scale if bidirectional")
    parser.add_argument("--base_guidance_scale", type=float, default=2.0,
                        help="Base guidance scale for non-harmful regions")

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


def load_mode_scales(path: Optional[str], n_clusters: int) -> Dict[int, float]:
    """Load per-cluster guidance scales."""
    if path is not None and os.path.exists(path):
        with open(path, 'r') as f:
            scales = json.load(f)
        return {int(k): v for k, v in scales.items()}
    # Default: same scale for all clusters
    return {i: 1.0 for i in range(n_clusters)}


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


def visualize_mode_aware_guidance(
    mask_generator: ModeAwareSpatialMaskGenerator,
    output_dir: Path,
    prefix: str = "mode_aware_guidance",
    generated_image: Optional[Image.Image] = None
):
    """Visualize mode-aware spatial guidance statistics."""
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
    cluster_ids = [h['cluster_id'] for h in history]
    scale_multipliers = [h['scale_multiplier'] for h in history]

    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 14))

    # Plot 1: Cluster Assignment
    ax1 = axes[0]
    ax1.scatter(steps, cluster_ids, c=cluster_ids, cmap='tab10', s=50)
    ax1.set_xlabel('Denoising Step', fontsize=12)
    ax1.set_ylabel('Cluster ID', fontsize=12)
    ax1.set_title('Mode (Cluster) Assignment Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scale Multiplier
    ax2 = axes[1]
    ax2.plot(steps, scale_multipliers, marker='o', linewidth=2, markersize=4, color='purple')
    ax2.set_xlabel('Denoising Step', fontsize=12)
    ax2.set_ylabel('Scale Multiplier', fontsize=12)
    ax2.set_title('Mode-Specific Guidance Scale Multiplier', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Spatial Threshold & Mask Ratio
    ax3 = axes[2]
    ax3.plot(steps, spatial_thresholds, marker='o', linewidth=2, markersize=4, color='blue', label='Threshold')
    ax3.plot(steps, mask_ratios, marker='s', linewidth=2, markersize=4, color='orange', label='Mask Ratio')
    ax3.set_xlabel('Denoising Step', fontsize=12)
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Adaptive Spatial Threshold & Mask Coverage', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])

    # Plot 4: Heatmap Statistics
    ax4 = axes[3]
    ax4.plot(steps, heatmap_means, marker='o', color='green', linewidth=2, markersize=4, label='Mean')
    ax4.set_xlabel('Denoising Step', fontsize=12)
    ax4.set_ylabel('Heatmap Value', fontsize=12)
    ax4.set_title('Grad-CAM Heatmap Mean', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])

    plt.tight_layout()

    save_path = output_dir / f"{prefix}_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: {save_path}")
    plt.close()

    # Print summary
    print("\n" + "="*60)
    print("MODE-AWARE SPATIAL GUIDANCE SUMMARY")
    print("="*60)
    print(f"Total steps:              {stats['total_steps']}")
    print(f"Threshold range:          {spatial_thresholds[0]:.3f} → {spatial_thresholds[-1]:.3f}")
    print(f"Avg mask ratio:           {np.mean(mask_ratios):.1%}")
    print(f"Unique clusters visited:  {len(set(cluster_ids))}")
    print(f"Cluster distribution:     {dict(zip(*np.unique(cluster_ids, return_counts=True)))}")
    print(f"Avg scale multiplier:     {np.mean(scale_multipliers):.2f}")
    print("="*60 + "\n")


# =========================
# Main Generation Function
# =========================
def generate_with_mode_aware_guidance(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    mask_generator: ModeAwareSpatialMaskGenerator,
    guidance_module: ModeAwareSpatialGuidance,
    threshold_scheduler: AdaptiveSpatialThresholdScheduler,
    args,
    output_dir: Path
):
    """Generate images with mode-aware adaptive spatial guidance."""
    device = pipe.device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("MODE-AWARE ADAPTIVE SPATIAL CLASSIFIER GUIDANCE")
    print("="*80)
    print(f"Total prompts: {len(prompts)}")
    print(f"Samples per prompt: {args.nsamples}")
    print(f"Base guidance scale: {args.guidance_scale}")
    print(f"Spatial threshold: {args.spatial_threshold_start} → {args.spatial_threshold_end}")
    print(f"Threshold strategy: {args.threshold_strategy}")
    print(f"Bidirectional: {args.use_bidirectional}")
    print(f"Active steps: {args.guidance_start_step} → {args.guidance_end_step}")
    print(f"Centroids: {args.centroids_path}")
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

                    # Generate mask with mode-aware info
                    spatial_mask, heatmap, mode_info = mask_generator.generate_mask(
                        latent=latents,
                        timestep=timestep,
                        spatial_threshold=spatial_threshold,
                        current_step=step,
                        return_heatmap=args.save_visualizations
                    )

                    # Apply guidance with mode-specific scaling
                    guided_latents = guidance_module.apply_guidance(
                        latent=latents,
                        timestep=timestep,
                        spatial_mask=spatial_mask,
                        base_guidance_scale=args.guidance_scale,
                        harmful_scale=args.harmful_scale,
                        mode_scale_multiplier=mode_info['scale_multiplier'],
                        weak_guidance_scale=args.base_guidance_scale
                    )

                    callback_kwargs["latents"] = guided_latents

                    if args.debug:
                        mask_ratio = spatial_mask.mean().item()
                        print(f"  [Step {step}] cluster={mode_info['cluster_id']}, "
                              f"scale={mode_info['scale_multiplier']:.2f}, "
                              f"threshold={spatial_threshold:.3f}, mask={mask_ratio:.1%}")

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
                    visualize_mode_aware_guidance(
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
    print("MODE-AWARE ADAPTIVE SPATIAL CG - INITIALIZATION")
    print("="*80)
    print(f"Model: {args.ckpt_path}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print("="*80 + "\n")

    # Load GradCAM statistics
    gradcam_stats = None
    if args.gradcam_stats_file:
        print(f"[1/7] Loading GradCAM statistics from {args.gradcam_stats_file}...")
        with open(args.gradcam_stats_file, 'r') as f:
            gradcam_stats = json.load(f)
        print(f"  Mean: {gradcam_stats['mean']:.6f}, Std: {gradcam_stats['std']:.6f}")
    else:
        print(f"[1/7] No GradCAM statistics - using per-image normalization")

    # Load prompts
    print(f"\n[2/7] Loading prompts from {args.prompt_file}...")
    prompts = load_prompts(args.prompt_file)
    print(f"  Loaded {len(prompts)} prompts")

    # Load pipeline
    print(f"\n[3/7] Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    print(f"  Pipeline loaded successfully")

    # Load classifier
    print(f"\n[4/7] Loading classifier from {args.classifier_ckpt}...")
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=3
    ).to(device)
    classifier.eval()
    print(f"  Classifier loaded")

    # Load cluster manager (MODE-AWARE)
    print(f"\n[5/7] Loading cluster centroids from {args.centroids_path}...")
    cluster_manager = ClusterManager()
    cluster_manager.load(args.centroids_path)
    print(f"  Loaded {cluster_manager.n_clusters} clusters")

    # Load mode scales
    mode_scales = load_mode_scales(args.mode_scales_file, cluster_manager.n_clusters)
    print(f"\n[6/7] Mode scales:")
    for cluster_id, scale in mode_scales.items():
        print(f"  Cluster {cluster_id}: {scale:.2f}x")

    # Initialize modules
    print(f"\n[7/7] Initializing modules...")

    threshold_scheduler = AdaptiveSpatialThresholdScheduler(
        strategy=args.threshold_strategy,
        start_value=args.spatial_threshold_start,
        end_value=args.spatial_threshold_end,
        total_steps=args.num_inference_steps
    )

    mask_generator = ModeAwareSpatialMaskGenerator(
        classifier_model=classifier,
        cluster_manager=cluster_manager,
        harmful_class=args.harmful_class,
        gradcam_layer=args.gradcam_layer,
        device=device,
        debug=args.debug,
        gradcam_stats=gradcam_stats,
        mode_scales=mode_scales
    )

    guidance_module = ModeAwareSpatialGuidance(
        classifier_model=classifier,
        safe_class=args.safe_class,
        harmful_class=args.harmful_class,
        device=device,
        use_bidirectional=args.use_bidirectional
    )

    print(f"  All modules initialized")

    # Generate
    print(f"\n✓ Starting generation...")
    output_dir = Path(args.output_dir)

    generate_with_mode_aware_guidance(
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
