#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cluster-Aware Adaptive Spatial Classifier Guidance

Combines:
1. Clustering: Identifies which harmful mode the latent belongs to
2. Spatial Guidance: GradCAM-based spatial masking for targeted guidance

Key Features:
- Cluster-specific guidance scales (different modes get different treatment)
- Spatial masking using Grad-CAM heatmaps
- Adaptive threshold scheduling
- Cluster assignment visualization

Flow:
1. At each denoising step:
   a. Assign latent to nearest cluster (mode detection)
   b. Get cluster-specific guidance scale
   c. Generate Grad-CAM spatial mask
   d. Apply spatially-weighted classifier gradient
   e. Save cluster info for visualization

Hyperparameters:
  - Clustering:
    * centroids_path: Path to pre-computed cluster centroids
    * cluster_scales: Per-cluster guidance scales (optional)

  - Spatial:
    * guidance_scale: Base classifier guidance strength
    * spatial_threshold_start/end: Adaptive spatial threshold range
    * threshold_strategy: Scheduling strategy
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
from typing import List, Optional, Dict
from collections import defaultdict

from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDIMScheduler

from geo_models.classifier.classifier import load_discriminator
from geo_utils.classifier_interpretability import ClassifierGradCAM
from geo_utils.mode_aware_gradient_model import ClusterManager


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
# Cluster-Aware Spatial Mask Generator
# =========================
class ClusterAwareSpatialMaskGenerator:
    """
    Generates spatial masks using Grad-CAM with cluster awareness.
    Tracks which cluster each latent belongs to.
    """

    def __init__(
        self,
        classifier_model,
        cluster_manager: ClusterManager,
        harmful_class: int = 2,
        gradcam_layer: str = "encoder_model.middle_block.2",
        device: str = "cuda",
        debug: bool = False,
        gradcam_stats: Optional[Dict] = None
    ):
        self.classifier = classifier_model
        self.cluster_manager = cluster_manager
        self.harmful_class = harmful_class
        self.device = device
        self.debug = debug
        self.classifier_dtype = next(self.classifier.parameters()).dtype

        # GradCAM statistics for absolute normalization
        self.gradcam_stats = gradcam_stats
        if gradcam_stats:
            print(f"  Using GradCAM statistics: mean={gradcam_stats['mean']:.4f}, std={gradcam_stats['std']:.4f}")

        # Initialize Grad-CAM
        self.gradcam = ClassifierGradCAM(
            classifier_model=classifier_model,
            target_layer_name=gradcam_layer
        )

        self.classifier = self.classifier.to(device)
        self.classifier.eval()
        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

        # Statistics tracking
        self.stats = {
            'total_steps': 0,
            'step_history': [],
            'cluster_counts': defaultdict(int)
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
        Generate spatial mask and detect cluster.

        Returns:
            mask: [B, H, W] binary mask
            heatmap: [B, H, W] optional Grad-CAM heatmap
            cluster_info: dict with cluster assignment info
        """
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)

        B = latent.shape[0]
        if timestep.shape[0] != B:
            timestep = timestep.expand(B)

        # 1. Cluster assignment
        cluster_indices, cluster_distances = self.cluster_manager.get_nearest_cluster(latent.float())
        cluster_idx = cluster_indices[0].item()
        cluster_dist = cluster_distances[0].item()

        # 2. Generate Grad-CAM heatmap
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

        # 3. Create binary mask
        mask = (heatmap >= spatial_threshold).float()

        # 4. Track statistics
        cluster_info = {
            'cluster_idx': cluster_idx,
            'cluster_distance': cluster_dist,
        }

        if current_step is not None:
            self.stats['total_steps'] += 1
            self.stats['cluster_counts'][cluster_idx] += 1

            step_info = {
                'step': current_step,
                'spatial_threshold': spatial_threshold,
                'mask_ratio': mask.mean().item(),
                'heatmap_mean': heatmap.mean().item(),
                'heatmap_max': heatmap.max().item(),
                'cluster_idx': cluster_idx,
                'cluster_distance': cluster_dist,
                'heatmap': heatmap.detach().cpu() if return_heatmap else None
            }
            self.stats['step_history'].append(step_info)

        if return_heatmap:
            return mask, heatmap, cluster_info
        else:
            return mask, None, cluster_info

    def get_statistics(self) -> Dict:
        return self.stats.copy()

    def reset_statistics(self):
        self.stats = {
            'total_steps': 0,
            'step_history': [],
            'cluster_counts': defaultdict(int)
        }


# =========================
# Cluster-Aware Spatial Guidance
# =========================
class ClusterAwareSpatialGuidance:
    """
    Applies classifier guidance with cluster-specific scales and spatial masking.
    """

    def __init__(
        self,
        classifier_model,
        safe_class: int = 1,
        harmful_class: int = 2,
        device: str = "cuda",
        use_bidirectional: bool = True,
        cluster_scales: Optional[Dict[int, float]] = None
    ):
        self.classifier = classifier_model
        self.safe_class = safe_class
        self.harmful_class = harmful_class
        self.device = device
        self.use_bidirectional = use_bidirectional
        self.cluster_scales = cluster_scales or {}
        self.classifier_dtype = next(self.classifier.parameters()).dtype

        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

    def get_cluster_scale(self, cluster_idx: int, base_scale: float) -> float:
        """Get guidance scale for specific cluster."""
        if cluster_idx in self.cluster_scales:
            return self.cluster_scales[cluster_idx]
        return base_scale

    def compute_gradient(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        spatial_mask: torch.Tensor,
        guidance_scale: float = 5.0,
        harmful_scale: float = 1.0,
        base_guidance_scale: float = 2.0,
        cluster_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Compute spatially-weighted classifier gradient."""

        # Apply cluster-specific scale if available
        if cluster_idx is not None:
            guidance_scale = self.get_cluster_scale(cluster_idx, guidance_scale)

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

        mask_expanded = spatial_mask.unsqueeze(1)
        weight_map = mask_expanded * guidance_scale + (1 - mask_expanded) * base_guidance_scale
        weighted_grad = grad * weight_map
        weighted_grad = weighted_grad.to(dtype=latent.dtype)

        return weighted_grad.detach()

    def apply_guidance(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        spatial_mask: torch.Tensor,
        guidance_scale: float = 5.0,
        harmful_scale: float = 1.0,
        base_guidance_scale: float = 2.0,
        cluster_idx: Optional[int] = None
    ) -> torch.Tensor:
        weighted_grad = self.compute_gradient(
            latent=latent,
            timestep=timestep,
            spatial_mask=spatial_mask,
            guidance_scale=guidance_scale,
            harmful_scale=harmful_scale,
            base_guidance_scale=base_guidance_scale,
            cluster_idx=cluster_idx
        )
        return latent + weighted_grad


# =========================
# Arguments
# =========================
def parse_args():
    parser = ArgumentParser(description="Cluster-Aware Adaptive Spatial CG")

    # Model & Generation
    parser.add_argument("ckpt_path", type=str, help="Path to pretrained SD model")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts")
    parser.add_argument("--output_dir", type=str, default="output_img/cluster_spatial_cg")

    # Generation parameters
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)

    # Classifier
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--harmful_class", type=int, default=2)
    parser.add_argument("--safe_class", type=int, default=1)
    parser.add_argument("--gradcam_layer", type=str, default="encoder_model.middle_block.2")
    parser.add_argument("--gradcam_stats_file", type=str, default=None)

    # Clustering
    parser.add_argument("--centroids_path", type=str, required=True,
                        help="Path to cluster centroids (.pt file)")
    parser.add_argument("--cluster_timestep", type=int, default=None,
                        help="Which timestep's centroids to use (default: smallest)")
    parser.add_argument("--cluster_scales", type=str, default=None,
                        help="JSON string for per-cluster scales, e.g., '{\"0\": 15.0, \"1\": 10.0}'")

    # Spatial guidance
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--spatial_threshold_start", type=float, default=0.5)
    parser.add_argument("--spatial_threshold_end", type=float, default=0.2)
    parser.add_argument("--threshold_strategy", type=str, default="cosine_anneal",
                        choices=["constant", "linear_decrease", "linear_increase", "cosine_anneal"])

    # Bidirectional guidance
    parser.add_argument("--use_bidirectional", action="store_true")
    parser.add_argument("--harmful_scale", type=float, default=1.5)
    parser.add_argument("--base_guidance_scale", type=float, default=2.0)

    # Active step range
    parser.add_argument("--guidance_start_step", type=int, default=0)
    parser.add_argument("--guidance_end_step", type=int, default=50)

    # Debug & Visualization
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_visualizations", action="store_true")

    return parser.parse_args()


# =========================
# Utilities
# =========================
def load_prompts(prompt_file: str) -> List[str]:
    with open(prompt_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


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


def visualize_cluster_spatial_guidance(
    mask_generator: ClusterAwareSpatialMaskGenerator,
    output_dir: Path,
    prefix: str,
    generated_image: Optional[Image.Image] = None,
    n_clusters: int = 10
):
    """Visualize cluster assignments and spatial guidance."""
    import matplotlib.pyplot as plt
    import cv2

    stats = mask_generator.get_statistics()
    history = stats.get('step_history', [])
    cluster_counts = stats.get('cluster_counts', {})

    if not history:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    steps = [h['step'] for h in history]
    spatial_thresholds = [h['spatial_threshold'] for h in history]
    mask_ratios = [h['mask_ratio'] for h in history]
    heatmap_means = [h['heatmap_mean'] for h in history]
    cluster_indices = [h['cluster_idx'] for h in history]
    cluster_distances = [h['cluster_distance'] for h in history]

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Cluster assignment over steps
    ax1 = axes[0, 0]
    ax1.scatter(steps, cluster_indices, c=cluster_indices, cmap='tab10', s=50, alpha=0.7)
    ax1.set_xlabel('Denoising Step', fontsize=11)
    ax1.set_ylabel('Cluster Index', fontsize=11)
    ax1.set_title('Cluster Assignment Over Time', fontsize=12, fontweight='bold')
    ax1.set_yticks(range(n_clusters))
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cluster distance over steps
    ax2 = axes[0, 1]
    ax2.plot(steps, cluster_distances, marker='o', linewidth=2, markersize=4, color='purple')
    ax2.set_xlabel('Denoising Step', fontsize=11)
    ax2.set_ylabel('Distance to Centroid', fontsize=11)
    ax2.set_title('Cluster Distance Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cluster distribution (pie chart)
    ax3 = axes[1, 0]
    if cluster_counts:
        labels = [f'C{k}' for k in sorted(cluster_counts.keys())]
        sizes = [cluster_counts[k] for k in sorted(cluster_counts.keys())]
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Cluster Distribution', fontsize=12, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No cluster data', ha='center', va='center')
        ax3.set_title('Cluster Distribution', fontsize=12, fontweight='bold')

    # Plot 4: Spatial threshold and mask ratio
    ax4 = axes[1, 1]
    ax4.plot(steps, spatial_thresholds, marker='o', linewidth=2, markersize=4, color='blue', label='Threshold')
    ax4.plot(steps, mask_ratios, marker='s', linewidth=2, markersize=4, color='orange', label='Mask Ratio')
    ax4.set_xlabel('Denoising Step', fontsize=11)
    ax4.set_ylabel('Value', fontsize=11)
    ax4.set_title('Spatial Threshold & Mask Ratio', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])

    plt.tight_layout()
    save_path = output_dir / f"{prefix}_cluster_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Heatmap visualization on generated image
    if generated_image is not None:
        img_array = np.array(generated_image.resize((512, 512)))
        heatmaps_to_visualize = [len(history)//4, len(history)//2, len(history)-1]

        for idx in heatmaps_to_visualize:
            if 0 <= idx < len(history) and history[idx].get('heatmap') is not None:
                step_info = history[idx]
                heatmap = step_info['heatmap'].squeeze().numpy()
                heatmap_upsampled = cv2.resize(heatmap, (512, 512), interpolation=cv2.INTER_LINEAR)
                threshold = step_info['spatial_threshold']
                mask_upsampled = (heatmap_upsampled >= threshold).astype(float)

                fig, axes = plt.subplots(1, 4, figsize=(20, 4))

                axes[0].imshow(img_array)
                axes[0].set_title(f"Generated Image", fontsize=12)
                axes[0].axis('off')

                axes[1].imshow(img_array)
                im1 = axes[1].imshow(heatmap_upsampled, cmap='hot', alpha=0.5, vmin=0, vmax=1)
                axes[1].set_title(f"Step {step_info['step']}: Heatmap\nCluster {step_info['cluster_idx']}", fontsize=12)
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                axes[2].imshow(img_array)
                mask_colored = np.zeros((512, 512, 4))
                mask_colored[:, :, 0] = 1.0
                mask_colored[:, :, 3] = mask_upsampled * 0.6
                axes[2].imshow(mask_colored)
                axes[2].set_title(f"Mask (thresh={threshold:.3f})\nratio={step_info['mask_ratio']:.1%}", fontsize=12)
                axes[2].axis('off')

                import matplotlib.cm as cm
                axes[3].imshow(img_array)
                heatmap_colored = cm.hot(heatmap_upsampled)
                heatmap_colored[:, :, 3] = mask_upsampled * 0.7
                axes[3].imshow(heatmap_colored)
                axes[3].set_title(f"Guided Regions", fontsize=12)
                axes[3].axis('off')

                plt.tight_layout()
                heatmap_path = output_dir / f"{prefix}_heatmap_step{step_info['step']:02d}.png"
                plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                plt.close()

    # Print summary
    print("\n" + "="*60)
    print("CLUSTER-AWARE SPATIAL GUIDANCE SUMMARY")
    print("="*60)
    print(f"Total steps: {stats['total_steps']}")
    print(f"\nCluster distribution:")
    for k, v in sorted(cluster_counts.items()):
        pct = 100 * v / stats['total_steps']
        print(f"  Cluster {k}: {v} steps ({pct:.1f}%)")
    print(f"\nAvg mask ratio: {np.mean(mask_ratios):.1%}")
    print(f"Avg heatmap mean: {np.mean(heatmap_means):.3f}")
    print("="*60 + "\n")


# =========================
# Main Generation Function
# =========================
def generate_with_cluster_spatial_guidance(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    mask_generator: ClusterAwareSpatialMaskGenerator,
    guidance_module: ClusterAwareSpatialGuidance,
    threshold_scheduler: AdaptiveSpatialThresholdScheduler,
    args,
    output_dir: Path
):
    device = pipe.device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("CLUSTER-AWARE ADAPTIVE SPATIAL CLASSIFIER GUIDANCE")
    print("="*80)
    print(f"Total prompts: {len(prompts)}")
    print(f"Samples per prompt: {args.nsamples}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Spatial threshold: {args.spatial_threshold_start} -> {args.spatial_threshold_end}")
    print(f"Centroids: {args.centroids_path}")
    print("="*80 + "\n")

    total_images = 0
    all_cluster_stats = defaultdict(int)

    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        print(f"\n[Prompt {prompt_idx+1}/{len(prompts)}] {prompt}")

        for sample_idx in range(args.nsamples):
            mask_generator.reset_statistics()

            def callback_on_step_end(pipe, step, timestep, callback_kwargs):
                latents = callback_kwargs["latents"]

                if args.guidance_start_step <= step <= args.guidance_end_step:
                    spatial_threshold = threshold_scheduler.get_threshold(step)

                    spatial_mask, heatmap, cluster_info = mask_generator.generate_mask(
                        latent=latents,
                        timestep=timestep,
                        spatial_threshold=spatial_threshold,
                        current_step=step,
                        return_heatmap=args.save_visualizations
                    )

                    guided_latents = guidance_module.apply_guidance(
                        latent=latents,
                        timestep=timestep,
                        spatial_mask=spatial_mask,
                        guidance_scale=args.guidance_scale,
                        harmful_scale=args.harmful_scale,
                        base_guidance_scale=args.base_guidance_scale,
                        cluster_idx=cluster_info['cluster_idx']
                    )

                    callback_kwargs["latents"] = guided_latents

                    if args.debug:
                        print(f"  [Step {step}] cluster={cluster_info['cluster_idx']}, "
                              f"dist={cluster_info['cluster_distance']:.3f}, "
                              f"thresh={spatial_threshold:.3f}, "
                              f"mask={spatial_mask.mean().item():.1%}")

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

            # Aggregate cluster stats
            stats = mask_generator.get_statistics()
            for k, v in stats['cluster_counts'].items():
                all_cluster_stats[k] += v

            if args.save_visualizations:
                if stats['total_steps'] > 0:
                    viz_dir = output_dir / "visualizations"
                    visualize_cluster_spatial_guidance(
                        mask_generator=mask_generator,
                        output_dir=viz_dir,
                        prefix=f"{prompt_idx:04d}_{sample_idx:02d}_{safe_prompt}",
                        generated_image=image,
                        n_clusters=mask_generator.cluster_manager.n_clusters
                    )

    # Final summary
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"Total images: {total_images}")
    print(f"\nOverall cluster distribution:")
    total_steps = sum(all_cluster_stats.values())
    for k, v in sorted(all_cluster_stats.items()):
        pct = 100 * v / total_steps if total_steps > 0 else 0
        print(f"  Cluster {k}: {v} ({pct:.1f}%)")
    print(f"\nOutput: {output_dir}")
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
    print("CLUSTER-AWARE SPATIAL CG - INITIALIZATION")
    print("="*80)

    # Load GradCAM statistics
    gradcam_stats = None
    if args.gradcam_stats_file:
        print(f"[1/7] Loading GradCAM statistics...")
        with open(args.gradcam_stats_file, 'r') as f:
            gradcam_stats = json.load(f)
        print(f"  mean={gradcam_stats['mean']:.6f}, std={gradcam_stats['std']:.6f}")
    else:
        print(f"[1/7] No GradCAM stats file - using per-image normalization")

    # Load prompts
    print(f"\n[2/7] Loading prompts...")
    prompts = load_prompts(args.prompt_file)
    print(f"  Loaded {len(prompts)} prompts")

    # Load pipeline
    print(f"\n[3/7] Loading Stable Diffusion...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Load classifier
    print(f"\n[4/7] Loading classifier...")
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=3
    ).to(device)
    classifier.eval()

    # Load cluster centroids
    print(f"\n[5/7] Loading cluster centroids from {args.centroids_path}...")
    cluster_manager = ClusterManager()
    cluster_manager.load(args.centroids_path, timestep=args.cluster_timestep)

    # Parse cluster scales if provided
    cluster_scales = None
    if args.cluster_scales:
        cluster_scales = {int(k): float(v) for k, v in json.loads(args.cluster_scales).items()}
        print(f"  Cluster-specific scales: {cluster_scales}")

    # Initialize threshold scheduler
    print(f"\n[6/7] Initializing threshold scheduler...")
    threshold_scheduler = AdaptiveSpatialThresholdScheduler(
        strategy=args.threshold_strategy,
        start_value=args.spatial_threshold_start,
        end_value=args.spatial_threshold_end,
        total_steps=args.num_inference_steps
    )

    # Initialize mask generator
    print(f"\n[7/7] Initializing cluster-aware mask generator...")
    mask_generator = ClusterAwareSpatialMaskGenerator(
        classifier_model=classifier,
        cluster_manager=cluster_manager,
        harmful_class=args.harmful_class,
        gradcam_layer=args.gradcam_layer,
        device=device,
        debug=args.debug,
        gradcam_stats=gradcam_stats
    )

    # Initialize guidance module
    guidance_module = ClusterAwareSpatialGuidance(
        classifier_model=classifier,
        safe_class=args.safe_class,
        harmful_class=args.harmful_class,
        device=device,
        use_bidirectional=args.use_bidirectional,
        cluster_scales=cluster_scales
    )

    print(f"\n All modules ready - Starting generation...")

    generate_with_cluster_spatial_guidance(
        pipe=pipe,
        prompts=prompts,
        mask_generator=mask_generator,
        guidance_module=guidance_module,
        threshold_scheduler=threshold_scheduler,
        args=args,
        output_dir=Path(args.output_dir)
    )

    print("\n Done!")


if __name__ == "__main__":
    main()
