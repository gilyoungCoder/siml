#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Concept Always-On Adaptive Spatial Classifier Guidance

Extends the simplified always-on adaptive CG to multiple concepts using
Restricted Gradient (RG) method for conflict resolution.

Key Features:
  - Always applies guidance (no detection threshold)
  - Adaptive spatial threshold scheduling
  - Pixel-wise Restricted Gradient for multi-concept conflicts
  - Simplified hyperparameters (guidance_scale, spatial_threshold, strategy)

Multi-Concept Handling:
  - Single concept: Direct gradient application
  - Two concepts: Pixel-wise RG to resolve conflicts
  - Fallback: Average gradients by direction
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
# Multi-Concept Spatial Mask Generator
# =========================
class MultiConceptSpatialMaskGenerator:
    """Generates spatial masks for multiple concepts."""

    def __init__(
        self,
        classifiers: Dict[str, torch.nn.Module],
        harmful_classes: Dict[str, int],
        gradcam_layer: str = "encoder_model.middle_block.2",
        device: str = "cuda",
        debug: bool = False
    ):
        """
        Args:
            classifiers: Dict of {concept_name: classifier_model}
            harmful_classes: Dict of {concept_name: harmful_class_idx}
            gradcam_layer: Target layer for Grad-CAM
            device: Device to run on
            debug: Enable debug mode
        """
        self.classifiers = classifiers
        self.harmful_classes = harmful_classes
        self.device = device
        self.debug = debug

        # Initialize Grad-CAM for each concept
        self.gradcams = {}
        for concept_name, classifier in classifiers.items():
            self.gradcams[concept_name] = ClassifierGradCAM(
                classifier_model=classifier,
                target_layer_name=gradcam_layer
            )
            classifier.to(device).eval()

        # Statistics tracking
        self.stats = {'total_steps': 0, 'step_history': []}

    def generate_masks(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        spatial_threshold: float,
        current_step: Optional[int] = None,
        return_heatmaps: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generate spatial masks for all concepts.

        Returns:
            masks: Dict of {concept_name: [B, H, W] binary mask}
            heatmaps: Optional dict of {concept_name: [B, H, W] heatmap}
        """
        # Ensure timestep is tensor
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)

        B = latent.shape[0]
        if timestep.shape[0] != B:
            timestep = timestep.expand(B)

        masks = {}
        heatmaps = {}

        for concept_name, gradcam in self.gradcams.items():
            classifier = self.classifiers[concept_name]
            harmful_class = self.harmful_classes[concept_name]
            classifier_dtype = next(classifier.parameters()).dtype

            latent_input = latent.to(dtype=classifier_dtype)

            with torch.enable_grad():
                heatmap, _ = gradcam.generate_heatmap(
                    latent=latent_input,
                    timestep=timestep,
                    target_class=harmful_class,
                    normalize=True
                )

            # Create binary mask
            mask = (heatmap >= spatial_threshold).float()
            masks[concept_name] = mask

            if return_heatmaps:
                heatmaps[concept_name] = heatmap

        # Track statistics
        if current_step is not None:
            self.stats['total_steps'] += 1
            step_info = {
                'step': current_step,
                'spatial_threshold': spatial_threshold,
            }
            for concept_name in masks.keys():
                step_info[f'{concept_name}_mask_ratio'] = masks[concept_name].mean().item()
            self.stats['step_history'].append(step_info)

        if return_heatmaps:
            return masks, heatmaps
        else:
            return masks, None

    def reset_statistics(self):
        """Reset statistics for new generation."""
        self.stats = {'total_steps': 0, 'step_history': []}


# =========================
# Multi-Concept Guidance Manager
# =========================
class MultiConceptGuidanceManager:
    """Manages multi-concept guidance with Restricted Gradient method."""

    def __init__(
        self,
        classifiers: Dict[str, torch.nn.Module],
        safe_classes: Dict[str, int],
        harmful_classes: Dict[str, int],
        device: str = "cuda",
        use_bidirectional: bool = True
    ):
        """
        Args:
            classifiers: Dict of {concept_name: classifier_model}
            safe_classes: Dict of {concept_name: safe_class_idx}
            harmful_classes: Dict of {concept_name: harmful_class_idx}
            device: Device to run on
            use_bidirectional: Use bidirectional guidance (safe pull + harmful push)
        """
        self.classifiers = classifiers
        self.safe_classes = safe_classes
        self.harmful_classes = harmful_classes
        self.device = device
        self.use_bidirectional = use_bidirectional

    def compute_concept_gradients(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        spatial_masks: Dict[str, torch.Tensor],
        guidance_scale: float = 5.0,
        harmful_scale: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute raw gradients for each concept (before masking).

        Returns:
            concept_grads_raw: Dict of {concept_name: [B, C, H, W] gradient}
        """
        # Ensure timestep is tensor
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).to(latent.device)
        else:
            timestep = timestep.to(latent.device)

        B = latent.shape[0]
        if timestep.shape[0] != B:
            timestep = timestep.expand(B).to(latent.device)

        concept_grads_raw = {}

        for concept_name, classifier in self.classifiers.items():
            classifier_dtype = next(classifier.parameters()).dtype

            with torch.enable_grad():
                latent_input = latent.detach().to(dtype=classifier_dtype).requires_grad_(True)

                if self.use_bidirectional:
                    # Pull to safe
                    latent_for_safe = latent_input.detach().requires_grad_(True)
                    logits_safe = classifier(latent_for_safe, timestep)
                    safe_logit = logits_safe[:, self.safe_classes[concept_name]].sum()
                    grad_safe = torch.autograd.grad(safe_logit, latent_for_safe)[0]

                    # Push from harmful
                    latent_for_harmful = latent_input.detach().requires_grad_(True)
                    logits_harmful = classifier(latent_for_harmful, timestep)
                    harmful_logit = logits_harmful[:, self.harmful_classes[concept_name]].sum()
                    grad_harmful = torch.autograd.grad(harmful_logit, latent_for_harmful)[0]

                    grad = grad_safe - harmful_scale * grad_harmful
                else:
                    # Unidirectional
                    logits = classifier(latent_input, timestep)
                    safe_logit = logits[:, self.safe_classes[concept_name]].sum()
                    grad = torch.autograd.grad(safe_logit, latent_input)[0]

            # Convert back to latent dtype and detach
            grad = grad.to(dtype=latent.dtype).detach()
            concept_grads_raw[concept_name] = grad

        return concept_grads_raw

    def combine_gradients_with_rg(
        self,
        concept_grads_raw: Dict[str, torch.Tensor],
        spatial_masks: Dict[str, torch.Tensor],
        guidance_scales: Dict[str, float]
    ) -> torch.Tensor:
        """
        Combine multiple concept gradients using Restricted Gradient method.

        Args:
            concept_grads_raw: Dict of {concept_name: [B, C, H, W] raw gradient}
            spatial_masks: Dict of {concept_name: [B, H, W] binary mask}
            guidance_scales: Dict of {concept_name: guidance_scale}

        Returns:
            combined_gradient: [B, C, H, W] combined gradient tensor
        """
        if len(concept_grads_raw) == 0:
            B, C, H, W = next(iter(spatial_masks.values())).shape[0], 4, 64, 64
            return torch.zeros((B, C, H, W), device=self.device)

        # Single concept case
        if len(concept_grads_raw) == 1:
            concept_name = list(concept_grads_raw.keys())[0]
            grad_raw = concept_grads_raw[concept_name]
            mask = spatial_masks[concept_name]
            scale = guidance_scales[concept_name]

            # Apply mask and scale
            mask_expanded = mask.unsqueeze(1)  # [B, 1, H, W]
            masked_grad = grad_raw * mask_expanded * scale
            return masked_grad

        # Multi-concept case: Pixel-wise RG
        concept_names = list(concept_grads_raw.keys())

        if len(concept_names) == 2:
            # Two concepts: Apply pixel-wise RG
            concept_a_name, concept_b_name = concept_names[0], concept_names[1]

            g_a = concept_grads_raw[concept_a_name]  # [B, C, H, W]
            g_b = concept_grads_raw[concept_b_name]

            mask_a = spatial_masks[concept_a_name]  # [B, H, W]
            mask_b = spatial_masks[concept_b_name]

            scale_a = guidance_scales[concept_a_name]
            scale_b = guidance_scales[concept_b_name]

            B, C, H, W = g_a.shape
            eps = 1e-8

            # Reshape to [B*C, H*W] for vectorized pixel-wise operations
            g_a_flat = g_a.reshape(B * C, H * W)
            g_b_flat = g_b.reshape(B * C, H * W)

            # Compute norms for each pixel
            norms_a = torch.norm(g_a_flat, dim=0)  # [H*W]
            norms_b = torch.norm(g_b_flat, dim=0)

            # Normalize to unit vectors
            dir_a = g_a_flat / (norms_a.unsqueeze(0) + eps)
            dir_b = g_b_flat / (norms_b.unsqueeze(0) + eps)

            # Compute dot products
            dot_products = (dir_a * dir_b).sum(dim=0)  # [H*W]

            # Restricted Gradient (orthogonal projection)
            proj_a_on_b = dot_products.unsqueeze(0) * dir_b
            restricted_a_dir = dir_a - proj_a_on_b

            proj_b_on_a = dot_products.unsqueeze(0) * dir_a
            restricted_b_dir = dir_b - proj_b_on_a

            # Scale back to original magnitudes AND apply concept-specific guidance scale
            restricted_a = restricted_a_dir * norms_a.unsqueeze(0) * scale_a
            restricted_b = restricted_b_dir * norms_b.unsqueeze(0) * scale_b

            # Reshape back to [B, C, H, W]
            restricted_a = restricted_a.reshape(B, C, H, W)
            restricted_b = restricted_b.reshape(B, C, H, W)

            # Identify regions
            mask_a_flat = mask_a.reshape(B, 1, H * W)
            mask_b_flat = mask_b.reshape(B, 1, H * W)

            mask_both_flat = (mask_a_flat * mask_b_flat).bool()  # [B, 1, H*W]
            mask_only_a_flat = ((mask_a_flat > 0) & ~mask_both_flat)
            mask_only_b_flat = ((mask_b_flat > 0) & ~mask_both_flat)

            # Initialize combined gradient
            combined_flat = torch.zeros(B, C, H * W, device=g_a.device, dtype=g_a.dtype)

            # Apply gradients to respective regions
            restricted_a_flat = restricted_a.reshape(B, C, H * W)
            restricted_b_flat = restricted_b.reshape(B, C, H * W)

            # Both region: RG applied
            combined_flat = torch.where(mask_both_flat.expand(-1, C, -1),
                                       restricted_a_flat + restricted_b_flat,
                                       combined_flat)

            # Only A region
            combined_flat = torch.where(mask_only_a_flat.expand(-1, C, -1),
                                       restricted_a_flat,
                                       combined_flat)

            # Only B region
            combined_flat = torch.where(mask_only_b_flat.expand(-1, C, -1),
                                       restricted_b_flat,
                                       combined_flat)

            # Reshape back to [B, C, H, W]
            combined_grad = combined_flat.reshape(B, C, H, W)
            return combined_grad

        else:
            # More than 2 concepts: Fallback to average with masking
            B, C, H, W = next(iter(concept_grads_raw.values())).shape
            combined_grad = torch.zeros((B, C, H, W), device=self.device, dtype=next(iter(concept_grads_raw.values())).dtype)

            for concept_name, grad_raw in concept_grads_raw.items():
                mask = spatial_masks[concept_name]
                scale = guidance_scales[concept_name]  # Use concept-specific scale
                mask_expanded = mask.unsqueeze(1)  # [B, 1, H, W]
                masked_grad = grad_raw * mask_expanded * scale
                combined_grad += masked_grad

            # Average
            combined_grad = combined_grad / len(concept_grads_raw)
            return combined_grad

    def apply_guidance(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        spatial_masks: Dict[str, torch.Tensor],
        guidance_scales: Dict[str, float],
        harmful_scale: float = 1.0
    ) -> torch.Tensor:
        """Apply multi-concept guidance to latent."""
        # Compute raw gradients for each concept
        concept_grads_raw = self.compute_concept_gradients(
            latent=latent,
            timestep=timestep,
            spatial_masks=spatial_masks,
            guidance_scale=1.0,  # Will apply in combine
            harmful_scale=harmful_scale
        )

        # Combine using RG with concept-specific scales
        combined_grad = self.combine_gradients_with_rg(
            concept_grads_raw=concept_grads_raw,
            spatial_masks=spatial_masks,
            guidance_scales=guidance_scales
        )

        # Apply guidance
        guided_latent = latent + combined_grad
        return guided_latent


# =========================
# Arguments
# =========================
def parse_args():
    parser = ArgumentParser(description="Multi-Concept Always-On Adaptive Spatial CG")

    # Model & Generation
    parser.add_argument("ckpt_path", type=str, help="Path to pretrained SD model")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts")
    parser.add_argument("--output_dir", type=str, default="output_img/multi_concept_always_adaptive_cg",
                        help="Output directory")

    # Generation parameters
    parser.add_argument("--nsamples", type=int, default=1, help="Samples per prompt")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # Concept Selection (choose 2 or 3)
    parser.add_argument("--nudity_enabled", action="store_true", help="Enable nudity removal")
    parser.add_argument("--violence_enabled", action="store_true", help="Enable violence removal")
    parser.add_argument("--vangogh_enabled", action="store_true", help="Enable VanGogh style removal")

    # Classifiers (will be used based on enabled concepts)
    parser.add_argument("--nudity_classifier", type=str,
                        default="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth",
                        help="Nudity classifier checkpoint")
    parser.add_argument("--violence_classifier", type=str,
                        default="./work_dirs/violence_three_class_diff/checkpoint/step_11400/classifier.pth",
                        help="Violence classifier checkpoint")
    parser.add_argument("--vangogh_classifier", type=str,
                        default="./work_dirs/vangogh_three_class_diff/checkpoint/step_7200/classifier.pth",
                        help="VanGogh classifier checkpoint")

    parser.add_argument("--gradcam_layer", type=str,
                        default="encoder_model.middle_block.2",
                        help="Target layer for Grad-CAM")

    # === CONCEPT-SPECIFIC HYPERPARAMETERS ===
    # Each concept can have its own tuned values!

    # Nudity parameters
    parser.add_argument("--nudity_guidance_scale", type=float, default=10.0,
                        help="Nudity guidance scale")
    parser.add_argument("--nudity_spatial_start", type=float, default=0.4,
                        help="Nudity initial spatial threshold")
    parser.add_argument("--nudity_spatial_end", type=float, default=0.1,
                        help="Nudity final spatial threshold")

    # Violence parameters
    parser.add_argument("--violence_guidance_scale", type=float, default=10.0,
                        help="Violence guidance scale")
    parser.add_argument("--violence_spatial_start", type=float, default=0.4,
                        help="Violence initial spatial threshold")
    parser.add_argument("--violence_spatial_end", type=float, default=0.1,
                        help="Violence final spatial threshold")

    # VanGogh parameters
    parser.add_argument("--vangogh_guidance_scale", type=float, default=20.0,
                        help="VanGogh guidance scale")
    parser.add_argument("--vangogh_spatial_start", type=float, default=0.05,
                        help="VanGogh initial spatial threshold")
    parser.add_argument("--vangogh_spatial_end", type=float, default=0.1,
                        help="VanGogh final spatial threshold")

    # Shared parameters
    parser.add_argument("--threshold_strategy", type=str, default="cosine_anneal",
                        choices=["constant", "linear_decrease", "linear_increase", "cosine_anneal"],
                        help="Spatial threshold scheduling strategy (shared)")

    # Bidirectional guidance (optional)
    parser.add_argument("--use_bidirectional", action="store_true",
                        help="[Optional] Enable bidirectional guidance")
    parser.add_argument("--harmful_scale", type=float, default=1.5,
                        help="[Optional] Harmful repulsion scale if bidirectional (shared)")

    # 4. Active step range (optional)
    parser.add_argument("--guidance_start_step", type=int, default=0,
                        help="[Optional] Step to start guidance")
    parser.add_argument("--guidance_end_step", type=int, default=50,
                        help="[Optional] Step to end guidance")

    # Debug & Visualization
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Save visualizations")

    args = parser.parse_args()

    # Validate at least one concept is enabled
    if not (args.nudity_enabled or args.violence_enabled or args.vangogh_enabled):
        parser.error("At least one concept must be enabled (--nudity_enabled, --violence_enabled, or --vangogh_enabled)")

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


# =========================
# Main Generation Function
# =========================
def generate_with_multi_concept_guidance(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    mask_generator: MultiConceptSpatialMaskGenerator,
    guidance_module: MultiConceptGuidanceManager,
    threshold_schedulers: Dict[str, AdaptiveSpatialThresholdScheduler],
    guidance_scales: Dict[str, float],
    args,
    output_dir: Path,
    enabled_concepts: List[str]
):
    """Generate images with multi-concept always-on guidance."""
    device = pipe.device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("MULTI-CONCEPT ALWAYS-ON ADAPTIVE SPATIAL CG")
    print("="*80)
    print(f"Enabled concepts: {', '.join(enabled_concepts)}")
    print(f"Total prompts: {len(prompts)}")
    print(f"Samples per prompt: {args.nsamples}")
    print(f"Concept-specific guidance scales:")
    for concept, scale in guidance_scales.items():
        print(f"  - {concept}: {scale}")
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
            # Reset statistics
            mask_generator.reset_statistics()

            # Define callback
            def callback_on_step_end(pipe, step, timestep, callback_kwargs):
                latents = callback_kwargs["latents"]

                # Check if guidance should be applied at this step
                if args.guidance_start_step <= step <= args.guidance_end_step:
                    # Get per-concept adaptive spatial thresholds
                    spatial_thresholds = {
                        concept: scheduler.get_threshold(step)
                        for concept, scheduler in threshold_schedulers.items()
                    }

                    # Generate spatial masks for all concepts using per-concept thresholds
                    spatial_masks = {}
                    for concept_name in threshold_schedulers.keys():
                        # Generate mask for this concept with its specific threshold
                        concept_masks, _ = mask_generator.generate_masks(
                            latent=latents,
                            timestep=timestep,
                            spatial_threshold=spatial_thresholds[concept_name],
                            current_step=step,
                            return_heatmaps=False
                        )
                        # Extract just this concept's mask
                        spatial_masks[concept_name] = concept_masks[concept_name]

                    # Apply multi-concept guidance with concept-specific scales
                    guided_latents = guidance_module.apply_guidance(
                        latent=latents,
                        timestep=timestep,
                        spatial_masks=spatial_masks,
                        guidance_scales=guidance_scales,
                        harmful_scale=args.harmful_scale
                    )

                    callback_kwargs["latents"] = guided_latents

                    if args.debug:
                        mask_ratios = {name: mask.mean().item() for name, mask in spatial_masks.items()}
                        threshold_str = ", ".join([f"{name}={t:.3f}" for name, t in spatial_thresholds.items()])
                        print(f"  [Step {step}] thresholds=({threshold_str}), masks={mask_ratios}")

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

    # Determine enabled concepts
    enabled_concepts = []
    if args.nudity_enabled:
        enabled_concepts.append('nudity')
    if args.violence_enabled:
        enabled_concepts.append('violence')
    if args.vangogh_enabled:
        enabled_concepts.append('vangogh')

    print("\n" + "="*80)
    print("MULTI-CONCEPT ALWAYS-ON ADAPTIVE SPATIAL CG - INITIALIZATION")
    print("="*80)
    print(f"Model: {args.ckpt_path}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Enabled concepts: {', '.join(enabled_concepts)}")
    print("="*80 + "\n")

    # Load prompts
    print(f"[1/5] Loading prompts from {args.prompt_file}...")
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

    # Load classifiers for enabled concepts
    print(f"\n[3/5] Loading classifiers...")
    classifiers = {}
    safe_classes = {}
    harmful_classes = {}

    # Nudity: [0=NotPeople, 1=Clothed, 2=Nude]
    if args.nudity_enabled:
        print(f"  Loading nudity classifier...")
        classifiers['nudity'] = load_discriminator(
            ckpt_path=args.nudity_classifier,
            condition=None,
            eval=True,
            channel=4,
            num_classes=3
        ).to(device)
        safe_classes['nudity'] = 1
        harmful_classes['nudity'] = 2

    # Violence: [0=NotRelevant, 1=Safe, 2=Violence]
    if args.violence_enabled:
        print(f"  Loading violence classifier...")
        classifiers['violence'] = load_discriminator(
            ckpt_path=args.violence_classifier,
            condition=None,
            eval=True,
            channel=4,
            num_classes=3
        ).to(device)
        safe_classes['violence'] = 1
        harmful_classes['violence'] = 2

    # VanGogh: [0=NotPaint, 1=OtherPaint, 2=VanGoghStyle]
    if args.vangogh_enabled:
        print(f"  Loading VanGogh classifier...")
        classifiers['vangogh'] = load_discriminator(
            ckpt_path=args.vangogh_classifier,
            condition=None,
            eval=True,
            channel=4,
            num_classes=3
        ).to(device)
        safe_classes['vangogh'] = 1
        harmful_classes['vangogh'] = 2

    print(f"  {len(classifiers)} classifiers loaded")

    # Initialize per-concept threshold schedulers
    print(f"\n[4/5] Initializing adaptive spatial threshold schedulers...")
    threshold_schedulers = {}
    guidance_scales = {}

    if args.nudity_enabled:
        threshold_schedulers['nudity'] = AdaptiveSpatialThresholdScheduler(
            strategy=args.threshold_strategy,
            start_value=args.nudity_spatial_start,
            end_value=args.nudity_spatial_end,
            total_steps=args.num_inference_steps
        )
        guidance_scales['nudity'] = args.nudity_guidance_scale
        print(f"  Nudity: guidance={args.nudity_guidance_scale}, threshold={args.nudity_spatial_start} → {args.nudity_spatial_end}")

    if args.violence_enabled:
        threshold_schedulers['violence'] = AdaptiveSpatialThresholdScheduler(
            strategy=args.threshold_strategy,
            start_value=args.violence_spatial_start,
            end_value=args.violence_spatial_end,
            total_steps=args.num_inference_steps
        )
        guidance_scales['violence'] = args.violence_guidance_scale
        print(f"  Violence: guidance={args.violence_guidance_scale}, threshold={args.violence_spatial_start} → {args.violence_spatial_end}")

    if args.vangogh_enabled:
        threshold_schedulers['vangogh'] = AdaptiveSpatialThresholdScheduler(
            strategy=args.threshold_strategy,
            start_value=args.vangogh_spatial_start,
            end_value=args.vangogh_spatial_end,
            total_steps=args.num_inference_steps
        )
        guidance_scales['vangogh'] = args.vangogh_guidance_scale
        print(f"  VanGogh: guidance={args.vangogh_guidance_scale}, threshold={args.vangogh_spatial_start} → {args.vangogh_spatial_end}")

    print(f"  Strategy: {args.threshold_strategy}")

    # Initialize mask generator
    print(f"\n[5/5] Initializing multi-concept modules...")
    mask_generator = MultiConceptSpatialMaskGenerator(
        classifiers=classifiers,
        harmful_classes=harmful_classes,
        gradcam_layer=args.gradcam_layer,
        device=device,
        debug=args.debug
    )

    # Initialize guidance module
    guidance_module = MultiConceptGuidanceManager(
        classifiers=classifiers,
        safe_classes=safe_classes,
        harmful_classes=harmful_classes,
        device=device,
        use_bidirectional=args.use_bidirectional
    )
    guidance_mode = "bidirectional" if args.use_bidirectional else "unidirectional"
    print(f"  Multi-concept modules initialized ({guidance_mode})")

    # Generate images
    print(f"\n✓ All modules ready - Starting generation...")
    output_dir = Path(args.output_dir)

    generate_with_multi_concept_guidance(
        pipe=pipe,
        prompts=prompts,
        mask_generator=mask_generator,
        guidance_module=guidance_module,
        threshold_schedulers=threshold_schedulers,
        guidance_scales=guidance_scales,
        args=args,
        output_dir=output_dir,
        enabled_concepts=enabled_concepts
    )

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
