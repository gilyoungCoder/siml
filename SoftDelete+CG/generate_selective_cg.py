#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Selective Classifier Guidance for Machine Unlearning

Novel Approach:
  - Apply classifier guidance ONLY when harmful signal detected
  - Spatially-aware guidance using Grad-CAM localization
  - Prevents unnecessary intervention on benign prompts

Key Innovation vs Previous Methods:
  1. Previous (generate_classifier_masked.py):
     - Applies Grad-CAM masking at ALL timesteps
     - Can degrade benign prompt quality (GENEVAL score drop)

  2. This approach (Selective CG):
     - Monitors latent with classifier at each step
     - Applies guidance ONLY if harmful_score > threshold
     - Spatially masks guidance to harmful regions
     - Benign prompts: minimal/no intervention → preserves quality

Benefits:
  - Better GENEVAL scores on benign data
  - Selective intervention reduces computational cost
  - Maintains safety on harmful prompts
  - Spatial precision via Grad-CAM

Technical Flow:
  1. Each denoising step:
     a. Monitor latent with classifier
     b. Compute harmful score (nude class logit/probability)

  2. If harmful_score > threshold:
     a. Compute Grad-CAM heatmap for harmful class
     b. Create spatial mask of harmful regions
     c. Compute classifier gradient toward safe class
     d. Apply gradient masked to harmful regions only

  3. If harmful_score <= threshold:
     - Skip guidance entirely
     - Let vanilla diffusion proceed

Result:
  - Benign prompts: Clean, unaffected generation
  - Harmful prompts: Targeted suppression in harmful regions
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
from geo_utils.selective_guidance_utils import (
    SelectiveGuidanceMonitor,
    SpatiallyMaskedGuidance,
    WeightScheduler,
    visualize_selective_guidance
)


# =========================
# General Classifier Guidance (Always-on, non-selective)
# =========================
class GeneralClassifierGuidance:
    """
    General classifier guidance - applies to ALL steps regardless of detection.

    Difference from Selective CG:
    - Selective CG: Only applies when harmful detected + spatial masking
    - General CG: Always applies to entire latent (global guidance)

    Can be combined with Selective CG for maximum effect.
    """

    def __init__(
        self,
        classifier_model,
        safe_class: int = 1,
        harmful_class: int = 2,
        use_bidirectional: bool = False,
        device: str = "cuda"
    ):
        self.classifier = classifier_model
        self.safe_class = safe_class
        self.harmful_class = harmful_class
        self.use_bidirectional = use_bidirectional
        self.device = device
        self.classifier_dtype = next(self.classifier.parameters()).dtype

    def compute_gradient(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        guidance_scale: float = 5.0,
        harmful_scale: float = 1.0
    ) -> torch.Tensor:
        """Compute classifier gradient (bidirectional or unidirectional)."""
        with torch.enable_grad():
            latent_input = latent.detach().to(dtype=self.classifier_dtype).requires_grad_(True)

            # Ensure timestep is tensor
            if not isinstance(timestep, torch.Tensor):
                timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
            elif timestep.dim() == 0:
                timestep = timestep.unsqueeze(0)

            B = latent_input.shape[0]
            if timestep.shape[0] != B:
                timestep = timestep.expand(B)

            if self.use_bidirectional:
                # Bidirectional: pull to safe + push from harmful
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
                # Unidirectional: pull to safe only
                logits = self.classifier(latent_input, timestep)
                safe_logit = logits[:, self.safe_class].sum()
                grad = torch.autograd.grad(safe_logit, latent_input)[0]

        grad = grad * guidance_scale
        grad = grad.to(dtype=latent.dtype)
        return grad.detach()

    def apply_guidance(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        guidance_scale: float = 5.0,
        harmful_scale: float = 1.0
    ) -> torch.Tensor:
        """Apply general CG to latent."""
        grad = self.compute_gradient(latent, timestep, guidance_scale, harmful_scale)
        guided_latent = latent + grad
        return guided_latent


# =========================
# Arguments
# =========================
def parse_args():
    parser = ArgumentParser(description="Selective Classifier Guidance for Unlearning")

    # Model & Generation
    parser.add_argument("ckpt_path", type=str, help="Path to pretrained SD model")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts")
    parser.add_argument("--output_dir", type=str, default="output_img/selective_cg",
                        help="Output directory")

    # Generation parameters
    parser.add_argument("--nsamples", type=int, default=1, help="Samples per prompt")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # Selective Classifier Guidance
    parser.add_argument("--selective_guidance", action="store_true",
                        help="Enable selective classifier guidance")
    parser.add_argument("--classifier_ckpt", type=str,
                        default="./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth",
                        help="Classifier checkpoint path")

    # Detection parameters
    parser.add_argument("--harmful_threshold", type=float, default=0.5,
                        help="Threshold for harmful detection (logit or probability)")
    parser.add_argument("--harmful_class", type=int, default=2,
                        help="Harmful class index (2 = nude)")
    parser.add_argument("--safe_class", type=int, default=1,
                        help="Safe class for guidance target (1 = clothed)")

    # Spatial masking parameters
    parser.add_argument("--spatial_threshold", type=float, default=0.5,
                        help="Grad-CAM threshold for spatial masking")
    parser.add_argument("--use_percentile", action="store_true",
                        help="Use percentile instead of fixed threshold")
    parser.add_argument("--spatial_percentile", type=float, default=0.3,
                        help="Top percentile to mask (e.g., 0.3 = top 30%)")

    # Guidance parameters
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="Classifier guidance scale")
    parser.add_argument("--harmful_scale", type=float, default=1.0,
                        help="Harmful repulsion scale (relative to guidance_scale, default: 1.0)")
    parser.add_argument("--use_bidirectional", action="store_true",
                        help="Enable bidirectional guidance (pull to safe + push from harmful)")
    parser.add_argument("--guidance_start_step", type=int, default=0,
                        help="Step to start guidance")
    parser.add_argument("--guidance_end_step", type=int, default=50,
                        help="Step to end guidance")

    # Grad-CAM layer
    parser.add_argument("--gradcam_layer", type=str,
                        default="encoder_model.middle_block.2",
                        help="Target layer for Grad-CAM")

    # General Classifier Guidance (NEW: Always-on CG)
    parser.add_argument("--general_cg", action="store_true",
                        help="Enable general (always-on) classifier guidance")
    parser.add_argument("--general_cg_scale", type=float, default=5.0,
                        help="Gradient scale for general CG")
    parser.add_argument("--general_cg_harmful_scale", type=float, default=1.0,
                        help="Harmful repulsion scale for general CG (if bidirectional)")
    parser.add_argument("--general_cg_use_bidirectional", action="store_true",
                        help="Enable bidirectional guidance for general CG")
    parser.add_argument("--general_cg_start_step", type=int, default=0,
                        help="Step to start general CG")
    parser.add_argument("--general_cg_end_step", type=int, default=50,
                        help="Step to end general CG")

    # ========== Adaptive CG V2 Parameters ==========

    # Adaptive Threshold Scheduling
    parser.add_argument("--use_adaptive_threshold", action="store_true",
                        help="Enable adaptive threshold scheduling over timesteps")
    parser.add_argument("--threshold_strategy", type=str, default="constant",
                        choices=["constant", "linear_increase", "linear_decrease", "cosine_anneal"],
                        help="Threshold scheduling strategy")
    parser.add_argument("--threshold_start_value", type=float, default=0.5,
                        help="Starting spatial threshold value")
    parser.add_argument("--threshold_end_value", type=float, default=0.5,
                        help="Ending spatial threshold value")

    # Heatmap-Weighted Guidance
    parser.add_argument("--use_heatmap_weighted_guidance", action="store_true",
                        help="Use GradCAM heatmap values to weight guidance spatially")

    # Weight scheduling parameters (Guidance Scale Scheduling)
    parser.add_argument("--use_weight_scheduling", action="store_true",
                        help="Enable adaptive guidance scale scheduling")
    parser.add_argument("--weight_strategy", type=str, default="constant",
                        choices=["constant", "linear_increase", "linear_decrease",
                                 "cosine_anneal", "exponential_decay"],
                        help="Guidance scale scheduling strategy")
    parser.add_argument("--weight_start_value", type=float, default=1.0,
                        help="Starting guidance scale multiplier")
    parser.add_argument("--weight_end_value", type=float, default=1.0,
                        help="Ending guidance scale multiplier")
    parser.add_argument("--weight_decay_rate", type=float, default=0.1,
                        help="Decay rate for exponential strategy")

    # Gradient normalization
    parser.add_argument("--normalize_gradient", action="store_true",
                        help="Normalize gradients for stability")
    parser.add_argument("--gradient_norm_type", type=str, default="l2",
                        choices=["l2", "layer"],
                        help="Type of gradient normalization")

    # Debug options
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


# =========================
# Main Generation Function
# =========================
def generate_with_selective_guidance(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    monitor: Optional[SelectiveGuidanceMonitor],
    guidance_module: Optional[SpatiallyMaskedGuidance],
    general_cg: Optional[GeneralClassifierGuidance],
    args,
    output_dir: Path
):
    """
    Generate images with selective classifier guidance.

    Args:
        pipe: Stable Diffusion pipeline
        prompts: List of text prompts
        monitor: Selective guidance monitor
        guidance_module: Spatially masked guidance module
        args: Arguments
        output_dir: Output directory
    """
    device = pipe.device

    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("STARTING GENERATION WITH SELECTIVE CLASSIFIER GUIDANCE")
    print("="*80)
    print(f"Total prompts: {len(prompts)}")
    print(f"Samples per prompt: {args.nsamples}")
    print(f"Selective guidance: {args.selective_guidance}")
    if args.selective_guidance:
        print(f"  Harmful threshold: {args.harmful_threshold}")
        print(f"  Guidance scale: {args.guidance_scale}")
        print(f"  Spatial masking: {'Percentile' if args.use_percentile else 'Threshold'}")
        if args.use_percentile:
            print(f"  Spatial percentile: {args.spatial_percentile}")
        else:
            print(f"  Spatial threshold: {args.spatial_threshold}")
    print("="*80 + "\n")

    # Track overall statistics
    total_images = 0
    guidance_stats = []

    # Process each prompt
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        print(f"\n[Prompt {prompt_idx+1}/{len(prompts)}] {prompt}")

        for sample_idx in range(args.nsamples):
            # Reset monitor statistics for this generation
            if monitor is not None:
                monitor.reset_statistics()

            # Define callback for selective guidance + general CG
            def callback_on_step_end(pipe, step, timestep, callback_kwargs):
                latents = callback_kwargs["latents"]

                # Apply SELECTIVE guidance (only when harmful detected)
                if args.selective_guidance and monitor is not None and guidance_module is not None:
                    # Check step range
                    if args.guidance_start_step <= step <= args.guidance_end_step:
                        # Monitor latent and decide
                        should_guide, spatial_mask, info = monitor.should_apply_guidance(
                            latent=latents,
                            timestep=timestep,
                            step=step
                        )

                        if should_guide and spatial_mask is not None:
                            # Apply spatially-masked bidirectional guidance with weight scheduling
                            guided_latents = guidance_module.apply_guidance(
                                latent=latents,
                                timestep=timestep,
                                spatial_mask=spatial_mask,
                                guidance_scale=args.guidance_scale,
                                harmful_scale=args.harmful_scale,
                                scheduler=pipe.scheduler,
                                current_step=step  # NEW: Pass current step for weight scheduling
                            )
                            callback_kwargs["latents"] = guided_latents
                            latents = guided_latents  # Update for next step

                # Apply GENERAL CG (always-on, global guidance)
                if general_cg is not None and (args.general_cg_start_step <= step <= args.general_cg_end_step):
                    guided_latents = general_cg.apply_guidance(
                        latent=latents,
                        timestep=timestep,
                        guidance_scale=args.general_cg_scale,
                        harmful_scale=args.general_cg_harmful_scale
                    )
                    callback_kwargs["latents"] = guided_latents

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

            # Get statistics
            if monitor is not None:
                stats = monitor.get_statistics()
                guidance_stats.append(stats)

                if args.debug:
                    print(f"\n  Generation stats:")
                    print(f"    Total steps: {stats['total_steps']}")
                    print(f"    Harmful detected: {stats['harmful_steps']} ({stats.get('harmful_ratio', 0):.1%})")
                    print(f"    Guidance applied: {stats['guidance_applied']} ({stats.get('guidance_ratio', 0):.1%})")

                # Save visualization
                if args.save_visualizations and stats['guidance_applied'] > 0:
                    viz_dir = output_dir / "visualizations"
                    viz_dir.mkdir(exist_ok=True)
                    visualize_selective_guidance(
                        monitor=monitor,
                        output_dir=viz_dir,
                        prefix=f"{prompt_idx:04d}_{sample_idx:02d}"
                    )

    # Final summary
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"Total images generated: {total_images}")
    print(f"Output directory: {output_dir}")

    if guidance_stats:
        # Aggregate statistics
        total_steps = sum(s['total_steps'] for s in guidance_stats)
        total_harmful = sum(s['harmful_steps'] for s in guidance_stats)
        total_guided = sum(s['guidance_applied'] for s in guidance_stats)

        print(f"\nOverall Selective Guidance Statistics:")
        print(f"  Total denoising steps: {total_steps}")
        print(f"  Harmful detected: {total_harmful} ({total_harmful/max(1,total_steps):.1%})")
        print(f"  Guidance applied: {total_guided} ({total_guided/max(1,total_steps):.1%})")
        print(f"  Steps saved (no guidance): {total_steps - total_guided} ({(total_steps-total_guided)/max(1,total_steps):.1%})")

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
    print("SELECTIVE CLASSIFIER GUIDANCE - INITIALIZATION")
    print("="*80)
    print(f"Model: {args.ckpt_path}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print("="*80 + "\n")

    # Load prompts
    print(f"[1/4] Loading prompts from {args.prompt_file}...")
    prompts = load_prompts(args.prompt_file)
    print(f"  Loaded {len(prompts)} prompts")

    # Load Stable Diffusion pipeline
    print(f"\n[2/4] Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)

    # Use DDIM scheduler for deterministic generation
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    print(f"  Pipeline loaded successfully")

    # Initialize selective guidance modules
    monitor = None
    guidance_module = None
    general_cg = None

    if args.selective_guidance:
        print(f"\n[3/4] Initializing selective guidance modules...")

        # Load classifier
        print(f"  Loading classifier from {args.classifier_ckpt}...")
        classifier = load_discriminator(
            ckpt_path=args.classifier_ckpt,
            condition=None,
            eval=True,
            channel=4,
            num_classes=3  # [not_people, clothed, nude]
        ).to(device)
        # Keep classifier in float32 (default)
        # Latent will be converted to float32 when needed
        classifier.eval()
        print(f"  Classifier loaded (dtype: float32)")

        # Initialize THRESHOLD SCHEDULER (Adaptive CG V2)
        threshold_scheduler = None
        if args.use_adaptive_threshold:
            from geo_utils.selective_guidance_utils import ThresholdScheduler
            print(f"  Creating threshold scheduler...")
            threshold_scheduler = ThresholdScheduler(
                strategy=args.threshold_strategy,
                start_value=args.threshold_start_value,
                end_value=args.threshold_end_value,
                total_steps=args.num_inference_steps
            )
            print(f"    Strategy: {args.threshold_strategy}")
            print(f"    Threshold: {args.threshold_start_value} → {args.threshold_end_value}")

        # Initialize monitor (Adaptive CG V2)
        print(f"  Initializing selective guidance monitor...")
        monitor = SelectiveGuidanceMonitor(
            classifier_model=classifier,
            harmful_threshold=args.harmful_threshold,
            harmful_class=args.harmful_class,
            safe_class=args.safe_class,
            spatial_threshold=args.spatial_threshold,
            use_percentile=args.use_percentile,
            spatial_percentile=args.spatial_percentile,
            gradcam_layer=args.gradcam_layer,
            device=device,
            debug=args.debug,
            # Adaptive CG V2 parameters
            use_adaptive_threshold=args.use_adaptive_threshold,
            threshold_scheduler=threshold_scheduler,
            use_heatmap_weighted_guidance=args.use_heatmap_weighted_guidance
        )
        print(f"  Monitor initialized")
        if args.use_adaptive_threshold:
            print(f"    Adaptive threshold: {args.threshold_strategy} ({args.threshold_start_value} → {args.threshold_end_value})")
        if args.use_heatmap_weighted_guidance:
            print(f"    Heatmap-weighted guidance: ENABLED (pixel-wise adaptive)")
        else:
            print(f"    Binary masking")

        # Create WEIGHT SCHEDULER
        weight_scheduler = None
        if args.use_weight_scheduling:
            print(f"  Creating weight scheduler...")
            weight_scheduler = WeightScheduler(
                strategy=args.weight_strategy,
                start_step=args.guidance_start_step,
                end_step=args.guidance_end_step,
                start_weight=args.weight_start_value,
                end_weight=args.weight_end_value,
                decay_rate=args.weight_decay_rate
            )
            print(f"    Strategy: {args.weight_strategy}")
            print(f"    Weight range: {args.weight_start_value} → {args.weight_end_value}")

        # Initialize guidance module with WEIGHT SCHEDULER and NORMALIZATION
        print(f"  Initializing spatially-masked guidance...")
        guidance_module = SpatiallyMaskedGuidance(
            classifier_model=classifier,
            safe_class=args.safe_class,
            harmful_class=args.harmful_class,
            device=device,
            use_bidirectional=args.use_bidirectional,
            # NEW: Weight scheduler
            weight_scheduler=weight_scheduler,
            # NEW: Gradient normalization
            normalize_gradient=args.normalize_gradient,
            gradient_norm_type=args.gradient_norm_type
        )
        guidance_mode = "bidirectional" if args.use_bidirectional else "unidirectional"
        print(f"  Guidance module initialized ({guidance_mode})")
        if args.use_bidirectional:
            print(f"  Harmful repulsion scale: {args.harmful_scale}")
        print(f"  ✓ Selective guidance ready")
    else:
        print(f"\n[3/4] Selective guidance disabled - vanilla generation")

    # Initialize GENERAL CG (always-on, non-selective)
    if args.general_cg:
        print(f"\n[3.5/4] Initializing general classifier guidance...")

        # Load classifier if not already loaded
        if args.selective_guidance:
            # Reuse classifier from selective guidance
            classifier = monitor.classifier
            print(f"  Reusing classifier from selective guidance")
        else:
            # Load classifier separately
            print(f"  Loading classifier from {args.classifier_ckpt}...")
            classifier = load_discriminator(
                ckpt_path=args.classifier_ckpt,
                condition=None,
                eval=True,
                channel=4,
                num_classes=3
            ).to(device)
            classifier.eval()
            print(f"  Classifier loaded")

        # Initialize general CG
        general_cg = GeneralClassifierGuidance(
            classifier_model=classifier,
            safe_class=args.safe_class,
            harmful_class=args.harmful_class,
            use_bidirectional=args.general_cg_use_bidirectional,
            device=device
        )
        guidance_mode = "bidirectional" if args.general_cg_use_bidirectional else "unidirectional"
        print(f"  General CG initialized ({guidance_mode})")
        print(f"  Scale: {args.general_cg_scale}")
        if args.general_cg_use_bidirectional:
            print(f"  Harmful repulsion scale: {args.general_cg_harmful_scale}")
        print(f"  Active steps: {args.general_cg_start_step} → {args.general_cg_end_step}")
        print(f"  ✓ General CG ready")

    # Generate images
    print(f"\n[4/4] Starting image generation...")
    output_dir = Path(args.output_dir)

    generate_with_selective_guidance(
        pipe=pipe,
        prompts=prompts,
        monitor=monitor,
        guidance_module=guidance_module,
        general_cg=general_cg,
        args=args,
        output_dir=output_dir
    )

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
