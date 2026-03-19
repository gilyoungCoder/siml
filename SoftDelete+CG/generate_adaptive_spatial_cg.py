#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adaptive Spatial Classifier Guidance with Skip Mode

Two modes:
  1. skip_safe=True (skip mode): Only apply guidance when harmful content detected
  2. skip_safe=False (always mode): Always apply guidance

Based on generate_always_adaptive_spatial_cg.py with added skip functionality.
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
# Spatial Mask Generator with Skip Detection
# =========================
class AdaptiveSpatialMaskGenerator:
    """Generates spatial masks using Grad-CAM with optional skip detection."""

    def __init__(
        self,
        classifier_model,
        harmful_class: int = 2,
        safe_class: int = 1,
        gradcam_layer: str = "encoder_model.middle_block.2",
        device: str = "cuda",
        debug: bool = False,
        gradcam_stats: Optional[Dict] = None,
        harmful_threshold: float = 0.5  # For skip detection
    ):
        self.classifier = classifier_model
        self.harmful_class = harmful_class
        self.safe_class = safe_class
        self.device = device
        self.debug = debug
        self.classifier_dtype = next(self.classifier.parameters()).dtype
        self.harmful_threshold = harmful_threshold

        self.gradcam_stats = gradcam_stats
        if gradcam_stats:
            print(f"✓ Using GradCAM statistics: Mean={gradcam_stats['mean']:.4f}, Std={gradcam_stats['std']:.4f}")

        self.gradcam = ClassifierGradCAM(
            classifier_model=classifier_model,
            target_layer_name=gradcam_layer
        )

        self.classifier = self.classifier.to(device)
        self.classifier.eval()
        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

        self.stats = {'total_steps': 0, 'skipped_steps': 0, 'guided_steps': 0, 'step_history': []}

    def is_harmful(self, latent: torch.Tensor, timestep: torch.Tensor) -> bool:
        """Check if current latent is classified as harmful using argmax."""
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)

        B = latent.shape[0]
        if timestep.shape[0] != B:
            timestep = timestep.expand(B)

        norm_timestep = timestep.float() / 1000.0
        latent_input = latent.to(dtype=self.classifier_dtype)

        with torch.no_grad():
            logits = self.classifier(latent_input, norm_timestep)
            predicted_class = logits.argmax(dim=-1)  # argmax로 예측 클래스 결정

        # harmful class로 예측되면 True (guidance 적용)
        return (predicted_class == self.harmful_class).any().item()

    def generate_mask(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        spatial_threshold: float,
        current_step: Optional[int] = None,
        return_heatmap: bool = False
    ) -> tuple:
        """Generate spatial mask for current latent."""
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)

        B = latent.shape[0]
        if timestep.shape[0] != B:
            timestep = timestep.expand(B)

        latent_input = latent.to(dtype=self.classifier_dtype)
        use_raw = self.gradcam_stats is not None
        norm_timestep = timestep.float() / 1000.0

        with torch.enable_grad():
            heatmap, info = self.gradcam.generate_heatmap(
                latent=latent_input,
                timestep=norm_timestep,
                target_class=self.harmful_class,
                normalize=not use_raw
            )

        if self.gradcam_stats:
            mean = self.gradcam_stats['mean']
            std = self.gradcam_stats['std']
            heatmap_standardized = (heatmap - mean) / (std + 1e-8)
            from torch.distributions import Normal
            normal = Normal(torch.tensor(0.0, device=heatmap.device),
                           torch.tensor(1.0, device=heatmap.device))
            heatmap = normal.cdf(heatmap_standardized)

        mask = (heatmap >= spatial_threshold).float()

        if current_step is not None:
            self.stats['total_steps'] += 1
            step_info = {
                'step': current_step,
                'spatial_threshold': spatial_threshold,
                'mask_ratio': mask.mean().item(),
                'heatmap_mean': heatmap.mean().item(),
                'heatmap_max': heatmap.max().item(),
            }
            self.stats['step_history'].append(step_info)

        if return_heatmap:
            return mask, heatmap
        else:
            return mask, None

    def reset_statistics(self):
        self.stats = {'total_steps': 0, 'skipped_steps': 0, 'guided_steps': 0, 'step_history': []}


# =========================
# Spatially Masked Guidance
# =========================
class SpatialGuidance:
    """Applies classifier guidance with spatial masking."""

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
        """Compute spatially-weighted classifier gradient."""
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

            norm_timestep = timestep.float() / 1000.0

            if self.use_bidirectional:
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
                logits = self.classifier(latent_input, norm_timestep)
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
        return latent + weighted_grad


# =========================
# Arguments
# =========================
def parse_args():
    parser = ArgumentParser(description="Adaptive Spatial CG with Skip Mode")

    # Model & Generation
    parser.add_argument("ckpt_path", type=str, help="Path to pretrained SD model")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts")
    parser.add_argument("--output_dir", type=str, default="output_img/adaptive_spatial_cg",
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
    parser.add_argument("--harmful_class", type=int, default=2, help="Harmful class index")
    parser.add_argument("--safe_class", type=int, default=1, help="Safe class index")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes")
    parser.add_argument("--gradcam_layer", type=str, default="encoder_model.middle_block.2",
                        help="Target layer for Grad-CAM")
    parser.add_argument("--gradcam_stats_dir", type=str, default=None,
                        help="Directory with GradCAM statistics JSON files")

    # === SKIP MODE ===
    parser.add_argument("--skip_safe", action="store_true",
                        help="[Skip Mode] Only apply guidance when harmful content detected")
    parser.add_argument("--harmful_threshold", type=float, default=0.5,
                        help="[Skip Mode] Threshold for harmful detection (default: 0.5)")

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
                        help="Enable bidirectional guidance")
    parser.add_argument("--harmful_scale", type=float, default=1.0,
                        help="Harmful repulsion scale")
    parser.add_argument("--base_guidance_scale", type=float, default=2.0,
                        help="Base guidance scale for non-harmful regions")

    # Active step range
    parser.add_argument("--guidance_start_step", type=int, default=0,
                        help="Step to start guidance")
    parser.add_argument("--guidance_end_step", type=int, default=50,
                        help="Step to end guidance")

    # Debug
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Save visualizations")

    args = parser.parse_args()
    return args


# =========================
# Utilities
# =========================
def load_prompts(prompt_file: str) -> List[str]:
    if prompt_file.endswith('.csv'):
        import csv
        with open(prompt_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header
            prompts = [row[0] for row in reader if row]
    else:
        with open(prompt_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    return prompts


def save_image(image, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.resize((512, 512))
    image.save(filepath)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_gradcam_stats(stats_dir: str, harmful_class: int) -> Optional[Dict]:
    """Load GradCAM statistics from directory."""
    if not stats_dir:
        return None

    stats_file = Path(stats_dir) / f"class_{harmful_class}_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            return json.load(f)

    # Try alternative naming
    for f in Path(stats_dir).glob("*.json"):
        with open(f, 'r') as fp:
            stats = json.load(fp)
            if 'mean' in stats and 'std' in stats:
                return stats

    return None


# =========================
# Main Generation Function
# =========================
def generate_with_guidance(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    mask_generator: AdaptiveSpatialMaskGenerator,
    guidance_module: SpatialGuidance,
    threshold_scheduler: AdaptiveSpatialThresholdScheduler,
    args,
    output_dir: Path
):
    """Generate images with adaptive spatial guidance."""
    device = pipe.device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "SKIP" if args.skip_safe else "ALWAYS"
    print("\n" + "="*80)
    print(f"ADAPTIVE SPATIAL CLASSIFIER GUIDANCE ({mode} MODE)")
    print("="*80)
    print(f"Total prompts: {len(prompts)}")
    print(f"Skip safe: {args.skip_safe}")
    if args.skip_safe:
        print(f"Harmful threshold: {args.harmful_threshold}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Spatial threshold: {args.spatial_threshold_start} → {args.spatial_threshold_end}")
    print(f"Strategy: {args.threshold_strategy}")
    print("="*80 + "\n")

    total_images = 0
    total_skipped = 0
    total_guided = 0

    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        print(f"\n[Prompt {prompt_idx+1}/{len(prompts)}] {prompt[:80]}...")

        for sample_idx in range(args.nsamples):
            mask_generator.reset_statistics()
            prompt_skipped = 0
            prompt_guided = 0

            def callback_on_step_end(pipe, step, timestep, callback_kwargs):
                nonlocal prompt_skipped, prompt_guided
                latents = callback_kwargs["latents"]

                if args.guidance_start_step <= step <= args.guidance_end_step:
                    # Skip mode: check if harmful first
                    if args.skip_safe:
                        is_harmful = mask_generator.is_harmful(latents, timestep)
                        if not is_harmful:
                            prompt_skipped += 1
                            if args.debug:
                                print(f"  [Step {step}] SKIPPED (not harmful)")
                            return callback_kwargs

                    prompt_guided += 1
                    spatial_threshold = threshold_scheduler.get_threshold(step)

                    spatial_mask, _ = mask_generator.generate_mask(
                        latent=latents,
                        timestep=timestep,
                        spatial_threshold=spatial_threshold,
                        current_step=step,
                        return_heatmap=False
                    )

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
                        print(f"  [Step {step}] GUIDED: threshold={spatial_threshold:.3f}, mask={mask_ratio:.1%}")

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
            print(f"  Saved: {filename} (guided={prompt_guided}, skipped={prompt_skipped})")

            total_images += 1
            total_skipped += prompt_skipped
            total_guided += prompt_guided

    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"Total images: {total_images}")
    print(f"Total guided steps: {total_guided}")
    print(f"Total skipped steps: {total_skipped}")
    if total_guided + total_skipped > 0:
        print(f"Skip ratio: {total_skipped / (total_guided + total_skipped):.1%}")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")


# =========================
# Main
# =========================
def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator()
    device = accelerator.device

    mode = "SKIP" if args.skip_safe else "ALWAYS"
    print("\n" + "="*80)
    print(f"ADAPTIVE SPATIAL CG ({mode} MODE) - INITIALIZATION")
    print("="*80)
    print(f"Model: {args.ckpt_path}")
    print(f"Device: {device}")
    print("="*80 + "\n")

    # Load GradCAM statistics
    gradcam_stats = load_gradcam_stats(args.gradcam_stats_dir, args.harmful_class)
    if gradcam_stats:
        print(f"✓ Loaded GradCAM stats: mean={gradcam_stats['mean']:.4f}, std={gradcam_stats['std']:.4f}")

    # Load prompts
    print(f"Loading prompts from {args.prompt_file}...")
    prompts = load_prompts(args.prompt_file)
    print(f"  Loaded {len(prompts)} prompts")

    # Load SD pipeline
    print(f"Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Load classifier
    print(f"Loading classifier from {args.classifier_ckpt}...")
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=args.num_classes
    ).to(device)
    classifier.eval()

    # Initialize modules
    threshold_scheduler = AdaptiveSpatialThresholdScheduler(
        strategy=args.threshold_strategy,
        start_value=args.spatial_threshold_start,
        end_value=args.spatial_threshold_end,
        total_steps=args.num_inference_steps
    )

    mask_generator = AdaptiveSpatialMaskGenerator(
        classifier_model=classifier,
        harmful_class=args.harmful_class,
        safe_class=args.safe_class,
        gradcam_layer=args.gradcam_layer,
        device=device,
        debug=args.debug,
        gradcam_stats=gradcam_stats,
        harmful_threshold=args.harmful_threshold
    )

    guidance_module = SpatialGuidance(
        classifier_model=classifier,
        safe_class=args.safe_class,
        harmful_class=args.harmful_class,
        device=device,
        use_bidirectional=args.use_bidirectional
    )

    print(f"\n✓ All modules ready - Starting generation...")

    generate_with_guidance(
        pipe=pipe,
        prompts=prompts,
        mask_generator=mask_generator,
        guidance_module=guidance_module,
        threshold_scheduler=threshold_scheduler,
        args=args,
        output_dir=Path(args.output_dir)
    )

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
