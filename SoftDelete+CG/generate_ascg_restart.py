#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ASCG + Guided Restart Sampling (GRS)

Combines Adaptive Spatial Classifier Guidance with Restart Sampling
for manifold recovery after strong safety guidance.

Motivation:
  - Strong safety guidance pushes latents off the data manifold
  - This causes FID degradation (e.g., 22.7 vs 8.96 baseline)
  - Restart sampling allows the model to "self-heal" quality artifacts
  - The safety correction from Phase 1 persists through the restart

Algorithm:
  Phase 1 (Guided Denoising):
    - Run standard ASCG with spatial Grad-CAM guidance for all 50 steps
    - Produces safe but potentially degraded latent z_0_guided

  Phase 2 (Selective Restart):
    - Re-noise z_0_guided back to intermediate timestep T_restart
    - Re-denoise from T_restart to 0 WITHOUT classifier guidance
    - Only standard CFG is applied (text conditioning preserved)
    - Model self-heals quality while content stays safe

  Phase 3 (Safety Verification) [optional]:
    - Run classifier on final latent
    - If unsafe, fall back to Phase 1 result (safety lock)

Inspired by:
  - Restart Sampling (Xu et al., NeurIPS 2023)
  - Self-Recurrence (Bansal et al., Universal Guidance, ICLR 2024)
  - Time-Travel (FreeDoM, ICCV 2023)

Usage:
  python generate_ascg_restart.py CompVis/stable-diffusion-v1-4 \
    --prompt_file datasets/nudity-ring-a-bell.csv \
    --classifier_ckpt work_dirs/nudity_4class_ringabell/classifier_final.pth \
    --gradcam_stats_file gradcam_stats/nudity_4class_ringabell/gradcam_stats_harm_nude_class2.json \
    --guidance_scale 10.0 \
    --spatial_threshold_start 0.3 \
    --spatial_threshold_end 0.5 \
    --restart_timestep 200 \
    --restart_guidance_fraction 0.0 \
    --output_dir scg_outputs/restart_poc
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

from diffusers import StableDiffusionPipeline, DDIMScheduler

from geo_models.classifier.classifier import load_discriminator
from geo_utils.classifier_interpretability import ClassifierGradCAM

# Reuse components from the existing ASCG pipeline
from generate_always_adaptive_spatial_cg import (
    AdaptiveSpatialThresholdScheduler,
    AdaptiveSpatialMaskGenerator,
    AlwaysOnSpatialGuidance,
    set_seed,
    save_image,
)


# =========================
# Restart Denoising Module
# =========================
class RestartDenoiser:
    """
    Performs restart denoising: re-noise a clean latent back to an
    intermediate timestep, then re-denoise using only text CFG
    (no classifier guidance).

    This allows the diffusion model to "self-heal" quality artifacts
    caused by strong classifier guidance, while maintaining the
    semantic content that was already steered toward safety.
    """

    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        classifier_model=None,
        harmful_class: int = 2,
        safe_class: int = 1,
        device: str = "cuda",
    ):
        self.pipe = pipe
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler
        self.vae = pipe.vae
        self.classifier = classifier_model
        self.harmful_class = harmful_class
        self.safe_class = safe_class
        self.device = device

    def add_noise_to_timestep(
        self,
        clean_latent: torch.Tensor,
        target_timestep: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Add noise to a clean latent to reach a specific noise level.

        Uses the forward diffusion process:
        z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            clean_latent: [B, 4, H, W] denoised latent (z_0)
            target_timestep: Timestep to noise up to
            generator: Optional random generator for reproducibility

        Returns:
            noisy_latent: [B, 4, H, W] noisy latent at target_timestep
        """
        noise = torch.randn(
            clean_latent.shape,
            device=clean_latent.device,
            dtype=clean_latent.dtype,
            generator=generator,
        )
        timesteps = torch.tensor(
            [target_timestep], device=clean_latent.device, dtype=torch.long
        )
        noisy_latent = self.scheduler.add_noise(clean_latent, noise, timesteps)
        return noisy_latent

    @torch.no_grad()
    def restart_denoise(
        self,
        latent_guided: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        restart_timestep: int = 200,
        cfg_scale: float = 7.5,
        num_inference_steps: int = 50,
        restart_guidance_fraction: float = 0.0,
        guidance_module: Optional[AlwaysOnSpatialGuidance] = None,
        mask_generator: Optional[AdaptiveSpatialMaskGenerator] = None,
        threshold_scheduler: Optional[AdaptiveSpatialThresholdScheduler] = None,
        guidance_scale: float = 5.0,
        harmful_scale: float = 1.0,
        base_guidance_scale: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Perform restart denoising.

        Args:
            latent_guided: [B, 4, H, W] guided latent from Phase 1
            prompt_embeds: [B, seq_len, dim] text embeddings
            negative_prompt_embeds: [B, seq_len, dim] uncond embeddings
            restart_timestep: Timestep to re-noise to (higher = more noise)
            cfg_scale: Classifier-free guidance scale
            num_inference_steps: Total inference steps for schedule
            restart_guidance_fraction: Fraction of restart steps to apply
                classifier guidance (0.0 = no guidance, 1.0 = full guidance)
            guidance_module: Optional ASCG guidance for partial restart guidance
            mask_generator: Optional mask generator for spatial guidance
            threshold_scheduler: Optional threshold scheduler
            guidance_scale: Classifier guidance scale (if applying partial guidance)
            harmful_scale: Harmful repulsion scale
            base_guidance_scale: Base guidance for non-masked regions
            generator: Random generator

        Returns:
            latent_restarted: [B, 4, H, W] quality-recovered latent
            info: Dictionary with restart statistics
        """
        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        all_timesteps = self.scheduler.timesteps

        # Find the index where restart begins
        restart_idx = None
        for i, t in enumerate(all_timesteps):
            if t.item() <= restart_timestep:
                restart_idx = i
                break

        if restart_idx is None:
            print(f"[WARNING] restart_timestep {restart_timestep} not found, using step 0")
            restart_idx = 0

        restart_timesteps = all_timesteps[restart_idx:]
        total_restart_steps = len(restart_timesteps)

        # Determine which restart steps get guidance (if any)
        guided_restart_steps = int(total_restart_steps * restart_guidance_fraction)

        print(f"  [Restart] Re-noising to t={restart_timestep} "
              f"(step {restart_idx}/{num_inference_steps})")
        print(f"  [Restart] Re-denoising {total_restart_steps} steps "
              f"({guided_restart_steps} guided, "
              f"{total_restart_steps - guided_restart_steps} unguided)")

        # Step 1: Re-noise the guided latent
        latent = self.add_noise_to_timestep(
            latent_guided, restart_timestep, generator=generator
        )

        # Step 2: Re-denoise from restart_timestep to 0
        # Concatenate embeddings for CFG
        do_cfg = cfg_scale > 1.0
        if do_cfg:
            prompt_embeds_combined = torch.cat(
                [negative_prompt_embeds, prompt_embeds], dim=0
            )

        info = {
            "restart_timestep": restart_timestep,
            "restart_idx": restart_idx,
            "total_restart_steps": total_restart_steps,
            "guided_restart_steps": guided_restart_steps,
            "step_details": [],
        }

        for step_i, t in enumerate(restart_timesteps):
            # Expand latent for CFG
            if do_cfg:
                latent_model_input = torch.cat([latent, latent], dim=0)
            else:
                latent_model_input = latent

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t
            )

            # Predict noise
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_combined if do_cfg else prompt_embeds,
            ).sample

            # Apply CFG
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # DDIM step
            latent = self.scheduler.step(noise_pred, t, latent).prev_sample

            # Optional: apply light classifier guidance for first few restart steps
            apply_guidance = (
                step_i < guided_restart_steps
                and guidance_module is not None
                and mask_generator is not None
            )

            step_info = {
                "step": restart_idx + step_i,
                "timestep": t.item(),
                "guided": apply_guidance,
            }

            if apply_guidance:
                # Use reduced guidance during restart
                restart_gs = guidance_scale * 0.3  # 30% of original
                global_step = restart_idx + step_i
                spatial_threshold = (
                    threshold_scheduler.get_threshold(global_step)
                    if threshold_scheduler
                    else 0.5
                )

                spatial_mask, _ = mask_generator.generate_mask(
                    latent=latent,
                    timestep=t,
                    spatial_threshold=spatial_threshold,
                    current_step=global_step,
                )

                latent = guidance_module.apply_guidance(
                    latent=latent,
                    timestep=t,
                    spatial_mask=spatial_mask,
                    guidance_scale=restart_gs,
                    harmful_scale=harmful_scale * 0.3,
                    base_guidance_scale=0.0,
                )
                step_info["guidance_scale_used"] = restart_gs
                step_info["mask_ratio"] = spatial_mask.mean().item()

            info["step_details"].append(step_info)

        return latent, info

    @torch.no_grad()
    def check_safety(
        self,
        latent: torch.Tensor,
        timestep: int = 0,
    ) -> Tuple[bool, Dict]:
        """
        Check if the latent is classified as safe.

        Args:
            latent: [B, 4, H, W] latent to check
            timestep: Timestep for classifier (0 for clean latent)

        Returns:
            is_safe: True if classified as safe
            probs: Class probabilities
        """
        if self.classifier is None:
            return True, {}

        classifier_dtype = next(self.classifier.parameters()).dtype
        latent_input = latent.to(dtype=classifier_dtype)
        norm_t = torch.tensor(
            [timestep / 1000.0], device=latent.device, dtype=classifier_dtype
        )

        with torch.no_grad():
            logits = self.classifier(latent_input, norm_t)
            probs = F.softmax(logits, dim=-1)

        harmful_prob = probs[0, self.harmful_class].item()
        safe_prob = probs[0, self.safe_class].item()
        pred_class = probs.argmax(dim=-1).item()

        is_safe = pred_class != self.harmful_class

        return is_safe, {
            "pred_class": pred_class,
            "harmful_prob": harmful_prob,
            "safe_prob": safe_prob,
            "all_probs": probs[0].cpu().tolist(),
        }


# =========================
# Main Generation Pipeline
# =========================
def generate_with_restart(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    mask_generator: AdaptiveSpatialMaskGenerator,
    guidance_module: AlwaysOnSpatialGuidance,
    threshold_scheduler: AdaptiveSpatialThresholdScheduler,
    restart_denoiser: RestartDenoiser,
    args,
    output_dir: Path,
):
    """
    Generate images with ASCG + Restart Sampling.
    """
    device = pipe.device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("ASCG + GUIDED RESTART SAMPLING (GRS)")
    print("=" * 80)
    print(f"Total prompts: {len(prompts)}")
    print(f"Phase 1 - ASCG guidance scale: {args.guidance_scale}")
    print(f"Phase 1 - Spatial threshold: {args.spatial_threshold_start} -> {args.spatial_threshold_end}")
    print(f"Phase 2 - Restart timestep: {args.restart_timestep}")
    print(f"Phase 2 - Restart guidance fraction: {args.restart_guidance_fraction}")
    print(f"Phase 2 - Restart count: {args.restart_count}")
    print(f"Phase 3 - Safety check: {args.safety_check}")
    print("=" * 80 + "\n")

    total_images = 0
    stats = {
        "total": 0,
        "restart_applied": 0,
        "safety_fallback": 0,
        "per_image": [],
    }

    # Pre-compute text embeddings for efficiency
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Generating")):
        for sample_idx in range(args.nsamples):
            seed = args.seed + prompt_idx * args.nsamples + sample_idx
            set_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)

            # Reset mask generator statistics
            mask_generator.reset_statistics()

            image_info = {
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "seed": seed,
                "prompt": prompt,
            }

            # ===== PHASE 1: Guided ASCG Denoising =====
            # Track per-step guidance strength
            guidance_norms = []

            def callback_phase1(pipe_obj, step, timestep, callback_kwargs):
                latents = callback_kwargs["latents"]

                if args.guidance_start_step <= step <= args.guidance_end_step:
                    spatial_threshold = threshold_scheduler.get_threshold(step)

                    spatial_mask, _ = mask_generator.generate_mask(
                        latent=latents,
                        timestep=timestep,
                        spatial_threshold=spatial_threshold,
                        current_step=step,
                    )

                    # Compute gradient (for norm tracking)
                    grad = guidance_module.compute_gradient(
                        latent=latents,
                        timestep=timestep,
                        spatial_mask=spatial_mask,
                        guidance_scale=args.guidance_scale,
                        harmful_scale=args.harmful_scale,
                        base_guidance_scale=args.base_guidance_scale,
                    )

                    grad_norm = grad.norm().item()
                    guidance_norms.append(
                        {"step": step, "norm": grad_norm, "timestep": timestep.item()}
                    )

                    # Apply guidance
                    guided_latents = latents + grad
                    callback_kwargs["latents"] = guided_latents

                return callback_kwargs

            # Run Phase 1
            with torch.no_grad():
                output_phase1 = pipe(
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.cfg_scale,
                    callback_on_step_end=callback_phase1,
                    callback_on_step_end_tensor_inputs=["latents"],
                    generator=generator,
                    output_type="latent",  # Get latent instead of image
                )

            latent_phase1 = output_phase1.images  # [B, 4, H, W] latent

            image_info["phase1_guidance_norms"] = guidance_norms
            avg_guidance_norm = (
                np.mean([g["norm"] for g in guidance_norms]) if guidance_norms else 0
            )
            image_info["avg_guidance_norm"] = avg_guidance_norm

            # ===== PHASE 2: Restart Denoising =====
            # Get text embeddings for manual denoising
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            prompt_embeds = text_encoder(text_inputs.input_ids)[0]

            uncond_inputs = tokenizer(
                "",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).to(device)
            negative_prompt_embeds = text_encoder(uncond_inputs.input_ids)[0]

            latent_current = latent_phase1
            restart_infos = []

            skip_restart = (args.restart_count <= 0 or args.restart_timestep <= 0)

            for restart_i in range(max(0, args.restart_count) if not skip_restart else 0):
                # Re-seed for restart reproducibility
                restart_gen = torch.Generator(device=device).manual_seed(
                    seed + 10000 * (restart_i + 1)
                )

                latent_restarted, restart_info = restart_denoiser.restart_denoise(
                    latent_guided=latent_current,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    restart_timestep=args.restart_timestep,
                    cfg_scale=args.cfg_scale,
                    num_inference_steps=args.num_inference_steps,
                    restart_guidance_fraction=args.restart_guidance_fraction,
                    guidance_module=guidance_module if args.restart_guidance_fraction > 0 else None,
                    mask_generator=mask_generator if args.restart_guidance_fraction > 0 else None,
                    threshold_scheduler=threshold_scheduler if args.restart_guidance_fraction > 0 else None,
                    guidance_scale=args.guidance_scale,
                    harmful_scale=args.harmful_scale,
                    base_guidance_scale=args.base_guidance_scale,
                    generator=restart_gen,
                )

                latent_current = latent_restarted
                restart_infos.append(restart_info)

            # ===== PHASE 3: Safety Verification =====
            use_restarted = not skip_restart
            if args.safety_check and use_restarted:
                is_safe, safety_info = restart_denoiser.check_safety(latent_current)
                image_info["safety_check"] = safety_info

                if not is_safe:
                    print(
                        f"  [Safety Lock] Restart result unsafe "
                        f"(harm_prob={safety_info['harmful_prob']:.3f}), "
                        f"falling back to Phase 1"
                    )
                    latent_current = latent_phase1
                    use_restarted = False
                    stats["safety_fallback"] += 1

            image_info["used_restart"] = use_restarted
            image_info["restart_infos"] = restart_infos

            # Decode latent to image
            with torch.no_grad():
                latent_decoded = latent_current / pipe.vae.config.scaling_factor
                image_tensor = pipe.vae.decode(
                    latent_decoded.to(pipe.vae.dtype)
                ).sample
                image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
                image_np = (
                    image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()[0] * 255
                ).astype(np.uint8)
                image = Image.fromarray(image_np)

            # Also decode Phase 1 result for comparison (if restart was used)
            if use_restarted and args.save_comparison:
                with torch.no_grad():
                    latent_p1_decoded = latent_phase1 / pipe.vae.config.scaling_factor
                    image_p1_tensor = pipe.vae.decode(
                        latent_p1_decoded.to(pipe.vae.dtype)
                    ).sample
                    image_p1_tensor = (image_p1_tensor / 2 + 0.5).clamp(0, 1)
                    image_p1_np = (
                        image_p1_tensor.cpu().permute(0, 2, 3, 1).float().numpy()[0]
                        * 255
                    ).astype(np.uint8)
                    image_phase1 = Image.fromarray(image_p1_np)

            # Save images
            safe_prompt = "".join(
                c if c.isalnum() or c in [" ", "-", "_"] else "_" for c in prompt
            )
            safe_prompt = safe_prompt[:50].strip().replace(" ", "_")

            filename = f"{prompt_idx:04d}_{sample_idx:02d}_{safe_prompt}.png"
            save_path = output_dir / filename
            save_image(image, save_path)

            if use_restarted and args.save_comparison:
                phase1_dir = output_dir / "phase1_only"
                phase1_dir.mkdir(exist_ok=True)
                save_image(image_phase1, phase1_dir / filename)

            total_images += 1
            stats["total"] += 1
            if use_restarted:
                stats["restart_applied"] += 1
            stats["per_image"].append(image_info)

    # Save generation statistics
    stats_path = output_dir / "generation_stats.json"

    # Make stats JSON serializable
    serializable_stats = {
        "args": vars(args),
        "summary": {
            "total": stats["total"],
            "restart_applied": stats["restart_applied"],
            "safety_fallback": stats["safety_fallback"],
        },
        "per_image": [
            {
                k: v
                for k, v in img.items()
                if k not in ["phase1_guidance_norms"]  # Skip large data
            }
            for img in stats["per_image"]
        ],
    }
    with open(stats_path, "w") as f:
        json.dump(serializable_stats, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total images: {stats['total']}")
    print(f"Restart applied: {stats['restart_applied']}")
    print(f"Safety fallbacks: {stats['safety_fallback']}")
    print(f"Output: {output_dir}")
    print("=" * 80)


# =========================
# Arguments
# =========================
def parse_args():
    parser = ArgumentParser(description="ASCG + Guided Restart Sampling")

    # Model & Generation
    parser.add_argument("ckpt_path", type=str, help="Pretrained SD model path")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument(
        "--output_dir", type=str, default="scg_outputs/restart_poc"
    )

    # Generation parameters
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)

    # Classifier
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--harmful_class", type=int, default=2)
    parser.add_argument("--safe_class", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument(
        "--gradcam_layer", type=str, default="encoder_model.middle_block.2"
    )
    parser.add_argument("--gradcam_stats_file", type=str, default=None)

    # Phase 1: ASCG parameters
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--spatial_threshold_start", type=float, default=0.3)
    parser.add_argument("--spatial_threshold_end", type=float, default=0.5)
    parser.add_argument(
        "--threshold_strategy",
        type=str,
        default="cosine_anneal",
        choices=["constant", "linear_decrease", "linear_increase", "cosine_anneal"],
    )
    parser.add_argument("--use_bidirectional", action="store_true")
    parser.add_argument("--harmful_scale", type=float, default=1.0)
    parser.add_argument("--base_guidance_scale", type=float, default=0.0)
    parser.add_argument("--guidance_start_step", type=int, default=0)
    parser.add_argument("--guidance_end_step", type=int, default=50)

    # Phase 2: Restart parameters
    parser.add_argument(
        "--restart_timestep",
        type=int,
        default=200,
        help="Timestep to re-noise back to (0-999). "
        "Higher = more noise = more quality recovery but more risk. "
        "Recommended: 100-300",
    )
    parser.add_argument(
        "--restart_guidance_fraction",
        type=float,
        default=0.0,
        help="Fraction of restart steps with classifier guidance "
        "(0.0 = pure restart, 1.0 = full guidance). Recommended: 0.0-0.3",
    )
    parser.add_argument(
        "--restart_count",
        type=int,
        default=1,
        help="Number of restart cycles. More = better quality but slower.",
    )

    # Phase 3: Safety parameters
    parser.add_argument(
        "--safety_check",
        action="store_true",
        help="Enable safety verification after restart (fallback to Phase 1 if unsafe)",
    )

    # Output options
    parser.add_argument(
        "--save_comparison",
        action="store_true",
        help="Save Phase 1 images alongside restart images for comparison",
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args


# =========================
# Load prompts
# =========================
def load_prompts(prompt_file: str) -> List[str]:
    """Load prompts from CSV or text file."""
    prompts = []
    if prompt_file.endswith(".csv"):
        import csv

        with open(prompt_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

            # Detect column: 'prompt', 'sensitive prompt', or first column
            prompt_col = 0
            for i, col in enumerate(header):
                if col.lower() in ["prompt", "sensitive prompt"]:
                    prompt_col = i
                    break

            for row in reader:
                if len(row) > prompt_col:
                    prompt_text = row[prompt_col].strip().strip('"')
                    if prompt_text:
                        prompts.append(prompt_text)
    else:
        with open(prompt_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]

    return prompts


# =========================
# Main
# =========================
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print("ASCG + GUIDED RESTART SAMPLING - INITIALIZATION")
    print("=" * 80)
    print(f"Model: {args.ckpt_path}")
    print(f"Device: {device}")
    print("=" * 80)

    # Load GradCAM statistics
    gradcam_stats = None
    if args.gradcam_stats_file:
        with open(args.gradcam_stats_file, "r") as f:
            gradcam_stats = json.load(f)
        print(f"[1/6] GradCAM stats loaded (mean={gradcam_stats['mean']:.4f}, "
              f"std={gradcam_stats['std']:.4f})")

    # Load prompts
    prompts = load_prompts(args.prompt_file)
    if args.end_idx > 0:
        prompts = prompts[args.start_idx : args.end_idx]
    elif args.start_idx > 0:
        prompts = prompts[args.start_idx :]
    print(f"[2/6] Loaded {len(prompts)} prompts")

    # Load pipeline
    print(f"[3/6] Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Load classifier
    print(f"[4/6] Loading classifier...")
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=args.num_classes,
    ).to(device)
    classifier.eval()

    # Initialize ASCG components
    print(f"[5/6] Initializing ASCG components...")
    threshold_scheduler = AdaptiveSpatialThresholdScheduler(
        strategy=args.threshold_strategy,
        start_value=args.spatial_threshold_start,
        end_value=args.spatial_threshold_end,
        total_steps=args.num_inference_steps,
    )

    mask_generator = AdaptiveSpatialMaskGenerator(
        classifier_model=classifier,
        harmful_class=args.harmful_class,
        gradcam_layer=args.gradcam_layer,
        device=str(device),
        debug=args.debug,
        gradcam_stats=gradcam_stats,
    )

    guidance_module = AlwaysOnSpatialGuidance(
        classifier_model=classifier,
        safe_class=args.safe_class,
        harmful_class=args.harmful_class,
        device=str(device),
        use_bidirectional=args.use_bidirectional,
    )

    # Initialize Restart Denoiser
    print(f"[6/6] Initializing Restart Denoiser...")
    restart_denoiser = RestartDenoiser(
        pipe=pipe,
        classifier_model=classifier,
        harmful_class=args.harmful_class,
        safe_class=args.safe_class,
        device=str(device),
    )

    print(f"\nAll modules ready. Starting generation...\n")

    # Generate
    generate_with_restart(
        pipe=pipe,
        prompts=prompts,
        mask_generator=mask_generator,
        guidance_module=guidance_module,
        threshold_scheduler=threshold_scheduler,
        restart_denoiser=restart_denoiser,
        args=args,
        output_dir=Path(args.output_dir),
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
