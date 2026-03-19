#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Noise-Space Anchor Projection v5: Training-Free Safe Generation

Architecture:
  WHEN (Global CAS): cos(d_prompt, d_target) > tau  -> trigger guidance (sticky mode)
  WHERE (Anchor Projection): project d_prompt onto (d_target - d_anchor) axis per pixel
                              -> sigmoid soft mask with Gaussian blur smoothing
  HOW (Guided denoising):
       - anchor_inpaint: eps_cfg * (1 - s*M) + eps_anchor_cfg * (s*M)
       - sld:            eps_cfg - s * M * (eps_target - eps_null)
       - hybrid:         eps_cfg - s * M * (eps_target - eps_anchor)
       - proj_subtract:  eps_cfg - s * M * proj_component (remove nudity-axis component)

Key differences from v4:
  - Anchor Projection instead of Spatial CAS for WHERE detection
  - Projects d_prompt onto the "nudity axis" (d_target - d_anchor) at each pixel
  - Positive projection = nudity-leaning, Negative = clothed-leaning
  - New proj_subtract guidance mode: removes only the nudity-axis component
  - 4 UNet forwards every guided step (anchor always needed for projection)
"""

import os
import sys
import json
import math
import random
import csv
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from diffusers import StableDiffusionPipeline, DDIMScheduler


# =============================================================================
# Global CAS: Concept Alignment Score (WHEN)
# =============================================================================
class GlobalCAS:
    """Detect harmful prompt alignment via cosine(d_prompt, d_target)."""

    def __init__(self, threshold: float = 0.3, sticky: bool = True):
        self.threshold = threshold
        self.sticky = sticky
        self.triggered = False

    def reset(self):
        self.triggered = False

    def compute(self, eps_prompt, eps_null, eps_target):
        """
        Compute global CAS from noise predictions.

        Args:
            eps_prompt: noise prediction conditioned on prompt [1, 4, 64, 64]
            eps_null:   unconditional noise prediction [1, 4, 64, 64]
            eps_target: noise prediction conditioned on target concept [1, 4, 64, 64]

        Returns:
            cas_value: float, cosine similarity between prompt and target directions
            should_guide: bool, whether guidance should be applied this step
        """
        d_prompt = (eps_prompt - eps_null).reshape(1, -1).float()
        d_target = (eps_target - eps_null).reshape(1, -1).float()
        cas = F.cosine_similarity(d_prompt, d_target, dim=-1).item()

        if math.isnan(cas) or math.isinf(cas):
            return 0.0, self.triggered if self.sticky else False

        # Sticky mode: once triggered, stays on
        if self.sticky and self.triggered:
            return cas, True

        if cas > self.threshold:
            if self.sticky:
                self.triggered = True
            return cas, True

        return cas, False


# =============================================================================
# Anchor Projection: Per-Pixel Projection onto Nudity Axis (WHERE)
# =============================================================================
def compute_anchor_projection(
    eps_prompt: torch.Tensor,   # [1, 4, 64, 64]
    eps_null: torch.Tensor,
    eps_target: torch.Tensor,
    eps_anchor: torch.Tensor,
    neighborhood_size: int = 3,
) -> torch.Tensor:
    """
    Compute per-pixel projection of d_prompt onto the nudity axis (d_target - d_anchor).

    The "nudity axis" at each pixel is the direction from clothed to nude.
    Projecting d_prompt onto this axis measures "how much more like nudity than
    clothed" the generation is at each spatial location.

    Uses neighborhood pooling (unfold) for robustness, same as V4's spatial CAS.

    Args:
        eps_prompt: prompt noise prediction [1, 4, H, W]
        eps_null:   unconditional noise prediction [1, 4, H, W]
        eps_target: target concept noise prediction [1, 4, H, W]
        eps_anchor: anchor concept noise prediction [1, 4, H, W]
        neighborhood_size: kernel size for neighborhood pooling (default: 3)

    Returns:
        projection: per-pixel projection values [H, W] (positive = nudity-leaning)
    """
    d_prompt = (eps_prompt - eps_null).float()  # [1, 4, H, W]
    d_target = (eps_target - eps_null).float()
    d_anchor = (eps_anchor - eps_null).float()

    # The nudity axis: direction from clothed to nude at each pixel
    axis = d_target - d_anchor  # [1, 4, H, W]

    H, W = d_prompt.shape[2], d_prompt.shape[3]
    pad = neighborhood_size // 2

    # Unfold: extract neighborhoods -> [1, C * k * k, H * W]
    prompt_unf = F.unfold(d_prompt, kernel_size=neighborhood_size, padding=pad)
    axis_unf = F.unfold(axis, kernel_size=neighborhood_size, padding=pad)

    # Project d_prompt onto the nudity axis at each pixel
    # projection = dot(d_prompt, axis) / ||axis||
    dot_product = (prompt_unf * axis_unf).sum(dim=1)  # [1, H*W]
    axis_norm = axis_unf.norm(dim=1).clamp(min=1e-8)  # [1, H*W]
    projection = dot_product / axis_norm  # [1, H*W]
    projection = projection.reshape(H, W)

    return projection


def compute_soft_mask(
    projection: torch.Tensor,    # [H, W]
    proj_threshold: float = 0.0,
    sigmoid_alpha: float = 10.0,
    blur_sigma: float = 1.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Convert projection map to a soft guidance mask using sigmoid + Gaussian blur.

    Positive projection (nudity-leaning) -> 1 (needs guidance)
    Negative projection (clothed-leaning) -> 0 (keep original)

    Args:
        projection: per-pixel projection values [H, W]
        proj_threshold: center of sigmoid transition (default 0.0)
        sigmoid_alpha: sharpness of sigmoid (higher = sharper transition)
        blur_sigma: sigma for Gaussian blur smoothing (0 = no blur)

    Returns:
        soft_mask: [1, 1, H, W] smooth mask in [0, 1]
    """
    # Sigmoid: maps projection values around threshold to [0, 1]
    soft_mask = torch.sigmoid(sigmoid_alpha * (projection - proj_threshold))
    soft_mask = soft_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # Gaussian blur for smoother boundaries
    if blur_sigma > 0:
        soft_mask = gaussian_blur_2d(soft_mask, kernel_size=5, sigma=blur_sigma)

    return soft_mask.clamp(0, 1)


def gaussian_blur_2d(
    x: torch.Tensor,         # [B, C, H, W]
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Apply Gaussian blur to a 2D tensor."""
    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()

    # Separable 2D convolution
    kernel_h = g.view(1, 1, kernel_size, 1)
    kernel_w = g.view(1, 1, 1, kernel_size)
    pad_h = kernel_size // 2
    pad_w = kernel_size // 2

    # Apply along height then width
    x = F.pad(x, [0, 0, pad_h, pad_h], mode='reflect')
    x = F.conv2d(x, kernel_h.expand(x.shape[1], -1, -1, -1), groups=x.shape[1])
    x = F.pad(x, [pad_w, pad_w, 0, 0], mode='reflect')
    x = F.conv2d(x, kernel_w.expand(x.shape[1], -1, -1, -1), groups=x.shape[1])

    return x


# =============================================================================
# Guidance Application (HOW)
# =============================================================================
def apply_guidance(
    eps_cfg: torch.Tensor,        # Standard CFG: eps_null + cfg_scale * (eps_prompt - eps_null)
    eps_null: torch.Tensor,
    eps_prompt: torch.Tensor,
    eps_target: torch.Tensor,
    eps_anchor: torch.Tensor,
    soft_mask: torch.Tensor,      # [1, 1, H, W] soft mask in [0, 1]
    guide_mode: str = "anchor_inpaint",
    safety_scale: float = 1.0,
    cfg_scale: float = 7.5,
) -> torch.Tensor:
    """
    Apply spatially-masked guidance to remove unsafe content.

    Modes:
        anchor_inpaint: Blend original CFG with anchor CFG in masked regions.
            eps_final = eps_cfg * (1 - s*M) + eps_anchor_cfg * (s*M)

        sld: Subtract target concept direction in masked regions (SLD-style).
            eps_final = eps_cfg - s * M * (eps_target - eps_null)

        hybrid: Subtract target AND add anchor direction.
            eps_final = eps_cfg - s * M * (eps_target - eps_anchor)

        proj_subtract: Remove only the nudity-axis component from noise at each pixel.
            axis_norm = (d_target - d_anchor) / ||d_target - d_anchor||
            proj_component = dot(d_prompt, axis_norm) * axis_norm
            eps_final = eps_cfg - s * M * proj_component

    Args:
        eps_cfg:      standard CFG noise prediction
        eps_null:     unconditional noise prediction
        eps_prompt:   prompt-conditioned noise prediction
        eps_target:   target concept noise prediction
        eps_anchor:   anchor concept noise prediction
        soft_mask:    [1, 1, H, W] guidance mask
        guide_mode:   "anchor_inpaint", "sld", "hybrid", or "proj_subtract"
        safety_scale: guidance strength
        cfg_scale:    classifier-free guidance scale

    Returns:
        eps_final: guided noise prediction
    """
    mask = soft_mask.to(eps_cfg.dtype)

    if guide_mode == "anchor_inpaint":
        # Anchor CFG: what the anchor concept would generate
        eps_anchor_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
        # Soft inpainting: blend original and anchor based on mask
        eps_final = eps_cfg * (1.0 - safety_scale * mask) + eps_anchor_cfg * (safety_scale * mask)

    elif guide_mode == "sld":
        # SLD-style: subtract target direction in masked regions
        eps_safe_direction = eps_target - eps_null
        eps_final = eps_cfg - safety_scale * mask * eps_safe_direction

    elif guide_mode == "hybrid":
        # Hybrid: subtract target AND add anchor (push from target toward anchor)
        eps_final = eps_cfg - safety_scale * mask * (eps_target - eps_anchor)

    elif guide_mode == "proj_subtract":
        # Remove only the nudity-axis component from noise at each pixel
        d_prompt = (eps_prompt - eps_null).float()
        axis = (eps_target - eps_anchor).float()  # [1, 4, H, W]
        # Normalize axis per pixel (across channel dim)
        axis_norm_val = axis.norm(dim=1, keepdim=True).clamp(min=1e-8)  # [1, 1, H, W]
        axis_normalized = axis / axis_norm_val  # [1, 4, H, W]
        # Project d_prompt onto normalized axis: scalar projection per pixel
        proj_scalar = (d_prompt * axis_normalized).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        # Vector projection: the nudity-axis component of d_prompt
        proj_component = proj_scalar * axis_normalized  # [1, 4, H, W]
        eps_final = eps_cfg - safety_scale * mask * proj_component.to(eps_cfg.dtype)

    else:
        raise ValueError(f"Unknown guide_mode: {guide_mode}")

    # NaN/Inf guard
    if torch.isnan(eps_final).any() or torch.isinf(eps_final).any():
        eps_final = torch.where(torch.isfinite(eps_final), eps_final, eps_cfg)

    return eps_final


# =============================================================================
# Utils
# =============================================================================
def load_prompts(filepath):
    """Load prompts from CSV (col: 'sensitive prompt' etc.) or TXT (one per line)."""
    filepath = Path(filepath)
    if filepath.suffix == ".csv":
        prompts = []
        with open(filepath, "r") as fp:
            reader = csv.DictReader(fp)
            fieldnames = reader.fieldnames
            prompt_col = None
            for col in ['sensitive prompt', 'adv_prompt', 'prompt', 'target_prompt',
                        'text', 'Prompt', 'Text']:
                if col in fieldnames:
                    prompt_col = col
                    break
            if prompt_col is None:
                raise ValueError(f"No known prompt column in {fieldnames}")
            for row in reader:
                p = row[prompt_col].strip()
                if p:
                    prompts.append(p)
        return prompts
    else:
        return [line.strip() for line in open(filepath) if line.strip()]


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_safe_filename(prompt, max_len=50):
    """Create a filesystem-safe slug from a prompt string."""
    safe = "".join(c if c.isalnum() or c in ' -_' else '_' for c in prompt)
    return safe[:max_len].replace(' ', '_')


def encode_concepts(text_encoder, tokenizer, concepts, device):
    """
    Encode a list of concept strings and average their embeddings.

    Args:
        text_encoder: CLIP text encoder
        tokenizer: CLIP tokenizer
        concepts: list of concept strings (e.g., ["nudity", "nude person"])
        device: torch device

    Returns:
        averaged embedding: [1, seq_len, dim]
    """
    all_embeds = []
    for concept in concepts:
        inputs = tokenizer(
            concept,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        embeds = text_encoder(inputs.input_ids.to(device))[0]
        all_embeds.append(embeds)

    # Average embeddings across concepts
    return torch.stack(all_embeds).mean(dim=0)


def parse_args():
    p = ArgumentParser(description="Noise-Space Anchor Projection v5")
    p.add_argument("--ckpt", type=str, default="CompVis/stable-diffusion-v1-4",
                    help="Model checkpoint")
    p.add_argument("--prompts", type=str, required=True,
                    help="Path to CSV or TXT prompt file")
    p.add_argument("--outdir", type=str, required=True,
                    help="Output directory")
    p.add_argument("--nsamples", type=int, default=4,
                    help="Images per prompt")
    p.add_argument("--steps", type=int, default=50,
                    help="Denoising steps")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    p.add_argument("--cfg_scale", type=float, default=7.5,
                    help="Classifier-free guidance scale")

    # CAS (When)
    p.add_argument("--cas_threshold", type=float, default=0.3,
                    help="Global CAS trigger threshold")
    p.add_argument("--cas_sticky", action="store_true", default=True,
                    help="Once triggered, guidance stays on")
    p.add_argument("--cas_no_sticky", action="store_true",
                    help="Disable sticky CAS mode")

    # Anchor Projection (Where)
    p.add_argument("--proj_threshold", type=float, default=0.0,
                    help="Projection threshold (sigmoid center); positive = nudity-leaning")
    p.add_argument("--spatial_threshold", type=float, default=0.3,
                    help="Alias for proj_threshold (backward compat, overridden by proj_threshold if both set)")
    p.add_argument("--sigmoid_alpha", type=float, default=10.0,
                    help="Sharpness of sigmoid mask transition")
    p.add_argument("--neighborhood_size", type=int, default=3,
                    help="Pooling kernel size for robust per-pixel projection")
    p.add_argument("--blur_sigma", type=float, default=1.0,
                    help="Gaussian blur sigma for mask smoothing (0=no blur)")

    # Guidance (How)
    p.add_argument("--guide_mode", type=str, default="anchor_inpaint",
                    choices=["anchor_inpaint", "sld", "hybrid", "proj_subtract"],
                    help="Guidance mode")
    p.add_argument("--safety_scale", type=float, default=1.0,
                    help="Guidance strength (anchor_inpaint: 0-1, sld: 1-10)")
    p.add_argument("--guide_start_frac", type=float, default=0.0,
                    help="Fraction of total steps before guidance starts (0=all steps)")

    # Concepts
    p.add_argument("--target_concepts", type=str, nargs="+",
                    default=["nudity", "nude person", "naked body"],
                    help="Target concepts to erase")
    p.add_argument("--anchor_concepts", type=str, nargs="+",
                    default=["clothed person", "person wearing clothes"],
                    help="Anchor concepts to guide toward")

    # Misc
    p.add_argument("--save_maps", action="store_true",
                    help="Save projection maps and masks for debugging")
    p.add_argument("--debug", action="store_true",
                    help="Print per-step debug info")
    p.add_argument("--start_idx", type=int, default=0,
                    help="Start prompt index (for parallel runs)")
    p.add_argument("--end_idx", type=int, default=-1,
                    help="End prompt index (-1 = all)")

    args = p.parse_args()
    if args.cas_no_sticky:
        args.cas_sticky = False
    return args


# =============================================================================
# Main Generation Loop
# =============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"Noise-Space Anchor Projection v5")
    print(f"{'='*70}")
    print(f"  WHEN:  Global CAS threshold={args.cas_threshold}, sticky={args.cas_sticky}")
    print(f"  WHERE: Anchor Projection, neighborhood={args.neighborhood_size}, "
          f"proj_threshold={args.proj_threshold}, alpha={args.sigmoid_alpha}, blur={args.blur_sigma}")
    print(f"  HOW:   {args.guide_mode}, safety_scale={args.safety_scale}, "
          f"start_frac={args.guide_start_frac}")
    print(f"  Target concepts: {args.target_concepts}")
    print(f"  Anchor concepts: {args.anchor_concepts}")
    print(f"  Model: {args.ckpt}")
    print(f"  Steps: {args.steps}, CFG: {args.cfg_scale}, Samples/prompt: {args.nsamples}")
    print(f"{'='*70}\n")

    # ---- Load prompts ----
    all_prompts = load_prompts(args.prompts)
    end_idx = args.end_idx if args.end_idx > 0 else len(all_prompts)
    prompts_with_idx = list(enumerate(all_prompts))[args.start_idx:end_idx]
    print(f"Loaded {len(all_prompts)} prompts, processing [{args.start_idx}:{end_idx}]")

    # ---- Load pipeline (safety_checker=None) ----
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.feature_extractor = None  # Fully disable safety filtering

    unet = pipe.unet
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    scheduler = pipe.scheduler

    # ---- Pre-encode concept embeddings ----
    with torch.no_grad():
        # Target: average embeddings of all target concepts
        target_embeds = encode_concepts(text_encoder, tokenizer,
                                        args.target_concepts, device)
        # Anchor: average embeddings of all anchor concepts
        anchor_embeds = encode_concepts(text_encoder, tokenizer,
                                        args.anchor_concepts, device)
        # Unconditional (empty string)
        uncond_inputs = tokenizer(
            "", padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]

    print(f"Embeddings encoded: target={len(args.target_concepts)} concepts (averaged), "
          f"anchor={len(args.anchor_concepts)} concepts (averaged)")

    # ---- Init CAS ----
    cas = GlobalCAS(threshold=args.cas_threshold, sticky=args.cas_sticky)

    # ---- Output directory ----
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.save_maps:
        (outdir / "maps").mkdir(exist_ok=True)

    all_stats = []

    # ---- Generation loop ----
    for prompt_idx, prompt in tqdm(prompts_with_idx, desc="Generating"):
        if not prompt.strip():
            continue

        for sample_idx in range(args.nsamples):
            seed = args.seed + prompt_idx * args.nsamples + sample_idx
            set_seed(seed)
            cas.reset()

            guided_count = 0
            cas_values = []
            mask_areas = []
            proj_values = []

            # Encode the prompt
            with torch.no_grad():
                prompt_inputs = tokenizer(
                    prompt, padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt"
                )
                prompt_embeds = text_encoder(prompt_inputs.input_ids.to(device))[0]

            # Init latents
            set_seed(seed)
            latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
            latents = latents * scheduler.init_noise_sigma
            scheduler.set_timesteps(args.steps, device=device)
            total_steps = len(scheduler.timesteps)

            guide_start_step = int(total_steps * args.guide_start_frac)

            # ---- Denoising loop ----
            for step_idx, t in enumerate(scheduler.timesteps):

                # =========================================================
                # Step 1: UNet forward passes (4 passes)
                # =========================================================
                lat_in = scheduler.scale_model_input(latents, t)

                with torch.no_grad():
                    # Pass 1 & 2: unconditional + prompt (batched)
                    lat_batch = torch.cat([lat_in, lat_in])
                    embed_batch = torch.cat([uncond_embeds, prompt_embeds])
                    raw = unet(lat_batch, t, encoder_hidden_states=embed_batch).sample
                    eps_null, eps_prompt = raw.chunk(2)

                    # Pass 3: target concept
                    eps_target = unet(lat_in, t,
                                      encoder_hidden_states=target_embeds).sample

                # Standard CFG
                eps_cfg = eps_null + args.cfg_scale * (eps_prompt - eps_null)

                # =========================================================
                # Step 2: Global CAS (WHEN to guide)
                # =========================================================
                cas_val, should_trigger = cas.compute(eps_prompt, eps_null, eps_target)
                cas_values.append(cas_val)

                in_window = step_idx >= guide_start_step
                should_guide = should_trigger and in_window

                if should_guide:
                    # =====================================================
                    # Pass 4: anchor concept (needed for projection)
                    # =====================================================
                    with torch.no_grad():
                        eps_anchor = unet(lat_in, t,
                                          encoder_hidden_states=anchor_embeds).sample

                    # =====================================================
                    # Step 3: Anchor Projection (WHERE to guide)
                    # =====================================================
                    projection = compute_anchor_projection(
                        eps_prompt, eps_null, eps_target, eps_anchor,
                        neighborhood_size=args.neighborhood_size,
                    )

                    avg_proj = float(projection.mean().item())
                    proj_values.append(avg_proj)

                    # =====================================================
                    # Step 4: Soft mask with sigmoid + Gaussian blur
                    # =====================================================
                    soft_mask = compute_soft_mask(
                        projection,
                        proj_threshold=args.proj_threshold,
                        sigmoid_alpha=args.sigmoid_alpha,
                        blur_sigma=args.blur_sigma,
                        device=device,
                    )

                    # =====================================================
                    # Step 5: Apply guidance
                    # =====================================================
                    eps_final = apply_guidance(
                        eps_cfg=eps_cfg,
                        eps_null=eps_null,
                        eps_prompt=eps_prompt,
                        eps_target=eps_target,
                        eps_anchor=eps_anchor,
                        soft_mask=soft_mask,
                        guide_mode=args.guide_mode,
                        safety_scale=args.safety_scale,
                        cfg_scale=args.cfg_scale,
                    )

                    guided_count += 1
                    area_val = float(soft_mask.mean().item())
                    mask_areas.append(area_val)

                    # Save projection maps for debugging
                    if args.save_maps and step_idx % 10 == 0:
                        # Save projection map (centered around 0)
                        proj_np = projection.float().cpu().numpy()
                        proj_np = np.nan_to_num(proj_np, nan=0.0)
                        # Map to [0, 255]: negative=dark, zero=mid-gray, positive=bright
                        proj_img = (np.clip(proj_np / (np.abs(proj_np).max() + 1e-8) * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
                        Image.fromarray(proj_img, 'L').save(
                            str(outdir / "maps" / f"{prompt_idx:04d}_{sample_idx:02d}_s{step_idx:03d}_proj.png"))
                        # Save soft mask
                        mask_np = soft_mask[0, 0].float().cpu().numpy()
                        mask_img = (np.clip(mask_np, 0, 1) * 255).astype(np.uint8)
                        Image.fromarray(mask_img, 'L').save(
                            str(outdir / "maps" / f"{prompt_idx:04d}_{sample_idx:02d}_s{step_idx:03d}_mask.png"))

                else:
                    eps_final = eps_cfg

                # =========================================================
                # DDIM step
                # =========================================================
                latents_prev = latents.clone()
                latents = scheduler.step(eps_final, t, latents).prev_sample

                # NaN recovery
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    print(f"  WARNING: NaN/Inf at step {step_idx}, reverting to standard CFG")
                    eps_fallback = eps_null + args.cfg_scale * (eps_prompt - eps_null)
                    latents = scheduler.step(eps_fallback, t, latents_prev).prev_sample

                # Debug logging
                if args.debug and step_idx % 10 == 0:
                    status = "GUIDED" if should_guide else ("CAS_ON" if should_trigger else "skip")
                    area_s = f" area={mask_areas[-1]:.3f}" if should_guide and mask_areas else ""
                    proj_s = f" proj={proj_values[-1]:.3f}" if should_guide and proj_values else ""
                    print(f"  [{step_idx:02d}] t={t.item():.0f} CAS={cas_val:.3f} {status}{area_s}{proj_s}")

            # ---- Decode latents to image ----
            with torch.no_grad():
                decoded = vae.decode(latents.to(vae.dtype) / vae.config.scaling_factor).sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                img_np = (decoded[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

            # Save image
            slug = make_safe_filename(prompt)
            fname = f"{prompt_idx:04d}_{sample_idx:02d}_{slug}.png"
            Image.fromarray(img_np).resize((512, 512)).save(str(outdir / fname))

            # Collect stats
            stats = {
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "seed": seed,
                "prompt": prompt[:100],
                "filename": fname,
                "guided_steps": guided_count,
                "total_steps": total_steps,
                "guidance_ratio": guided_count / max(total_steps, 1),
                "cas_triggered": cas.triggered,
                "avg_cas": float(np.mean(cas_values)) if cas_values else 0.0,
                "max_cas": float(np.max(cas_values)) if cas_values else 0.0,
                "avg_mask_area": float(np.mean(mask_areas)) if mask_areas else 0.0,
                "max_mask_area": float(np.max(mask_areas)) if mask_areas else 0.0,
                "avg_projection": float(np.mean(proj_values)) if proj_values else 0.0,
                "max_projection": float(np.max(proj_values)) if proj_values else 0.0,
            }
            all_stats.append(stats)

            if prompt_idx % 10 == 0 or args.debug:
                tqdm.write(
                    f"  [{prompt_idx:03d}_{sample_idx}] guided={guided_count}/{total_steps} "
                    f"CAS avg={stats['avg_cas']:.3f} max={stats['max_cas']:.3f} "
                    f"mask_area={stats['avg_mask_area']:.3f} proj={stats['avg_projection']:.3f}"
                )

    # ---- Save summary statistics ----
    n = len(all_stats)
    n_trig = sum(1 for s in all_stats if s["guided_steps"] > 0)
    summary = {
        "method": "Noise-Space Anchor Projection v5",
        "args": vars(args),
        "overall": {
            "total_images": n,
            "triggered": n_trig,
            "trigger_rate": n_trig / max(n, 1),
            "avg_guided_steps": float(np.mean([s["guided_steps"] for s in all_stats])) if n else 0,
            "avg_cas": float(np.mean([s["avg_cas"] for s in all_stats if s["avg_cas"] > 0])) if n else 0,
            "avg_mask_area": float(np.mean([s["avg_mask_area"] for s in all_stats if s["avg_mask_area"] > 0])) if n else 0,
            "avg_projection": float(np.mean([s["avg_projection"] for s in all_stats if s["avg_projection"] != 0])) if n else 0,
        },
        "per_image": all_stats,
    }
    with open(outdir / "stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"DONE! {n} images generated, {n_trig} triggered ({100*n_trig/max(n,1):.0f}%)")
    print(f"  Guide mode: {args.guide_mode}, safety_scale={args.safety_scale}")
    print(f"  Avg guided steps: {summary['overall']['avg_guided_steps']:.1f}/{total_steps if n else 0}")
    print(f"  Avg mask area: {summary['overall']['avg_mask_area']:.3f}")
    print(f"  Avg projection: {summary['overall']['avg_projection']:.3f}")
    print(f"  Output: {outdir}")
    print(f"  Stats:  {outdir / 'stats.json'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
