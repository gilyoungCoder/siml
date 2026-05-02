#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GLASS Exemplar-Based Safe Generation v7: Training-Free Safe Generation

Architecture:
  WHEN (Global CAS): cos(d_prompt, d_exemplar_target) > tau
  WHERE (Spatial CAS): per-pixel cosine with pre-computed exemplar direction
  HOW: hybrid mode with exemplar-based target/anchor directions

Key differences from v4:
  - Pre-computed anchor direction from diverse clothed exemplar images (offline)
  - Pass 3 (target) kept online for accurate CAS, Pass 4 (anchor) uses exemplar
  - 3 UNet calls per guided step (vs v4's 4) -> ~25% speedup
  - Exemplar anchor direction captures real "clothed" concept from diverse images
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
# Global CAS v7: Concept Alignment Score with pre-computed directions (WHEN)
# =============================================================================
class GlobalCAS_v7:
    """Detect harmful prompt alignment using pre-computed exemplar directions."""

    def __init__(self, threshold: float = 0.6, sticky: bool = True):
        self.threshold = threshold
        self.sticky = sticky
        self.triggered = False

    def reset(self):
        self.triggered = False

    def compute(self, eps_prompt, eps_null, eps_target=None, d_target_global=None):
        """
        Compute global CAS from noise predictions.

        Supports two modes:
        1. Online (text mode): pass eps_target, computes direction on the fly
        2. Exemplar mode: pass d_target_global (pre-computed global direction)

        Args:
            eps_prompt: noise prediction conditioned on prompt [1, 4, 64, 64]
            eps_null:   unconditional noise prediction [1, 4, 64, 64]
            eps_target: (optional) noise prediction conditioned on target concept [1, 4, 64, 64]
            d_target_global: (optional) pre-computed target global direction [16384]

        Returns:
            cas_value: float, cosine similarity between prompt and target directions
            should_guide: bool, whether guidance should be applied this step
        """
        d_prompt = (eps_prompt - eps_null).reshape(1, -1).float()

        if d_target_global is not None:
            # Exemplar mode: use pre-computed global direction
            d_target = d_target_global.unsqueeze(0).float() if d_target_global.dim() == 1 else d_target_global.float()
        elif eps_target is not None:
            # Online (text) mode: compute from UNet output
            d_target = (eps_target - eps_null).reshape(1, -1).float()
        else:
            raise ValueError("Either eps_target or d_target_global must be provided")

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
# Spatial CAS: Per-Pixel Concept Alignment (WHERE)
# =============================================================================
def compute_spatial_cas(
    eps_prompt: torch.Tensor,   # [1, 4, 64, 64]
    eps_null: torch.Tensor,
    eps_target: torch.Tensor,
    neighborhood_size: int = 3,
) -> torch.Tensor:
    """
    Compute per-pixel Spatial CAS using neighborhood pooling.

    For each pixel (h, w), computes cosine similarity between the prompt
    direction and target direction vectors. Raw 4-dim vectors are too noisy,
    so we use neighborhood pooling (unfold with kernel_size=neighborhood_size)
    to get more robust vectors (e.g., 3x3 * 4 channels = 36-dim).

    Args:
        eps_prompt: prompt noise prediction [1, 4, H, W]
        eps_null:   unconditional noise prediction [1, 4, H, W]
        eps_target: target concept noise prediction [1, 4, H, W]
        neighborhood_size: kernel size for neighborhood pooling (default: 3)

    Returns:
        spatial_cas: per-pixel CAS map [H, W] with values in [-1, 1]
    """
    d_prompt = (eps_prompt - eps_null).float()  # [1, 4, H, W]
    d_target = (eps_target - eps_null).float()

    H, W = d_prompt.shape[2], d_prompt.shape[3]
    pad = neighborhood_size // 2

    # Unfold: extract neighborhoods -> [1, C * k * k, H * W]
    d_prompt_unfolded = F.unfold(d_prompt, kernel_size=neighborhood_size, padding=pad)
    d_target_unfolded = F.unfold(d_target, kernel_size=neighborhood_size, padding=pad)

    # Per-pixel cosine similarity along the feature dimension
    spatial_cas = F.cosine_similarity(d_prompt_unfolded, d_target_unfolded, dim=1)  # [1, H*W]
    spatial_cas = spatial_cas.reshape(H, W)

    return spatial_cas


def compute_spatial_cas_with_dir(
    d_prompt: torch.Tensor,     # [1, 4, H, W] already computed (eps_prompt - eps_null)
    d_target: torch.Tensor,     # [1, 4, H, W] pre-computed exemplar direction
    neighborhood_size: int = 3,
) -> torch.Tensor:
    """
    Compute per-pixel Spatial CAS using pre-computed directions (exemplar mode).

    Same as compute_spatial_cas but takes d_prompt and d_target directly,
    instead of computing them from eps values.

    Args:
        d_prompt: prompt direction [1, 4, H, W] (eps_prompt - eps_null)
        d_target: target direction [1, 4, H, W] (pre-computed exemplar direction)
        neighborhood_size: kernel size for neighborhood pooling (default: 3)

    Returns:
        spatial_cas: per-pixel CAS map [H, W] with values in [-1, 1]
    """
    d_prompt_f = d_prompt.float()
    d_target_f = d_target.float()

    H, W = d_prompt_f.shape[2], d_prompt_f.shape[3]
    pad = neighborhood_size // 2

    # Unfold: extract neighborhoods -> [1, C * k * k, H * W]
    d_prompt_unfolded = F.unfold(d_prompt_f, kernel_size=neighborhood_size, padding=pad)
    d_target_unfolded = F.unfold(d_target_f, kernel_size=neighborhood_size, padding=pad)

    # Per-pixel cosine similarity along the feature dimension
    spatial_cas = F.cosine_similarity(d_prompt_unfolded, d_target_unfolded, dim=1)  # [1, H*W]
    spatial_cas = spatial_cas.reshape(H, W)

    return spatial_cas


def compute_soft_mask(
    spatial_cas: torch.Tensor,    # [H, W]
    spatial_threshold: float = 0.3,
    sigmoid_alpha: float = 10.0,
    blur_sigma: float = 1.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Convert spatial CAS map to a soft guidance mask using sigmoid + Gaussian blur.

    High CAS -> 1 (needs guidance), Low CAS -> 0 (keep original).

    Args:
        spatial_cas: per-pixel CAS map [H, W]
        spatial_threshold: center of sigmoid transition
        sigmoid_alpha: sharpness of sigmoid (higher = sharper transition)
        blur_sigma: sigma for Gaussian blur smoothing (0 = no blur)

    Returns:
        soft_mask: [1, 1, H, W] smooth mask in [0, 1]
    """
    # Sigmoid: maps CAS values around threshold to [0, 1]
    soft_mask = torch.sigmoid(sigmoid_alpha * (spatial_cas - spatial_threshold))
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
    **kwargs,
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

    Args:
        eps_cfg:      standard CFG noise prediction
        eps_null:     unconditional noise prediction
        eps_prompt:   prompt-conditioned noise prediction
        eps_target:   target concept noise prediction
        eps_anchor:   anchor concept noise prediction
        soft_mask:    [1, 1, H, W] guidance mask
        guide_mode:   "anchor_inpaint", "sld", or "hybrid"
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
        # Hybrid: subtract target direction AND add anchor direction (both relative to null)
        # eps_final = eps_cfg - t_s*M*(eps_target - eps_null) + a_s*M*(eps_anchor - eps_null)
        t_scale = kwargs.get("target_scale", safety_scale)
        a_scale = kwargs.get("anchor_scale", safety_scale)
        eps_final = eps_cfg \
                    - t_scale * mask * (eps_target - eps_null) \
                    + a_scale * mask * (eps_anchor - eps_null)

    elif guide_mode == "hybrid_proj":
        # Hybrid + Projection: project out nudity from prompt, then add anchor
        # Step 1: Remove nudity component from prompt direction
        # Step 2: Add anchor direction on top
        # eps_safe_cfg = eps_null + cfg_scale * (d_prompt - proj(d_prompt, d_target))
        # eps_final = eps_safe_cfg*(1-a*M) + eps_anchor_cfg*(a*M)
        d_prompt = eps_prompt - eps_null
        d_target = eps_target - eps_null

        # Project out nudity
        dot = (d_prompt * d_target).sum(dim=1, keepdim=True)
        norm_sq = (d_target * d_target).sum(dim=1, keepdim=True).clamp(min=1e-8)
        proj = (dot / norm_sq) * d_target

        # Safe prompt = prompt minus nudity, scaled by projection strength
        p_scale = kwargs.get("proj_scale", 1.0)
        d_safe = d_prompt - p_scale * proj

        # Reconstruct safe CFG (prompt direction with nudity removed)
        eps_safe_cfg = eps_null + cfg_scale * d_safe

        # Anchor CFG
        a_scale = kwargs.get("anchor_scale", safety_scale)
        eps_anchor_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)

        # Blend: safe_cfg in unmasked, anchor in heavily masked
        eps_final = eps_safe_cfg * (1.0 - a_scale * mask) + eps_anchor_cfg * (a_scale * mask)

    elif guide_mode == "projection":
        # Projection: remove nudity component from prompt direction, preserve rest
        # d_prompt = eps_prompt - eps_null (prompt direction)
        # d_target = eps_target - eps_null (nudity direction)
        # d_safe = d_prompt - proj(d_prompt, d_target) (nudity removed)
        # eps_safe_cfg = eps_null + cfg_scale * d_safe
        # eps_final = eps_cfg*(1-s*M) + eps_safe_cfg*(s*M)
        d_prompt = eps_prompt - eps_null
        d_target = eps_target - eps_null

        # Project out the target (nudity) component from prompt direction
        # proj(a, b) = (a*b / b*b) * b
        dot = (d_prompt * d_target).sum(dim=1, keepdim=True)
        norm_sq = (d_target * d_target).sum(dim=1, keepdim=True).clamp(min=1e-8)
        proj = (dot / norm_sq) * d_target  # nudity component of prompt

        # Safe prompt direction = prompt minus nudity component
        d_safe = d_prompt - safety_scale * proj  # s controls how much nudity to remove

        # Reconstruct CFG with safe direction
        cfg_scale = kwargs.get("cfg_scale", 7.5)
        eps_safe_cfg = eps_null + cfg_scale * d_safe

        # Blend: use safe CFG in masked regions, original CFG elsewhere
        eps_final = eps_cfg * (1.0 - mask) + eps_safe_cfg * mask

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
    p = ArgumentParser(description="GLASS Exemplar-Based Safe Generation v7")
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
    p.add_argument("--cas_threshold", type=float, default=0.6,
                    help="Global CAS trigger threshold")
    p.add_argument("--cas_sticky", action="store_true", default=True,
                    help="Once triggered, guidance stays on")
    p.add_argument("--cas_no_sticky", action="store_true",
                    help="Disable sticky CAS mode")

    # Spatial CAS (Where)
    p.add_argument("--spatial_threshold", type=float, default=0.3,
                    help="Per-pixel CAS threshold (sigmoid center)")
    p.add_argument("--sigmoid_alpha", type=float, default=10.0,
                    help="Sharpness of sigmoid mask transition")
    p.add_argument("--neighborhood_size", type=int, default=3,
                    help="Pooling kernel size for robust per-pixel CAS")
    p.add_argument("--blur_sigma", type=float, default=1.0,
                    help="Gaussian blur sigma for mask smoothing (0=no blur)")

    # Guidance (How)
    p.add_argument("--guide_mode", type=str, default="hybrid",
                    choices=["anchor_inpaint", "sld", "hybrid", "projection", "hybrid_proj"],
                    help="Guidance mode")
    p.add_argument("--safety_scale", type=float, default=1.0,
                    help="Guidance strength (anchor_inpaint: 0-1, sld: 1-10)")
    p.add_argument("--target_scale", type=float, default=-1.0,
                    help="Hybrid only: target repulsion scale (default: same as safety_scale)")
    p.add_argument("--anchor_scale", type=float, default=-1.0,
                    help="Hybrid only: anchor attraction scale (default: same as safety_scale)")
    p.add_argument("--proj_scale", type=float, default=1.0,
                    help="hybrid_proj only: projection strength for nudity removal (default: 1.0)")
    p.add_argument("--guide_start_frac", type=float, default=0.0,
                    help="Fraction of total steps before guidance starts (0=all steps)")

    # Concepts (used in text and hybrid_exemplar modes)
    p.add_argument("--target_concepts", type=str, nargs="+",
                    default=["nudity", "nude person", "naked body"],
                    help="Target concepts to erase")
    p.add_argument("--anchor_concepts", type=str, nargs="+",
                    default=["clothed person", "person wearing clothes"],
                    help="Anchor concepts to guide toward")

    # v7: Exemplar mode args
    p.add_argument("--concept_dir_path", type=str, default=None,
                    help="Path to concept_directions.pt (required for exemplar mode)")
    p.add_argument("--exemplar_mode", type=str, default="exemplar",
                    choices=["exemplar", "text", "hybrid_exemplar"],
                    help="Direction source: exemplar (pre-computed), text (online UNet), "
                         "hybrid_exemplar (weighted blend)")
    p.add_argument("--exemplar_weight", type=float, default=0.7,
                    help="Weight for exemplar directions in hybrid_exemplar mode (0-1)")

    # Misc
    p.add_argument("--save_maps", action="store_true",
                    help="Save spatial CAS maps and masks for debugging")
    p.add_argument("--debug", action="store_true",
                    help="Print per-step debug info")
    p.add_argument("--start_idx", type=int, default=0,
                    help="Start prompt index (for parallel runs)")
    p.add_argument("--end_idx", type=int, default=-1,
                    help="End prompt index (-1 = all)")

    args = p.parse_args()
    if args.cas_no_sticky:
        args.cas_sticky = False

    # Validate exemplar mode requirements
    if args.exemplar_mode in ("exemplar", "hybrid_exemplar") and args.concept_dir_path is None:
        p.error("--concept_dir_path is required for exemplar and hybrid_exemplar modes")

    return args


# =============================================================================
# Main Generation Loop
# =============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"GLASS Exemplar-Based Safe Generation v7")
    print(f"{'='*70}")
    print(f"  MODE:  {args.exemplar_mode}"
          + (f" (weight={args.exemplar_weight})" if args.exemplar_mode == "hybrid_exemplar" else ""))
    print(f"  WHEN:  Global CAS threshold={args.cas_threshold}, sticky={args.cas_sticky}")
    print(f"  WHERE: Spatial CAS, neighborhood={args.neighborhood_size}, "
          f"threshold={args.spatial_threshold}, alpha={args.sigmoid_alpha}, blur={args.blur_sigma}")
    print(f"  HOW:   {args.guide_mode}, safety_scale={args.safety_scale}, "
          f"start_frac={args.guide_start_frac}")
    if args.exemplar_mode in ("exemplar", "hybrid_exemplar"):
        print(f"  Concept directions: {args.concept_dir_path}")
    if args.exemplar_mode in ("text", "hybrid_exemplar"):
        print(f"  Target concepts: {args.target_concepts}")
        print(f"  Anchor concepts: {args.anchor_concepts}")
    print(f"  Model: {args.ckpt}")
    print(f"  Steps: {args.steps}, CFG: {args.cfg_scale}, Samples/prompt: {args.nsamples}")
    print(f"{'='*70}\n")

    # ---- Load pre-computed concept directions (exemplar / hybrid_exemplar) ----
    target_dirs = None
    anchor_dirs = None
    target_global = None
    anchor_global = None

    if args.exemplar_mode in ("exemplar", "hybrid_exemplar"):
        print(f"Loading concept directions from {args.concept_dir_path} ...")
        concept_data = torch.load(args.concept_dir_path, map_location=device)
        target_dirs = concept_data['target_directions']    # {int: [4,64,64]}
        anchor_dirs = concept_data['anchor_directions']    # {int: [4,64,64]}
        target_global = concept_data['target_global']      # {int: [16384]}
        anchor_global = concept_data['anchor_global']      # {int: [16384]}
        print(f"  Loaded directions for {len(target_dirs)} timesteps")

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
    # Target embeds always needed (CAS uses online eps_target in all modes)
    # Anchor embeds needed for text and hybrid_exemplar modes (exemplar mode uses pre-computed)
    with torch.no_grad():
        target_embeds = encode_concepts(text_encoder, tokenizer,
                                        args.target_concepts, device)
    anchor_embeds = None
    if args.exemplar_mode in ("text", "hybrid_exemplar"):
        with torch.no_grad():
            anchor_embeds = encode_concepts(text_encoder, tokenizer,
                                            args.anchor_concepts, device)
    print(f"Embeddings encoded: target={len(args.target_concepts)} concepts (averaged)"
          + (f", anchor={len(args.anchor_concepts)} concepts (averaged)" if anchor_embeds is not None else ", anchor=exemplar"))

    # Unconditional (empty string) — always needed
    with torch.no_grad():
        uncond_inputs = tokenizer(
            "", padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]

    # ---- Init CAS ----
    cas = GlobalCAS_v7(threshold=args.cas_threshold, sticky=args.cas_sticky)

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
                # Step 1: UNet forward passes
                # =========================================================
                lat_in = scheduler.scale_model_input(latents, t)
                t_int = t.item()

                with torch.no_grad():
                    # Pass 1 & 2: unconditional + prompt (batched) — always needed
                    lat_batch = torch.cat([lat_in, lat_in])
                    embed_batch = torch.cat([uncond_embeds, prompt_embeds])
                    raw = unet(lat_batch, t, encoder_hidden_states=embed_batch).sample
                    eps_null, eps_prompt = raw.chunk(2)

                # Standard CFG
                eps_cfg = eps_null + args.cfg_scale * (eps_prompt - eps_null)

                # =========================================================
                # Step 2: Get target/anchor directions based on mode
                # =========================================================
                if args.exemplar_mode == "exemplar":
                    # --- Exemplar mode: Pass 3 online (CAS needs current z_t), Pass 4 exemplar ---
                    # Pass 3: target concept (online, needed for accurate CAS)
                    with torch.no_grad():
                        eps_target_online = unet(lat_in, t,
                                                 encoder_hidden_states=target_embeds).sample
                    eps_target_use = eps_target_online

                    # Anchor: use pre-computed exemplar direction (no Pass 4!)
                    d_anchor = anchor_dirs[t_int].to(device, dtype=torch.float16).unsqueeze(0)
                    eps_anchor_use = eps_null + d_anchor

                    # Global CAS with online target (accurate, image-specific)
                    cas_val, should_trigger = cas.compute(
                        eps_prompt, eps_null, eps_target=eps_target_online
                    )

                elif args.exemplar_mode == "text":
                    # --- Text mode: same as v4, online UNet calls ---
                    with torch.no_grad():
                        eps_target_online = unet(lat_in, t,
                                                 encoder_hidden_states=target_embeds).sample

                    eps_target_use = eps_target_online
                    # eps_anchor computed later only if needed

                    # Global CAS with online target
                    cas_val, should_trigger = cas.compute(
                        eps_prompt, eps_null, eps_target=eps_target_online
                    )

                elif args.exemplar_mode == "hybrid_exemplar":
                    # --- Hybrid exemplar: blend pre-computed + online directions ---
                    d_target_exemplar = target_dirs[t_int].to(device, dtype=torch.float16).unsqueeze(0)
                    d_anchor_exemplar = anchor_dirs[t_int].to(device, dtype=torch.float16).unsqueeze(0)
                    d_target_global_vec = target_global[t_int].to(device)

                    with torch.no_grad():
                        eps_target_online = unet(lat_in, t,
                                                 encoder_hidden_states=target_embeds).sample
                    d_target_online = eps_target_online - eps_null

                    # Weighted blend of exemplar and online directions
                    w = args.exemplar_weight
                    d_target_blended = w * d_target_exemplar + (1 - w) * d_target_online

                    eps_target_use = eps_null + d_target_blended
                    # eps_anchor computed later only if needed, also blended

                    # Global CAS: blend global directions too
                    d_target_online_global = (eps_target_online - eps_null).reshape(1, -1).float()
                    d_target_exemplar_global = d_target_global_vec.unsqueeze(0).float()
                    d_target_blended_global = w * d_target_exemplar_global + (1 - w) * d_target_online_global
                    cas_val, should_trigger = cas.compute(
                        eps_prompt, eps_null,
                        d_target_global=d_target_blended_global.squeeze(0)
                    )

                cas_values.append(cas_val)

                in_window = step_idx >= guide_start_step
                should_guide = should_trigger and in_window

                if should_guide:
                    # =====================================================
                    # Get anchor (mode-dependent)
                    # =====================================================
                    if args.exemplar_mode == "exemplar":
                        # eps_anchor_use already set from pre-computed direction above
                        pass

                    elif args.exemplar_mode == "text":
                        # Online anchor UNet call (only when guidance needed)
                        with torch.no_grad():
                            eps_anchor_online = unet(lat_in, t,
                                                     encoder_hidden_states=anchor_embeds).sample
                        eps_anchor_use = eps_anchor_online

                    elif args.exemplar_mode == "hybrid_exemplar":
                        # Blend exemplar + online anchor
                        with torch.no_grad():
                            eps_anchor_online = unet(lat_in, t,
                                                     encoder_hidden_states=anchor_embeds).sample
                        d_anchor_online = eps_anchor_online - eps_null
                        w = args.exemplar_weight
                        d_anchor_blended = w * d_anchor_exemplar + (1 - w) * d_anchor_online
                        eps_anchor_use = eps_null + d_anchor_blended

                    # =====================================================
                    # Step 3: Spatial CAS (WHERE to guide)
                    # =====================================================
                    if args.exemplar_mode == "exemplar":
                        # Use online target for spatial CAS (image-specific)
                        spatial_cas = compute_spatial_cas(
                            eps_prompt, eps_null, eps_target_use,
                            neighborhood_size=args.neighborhood_size,
                        )
                    elif args.exemplar_mode == "hybrid_exemplar":
                        # Use blended direction
                        d_prompt_spatial = (eps_prompt - eps_null).float()
                        spatial_cas = compute_spatial_cas_with_dir(
                            d_prompt_spatial, d_target_blended.float(),
                            neighborhood_size=args.neighborhood_size,
                        )
                    else:
                        # Text mode: compute from eps values (same as v4)
                        spatial_cas = compute_spatial_cas(
                            eps_prompt, eps_null, eps_target_use,
                            neighborhood_size=args.neighborhood_size,
                        )

                    # =====================================================
                    # Step 4: Soft mask with sigmoid + Gaussian blur
                    # =====================================================
                    soft_mask = compute_soft_mask(
                        spatial_cas,
                        spatial_threshold=args.spatial_threshold,
                        sigmoid_alpha=args.sigmoid_alpha,
                        blur_sigma=args.blur_sigma,
                        device=device,
                    )

                    # =====================================================
                    # Step 5 & 6: Apply guidance
                    # =====================================================
                    eps_final = apply_guidance(
                        eps_cfg=eps_cfg,
                        eps_null=eps_null,
                        eps_prompt=eps_prompt,
                        eps_target=eps_target_use,
                        eps_anchor=eps_anchor_use,
                        soft_mask=soft_mask,
                        guide_mode=args.guide_mode,
                        safety_scale=args.safety_scale,
                        cfg_scale=args.cfg_scale,
                        target_scale=args.target_scale if args.target_scale > 0 else args.safety_scale,
                        anchor_scale=args.anchor_scale if args.anchor_scale > 0 else args.safety_scale,
                        proj_scale=args.proj_scale,
                    )

                    guided_count += 1
                    area_val = float(soft_mask.mean().item())
                    mask_areas.append(area_val)

                    # Save spatial maps for debugging
                    if args.save_maps and step_idx % 10 == 0:
                        # Save spatial CAS map
                        cas_map_np = spatial_cas.float().cpu().numpy()
                        cas_map_np = np.nan_to_num(cas_map_np, nan=0.0)
                        cas_map_img = (np.clip((cas_map_np + 1) / 2, 0, 1) * 255).astype(np.uint8)
                        Image.fromarray(cas_map_img, 'L').save(
                            str(outdir / "maps" / f"{prompt_idx:04d}_{sample_idx:02d}_s{step_idx:03d}_cas.png"))
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
                    print(f"  [{step_idx:02d}] t={t.item():.0f} CAS={cas_val:.3f} {status}{area_s}")

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
                "exemplar_mode": args.exemplar_mode,
            }
            all_stats.append(stats)

            if prompt_idx % 10 == 0 or args.debug:
                tqdm.write(
                    f"  [{prompt_idx:03d}_{sample_idx}] guided={guided_count}/{total_steps} "
                    f"CAS avg={stats['avg_cas']:.3f} max={stats['max_cas']:.3f} "
                    f"mask_area={stats['avg_mask_area']:.3f}"
                )

    # ---- Save summary statistics ----
    n = len(all_stats)
    n_trig = sum(1 for s in all_stats if s["guided_steps"] > 0)
    summary = {
        "method": "GLASS Exemplar-Based Safe Generation v7",
        "exemplar_mode": args.exemplar_mode,
        "args": vars(args),
        "overall": {
            "total_images": n,
            "triggered": n_trig,
            "trigger_rate": n_trig / max(n, 1),
            "avg_guided_steps": float(np.mean([s["guided_steps"] for s in all_stats])) if n else 0,
            "avg_cas": float(np.mean([s["avg_cas"] for s in all_stats if s["avg_cas"] > 0])) if n else 0,
            "avg_mask_area": float(np.mean([s["avg_mask_area"] for s in all_stats if s["avg_mask_area"] > 0])) if n else 0,
        },
        "per_image": all_stats,
    }
    with open(outdir / "stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"DONE! {n} images generated, {n_trig} triggered ({100*n_trig/max(n,1):.0f}%)")
    print(f"  Exemplar mode: {args.exemplar_mode}")
    print(f"  Guide mode: {args.guide_mode}, safety_scale={args.safety_scale}")
    print(f"  Avg guided steps: {summary['overall']['avg_guided_steps']:.1f}/{total_steps if n else 0}")
    print(f"  Avg mask area: {summary['overall']['avg_mask_area']:.3f}")
    print(f"  Output: {outdir}")
    print(f"  Stats:  {outdir / 'stats.json'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
