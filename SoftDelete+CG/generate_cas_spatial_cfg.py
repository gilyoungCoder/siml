#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAS + Spatial CFG: Training-Free Safe Generation

Pipeline (fully sample-based, no classifier needed):
  - WHEN (CAS): Concept Alignment Score
      d_prompt = ε(prompt) - ε(∅)     # already computed in CFG — free
      d_target = ε("nudity") - ε(∅)   # 1 extra UNet forward
      CAS = cos(d_prompt, d_target)    # global cosine similarity
      If CAS > τ → trigger guidance (sticky mode available)

  - WHERE (Spatial CFG):
      d_target = ε("nudity") - ε(∅)        at spatial level (4×64×64)
      d_anchor = ε("clothed person") - ε(∅)  at spatial level
      spatial_score[h,w] = cos(d_target, d_anchor) per pixel
      Where target strongly differs from anchor → nudity-specific region
      Apply safety guidance only there, with adaptive strength

  - GUIDANCE (Safety Guidance, DAG-inspired):
      ε_safe = ε_cfg - S · M · (ε_target - ε_anchor)
      S = spatial scale map (stronger where target deviates from anchor)
      M = binary/soft mask from spatial score

Key advantages:
  - Completely training-free: no classifier, no fine-tuning
  - Sample-adaptive: only guides when CAS detects harmful alignment
  - Spatially precise: only edits regions where nudity is localized
  - Minimal overhead: 2 extra UNet forwards (target + anchor) per guided step
"""

import os
import sys
import json
import math
import random
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
# CAS: Concept Alignment Score (the "WHEN")
# =============================================================================
class ConceptAlignmentScore:
    """
    Detects whether the current prompt is generating harmful content
    by measuring cosine alignment between prompt direction and target direction.

    d_prompt = ε(prompt) - ε(∅)      # direction the prompt pushes generation
    d_target = ε("nudity") - ε(∅)   # direction the target concept pushes
    CAS = cos(d_prompt, d_target)

    High CAS → prompt aligns with harmful concept → trigger guidance
    Low CAS  → prompt is orthogonal to harmful → skip guidance
    """

    def __init__(self, threshold: float = 0.3, sticky: bool = True):
        self.threshold = threshold
        self.sticky = sticky
        self.triggered = False

    def reset(self):
        self.triggered = False

    def compute(
        self,
        noise_pred_text: torch.Tensor,  # ε(prompt)
        noise_pred_uncond: torch.Tensor,  # ε(∅)
        noise_pred_target: torch.Tensor,  # ε("nudity")
    ) -> Tuple[float, bool]:
        """
        Compute CAS and decide whether to trigger guidance.

        Returns:
            cas_value: float, cosine similarity in [-1, 1]
            should_guide: bool
        """
        # Compute directions (global, flattened)
        d_prompt = (noise_pred_text - noise_pred_uncond).reshape(1, -1)
        d_target = (noise_pred_target - noise_pred_uncond).reshape(1, -1)

        cas = F.cosine_similarity(d_prompt, d_target, dim=-1).item()

        if self.sticky and self.triggered:
            return cas, True

        should_guide = cas > self.threshold

        if should_guide and self.sticky:
            self.triggered = True

        return cas, should_guide


# =============================================================================
# Spatial CFG: Noise Direction Difference Map (the "WHERE")
# =============================================================================
class SpatialCFGGuidance:
    """
    Computes spatial guidance map by comparing noise prediction directions.

    Key insight: noise predictions are 4-channel latent space (too few dims for
    reliable per-pixel cosine). Solutions:
      - Patch-based methods: use 3×3 neighborhoods (36 dims) for robust cosine
      - Percentile normalization: robust to outliers unlike min-max

    Methods:
      1. "spatial_cas": per-pixel cos(d_prompt, d_target)
         → WHERE the prompt aligns with target concept
      2. "patch_cas": 3×3 patch cos(d_prompt, d_target)
         → same but with 36-dim patches for robustness
      3. "weighted_cas": spatial_cas × ||d_target||
         → alignment weighted by target strength
      4. "target_strength": ||d_target|| - ||d_anchor|| per pixel
      5. "diff_norm": ||d_target - d_anchor|| per pixel
      6. "cosine_diff": 1 - cos(d_target, d_anchor) per pixel
      7. "target_projection": proj of d_prompt onto (d_target - d_anchor)
    """

    def __init__(
        self,
        spatial_method: str = "spatial_cas",
        spatial_threshold: float = 0.3,
        guidance_scale_high: float = 5.0,
        guidance_scale_low: float = 0.0,
        soft_mask: bool = True,
        mask_blur_sigma: float = 2.0,
        norm_method: str = "percentile",  # "percentile" or "minmax"
        patch_size: int = 3,
    ):
        self.spatial_method = spatial_method
        self.spatial_threshold = spatial_threshold
        self.guidance_scale_high = guidance_scale_high
        self.guidance_scale_low = guidance_scale_low
        self.soft_mask = soft_mask
        self.mask_blur_sigma = mask_blur_sigma
        self.norm_method = norm_method
        self.patch_size = patch_size

    def _extract_patches(self, x: torch.Tensor, patch_size: int = 3) -> torch.Tensor:
        """Extract overlapping patches: (1, C, H, W) → (1, C*p*p, H, W)."""
        pad = patch_size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        # Unfold into patches
        patches = x_padded.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
        # (1, C, H, W, p, p) → (1, C*p*p, H, W)
        B, C, H, W, p1, p2 = patches.shape
        return patches.reshape(B, C * p1 * p2, H, W)

    def compute_spatial_map(
        self,
        noise_pred_uncond: torch.Tensor,   # ε(∅): (1, 4, 64, 64)
        noise_pred_target: torch.Tensor,   # ε("nudity"): (1, 4, 64, 64)
        noise_pred_anchor: torch.Tensor,   # ε("clothed person"): (1, 4, 64, 64)
        noise_pred_text: torch.Tensor = None,  # ε(prompt): for CAS-based methods
    ) -> torch.Tensor:
        """
        Compute spatial guidance map.

        Returns:
            spatial_map: (1, 1, H, W) in [0, 1], higher = more guidance needed
        """
        d_target = noise_pred_target - noise_pred_uncond  # (1, 4, 64, 64)
        d_anchor = noise_pred_anchor - noise_pred_uncond  # (1, 4, 64, 64)

        if self.spatial_method == "spatial_cas":
            # Per-pixel cosine between d_prompt and d_target
            # Shows WHERE the prompt aligns with target concept
            assert noise_pred_text is not None
            d_prompt = noise_pred_text - noise_pred_uncond
            cos_sim = F.cosine_similarity(d_prompt, d_target, dim=1)  # (1, H, W)
            # Only positive alignment matters (negative = pushing away from target)
            spatial_score = F.relu(cos_sim).unsqueeze(1)  # (1, 1, H, W)
            # Subtract anchor alignment to get target-specific signal
            cos_anchor = F.cosine_similarity(d_prompt, d_anchor, dim=1)
            anchor_score = F.relu(cos_anchor).unsqueeze(1)
            # Where prompt aligns MORE with target than anchor = nudity region
            spatial_score = F.relu(spatial_score - anchor_score)

        elif self.spatial_method == "patch_cas":
            # Same as spatial_cas but with 3×3 patches (36 dims) for robustness
            assert noise_pred_text is not None
            d_prompt = noise_pred_text - noise_pred_uncond
            p = self.patch_size
            d_prompt_patches = self._extract_patches(d_prompt, p)  # (1, 4*p*p, H, W)
            d_target_patches = self._extract_patches(d_target, p)
            d_anchor_patches = self._extract_patches(d_anchor, p)
            # Per-pixel cosine with 36 dimensions — much more reliable
            cos_target = F.cosine_similarity(d_prompt_patches, d_target_patches, dim=1)  # (1, H, W)
            cos_anchor = F.cosine_similarity(d_prompt_patches, d_anchor_patches, dim=1)
            # Target-specific: where prompt aligns more with target than anchor
            spatial_score = F.relu(cos_target - cos_anchor).unsqueeze(1)  # (1, 1, H, W)

        elif self.spatial_method == "weighted_cas":
            # spatial_cas weighted by target direction magnitude
            # High where: prompt aligns with target AND target is strong
            assert noise_pred_text is not None
            d_prompt = noise_pred_text - noise_pred_uncond
            cos_sim = F.cosine_similarity(d_prompt, d_target, dim=1).unsqueeze(1)  # (1, 1, H, W)
            cos_anchor = F.cosine_similarity(d_prompt, d_anchor, dim=1).unsqueeze(1)
            alignment = F.relu(cos_sim - cos_anchor)
            # Weight by target magnitude
            target_mag = d_target.norm(dim=1, keepdim=True)
            spatial_score = alignment * target_mag

        elif self.spatial_method == "diff_norm":
            diff = d_target - d_anchor
            spatial_score = diff.norm(dim=1, keepdim=True)

        elif self.spatial_method == "target_strength":
            target_norm = d_target.norm(dim=1, keepdim=True)
            anchor_norm = d_anchor.norm(dim=1, keepdim=True)
            spatial_score = F.relu(target_norm - anchor_norm)

        elif self.spatial_method == "cosine_diff":
            cos_sim = F.cosine_similarity(d_target, d_anchor, dim=1)
            spatial_score = F.relu(1.0 - cos_sim).unsqueeze(1)

        elif self.spatial_method == "target_projection":
            assert noise_pred_text is not None
            d_prompt = noise_pred_text - noise_pred_uncond
            diff_dir = d_target - d_anchor
            projection = (d_prompt * diff_dir).sum(dim=1, keepdim=True)
            diff_norm = diff_dir.norm(dim=1, keepdim=True) + 1e-8
            projection = projection / diff_norm
            spatial_score = F.relu(projection)

        else:
            raise ValueError(f"Unknown spatial method: {self.spatial_method}")

        # Normalize to [0, 1]
        spatial_map = self._normalize_spatial_map(spatial_score)

        # Gaussian blur for smoother guidance regions
        if self.mask_blur_sigma > 0:
            spatial_map = self._gaussian_blur(spatial_map, self.mask_blur_sigma)
            # Re-normalize after blur
            spatial_map = self._normalize_spatial_map(spatial_map)

        return spatial_map

    def _normalize_spatial_map(self, score: torch.Tensor) -> torch.Tensor:
        """Normalize spatial score to [0, 1]."""
        if self.norm_method == "percentile":
            # Percentile-based normalization: robust to outliers
            flat = score.reshape(-1)
            if flat.max() - flat.min() < 1e-8:
                return torch.zeros_like(score)
            # Use 2nd and 98th percentile to avoid outlier squashing
            p_low = torch.quantile(flat.float(), 0.02)
            p_high = torch.quantile(flat.float(), 0.98)
            if p_high - p_low < 1e-8:
                p_low = flat.min()
                p_high = flat.max()
            normalized = (score - p_low) / (p_high - p_low + 1e-8)
            return normalized.clamp(0, 1)
        else:
            # Min-max normalization
            s_min = score.min()
            s_max = score.max()
            if s_max - s_min < 1e-8:
                return torch.zeros_like(score)
            return (score - s_min) / (s_max - s_min)

    def _gaussian_blur(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur to spatial map for smoother guidance regions."""
        if sigma <= 0:
            return x
        kernel_size = int(2 * math.ceil(2 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_h = kernel_1d.view(1, 1, -1, 1)
        kernel_w = kernel_1d.view(1, 1, 1, -1)

        pad = kernel_size // 2
        out = F.pad(x, (0, 0, pad, pad), mode='reflect')
        out = F.conv2d(out, kernel_h, groups=1)
        out = F.pad(out, (pad, pad, 0, 0), mode='reflect')
        out = F.conv2d(out, kernel_w, groups=1)

        return out

    def compute_guidance_weight_map(
        self,
        spatial_map: torch.Tensor,  # (1, 1, H, W) in [0, 1]
        adaptive_area_scale: bool = True,
    ) -> torch.Tensor:
        """
        Convert spatial map to guidance weight map.

        DAG-inspired adaptive scaling:
        - Detected area inversely scales per-pixel guidance
        - Soft mask: continuous weight proportional to confidence

        Returns:
            weight_map: (1, 1, H, W)
        """
        H, W = spatial_map.shape[-2:]

        mask = (spatial_map >= self.spatial_threshold).float()

        if self.soft_mask:
            # Continuous weight: map [threshold, 1] → [1, guidance_scale_high]
            soft_weight = torch.where(
                spatial_map >= self.spatial_threshold,
                1.0 + (spatial_map - self.spatial_threshold) / (1.0 - self.spatial_threshold + 1e-8)
                    * (self.guidance_scale_high - 1.0),
                torch.full_like(spatial_map, self.guidance_scale_low),
            )
        else:
            soft_weight = mask * self.guidance_scale_high + (1 - mask) * self.guidance_scale_low

        if adaptive_area_scale:
            area_fraction = mask.sum() / (H * W)
            if area_fraction > 0:
                area_scale = min(5.0, 1.0 / (area_fraction.sqrt() + 1e-4))
                soft_weight = soft_weight * area_scale

        return soft_weight


# =============================================================================
# Safety Guidance Application
# =============================================================================
def apply_safety_guidance(
    noise_pred_cfg: torch.Tensor,      # Standard CFG noise prediction
    noise_pred_uncond: torch.Tensor,   # ε(∅)
    noise_pred_target: torch.Tensor,   # ε("nudity")
    noise_pred_anchor: torch.Tensor,   # ε("clothed person")
    weight_map: torch.Tensor,          # Spatial weight map (1, 1, H, W)
    guidance_mode: str = "anchor_shift",
    global_scale: float = 1.0,
) -> torch.Tensor:
    """
    Apply spatially-weighted safety guidance to the noise prediction.

    Modes:
      - "anchor_shift": Push from target toward anchor direction
          ε_safe = ε_cfg + scale * weight * (ε_anchor - ε_target)
      - "target_negate": Negate the target direction (SLD-style)
          ε_safe = ε_cfg - scale * weight * (ε_target - ε_uncond)
      - "dual": Combine both
          ε_safe = ε_cfg + scale * weight * [(ε_anchor - ε_uncond) - (ε_target - ε_uncond)]
                 = ε_cfg + scale * weight * (ε_anchor - ε_target)

    Returns:
        noise_pred_safe: Modified noise prediction
    """
    if guidance_mode == "anchor_shift":
        # Push from target direction toward anchor direction
        safety_direction = noise_pred_anchor - noise_pred_target  # (1, 4, H, W)
        noise_pred_safe = noise_pred_cfg + global_scale * weight_map * safety_direction

    elif guidance_mode == "target_negate":
        # Negate the target concept direction (SLD-style)
        target_direction = noise_pred_target - noise_pred_uncond  # (1, 4, H, W)
        noise_pred_safe = noise_pred_cfg - global_scale * weight_map * target_direction

    elif guidance_mode == "dual":
        # Both negate target and amplify anchor
        target_dir = noise_pred_target - noise_pred_uncond
        anchor_dir = noise_pred_anchor - noise_pred_uncond
        safety_direction = anchor_dir - target_dir  # = anchor - target
        noise_pred_safe = noise_pred_cfg + global_scale * weight_map * safety_direction

    else:
        raise ValueError(f"Unknown guidance mode: {guidance_mode}")

    return noise_pred_safe


# =============================================================================
# Utils
# =============================================================================
def load_prompts(f):
    import csv
    f = Path(f)
    if f.suffix == ".csv":
        prompts = []
        with open(f, "r") as fp:
            reader = csv.DictReader(fp)
            fieldnames = reader.fieldnames
            prompt_col = None
            for col in ['adv_prompt', 'sensitive prompt', 'prompt', 'target_prompt',
                        'text', 'Prompt', 'Text']:
                if col in fieldnames:
                    prompt_col = col
                    break
            if prompt_col is None:
                raise ValueError(f"No known prompt column in {fieldnames}")
            for row in reader:
                prompts.append(row[prompt_col].strip())
        return prompts
    else:
        return [l.strip() for l in open(f) if l.strip()]


def save_image(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.resize((512, 512)).save(path)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# =============================================================================
# Args
# =============================================================================
def parse_args():
    parser = ArgumentParser(description="CAS + Spatial CFG: Training-Free Safe Generation")
    parser.add_argument("--ckpt_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./scg_outputs/cas_spatial_cfg")
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # --- CAS (When) ---
    parser.add_argument("--cas_threshold", type=float, default=0.3,
                        help="CAS threshold to trigger guidance. ~0.3-0.6 typical range")
    parser.add_argument("--cas_sticky", action="store_true", default=True,
                        help="Once CAS triggers, keep guidance on for all remaining steps")
    parser.add_argument("--cas_no_sticky", action="store_true",
                        help="Disable sticky mode (re-evaluate CAS every step)")
    parser.add_argument("--cas_window_start", type=int, default=1000,
                        help="Only evaluate CAS in this timestep window (start)")
    parser.add_argument("--cas_window_end", type=int, default=0,
                        help="Only evaluate CAS in this timestep window (end)")

    # --- Target / Anchor concepts ---
    parser.add_argument("--target_concepts", type=str, nargs="+",
                        default=["nudity", "nude person", "nsfw person", "naked body"],
                        help="Target concepts to erase")
    parser.add_argument("--anchor_concepts", type=str, nargs="+",
                        default=["clothed person", "person wearing clothes", "fully dressed person"],
                        help="Anchor concepts (safe counterpart)")

    # --- Spatial CFG (Where) ---
    parser.add_argument("--spatial_method", type=str, default="spatial_cas",
                        choices=["spatial_cas", "patch_cas", "weighted_cas",
                                 "diff_norm", "target_strength", "cosine_diff", "target_projection"],
                        help="How to compute spatial guidance map")
    parser.add_argument("--spatial_threshold", type=float, default=0.3,
                        help="Threshold for spatial map to trigger per-pixel guidance")
    parser.add_argument("--spatial_scale_high", type=float, default=5.0,
                        help="Guidance scale for high-confidence unsafe pixels")
    parser.add_argument("--spatial_scale_low", type=float, default=0.0,
                        help="Guidance scale for low-confidence safe pixels")
    parser.add_argument("--soft_mask", action="store_true", default=True,
                        help="Use soft spatial mask (gradient) vs hard binary mask")
    parser.add_argument("--hard_mask", action="store_true",
                        help="Use hard binary mask instead of soft")
    parser.add_argument("--mask_blur_sigma", type=float, default=1.0,
                        help="Gaussian blur sigma for spatial mask smoothing")
    parser.add_argument("--adaptive_area_scale", action="store_true", default=True,
                        help="Scale guidance inversely with detected unsafe area")

    # --- Safety Guidance ---
    parser.add_argument("--guidance_mode", type=str, default="anchor_shift",
                        choices=["anchor_shift", "target_negate", "dual"],
                        help="How to apply safety guidance")
    parser.add_argument("--safety_scale", type=float, default=1.0,
                        help="Global multiplier for safety guidance strength")
    parser.add_argument("--guidance_schedule", type=str, default="constant",
                        choices=["constant", "linear_decay", "cosine"],
                        help="Schedule for safety guidance strength over timesteps")
    parser.add_argument("--warmup_steps", type=int, default=5,
                        help="Number of warmup steps before applying guidance at full strength")

    # --- Misc ---
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_spatial_maps", action="store_true",
                        help="Save spatial guidance maps as images for visualization")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)

    args = parser.parse_args()

    # Handle flag overrides
    if args.cas_no_sticky:
        args.cas_sticky = False
    if args.hard_mask:
        args.soft_mask = False

    return args


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"CAS + Spatial CFG: Training-Free Safe Generation")
    print(f"{'='*70}")
    print(f"  [WHEN] CAS threshold: {args.cas_threshold}, sticky: {args.cas_sticky}")
    print(f"         CAS window: t=[{args.cas_window_end}, {args.cas_window_start}]")
    print(f"  [WHERE] Method: {args.spatial_method}, threshold: {args.spatial_threshold}")
    print(f"          Scale: high={args.spatial_scale_high}, low={args.spatial_scale_low}")
    print(f"          Soft mask: {args.soft_mask}, blur σ={args.mask_blur_sigma}")
    print(f"  [HOW]  Mode: {args.guidance_mode}, scale: {args.safety_scale}")
    print(f"         Schedule: {args.guidance_schedule}")
    print(f"  Target: {args.target_concepts}")
    print(f"  Anchor: {args.anchor_concepts}")
    print(f"{'='*70}\n")

    # Load prompts
    all_prompts = load_prompts(args.prompt_file)
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx > 0 else len(all_prompts)
    prompts_with_idx = list(enumerate(all_prompts))[start_idx:end_idx]
    print(f"Loaded {len(all_prompts)} prompts, processing [{start_idx}:{end_idx}] = {len(prompts_with_idx)}")

    # Load SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    unet = pipe.unet
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    scheduler = pipe.scheduler

    # Pre-encode target and anchor concept embeddings (once, reuse for all prompts)
    with torch.no_grad():
        # Target concepts (combine into one)
        target_text = ", ".join(args.target_concepts)
        target_inputs = tokenizer(
            target_text, padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        target_embeds = text_encoder(target_inputs.input_ids.to(device))[0]  # (1, 77, 768)

        # Anchor concepts (combine into one)
        anchor_text = ", ".join(args.anchor_concepts)
        anchor_inputs = tokenizer(
            anchor_text, padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        anchor_embeds = text_encoder(anchor_inputs.input_ids.to(device))[0]  # (1, 77, 768)

        # Unconditional embedding (shared)
        uncond_inputs = tokenizer(
            "", padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]  # (1, 77, 768)

    print(f"[Embeddings] Target: '{target_text}'")
    print(f"[Embeddings] Anchor: '{anchor_text}'")

    # Initialize modules
    cas = ConceptAlignmentScore(
        threshold=args.cas_threshold,
        sticky=args.cas_sticky,
    )
    spatial_guidance = SpatialCFGGuidance(
        spatial_method=args.spatial_method,
        spatial_threshold=args.spatial_threshold,
        guidance_scale_high=args.spatial_scale_high,
        guidance_scale_low=args.spatial_scale_low,
        soft_mask=args.soft_mask,
        mask_blur_sigma=args.mask_blur_sigma,
        norm_method="percentile",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_spatial_maps:
        maps_dir = output_dir / "spatial_maps"
        maps_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []

    for prompt_idx, prompt in tqdm(prompts_with_idx, desc="Generating"):
        for sample_idx in range(args.nsamples):
            current_seed = args.seed + prompt_idx * args.nsamples + sample_idx
            set_seed(current_seed)

            # Reset CAS state for each sample
            cas.reset()

            guided_steps_count = 0
            skipped_steps_count = 0
            step_history = []
            cas_values = []

            # Encode prompt text
            with torch.no_grad():
                text_inputs = tokenizer(
                    prompt, padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt"
                )
                text_embeds = text_encoder(text_inputs.input_ids.to(device))[0]

            # Initialize latents
            set_seed(current_seed)
            latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
            latents = latents * scheduler.init_noise_sigma
            scheduler.set_timesteps(args.num_inference_steps, device=device)

            total_steps = len(scheduler.timesteps)

            for step_idx, t in enumerate(scheduler.timesteps):
                t_val = t.item()

                # ---- Step 1: Standard CFG forward (uncond + cond) ----
                # Batch: [uncond, cond] for standard CFG
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                with torch.no_grad():
                    noise_pred_raw = unet(
                        latent_model_input, t,
                        encoder_hidden_states=torch.cat([uncond_embeds, text_embeds])
                    ).sample

                noise_pred_uncond, noise_pred_text = noise_pred_raw.chunk(2)

                # Standard CFG
                noise_pred_cfg = noise_pred_uncond + args.cfg_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # ---- Step 2: Check CAS (WHEN) ----
                in_window = (args.cas_window_end <= t_val <= args.cas_window_start)
                should_guide = False
                cas_value = None

                if in_window:
                    # Need target noise prediction for CAS
                    latent_single = scheduler.scale_model_input(latents, t)

                    with torch.no_grad():
                        noise_pred_target = unet(
                            latent_single, t,
                            encoder_hidden_states=target_embeds
                        ).sample

                    cas_value, should_guide = cas.compute(
                        noise_pred_text, noise_pred_uncond, noise_pred_target
                    )
                    cas_values.append(cas_value)

                info = {
                    "step": step_idx,
                    "timestep": t_val,
                    "in_window": in_window,
                    "cas_value": round(cas_value, 6) if cas_value is not None else None,
                    "should_guide": should_guide,
                }

                # ---- Step 3: Spatial CFG + Safety Guidance (WHERE + HOW) ----
                if should_guide:
                    # We already have noise_pred_target from CAS computation
                    # Need anchor noise prediction
                    with torch.no_grad():
                        noise_pred_anchor = unet(
                            latent_single, t,
                            encoder_hidden_states=anchor_embeds
                        ).sample

                    # Compute spatial map (WHERE)
                    spatial_map = spatial_guidance.compute_spatial_map(
                        noise_pred_uncond=noise_pred_uncond,
                        noise_pred_target=noise_pred_target,
                        noise_pred_anchor=noise_pred_anchor,
                        noise_pred_text=noise_pred_text,
                    )

                    # Compute guidance weight map
                    weight_map = spatial_guidance.compute_guidance_weight_map(
                        spatial_map,
                        adaptive_area_scale=args.adaptive_area_scale,
                    )

                    # Compute guidance schedule factor
                    if args.guidance_schedule == "constant":
                        schedule_factor = 1.0
                    elif args.guidance_schedule == "linear_decay":
                        progress = step_idx / max(total_steps - 1, 1)
                        schedule_factor = 1.0 - progress
                    elif args.guidance_schedule == "cosine":
                        progress = step_idx / max(total_steps - 1, 1)
                        schedule_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                    else:
                        schedule_factor = 1.0

                    # Warmup: ramp up guidance in first few steps
                    if guided_steps_count < args.warmup_steps:
                        warmup_factor = (guided_steps_count + 1) / args.warmup_steps
                        schedule_factor *= warmup_factor

                    # Apply safety guidance (HOW)
                    noise_pred_safe = apply_safety_guidance(
                        noise_pred_cfg=noise_pred_cfg,
                        noise_pred_uncond=noise_pred_uncond,
                        noise_pred_target=noise_pred_target,
                        noise_pred_anchor=noise_pred_anchor,
                        weight_map=weight_map,
                        guidance_mode=args.guidance_mode,
                        global_scale=args.safety_scale * schedule_factor,
                    )

                    # Use the safe noise prediction for this step
                    noise_pred_cfg = noise_pred_safe

                    guided_steps_count += 1
                    info["guided"] = True
                    info["spatial_area"] = float((spatial_map >= args.spatial_threshold).float().mean().item())
                    info["weight_map_mean"] = float(weight_map.mean().item())
                    info["schedule_factor"] = round(schedule_factor, 4)

                    # Save spatial map visualization
                    if args.save_spatial_maps and step_idx % 10 == 0:
                        map_np = spatial_map[0, 0].float().cpu().numpy()
                        map_np = np.nan_to_num(map_np, nan=0.0)
                        map_img = (np.clip(map_np, 0, 1) * 255).astype(np.uint8)
                        map_pil = Image.fromarray(map_img, mode='L')
                        map_path = maps_dir / f"{prompt_idx:04d}_{sample_idx:02d}_step{step_idx:03d}.png"
                        map_pil.save(str(map_path))
                else:
                    skipped_steps_count += 1
                    info["guided"] = False

                # ---- Step 4: DDIM step ----
                latents = scheduler.step(noise_pred_cfg, t, latents).prev_sample

                step_history.append(info)

                if args.debug and step_idx % 5 == 0:
                    cas_str = f"CAS={cas_value:.4f}" if cas_value is not None else "CAS=N/A"
                    status = "GUIDED" if info.get("guided", False) else "skip"
                    area_str = f" area={info.get('spatial_area', 0):.3f}" if info.get("guided") else ""
                    print(f"  Step {step_idx}: t={t_val}, {cas_str}, {status}{area_str}")

            # ---- Decode ----
            with torch.no_grad():
                latents_dec = 1.0 / vae.config.scaling_factor * latents
                image = vae.decode(latents_dec.to(vae.dtype)).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()
                image = (image[0] * 255).round().astype(np.uint8)

            safe_prompt = "".join(
                c if c.isalnum() or c in ' -_' else '_'
                for c in prompt
            )[:50].replace(' ', '_')
            img_filename = f"{prompt_idx:04d}_{sample_idx:02d}_{safe_prompt}.png"
            save_image(image, output_dir / img_filename)

            total_steps = guided_steps_count + skipped_steps_count

            img_stats = {
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "seed": current_seed,
                "prompt": prompt[:100],
                "filename": img_filename,
                "guided_steps": guided_steps_count,
                "skipped_steps": skipped_steps_count,
                "total_steps": total_steps,
                "guidance_ratio": guided_steps_count / max(total_steps, 1),
                "cas_triggered": cas.triggered,
                "avg_cas": float(np.mean(cas_values)) if cas_values else 0.0,
                "max_cas": float(np.max(cas_values)) if cas_values else 0.0,
                "min_cas": float(np.min(cas_values)) if cas_values else 0.0,
            }
            if args.debug:
                img_stats["step_history"] = step_history
            all_stats.append(img_stats)

            print(
                f"  [{prompt_idx:03d}] Guided: {guided_steps_count}/{total_steps} "
                f"({img_stats['guidance_ratio']*100:.1f}%) "
                f"CAS avg={img_stats['avg_cas']:.4f} max={img_stats['max_cas']:.4f} "
                f"triggered={cas.triggered}"
            )

    # Summary
    total_images = len(all_stats)
    avg_guided = np.mean([s["guided_steps"] for s in all_stats]) if all_stats else 0
    avg_ratio = np.mean([s["guidance_ratio"] for s in all_stats]) if all_stats else 0
    n_triggered = sum(1 for s in all_stats if s["guided_steps"] > 0)
    avg_cas_all = np.mean([s["avg_cas"] for s in all_stats if s["avg_cas"] > 0]) if all_stats else 0

    summary = {
        "method": "CAS + Spatial CFG",
        "args": vars(args),
        "overall": {
            "total_images": total_images,
            "avg_guided_steps": float(avg_guided),
            "avg_guidance_ratio": float(avg_ratio),
            "triggered_count": n_triggered,
            "no_guidance_count": total_images - n_triggered,
            "avg_cas_score": float(avg_cas_all),
        },
        "per_image_stats": all_stats,
    }
    with open(output_dir / "generation_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"CAS + SPATIAL CFG COMPLETE!")
    print(f"{'='*70}")
    print(f"Output: {output_dir}")
    print(f"Total images: {total_images}")
    print(f"Triggered: {n_triggered}/{total_images} ({100*n_triggered/max(total_images,1):.1f}%)")
    print(f"Avg guided steps: {avg_guided:.1f}/{args.num_inference_steps}")
    print(f"Avg CAS (triggered): {avg_cas_all:.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
