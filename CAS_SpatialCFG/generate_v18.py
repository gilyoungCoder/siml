#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Timestep-Adaptive Mask Sharpening v18: Training-Free Safe Generation

Architecture:
  WHEN (Global CAS): cos(d_prompt, d_target) > tau, with optional adaptive threshold
  WHERE (Hybrid Fusion): cross-attention probe + noise-based spatial CAS
  HOW: hybrid/sld/dag_adaptive with timestep-adaptive safety_scale

Key innovation (v18):
  Dynamically adjust spatial mask aggressiveness and guidance strength based on
  the denoising timestep. Early steps (layout phase) use broad + strong guidance
  to prevent nudity layout from forming. Late steps (detail phase) use focused +
  gentle guidance to preserve fine details.

  Inspired by:
  - SAFREE's f_beta scheduling (binary on/off per timestep)
  - DAG's observation that hard-to-erase nudity collapses in the first few steps

  Our approach is more fine-grained: continuously adapt guidance parameters
  instead of a binary on/off switch.

Schedule types:
  - linear:  Linear interpolation from early (aggressive) to late (gentle)
  - cosine:  Cosine annealing (smoother S-curve transition)
  - step:    Binary switch at warmup_frac (aggressive before, gentle after)
  - none:    Fixed params throughout (same as v14, for ablation)

Base features (from v14 = v6 crossattn + v7 noise CAS + v12 hybrid_mask):
  - Cross-attention probe spatial mask (v6)
  - Noise-based spatial CAS mask (v7)
  - Hybrid mask fusion with configurable weight (v12)
  - Exemplar concept directions support (v7)
  - Multiple guidance modes: hybrid, sld, dag_adaptive, anchor_inpaint, projection
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

from attention_probe import (
    AttentionProbeStore,
    precompute_target_keys,
    register_attention_probe,
    register_dual_attention_probe,
    restore_processors,
    compute_attention_spatial_mask,
    find_token_indices,
)
from concept_pack_loader import load_concept_pack, load_multiple_packs, get_combined_target_words


# =============================================================================
# Timestep-Adaptive Schedule Functions
# =============================================================================
def schedule_linear(frac: float) -> float:
    """Linear interpolation: 1.0 at start -> 0.0 at end."""
    return 1.0 - frac


def schedule_cosine(frac: float) -> float:
    """Cosine annealing: smooth S-curve from 1.0 -> 0.0."""
    return 0.5 * (1.0 + math.cos(math.pi * frac))


def schedule_step(frac: float, warmup_frac: float = 0.3) -> float:
    """Binary step: 1.0 before warmup_frac, 0.0 after."""
    return 1.0 if frac < warmup_frac else 0.0


SCHEDULE_FNS = {
    "linear": schedule_linear,
    "cosine": schedule_cosine,
    "step": schedule_step,
}


def get_adaptive_params(
    step_idx: int,
    total_steps: int,
    base_safety_scale: float,
    base_threshold: float,
    base_alpha: float,
    base_cas_threshold: float,
    schedule_type: str = "linear",
    scale_boost: float = 1.0,
    threshold_range: Tuple[float, float] = (0.5, 1.0),
    alpha_range: Tuple[float, float] = (0.5, 1.0),
    warmup_frac: float = 0.3,
    adaptive_cas: bool = False,
    cas_early_factor: float = 0.7,
) -> Dict[str, float]:
    """
    Compute timestep-adaptive guidance parameters.

    Early denoising (layout phase): strong + broad guidance to prevent nudity.
    Late denoising (detail phase): gentle + focused guidance to preserve quality.

    Args:
        step_idx:           Current step index (0 = start of denoising)
        total_steps:        Total number of denoising steps
        base_safety_scale:  Base safety guidance scale
        base_threshold:     Base spatial mask threshold
        base_alpha:         Base sigmoid sharpness alpha
        base_cas_threshold: Base global CAS threshold
        schedule_type:      "linear", "cosine", "step", or "none"
        scale_boost:        How much stronger early guidance is (multiplicative)
        threshold_range:    (early_frac, late_frac) of base_threshold
        alpha_range:        (early_frac, late_frac) of base_alpha
        warmup_frac:        Fraction for step schedule binary switch
        adaptive_cas:       Whether to also adapt CAS threshold
        cas_early_factor:   CAS threshold multiplier for early steps (lower = more sensitive)

    Returns:
        Dict with adaptive parameter values for this timestep.
    """
    if schedule_type == "none" or total_steps <= 1:
        return {
            "safety_scale": base_safety_scale,
            "spatial_threshold": base_threshold,
            "sigmoid_alpha": base_alpha,
            "cas_threshold": base_cas_threshold,
        }

    frac = step_idx / max(total_steps - 1, 1)  # 0.0 (start) -> 1.0 (end)

    # Get schedule value: 1.0 at start (aggressive) -> 0.0 at end (gentle)
    if schedule_type == "step":
        s = schedule_step(frac, warmup_frac)
    else:
        s = SCHEDULE_FNS[schedule_type](frac)

    # Safety scale: boosted early, base late
    # s=1.0 -> base * (1 + scale_boost), s=0.0 -> base
    safety_scale = base_safety_scale * (1.0 + s * scale_boost)

    # Spatial threshold: low early (broad mask), high late (focused mask)
    # s=1.0 -> threshold_range[0] * base, s=0.0 -> threshold_range[1] * base
    t_lo, t_hi = threshold_range
    threshold_factor = t_lo + (1.0 - s) * (t_hi - t_lo)
    spatial_threshold = base_threshold * threshold_factor

    # Sigmoid alpha: soft early (gradual transition), sharp late (crisp mask)
    # s=1.0 -> alpha_range[0] * base, s=0.0 -> alpha_range[1] * base
    a_lo, a_hi = alpha_range
    alpha_factor = a_lo + (1.0 - s) * (a_hi - a_lo)
    sigmoid_alpha = base_alpha * alpha_factor

    # CAS threshold: lower early (more sensitive), standard late
    if adaptive_cas:
        cas_factor = cas_early_factor + (1.0 - s) * (1.0 - cas_early_factor)
        cas_threshold = base_cas_threshold * cas_factor
    else:
        cas_threshold = base_cas_threshold

    return {
        "safety_scale": safety_scale,
        "spatial_threshold": spatial_threshold,
        "sigmoid_alpha": sigmoid_alpha,
        "cas_threshold": cas_threshold,
    }


# =============================================================================
# Global CAS: Concept Alignment Score (WHEN)
# =============================================================================
class GlobalCAS:
    """Detect harmful prompt alignment via cosine(d_prompt, d_target)."""

    def __init__(self, threshold: float = 0.6, sticky: bool = True):
        self.threshold = threshold
        self.sticky = sticky
        self.triggered = False

    def reset(self):
        self.triggered = False

    def compute(self, eps_prompt, eps_null, eps_target=None, d_target_global=None,
                threshold_override=None):
        """
        Compute global CAS from noise predictions.

        Args:
            eps_prompt:         Prompt-conditioned noise prediction [1, 4, 64, 64]
            eps_null:           Unconditional noise prediction [1, 4, 64, 64]
            eps_target:         Target concept noise prediction (online mode)
            d_target_global:    Pre-computed target global direction (exemplar mode)
            threshold_override: Per-step threshold (for adaptive CAS)

        Returns:
            cas_value: float, cosine similarity
            should_guide: bool
        """
        d_prompt = (eps_prompt - eps_null).reshape(1, -1).float()

        if d_target_global is not None:
            d_target = d_target_global.unsqueeze(0).float() if d_target_global.dim() == 1 else d_target_global.float()
        elif eps_target is not None:
            d_target = (eps_target - eps_null).reshape(1, -1).float()
        else:
            raise ValueError("Either eps_target or d_target_global must be provided")

        cas = F.cosine_similarity(d_prompt, d_target, dim=-1).item()

        if math.isnan(cas) or math.isinf(cas):
            return 0.0, self.triggered if self.sticky else False

        thresh = threshold_override if threshold_override is not None else self.threshold

        if self.sticky and self.triggered:
            return cas, True

        if cas > thresh:
            if self.sticky:
                self.triggered = True
            return cas, True

        return cas, False


# =============================================================================
# Spatial CAS: Per-Pixel Concept Alignment (WHERE - noise mode)
# =============================================================================
def compute_spatial_cas(
    eps_prompt: torch.Tensor,
    eps_null: torch.Tensor,
    eps_target: torch.Tensor,
    neighborhood_size: int = 3,
) -> torch.Tensor:
    """
    Compute per-pixel Spatial CAS using neighborhood pooling.

    Returns:
        spatial_cas: per-pixel CAS map [H, W] with values in [-1, 1]
    """
    d_prompt = (eps_prompt - eps_null).float()
    d_target = (eps_target - eps_null).float()

    H, W = d_prompt.shape[2], d_prompt.shape[3]
    pad = neighborhood_size // 2

    d_prompt_unfolded = F.unfold(d_prompt, kernel_size=neighborhood_size, padding=pad)
    d_target_unfolded = F.unfold(d_target, kernel_size=neighborhood_size, padding=pad)

    spatial_cas = F.cosine_similarity(d_prompt_unfolded, d_target_unfolded, dim=1)
    spatial_cas = spatial_cas.reshape(H, W)

    return spatial_cas


def compute_spatial_cas_with_dir(
    d_prompt: torch.Tensor,
    d_target: torch.Tensor,
    neighborhood_size: int = 3,
) -> torch.Tensor:
    """Compute per-pixel Spatial CAS using pre-computed directions (exemplar mode)."""
    d_prompt_f = d_prompt.float()
    d_target_f = d_target.float()

    H, W = d_prompt_f.shape[2], d_prompt_f.shape[3]
    pad = neighborhood_size // 2

    d_prompt_unfolded = F.unfold(d_prompt_f, kernel_size=neighborhood_size, padding=pad)
    d_target_unfolded = F.unfold(d_target_f, kernel_size=neighborhood_size, padding=pad)

    spatial_cas = F.cosine_similarity(d_prompt_unfolded, d_target_unfolded, dim=1)
    spatial_cas = spatial_cas.reshape(H, W)

    return spatial_cas


def compute_soft_mask(
    spatial_cas: torch.Tensor,
    spatial_threshold: float = 0.3,
    sigmoid_alpha: float = 10.0,
    blur_sigma: float = 1.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Convert spatial CAS map to a soft guidance mask using sigmoid + Gaussian blur.

    Returns:
        soft_mask: [1, 1, H, W] smooth mask in [0, 1]
    """
    soft_mask = torch.sigmoid(sigmoid_alpha * (spatial_cas - spatial_threshold))
    soft_mask = soft_mask.unsqueeze(0).unsqueeze(0)

    if blur_sigma > 0:
        soft_mask = gaussian_blur_2d(soft_mask, kernel_size=5, sigma=blur_sigma)

    return soft_mask.clamp(0, 1)


def gaussian_blur_2d(
    x: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Apply Gaussian blur to a 2D tensor."""
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()

    kernel_h = g.view(1, 1, kernel_size, 1)
    kernel_w = g.view(1, 1, 1, kernel_size)
    pad_h = kernel_size // 2
    pad_w = kernel_size // 2

    x = F.pad(x, [0, 0, pad_h, pad_h], mode='reflect')
    x = F.conv2d(x, kernel_h.expand(x.shape[1], -1, -1, -1), groups=x.shape[1])
    x = F.pad(x, [pad_w, pad_w, 0, 0], mode='reflect')
    x = F.conv2d(x, kernel_w.expand(x.shape[1], -1, -1, -1), groups=x.shape[1])

    return x


def compute_crossattn_soft_mask(
    probe_store: AttentionProbeStore,
    token_indices: Optional[List[int]],
    attn_threshold: float = 0.3,
    sigmoid_alpha: float = 10.0,
    blur_sigma: float = 1.0,
    target_resolution: int = 64,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Convert cross-attention probe maps into a soft spatial mask.

    Returns:
        soft_mask: [1, 1, H, W] in [0, 1]
    """
    attn_map = compute_attention_spatial_mask(
        probe_store,
        token_indices=token_indices,
        target_resolution=target_resolution,
    )

    if device is not None:
        attn_map = attn_map.to(device)

    soft_mask = torch.sigmoid(sigmoid_alpha * (attn_map - attn_threshold))
    soft_mask = soft_mask.unsqueeze(0).unsqueeze(0)

    if blur_sigma > 0:
        soft_mask = gaussian_blur_2d(soft_mask, kernel_size=5, sigma=blur_sigma)

    return soft_mask.clamp(0, 1)


# =============================================================================
# Guidance Application (HOW)
# =============================================================================
def apply_guidance(
    eps_cfg: torch.Tensor,
    eps_null: torch.Tensor,
    eps_prompt: torch.Tensor,
    eps_target: torch.Tensor,
    eps_anchor: torch.Tensor,
    soft_mask: torch.Tensor,
    guide_mode: str = "hybrid",
    safety_scale: float = 1.0,
    cfg_scale: float = 7.5,
    **kwargs,
) -> torch.Tensor:
    """
    Apply spatially-masked guidance to remove unsafe content.

    Modes:
        hybrid:          Subtract target + add anchor direction.
        sld:             Subtract target direction (SLD-style).
        dag_adaptive:    DAG-style area+magnitude scaling.
        anchor_inpaint:  Blend original CFG with anchor CFG in masked regions.
        projection:      Project out nudity component from prompt direction.
        hybrid_proj:     Projection + anchor blending.
    """
    mask = soft_mask.to(eps_cfg.dtype)

    if guide_mode == "hybrid":
        t_scale = kwargs.get("target_scale", safety_scale)
        a_scale = kwargs.get("anchor_scale", safety_scale)
        eps_final = eps_cfg \
                    - t_scale * mask * (eps_target - eps_null) \
                    + a_scale * mask * (eps_anchor - eps_null)

    elif guide_mode == "sld":
        eps_safe_direction = eps_target - eps_null
        eps_final = eps_cfg - safety_scale * mask * eps_safe_direction

    elif guide_mode == "dag_adaptive":
        area = mask.sum() / mask.numel()
        area_scale = 5.0 / (mask.shape[-1] * mask.shape[-2])
        area_factor = area_scale * area * mask.numel()
        mag_scale = 1.0 + 4.0 * soft_mask.to(eps_cfg.dtype)
        d_target = eps_target - eps_null
        correction = safety_scale * area_factor * mag_scale * mask * d_target
        eps_final = eps_cfg - correction

    elif guide_mode == "anchor_inpaint":
        eps_anchor_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
        eps_final = eps_cfg * (1.0 - safety_scale * mask) + eps_anchor_cfg * (safety_scale * mask)

    elif guide_mode == "projection":
        d_prompt = (eps_prompt - eps_null).float()
        d_target = (eps_target - eps_null).float()
        dot = (d_prompt * d_target).sum(dim=1, keepdim=True)
        norm_sq = (d_target * d_target).sum(dim=1, keepdim=True).clamp(min=1e-8)
        proj = (dot / norm_sq) * d_target
        d_safe = d_prompt - safety_scale * proj
        eps_safe_cfg = (eps_null.float() + cfg_scale * d_safe).to(eps_cfg.dtype)
        eps_final = eps_cfg * (1.0 - mask) + eps_safe_cfg * mask

    elif guide_mode == "hybrid_proj":
        d_prompt = (eps_prompt - eps_null).float()
        d_target = (eps_target - eps_null).float()
        dot = (d_prompt * d_target).sum(dim=1, keepdim=True)
        norm_sq = (d_target * d_target).sum(dim=1, keepdim=True).clamp(min=1e-8)
        proj = (dot / norm_sq) * d_target
        p_scale = kwargs.get("proj_scale", 1.0)
        d_safe = d_prompt - p_scale * proj
        eps_safe_cfg = (eps_null.float() + cfg_scale * d_safe).to(eps_cfg.dtype)
        a_scale = kwargs.get("anchor_scale", safety_scale)
        eps_anchor_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
        eps_final = eps_safe_cfg * (1.0 - a_scale * mask) + eps_anchor_cfg * (a_scale * mask)

    else:
        raise ValueError(f"Unknown guide_mode: {guide_mode}")

    if torch.isnan(eps_final).any() or torch.isinf(eps_final).any():
        eps_final = torch.where(torch.isfinite(eps_final), eps_final, eps_cfg)

    return eps_final


# =============================================================================
# Utils
# =============================================================================
def load_prompts(filepath):
    """Load prompts from CSV or TXT file."""
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
    """Encode a list of concept strings and average their embeddings."""
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
    return torch.stack(all_embeds).mean(dim=0)


# =============================================================================
# Argument Parser
# =============================================================================
def parse_args():
    p = ArgumentParser(description="Timestep-Adaptive Mask Sharpening v18")
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

    # WHERE mode — hybrid fusion of crossattn + noise
    p.add_argument("--where_mode", type=str, default="hybrid",
                    choices=["crossattn", "noise", "hybrid"],
                    help="Spatial detection: crossattn (v6), noise (v7), "
                         "hybrid (weighted fusion of both)")
    p.add_argument("--hybrid_mask_weight", type=float, default=0.5,
                    help="Weight for cross-attention mask in hybrid mode (0=noise only, 1=attn only)")
    p.add_argument("--attn_resolutions", type=int, nargs="+", default=[16, 32],
                    help="Cross-attention resolutions to probe")
    p.add_argument("--attn_threshold", type=float, default=0.3,
                    help="Cross-attention mask threshold (sigmoid center)")
    p.add_argument("--target_words", type=str, nargs="+",
                    default=["nude", "naked", "nudity", "nsfw", "bare", "body"],
                    help="Keywords for target token indices in attention probing")

    # Probe source (orthogonal to where_mode; only affects crossattn/hybrid WHERE modes)
    p.add_argument("--probe_source", type=str, default="text",
                    choices=["text", "image", "both"],
                    help="Key source for cross-attention probe: "
                         "text (text keyword probe, default/v14 behavior), "
                         "image (CLIP image exemplar probe), "
                         "both (union: max of text and image masks)")
    p.add_argument("--clip_embeddings", type=str, default=None,
                    help="Path to pre-computed CLIP exemplar embeddings (.pt), "
                         "required when --probe_source is 'image' or 'both'")

    # Spatial CAS / Soft Mask (Where)
    p.add_argument("--spatial_threshold", type=float, default=0.3,
                    help="Per-pixel CAS threshold for noise mode (sigmoid center)")
    p.add_argument("--sigmoid_alpha", type=float, default=10.0,
                    help="Sharpness of sigmoid mask transition")
    p.add_argument("--neighborhood_size", type=int, default=3,
                    help="Pooling kernel size for noise-based spatial CAS")
    p.add_argument("--blur_sigma", type=float, default=1.0,
                    help="Gaussian blur sigma for mask smoothing (0=no blur)")

    # Guidance (How)
    p.add_argument("--guide_mode", type=str, default="hybrid",
                    choices=["hybrid", "sld", "dag_adaptive", "anchor_inpaint",
                             "projection", "hybrid_proj"],
                    help="Guidance mode")
    p.add_argument("--safety_scale", type=float, default=1.0,
                    help="Base guidance strength")
    p.add_argument("--target_scale", type=float, default=-1.0,
                    help="Hybrid: target repulsion scale (default: same as safety_scale)")
    p.add_argument("--anchor_scale", type=float, default=-1.0,
                    help="Hybrid: anchor attraction scale (default: same as safety_scale)")
    p.add_argument("--proj_scale", type=float, default=1.0,
                    help="hybrid_proj: projection strength for nudity removal")
    p.add_argument("--guide_start_frac", type=float, default=0.0,
                    help="Fraction of total steps before guidance starts (0=all steps)")

    # v18: Timestep-adaptive schedule parameters
    p.add_argument("--schedule_type", type=str, default="cosine",
                    choices=["linear", "cosine", "step", "none"],
                    help="Adaptive schedule: linear, cosine, step (binary), none (fixed/ablation)")
    p.add_argument("--scale_boost", type=float, default=1.0,
                    help="How much stronger early guidance is (multiplicative on base safety_scale)")
    p.add_argument("--threshold_range_lo", type=float, default=0.5,
                    help="Fraction of base threshold at early steps (lower = broader mask)")
    p.add_argument("--threshold_range_hi", type=float, default=1.0,
                    help="Fraction of base threshold at late steps")
    p.add_argument("--alpha_range_lo", type=float, default=0.5,
                    help="Fraction of base sigmoid_alpha at early steps (lower = softer mask)")
    p.add_argument("--alpha_range_hi", type=float, default=1.0,
                    help="Fraction of base sigmoid_alpha at late steps")
    p.add_argument("--warmup_frac", type=float, default=0.3,
                    help="Fraction for step schedule binary switch point")
    p.add_argument("--adaptive_cas", action="store_true",
                    help="Also adapt CAS threshold over timesteps (lower early = more sensitive)")
    p.add_argument("--cas_early_factor", type=float, default=0.7,
                    help="CAS threshold multiplier for early steps (with --adaptive_cas)")

    # Concepts
    p.add_argument("--target_concepts", type=str, nargs="+",
                    default=["nudity", "nude person", "naked body"],
                    help="Target concepts to erase")
    p.add_argument("--anchor_concepts", type=str, nargs="+",
                    default=["clothed person", "person wearing clothes"],
                    help="Anchor concepts to guide toward")

    # Exemplar support (v7 compatibility)
    p.add_argument("--concept_dir_path", type=str, default=None,
                    help="Path to concept_directions.pt (for exemplar mode)")
    p.add_argument("--exemplar_mode", type=str, default="text",
                    choices=["exemplar", "text", "hybrid_exemplar"],
                    help="Direction source: exemplar, text (online), hybrid_exemplar")
    p.add_argument("--exemplar_weight", type=float, default=0.7,
                    help="Weight for exemplar directions in hybrid_exemplar mode")

    # Multi-concept
    p.add_argument("--concept_packs", type=str, nargs="+", default=None,
                    help="Concept pack directories for multi-concept erasing. "
                         "If set, overrides --target_concepts, --anchor_concepts, etc.")
    p.add_argument("--family_level", action="store_true", default=False,
                    help="Apply corrections at family level (per sub-concept) instead of concept level")

    # Misc
    p.add_argument("--save_maps", action="store_true",
                    help="Save spatial maps and masks for debugging")
    p.add_argument("--debug", action="store_true",
                    help="Print per-step debug info")
    p.add_argument("--start_idx", type=int, default=0,
                    help="Start prompt index")
    p.add_argument("--end_idx", type=int, default=-1,
                    help="End prompt index (-1 = all)")

    args = p.parse_args()
    if args.cas_no_sticky:
        args.cas_sticky = False

    # Validate (skip if using concept_packs)
    if args.concept_packs is None:
        if args.exemplar_mode in ("exemplar", "hybrid_exemplar") and args.concept_dir_path is None:
            p.error("--concept_dir_path is required for exemplar and hybrid_exemplar modes")
        if args.probe_source in ("image", "both") and args.clip_embeddings is None:
            p.error("--clip_embeddings is required when --probe_source is 'image' or 'both'")

    return args


# =============================================================================
# Main Generation Loop
# =============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_concept = args.concept_packs is not None

    threshold_range = (args.threshold_range_lo, args.threshold_range_hi)
    alpha_range = (args.alpha_range_lo, args.alpha_range_hi)

    print(f"\n{'='*70}")
    print(f"Timestep-Adaptive Mask Sharpening v18")
    print(f"{'='*70}")
    print(f"  WHEN:     Global CAS threshold={args.cas_threshold}, sticky={args.cas_sticky}"
          + (f", adaptive (early x{args.cas_early_factor})" if args.adaptive_cas else ""))
    print(f"  WHERE:    {args.where_mode}"
          + (f" (attn_weight={args.hybrid_mask_weight})" if args.where_mode == "hybrid" else "")
          + (f", probe_source={args.probe_source}" if args.where_mode != "noise" else ""))
    print(f"  HOW:      {args.guide_mode}, base safety_scale={args.safety_scale}")
    print(f"  SCHEDULE: {args.schedule_type}, scale_boost={args.scale_boost}, "
          f"warmup_frac={args.warmup_frac}")
    print(f"            threshold_range={threshold_range}, alpha_range={alpha_range}")
    print(f"  Exemplar: {args.exemplar_mode}"
          + (f" (weight={args.exemplar_weight})" if args.exemplar_mode == "hybrid_exemplar" else ""))
    print(f"  Target:   {args.target_concepts}")
    print(f"  Anchor:   {args.anchor_concepts}")
    print(f"  Model:    {args.ckpt}")
    print(f"  Steps: {args.steps}, CFG: {args.cfg_scale}, Samples/prompt: {args.nsamples}")
    print(f"{'='*70}\n")

    # ==================================================================
    # Multi-concept vs single-concept setup
    # ==================================================================
    concept_packs_list = None
    pack_cas_gates = None
    pack_target_embeds_list = None
    pack_anchor_embeds_list = None
    pack_probe_token_indices = None

    # ---- Load pre-computed concept directions (exemplar / hybrid_exemplar) ----
    target_dirs = anchor_dirs = target_global = anchor_global = None

    if not multi_concept and args.exemplar_mode in ("exemplar", "hybrid_exemplar"):
        print(f"Loading concept directions from {args.concept_dir_path} ...")
        concept_data = torch.load(args.concept_dir_path, map_location=device)
        target_dirs = concept_data['target_directions']
        anchor_dirs = concept_data['anchor_directions']
        target_global = concept_data['target_global']
        anchor_global = concept_data['anchor_global']
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
    pipe.feature_extractor = None

    unet = pipe.unet
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    scheduler = pipe.scheduler

    # ---- Pre-encode concept embeddings ----
    with torch.no_grad():
        target_embeds = encode_concepts(text_encoder, tokenizer,
                                        args.target_concepts, device)
    anchor_embeds = None
    if args.exemplar_mode in ("text", "hybrid_exemplar"):
        with torch.no_grad():
            anchor_embeds = encode_concepts(text_encoder, tokenizer,
                                            args.anchor_concepts, device)

    with torch.no_grad():
        uncond_inputs = tokenizer(
            "", padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]

    print(f"Embeddings encoded: target={len(args.target_concepts)} concepts"
          + (f", anchor={len(args.anchor_concepts)} concepts" if anchor_embeds is not None
             else ", anchor=exemplar"))

    # ---- Load CLIP exemplar embeddings (for image/both probe_source) ----
    clip_data = None
    clip_target_embeds = None
    if not multi_concept and args.probe_source in ("image", "both"):
        clip_data = torch.load(args.clip_embeddings, map_location=device)
        clip_target_embeds = clip_data["target_clip_embeds"].to(device=device, dtype=unet.dtype)
        print(f"Loaded CLIP exemplar embeddings: {clip_target_embeds.shape}")
        if "config" in clip_data:
            print(f"  CLIP config: {clip_data['config']}")

    # ---- Setup cross-attention probing (for crossattn and hybrid WHERE modes) ----
    use_crossattn = args.where_mode in ("crossattn", "hybrid")
    probe_store = None
    original_processors = None
    probe_token_indices = None
    image_probe_store = None
    image_probe_token_indices = None

    if multi_concept:
        # ---- Multi-concept: load packs ----
        print(f"\nLoading {len(args.concept_packs)} concept packs...")
        concept_packs_list = load_multiple_packs(args.concept_packs, device=device)

        pack_cas_gates = []
        pack_target_embeds_list = []
        pack_anchor_embeds_list = []
        pack_probe_token_indices = []

        for i, pack in enumerate(concept_packs_list):
            pack_threshold = pack.get("cas_threshold", args.cas_threshold)
            pack_cas_gates.append(GlobalCAS(threshold=pack_threshold, sticky=args.cas_sticky))

            pack_target_concepts = pack.get("target_concepts", args.target_concepts)
            with torch.no_grad():
                t_emb = encode_concepts(text_encoder, tokenizer, pack_target_concepts, device)
            pack_target_embeds_list.append(t_emb)

            pack_anchor_concepts = pack.get("anchor_concepts", args.anchor_concepts)
            with torch.no_grad():
                a_emb = encode_concepts(text_encoder, tokenizer, pack_anchor_concepts, device)
            pack_anchor_embeds_list.append(a_emb)

            pack_target_words = pack.get("target_words", args.target_words)
            target_text = ", ".join(pack_target_concepts)
            p_indices = find_token_indices(target_text, pack_target_words, tokenizer)
            pack_probe_token_indices.append(p_indices)

            print(f"  Pack [{i}] '{pack.get('concept', 'unknown')}': "
                  f"threshold={pack_threshold}, "
                  f"targets={pack_target_concepts[:3]}{'...' if len(pack_target_concepts) > 3 else ''}, "
                  f"probe_tokens={p_indices[:6]}{'...' if len(p_indices) > 6 else ''}")

        # Shared probe store for multi-concept
        if use_crossattn:
            probe_store = AttentionProbeStore()
            combined_target_words = get_combined_target_words(concept_packs_list)
            combined_target_concepts = []
            for pack in concept_packs_list:
                combined_target_concepts.extend(pack.get("target_concepts", []))
            if combined_target_concepts:
                with torch.no_grad():
                    combined_embeds = encode_concepts(
                        text_encoder, tokenizer, combined_target_concepts, device
                    )
                combined_keys = precompute_target_keys(
                    unet, combined_embeds.to(dtype=next(unet.parameters()).dtype),
                    args.attn_resolutions
                )
                original_processors = register_attention_probe(
                    unet, probe_store, combined_keys, args.attn_resolutions
                )
                combined_text = ", ".join(combined_target_concepts)
                probe_token_indices = find_token_indices(
                    combined_text, combined_target_words, tokenizer
                )
                print(f"  Multi-concept probe (text): combined_tokens={probe_token_indices[:8]}"
                      f"{'...' if len(probe_token_indices) > 8 else ''}")

        cas = GlobalCAS(threshold=args.cas_threshold, sticky=args.cas_sticky)

    else:
        # ---- Single-concept setup (original behavior) ----

        if use_crossattn:
            if args.probe_source == "text":
                probe_store = AttentionProbeStore()
                target_keys = precompute_target_keys(unet, target_embeds.to(dtype=next(unet.parameters()).dtype), args.attn_resolutions)
                original_processors = register_attention_probe(
                    unet, probe_store, target_keys, args.attn_resolutions
                )
                target_text = ", ".join(args.target_concepts)
                probe_token_indices = find_token_indices(target_text, args.target_words, tokenizer)
                print(f"  Text probe token indices: {probe_token_indices}")

            elif args.probe_source == "image":
                probe_store = AttentionProbeStore()
                clip_keys = precompute_target_keys(unet, clip_target_embeds, args.attn_resolutions)
                original_processors = register_attention_probe(
                    unet, probe_store, clip_keys, args.attn_resolutions
                )
                n_tokens = clip_data["config"]["n_tokens"] if (clip_data and "config" in clip_data) else 4
                probe_token_indices = list(range(1, 1 + n_tokens))
                print(f"  Image (CLIP) probe token indices: {probe_token_indices}")

            elif args.probe_source == "both":
                _text_target_keys = precompute_target_keys(unet, target_embeds.to(dtype=next(unet.parameters()).dtype), args.attn_resolutions)
                _clip_target_keys = precompute_target_keys(unet, clip_target_embeds.to(dtype=next(unet.parameters()).dtype), args.attn_resolutions)

                probe_store = AttentionProbeStore()
                image_probe_store = AttentionProbeStore()
                original_processors = register_dual_attention_probe(
                    unet, image_probe_store, probe_store,
                    _clip_target_keys, _text_target_keys, args.attn_resolutions
                )
                target_text = ", ".join(args.target_concepts)
                probe_token_indices = find_token_indices(target_text, args.target_words, tokenizer)
                print(f"  Text probe token indices: {probe_token_indices}")

                n_tokens = clip_data["config"]["n_tokens"] if (clip_data and "config" in clip_data) else 4
                image_probe_token_indices = list(range(1, 1 + n_tokens))
                print(f"  Image (CLIP) probe token indices: {image_probe_token_indices}")

        # Init CAS
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
            adaptive_log = []  # Per-step adaptive params for stats

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

                # =============================================================
                # Compute adaptive params for this timestep
                # =============================================================
                adaptive = get_adaptive_params(
                    step_idx=step_idx,
                    total_steps=total_steps,
                    base_safety_scale=args.safety_scale,
                    base_threshold=args.spatial_threshold,
                    base_alpha=args.sigmoid_alpha,
                    base_cas_threshold=args.cas_threshold,
                    schedule_type=args.schedule_type,
                    scale_boost=args.scale_boost,
                    threshold_range=threshold_range,
                    alpha_range=alpha_range,
                    warmup_frac=args.warmup_frac,
                    adaptive_cas=args.adaptive_cas,
                    cas_early_factor=args.cas_early_factor,
                )

                current_safety_scale = adaptive["safety_scale"]
                current_threshold = adaptive["spatial_threshold"]
                current_alpha = adaptive["sigmoid_alpha"]
                current_cas_threshold = adaptive["cas_threshold"]

                # =============================================================
                # Step 1: UNet forward passes
                # =============================================================
                lat_in = scheduler.scale_model_input(latents, t)
                t_int = t.item()

                # Pass 1+2: CFG (with probe active for crossattn mode)
                if use_crossattn:
                    probe_store.active = True
                    probe_store.reset()
                    if image_probe_store is not None:
                        image_probe_store.active = True
                        image_probe_store.reset()

                with torch.no_grad():
                    lat_batch = torch.cat([lat_in, lat_in])
                    embed_batch = torch.cat([uncond_embeds, prompt_embeds])
                    raw = unet(lat_batch, t, encoder_hidden_states=embed_batch).sample
                    eps_null, eps_prompt = raw.chunk(2)

                if use_crossattn:
                    probe_store.active = False
                    if image_probe_store is not None:
                        image_probe_store.active = False

                # Standard CFG
                eps_cfg = eps_null + args.cfg_scale * (eps_prompt - eps_null)

                # =============================================================
                # Step 2-4: WHEN/WHERE/HOW — Multi-concept or single-concept
                # =============================================================
                if multi_concept:
                    # ===================================================
                    # MULTI-CONCEPT: iterate over packs, accumulate corrections
                    # ===================================================
                    any_triggered = False
                    total_correction = torch.zeros_like(eps_cfg)
                    max_cas_val = 0.0

                    for pack_idx, pack in enumerate(concept_packs_list):
                        pack_cas_gate = pack_cas_gates[pack_idx]
                        p_target_embeds = pack_target_embeds_list[pack_idx]
                        p_anchor_embeds = pack_anchor_embeds_list[pack_idx]
                        p_probe_indices = pack_probe_token_indices[pack_idx]

                        p_guide_mode = pack.get("guide_mode", args.guide_mode)
                        p_safety_scale = pack.get("safety_scale", current_safety_scale)

                        # Target direction for this pack (text mode)
                        with torch.no_grad():
                            eps_target_pack = unet(lat_in, t,
                                                   encoder_hidden_states=p_target_embeds).sample

                        cas_val_pack, should_trigger_pack = pack_cas_gate.compute(
                            eps_prompt, eps_null, eps_target=eps_target_pack,
                            threshold_override=current_cas_threshold if args.adaptive_cas else None,
                        )
                        max_cas_val = max(max_cas_val, cas_val_pack)

                        in_window = step_idx >= guide_start_step
                        if not (should_trigger_pack and in_window):
                            continue

                        any_triggered = True

                        # Anchor: compute online
                        with torch.no_grad():
                            eps_anchor_pack = unet(lat_in, t,
                                                   encoder_hidden_states=p_anchor_embeds).sample

                        # WHERE: compute per-pack mask
                        if use_crossattn and probe_store is not None:
                            attn_spatial_pack = compute_attention_spatial_mask(
                                probe_store,
                                token_indices=p_probe_indices,
                                target_resolution=64,
                                resolutions_to_use=args.attn_resolutions,
                            )
                            soft_mask_pack = compute_soft_mask(
                                attn_spatial_pack,
                                spatial_threshold=current_threshold,
                                sigmoid_alpha=current_alpha,
                                blur_sigma=args.blur_sigma,
                                device=device,
                            )
                        else:
                            # Noise-based CAS mask
                            spatial_cas_map = compute_spatial_cas(
                                eps_prompt, eps_null, eps_target_pack,
                                neighborhood_size=args.neighborhood_size,
                            )
                            soft_mask_pack = compute_soft_mask(
                                spatial_cas_map,
                                spatial_threshold=current_threshold,
                                sigmoid_alpha=current_alpha,
                                blur_sigma=args.blur_sigma,
                                device=device,
                            )

                        # HOW: compute per-pack correction
                        t_scale_val = args.target_scale if args.target_scale > 0 else p_safety_scale
                        a_scale_val = args.anchor_scale if args.anchor_scale > 0 else p_safety_scale

                        eps_guided_pack = apply_guidance(
                            eps_cfg=eps_cfg,
                            eps_null=eps_null,
                            eps_prompt=eps_prompt,
                            eps_target=eps_target_pack,
                            eps_anchor=eps_anchor_pack,
                            soft_mask=soft_mask_pack,
                            guide_mode=p_guide_mode,
                            safety_scale=p_safety_scale,
                            cfg_scale=args.cfg_scale,
                            target_scale=t_scale_val,
                            anchor_scale=a_scale_val,
                            proj_scale=args.proj_scale,
                        )
                        pack_correction = eps_cfg - eps_guided_pack
                        total_correction = total_correction + pack_correction

                        mask_areas.append(float(soft_mask_pack.mean().item()))

                    cas_val = max_cas_val
                    cas_values.append(cas_val)

                    if any_triggered:
                        eps_final = eps_cfg - total_correction
                        if torch.isnan(eps_final).any() or torch.isinf(eps_final).any():
                            eps_final = torch.where(torch.isfinite(eps_final), eps_final, eps_cfg)
                        guided_count += 1

                        adaptive_log.append({
                            "step": step_idx,
                            "safety_scale_t": round(current_safety_scale, 4),
                            "threshold_t": round(current_threshold, 4),
                            "alpha_t": round(current_alpha, 4),
                            "cas_threshold_t": round(current_cas_threshold, 4),
                            "mask_area": round(float(mask_areas[-1]), 4) if mask_areas else 0.0,
                        })
                    else:
                        eps_final = eps_cfg

                else:
                    # ===================================================
                    # SINGLE-CONCEPT: original v18 behavior
                    # ===================================================
                    if args.exemplar_mode == "exemplar":
                        with torch.no_grad():
                            eps_target_online = unet(lat_in, t,
                                                     encoder_hidden_states=target_embeds).sample
                        eps_target_use = eps_target_online
                        d_anchor = anchor_dirs[t_int].to(device, dtype=torch.float16).unsqueeze(0)
                        eps_anchor_use = eps_null + d_anchor

                        cas_val, should_trigger = cas.compute(
                            eps_prompt, eps_null, eps_target=eps_target_online,
                            threshold_override=current_cas_threshold if args.adaptive_cas else None,
                        )

                    elif args.exemplar_mode == "text":
                        with torch.no_grad():
                            eps_target_online = unet(lat_in, t,
                                                     encoder_hidden_states=target_embeds).sample
                        eps_target_use = eps_target_online

                        cas_val, should_trigger = cas.compute(
                            eps_prompt, eps_null, eps_target=eps_target_online,
                            threshold_override=current_cas_threshold if args.adaptive_cas else None,
                        )

                    elif args.exemplar_mode == "hybrid_exemplar":
                        d_target_exemplar = target_dirs[t_int].to(device, dtype=torch.float16).unsqueeze(0)
                        d_anchor_exemplar = anchor_dirs[t_int].to(device, dtype=torch.float16).unsqueeze(0)
                        d_target_global_vec = target_global[t_int].to(device)

                        with torch.no_grad():
                            eps_target_online = unet(lat_in, t,
                                                     encoder_hidden_states=target_embeds).sample
                        d_target_online = eps_target_online - eps_null

                        w = args.exemplar_weight
                        d_target_blended = w * d_target_exemplar + (1 - w) * d_target_online
                        eps_target_use = eps_null + d_target_blended

                        d_target_online_global = (eps_target_online - eps_null).reshape(1, -1).float()
                        d_target_exemplar_global = d_target_global_vec.unsqueeze(0).float()
                        d_target_blended_global = w * d_target_exemplar_global + (1 - w) * d_target_online_global
                        cas_val, should_trigger = cas.compute(
                            eps_prompt, eps_null,
                            d_target_global=d_target_blended_global.squeeze(0),
                            threshold_override=current_cas_threshold if args.adaptive_cas else None,
                        )

                    cas_values.append(cas_val)

                    in_window = step_idx >= guide_start_step
                    should_guide = should_trigger and in_window

                    if should_guide:
                        # =========================================================
                        # Get anchor (mode-dependent)
                        # =========================================================
                        if args.exemplar_mode == "exemplar":
                            pass  # eps_anchor_use already set above

                        elif args.exemplar_mode == "text":
                            with torch.no_grad():
                                eps_anchor_online = unet(lat_in, t,
                                                         encoder_hidden_states=anchor_embeds).sample
                            eps_anchor_use = eps_anchor_online

                        elif args.exemplar_mode == "hybrid_exemplar":
                            with torch.no_grad():
                                eps_anchor_online = unet(lat_in, t,
                                                         encoder_hidden_states=anchor_embeds).sample
                            d_anchor_online = eps_anchor_online - eps_null
                            w = args.exemplar_weight
                            d_anchor_blended = w * d_anchor_exemplar + (1 - w) * d_anchor_online
                            eps_anchor_use = eps_null + d_anchor_blended

                        # =========================================================
                        # Step 3: WHERE — compute spatial mask with adaptive params
                        # =========================================================
                        if args.where_mode == "crossattn":
                            if args.probe_source == "both":
                                text_mask = compute_crossattn_soft_mask(
                                    probe_store, probe_token_indices,
                                    attn_threshold=current_threshold,
                                    sigmoid_alpha=current_alpha,
                                    blur_sigma=args.blur_sigma,
                                    target_resolution=64,
                                    device=device,
                                )
                                img_mask = compute_crossattn_soft_mask(
                                    image_probe_store, image_probe_token_indices,
                                    attn_threshold=current_threshold,
                                    sigmoid_alpha=current_alpha,
                                    blur_sigma=args.blur_sigma,
                                    target_resolution=64,
                                    device=device,
                                )
                                soft_mask = torch.max(text_mask, img_mask)
                            else:
                                soft_mask = compute_crossattn_soft_mask(
                                    probe_store, probe_token_indices,
                                    attn_threshold=current_threshold,
                                    sigmoid_alpha=current_alpha,
                                    blur_sigma=args.blur_sigma,
                                    target_resolution=64,
                                    device=device,
                                )

                        elif args.where_mode == "noise":
                            if args.exemplar_mode == "hybrid_exemplar":
                                d_prompt_spatial = (eps_prompt - eps_null).float()
                                spatial_cas_map = compute_spatial_cas_with_dir(
                                    d_prompt_spatial, d_target_blended.float(),
                                    neighborhood_size=args.neighborhood_size,
                                )
                            else:
                                spatial_cas_map = compute_spatial_cas(
                                    eps_prompt, eps_null, eps_target_use,
                                    neighborhood_size=args.neighborhood_size,
                                )
                            soft_mask = compute_soft_mask(
                                spatial_cas_map,
                                spatial_threshold=current_threshold,
                                sigmoid_alpha=current_alpha,
                                blur_sigma=args.blur_sigma,
                                device=device,
                            )

                        elif args.where_mode == "hybrid":
                            if args.probe_source == "both":
                                text_attn_mask = compute_crossattn_soft_mask(
                                    probe_store, probe_token_indices,
                                    attn_threshold=current_threshold,
                                    sigmoid_alpha=current_alpha,
                                    blur_sigma=args.blur_sigma,
                                    target_resolution=64,
                                    device=device,
                                )
                                img_attn_mask = compute_crossattn_soft_mask(
                                    image_probe_store, image_probe_token_indices,
                                    attn_threshold=current_threshold,
                                    sigmoid_alpha=current_alpha,
                                    blur_sigma=args.blur_sigma,
                                    target_resolution=64,
                                    device=device,
                                )
                                attn_mask = torch.max(text_attn_mask, img_attn_mask)
                            else:
                                attn_mask = compute_crossattn_soft_mask(
                                    probe_store, probe_token_indices,
                                    attn_threshold=current_threshold,
                                    sigmoid_alpha=current_alpha,
                                    blur_sigma=args.blur_sigma,
                                    target_resolution=64,
                                    device=device,
                                )

                            # Noise-based spatial CAS mask
                            if args.exemplar_mode == "hybrid_exemplar":
                                d_prompt_spatial = (eps_prompt - eps_null).float()
                                spatial_cas_map = compute_spatial_cas_with_dir(
                                    d_prompt_spatial, d_target_blended.float(),
                                    neighborhood_size=args.neighborhood_size,
                                )
                            else:
                                spatial_cas_map = compute_spatial_cas(
                                    eps_prompt, eps_null, eps_target_use,
                                    neighborhood_size=args.neighborhood_size,
                                )
                            noise_mask = compute_soft_mask(
                                spatial_cas_map,
                                spatial_threshold=current_threshold,
                                sigmoid_alpha=current_alpha,
                                blur_sigma=args.blur_sigma,
                                device=device,
                            )

                            # Weighted fusion
                            w_attn = args.hybrid_mask_weight
                            soft_mask = w_attn * attn_mask + (1 - w_attn) * noise_mask

                        # =========================================================
                        # Step 4: HOW — Apply guidance with adaptive safety_scale
                        # =========================================================
                        t_scale_val = args.target_scale if args.target_scale > 0 else current_safety_scale
                        a_scale_val = args.anchor_scale if args.anchor_scale > 0 else current_safety_scale

                        eps_final = apply_guidance(
                            eps_cfg=eps_cfg,
                            eps_null=eps_null,
                            eps_prompt=eps_prompt,
                            eps_target=eps_target_use,
                            eps_anchor=eps_anchor_use,
                            soft_mask=soft_mask,
                            guide_mode=args.guide_mode,
                            safety_scale=current_safety_scale,
                            cfg_scale=args.cfg_scale,
                            target_scale=t_scale_val,
                            anchor_scale=a_scale_val,
                            proj_scale=args.proj_scale,
                        )

                        guided_count += 1
                        area_val = float(soft_mask.mean().item())
                        mask_areas.append(area_val)

                        # Log adaptive params for this step
                        adaptive_log.append({
                            "step": step_idx,
                            "safety_scale_t": round(current_safety_scale, 4),
                            "threshold_t": round(current_threshold, 4),
                            "alpha_t": round(current_alpha, 4),
                            "cas_threshold_t": round(current_cas_threshold, 4),
                            "mask_area": round(area_val, 4),
                        })

                        # Save spatial maps for debugging
                        if args.save_maps and step_idx % 10 == 0:
                            if args.where_mode in ("crossattn", "hybrid"):
                                attn_spatial = compute_attention_spatial_mask(
                                    probe_store,
                                    token_indices=probe_token_indices,
                                    target_resolution=64,
                                )
                                attn_map_np = attn_spatial.float().cpu().numpy()
                                attn_map_np = np.nan_to_num(attn_map_np, nan=0.0)
                                attn_map_img = (np.clip(attn_map_np, 0, 1) * 255).astype(np.uint8)
                                Image.fromarray(attn_map_img, 'L').save(
                                    str(outdir / "maps" / f"{prompt_idx:04d}_{sample_idx:02d}_s{step_idx:03d}_attn.png"))

                            if args.where_mode in ("noise", "hybrid"):
                                cas_map_np = spatial_cas_map.float().cpu().numpy()
                                cas_map_np = np.nan_to_num(cas_map_np, nan=0.0)
                                cas_map_img = (np.clip((cas_map_np + 1) / 2, 0, 1) * 255).astype(np.uint8)
                                Image.fromarray(cas_map_img, 'L').save(
                                    str(outdir / "maps" / f"{prompt_idx:04d}_{sample_idx:02d}_s{step_idx:03d}_cas.png"))

                            mask_np = soft_mask[0, 0].float().cpu().numpy()
                            mask_img = (np.clip(mask_np, 0, 1) * 255).astype(np.uint8)
                            Image.fromarray(mask_img, 'L').save(
                                str(outdir / "maps" / f"{prompt_idx:04d}_{sample_idx:02d}_s{step_idx:03d}_mask.png"))

                    else:
                        eps_final = eps_cfg

                # =============================================================
                # DDIM step
                # =============================================================
                latents_prev = latents.clone()
                latents = scheduler.step(eps_final, t, latents).prev_sample

                # NaN recovery
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    print(f"  WARNING: NaN/Inf at step {step_idx}, reverting to standard CFG")
                    eps_fallback = eps_null + args.cfg_scale * (eps_prompt - eps_null)
                    latents = scheduler.step(eps_fallback, t, latents_prev).prev_sample

                # Debug logging
                if args.debug and step_idx % 10 == 0:
                    if multi_concept:
                        status = "GUIDED" if (any_triggered if 'any_triggered' in dir() else False) else "skip"
                        area_s = f" area={mask_areas[-1]:.3f}" if mask_areas else ""
                    else:
                        status = "GUIDED" if should_guide else ("CAS_ON" if should_trigger else "skip")
                        area_s = f" area={mask_areas[-1]:.3f}" if should_guide and mask_areas else ""
                    sched_s = f" ss={current_safety_scale:.2f} th={current_threshold:.3f} al={current_alpha:.1f}"
                    print(f"  [{step_idx:02d}] t={t.item():.0f} CAS={cas_val:.3f} "
                          f"{status}{area_s}{sched_s}")

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
                "where_mode": args.where_mode,
                "probe_source": args.probe_source,
                "schedule_type": args.schedule_type,
                "exemplar_mode": args.exemplar_mode,
                "adaptive_params": adaptive_log,
            }
            all_stats.append(stats)

            if prompt_idx % 10 == 0 or args.debug:
                tqdm.write(
                    f"  [{prompt_idx:03d}_{sample_idx}] guided={guided_count}/{total_steps} "
                    f"CAS avg={stats['avg_cas']:.3f} max={stats['max_cas']:.3f} "
                    f"mask_area={stats['avg_mask_area']:.3f} sched={args.schedule_type}"
                )

    # ---- Cleanup cross-attention probing ----
    if use_crossattn and original_processors is not None:
        restore_processors(unet, original_processors)

    # ---- Save summary statistics ----
    n = len(all_stats)
    n_trig = sum(1 for s in all_stats if s["guided_steps"] > 0)
    summary = {
        "method": "Timestep-Adaptive Mask Sharpening v18",
        "schedule_type": args.schedule_type,
        "where_mode": args.where_mode,
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
    print(f"  Schedule: {args.schedule_type}, scale_boost={args.scale_boost}")
    print(f"  WHERE: {args.where_mode}, Exemplar: {args.exemplar_mode}")
    print(f"  HOW:   {args.guide_mode}, base safety_scale={args.safety_scale}")
    print(f"  Avg guided steps: {summary['overall']['avg_guided_steps']:.1f}/{total_steps if n else 0}")
    print(f"  Avg mask area: {summary['overall']['avg_mask_area']:.3f}")
    print(f"  Output: {outdir}")
    print(f"  Stats:  {outdir / 'stats.json'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
