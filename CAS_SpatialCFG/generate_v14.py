#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hybrid WHERE Fusion v14: Training-Free Safe Generation

Architecture:
  WHEN (Global CAS): cos(d_prompt, d_target) > tau (same as v6, threshold=0.6)
  WHERE (Hybrid Fusion): Combines v6's focused cross-attention probe mask with
        v7's noise-based spatial CAS as a confirmation signal.
        mask_fused = mask_attn * sigmoid(alpha * (mask_cas - threshold))
        -> intersection: ~0.15-0.25 area (body regions that ARE generating nudity)
  HOW:  v7's hybrid mode with exemplar anchor direction + dag_adaptive option

Key innovation over v6 and v7:
  - v6 crossattn probe: focused (~0.31 area) but can fire on non-nudity body parts
  - v7 noise CAS: broad (~0.88 area) but confirms actual nudity generation
  - v14 fused mask: intersection of both signals -> precise body regions that
    ARE currently generating nudity content
  - Supports --where_mode {fused, crossattn_only, noise_only} for ablation

UNet call budget per guided step:
  - exemplar mode: 3 calls (null+prompt batched, target online; anchor from exemplar)
  - text mode: 4 calls (null+prompt batched, target online, anchor online)
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
    restore_processors,
    compute_attention_spatial_mask,
    find_token_indices,
)
from concept_pack_loader import load_concept_pack, load_multiple_packs, get_combined_target_words


# =============================================================================
# Global CAS: Concept Alignment Score (WHEN)
# =============================================================================
class GlobalCAS:
    """Detect harmful prompt alignment via cosine(d_prompt, d_target).

    Supports both online (eps_target from UNet) and exemplar (pre-computed
    global direction) modes for target direction.
    """

    def __init__(self, threshold: float = 0.6, sticky: bool = True):
        self.threshold = threshold
        self.sticky = sticky
        self.triggered = False

    def reset(self):
        self.triggered = False

    def compute(self, eps_prompt, eps_null, eps_target=None, d_target_global=None):
        """
        Compute global CAS from noise predictions.

        Args:
            eps_prompt: noise prediction conditioned on prompt [1, 4, 64, 64]
            eps_null:   unconditional noise prediction [1, 4, 64, 64]
            eps_target: (optional) noise prediction conditioned on target [1, 4, 64, 64]
            d_target_global: (optional) pre-computed target global direction [16384]

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

        if self.sticky and self.triggered:
            return cas, True

        if cas > self.threshold:
            if self.sticky:
                self.triggered = True
            return cas, True

        return cas, False


# =============================================================================
# Spatial CAS: Per-Pixel Concept Alignment (WHERE — noise-based)
# =============================================================================
def compute_spatial_cas(
    eps_prompt: torch.Tensor,   # [1, 4, 64, 64]
    eps_null: torch.Tensor,
    eps_target: torch.Tensor,
    neighborhood_size: int = 3,
) -> torch.Tensor:
    """
    Compute per-pixel Spatial CAS using 3x3 neighborhood pooling (unfold).

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
    d_prompt: torch.Tensor,     # [1, 4, H, W]
    d_target: torch.Tensor,     # [1, 4, H, W]
    neighborhood_size: int = 3,
) -> torch.Tensor:
    """
    Compute per-pixel Spatial CAS using pre-computed directions (exemplar mode).
    """
    d_prompt_f = d_prompt.float()
    d_target_f = d_target.float()

    H, W = d_prompt_f.shape[2], d_prompt_f.shape[3]
    pad = neighborhood_size // 2

    d_prompt_unfolded = F.unfold(d_prompt_f, kernel_size=neighborhood_size, padding=pad)
    d_target_unfolded = F.unfold(d_target_f, kernel_size=neighborhood_size, padding=pad)

    spatial_cas = F.cosine_similarity(d_prompt_unfolded, d_target_unfolded, dim=1)
    spatial_cas = spatial_cas.reshape(H, W)

    return spatial_cas


# =============================================================================
# Soft Mask Utilities
# =============================================================================
def compute_soft_mask(
    spatial_cas: torch.Tensor,    # [H, W]
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
    x: torch.Tensor,         # [B, C, H, W]
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Apply separable Gaussian blur to a 2D tensor."""
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
    guide_mode: str = "hybrid",
    safety_scale: float = 1.0,
    cfg_scale: float = 7.5,
    **kwargs,
) -> torch.Tensor:
    """
    Apply spatially-masked guidance to remove unsafe content.

    Modes:
        hybrid: Subtract target AND add anchor direction.
            eps_final = eps_cfg - t_s*M*(eps_target - eps_null) + a_s*M*(eps_anchor - eps_null)

        sld: Subtract target concept direction in masked regions.
            eps_final = eps_cfg - s * M * (eps_target - eps_null)

        dag_adaptive: Area-proportional + magnitude-proportional scaling.
            eps_final = eps_cfg - s * (area_factor * mag_scale * M) * d_target

        anchor_inpaint: Blend original CFG with anchor CFG in masked regions.
            eps_final = eps_cfg * (1 - s*M) + eps_anchor_cfg * (s*M)

        hybrid_proj: Project out nudity from prompt, then blend with anchor.
        projection: Project out nudity component from prompt direction.
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
        # DAG: area-based + magnitude-based adaptive scaling
        d_target = eps_target - eps_null
        binary_mask = (mask > 0.5).to(mask.dtype)
        area = binary_mask.sum() / binary_mask.numel()
        area_scale = 5.0 / (mask.shape[-1] * mask.shape[-2])
        area_factor = area_scale * area * mask.numel()  # = 5 * area_fraction

        # Magnitude scaling: rescale weight from [0,1] to [1, 5]
        mag_scale = 1.0 + 4.0 * mask
        correction = safety_scale * area_factor * mag_scale * binary_mask * d_target
        eps_final = eps_cfg - correction

    elif guide_mode == "anchor_inpaint":
        eps_anchor_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
        eps_final = eps_cfg * (1.0 - safety_scale * mask) + eps_anchor_cfg * (safety_scale * mask)

    elif guide_mode == "hybrid_proj":
        d_prompt = eps_prompt - eps_null
        d_target = eps_target - eps_null
        dot = (d_prompt * d_target).sum(dim=1, keepdim=True)
        norm_sq = (d_target * d_target).sum(dim=1, keepdim=True).clamp(min=1e-8)
        proj = (dot / norm_sq) * d_target
        p_scale = kwargs.get("proj_scale", 1.0)
        d_safe = d_prompt - p_scale * proj
        eps_safe_cfg = eps_null + cfg_scale * d_safe
        a_scale = kwargs.get("anchor_scale", safety_scale)
        eps_anchor_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
        eps_final = eps_safe_cfg * (1.0 - a_scale * mask) + eps_anchor_cfg * (a_scale * mask)

    elif guide_mode == "projection":
        d_prompt = eps_prompt - eps_null
        d_target = eps_target - eps_null
        dot = (d_prompt * d_target).sum(dim=1, keepdim=True)
        norm_sq = (d_target * d_target).sum(dim=1, keepdim=True).clamp(min=1e-8)
        proj = (dot / norm_sq) * d_target
        d_safe = d_prompt - safety_scale * proj
        eps_safe_cfg = eps_null + cfg_scale * d_safe
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
# Argument Parsing
# =============================================================================
def parse_args():
    p = ArgumentParser(description="Hybrid WHERE Fusion v14: CrossAttn Probe + Noise CAS")
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

    # WHERE mode — the key v14 parameter
    p.add_argument("--where_mode", type=str, default="fused",
                    choices=["fused", "crossattn_only", "noise_only"],
                    help="Spatial detection: fused (v14, crossattn*noise_cas), "
                         "crossattn_only (v6), noise_only (v7 fallback)")

    # Cross-attention probe parameters
    p.add_argument("--attn_resolutions", type=int, nargs="+", default=[16, 32],
                    help="Cross-attention resolutions to probe")
    p.add_argument("--target_words", type=str, nargs="+",
                    default=["nude", "naked", "nudity", "nsfw", "bare", "body"],
                    help="Keywords to find target token indices for attention probing")

    # Probe source — controls what K_target keys are used in cross-attention probing
    p.add_argument("--probe_source", type=str, default="text",
                    choices=["text", "image", "both"],
                    help="Cross-attention probe key source: "
                         "text (v6-style, keyword tokens), "
                         "image (CLIP exemplar tokens from --clip_embeddings), "
                         "both (union of text and image probe masks via combined embedding)")
    p.add_argument("--clip_embeddings", type=str, default=None,
                    help="Path to CLIP exemplar embeddings .pt file (required for "
                         "probe_source=image or both). Must contain 'target_clip_embeds' "
                         "key with shape [1, 77, 768] and config['n_tokens'].")

    # Spatial CAS / Soft mask (Where)
    p.add_argument("--spatial_threshold", type=float, default=0.3,
                    help="Per-pixel CAS threshold (sigmoid center for noise mask)")
    p.add_argument("--sigmoid_alpha", type=float, default=10.0,
                    help="Sharpness of sigmoid mask transition")
    p.add_argument("--neighborhood_size", type=int, default=3,
                    help="Pooling kernel size for robust per-pixel CAS (noise mode)")
    p.add_argument("--blur_sigma", type=float, default=1.0,
                    help="Gaussian blur sigma for final mask smoothing (0=no blur)")

    # Cross-attention mask threshold (for crossattn and fused modes)
    p.add_argument("--attn_threshold", type=float, default=0.3,
                    help="Threshold for binarizing cross-attention probe mask")
    p.add_argument("--attn_sigmoid_alpha", type=float, default=10.0,
                    help="Sigmoid sharpness for cross-attention mask softening")

    # Guidance (How)
    p.add_argument("--guide_mode", type=str, default="hybrid",
                    choices=["hybrid", "sld", "dag_adaptive", "anchor_inpaint",
                             "projection", "hybrid_proj"],
                    help="Guidance mode")
    p.add_argument("--safety_scale", type=float, default=1.0,
                    help="Guidance strength (anchor_inpaint: 0-1, sld: 1-10)")
    p.add_argument("--target_scale", type=float, default=-1.0,
                    help="Hybrid only: target repulsion scale (default: same as safety_scale)")
    p.add_argument("--anchor_scale", type=float, default=-1.0,
                    help="Hybrid only: anchor attraction scale (default: same as safety_scale)")
    p.add_argument("--proj_scale", type=float, default=1.0,
                    help="hybrid_proj only: projection strength for nudity removal")
    p.add_argument("--guide_start_frac", type=float, default=0.0,
                    help="Fraction of total steps before guidance starts (0=all steps)")

    # Concepts
    p.add_argument("--target_concepts", type=str, nargs="+",
                    default=["nudity", "nude person", "naked body"],
                    help="Target concepts to erase")
    p.add_argument("--anchor_concepts", type=str, nargs="+",
                    default=["clothed person", "person wearing clothes"],
                    help="Anchor concepts to guide toward")

    # v7 exemplar mode args
    p.add_argument("--concept_dir_path", type=str, default=None,
                    help="Path to concept_directions.pt (required for exemplar mode)")
    p.add_argument("--exemplar_mode", type=str, default="exemplar",
                    choices=["exemplar", "text", "hybrid_exemplar"],
                    help="Direction source: exemplar (pre-computed, 3 UNet calls), "
                         "text (online, 4 UNet calls), hybrid_exemplar (blended)")
    p.add_argument("--exemplar_weight", type=float, default=0.7,
                    help="Weight for exemplar directions in hybrid_exemplar mode (0-1)")

    # Multi-concept
    p.add_argument("--concept_packs", type=str, nargs="+", default=None,
                    help="Concept pack directories for multi-concept erasing. "
                         "If set, overrides --target_concepts, --anchor_concepts, etc.")
    p.add_argument("--family_level", action="store_true", default=False,
                    help="Apply corrections at family level (per sub-concept) instead of concept level")

    # Misc
    p.add_argument("--save_maps", action="store_true",
                    help="Save spatial maps, attention maps, and masks for debugging")
    p.add_argument("--debug", action="store_true",
                    help="Print per-step debug info")
    p.add_argument("--start_idx", type=int, default=0,
                    help="Start prompt index (for parallel runs)")
    p.add_argument("--end_idx", type=int, default=-1,
                    help="End prompt index (-1 = all)")

    args = p.parse_args()
    if args.cas_no_sticky:
        args.cas_sticky = False

    # Validate exemplar mode requirements (skip if using concept_packs)
    if args.concept_packs is None:
        if args.exemplar_mode in ("exemplar", "hybrid_exemplar") and args.concept_dir_path is None:
            p.error("--concept_dir_path is required for exemplar and hybrid_exemplar modes")
        if args.probe_source in ("image", "both") and args.clip_embeddings is None:
            p.error("--clip_embeddings is required for probe_source=image or both")

    return args


# =============================================================================
# Main Generation Loop
# =============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Multi-concept mode flag
    multi_concept = args.concept_packs is not None

    # Determine if we need crossattn probe (fused and crossattn_only modes)
    use_crossattn = args.where_mode in ("fused", "crossattn_only")
    # Determine if we need noise CAS (fused and noise_only modes)
    use_noise_cas = args.where_mode in ("fused", "noise_only")
    # Determine if image exemplar probe is needed
    use_image_probe = args.probe_source in ("image", "both") and not multi_concept

    where_label = {
        "fused": f"Fused (crossattn*noise_cas), attn_res={args.attn_resolutions}, "
                 f"attn_thr={args.attn_threshold}, noise_thr={args.spatial_threshold}",
        "crossattn_only": f"CrossAttn Probe only, res={args.attn_resolutions}, "
                          f"thr={args.attn_threshold}",
        "noise_only": f"Noise CAS only, neighborhood={args.neighborhood_size}, "
                      f"thr={args.spatial_threshold}",
    }[args.where_mode]

    print(f"\n{'='*70}")
    print(f"Hybrid WHERE Fusion v14: CrossAttn Probe + Noise CAS")
    if multi_concept:
        print(f"  ** MULTI-CONCEPT MODE ** ({len(args.concept_packs)} packs)")
    print(f"{'='*70}")
    print(f"  WHEN:  Global CAS threshold={args.cas_threshold}, sticky={args.cas_sticky}")
    print(f"  WHERE: {args.where_mode} -- {where_label}")
    print(f"         probe_source={args.probe_source}, sigmoid_alpha={args.sigmoid_alpha}, blur={args.blur_sigma}")
    print(f"  HOW:   {args.guide_mode}, safety_scale={args.safety_scale}, "
          f"start_frac={args.guide_start_frac}")
    if multi_concept:
        print(f"  Concept packs: {args.concept_packs}")
        print(f"  Family level: {args.family_level}")
    else:
        print(f"  Exemplar mode: {args.exemplar_mode}"
              + (f" (weight={args.exemplar_weight})" if args.exemplar_mode == "hybrid_exemplar" else ""))
        if args.exemplar_mode in ("exemplar", "hybrid_exemplar"):
            print(f"  Concept directions: {args.concept_dir_path}")
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

    # Unconditional (empty string)
    with torch.no_grad():
        uncond_inputs = tokenizer(
            "", padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]

    # ==================================================================
    # Multi-concept vs single-concept setup
    # ==================================================================
    concept_packs_list = None      # List of loaded pack dicts (multi-concept)
    pack_cas_gates = None          # Per-pack GlobalCAS instances
    pack_target_embeds = None      # Per-pack target embeddings
    pack_anchor_embeds = None      # Per-pack anchor embeddings
    pack_probe_stores = None       # Per-pack probe stores
    pack_probe_token_indices = None  # Per-pack token indices
    pack_original_processors = None  # We re-register probe per pack per step

    # Single-concept state (used when multi_concept is False)
    target_dirs = None
    anchor_dirs = None
    target_global = None
    anchor_global = None
    target_embeds = None
    anchor_embeds = None
    clip_target_embeds = None
    clip_n_tokens = None
    original_processors = None
    probe_store = None
    probe_token_indices = None

    if multi_concept:
        # ---- Multi-concept: load packs ----
        print(f"\nLoading {len(args.concept_packs)} concept packs...")
        concept_packs_list = load_multiple_packs(args.concept_packs, device=device)

        # For each pack, encode target/anchor embeddings and set up CAS gate
        pack_cas_gates = []
        pack_target_embeds = []
        pack_anchor_embeds = []
        pack_probe_token_indices = []

        for i, pack in enumerate(concept_packs_list):
            # CAS gate: use pack's threshold if available, else fallback to args
            pack_threshold = pack.get("cas_threshold", args.cas_threshold)
            pack_cas_gates.append(GlobalCAS(threshold=pack_threshold, sticky=args.cas_sticky))

            # Encode target concepts from pack
            pack_target_concepts = pack.get("target_concepts", args.target_concepts)
            with torch.no_grad():
                t_emb = encode_concepts(text_encoder, tokenizer, pack_target_concepts, device)
            pack_target_embeds.append(t_emb)

            # Encode anchor concepts from pack
            pack_anchor_concepts = pack.get("anchor_concepts", args.anchor_concepts)
            with torch.no_grad():
                a_emb = encode_concepts(text_encoder, tokenizer, pack_anchor_concepts, device)
            pack_anchor_embeds.append(a_emb)

            # Target words for cross-attention probing
            pack_target_words = pack.get("target_words", args.target_words)
            target_text = ", ".join(pack_target_concepts)
            p_indices = find_token_indices(target_text, pack_target_words, tokenizer)
            pack_probe_token_indices.append(p_indices)

            print(f"  Pack [{i}] '{pack.get('concept', 'unknown')}': "
                  f"threshold={pack_threshold}, "
                  f"targets={pack_target_concepts[:3]}{'...' if len(pack_target_concepts) > 3 else ''}, "
                  f"probe_tokens={p_indices[:6]}{'...' if len(p_indices) > 6 else ''}")

        # For multi-concept, we use a shared probe store but re-register per pack
        # during the denoising loop. Set up the initial probe store.
        if use_crossattn:
            probe_store = AttentionProbeStore()
            # Use combined target words from all packs for initial probe setup
            combined_target_words = get_combined_target_words(concept_packs_list)
            combined_target_concepts = []
            for pack in concept_packs_list:
                combined_target_concepts.extend(pack.get("target_concepts", []))
            if combined_target_concepts:
                with torch.no_grad():
                    combined_embeds = encode_concepts(
                        text_encoder, tokenizer, combined_target_concepts, device
                    )
                target_keys = precompute_target_keys(
                    unet, combined_embeds.to(dtype=next(unet.parameters()).dtype),
                    args.attn_resolutions
                )
                original_processors = register_attention_probe(
                    unet, probe_store, target_keys, args.attn_resolutions
                )
                # Combined token indices for capturing attention during forward pass
                combined_text = ", ".join(combined_target_concepts)
                probe_token_indices = find_token_indices(
                    combined_text, combined_target_words, tokenizer
                )
                print(f"  Multi-concept probe (text): combined_tokens={probe_token_indices[:8]}"
                      f"{'...' if len(probe_token_indices) > 8 else ''}")

        # Init single CAS (unused in multi-concept denoising, kept for interface)
        cas = GlobalCAS(threshold=args.cas_threshold, sticky=args.cas_sticky)

    else:
        # ---- Single-concept setup (original behavior) ----

        # Load pre-computed concept directions (exemplar / hybrid_exemplar)
        if args.exemplar_mode in ("exemplar", "hybrid_exemplar"):
            print(f"Loading concept directions from {args.concept_dir_path} ...")
            concept_data = torch.load(args.concept_dir_path, map_location=device)
            target_dirs = concept_data['target_directions']
            anchor_dirs = concept_data['anchor_directions']
            target_global = concept_data['target_global']
            anchor_global = concept_data['anchor_global']
            print(f"  Loaded directions for {len(target_dirs)} timesteps")

        # Pre-encode concept embeddings
        with torch.no_grad():
            target_embeds = encode_concepts(text_encoder, tokenizer,
                                            args.target_concepts, device)
        anchor_embeds = None
        if args.exemplar_mode in ("text", "hybrid_exemplar"):
            with torch.no_grad():
                anchor_embeds = encode_concepts(text_encoder, tokenizer,
                                                args.anchor_concepts, device)
        print(f"Embeddings encoded: target={len(args.target_concepts)} concepts (averaged)"
              + (f", anchor={len(args.anchor_concepts)} concepts (averaged)"
                 if anchor_embeds is not None else ", anchor=exemplar"))

        # Load CLIP exemplar embeddings (image / both probe modes)
        if use_image_probe:
            clip_data = torch.load(args.clip_embeddings, map_location=device)
            clip_target_embeds = clip_data["target_clip_embeds"].to(device=device, dtype=unet.dtype)
            clip_n_tokens = clip_data["config"]["n_tokens"]
            print(f"Loaded CLIP exemplar embeddings: shape={clip_target_embeds.shape}, "
                  f"n_tokens={clip_n_tokens}")

        # Setup cross-attention probing
        if use_crossattn:
            probe_store = AttentionProbeStore()

            if args.probe_source == "text":
                probe_embeds = target_embeds
                target_keys = precompute_target_keys(unet, probe_embeds.to(dtype=next(unet.parameters()).dtype), args.attn_resolutions)
                original_processors = register_attention_probe(
                    unet, probe_store, target_keys, args.attn_resolutions
                )
                target_text = ", ".join(args.target_concepts)
                probe_token_indices = find_token_indices(target_text, args.target_words, tokenizer)
                print(f"  Probe (text): token_indices={probe_token_indices}")

            elif args.probe_source == "image":
                target_keys = precompute_target_keys(unet, clip_target_embeds.to(dtype=next(unet.parameters()).dtype), args.attn_resolutions)
                original_processors = register_attention_probe(
                    unet, probe_store, target_keys, args.attn_resolutions
                )
                probe_token_indices = list(range(1, 1 + clip_n_tokens))
                print(f"  Probe (image): token_indices={probe_token_indices}")

            elif args.probe_source == "both":
                max_len = tokenizer.model_max_length
                text_content = target_embeds[:, 1:, :]
                image_content = clip_target_embeds[:, 1:1 + clip_n_tokens, :]

                n_img = clip_n_tokens
                n_text_max = max_len - 1 - n_img - 1
                text_content = text_content[:, :n_text_max, :]

                n_text_actual = text_content.shape[1]
                bos_embed = target_embeds[:, :1, :]
                eos_embed = target_embeds[:, -1:, :]

                n_combined = 1 + n_text_actual + n_img + 1
                n_pad = max_len - n_combined
                pad_embed = eos_embed.expand(1, n_pad, -1) if n_pad > 0 else None

                parts = [bos_embed, text_content, image_content, eos_embed]
                if pad_embed is not None:
                    parts.append(pad_embed)
                combined_embeds = torch.cat(parts, dim=1)

                target_keys = precompute_target_keys(unet, combined_embeds.to(dtype=next(unet.parameters()).dtype), args.attn_resolutions)
                original_processors = register_attention_probe(
                    unet, probe_store, target_keys, args.attn_resolutions
                )
                probe_token_indices = list(range(1, 1 + n_text_actual + n_img))
                print(f"  Probe (both): n_text={n_text_actual}, n_img={n_img}, "
                      f"token_indices={probe_token_indices[:8]}{'...' if len(probe_token_indices) > 8 else ''}")

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
            if multi_concept:
                for gate in pack_cas_gates:
                    gate.reset()

            guided_count = 0
            cas_values = []
            mask_areas_fused = []
            mask_areas_attn = []
            mask_areas_noise = []

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
                # Step 1: UNet forward passes (CFG with optional probe)
                # =========================================================
                lat_in = scheduler.scale_model_input(latents, t)
                t_int = t.item()

                # Pass 1+2: CFG (uncond + prompt batched)
                # Activate probe during this forward pass for crossattn modes
                if use_crossattn:
                    probe_store.active = True
                    probe_store.reset()

                with torch.no_grad():
                    lat_batch = torch.cat([lat_in, lat_in])
                    embed_batch = torch.cat([uncond_embeds, prompt_embeds])
                    raw = unet(lat_batch, t, encoder_hidden_states=embed_batch).sample
                    eps_null, eps_prompt = raw.chunk(2)

                if use_crossattn:
                    probe_store.active = False

                # Standard CFG
                eps_cfg = eps_null + args.cfg_scale * (eps_prompt - eps_null)

                # =========================================================
                # Step 2-4: WHEN/WHERE/HOW — Multi-concept or single-concept
                # =========================================================
                if multi_concept:
                    # ===================================================
                    # MULTI-CONCEPT: iterate over packs, accumulate corrections
                    # ===================================================
                    any_triggered = False
                    total_correction = torch.zeros_like(eps_cfg)
                    max_cas_val = 0.0

                    for pack_idx, pack in enumerate(concept_packs_list):
                        pack_cas_gate = pack_cas_gates[pack_idx]
                        p_target_embeds = pack_target_embeds[pack_idx]
                        p_anchor_embeds = pack_anchor_embeds[pack_idx]
                        p_probe_indices = pack_probe_token_indices[pack_idx]

                        # Get pack-specific config with fallbacks to args
                        p_guide_mode = pack.get("guide_mode", args.guide_mode)
                        p_safety_scale = pack.get("safety_scale", args.safety_scale)
                        p_exemplar_mode = pack.get("exemplar_mode", args.exemplar_mode)

                        # --- Compute target direction for this pack ---
                        if p_exemplar_mode == "exemplar" and "concept_directions" in pack:
                            pack_dirs = pack["concept_directions"]
                            with torch.no_grad():
                                eps_target_pack = unet(lat_in, t,
                                                       encoder_hidden_states=p_target_embeds).sample
                            eps_target_use_pack = eps_target_pack

                            pack_anchor_dirs = pack.get("anchor_directions", {})
                            if t_int in pack_anchor_dirs:
                                d_anchor_pack = pack_anchor_dirs[t_int].to(device, dtype=torch.float16).unsqueeze(0)
                                eps_anchor_use_pack = eps_null + d_anchor_pack
                            else:
                                eps_anchor_use_pack = eps_null  # fallback

                            cas_val_pack, should_trigger_pack = pack_cas_gate.compute(
                                eps_prompt, eps_null, eps_target=eps_target_pack
                            )
                        else:
                            # Text mode (default for multi-concept)
                            with torch.no_grad():
                                eps_target_pack = unet(lat_in, t,
                                                       encoder_hidden_states=p_target_embeds).sample
                            eps_target_use_pack = eps_target_pack

                            cas_val_pack, should_trigger_pack = pack_cas_gate.compute(
                                eps_prompt, eps_null, eps_target=eps_target_pack
                            )

                            # Anchor: compute online
                            with torch.no_grad():
                                eps_anchor_use_pack = unet(lat_in, t,
                                                           encoder_hidden_states=p_anchor_embeds).sample

                        max_cas_val = max(max_cas_val, cas_val_pack)

                        in_window = step_idx >= guide_start_step
                        if not (should_trigger_pack and in_window):
                            continue

                        any_triggered = True

                        # --- WHERE: compute per-pack mask ---
                        # Cross-attention mask (shared probe, pack-specific tokens)
                        mask_attn_pack = None
                        if use_crossattn and probe_store is not None:
                            attn_spatial = compute_attention_spatial_mask(
                                probe_store,
                                token_indices=p_probe_indices,
                                target_resolution=64,
                                resolutions_to_use=args.attn_resolutions,
                            )
                            mask_attn_pack = torch.sigmoid(
                                args.attn_sigmoid_alpha * (attn_spatial.to(device) - args.attn_threshold)
                            )
                            mask_attn_pack = mask_attn_pack.unsqueeze(0).unsqueeze(0)

                        # Noise CAS mask (per-pack target)
                        mask_cas_soft_pack = None
                        if use_noise_cas:
                            spatial_cas_map = compute_spatial_cas(
                                eps_prompt, eps_null, eps_target_use_pack,
                                neighborhood_size=args.neighborhood_size,
                            )
                            mask_cas_soft_pack = torch.sigmoid(
                                args.sigmoid_alpha * (spatial_cas_map.to(device) - args.spatial_threshold)
                            )
                            mask_cas_soft_pack = mask_cas_soft_pack.unsqueeze(0).unsqueeze(0)

                        # Fuse masks
                        if args.where_mode == "fused" and mask_attn_pack is not None and mask_cas_soft_pack is not None:
                            soft_mask_pack = mask_attn_pack * mask_cas_soft_pack
                        elif args.where_mode == "crossattn_only" and mask_attn_pack is not None:
                            soft_mask_pack = mask_attn_pack
                        elif args.where_mode == "noise_only" and mask_cas_soft_pack is not None:
                            soft_mask_pack = mask_cas_soft_pack
                        else:
                            # Fallback: use whichever mask is available
                            soft_mask_pack = mask_attn_pack if mask_attn_pack is not None else mask_cas_soft_pack
                            if soft_mask_pack is None:
                                soft_mask_pack = torch.ones(1, 1, 64, 64, device=device, dtype=eps_cfg.dtype)

                        if args.blur_sigma > 0:
                            soft_mask_pack = gaussian_blur_2d(soft_mask_pack, kernel_size=5, sigma=args.blur_sigma)
                        soft_mask_pack = soft_mask_pack.clamp(0, 1)

                        # --- HOW: compute per-pack correction ---
                        # Use apply_guidance to get corrected eps, then extract correction
                        eps_guided_pack = apply_guidance(
                            eps_cfg=eps_cfg,
                            eps_null=eps_null,
                            eps_prompt=eps_prompt,
                            eps_target=eps_target_use_pack,
                            eps_anchor=eps_anchor_use_pack,
                            soft_mask=soft_mask_pack,
                            guide_mode=p_guide_mode,
                            safety_scale=p_safety_scale,
                            cfg_scale=args.cfg_scale,
                            target_scale=args.target_scale if args.target_scale > 0 else p_safety_scale,
                            anchor_scale=args.anchor_scale if args.anchor_scale > 0 else p_safety_scale,
                            proj_scale=args.proj_scale,
                        )
                        # Correction = eps_cfg - eps_guided (what was subtracted)
                        pack_correction = eps_cfg - eps_guided_pack
                        total_correction = total_correction + pack_correction

                        # Track mask areas
                        if mask_attn_pack is not None:
                            mask_areas_attn.append(float(mask_attn_pack.mean().item()))
                        if mask_cas_soft_pack is not None:
                            mask_areas_noise.append(float(mask_cas_soft_pack.mean().item()))
                        mask_areas_fused.append(float(soft_mask_pack.mean().item()))

                    cas_val = max_cas_val
                    cas_values.append(cas_val)

                    if any_triggered:
                        eps_final = eps_cfg - total_correction
                        # NaN/Inf guard on accumulated correction
                        if torch.isnan(eps_final).any() or torch.isinf(eps_final).any():
                            eps_final = torch.where(torch.isfinite(eps_final), eps_final, eps_cfg)
                        guided_count += 1

                        # Save maps for debugging (combined fused mask of last pack)
                        if args.save_maps and step_idx % 10 == 0:
                            prefix = f"{prompt_idx:04d}_{sample_idx:02d}_s{step_idx:03d}"
                            if 'soft_mask_pack' in locals():
                                m = soft_mask_pack[0, 0].float().cpu().numpy()
                                m = np.nan_to_num(m, nan=0.0)
                                Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8), 'L').save(
                                    str(outdir / "maps" / f"{prefix}_fused.png"))
                    else:
                        eps_final = eps_cfg

                else:
                    # ===================================================
                    # SINGLE-CONCEPT: original v14 behavior
                    # ===================================================

                    # Step 2: Get target/anchor directions (mode-dependent)
                    if args.exemplar_mode == "exemplar":
                        with torch.no_grad():
                            eps_target_online = unet(lat_in, t,
                                                     encoder_hidden_states=target_embeds).sample
                        eps_target_use = eps_target_online

                        d_anchor = anchor_dirs[t_int].to(device, dtype=torch.float16).unsqueeze(0)
                        eps_anchor_use = eps_null + d_anchor

                        cas_val, should_trigger = cas.compute(
                            eps_prompt, eps_null, eps_target=eps_target_online
                        )

                    elif args.exemplar_mode == "text":
                        with torch.no_grad():
                            eps_target_online = unet(lat_in, t,
                                                     encoder_hidden_states=target_embeds).sample
                        eps_target_use = eps_target_online

                        cas_val, should_trigger = cas.compute(
                            eps_prompt, eps_null, eps_target=eps_target_online
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
                            d_target_global=d_target_blended_global.squeeze(0)
                        )

                    cas_values.append(cas_val)

                    in_window = step_idx >= guide_start_step
                    should_guide = should_trigger and in_window

                    if should_guide:
                        # Get anchor (mode-dependent)
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

                        # Step 3: WHERE — compute spatial masks
                        mask_attn = None
                        if use_crossattn:
                            attn_spatial = compute_attention_spatial_mask(
                                probe_store,
                                token_indices=probe_token_indices,
                                target_resolution=64,
                                resolutions_to_use=args.attn_resolutions,
                            )
                            mask_attn = torch.sigmoid(
                                args.attn_sigmoid_alpha * (attn_spatial.to(device) - args.attn_threshold)
                            )
                            mask_attn = mask_attn.unsqueeze(0).unsqueeze(0)

                        mask_cas_soft = None
                        if use_noise_cas:
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
                            mask_cas_soft = torch.sigmoid(
                                args.sigmoid_alpha * (spatial_cas_map.to(device) - args.spatial_threshold)
                            )
                            mask_cas_soft = mask_cas_soft.unsqueeze(0).unsqueeze(0)

                        # Fuse masks
                        if args.where_mode == "fused":
                            soft_mask = mask_attn * mask_cas_soft
                        elif args.where_mode == "crossattn_only":
                            soft_mask = mask_attn
                        elif args.where_mode == "noise_only":
                            soft_mask = mask_cas_soft

                        if args.blur_sigma > 0:
                            soft_mask = gaussian_blur_2d(soft_mask, kernel_size=5, sigma=args.blur_sigma)
                        soft_mask = soft_mask.clamp(0, 1)

                        # Step 4: HOW — Apply guidance
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

                        # Track mask areas separately for stats
                        fused_area = float(soft_mask.mean().item())
                        mask_areas_fused.append(fused_area)
                        if mask_attn is not None:
                            mask_areas_attn.append(float(mask_attn.mean().item()))
                        if mask_cas_soft is not None:
                            mask_areas_noise.append(float(mask_cas_soft.mean().item()))

                        # Save maps for debugging
                        if args.save_maps and step_idx % 10 == 0:
                            prefix = f"{prompt_idx:04d}_{sample_idx:02d}_s{step_idx:03d}"
                            if mask_attn is not None:
                                m = mask_attn[0, 0].float().cpu().numpy()
                                m = np.nan_to_num(m, nan=0.0)
                                Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8), 'L').save(
                                    str(outdir / "maps" / f"{prefix}_attn.png"))
                            if mask_cas_soft is not None:
                                m = mask_cas_soft[0, 0].float().cpu().numpy()
                                m = np.nan_to_num(m, nan=0.0)
                                Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8), 'L').save(
                                    str(outdir / "maps" / f"{prefix}_noise.png"))
                            m = soft_mask[0, 0].float().cpu().numpy()
                            m = np.nan_to_num(m, nan=0.0)
                            Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8), 'L').save(
                                str(outdir / "maps" / f"{prefix}_fused.png"))

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
                    if multi_concept:
                        was_guided = any_triggered
                        status = "GUIDED" if was_guided else "skip"
                    else:
                        was_guided = should_guide
                        status = "GUIDED" if should_guide else ("CAS_ON" if should_trigger else "skip")
                    area_s = ""
                    if was_guided and mask_areas_fused:
                        area_s = f" fused={mask_areas_fused[-1]:.3f}"
                        if mask_areas_attn:
                            area_s += f" attn={mask_areas_attn[-1]:.3f}"
                        if mask_areas_noise:
                            area_s += f" noise={mask_areas_noise[-1]:.3f}"
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
            if multi_concept:
                cas_triggered = any(g.triggered for g in pack_cas_gates)
            else:
                cas_triggered = cas.triggered
            stats = {
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "seed": seed,
                "prompt": prompt[:100],
                "filename": fname,
                "guided_steps": guided_count,
                "total_steps": total_steps,
                "guidance_ratio": guided_count / max(total_steps, 1),
                "cas_triggered": cas_triggered,
                "multi_concept": multi_concept,
                "n_concept_packs": len(concept_packs_list) if multi_concept else 0,
                "avg_cas": float(np.mean(cas_values)) if cas_values else 0.0,
                "max_cas": float(np.max(cas_values)) if cas_values else 0.0,
                "avg_mask_area_fused": float(np.mean(mask_areas_fused)) if mask_areas_fused else 0.0,
                "avg_mask_area_attn": float(np.mean(mask_areas_attn)) if mask_areas_attn else 0.0,
                "avg_mask_area_noise": float(np.mean(mask_areas_noise)) if mask_areas_noise else 0.0,
                "where_mode": args.where_mode,
                "exemplar_mode": args.exemplar_mode,
            }
            all_stats.append(stats)

            if prompt_idx % 10 == 0 or args.debug:
                tqdm.write(
                    f"  [{prompt_idx:03d}_{sample_idx}] guided={guided_count}/{total_steps} "
                    f"CAS avg={stats['avg_cas']:.3f} max={stats['max_cas']:.3f} "
                    f"fused={stats['avg_mask_area_fused']:.3f} "
                    f"attn={stats['avg_mask_area_attn']:.3f} "
                    f"noise={stats['avg_mask_area_noise']:.3f}"
                )

    # ---- Cleanup cross-attention probing ----
    if use_crossattn and original_processors is not None:
        restore_processors(unet, original_processors)

    # ---- Save summary statistics ----
    n = len(all_stats)
    n_trig = sum(1 for s in all_stats if s["guided_steps"] > 0)
    summary = {
        "method": "Hybrid WHERE Fusion v14: CrossAttn Probe + Noise CAS",
        "multi_concept": multi_concept,
        "n_concept_packs": len(concept_packs_list) if multi_concept else 0,
        "concept_pack_dirs": args.concept_packs if multi_concept else None,
        "where_mode": args.where_mode,
        "exemplar_mode": args.exemplar_mode,
        "args": vars(args),
        "overall": {
            "total_images": n,
            "triggered": n_trig,
            "trigger_rate": n_trig / max(n, 1),
            "avg_guided_steps": float(np.mean([s["guided_steps"] for s in all_stats])) if n else 0,
            "avg_cas": float(np.mean([s["avg_cas"] for s in all_stats if s["avg_cas"] > 0])) if n else 0,
            "avg_mask_area_fused": float(np.mean([s["avg_mask_area_fused"] for s in all_stats if s["avg_mask_area_fused"] > 0])) if n else 0,
            "avg_mask_area_attn": float(np.mean([s["avg_mask_area_attn"] for s in all_stats if s["avg_mask_area_attn"] > 0])) if n else 0,
            "avg_mask_area_noise": float(np.mean([s["avg_mask_area_noise"] for s in all_stats if s["avg_mask_area_noise"] > 0])) if n else 0,
        },
        "per_image": all_stats,
    }
    with open(outdir / "stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"DONE! {n} images generated, {n_trig} triggered ({100*n_trig/max(n,1):.0f}%)")
    if multi_concept:
        print(f"  Mode: MULTI-CONCEPT ({len(concept_packs_list)} packs)")
        for i, pack in enumerate(concept_packs_list):
            print(f"    Pack [{i}]: {pack.get('concept', 'unknown')}")
    print(f"  WHERE mode: {args.where_mode}")
    print(f"  Exemplar mode: {args.exemplar_mode}")
    print(f"  Guide mode: {args.guide_mode}, safety_scale={args.safety_scale}")
    print(f"  Avg guided steps: {summary['overall']['avg_guided_steps']:.1f}/{total_steps if n else 0}")
    print(f"  Avg mask area — fused: {summary['overall']['avg_mask_area_fused']:.3f}, "
          f"attn: {summary['overall']['avg_mask_area_attn']:.3f}, "
          f"noise: {summary['overall']['avg_mask_area_noise']:.3f}")
    print(f"  Output: {outdir}")
    print(f"  Stats:  {outdir / 'stats.json'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
