#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contrastive Image Direction Safe Generation v16

Architecture:
  WHEN (Global CAS): cos(d_prompt, d_target) > tau
       Optionally enhanced by projecting noise-space CAS onto CLIP contrastive
       direction for more concept-specific detection.
  WHERE (Contrastive Cross-Attention Probe): probe UNet hidden states with
        contrastive direction K (nude - clothed in CLIP space) -> spatial mask.
        Supports probe_only, noise_only, and fused WHERE modes.
  HOW: Same guidance modes as v13 (hybrid, sld, projection, etc.)

Key innovation vs v13:
  - Contrastive direction d = mean(CLIP(nude)) - mean(CLIP(clothed)) isolates
    nudity-specific features by removing shared person/pose/background features.
  - Three contrastive probe variants: cls, patch, mixed.
  - WHERE fusion: combine probe attention mask with noise-based spatial CAS.

Key innovation vs v7:
  - Uses CLIP image-space contrastive direction instead of noise-space exemplar
    directions, providing richer semantic concept representation.
  - Cross-attention probing for spatial localization (vs v7's noise-based WHERE).

probe_source option (new in v16):
  - image: contrastive image direction probe (original v16 behavior)
  - text:  text keyword cross-attention probe (v6 style, uses target_words)
  - both:  union mask = max(text_mask, image_mask)  [default]

Usage:
    # 1. First prepare contrastive embeddings:
    CUDA_VISIBLE_DEVICES=0 python prepare_contrastive_direction.py

    # 2. Then generate:
    CUDA_VISIBLE_DEVICES=0 python generate_v16.py \
        --prompts prompts/ringabell_anchor_subset.csv \
        --outdir outputs/v16/ringabell_cls_hybrid \
        --contrastive_embeddings exemplars/sd14/contrastive_embeddings.pt \
        --contrastive_mode cls \
        --where_mode probe_only \
        --probe_source both \
        --guide_mode hybrid --safety_scale 1.5
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

    def compute(self, eps_prompt, eps_null, eps_target):
        d_prompt = (eps_prompt - eps_null).reshape(1, -1).float()
        d_target = (eps_target - eps_null).reshape(1, -1).float()
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
# Spatial CAS (noise-based WHERE) — for fallback and fusion
# =============================================================================
def compute_spatial_cas(eps_prompt, eps_null, eps_target, neighborhood_size=3):
    d_prompt = (eps_prompt - eps_null).float()
    d_target = (eps_target - eps_null).float()
    H, W = d_prompt.shape[2], d_prompt.shape[3]
    pad = neighborhood_size // 2
    d_prompt_unfolded = F.unfold(d_prompt, kernel_size=neighborhood_size, padding=pad)
    d_target_unfolded = F.unfold(d_target, kernel_size=neighborhood_size, padding=pad)
    spatial_cas = F.cosine_similarity(d_prompt_unfolded, d_target_unfolded, dim=1)
    return spatial_cas.reshape(H, W)


def compute_soft_mask(spatial_cas, spatial_threshold=0.3, sigmoid_alpha=10.0,
                      blur_sigma=1.0, device=None):
    soft_mask = torch.sigmoid(sigmoid_alpha * (spatial_cas - spatial_threshold))
    soft_mask = soft_mask.unsqueeze(0).unsqueeze(0)
    if blur_sigma > 0:
        soft_mask = gaussian_blur_2d(soft_mask, kernel_size=5, sigma=blur_sigma)
    return soft_mask.clamp(0, 1)


def gaussian_blur_2d(x, kernel_size=5, sigma=1.0):
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
# Guidance Application (HOW) — same as v13
# =============================================================================
def apply_guidance(eps_cfg, eps_null, eps_prompt, eps_target, eps_anchor,
                   soft_mask, guide_mode="hybrid", safety_scale=1.0,
                   cfg_scale=7.5, **kwargs):
    mask = soft_mask.to(eps_cfg.dtype)

    if guide_mode == "anchor_inpaint":
        eps_anchor_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
        eps_final = eps_cfg * (1.0 - safety_scale * mask) + eps_anchor_cfg * (safety_scale * mask)

    elif guide_mode == "sld":
        eps_final = eps_cfg - safety_scale * mask * (eps_target - eps_null)

    elif guide_mode == "hybrid":
        t_scale = kwargs.get("target_scale", safety_scale)
        a_scale = kwargs.get("anchor_scale", safety_scale)
        eps_final = eps_cfg \
                    - t_scale * mask * (eps_target - eps_null) \
                    + a_scale * mask * (eps_anchor - eps_null)

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

    elif guide_mode == "dag_adaptive":
        # DAG-inspired: area-based + magnitude-based adaptive scaling
        d_target_dir = eps_target - eps_null
        binary_mask = (mask > 0.5).to(mask.dtype)
        area = binary_mask.sum() / binary_mask.numel()
        area_scale = 5.0 / (mask.shape[-1] * mask.shape[-2])
        area_factor = area_scale * area * mask.numel()
        mag_scale = 1.0 + 4.0 * mask
        correction = safety_scale * area_factor * mag_scale * binary_mask * d_target_dir
        eps_final = eps_cfg - correction

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

    if torch.isnan(eps_final).any() or torch.isinf(eps_final).any():
        eps_final = torch.where(torch.isfinite(eps_final), eps_final, eps_cfg)
    return eps_final


# =============================================================================
# Utils
# =============================================================================
def load_prompts(filepath):
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_safe_filename(prompt, max_len=50):
    safe = "".join(c if c.isalnum() or c in ' -_' else '_' for c in prompt)
    return safe[:max_len].replace(' ', '_')


def encode_concepts(text_encoder, tokenizer, concepts, device):
    all_embeds = []
    for concept in concepts:
        inputs = tokenizer(
            concept, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        embeds = text_encoder(inputs.input_ids.to(device))[0]
        all_embeds.append(embeds)
    return torch.stack(all_embeds).mean(dim=0)


# =============================================================================
# v16-specific: WHERE mode fusion
# =============================================================================
def compute_probe_mask(probe_store, token_indices, attn_resolutions,
                       spatial_threshold, sigmoid_alpha, blur_sigma, device):
    """
    Compute soft spatial mask from cross-attention probe maps.

    Args:
        probe_store: AttentionProbeStore with maps from current step
        token_indices: token positions to aggregate attention over
        attn_resolutions: which resolution groups to use
        spatial_threshold, sigmoid_alpha, blur_sigma: mask parameters
        device: torch device

    Returns:
        soft_mask: [1, 1, 64, 64]
    """
    attn_spatial = compute_attention_spatial_mask(
        probe_store,
        token_indices=token_indices,
        target_resolution=64,
        resolutions_to_use=attn_resolutions,
    )
    return compute_soft_mask(
        attn_spatial.to(device),
        spatial_threshold=spatial_threshold,
        sigmoid_alpha=sigmoid_alpha,
        blur_sigma=blur_sigma,
        device=device,
    )


def compute_where_mask(probe_store, probe_token_indices, eps_prompt, eps_null,
                       eps_target, where_mode, attn_resolutions,
                       spatial_threshold, sigmoid_alpha, blur_sigma,
                       neighborhood_size, fusion_weight, device,
                       probe_source="image",
                       text_probe_store=None, text_token_indices=None):
    """
    Compute spatial WHERE mask using probe attention, noise CAS, or fusion.

    Args:
        probe_store: AttentionProbeStore with maps from current step (image probe)
        probe_token_indices: token indices for image probe attention aggregation
        eps_prompt, eps_null, eps_target: noise predictions for spatial CAS
        where_mode: "probe_only", "noise_only", or "fused"
        attn_resolutions: resolutions for probe attention aggregation
        spatial_threshold, sigmoid_alpha, blur_sigma: mask parameters
        neighborhood_size: kernel size for noise-based spatial CAS
        fusion_weight: weight for probe mask in fused mode (0-1)
        device: torch device
        probe_source: "image", "text", or "both" — which probe key source to use
                      for cross-attention spatial mask (orthogonal to where_mode)
        text_probe_store: AttentionProbeStore with text-probe maps (used when
                          probe_source is "text" or "both")
        text_token_indices: token indices for text probe aggregation

    Returns:
        soft_mask: [1, 1, 64, 64] spatial guidance mask
    """
    def _get_probe_mask():
        """Return soft mask from probe (image/text/both) based on probe_source."""
        if probe_source == "image":
            return compute_probe_mask(
                probe_store, probe_token_indices, attn_resolutions,
                spatial_threshold, sigmoid_alpha, blur_sigma, device,
            )
        elif probe_source == "text":
            return compute_probe_mask(
                text_probe_store, text_token_indices, attn_resolutions,
                spatial_threshold, sigmoid_alpha, blur_sigma, device,
            )
        elif probe_source == "both":
            mask_img = compute_probe_mask(
                probe_store, probe_token_indices, attn_resolutions,
                spatial_threshold, sigmoid_alpha, blur_sigma, device,
            )
            mask_txt = compute_probe_mask(
                text_probe_store, text_token_indices, attn_resolutions,
                spatial_threshold, sigmoid_alpha, blur_sigma, device,
            )
            return torch.max(mask_img, mask_txt)
        else:
            raise ValueError(f"Unknown probe_source: {probe_source}")

    probe_mask = None
    noise_mask = None

    if where_mode in ("probe_only", "fused"):
        probe_mask = _get_probe_mask()

    if where_mode in ("noise_only", "fused"):
        noise_spatial = compute_spatial_cas(
            eps_prompt, eps_null, eps_target,
            neighborhood_size=neighborhood_size,
        )
        noise_mask = compute_soft_mask(
            noise_spatial,
            spatial_threshold=spatial_threshold,
            sigmoid_alpha=sigmoid_alpha,
            blur_sigma=blur_sigma,
            device=device,
        )

    if where_mode == "probe_only":
        return probe_mask
    elif where_mode == "noise_only":
        return noise_mask
    elif where_mode == "fused":
        # Weighted combination of probe and noise masks
        w = fusion_weight
        return w * probe_mask + (1.0 - w) * noise_mask
    else:
        raise ValueError(f"Unknown where_mode: {where_mode}")


# =============================================================================
# Argument parser
# =============================================================================
def parse_args():
    p = ArgumentParser(description="Contrastive Image Direction Safe Generation v16")
    p.add_argument("--ckpt", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--prompts", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--nsamples", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg_scale", type=float, default=7.5)

    # v16-specific: contrastive embeddings
    p.add_argument("--contrastive_embeddings", type=str,
                    default="exemplars/sd14/contrastive_embeddings.pt",
                    help="Path to pre-computed contrastive embeddings (.pt)")
    p.add_argument("--contrastive_mode", type=str, default="cls",
                    choices=["cls", "patch", "mixed"],
                    help="Which contrastive embedding variant to use for probe: "
                         "cls (CLS direction repeated), "
                         "patch (top-K discriminative patches), "
                         "mixed (CLS + top patches)")

    # WHERE mode
    p.add_argument("--where_mode", type=str, default="probe_only",
                    choices=["probe_only", "noise_only", "fused"],
                    help="WHERE source: probe_only (cross-attention), "
                         "noise_only (noise-based spatial CAS), "
                         "fused (weighted combination)")
    p.add_argument("--fusion_weight", type=float, default=0.6,
                    help="Weight for probe mask in fused mode (0=noise only, 1=probe only)")

    # Probe source (orthogonal to where_mode)
    p.add_argument("--probe_source", type=str, default="both",
                    choices=["text", "image", "both"],
                    help="Probe key source for cross-attention spatial mask: "
                         "image (contrastive image direction, original v16), "
                         "text (text keyword probe, v6 style), "
                         "both (union: max of text and image masks)")
    p.add_argument("--target_words", type=str, default="nude,naked,nudity,nsfw,bare,body",
                    help="Comma-separated keywords for text probe (used when "
                         "probe_source is 'text' or 'both')")

    # CAS (When)
    p.add_argument("--cas_threshold", type=float, default=0.6)
    p.add_argument("--cas_sticky", action="store_true", default=True)
    p.add_argument("--cas_no_sticky", action="store_true")

    # WHERE spatial parameters
    p.add_argument("--attn_resolutions", type=int, nargs="+", default=[16, 32])
    p.add_argument("--spatial_threshold", type=float, default=0.3)
    p.add_argument("--sigmoid_alpha", type=float, default=10.0)
    p.add_argument("--neighborhood_size", type=int, default=3)
    p.add_argument("--blur_sigma", type=float, default=1.0)

    # Guidance (How)
    p.add_argument("--guide_mode", type=str, default="hybrid",
                    choices=["anchor_inpaint", "sld", "hybrid", "projection", "hybrid_proj", "dag_adaptive"])
    p.add_argument("--safety_scale", type=float, default=1.5)
    p.add_argument("--target_scale", type=float, default=-1.0)
    p.add_argument("--anchor_scale", type=float, default=-1.0)
    p.add_argument("--proj_scale", type=float, default=1.0)
    p.add_argument("--guide_start_frac", type=float, default=0.0)

    # Concepts (for CAS WHEN + target/anchor UNet passes)
    p.add_argument("--target_concepts", type=str, nargs="+",
                    default=["nudity", "nude person", "naked body"])
    p.add_argument("--anchor_concepts", type=str, nargs="+",
                    default=["clothed person", "person wearing clothes"])

    # v7-style exemplar anchor support
    p.add_argument("--concept_dir_path", type=str, default=None,
                    help="Path to concept_directions.pt for exemplar anchor (optional)")
    p.add_argument("--use_exemplar_anchor", action="store_true",
                    help="Use pre-computed exemplar anchor directions (saves 1 UNet call)")

    # Multi-concept
    p.add_argument("--concept_packs", type=str, nargs="+", default=None,
                    help="Concept pack directories for multi-concept erasing. "
                         "If set, overrides --target_concepts, --anchor_concepts, etc.")
    p.add_argument("--family_level", action="store_true", default=False,
                    help="Apply corrections at family level (per sub-concept) instead of concept level")

    # Misc
    p.add_argument("--save_maps", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)

    args = p.parse_args()
    if args.cas_no_sticky:
        args.cas_sticky = False
    # Validate (skip if using concept_packs)
    if args.concept_packs is None:
        if args.use_exemplar_anchor and args.concept_dir_path is None:
            p.error("--concept_dir_path is required when --use_exemplar_anchor is set")
    return args


# =============================================================================
# Main Generation Loop
# =============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_concept = args.concept_packs is not None

    # Parse target_words into a list
    target_words = [w.strip() for w in args.target_words.split(",") if w.strip()]

    print(f"\n{'='*70}")
    print(f"Contrastive Image Direction Safe Generation v16")
    print(f"{'='*70}")
    print(f"  WHEN:  Global CAS threshold={args.cas_threshold}, sticky={args.cas_sticky}")
    print(f"  WHERE: Contrastive Probe, mode={args.contrastive_mode}, "
          f"where={args.where_mode}"
          + (f", fusion_w={args.fusion_weight}" if args.where_mode == "fused" else ""))
    print(f"         probe_source={args.probe_source}"
          + (f", target_words={target_words}" if args.probe_source in ("text", "both") else ""))
    print(f"         resolutions={args.attn_resolutions}, "
          f"spatial_threshold={args.spatial_threshold}, "
          f"sigmoid_alpha={args.sigmoid_alpha}, blur={args.blur_sigma}")
    print(f"  HOW:   {args.guide_mode}, safety_scale={args.safety_scale}")
    if args.use_exemplar_anchor:
        print(f"  Anchor: exemplar (from {args.concept_dir_path})")
    else:
        print(f"  Anchor: text concepts {args.anchor_concepts}")
    print(f"  Contrastive embeddings: {args.contrastive_embeddings}")
    print(f"  Model: {args.ckpt}")
    print(f"  Steps: {args.steps}, CFG: {args.cfg_scale}, Samples/prompt: {args.nsamples}")
    print(f"{'='*70}\n")

    # ---- Load prompts ----
    all_prompts = load_prompts(args.prompts)
    end_idx = args.end_idx if args.end_idx > 0 else len(all_prompts)
    prompts_with_idx = list(enumerate(all_prompts))[args.start_idx:end_idx]
    print(f"Loaded {len(all_prompts)} prompts, processing [{args.start_idx}:{end_idx}]")

    # ---- Load pipeline ----
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

    # ---- Pre-encode text concept embeddings ----
    with torch.no_grad():
        target_text_embeds = encode_concepts(text_encoder, tokenizer,
                                             args.target_concepts, device)
        anchor_text_embeds = encode_concepts(text_encoder, tokenizer,
                                             args.anchor_concepts, device)
        uncond_inputs = tokenizer(
            "", padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]

    # ==================================================================
    # Multi-concept vs single-concept setup
    # ==================================================================
    concept_packs_list = None
    pack_cas_gates = None
    pack_target_embeds = None
    pack_anchor_embeds = None
    pack_probe_token_indices = None

    # Single-concept state
    contrastive_data = None
    probe_embeds = None
    n_tokens = 0
    anchor_dirs = None
    original_processors = None
    probe_store = AttentionProbeStore()
    probe_token_indices = None
    text_probe_store = None
    text_probe_token_indices = None

    if multi_concept:
        # ---- Multi-concept: load packs ----
        print(f"\nLoading {len(args.concept_packs)} concept packs...")
        concept_packs_list = load_multiple_packs(args.concept_packs, device=device)

        pack_cas_gates = []
        pack_target_embeds = []
        pack_anchor_embeds = []
        pack_probe_token_indices = []

        for i, pack in enumerate(concept_packs_list):
            pack_threshold = pack.get("cas_threshold", args.cas_threshold)
            pack_cas_gates.append(GlobalCAS(threshold=pack_threshold, sticky=args.cas_sticky))

            pack_target_concepts = pack.get("target_concepts", args.target_concepts)
            with torch.no_grad():
                t_emb = encode_concepts(text_encoder, tokenizer, pack_target_concepts, device)
            pack_target_embeds.append(t_emb)

            pack_anchor_concepts = pack.get("anchor_concepts", args.anchor_concepts)
            with torch.no_grad():
                a_emb = encode_concepts(text_encoder, tokenizer, pack_anchor_concepts, device)
            pack_anchor_embeds.append(a_emb)

            pack_target_words = pack.get("target_words", target_words)
            target_text = ", ".join(pack_target_concepts)
            p_indices = find_token_indices(target_text, pack_target_words, tokenizer)
            pack_probe_token_indices.append(p_indices)

            print(f"  Pack [{i}] '{pack.get('concept', 'unknown')}': "
                  f"threshold={pack_threshold}, "
                  f"targets={pack_target_concepts[:3]}{'...' if len(pack_target_concepts) > 3 else ''}, "
                  f"probe_tokens={p_indices[:6]}{'...' if len(p_indices) > 6 else ''}")

        # Shared probe store with combined target words
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

        # Load contrastive embeddings
        contrastive_data = torch.load(args.contrastive_embeddings, map_location=device)
        contrastive_config = contrastive_data["config"]
        n_tokens = contrastive_config["n_tokens"]

        mode_key_map = {
            "cls": "target_embeds_cls",
            "patch": "target_embeds_patch",
            "mixed": "target_embeds_mixed",
        }
        probe_embeds = contrastive_data[mode_key_map[args.contrastive_mode]].to(device=device, dtype=unet.dtype)
        cls_direction = contrastive_data["cls_direction"].to(device=device, dtype=unet.dtype)

        print(f"Loaded contrastive embeddings: mode={args.contrastive_mode}, "
              f"probe_embeds={probe_embeds.shape}")
        print(f"  Contrastive config: n_tokens={n_tokens}, top_k={contrastive_config['top_k']}, "
              f"cosine_sim={contrastive_data['cosine_sim']:.4f}")

        # Load exemplar anchor directions (optional)
        if args.use_exemplar_anchor:
            concept_data = torch.load(args.concept_dir_path, map_location=device)
            anchor_dirs = concept_data['anchor_directions']
            print(f"Loaded exemplar anchor directions for {len(anchor_dirs)} timesteps")

        # Setup cross-attention probing
        probe_token_indices = list(range(1, 1 + n_tokens))
        target_keys = precompute_target_keys(unet, probe_embeds.to(dtype=next(unet.parameters()).dtype), args.attn_resolutions)

        if args.probe_source in ("text", "both"):
            with torch.no_grad():
                word_embeds = []
                for w in target_words:
                    inp = tokenizer(
                        w, padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt",
                    )
                    word_embeds.append(text_encoder(inp.input_ids.to(device))[0])
                text_probe_embeds = torch.stack(word_embeds).mean(dim=0)

            text_target_keys = precompute_target_keys(
                unet, text_probe_embeds, args.attn_resolutions
            )
            text_probe_store = AttentionProbeStore()
            text_probe_token_indices = list(range(1, min(1 + len(target_words) * 3, 20)))

            original_processors = register_dual_attention_probe(
                unet, probe_store, text_probe_store,
                target_keys, text_target_keys, args.attn_resolutions
            )
            print(f"  Contrastive probe (image): token_indices={probe_token_indices}")
            print(f"  Text probe: target_words={target_words}, "
                  f"token_indices={text_probe_token_indices}")
        else:
            original_processors = register_attention_probe(
                unet, probe_store, target_keys, args.attn_resolutions
            )
            print(f"  Contrastive probe (image): token_indices={probe_token_indices}")

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

                lat_in = scheduler.scale_model_input(latents, t)

                # ==== Pass 1+2: CFG (with probe active) ====
                probe_store.active = True
                probe_store.reset()
                if text_probe_store is not None:
                    text_probe_store.active = True
                    text_probe_store.reset()

                with torch.no_grad():
                    lat_batch = torch.cat([lat_in, lat_in])
                    embed_batch = torch.cat([uncond_embeds, prompt_embeds])
                    raw = unet(lat_batch, t, encoder_hidden_states=embed_batch).sample
                    eps_null, eps_prompt_pred = raw.chunk(2)

                probe_store.active = False
                if text_probe_store is not None:
                    text_probe_store.active = False

                # ==== Pass 3: target concept (for CAS WHEN) ====
                with torch.no_grad():
                    eps_target = unet(lat_in, t,
                                      encoder_hidden_states=target_text_embeds).sample

                # Standard CFG
                eps_cfg = eps_null + args.cfg_scale * (eps_prompt_pred - eps_null)

                # =========================================================
                # WHEN/WHERE/HOW — Multi-concept or single-concept
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

                        p_guide_mode = pack.get("guide_mode", args.guide_mode)
                        p_safety_scale = pack.get("safety_scale", args.safety_scale)

                        # Target direction for this pack (text mode)
                        with torch.no_grad():
                            eps_target_pack = unet(lat_in, t,
                                                   encoder_hidden_states=p_target_embeds).sample

                        cas_val_pack, should_trigger_pack = pack_cas_gate.compute(
                            eps_prompt_pred, eps_null, eps_target_pack
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

                        # WHERE: compute per-pack mask from cross-attention probe
                        attn_spatial_pack = compute_attention_spatial_mask(
                            probe_store,
                            token_indices=p_probe_indices,
                            target_resolution=64,
                            resolutions_to_use=args.attn_resolutions,
                        )
                        soft_mask_pack = compute_soft_mask(
                            attn_spatial_pack.to(device),
                            spatial_threshold=args.spatial_threshold,
                            sigmoid_alpha=args.sigmoid_alpha,
                            blur_sigma=args.blur_sigma,
                            device=device,
                        )

                        # HOW: compute per-pack correction
                        eps_guided_pack = apply_guidance(
                            eps_cfg=eps_cfg,
                            eps_null=eps_null,
                            eps_prompt=eps_prompt_pred,
                            eps_target=eps_target_pack,
                            eps_anchor=eps_anchor_pack,
                            soft_mask=soft_mask_pack,
                            guide_mode=p_guide_mode,
                            safety_scale=p_safety_scale,
                            cfg_scale=args.cfg_scale,
                            target_scale=args.target_scale if args.target_scale > 0 else p_safety_scale,
                            anchor_scale=args.anchor_scale if args.anchor_scale > 0 else p_safety_scale,
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
                    else:
                        eps_final = eps_cfg

                else:
                    # ===================================================
                    # SINGLE-CONCEPT: original v16 behavior
                    # ===================================================

                    # ==== Global CAS (WHEN) ====
                    cas_val, should_trigger = cas.compute(eps_prompt_pred, eps_null, eps_target)
                    cas_values.append(cas_val)

                    in_window = step_idx >= guide_start_step
                    should_guide = should_trigger and in_window

                    if should_guide:
                        # ==== Pass 4: anchor concept ====
                        if args.use_exemplar_anchor and anchor_dirs is not None:
                            t_int = t.item()
                            d_anchor = anchor_dirs[t_int].to(
                                device, dtype=torch.float16
                            ).unsqueeze(0)
                            eps_anchor = eps_null + d_anchor
                        else:
                            with torch.no_grad():
                                eps_anchor = unet(lat_in, t,
                                                  encoder_hidden_states=anchor_text_embeds).sample

                        # ==== WHERE: Compute spatial mask ====
                        soft_mask = compute_where_mask(
                            probe_store=probe_store,
                            probe_token_indices=probe_token_indices,
                            eps_prompt=eps_prompt_pred,
                            eps_null=eps_null,
                            eps_target=eps_target,
                            where_mode=args.where_mode,
                            attn_resolutions=args.attn_resolutions,
                            spatial_threshold=args.spatial_threshold,
                            sigmoid_alpha=args.sigmoid_alpha,
                            blur_sigma=args.blur_sigma,
                            neighborhood_size=args.neighborhood_size,
                            fusion_weight=args.fusion_weight,
                            device=device,
                            probe_source=args.probe_source,
                            text_probe_store=text_probe_store,
                            text_token_indices=text_probe_token_indices,
                        )

                        # ==== HOW: Apply guidance ====
                        eps_final = apply_guidance(
                            eps_cfg=eps_cfg,
                            eps_null=eps_null,
                            eps_prompt=eps_prompt_pred,
                            eps_target=eps_target,
                            eps_anchor=eps_anchor,
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

                        # Save maps
                        if args.save_maps and step_idx % 10 == 0:
                            pfx = f"{prompt_idx:04d}_{sample_idx:02d}_s{step_idx:03d}"
                            if args.where_mode != "noise_only":
                                if args.probe_source in ("image", "both"):
                                    attn_spatial = compute_attention_spatial_mask(
                                        probe_store,
                                        token_indices=probe_token_indices,
                                        target_resolution=64,
                                        resolutions_to_use=args.attn_resolutions,
                                    )
                                    attn_map_np = np.nan_to_num(
                                        attn_spatial.float().cpu().numpy(), nan=0.0)
                                    attn_map_img = (np.clip(attn_map_np, 0, 1) * 255).astype(np.uint8)
                                    Image.fromarray(attn_map_img, 'L').save(
                                        str(outdir / "maps" / f"{pfx}_attn_image.png"))
                                if args.probe_source in ("text", "both"):
                                    attn_spatial_txt = compute_attention_spatial_mask(
                                        text_probe_store,
                                        token_indices=text_probe_token_indices,
                                        target_resolution=64,
                                        resolutions_to_use=args.attn_resolutions,
                                    )
                                    attn_txt_np = np.nan_to_num(
                                        attn_spatial_txt.float().cpu().numpy(), nan=0.0)
                                    attn_txt_img = (np.clip(attn_txt_np, 0, 1) * 255).astype(np.uint8)
                                    Image.fromarray(attn_txt_img, 'L').save(
                                        str(outdir / "maps" / f"{pfx}_attn_text.png"))
                            mask_np = soft_mask[0, 0].float().cpu().numpy()
                            mask_img = (np.clip(mask_np, 0, 1) * 255).astype(np.uint8)
                            Image.fromarray(mask_img, 'L').save(
                                str(outdir / "maps" / f"{pfx}_mask.png"))

                    else:
                        eps_final = eps_cfg

                # ==== DDIM step ====
                latents_prev = latents.clone()
                latents = scheduler.step(eps_final, t, latents).prev_sample

                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    print(f"  WARNING: NaN/Inf at step {step_idx}, reverting")
                    eps_fallback = eps_null + args.cfg_scale * (eps_prompt_pred - eps_null)
                    latents = scheduler.step(eps_fallback, t, latents_prev).prev_sample

                if args.debug and step_idx % 10 == 0:
                    status = "GUIDED" if should_guide else ("CAS_ON" if should_trigger else "skip")
                    area_s = f" area={mask_areas[-1]:.3f}" if should_guide and mask_areas else ""
                    print(f"  [{step_idx:02d}] t={t.item():.0f} CAS={cas_val:.3f} {status}{area_s}")

            # ---- Decode ----
            with torch.no_grad():
                decoded = vae.decode(latents.to(vae.dtype) / vae.config.scaling_factor).sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                img_np = (decoded[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

            slug = make_safe_filename(prompt)
            fname = f"{prompt_idx:04d}_{sample_idx:02d}_{slug}.png"
            Image.fromarray(img_np).resize((512, 512)).save(str(outdir / fname))

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
                "contrastive_mode": args.contrastive_mode,
                "where_mode": args.where_mode,
                "probe_source": args.probe_source,
            }
            all_stats.append(stats)

            if prompt_idx % 10 == 0 or args.debug:
                tqdm.write(
                    f"  [{prompt_idx:03d}_{sample_idx}] guided={guided_count}/{total_steps} "
                    f"CAS avg={stats['avg_cas']:.3f} max={stats['max_cas']:.3f} "
                    f"mask_area={stats['avg_mask_area']:.3f} "
                    f"mode={args.contrastive_mode} where={args.where_mode} "
                    f"probe={args.probe_source}"
                )

    # ---- Cleanup ----
    restore_processors(unet, original_processors)

    # ---- Save stats ----
    n = len(all_stats)
    n_trig = sum(1 for s in all_stats if s["guided_steps"] > 0)
    summary = {
        "method": "Contrastive Image Direction Safe Generation v16",
        "contrastive_mode": args.contrastive_mode,
        "where_mode": args.where_mode,
        "probe_source": args.probe_source,
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
    print(f"DONE! {n} images, {n_trig} triggered ({100*n_trig/max(n,1):.0f}%)")
    print(f"  Contrastive mode: {args.contrastive_mode}")
    print(f"  WHERE mode: {args.where_mode}"
          + (f" (fusion_w={args.fusion_weight})" if args.where_mode == "fused" else ""))
    print(f"  Probe source: {args.probe_source}"
          + (f" (target_words={target_words})" if args.probe_source in ("text", "both") else ""))
    print(f"  Guide mode: {args.guide_mode}, safety_scale={args.safety_scale}")
    print(f"  Anchor: {'exemplar' if args.use_exemplar_anchor else 'text'}")
    print(f"  Avg guided steps: {summary['overall']['avg_guided_steps']:.1f}/{total_steps if n else 0}")
    print(f"  Avg mask area: {summary['overall']['avg_mask_area']:.3f}")
    print(f"  Output: {outdir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
