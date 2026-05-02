#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exemplar Cross-Attention Probe + CAS Safe Generation v13

Architecture:
  WHEN (Global CAS): cos(d_prompt, d_target) > tau (online, same as v6)
  WHERE (Exemplar Cross-Attention Probe): probe UNet hidden states with
        CLIP image exemplar K (instead of text "nude person" K)
        during eps_prompt forward pass -> spatial mask
  HOW: Same guidance modes as v6 (hybrid recommended)

Key innovations vs v6:
  - WHERE probe uses CLIP image embeddings from actual nude exemplar images
    instead of text "nude person" — richer, more grounded concept representation
  - CLIP ViT-L/14 class token (same model as SD1.4) projected into text space
  - Addresses DAG's background leakage problem training-free
  - Supports multiple probe sources: clip_exemplar, text (v6 fallback), both (union)

Key innovations vs DAG:
  - Completely training-free (no token optimization needed)
  - Uses exemplar images as information bottleneck (Paint by Example insight)

Usage:
    # 1. First prepare CLIP embeddings:
    CUDA_VISIBLE_DEVICES=0 python prepare_clip_exemplar.py

    # 2. Then generate:
    CUDA_VISIBLE_DEVICES=0 python generate_v13.py \
        --prompts prompts/ringabell_anchor_subset.csv \
        --outdir outputs/v13/ringabell_clip_hybrid \
        --clip_embeddings exemplars/sd14/clip_exemplar_embeddings.pt \
        --probe_source clip_exemplar \
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
    restore_processors,
    compute_attention_spatial_mask,
    find_token_indices,
)


# =============================================================================
# Global CAS: Concept Alignment Score (WHEN) — same as v6
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
# Spatial CAS (noise-based WHERE fallback) — same as v6
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
    coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
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
# Guidance Application (HOW) — same as v6
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
# Utils — same as v6
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
# v13-specific: Exemplar probe setup
# =============================================================================
def setup_exemplar_probe(unet, clip_embeds, text_embeds, probe_source,
                         attn_resolutions, device):
    """
    Setup cross-attention probing with exemplar or text embeddings.

    Args:
        unet: UNet model
        clip_embeds: [1, 77, 768] CLIP image exemplar embeddings (projected)
        text_embeds: [1, 77, 768] text concept embeddings ("nude person")
        probe_source: "clip_exemplar", "text", "both", "diff"
        attn_resolutions: list of resolutions to probe [16, 32]
        device: torch device

    Returns:
        probe_stores: dict of {name: (store, original_processors, token_indices)}
    """
    results = {}

    if probe_source in ("clip_exemplar", "both", "diff"):
        # CLIP exemplar probe
        clip_store = AttentionProbeStore()
        clip_keys = precompute_target_keys(unet, clip_embeds.to(device), attn_resolutions)
        clip_orig = register_attention_probe(unet, clip_store, clip_keys, attn_resolutions)
        # For CLIP exemplar, use tokens 1-4 (the concept tokens we placed)
        clip_token_indices = list(range(1, 5))
        results["clip"] = {
            "store": clip_store,
            "original_processors": clip_orig,
            "token_indices": clip_token_indices,
        }

    if probe_source in ("text", "both", "diff"):
        # Need to restore original processors first if clip was installed
        if "clip" in results:
            restore_processors(unet, results["clip"]["original_processors"])

        text_store = AttentionProbeStore()
        text_keys = precompute_target_keys(unet, text_embeds.to(device), attn_resolutions)
        text_orig = register_attention_probe(unet, text_store, text_keys, attn_resolutions)
        results["text"] = {
            "store": text_store,
            "original_processors": text_orig,
            "token_indices": None,  # Will be set per-prompt
        }

    return results


def setup_dual_probe(unet, clip_embeds, text_embeds, attn_resolutions, device):
    """
    For 'both' and 'diff' modes: we need to run TWO probes in one forward pass.
    We achieve this by concatenating clip and text embeddings and computing
    K for both simultaneously in a single probe processor.

    Alternative: Run probe sequentially (one per forward pass) — simpler but 2x cost.
    We go with sequential for clarity: probe_clip on eps_prompt forward,
    then use text-based probe as anchor reference.
    """
    # Primary probe: CLIP exemplar
    clip_store = AttentionProbeStore()
    clip_keys = precompute_target_keys(unet, clip_embeds.to(device), attn_resolutions)
    orig_procs = register_attention_probe(unet, clip_store, clip_keys, attn_resolutions)
    clip_token_indices = list(range(1, 5))

    return {
        "clip_store": clip_store,
        "original_processors": orig_procs,
        "clip_token_indices": clip_token_indices,
    }


# =============================================================================
# Argument parser
# =============================================================================
def parse_args():
    p = ArgumentParser(description="Exemplar Cross-Attention Probe + CAS Safe Generation v13")
    p.add_argument("--ckpt", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--prompts", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--nsamples", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg_scale", type=float, default=7.5)

    # v13-specific: CLIP exemplar embeddings
    p.add_argument("--clip_embeddings", type=str,
                    default="exemplars/sd14/clip_exemplar_embeddings.pt",
                    help="Path to pre-computed CLIP exemplar embeddings (.pt)")
    p.add_argument("--probe_source", type=str, default="clip_exemplar",
                    choices=["clip_exemplar", "text", "both", "diff"],
                    help="Source for cross-attention probe: "
                         "clip_exemplar (CLIP image features), "
                         "text (v6 fallback), "
                         "both (union of clip+text masks), "
                         "diff (clip mask minus text anchor mask)")

    # CAS (When)
    p.add_argument("--cas_threshold", type=float, default=0.6)
    p.add_argument("--cas_sticky", action="store_true", default=True)
    p.add_argument("--cas_no_sticky", action="store_true")

    # WHERE
    p.add_argument("--attn_resolutions", type=int, nargs="+", default=[16, 32])
    p.add_argument("--target_words", type=str, nargs="+",
                    default=["nude", "naked", "nudity", "nsfw", "bare", "body"])
    p.add_argument("--spatial_threshold", type=float, default=0.3)
    p.add_argument("--sigmoid_alpha", type=float, default=10.0)
    p.add_argument("--neighborhood_size", type=int, default=3)
    p.add_argument("--blur_sigma", type=float, default=1.0)

    # Noise-based WHERE fallback
    p.add_argument("--where_fallback", type=str, default="none",
                    choices=["none", "noise", "noise_union"],
                    help="Fallback WHERE: none (crossattn only), "
                         "noise (noise-based CAS), noise_union (union of both)")

    # Guidance (How)
    p.add_argument("--guide_mode", type=str, default="hybrid",
                    choices=["anchor_inpaint", "sld", "hybrid", "projection", "hybrid_proj"])
    p.add_argument("--safety_scale", type=float, default=1.5)
    p.add_argument("--target_scale", type=float, default=-1.0)
    p.add_argument("--anchor_scale", type=float, default=-1.0)
    p.add_argument("--proj_scale", type=float, default=1.0)
    p.add_argument("--guide_start_frac", type=float, default=0.0)

    # Concepts (for CAS WHEN + text probe fallback)
    p.add_argument("--target_concepts", type=str, nargs="+",
                    default=["nudity", "nude person", "naked body"])
    p.add_argument("--anchor_concepts", type=str, nargs="+",
                    default=["clothed person", "person wearing clothes"])

    # Misc
    p.add_argument("--save_maps", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)

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
    print(f"Exemplar Cross-Attention Probe + CAS Safe Generation v13")
    print(f"{'='*70}")
    print(f"  WHEN:  Global CAS threshold={args.cas_threshold}, sticky={args.cas_sticky}")
    print(f"  WHERE: Exemplar CrossAttn Probe, source={args.probe_source}, "
          f"resolutions={args.attn_resolutions}")
    print(f"         spatial_threshold={args.spatial_threshold}, "
          f"sigmoid_alpha={args.sigmoid_alpha}, blur={args.blur_sigma}")
    print(f"         fallback={args.where_fallback}")
    print(f"  HOW:   {args.guide_mode}, safety_scale={args.safety_scale}")
    print(f"  CLIP embeddings: {args.clip_embeddings}")
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

    # ---- Load CLIP exemplar embeddings ----
    clip_data = None
    clip_target_embeds = None
    clip_anchor_embeds = None

    if args.probe_source != "text":
        clip_data = torch.load(args.clip_embeddings, map_location=device)
        clip_target_embeds = clip_data["target_clip_embeds"].float().to(device)  # [1, 77, 768]
        clip_anchor_embeds = clip_data["anchor_clip_embeds"].float().to(device)
        print(f"Loaded CLIP exemplar embeddings: target={clip_target_embeds.shape}, "
              f"anchor={clip_anchor_embeds.shape}")
        print(f"  Projection method: {clip_data['config']['projection']}")
        print(f"  N_tokens: {clip_data['config']['n_tokens']}")

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

    # ---- Setup cross-attention probing ----
    probe_store = AttentionProbeStore()
    probe_token_indices = None

    if args.probe_source == "clip_exemplar":
        # Use CLIP exemplar embeddings for K_target
        target_keys = precompute_target_keys(unet, clip_target_embeds, args.attn_resolutions)
        original_processors = register_attention_probe(
            unet, probe_store, target_keys, args.attn_resolutions
        )
        # CLIP embeddings: concept tokens at positions 1..n_tokens
        n_tokens = clip_data["config"]["n_tokens"] if clip_data else 4
        probe_token_indices = list(range(1, 1 + n_tokens))
        print(f"  CLIP exemplar probe: token_indices={probe_token_indices}")

    elif args.probe_source == "text":
        # v6 style: text embeddings
        target_keys = precompute_target_keys(unet, target_text_embeds, args.attn_resolutions)
        original_processors = register_attention_probe(
            unet, probe_store, target_keys, args.attn_resolutions
        )
        target_text = ", ".join(args.target_concepts)
        probe_token_indices = find_token_indices(target_text, args.target_words, tokenizer)
        print(f"  Text probe: token_indices={probe_token_indices}")

    elif args.probe_source in ("both", "diff"):
        # Primary: CLIP exemplar, anchor reference: text
        # We'll compute text-based mask separately using noise-based spatial CAS
        target_keys = precompute_target_keys(unet, clip_target_embeds, args.attn_resolutions)
        original_processors = register_attention_probe(
            unet, probe_store, target_keys, args.attn_resolutions
        )
        n_tokens = clip_data["config"]["n_tokens"] if clip_data else 4
        probe_token_indices = list(range(1, 1 + n_tokens))
        print(f"  Dual probe (CLIP primary): token_indices={probe_token_indices}")

    # ---- Also setup anchor probe for 'diff' mode ----
    anchor_probe_store = None
    anchor_probe_token_indices = None
    if args.probe_source == "diff" and clip_data is not None:
        # We'll need a separate forward pass with anchor keys for the diff mask
        # For efficiency, we compute anchor attention mask using clothed CLIP embeddings
        # in a second probe pass (only when guidance is active)
        anchor_probe_store = AttentionProbeStore()
        anchor_probe_token_indices = list(range(1, 1 + (clip_data["config"]["n_tokens"])))

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

                with torch.no_grad():
                    lat_batch = torch.cat([lat_in, lat_in])
                    embed_batch = torch.cat([uncond_embeds, prompt_embeds])
                    raw = unet(lat_batch, t, encoder_hidden_states=embed_batch).sample
                    eps_null, eps_prompt = raw.chunk(2)

                probe_store.active = False

                # ==== Pass 3: target concept (for CAS WHEN) ====
                with torch.no_grad():
                    eps_target = unet(lat_in, t,
                                      encoder_hidden_states=target_text_embeds).sample

                # Standard CFG
                eps_cfg = eps_null + args.cfg_scale * (eps_prompt - eps_null)

                # ==== Global CAS (WHEN) ====
                cas_val, should_trigger = cas.compute(eps_prompt, eps_null, eps_target)
                cas_values.append(cas_val)

                in_window = step_idx >= guide_start_step
                should_guide = should_trigger and in_window

                if should_guide:
                    # ==== Pass 4: anchor concept ====
                    with torch.no_grad():
                        eps_anchor = unet(lat_in, t,
                                          encoder_hidden_states=anchor_text_embeds).sample

                    # ==== WHERE: Cross-attention spatial mask ====
                    # Primary mask from exemplar probe
                    attn_spatial = compute_attention_spatial_mask(
                        probe_store,
                        token_indices=probe_token_indices,
                        target_resolution=64,
                        resolutions_to_use=args.attn_resolutions,
                    )

                    if args.probe_source == "diff" and clip_data is not None:
                        # Compute anchor mask: temporarily install anchor probe
                        # Use noise-based approach for anchor (cheaper)
                        anchor_spatial = compute_spatial_cas(
                            eps_prompt, eps_null, eps_anchor,
                            neighborhood_size=args.neighborhood_size,
                        )
                        # Diff: target attention - anchor attention (like v3/DAG)
                        anchor_soft = torch.sigmoid(
                            args.sigmoid_alpha * (anchor_spatial - args.spatial_threshold)
                        )
                        target_soft = torch.sigmoid(
                            args.sigmoid_alpha * (attn_spatial.to(device) - args.spatial_threshold)
                        )
                        diff_mask = (target_soft - anchor_soft).clamp(0, 1)
                        soft_mask = diff_mask.unsqueeze(0).unsqueeze(0)
                        if args.blur_sigma > 0:
                            soft_mask = gaussian_blur_2d(soft_mask, kernel_size=5,
                                                         sigma=args.blur_sigma)
                        soft_mask = soft_mask.clamp(0, 1)

                    elif args.probe_source == "both":
                        # Union: combine CLIP probe mask with noise-based mask
                        noise_spatial = compute_spatial_cas(
                            eps_prompt, eps_null, eps_target,
                            neighborhood_size=args.neighborhood_size,
                        )
                        clip_mask = compute_soft_mask(
                            attn_spatial.to(device),
                            spatial_threshold=args.spatial_threshold,
                            sigmoid_alpha=args.sigmoid_alpha,
                            blur_sigma=args.blur_sigma,
                            device=device,
                        )
                        noise_mask = compute_soft_mask(
                            noise_spatial,
                            spatial_threshold=args.spatial_threshold,
                            sigmoid_alpha=args.sigmoid_alpha,
                            blur_sigma=args.blur_sigma,
                            device=device,
                        )
                        # Union: max of both masks
                        soft_mask = torch.max(clip_mask, noise_mask)

                    else:
                        # Default: clip_exemplar or text — just use probe mask
                        soft_mask = compute_soft_mask(
                            attn_spatial.to(device),
                            spatial_threshold=args.spatial_threshold,
                            sigmoid_alpha=args.sigmoid_alpha,
                            blur_sigma=args.blur_sigma,
                            device=device,
                        )

                    # Optional noise fallback union
                    if args.where_fallback == "noise_union":
                        noise_spatial = compute_spatial_cas(
                            eps_prompt, eps_null, eps_target,
                            neighborhood_size=args.neighborhood_size,
                        )
                        noise_mask = compute_soft_mask(
                            noise_spatial,
                            spatial_threshold=args.spatial_threshold,
                            sigmoid_alpha=args.sigmoid_alpha,
                            blur_sigma=args.blur_sigma,
                            device=device,
                        )
                        soft_mask = torch.max(soft_mask, noise_mask)
                    elif args.where_fallback == "noise":
                        noise_spatial = compute_spatial_cas(
                            eps_prompt, eps_null, eps_target,
                            neighborhood_size=args.neighborhood_size,
                        )
                        soft_mask = compute_soft_mask(
                            noise_spatial,
                            spatial_threshold=args.spatial_threshold,
                            sigmoid_alpha=args.sigmoid_alpha,
                            blur_sigma=args.blur_sigma,
                            device=device,
                        )

                    # ==== HOW: Apply guidance ====
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
                        target_scale=args.target_scale if args.target_scale > 0 else args.safety_scale,
                        anchor_scale=args.anchor_scale if args.anchor_scale > 0 else args.safety_scale,
                        proj_scale=args.proj_scale,
                    )

                    guided_count += 1
                    area_val = float(soft_mask.mean().item())
                    mask_areas.append(area_val)

                    # Save maps
                    if args.save_maps and step_idx % 10 == 0:
                        attn_map_np = attn_spatial.float().cpu().numpy()
                        attn_map_np = np.nan_to_num(attn_map_np, nan=0.0)
                        attn_map_img = (np.clip(attn_map_np, 0, 1) * 255).astype(np.uint8)
                        Image.fromarray(attn_map_img, 'L').save(
                            str(outdir / "maps" / f"{prompt_idx:04d}_{sample_idx:02d}_s{step_idx:03d}_attn.png"))
                        mask_np = soft_mask[0, 0].float().cpu().numpy()
                        mask_img = (np.clip(mask_np, 0, 1) * 255).astype(np.uint8)
                        Image.fromarray(mask_img, 'L').save(
                            str(outdir / "maps" / f"{prompt_idx:04d}_{sample_idx:02d}_s{step_idx:03d}_mask.png"))

                else:
                    eps_final = eps_cfg

                # ==== DDIM step ====
                latents_prev = latents.clone()
                latents = scheduler.step(eps_final, t, latents).prev_sample

                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    print(f"  WARNING: NaN/Inf at step {step_idx}, reverting")
                    eps_fallback = eps_null + args.cfg_scale * (eps_prompt - eps_null)
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
                "probe_source": args.probe_source,
            }
            all_stats.append(stats)

            if prompt_idx % 10 == 0 or args.debug:
                tqdm.write(
                    f"  [{prompt_idx:03d}_{sample_idx}] guided={guided_count}/{total_steps} "
                    f"CAS avg={stats['avg_cas']:.3f} max={stats['max_cas']:.3f} "
                    f"mask_area={stats['avg_mask_area']:.3f} probe={args.probe_source}"
                )

    # ---- Cleanup ----
    restore_processors(unet, original_processors)

    # ---- Save stats ----
    n = len(all_stats)
    n_trig = sum(1 for s in all_stats if s["guided_steps"] > 0)
    summary = {
        "method": "Exemplar Cross-Attention Probe + CAS v13",
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
    print(f"  Probe source: {args.probe_source}")
    print(f"  Guide mode: {args.guide_mode}, safety_scale={args.safety_scale}")
    print(f"  Avg guided steps: {summary['overall']['avg_guided_steps']:.1f}/{total_steps if n else 0}")
    print(f"  Avg mask area: {summary['overall']['avg_mask_area']:.3f}")
    print(f"  Output: {outdir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
