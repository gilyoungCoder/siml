#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Exemplar Diverse Probe Ensemble v19: Training-Free Safe Generation

Architecture:
  WHEN (Global CAS): cos(d_prompt, d_target) > tau (online, same as v13)
  WHERE (Multi-Exemplar Cross-Attention Probe): Instead of averaging all 16
        exemplar CLIP embeddings into one collapsed vector (v13), maintain
        individual exemplar embeddings as separate probe tokens. Each exemplar
        image captures a different nudity configuration (front/back/side,
        male/female). The probe's max-across-tokens naturally computes their
        union, giving broader spatial coverage with focused individual probes.
  HOW: Same guidance modes as v13 (hybrid recommended)

Key innovations vs v13:
  - v13 averages N exemplar CLIP features into 1 vector -> information collapse
  - v19 keeps N exemplars as N separate tokens in a single 77-token sequence:
    [BOS, ex1, ex2, ..., exN, EOS, PAD...]
  - attention_probe.py's max-across-tokens naturally unions per-exemplar probes
  - Supports exemplar selection strategies (all, top_k, diverse, random_k)
  - Supports aggregation modes (union, intersection, mean, threshold_vote)

Usage:
    # 1. First prepare CLIP embeddings (same as v13):
    CUDA_VISIBLE_DEVICES=0 python prepare_clip_exemplar.py

    # 2. Then generate:
    CUDA_VISIBLE_DEVICES=0 python generate_v19.py \
        --prompts prompts/ringabell_anchor_subset.csv \
        --outdir outputs/v19/ringabell_multi_exemplar \
        --clip_embeddings exemplars/sd14/clip_exemplar_embeddings.pt \
        --num_exemplars 16 --exemplar_selection all \
        --probe_aggregation union \
        --where_mode multi_probe \
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
from concept_pack_loader import load_concept_pack, load_multiple_packs, get_combined_target_words


# =============================================================================
# Global CAS: Concept Alignment Score (WHEN) -- same as v13
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
# Spatial CAS (noise-based WHERE fallback) -- same as v13
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
# Guidance Application (HOW) -- same as v13
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
# Utils -- same as v13
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
# v19-specific: Multi-Exemplar Embedding Construction
# =============================================================================
def select_exemplars(features, anchor_features=None, num_exemplars=16,
                     selection="all"):
    """
    Select a subset of exemplar CLIP features based on strategy.

    Args:
        features: [N, 768] individual CLIP image features (nudity exemplars)
        anchor_features: [M, 768] anchor CLIP features (clothed) for top_k
        num_exemplars: max number to select
        selection: strategy -- "all", "top_k", "diverse", "random_k"

    Returns:
        selected: [K, 768] selected features, K <= min(num_exemplars, N, 75)
    """
    N = features.shape[0]
    max_k = min(num_exemplars, N, 75)  # max 75 tokens (77 - BOS - EOS)

    if selection == "all":
        return features[:max_k]

    elif selection == "top_k":
        # Select K exemplars most different from anchor (most discriminative)
        if anchor_features is None:
            return features[:max_k]
        anchor_mean = anchor_features.mean(dim=0, keepdim=True)  # [1, 768]
        # Cosine distance from anchor centroid (lower cos = more different)
        cos_sim = F.cosine_similarity(features, anchor_mean.expand(N, -1), dim=-1)
        # Sort ascending: least similar to anchor first (most discriminative)
        indices = cos_sim.argsort()[:max_k]
        return features[indices]

    elif selection == "diverse":
        # Greedy farthest-point sampling for maximum diversity
        selected_idx = [0]
        remaining = set(range(1, N))

        while len(selected_idx) < max_k and remaining:
            selected_feats = features[selected_idx]  # [K_cur, 768]
            best_idx = -1
            best_min_dist = -1.0

            for idx in remaining:
                # Min cosine sim to any already-selected exemplar
                cos_sims = F.cosine_similarity(
                    features[idx:idx+1].expand(len(selected_idx), -1),
                    selected_feats, dim=-1
                )
                min_sim = cos_sims.min().item()
                dist = 1.0 - min_sim  # convert to distance; larger = farther
                if dist > best_min_dist or best_idx == -1:
                    best_min_dist = dist
                    best_idx = idx

            selected_idx.append(best_idx)
            remaining.discard(best_idx)

        return features[selected_idx]

    elif selection == "random_k":
        # Random subset for ablation
        perm = torch.randperm(N)[:max_k]
        return features[perm]

    else:
        raise ValueError(f"Unknown exemplar_selection: {selection}")


def build_multi_exemplar_embedding(selected_features, text_encoder, tokenizer,
                                   device):
    """
    Build a 77-token embedding with individual exemplar features as tokens.

    Format: [BOS, ex1, ex2, ..., exN, EOS, PAD, PAD, ...]
    Each exemplar gets its own token position. The probe's max-across-tokens
    naturally computes the union of per-exemplar attention maps.

    Args:
        selected_features: [K, 768] selected CLIP exemplar features (normalized)
        text_encoder: SD's CLIPTextModel (for BOS/EOS/PAD embeddings)
        tokenizer: SD's CLIPTokenizer
        device: torch device

    Returns:
        embeds: [1, 77, 768] multi-exemplar token sequence
        probe_token_indices: list of token positions [1, 2, ..., K]
    """
    K = selected_features.shape[0]
    token_embedding = text_encoder.text_model.embeddings.token_embedding

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    with torch.no_grad():
        bos_embed = token_embedding(torch.tensor([bos_id], device=device))  # [1, 768]
        eos_embed = token_embedding(torch.tensor([eos_id], device=device))
        pad_embed = token_embedding(torch.tensor([pad_id], device=device))

    # Build: [BOS, ex_1, ex_2, ..., ex_K, EOS, PAD...]
    tokens_list = [bos_embed]
    for i in range(K):
        feat = selected_features[i:i+1].to(device)  # [1, 768]
        tokens_list.append(feat)
    tokens_list.append(eos_embed)

    # Pad to 77
    n_pad = 77 - len(tokens_list)
    for _ in range(n_pad):
        tokens_list.append(pad_embed)

    token_embeds = torch.cat(tokens_list, dim=0).unsqueeze(0)  # [1, 77, 768]

    # Probe tokens: positions 1..K (skip BOS at 0)
    probe_token_indices = list(range(1, 1 + K))

    return token_embeds, probe_token_indices


# =============================================================================
# v19-specific: Multi-Exemplar Aggregated Mask Computation
# =============================================================================
def compute_multi_exemplar_mask(probe_store, probe_token_indices,
                                num_exemplars, aggregation="union",
                                target_resolution=64,
                                resolutions_to_use=[16, 32],
                                vote_threshold=0.5):
    """
    Compute spatial mask from multi-exemplar probe with configurable aggregation.

    For 'union' mode, this is identical to compute_attention_spatial_mask with
    all exemplar token indices (the max-across-tokens already does union).
    For other modes, we compute per-exemplar maps then aggregate differently.

    Args:
        probe_store: AttentionProbeStore with maps from current step
        probe_token_indices: list of token positions [1, 2, ..., K]
        num_exemplars: number of exemplar tokens
        aggregation: "union", "intersection", "mean", "threshold_vote"
        target_resolution: output spatial resolution (64 for latent)
        resolutions_to_use: which attention layer resolutions to use
        vote_threshold: fraction of exemplars that must agree (for threshold_vote)

    Returns:
        spatial_mask: [H, W] normalized attention mask
        per_exemplar_areas: list of per-exemplar mask areas (for stats)
    """
    if aggregation == "union":
        # Max across tokens = union. Use existing infrastructure directly.
        mask = compute_attention_spatial_mask(
            probe_store,
            token_indices=probe_token_indices,
            target_resolution=target_resolution,
            resolutions_to_use=resolutions_to_use,
        )
        # Compute per-exemplar areas for stats
        per_exemplar_areas = _compute_per_exemplar_areas(
            probe_store, probe_token_indices, target_resolution,
            resolutions_to_use)
        return mask, per_exemplar_areas

    # For non-union modes, compute per-exemplar masks individually
    exemplar_masks = []
    per_exemplar_areas = []

    for i, tok_idx in enumerate(probe_token_indices[:num_exemplars]):
        mask_i = compute_attention_spatial_mask(
            probe_store,
            token_indices=[tok_idx],
            target_resolution=target_resolution,
            resolutions_to_use=resolutions_to_use,
        )
        exemplar_masks.append(mask_i)
        per_exemplar_areas.append(float(mask_i.mean().item()))

    if not exemplar_masks:
        return torch.zeros(target_resolution, target_resolution), []

    stacked = torch.stack(exemplar_masks)  # [K, H, W]

    if aggregation == "intersection":
        combined = stacked.min(dim=0)[0]  # [H, W]
    elif aggregation == "mean":
        combined = stacked.mean(dim=0)  # [H, W]
    elif aggregation == "threshold_vote":
        # Binary vote: how many exemplars exceed threshold at each pixel
        binary = (stacked > 0.3).float()
        vote_frac = binary.mean(dim=0)  # fraction of exemplars voting "unsafe"
        combined = (vote_frac >= vote_threshold).float()
    else:
        raise ValueError(f"Unknown probe_aggregation: {aggregation}")

    # Normalize to [0, 1]
    flat = combined.reshape(-1)
    vmin, vmax = flat.min(), flat.max()
    if vmax - vmin > 1e-8:
        combined = (combined - vmin) / (vmax - vmin)

    return combined, per_exemplar_areas


def _compute_per_exemplar_areas(probe_store, probe_token_indices,
                                target_resolution, resolutions_to_use):
    """Compute per-exemplar mask areas for stats tracking."""
    per_areas = []
    for tok_idx in probe_token_indices:
        mask_i = compute_attention_spatial_mask(
            probe_store,
            token_indices=[tok_idx],
            target_resolution=target_resolution,
            resolutions_to_use=resolutions_to_use,
        )
        per_areas.append(float(mask_i.mean().item()))
    return per_areas


# =============================================================================
# Argument parser
# =============================================================================
def parse_args():
    p = ArgumentParser(description="Multi-Exemplar Diverse Probe Ensemble v19")
    p.add_argument("--ckpt", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--prompts", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--nsamples", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg_scale", type=float, default=7.5)

    # v19-specific: Multi-Exemplar probe
    p.add_argument("--clip_embeddings", type=str,
                    default="exemplars/sd14/clip_exemplar_embeddings.pt",
                    help="Path to pre-computed CLIP exemplar embeddings (.pt)")
    p.add_argument("--num_exemplars", type=int, default=16,
                    help="Number of exemplars to use (max 75, default: all available up to 16)")
    p.add_argument("--exemplar_selection", type=str, default="all",
                    choices=["all", "top_k", "diverse", "random_k"],
                    help="Exemplar selection strategy: "
                         "all (use first K), top_k (most discriminative vs anchor), "
                         "diverse (greedy farthest-point), random_k (random subset)")
    p.add_argument("--probe_aggregation", type=str, default="union",
                    choices=["union", "intersection", "mean", "threshold_vote"],
                    help="How to aggregate per-exemplar probe masks: "
                         "union (max, default), intersection (min), "
                         "mean (average), threshold_vote (majority vote)")
    p.add_argument("--vote_threshold", type=float, default=0.5,
                    help="Fraction of exemplars that must agree (for threshold_vote)")

    # Probe source: which probe to use for WHERE detection
    p.add_argument("--probe_source", type=str, default="both",
                    choices=["text", "image", "both"],
                    help="Probe source for WHERE: "
                         "text (text keyword probe only, v6 style), "
                         "image (multi-exemplar CLIP probe only, current v19), "
                         "both (union of text + image probes, default)")
    p.add_argument("--target_words", type=str, nargs="+",
                    default=["nude", "naked", "nudity"],
                    help="Keywords for text probe (used when probe_source=text or both)")

    # WHERE mode
    p.add_argument("--where_mode", type=str, default="multi_probe",
                    choices=["multi_probe", "fused", "noise_only"],
                    help="WHERE detection: multi_probe (v19 default), "
                         "fused (union of probe + noise CAS), "
                         "noise_only (v6 fallback)")

    # CAS (When)
    p.add_argument("--cas_threshold", type=float, default=0.6)
    p.add_argument("--cas_sticky", action="store_true", default=True)
    p.add_argument("--cas_no_sticky", action="store_true")

    # WHERE common params
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

    # Concepts (for CAS WHEN + text probe fallback)
    p.add_argument("--target_concepts", type=str, nargs="+",
                    default=["nudity", "nude person", "naked body"])
    p.add_argument("--anchor_concepts", type=str, nargs="+",
                    default=["clothed person", "person wearing clothes"])

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
    return args


# =============================================================================
# Main Generation Loop
# =============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_concept = args.concept_packs is not None

    print(f"\n{'='*70}")
    print(f"Multi-Exemplar Diverse Probe Ensemble v19")
    print(f"{'='*70}")
    print(f"  WHEN:  Global CAS threshold={args.cas_threshold}, sticky={args.cas_sticky}")
    print(f"  WHERE: Multi-Exemplar Probe, mode={args.where_mode}, "
          f"probe_source={args.probe_source}, "
          f"resolutions={args.attn_resolutions}")
    print(f"         num_exemplars={args.num_exemplars}, "
          f"selection={args.exemplar_selection}, "
          f"aggregation={args.probe_aggregation}")
    if args.probe_source in ("text", "both"):
        print(f"         target_words={args.target_words}")
    if args.probe_aggregation == "threshold_vote":
        print(f"         vote_threshold={args.vote_threshold}")
    print(f"         spatial_threshold={args.spatial_threshold}, "
          f"sigmoid_alpha={args.sigmoid_alpha}, blur={args.blur_sigma}")
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

    # ---- Pre-encode text concept embeddings (for CAS WHEN + anchor) ----
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
    pack_target_embeds_list = None
    pack_anchor_embeds_list = None
    pack_probe_token_indices = None

    # Single-concept state
    num_selected = 0
    use_probe = args.where_mode not in ("noise_only",)
    text_probe_token_indices = []
    probe_token_indices = []
    probe_store = AttentionProbeStore()
    original_processors = None
    exemplar_area_history_enabled = False

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
        exemplar_area_history_enabled = True

        # Load CLIP exemplar embeddings & build multi-exemplar sequence
        clip_data = torch.load(args.clip_embeddings, map_location="cpu")
        target_features = clip_data["target_clip_features"].float()
        anchor_features = clip_data.get("anchor_clip_features", None)
        if anchor_features is not None:
            anchor_features = anchor_features.float()

        print(f"Loaded CLIP exemplar features: {target_features.shape[0]} nudity exemplars")
        if anchor_features is not None:
            print(f"  Anchor features: {anchor_features.shape[0]} clothed exemplars")

        selected = select_exemplars(
            target_features,
            anchor_features=anchor_features,
            num_exemplars=args.num_exemplars,
            selection=args.exemplar_selection,
        )
        num_selected = selected.shape[0]
        print(f"  Selected {num_selected} exemplars via '{args.exemplar_selection}' strategy")

        multi_exemplar_embeds, probe_token_indices = build_multi_exemplar_embedding(
            selected, text_encoder, tokenizer, device
        )
        multi_exemplar_embeds = multi_exemplar_embeds.to(device=device, dtype=unet.dtype)
        print(f"  Multi-exemplar embedding: {multi_exemplar_embeds.shape}")
        print(f"  Probe token indices: {probe_token_indices}")

        # Build probe embedding based on probe_source
        if use_probe and args.probe_source == "text":
            target_words_text = " ".join(args.target_words)
            with torch.no_grad():
                tw_inputs = tokenizer(
                    target_words_text,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                probe_embeds = text_encoder(tw_inputs.input_ids.to(device))[0]
            text_probe_token_indices = find_token_indices(
                target_words_text, args.target_words, tokenizer
            )
            probe_token_indices_for_image = []
            print(f"  Text probe embedding built for words: {args.target_words}")
            print(f"  Text probe token indices: {text_probe_token_indices}")

        elif use_probe and args.probe_source == "both":
            target_words_text = " ".join(args.target_words)
            tw_inputs = tokenizer(
                target_words_text,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            tw_ids = tw_inputs.input_ids[0].tolist()
            bos_id = tokenizer.bos_token_id
            eos_id = tokenizer.eos_token_id
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
            tw_content_ids = []
            for tid in tw_ids:
                if tid == bos_id:
                    continue
                if tid == eos_id:
                    break
                tw_content_ids.append(tid)

            token_embedding = text_encoder.text_model.embeddings.token_embedding
            with torch.no_grad():
                bos_embed = token_embedding(torch.tensor([bos_id], device=device))
                eos_embed = token_embedding(torch.tensor([eos_id], device=device))
                pad_embed = token_embedding(torch.tensor([pad_id], device=device))

            tokens_list = [bos_embed]
            for i in range(num_selected):
                tokens_list.append(selected[i:i+1].to(device))
            probe_token_indices_for_image = list(range(1, 1 + num_selected))

            max_content = 75
            n_text = min(len(tw_content_ids), max_content - num_selected)
            with torch.no_grad():
                for tid in tw_content_ids[:n_text]:
                    tokens_list.append(
                        token_embedding(torch.tensor([tid], device=device))
                    )
            text_probe_token_indices = list(
                range(1 + num_selected, 1 + num_selected + n_text)
            )
            tokens_list.append(eos_embed)
            n_pad = 77 - len(tokens_list)
            for _ in range(n_pad):
                tokens_list.append(pad_embed)

            probe_embeds = torch.cat(tokens_list, dim=0).unsqueeze(0)
            print(f"  Combined probe: {num_selected} image exemplars + "
                  f"{n_text} text tokens (words={args.target_words})")
            print(f"  Image probe indices: {probe_token_indices_for_image}")
            print(f"  Text probe indices: {text_probe_token_indices}")

        else:
            probe_embeds = multi_exemplar_embeds
            probe_token_indices_for_image = probe_token_indices
            text_probe_token_indices = []

        if use_probe and args.probe_source != "text":
            probe_token_indices = probe_token_indices_for_image

        # Setup cross-attention probing
        if use_probe:
            target_keys = precompute_target_keys(
                unet, probe_embeds, args.attn_resolutions
            )
            original_processors = register_attention_probe(
                unet, probe_store, target_keys, args.attn_resolutions
            )
            if args.probe_source == "image":
                print(f"  Image probe installed: "
                      f"{num_selected} exemplar tokens at resolutions {args.attn_resolutions}")
            elif args.probe_source == "text":
                print(f"  Text probe installed at resolutions {args.attn_resolutions}")
            else:
                print(f"  Combined probe installed at resolutions {args.attn_resolutions}")
        else:
            original_processors = dict(unet.attn_processors)

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
            exemplar_area_history = []

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
                if use_probe:
                    probe_store.active = True
                    probe_store.reset()

                with torch.no_grad():
                    lat_batch = torch.cat([lat_in, lat_in])
                    embed_batch = torch.cat([uncond_embeds, prompt_embeds])
                    raw = unet(lat_batch, t, encoder_hidden_states=embed_batch).sample
                    eps_null, eps_prompt = raw.chunk(2)

                if use_probe:
                    probe_store.active = False

                # ==== Pass 3: target concept (for CAS WHEN) ====
                with torch.no_grad():
                    eps_target = unet(lat_in, t,
                                      encoder_hidden_states=target_text_embeds).sample

                # Standard CFG
                eps_cfg = eps_null + args.cfg_scale * (eps_prompt - eps_null)

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
                        p_target_embeds = pack_target_embeds_list[pack_idx]
                        p_anchor_embeds = pack_anchor_embeds_list[pack_idx]
                        p_probe_indices = pack_probe_token_indices[pack_idx]

                        p_guide_mode = pack.get("guide_mode", args.guide_mode)
                        p_safety_scale = pack.get("safety_scale", args.safety_scale)

                        # Target direction for this pack (text mode)
                        with torch.no_grad():
                            eps_target_pack = unet(lat_in, t,
                                                   encoder_hidden_states=p_target_embeds).sample

                        cas_val_pack, should_trigger_pack = pack_cas_gate.compute(
                            eps_prompt, eps_null, eps_target_pack
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
                            eps_prompt=eps_prompt,
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
                    # SINGLE-CONCEPT: original v19 behavior
                    # ===================================================

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
    
                        # ==== WHERE: Spatial mask ====
                        if args.where_mode == "multi_probe":
                            # Compute mask(s) based on probe_source
                            if args.probe_source == "text":
                                # Text-only probe: use keyword token indices
                                attn_spatial = compute_attention_spatial_mask(
                                    probe_store,
                                    token_indices=text_probe_token_indices,
                                    target_resolution=64,
                                    resolutions_to_use=args.attn_resolutions,
                                )
                                per_ex_areas = []
                                soft_mask = compute_soft_mask(
                                    attn_spatial.to(device),
                                    spatial_threshold=args.spatial_threshold,
                                    sigmoid_alpha=args.sigmoid_alpha,
                                    blur_sigma=args.blur_sigma,
                                    device=device,
                                )
                            elif args.probe_source == "image":
                                # Image-only probe: multi-exemplar (original v19 behavior)
                                attn_spatial, per_ex_areas = compute_multi_exemplar_mask(
                                    probe_store,
                                    probe_token_indices=probe_token_indices,
                                    num_exemplars=num_selected,
                                    aggregation=args.probe_aggregation,
                                    target_resolution=64,
                                    resolutions_to_use=args.attn_resolutions,
                                    vote_threshold=args.vote_threshold,
                                )
                                soft_mask = compute_soft_mask(
                                    attn_spatial.to(device),
                                    spatial_threshold=args.spatial_threshold,
                                    sigmoid_alpha=args.sigmoid_alpha,
                                    blur_sigma=args.blur_sigma,
                                    device=device,
                                )
                            else:
                                # both: union of image mask and text mask
                                img_spatial, per_ex_areas = compute_multi_exemplar_mask(
                                    probe_store,
                                    probe_token_indices=probe_token_indices,
                                    num_exemplars=num_selected,
                                    aggregation=args.probe_aggregation,
                                    target_resolution=64,
                                    resolutions_to_use=args.attn_resolutions,
                                    vote_threshold=args.vote_threshold,
                                )
                                img_mask = compute_soft_mask(
                                    img_spatial.to(device),
                                    spatial_threshold=args.spatial_threshold,
                                    sigmoid_alpha=args.sigmoid_alpha,
                                    blur_sigma=args.blur_sigma,
                                    device=device,
                                )
                                if text_probe_token_indices:
                                    txt_spatial = compute_attention_spatial_mask(
                                        probe_store,
                                        token_indices=text_probe_token_indices,
                                        target_resolution=64,
                                        resolutions_to_use=args.attn_resolutions,
                                    )
                                    txt_mask = compute_soft_mask(
                                        txt_spatial.to(device),
                                        spatial_threshold=args.spatial_threshold,
                                        sigmoid_alpha=args.sigmoid_alpha,
                                        blur_sigma=args.blur_sigma,
                                        device=device,
                                    )
                                    soft_mask = torch.max(img_mask, txt_mask)
                                    attn_spatial = torch.max(img_spatial, txt_spatial.to(img_spatial.device))
                                else:
                                    soft_mask = img_mask
                                    attn_spatial = img_spatial
                            exemplar_area_history.append(per_ex_areas)
    
                        elif args.where_mode == "fused":
                            # Union of probe mask(s) + noise CAS
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
    
                            if args.probe_source == "text":
                                txt_spatial = compute_attention_spatial_mask(
                                    probe_store,
                                    token_indices=text_probe_token_indices,
                                    target_resolution=64,
                                    resolutions_to_use=args.attn_resolutions,
                                )
                                attn_spatial = txt_spatial
                                probe_mask = compute_soft_mask(
                                    txt_spatial.to(device),
                                    spatial_threshold=args.spatial_threshold,
                                    sigmoid_alpha=args.sigmoid_alpha,
                                    blur_sigma=args.blur_sigma,
                                    device=device,
                                )
                                per_ex_areas = []
                            elif args.probe_source == "image":
                                attn_spatial, per_ex_areas = compute_multi_exemplar_mask(
                                    probe_store,
                                    probe_token_indices=probe_token_indices,
                                    num_exemplars=num_selected,
                                    aggregation=args.probe_aggregation,
                                    target_resolution=64,
                                    resolutions_to_use=args.attn_resolutions,
                                    vote_threshold=args.vote_threshold,
                                )
                                probe_mask = compute_soft_mask(
                                    attn_spatial.to(device),
                                    spatial_threshold=args.spatial_threshold,
                                    sigmoid_alpha=args.sigmoid_alpha,
                                    blur_sigma=args.blur_sigma,
                                    device=device,
                                )
                            else:
                                # both: union of image + text probe masks
                                img_spatial, per_ex_areas = compute_multi_exemplar_mask(
                                    probe_store,
                                    probe_token_indices=probe_token_indices,
                                    num_exemplars=num_selected,
                                    aggregation=args.probe_aggregation,
                                    target_resolution=64,
                                    resolutions_to_use=args.attn_resolutions,
                                    vote_threshold=args.vote_threshold,
                                )
                                attn_spatial = img_spatial  # for save_maps
                                img_mask = compute_soft_mask(
                                    img_spatial.to(device),
                                    spatial_threshold=args.spatial_threshold,
                                    sigmoid_alpha=args.sigmoid_alpha,
                                    blur_sigma=args.blur_sigma,
                                    device=device,
                                )
                                if text_probe_token_indices:
                                    txt_spatial = compute_attention_spatial_mask(
                                        probe_store,
                                        token_indices=text_probe_token_indices,
                                        target_resolution=64,
                                        resolutions_to_use=args.attn_resolutions,
                                    )
                                    txt_mask = compute_soft_mask(
                                        txt_spatial.to(device),
                                        spatial_threshold=args.spatial_threshold,
                                        sigmoid_alpha=args.sigmoid_alpha,
                                        blur_sigma=args.blur_sigma,
                                        device=device,
                                    )
                                    probe_mask = torch.max(img_mask, txt_mask)
                                else:
                                    probe_mask = img_mask
    
                            soft_mask = torch.max(probe_mask, noise_mask)
                            exemplar_area_history.append(per_ex_areas)
    
                        elif args.where_mode == "noise_only":
                            # Noise-based CAS only (v6 fallback, no probe)
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
                            attn_map_np = attn_spatial.float().cpu().numpy() if args.where_mode != "noise_only" else noise_spatial.float().cpu().numpy()
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
                    if multi_concept:
                        status = "GUIDED" if any_triggered else "skip"
                        area_s = f" area={mask_areas[-1]:.3f}" if mask_areas else ""
                        n_active = ""
                    else:
                        status = "GUIDED" if should_guide else ("CAS_ON" if should_trigger else "skip")
                        area_s = f" area={mask_areas[-1]:.3f}" if should_guide and mask_areas else ""
                        n_active = ""
                        if should_guide and exemplar_area_history:
                            last_areas = exemplar_area_history[-1]
                            active = sum(1 for a in last_areas if a > 0.01)
                            n_active = f" active_ex={active}/{num_selected}"
                    print(f"  [{step_idx:02d}] t={t.item():.0f} CAS={cas_val:.3f} "
                          f"{status}{area_s}{n_active}")

            # ---- Decode ----
            with torch.no_grad():
                decoded = vae.decode(latents.to(vae.dtype) / vae.config.scaling_factor).sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                img_np = (decoded[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

            slug = make_safe_filename(prompt)
            fname = f"{prompt_idx:04d}_{sample_idx:02d}_{slug}.png"
            Image.fromarray(img_np).resize((512, 512)).save(str(outdir / fname))

            # Per-exemplar area stats (averaged across guided steps)
            avg_per_ex_areas = []
            if exemplar_area_history:
                for ex_i in range(num_selected):
                    areas_for_ex = [step_areas[ex_i] for step_areas in exemplar_area_history
                                    if ex_i < len(step_areas)]
                    avg_per_ex_areas.append(float(np.mean(areas_for_ex)) if areas_for_ex else 0.0)

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
                "num_exemplars": num_selected,
                "exemplar_selection": args.exemplar_selection,
                "probe_aggregation": args.probe_aggregation,
                "where_mode": args.where_mode,
                "probe_source": args.probe_source,
                "avg_per_exemplar_areas": avg_per_ex_areas,
            }
            all_stats.append(stats)

            if prompt_idx % 10 == 0 or args.debug:
                active_str = ""
                if avg_per_ex_areas:
                    n_active = sum(1 for a in avg_per_ex_areas if a > 0.01)
                    active_str = f" active_ex={n_active}/{num_selected}"
                tqdm.write(
                    f"  [{prompt_idx:03d}_{sample_idx}] guided={guided_count}/{total_steps} "
                    f"CAS avg={stats['avg_cas']:.3f} max={stats['max_cas']:.3f} "
                    f"mask_area={stats['avg_mask_area']:.3f}{active_str}"
                )

    # ---- Cleanup ----
    if use_probe:
        restore_processors(unet, original_processors)

    # ---- Save stats ----
    n = len(all_stats)
    n_trig = sum(1 for s in all_stats if s["guided_steps"] > 0)

    # Compute exemplar-level summary stats
    all_per_ex_areas = [s["avg_per_exemplar_areas"] for s in all_stats
                        if s["avg_per_exemplar_areas"]]
    exemplar_summary = {}
    if all_per_ex_areas:
        for ex_i in range(num_selected):
            areas = [s[ex_i] for s in all_per_ex_areas if ex_i < len(s)]
            exemplar_summary[f"exemplar_{ex_i}"] = {
                "avg_area": float(np.mean(areas)) if areas else 0.0,
                "max_area": float(np.max(areas)) if areas else 0.0,
                "active_frac": float(np.mean([1.0 if a > 0.01 else 0.0 for a in areas])) if areas else 0.0,
            }

    summary = {
        "method": "Multi-Exemplar Diverse Probe Ensemble v19",
        "args": vars(args),
        "overall": {
            "total_images": n,
            "triggered": n_trig,
            "trigger_rate": n_trig / max(n, 1),
            "avg_guided_steps": float(np.mean([s["guided_steps"] for s in all_stats])) if n else 0,
            "avg_cas": float(np.mean([s["avg_cas"] for s in all_stats if s["avg_cas"] > 0])) if n else 0,
            "avg_mask_area": float(np.mean([s["avg_mask_area"] for s in all_stats if s["avg_mask_area"] > 0])) if n else 0,
            "num_exemplars_used": num_selected,
            "exemplar_selection": args.exemplar_selection,
            "probe_aggregation": args.probe_aggregation,
            "where_mode": args.where_mode,
            "probe_source": args.probe_source,
        },
        "exemplar_stats": exemplar_summary,
        "per_image": all_stats,
    }
    with open(outdir / "stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"DONE! {n} images, {n_trig} triggered ({100*n_trig/max(n,1):.0f}%)")
    print(f"  Multi-Exemplar Probe: {num_selected} exemplars, "
          f"selection={args.exemplar_selection}, aggregation={args.probe_aggregation}")
    print(f"  WHERE mode: {args.where_mode}")
    print(f"  Guide mode: {args.guide_mode}, safety_scale={args.safety_scale}")
    print(f"  Avg guided steps: {summary['overall']['avg_guided_steps']:.1f}/{total_steps if n else 0}")
    print(f"  Avg mask area: {summary['overall']['avg_mask_area']:.3f}")
    if exemplar_summary:
        active_exemplars = sum(1 for v in exemplar_summary.values()
                               if v["active_frac"] > 0.1)
        print(f"  Active exemplars (>10% of triggered images): "
              f"{active_exemplars}/{num_selected}")
    print(f"  Output: {outdir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
