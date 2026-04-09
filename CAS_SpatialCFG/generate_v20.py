#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v20: v4 Anchor-Inpaint + CLIP Image Exemplar WHERE Enhancement

Base architecture from v4 (96.5% SR, the best):
  WHEN: Noise CAS (threshold=0.6, sticky)
  WHERE: Spatial CAS (per-pixel noise cosine similarity, 3x3 neighborhood)
  HOW: Anchor inpainting (blend CFG with anchor-CFG in masked regions)

v20 enhancement: optionally refine WHERE mask using CLIP image exemplar
cross-attention probing. The UNet's cross-attention layers reveal WHERE
the model is attending to unsafe visual patterns.

Image embedding strategies (--img_pool):
  none:       Pure v4 baseline (no image probe)
  cls_mean:   Mean-pool K exemplar CLS features -> 1 probe token
  cls_multi:  Each exemplar CLS as separate token -> K probe tokens (union)
  patch_disc: Top-K discriminative patch tokens (body-specific spatial patches)

Fusion of noise_mask and attn_mask (--fusion):
  noise_only:  Pure v4 spatial CAS (ignores attn even if computed)
  multiply:    noise * attn (intersection — guide only where BOTH agree)
  noise_boost: noise * (1 + boost_alpha * attn) — attn amplifies noise

Evidence:
  v4 anchor_inpaint = 96.5% SR (Qwen3-VL, Ring-A-Bell)
  v3 dag_adaptive   = 92.1% SR
  v14 fused noise*crossattn = 72.5% SR (too aggressive intersection)
  v19 multi-exemplar = 73.4% SR
  -> v20 hypothesis: v4 base + light attn boost (not hard intersection) can beat v4
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
# Global CAS (WHEN) — identical to v4
# =============================================================================
class GlobalCAS:
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
# Spatial CAS (WHERE — noise-based) — identical to v4
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
# Guidance (HOW) — v4 anchor_inpaint + v3 dag_adaptive
# =============================================================================
def apply_guidance(eps_cfg, eps_null, eps_prompt, eps_target, eps_anchor,
                   soft_mask, guide_mode="anchor_inpaint", safety_scale=1.0,
                   cfg_scale=7.5):
    mask = soft_mask.to(eps_cfg.dtype)

    if guide_mode == "anchor_inpaint":
        eps_anchor_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
        eps_final = eps_cfg * (1.0 - safety_scale * mask) + \
                    eps_anchor_cfg * (safety_scale * mask)

    elif guide_mode == "dag_adaptive":
        d_target = eps_target - eps_null
        binary_mask = (mask > 0.5).to(mask.dtype)
        area = binary_mask.sum() / binary_mask.numel()
        area_scale = 5.0 / (mask.shape[-1] * mask.shape[-2])
        area_factor = area_scale * area * mask.numel()
        mag_scale = 1.0 + 4.0 * mask
        correction = safety_scale * area_factor * mag_scale * binary_mask * d_target
        eps_final = eps_cfg - correction

    elif guide_mode == "hybrid":
        # Subtract target + add anchor (both relative to null)
        eps_final = eps_cfg \
                    - safety_scale * mask * (eps_target - eps_null) \
                    + safety_scale * mask * (eps_anchor - eps_null)

    elif guide_mode == "sld":
        eps_final = eps_cfg - safety_scale * mask * (eps_target - eps_null)

    else:
        raise ValueError(f"Unknown guide_mode: {guide_mode}")

    if torch.isnan(eps_final).any() or torch.isinf(eps_final).any():
        eps_final = torch.where(torch.isfinite(eps_final), eps_final, eps_cfg)
    return eps_final


# =============================================================================
# CLIP Image Probe Embedding Builders (v20 innovation)
# =============================================================================
def build_cls_mean_embedding(clip_features, text_encoder, tokenizer, device,
                             n_repeat=4):
    """
    Mean-pool K exemplar CLS features -> single [768] vector,
    repeat n_repeat times as probe tokens.

    Format: [BOS, tok, tok, ..., EOS, PAD, PAD, ...]
    Probe token indices: [1, 2, ..., n_repeat]
    """
    avg_feat = clip_features.mean(dim=0)  # [768]
    avg_feat = avg_feat / avg_feat.norm()

    token_embedding = text_encoder.text_model.embeddings.token_embedding
    bos_embed = token_embedding(torch.tensor([tokenizer.bos_token_id], device=device))
    eos_embed = token_embedding(torch.tensor([tokenizer.eos_token_id], device=device))
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_embed = token_embedding(torch.tensor([pad_id], device=device))

    concept_embed = avg_feat.unsqueeze(0).to(device)  # [1, 768]
    tokens_list = [bos_embed]
    for _ in range(n_repeat):
        tokens_list.append(concept_embed)
    tokens_list.append(eos_embed)
    n_pad = 77 - len(tokens_list)
    for _ in range(n_pad):
        tokens_list.append(pad_embed)

    embeds = torch.cat(tokens_list, dim=0).unsqueeze(0)  # [1, 77, 768]
    probe_indices = list(range(1, 1 + n_repeat))
    return embeds, probe_indices


def build_cls_multi_embedding(clip_features, text_encoder, tokenizer, device,
                              max_tokens=16):
    """
    Each exemplar CLS feature as a separate probe token.
    Union via max-across-tokens captures diverse nudity patterns.

    Format: [BOS, ex_1, ex_2, ..., ex_K, EOS, PAD, PAD, ...]
    Probe token indices: [1, 2, ..., K]
    """
    K = min(clip_features.shape[0], max_tokens, 75)

    token_embedding = text_encoder.text_model.embeddings.token_embedding
    bos_embed = token_embedding(torch.tensor([tokenizer.bos_token_id], device=device))
    eos_embed = token_embedding(torch.tensor([tokenizer.eos_token_id], device=device))
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_embed = token_embedding(torch.tensor([pad_id], device=device))

    # Normalize each exemplar feature
    feats = clip_features[:K].to(device).float()
    feats = F.normalize(feats, dim=-1)

    tokens_list = [bos_embed]
    for i in range(K):
        tokens_list.append(feats[i:i+1])  # [1, 768]
    tokens_list.append(eos_embed)
    n_pad = 77 - len(tokens_list)
    for _ in range(n_pad):
        tokens_list.append(pad_embed)

    embeds = torch.cat(tokens_list, dim=0).unsqueeze(0)  # [1, 77, 768]
    probe_indices = list(range(1, 1 + K))
    return embeds, probe_indices


def build_patch_disc_embedding(patch_data, text_encoder, tokenizer, device):
    """
    Use pre-computed discriminative patch tokens from prepare_clip_patch_tokens.py.
    These are the spatial patches that differ most between nude/clothed exemplars.

    Expects patch_data dict with 'target_patches' [K, 768].
    """
    target_patches = patch_data["target_patches"].to(device).float()
    K = min(target_patches.shape[0], 75)
    target_patches = F.normalize(target_patches[:K], dim=-1)

    token_embedding = text_encoder.text_model.embeddings.token_embedding
    bos_embed = token_embedding(torch.tensor([tokenizer.bos_token_id], device=device))
    eos_embed = token_embedding(torch.tensor([tokenizer.eos_token_id], device=device))
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_embed = token_embedding(torch.tensor([pad_id], device=device))

    tokens_list = [bos_embed]
    for i in range(K):
        tokens_list.append(target_patches[i:i+1])
    tokens_list.append(eos_embed)
    n_pad = 77 - len(tokens_list)
    for _ in range(n_pad):
        tokens_list.append(pad_embed)

    embeds = torch.cat(tokens_list, dim=0).unsqueeze(0)
    probe_indices = list(range(1, 1 + K))
    return embeds, probe_indices


# =============================================================================
# Mask Fusion (v20 core: how to combine noise_mask and attn_mask)
# =============================================================================
def fuse_masks(noise_mask, attn_mask, fusion_mode, boost_alpha=2.0):
    """
    Combine noise-based spatial CAS mask with cross-attention probe mask.

    Args:
        noise_mask: [1, 1, H, W] from spatial CAS (v4)
        attn_mask:  [1, 1, H, W] from cross-attention probe (normalized [0,1])
        fusion_mode: 'noise_only', 'multiply', 'noise_boost'
        boost_alpha: scaling for noise_boost mode
    """
    if fusion_mode == "noise_only" or attn_mask is None:
        return noise_mask

    # Ensure same shape
    if attn_mask.shape != noise_mask.shape:
        attn_mask = F.interpolate(attn_mask, size=noise_mask.shape[-2:],
                                  mode='bilinear', align_corners=False)

    if fusion_mode == "multiply":
        # Intersection: only guide where BOTH noise and attn agree
        return (noise_mask * attn_mask).clamp(0, 1)

    elif fusion_mode == "noise_boost":
        # Attn amplifies noise confidence: high attn -> stronger noise mask
        # noise * (1 + alpha * attn) keeps noise as base, attn boosts
        boosted = noise_mask * (1.0 + boost_alpha * attn_mask)
        return boosted.clamp(0, 1)

    else:
        raise ValueError(f"Unknown fusion_mode: {fusion_mode}")


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
        inputs = tokenizer(concept, padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt")
        embeds = text_encoder(inputs.input_ids.to(device))[0]
        all_embeds.append(embeds)
    return torch.stack(all_embeds).mean(dim=0)


# =============================================================================
# Arguments
# =============================================================================
def parse_args():
    p = ArgumentParser(description="v20: v4 Anchor-Inpaint + CLIP Image WHERE")

    # Model & I/O
    p.add_argument("--ckpt", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--prompts", required=True, help="CSV or TXT prompt file")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--nsamples", type=int, default=1)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg_scale", type=float, default=7.5)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)

    # WHEN — Global CAS
    p.add_argument("--cas_threshold", type=float, default=0.6)
    p.add_argument("--cas_sticky", action="store_true", default=True)
    p.add_argument("--cas_no_sticky", action="store_true")

    # WHERE — Noise spatial CAS (v4)
    p.add_argument("--spatial_threshold", type=float, default=0.3)
    p.add_argument("--sigmoid_alpha", type=float, default=10.0)
    p.add_argument("--neighborhood_size", type=int, default=3)
    p.add_argument("--blur_sigma", type=float, default=1.0)

    # WHERE — CLIP Image Probe (v20 enhancement)
    p.add_argument("--img_pool", type=str, default="none",
                   choices=["none", "cls_mean", "cls_multi", "patch_disc"],
                   help="Image embedding strategy for cross-attention probe. "
                        "none = pure v4 (no probe)")
    p.add_argument("--clip_embeddings", type=str, default=None,
                   help="Path to CLIP exemplar .pt file (for cls_mean/cls_multi)")
    p.add_argument("--patch_embeddings", type=str, default=None,
                   help="Path to CLIP patch tokens .pt file (for patch_disc)")
    p.add_argument("--n_repeat", type=int, default=4,
                   help="cls_mean: number of repeated probe tokens")
    p.add_argument("--max_exemplars", type=int, default=16,
                   help="cls_multi: max number of exemplar tokens")
    p.add_argument("--attn_resolutions", type=int, nargs="+", default=[16, 32])
    p.add_argument("--attn_threshold", type=float, default=0.3,
                   help="Sigmoid center for cross-attention mask")
    p.add_argument("--attn_sigmoid_alpha", type=float, default=10.0)

    # WHERE — Fusion
    p.add_argument("--fusion", type=str, default="noise_only",
                   choices=["noise_only", "multiply", "noise_boost"],
                   help="How to combine noise and attn masks")
    p.add_argument("--boost_alpha", type=float, default=2.0,
                   help="noise_boost: attn amplification strength")

    # HOW — Guidance
    p.add_argument("--guide_mode", type=str, default="anchor_inpaint",
                   choices=["anchor_inpaint", "dag_adaptive", "hybrid", "sld"])
    p.add_argument("--safety_scale", type=float, default=1.0,
                   help="Guidance strength (anchor_inpaint: ~0.9, dag: ~3.0)")
    p.add_argument("--guide_start_frac", type=float, default=0.0)

    # Concepts
    p.add_argument("--target_concepts", type=str, nargs="+",
                   default=["nudity", "nude person", "naked body"])
    p.add_argument("--anchor_concepts", type=str, nargs="+",
                   default=["clothed person", "person wearing clothes"])

    # Debug
    p.add_argument("--save_maps", action="store_true")
    p.add_argument("--debug", action="store_true")

    args = p.parse_args()
    if args.cas_no_sticky:
        args.cas_sticky = False

    # Validate
    if args.img_pool in ("cls_mean", "cls_multi") and args.clip_embeddings is None:
        p.error("--clip_embeddings required for img_pool=cls_mean/cls_multi")
    if args.img_pool == "patch_disc" and args.patch_embeddings is None:
        p.error("--patch_embeddings required for img_pool=patch_disc")

    # img_pool=none -> force fusion=noise_only
    if args.img_pool == "none" and args.fusion != "noise_only":
        print(f"[v20] img_pool=none: forcing fusion=noise_only")
        args.fusion = "noise_only"

    return args


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_probe = args.img_pool != "none"

    print(f"\n{'='*70}")
    print(f"v20: v4 Anchor-Inpaint + CLIP Image WHERE Enhancement")
    print(f"{'='*70}")
    print(f"  WHEN:    CAS threshold={args.cas_threshold}, sticky={args.cas_sticky}")
    print(f"  WHERE:   noise(nbr={args.neighborhood_size}, thr={args.spatial_threshold})"
          f" + img_pool={args.img_pool}, fusion={args.fusion}")
    if use_probe:
        print(f"           attn_res={args.attn_resolutions}, attn_thr={args.attn_threshold}")
        if args.fusion == "noise_boost":
            print(f"           boost_alpha={args.boost_alpha}")
    print(f"  HOW:     {args.guide_mode}, safety_scale={args.safety_scale}")
    print(f"  Targets: {args.target_concepts}")
    print(f"  Anchors: {args.anchor_concepts}")
    print(f"  Steps={args.steps}, CFG={args.cfg_scale}, nsamples={args.nsamples}")
    print(f"{'='*70}\n")

    # ---- Load prompts ----
    all_prompts = load_prompts(args.prompts)
    end_idx = args.end_idx if args.end_idx > 0 else len(all_prompts)
    prompts_with_idx = list(enumerate(all_prompts))[args.start_idx:end_idx]
    print(f"Loaded {len(all_prompts)} prompts, processing [{args.start_idx}:{end_idx}]")

    # ---- Pipeline ----
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

    # ---- Encode concepts ----
    with torch.no_grad():
        target_embeds = encode_concepts(text_encoder, tokenizer,
                                        args.target_concepts, device)
        anchor_embeds = encode_concepts(text_encoder, tokenizer,
                                        args.anchor_concepts, device)
        uncond_inputs = tokenizer("", padding="max_length",
                                  max_length=tokenizer.model_max_length,
                                  truncation=True, return_tensors="pt")
        uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]

    # ---- Setup cross-attention probe (v20 enhancement) ----
    probe_store = None
    probe_token_indices = None
    original_processors = None

    if use_probe:
        probe_store = AttentionProbeStore()

        if args.img_pool == "cls_mean":
            clip_data = torch.load(args.clip_embeddings, map_location="cpu")
            clip_features = clip_data["target_clip_features"].float()
            probe_embeds, probe_token_indices = build_cls_mean_embedding(
                clip_features, text_encoder, tokenizer, device,
                n_repeat=args.n_repeat)
            print(f"  Probe: cls_mean, {clip_features.shape[0]} exemplars -> "
                  f"{args.n_repeat} tokens")

        elif args.img_pool == "cls_multi":
            clip_data = torch.load(args.clip_embeddings, map_location="cpu")
            clip_features = clip_data["target_clip_features"].float()
            probe_embeds, probe_token_indices = build_cls_multi_embedding(
                clip_features, text_encoder, tokenizer, device,
                max_tokens=args.max_exemplars)
            print(f"  Probe: cls_multi, {len(probe_token_indices)} individual "
                  f"exemplar tokens")

        elif args.img_pool == "patch_disc":
            patch_data = torch.load(args.patch_embeddings, map_location="cpu")
            probe_embeds, probe_token_indices = build_patch_disc_embedding(
                patch_data, text_encoder, tokenizer, device)
            K = patch_data["target_patches"].shape[0]
            print(f"  Probe: patch_disc, {K} discriminative patches")

        # Precompute target keys and register probe processors
        target_keys = precompute_target_keys(
            unet, probe_embeds.to(dtype=next(unet.parameters()).dtype),
            args.attn_resolutions)
        original_processors = register_attention_probe(
            unet, probe_store, target_keys, args.attn_resolutions)
        print(f"  Probe token indices: {probe_token_indices}")

    # ---- CAS ----
    cas = GlobalCAS(threshold=args.cas_threshold, sticky=args.cas_sticky)

    # ---- Output ----
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

            with torch.no_grad():
                prompt_inputs = tokenizer(
                    prompt, padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt")
                prompt_embeds = text_encoder(prompt_inputs.input_ids.to(device))[0]

            set_seed(seed)
            latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
            latents = latents * scheduler.init_noise_sigma
            scheduler.set_timesteps(args.steps, device=device)
            total_steps = len(scheduler.timesteps)
            guide_start_step = int(total_steps * args.guide_start_frac)

            for step_idx, t in enumerate(scheduler.timesteps):
                lat_in = scheduler.scale_model_input(latents, t)

                # ---- UNet forward: null + prompt (batched) ----
                with torch.no_grad():
                    if use_probe:
                        probe_store.active = True
                        probe_store.reset()

                    lat_batch = torch.cat([lat_in, lat_in])
                    embed_batch = torch.cat([uncond_embeds, prompt_embeds])
                    raw = unet(lat_batch, t,
                               encoder_hidden_states=embed_batch).sample
                    eps_null, eps_prompt_pred = raw.chunk(2)

                    if use_probe:
                        probe_store.active = False

                    # Target concept
                    eps_target = unet(lat_in, t,
                                      encoder_hidden_states=target_embeds).sample

                # Standard CFG
                eps_cfg = eps_null + args.cfg_scale * (eps_prompt_pred - eps_null)

                # ---- WHEN: Global CAS ----
                cas_val, should_trigger = cas.compute(
                    eps_prompt_pred, eps_null, eps_target)
                cas_values.append(cas_val)
                in_window = step_idx >= guide_start_step
                should_guide = should_trigger and in_window

                if should_guide:
                    # ---- Anchor UNet call (only when guided) ----
                    with torch.no_grad():
                        eps_anchor = unet(lat_in, t,
                                          encoder_hidden_states=anchor_embeds).sample

                    # ---- WHERE: Noise spatial CAS mask ----
                    spatial_cas = compute_spatial_cas(
                        eps_prompt_pred, eps_null, eps_target,
                        neighborhood_size=args.neighborhood_size)
                    noise_mask = compute_soft_mask(
                        spatial_cas,
                        spatial_threshold=args.spatial_threshold,
                        sigmoid_alpha=args.sigmoid_alpha,
                        blur_sigma=args.blur_sigma,
                        device=device)

                    # ---- WHERE: Cross-attention probe mask (v20) ----
                    attn_mask = None
                    if use_probe and probe_store.get_maps():
                        attn_spatial = compute_attention_spatial_mask(
                            probe_store,
                            token_indices=probe_token_indices,
                            target_resolution=64,
                            resolutions_to_use=args.attn_resolutions)
                        # Sigmoid soft mask
                        attn_mask = torch.sigmoid(
                            args.attn_sigmoid_alpha *
                            (attn_spatial.to(device) - args.attn_threshold))
                        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                        if args.blur_sigma > 0:
                            attn_mask = gaussian_blur_2d(
                                attn_mask, kernel_size=5,
                                sigma=args.blur_sigma)
                        attn_mask = attn_mask.clamp(0, 1)

                    # ---- WHERE: Fuse masks ----
                    final_mask = fuse_masks(
                        noise_mask, attn_mask,
                        fusion_mode=args.fusion,
                        boost_alpha=args.boost_alpha)

                    # ---- HOW: Apply guidance ----
                    eps_final = apply_guidance(
                        eps_cfg=eps_cfg,
                        eps_null=eps_null,
                        eps_prompt=eps_prompt_pred,
                        eps_target=eps_target,
                        eps_anchor=eps_anchor,
                        soft_mask=final_mask,
                        guide_mode=args.guide_mode,
                        safety_scale=args.safety_scale,
                        cfg_scale=args.cfg_scale)

                    guided_count += 1
                    area_val = float(final_mask.mean().item())
                    mask_areas.append(area_val)

                    # Save debug maps
                    if args.save_maps and step_idx % 10 == 0:
                        prefix = f"{prompt_idx:04d}_{sample_idx:02d}_s{step_idx:03d}"
                        maps_dir = outdir / "maps"
                        # Noise CAS
                        cas_np = spatial_cas.float().cpu().numpy()
                        cas_np = np.nan_to_num(cas_np, nan=0.0)
                        cas_img = (np.clip((cas_np + 1) / 2, 0, 1) * 255).astype(np.uint8)
                        Image.fromarray(cas_img, 'L').save(
                            str(maps_dir / f"{prefix}_noise_cas.png"))
                        # Noise mask
                        nm_np = noise_mask[0, 0].float().cpu().numpy()
                        Image.fromarray((np.clip(nm_np, 0, 1) * 255).astype(np.uint8), 'L').save(
                            str(maps_dir / f"{prefix}_noise_mask.png"))
                        # Attn mask (if exists)
                        if attn_mask is not None:
                            am_np = attn_mask[0, 0].float().cpu().numpy()
                            Image.fromarray((np.clip(am_np, 0, 1) * 255).astype(np.uint8), 'L').save(
                                str(maps_dir / f"{prefix}_attn_mask.png"))
                        # Final fused mask
                        fm_np = final_mask[0, 0].float().cpu().numpy()
                        Image.fromarray((np.clip(fm_np, 0, 1) * 255).astype(np.uint8), 'L').save(
                            str(maps_dir / f"{prefix}_final_mask.png"))

                    if args.debug and step_idx % 10 == 0:
                        attn_area = float(attn_mask.mean().item()) if attn_mask is not None else 0
                        print(f"  [{step_idx:02d}] CAS={cas_val:.3f} GUIDED "
                              f"noise_area={float(noise_mask.mean()):.3f} "
                              f"attn_area={attn_area:.3f} "
                              f"final_area={area_val:.3f}")
                else:
                    eps_final = eps_cfg
                    if args.debug and step_idx % 10 == 0:
                        status = "CAS_ON" if should_trigger else "skip"
                        print(f"  [{step_idx:02d}] CAS={cas_val:.3f} {status}")

                # ---- DDIM step ----
                latents_prev = latents.clone()
                latents = scheduler.step(eps_final, t, latents).prev_sample

                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    print(f"  WARNING: NaN/Inf at step {step_idx}, falling back")
                    eps_fallback = eps_null + args.cfg_scale * (eps_prompt_pred - eps_null)
                    latents = scheduler.step(eps_fallback, t, latents_prev).prev_sample

            # ---- Decode ----
            with torch.no_grad():
                decoded = vae.decode(
                    latents.to(vae.dtype) / vae.config.scaling_factor).sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                img_np = (decoded[0].cpu().permute(1, 2, 0).numpy() * 255
                          ).round().astype(np.uint8)

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
                "max_cas": max(cas_values) if cas_values else 0,
                "mean_mask_area": np.mean(mask_areas) if mask_areas else 0,
            }
            all_stats.append(stats)

    # ---- Save stats ----
    stats_path = outdir / "generation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    # Save args
    args_path = outdir / "args.json"
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    guided_imgs = sum(1 for s in all_stats if s["guided_steps"] > 0)
    print(f"\nDone! {len(all_stats)} images generated.")
    print(f"  Guided: {guided_imgs}/{len(all_stats)} "
          f"({100*guided_imgs/max(len(all_stats),1):.1f}%)")
    print(f"  Stats: {stats_path}")

    if original_processors is not None:
        restore_processors(unet, original_processors)


if __name__ == "__main__":
    main()
