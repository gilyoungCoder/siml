#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v10: Prompt-Fidelity Projection Guidance — Training-Free Safe Generation

Key Innovation over v7:
  Instead of subtracting the FULL target direction (which destroys semantics),
  project out only the nudity-aligned component from the prompt direction.
  This surgically removes nudity while preserving all other prompt semantics.

New Guide Modes:
  - proj_anchor: Per-pixel projection removes nudity from prompt, anchor fills in
  - hybrid_fidelity: v7 hybrid + prompt fidelity regularization (clamped deviation)

Additional Features:
  - Adaptive CAS scaling: guidance strength proportional to CAS value
  - Deviation clamping: limit max change from original CFG to prevent content destruction
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
# Global CAS (WHEN) — same as v7
# =============================================================================
class GlobalCAS:
    def __init__(self, threshold: float = 0.6, sticky: bool = True):
        self.threshold = threshold
        self.sticky = sticky
        self.triggered = False

    def reset(self):
        self.triggered = False

    def compute(self, eps_prompt, eps_null, eps_target=None, d_target_global=None):
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
# Spatial CAS (WHERE) — same as v7
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


def compute_spatial_cas_with_dir(d_prompt, d_target, neighborhood_size=3):
    d_prompt_f = d_prompt.float()
    d_target_f = d_target.float()
    H, W = d_prompt_f.shape[2], d_prompt_f.shape[3]
    pad = neighborhood_size // 2
    d_prompt_unfolded = F.unfold(d_prompt_f, kernel_size=neighborhood_size, padding=pad)
    d_target_unfolded = F.unfold(d_target_f, kernel_size=neighborhood_size, padding=pad)
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
# Guidance Application (HOW) — v10 new modes
# =============================================================================
def apply_guidance(
    eps_cfg, eps_null, eps_prompt, eps_target, eps_anchor,
    soft_mask, guide_mode="proj_anchor", safety_scale=1.0, cfg_scale=7.5,
    cas_value=0.0, **kwargs,
):
    mask = soft_mask.to(eps_cfg.dtype)

    if guide_mode == "hybrid":
        # v7-compatible hybrid mode
        t_scale = kwargs.get("target_scale", safety_scale)
        a_scale = kwargs.get("anchor_scale", safety_scale)
        eps_final = eps_cfg \
                    - t_scale * mask * (eps_target - eps_null) \
                    + a_scale * mask * (eps_anchor - eps_null)

    elif guide_mode == "proj_anchor":
        # ============================================================
        # v10 KEY: Projection-based nudity removal + anchor redirection
        # ============================================================
        # Instead of subtracting the full target direction (which destroys
        # non-nudity content), project out ONLY the nudity-aligned component
        # from the prompt direction. Then add anchor for semantic redirection.
        #
        # d_safe = d_prompt - proj(d_prompt, d_target)  [nudity removed]
        # eps_safe_cfg = eps_null + cfg_scale * d_safe
        # eps_final = (1-a*M)*eps_safe_cfg + a*M*eps_anchor_cfg
        # ============================================================
        t_scale = kwargs.get("target_scale", safety_scale)
        a_scale = kwargs.get("anchor_scale", safety_scale)

        d_prompt = (eps_prompt - eps_null).float()
        d_target = (eps_target - eps_null).float()

        # Per-CHANNEL projection (4-channel latent space)
        # dot product per pixel: [1, 1, H, W]
        dot = (d_prompt * d_target).sum(dim=1, keepdim=True)
        norm_sq = (d_target * d_target).sum(dim=1, keepdim=True).clamp(min=1e-8)
        proj_coeff = dot / norm_sq  # [1, 1, H, W]

        # Only remove POSITIVE alignment (where prompt direction aligns with nudity)
        # Negative alignment means prompt goes opposite to nudity — keep that
        proj_coeff = proj_coeff.clamp(min=0)

        # Nudity component in prompt direction
        d_nudity_in_prompt = proj_coeff * d_target  # [1, 4, H, W]

        # Safe prompt direction: prompt minus nudity component, scaled by t_scale
        d_safe = d_prompt - t_scale * mask.float() * d_nudity_in_prompt

        # Reconstruct safe CFG
        eps_safe_cfg = (eps_null.float() + cfg_scale * d_safe).to(eps_cfg.dtype)

        # Anchor CFG for semantic redirection in masked regions
        eps_anchor_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)

        # Blend: safe CFG base + anchor boost in masked regions
        eps_final = (1.0 - a_scale * mask) * eps_safe_cfg + a_scale * mask * eps_anchor_cfg

    elif guide_mode == "hybrid_fidelity":
        # ============================================================
        # v10: Hybrid + Prompt Fidelity (deviation clamping)
        # Same as v7 hybrid but clamp maximum deviation from eps_cfg
        # to prevent content destruction
        # ============================================================
        t_scale = kwargs.get("target_scale", safety_scale)
        a_scale = kwargs.get("anchor_scale", safety_scale)
        max_dev = kwargs.get("max_deviation", 0.0)

        eps_guided = eps_cfg \
                     - t_scale * mask * (eps_target - eps_null) \
                     + a_scale * mask * (eps_anchor - eps_null)

        if max_dev > 0:
            # Clamp per-pixel deviation from original CFG
            delta = eps_guided - eps_cfg
            delta_norm = delta.norm(dim=1, keepdim=True)
            threshold = max_dev * eps_cfg.norm(dim=1, keepdim=True)
            scale_factor = (threshold / delta_norm.clamp(min=1e-8)).clamp(max=1.0)
            eps_final = eps_cfg + delta * scale_factor
        else:
            eps_final = eps_guided

    elif guide_mode == "proj_hybrid":
        # ============================================================
        # v10: Projection + Hybrid combo
        # Project out nudity from prompt, then apply hybrid guidance
        # on top (weaker scales since projection already removes nudity)
        # ============================================================
        t_scale = kwargs.get("target_scale", safety_scale)
        a_scale = kwargs.get("anchor_scale", safety_scale)

        d_prompt = (eps_prompt - eps_null).float()
        d_target = (eps_target - eps_null).float()

        # Per-pixel projection
        dot = (d_prompt * d_target).sum(dim=1, keepdim=True)
        norm_sq = (d_target * d_target).sum(dim=1, keepdim=True).clamp(min=1e-8)
        proj_coeff = (dot / norm_sq).clamp(min=0)
        d_nudity = proj_coeff * d_target

        # Safe prompt: remove nudity in masked regions
        d_safe = d_prompt - mask.float() * d_nudity
        eps_safe_cfg = (eps_null.float() + cfg_scale * d_safe).to(eps_cfg.dtype)

        # Additional anchor boost (on top of projection)
        eps_final = eps_safe_cfg + a_scale * mask * (eps_anchor - eps_null)

    elif guide_mode == "sld":
        eps_final = eps_cfg - safety_scale * mask * (eps_target - eps_null)

    elif guide_mode == "anchor_inpaint":
        eps_anchor_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
        eps_final = eps_cfg * (1.0 - safety_scale * mask) + eps_anchor_cfg * (safety_scale * mask)

    else:
        raise ValueError(f"Unknown guide_mode: {guide_mode}")

    # NaN/Inf guard
    if torch.isnan(eps_final).any() or torch.isinf(eps_final).any():
        eps_final = torch.where(torch.isfinite(eps_final), eps_final, eps_cfg)

    return eps_final


# =============================================================================
# Utils (same as v7)
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


def parse_args():
    p = ArgumentParser(description="v10: Prompt-Fidelity Projection Guidance")
    p.add_argument("--ckpt", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--prompts", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--nsamples", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg_scale", type=float, default=7.5)

    # CAS (When)
    p.add_argument("--cas_threshold", type=float, default=0.6)
    p.add_argument("--cas_sticky", action="store_true", default=True)
    p.add_argument("--cas_no_sticky", action="store_true")

    # Spatial CAS (Where)
    p.add_argument("--spatial_threshold", type=float, default=0.3)
    p.add_argument("--sigmoid_alpha", type=float, default=10.0)
    p.add_argument("--neighborhood_size", type=int, default=3)
    p.add_argument("--blur_sigma", type=float, default=1.0)

    # Guidance (How) — v10 new modes
    p.add_argument("--guide_mode", type=str, default="proj_anchor",
                    choices=["proj_anchor", "hybrid_fidelity", "proj_hybrid",
                             "hybrid", "sld", "anchor_inpaint"])
    p.add_argument("--safety_scale", type=float, default=1.0)
    p.add_argument("--target_scale", type=float, default=-1.0)
    p.add_argument("--anchor_scale", type=float, default=-1.0)
    p.add_argument("--guide_start_frac", type=float, default=0.0)

    # v10 new args
    p.add_argument("--max_deviation", type=float, default=0.0,
                    help="Max deviation from eps_cfg (0=unlimited, for hybrid_fidelity)")
    p.add_argument("--adaptive_cas", action="store_true",
                    help="Scale guidance strength by CAS value (higher CAS = stronger guidance)")
    p.add_argument("--adaptive_cas_min", type=float, default=0.3,
                    help="Min scale factor for adaptive CAS (when CAS ~ threshold)")

    # Concepts
    p.add_argument("--target_concepts", type=str, nargs="+",
                    default=["nudity", "nude person", "naked body"])
    p.add_argument("--anchor_concepts", type=str, nargs="+",
                    default=["clothed person", "person wearing clothes"])

    # Exemplar mode
    p.add_argument("--concept_dir_path", type=str, default=None)
    p.add_argument("--exemplar_mode", type=str, default="exemplar",
                    choices=["exemplar", "text", "hybrid_exemplar"])
    p.add_argument("--exemplar_weight", type=float, default=0.7)

    # Misc
    p.add_argument("--save_maps", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)

    args = p.parse_args()
    if args.cas_no_sticky:
        args.cas_sticky = False
    if args.exemplar_mode in ("exemplar", "hybrid_exemplar") and args.concept_dir_path is None:
        p.error("--concept_dir_path is required for exemplar and hybrid_exemplar modes")
    return args


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"v10: Prompt-Fidelity Projection Guidance")
    print(f"{'='*70}")
    print(f"  MODE:  {args.exemplar_mode}")
    print(f"  WHEN:  Global CAS threshold={args.cas_threshold}, sticky={args.cas_sticky}")
    print(f"  WHERE: Spatial CAS, neighborhood={args.neighborhood_size}, "
          f"threshold={args.spatial_threshold}, alpha={args.sigmoid_alpha}")
    print(f"  HOW:   {args.guide_mode}, safety_scale={args.safety_scale}")
    if args.adaptive_cas:
        print(f"  ADAPTIVE CAS: min_scale={args.adaptive_cas_min}")
    if args.max_deviation > 0:
        print(f"  MAX DEVIATION: {args.max_deviation}")
    print(f"{'='*70}\n")

    # Load pre-computed concept directions
    target_dirs = anchor_dirs = target_global = anchor_global = None
    if args.exemplar_mode in ("exemplar", "hybrid_exemplar"):
        print(f"Loading concept directions from {args.concept_dir_path} ...")
        concept_data = torch.load(args.concept_dir_path, map_location=device)
        target_dirs = concept_data['target_directions']
        anchor_dirs = concept_data['anchor_directions']
        target_global = concept_data['target_global']
        anchor_global = concept_data['anchor_global']
        print(f"  Loaded directions for {len(target_dirs)} timesteps")

    # Load prompts
    all_prompts = load_prompts(args.prompts)
    end_idx = args.end_idx if args.end_idx > 0 else len(all_prompts)
    prompts_with_idx = list(enumerate(all_prompts))[args.start_idx:end_idx]
    print(f"Loaded {len(all_prompts)} prompts, processing [{args.start_idx}:{end_idx}]")

    # Load pipeline
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

    # Encode concept embeddings
    with torch.no_grad():
        target_embeds = encode_concepts(text_encoder, tokenizer,
                                        args.target_concepts, device)
    anchor_embeds = None
    if args.exemplar_mode in ("text", "hybrid_exemplar"):
        with torch.no_grad():
            anchor_embeds = encode_concepts(text_encoder, tokenizer,
                                            args.anchor_concepts, device)

    with torch.no_grad():
        uncond_inputs = tokenizer("", padding="max_length",
                                  max_length=tokenizer.model_max_length,
                                  truncation=True, return_tensors="pt")
        uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]

    cas = GlobalCAS(threshold=args.cas_threshold, sticky=args.cas_sticky)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.save_maps:
        (outdir / "maps").mkdir(exist_ok=True)

    all_stats = []

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
                prompt_inputs = tokenizer(prompt, padding="max_length",
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
                t_int = t.item()

                with torch.no_grad():
                    lat_batch = torch.cat([lat_in, lat_in])
                    embed_batch = torch.cat([uncond_embeds, prompt_embeds])
                    raw = unet(lat_batch, t, encoder_hidden_states=embed_batch).sample
                    eps_null, eps_prompt_pred = raw.chunk(2)

                eps_cfg = eps_null + args.cfg_scale * (eps_prompt_pred - eps_null)

                # =========================================================
                # Get target/anchor directions based on mode
                # =========================================================
                if args.exemplar_mode == "exemplar":
                    with torch.no_grad():
                        eps_target_online = unet(lat_in, t,
                                                 encoder_hidden_states=target_embeds).sample
                    eps_target_use = eps_target_online
                    d_anchor = anchor_dirs[t_int].to(device, dtype=torch.float16).unsqueeze(0)
                    eps_anchor_use = eps_null + d_anchor
                    cas_val, should_trigger = cas.compute(
                        eps_prompt_pred, eps_null, eps_target=eps_target_online)

                elif args.exemplar_mode == "text":
                    with torch.no_grad():
                        eps_target_online = unet(lat_in, t,
                                                 encoder_hidden_states=target_embeds).sample
                    eps_target_use = eps_target_online
                    cas_val, should_trigger = cas.compute(
                        eps_prompt_pred, eps_null, eps_target=eps_target_online)

                elif args.exemplar_mode == "hybrid_exemplar":
                    d_target_exemplar = target_dirs[t_int].to(device, dtype=torch.float16).unsqueeze(0)
                    d_anchor_exemplar = anchor_dirs[t_int].to(device, dtype=torch.float16).unsqueeze(0)
                    with torch.no_grad():
                        eps_target_online = unet(lat_in, t,
                                                 encoder_hidden_states=target_embeds).sample
                    d_target_online = eps_target_online - eps_null
                    w = args.exemplar_weight
                    d_target_blended = w * d_target_exemplar + (1 - w) * d_target_online
                    eps_target_use = eps_null + d_target_blended
                    d_target_online_global = (eps_target_online - eps_null).reshape(1, -1).float()
                    d_target_exemplar_global = target_global[t_int].to(device).unsqueeze(0).float()
                    d_target_blended_global = w * d_target_exemplar_global + (1 - w) * d_target_online_global
                    cas_val, should_trigger = cas.compute(
                        eps_prompt_pred, eps_null,
                        d_target_global=d_target_blended_global.squeeze(0))

                cas_values.append(cas_val)
                in_window = step_idx >= guide_start_step
                should_guide = should_trigger and in_window

                if should_guide:
                    # Get anchor
                    if args.exemplar_mode == "exemplar":
                        pass  # already set
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
                        eps_anchor_use = eps_null + w * d_anchor_exemplar + (1 - w) * d_anchor_online

                    # Spatial CAS
                    if args.exemplar_mode == "exemplar":
                        spatial_cas = compute_spatial_cas(
                            eps_prompt_pred, eps_null, eps_target_use,
                            neighborhood_size=args.neighborhood_size)
                    elif args.exemplar_mode == "hybrid_exemplar":
                        d_prompt_spatial = (eps_prompt_pred - eps_null).float()
                        spatial_cas = compute_spatial_cas_with_dir(
                            d_prompt_spatial, d_target_blended.float(),
                            neighborhood_size=args.neighborhood_size)
                    else:
                        spatial_cas = compute_spatial_cas(
                            eps_prompt_pred, eps_null, eps_target_use,
                            neighborhood_size=args.neighborhood_size)

                    # Soft mask
                    soft_mask = compute_soft_mask(
                        spatial_cas,
                        spatial_threshold=args.spatial_threshold,
                        sigmoid_alpha=args.sigmoid_alpha,
                        blur_sigma=args.blur_sigma,
                        device=device)

                    # Adaptive CAS scaling
                    t_scale = args.target_scale if args.target_scale > 0 else args.safety_scale
                    a_scale = args.anchor_scale if args.anchor_scale > 0 else args.safety_scale
                    if args.adaptive_cas:
                        # Scale guidance by how far CAS exceeds threshold
                        cas_ratio = (cas_val - args.cas_threshold) / (1.0 - args.cas_threshold + 1e-8)
                        cas_ratio = max(args.adaptive_cas_min, min(1.0, cas_ratio))
                        t_scale = t_scale * cas_ratio
                        a_scale = a_scale * cas_ratio

                    # Apply guidance
                    eps_final = apply_guidance(
                        eps_cfg=eps_cfg,
                        eps_null=eps_null,
                        eps_prompt=eps_prompt_pred,
                        eps_target=eps_target_use,
                        eps_anchor=eps_anchor_use,
                        soft_mask=soft_mask,
                        guide_mode=args.guide_mode,
                        safety_scale=args.safety_scale,
                        cfg_scale=args.cfg_scale,
                        cas_value=cas_val,
                        target_scale=t_scale,
                        anchor_scale=a_scale,
                        max_deviation=args.max_deviation,
                    )

                    guided_count += 1
                    mask_areas.append(float(soft_mask.mean().item()))

                    if args.save_maps and step_idx % 10 == 0:
                        cas_map_np = spatial_cas.float().cpu().numpy()
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

                # DDIM step
                latents_prev = latents.clone()
                latents = scheduler.step(eps_final, t, latents).prev_sample

                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    print(f"  WARNING: NaN/Inf at step {step_idx}, reverting to standard CFG")
                    eps_fallback = eps_null + args.cfg_scale * (eps_prompt_pred - eps_null)
                    latents = scheduler.step(eps_fallback, t, latents_prev).prev_sample

                if args.debug and step_idx % 10 == 0:
                    status = "GUIDED" if should_guide else ("CAS_ON" if should_trigger else "skip")
                    area_s = f" area={mask_areas[-1]:.3f}" if should_guide and mask_areas else ""
                    print(f"  [{step_idx:02d}] t={t.item():.0f} CAS={cas_val:.3f} {status}{area_s}")

            # Decode
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
                "guide_mode": args.guide_mode,
            }
            all_stats.append(stats)

            if prompt_idx % 10 == 0 or args.debug:
                tqdm.write(
                    f"  [{prompt_idx:03d}_{sample_idx}] guided={guided_count}/{total_steps} "
                    f"CAS avg={stats['avg_cas']:.3f} max={stats['max_cas']:.3f} "
                    f"mask={stats['avg_mask_area']:.3f}")

    # Save stats
    n = len(all_stats)
    n_trig = sum(1 for s in all_stats if s["guided_steps"] > 0)
    summary = {
        "method": "v10: Prompt-Fidelity Projection Guidance",
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
    print(f"  Guide mode: {args.guide_mode}")
    print(f"  Output: {outdir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
