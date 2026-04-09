#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v21: Adaptive Anchor Inpainting — the final refinement of v4.

v4 achieved 96.5% SR / 0% Full / 3.5% NR (Qwen3-VL, Ring-A-Bell).
The 3.5% NR = over-erasure (person disappears entirely).

v21 diagnosis: NR occurs when mask covers most of the image AND
safety_scale is high — the anchor completely replaces the person.

v21 innovations (all on top of v4's proven architecture):

  1. Area-Adaptive Safety Scale (--area_damp):
     When mask area > area_threshold (default 0.4), linearly reduce
     safety_scale toward damp_min (default 0.3). Prevents person from
     disappearing when large regions are flagged.

     effective_ss = ss * (1 - damp_strength * clamp((area - area_thr) / (1 - area_thr)))

  2. Temporal Decay (--temporal_decay):
     Reduce guidance strength in later denoising steps. Early steps set
     composition (must be safe), late steps add fine details (lighter touch).

     temporal_factor = 1.0           if t_frac < decay_start
                     = linear decay  from 1.0 to decay_floor

  3. Mask Gamma (--mask_gamma):
     Apply power law to soft mask: mask^gamma. gamma > 1 concentrates
     strong erasure on high-confidence regions, leaves borderline areas
     with lighter guidance. Reduces collateral over-erasure.

  4. Minimum Preserve (--min_preserve):
     Hard floor: at least (1 - min_preserve) of the original CFG output
     is always preserved. Prevents complete replacement even in
     heavily-masked regions. Default 0.15 = always keep 15% original.

Architecture (same as v4):
  WHEN: Noise CAS (threshold=0.6, sticky)
  WHERE: Noise spatial CAS (3x3 neighborhood, sigmoid soft mask)
  HOW: Anchor inpainting with adaptive scaling

Evidence-based defaults:
  v4 best: ainp_s1.0_t0.1 → SR 96.5%, NR 3.5%, Full 0%
  v21 target: SR >= 96%, NR < 2%, Full 0%
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

from diffusers import StableDiffusionPipeline, DDIMScheduler


# =============================================================================
# Global CAS (WHEN) — same as v4
# =============================================================================
class GlobalCAS:
    def __init__(self, threshold=0.6, sticky=True):
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
# Spatial CAS + Soft Mask (WHERE) — same as v4
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


def gaussian_blur_2d(x, kernel_size=5, sigma=1.0):
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    kernel_h = g.view(1, 1, kernel_size, 1)
    kernel_w = g.view(1, 1, 1, kernel_size)
    p = kernel_size // 2
    x = F.pad(x, [0, 0, p, p], mode='reflect')
    x = F.conv2d(x, kernel_h.expand(x.shape[1], -1, -1, -1), groups=x.shape[1])
    x = F.pad(x, [p, p, 0, 0], mode='reflect')
    x = F.conv2d(x, kernel_w.expand(x.shape[1], -1, -1, -1), groups=x.shape[1])
    return x


def compute_soft_mask(spatial_cas, spatial_threshold=0.3, sigmoid_alpha=10.0,
                      blur_sigma=1.0, mask_gamma=1.0, device=None):
    """Compute soft mask with optional gamma power-law."""
    soft_mask = torch.sigmoid(sigmoid_alpha * (spatial_cas - spatial_threshold))
    soft_mask = soft_mask.unsqueeze(0).unsqueeze(0)
    if blur_sigma > 0:
        soft_mask = gaussian_blur_2d(soft_mask, kernel_size=5, sigma=blur_sigma)
    soft_mask = soft_mask.clamp(0, 1)

    # Gamma: concentrate strong erasure on high-confidence pixels
    if mask_gamma != 1.0:
        soft_mask = soft_mask.pow(mask_gamma)

    return soft_mask


# =============================================================================
# Adaptive Anchor Inpainting (HOW) — v21 innovation
# =============================================================================
def apply_adaptive_anchor_inpaint(
    eps_cfg, eps_null, eps_anchor, soft_mask, cfg_scale,
    safety_scale, step_frac,
    # v21 adaptive parameters
    area_damp=True, area_threshold=0.4, damp_strength=0.7,
    temporal_decay=True, decay_start=0.4, decay_floor=0.3,
    min_preserve=0.15,
):
    """
    Adaptive anchor inpainting with area dampening and temporal decay.

    Base formula (v4): eps_final = eps_cfg * (1 - ss*M) + eps_anchor_cfg * (ss*M)

    v21 adaptations:
    1. Area dampening: reduce ss when mask covers large portion of image
    2. Temporal decay: reduce ss in later denoising steps
    3. Min preserve: always keep at least (1-min_preserve) of original
    """
    mask = soft_mask.to(eps_cfg.dtype)
    mask_area = mask.mean().item()

    effective_ss = safety_scale

    # 1. Area dampening: prevent person from disappearing
    if area_damp and mask_area > area_threshold:
        # Linear reduction from 1.0 to (1 - damp_strength)
        excess = min((mask_area - area_threshold) / max(1.0 - area_threshold, 0.01), 1.0)
        area_factor = 1.0 - damp_strength * excess
        effective_ss *= area_factor

    # 2. Temporal decay: lighter touch in later steps
    if temporal_decay and step_frac > decay_start:
        # Linear decay from 1.0 to decay_floor
        decay_progress = min((step_frac - decay_start) / max(1.0 - decay_start, 0.01), 1.0)
        temporal_factor = 1.0 - (1.0 - decay_floor) * decay_progress
        effective_ss *= temporal_factor

    # 3. Compute anchor-guided output
    eps_anchor_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)

    blend_weight = effective_ss * mask

    # 4. Min preserve: always keep some original signal
    if min_preserve > 0:
        blend_weight = blend_weight.clamp(max=1.0 - min_preserve)

    eps_final = eps_cfg * (1.0 - blend_weight) + eps_anchor_cfg * blend_weight

    # NaN/Inf guard
    if torch.isnan(eps_final).any() or torch.isinf(eps_final).any():
        eps_final = torch.where(torch.isfinite(eps_final), eps_final, eps_cfg)

    return eps_final, effective_ss, mask_area


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
    p = ArgumentParser(description="v21: Adaptive Anchor Inpainting")

    # Model & I/O
    p.add_argument("--ckpt", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--prompts", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--nsamples", type=int, default=1)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg_scale", type=float, default=7.5)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)

    # WHEN — Global CAS (same as v4)
    p.add_argument("--cas_threshold", type=float, default=0.6)
    p.add_argument("--cas_sticky", action="store_true", default=True)
    p.add_argument("--cas_no_sticky", action="store_true")

    # WHERE — Noise Spatial CAS (same as v4)
    p.add_argument("--spatial_threshold", type=float, default=0.3)
    p.add_argument("--sigmoid_alpha", type=float, default=10.0)
    p.add_argument("--neighborhood_size", type=int, default=3)
    p.add_argument("--blur_sigma", type=float, default=1.0)

    # WHERE — Mask Gamma (v21)
    p.add_argument("--mask_gamma", type=float, default=1.0,
                   help="Power-law exponent for mask. >1 = concentrate on high-confidence. "
                        "1.0 = v4 behavior. Try 1.5 or 2.0.")

    # HOW — Base guidance (same as v4)
    p.add_argument("--safety_scale", type=float, default=1.0,
                   help="Base safety scale for anchor inpainting")
    p.add_argument("--guide_start_frac", type=float, default=0.0)

    # HOW — v21 Area Dampening
    p.add_argument("--area_damp", action="store_true", default=False,
                   help="Enable area-adaptive safety scale reduction")
    p.add_argument("--area_threshold", type=float, default=0.4,
                   help="Mask area above which dampening kicks in")
    p.add_argument("--damp_strength", type=float, default=0.7,
                   help="Max reduction factor (0.7 = reduce ss by up to 70%)")

    # HOW — v21 Temporal Decay
    p.add_argument("--temporal_decay", action="store_true", default=False,
                   help="Enable temporal decay of safety scale")
    p.add_argument("--decay_start", type=float, default=0.4,
                   help="Fraction of steps after which decay begins")
    p.add_argument("--decay_floor", type=float, default=0.3,
                   help="Minimum temporal factor (1.0=no decay, 0.0=fully off)")

    # HOW — v21 Min Preserve
    p.add_argument("--min_preserve", type=float, default=0.0,
                   help="Always preserve this fraction of original CFG. "
                        "0.15 = keep 15%% original even in fully-masked regions")

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
    return args


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Label for adaptive features
    features = []
    if args.area_damp:
        features.append(f"area_damp(thr={args.area_threshold},str={args.damp_strength})")
    if args.temporal_decay:
        features.append(f"temporal(start={args.decay_start},floor={args.decay_floor})")
    if args.mask_gamma != 1.0:
        features.append(f"gamma={args.mask_gamma}")
    if args.min_preserve > 0:
        features.append(f"preserve={args.min_preserve}")
    feat_str = " + ".join(features) if features else "none (v4 baseline)"

    print(f"\n{'='*70}")
    print(f"v21: Adaptive Anchor Inpainting")
    print(f"{'='*70}")
    print(f"  WHEN:     CAS threshold={args.cas_threshold}, sticky={args.cas_sticky}")
    print(f"  WHERE:    noise spatial CAS (nbr={args.neighborhood_size}, "
          f"thr={args.spatial_threshold}, gamma={args.mask_gamma})")
    print(f"  HOW:      anchor_inpaint, ss={args.safety_scale}")
    print(f"  ADAPTIVE: {feat_str}")
    print(f"  Targets:  {args.target_concepts}")
    print(f"  Anchors:  {args.anchor_concepts}")
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
            effective_scales = []

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
                step_frac = step_idx / max(total_steps - 1, 1)

                with torch.no_grad():
                    lat_batch = torch.cat([lat_in, lat_in])
                    embed_batch = torch.cat([uncond_embeds, prompt_embeds])
                    raw = unet(lat_batch, t, encoder_hidden_states=embed_batch).sample
                    eps_null, eps_prompt_pred = raw.chunk(2)
                    eps_target = unet(lat_in, t,
                                      encoder_hidden_states=target_embeds).sample

                eps_cfg = eps_null + args.cfg_scale * (eps_prompt_pred - eps_null)

                cas_val, should_trigger = cas.compute(
                    eps_prompt_pred, eps_null, eps_target)
                cas_values.append(cas_val)
                in_window = step_idx >= guide_start_step
                should_guide = should_trigger and in_window

                if should_guide:
                    with torch.no_grad():
                        eps_anchor = unet(lat_in, t,
                                          encoder_hidden_states=anchor_embeds).sample

                    # WHERE
                    spatial_cas = compute_spatial_cas(
                        eps_prompt_pred, eps_null, eps_target,
                        neighborhood_size=args.neighborhood_size)
                    soft_mask = compute_soft_mask(
                        spatial_cas,
                        spatial_threshold=args.spatial_threshold,
                        sigmoid_alpha=args.sigmoid_alpha,
                        blur_sigma=args.blur_sigma,
                        mask_gamma=args.mask_gamma,
                        device=device)

                    # HOW — Adaptive anchor inpainting
                    eps_final, eff_ss, area = apply_adaptive_anchor_inpaint(
                        eps_cfg=eps_cfg,
                        eps_null=eps_null,
                        eps_anchor=eps_anchor,
                        soft_mask=soft_mask,
                        cfg_scale=args.cfg_scale,
                        safety_scale=args.safety_scale,
                        step_frac=step_frac,
                        area_damp=args.area_damp,
                        area_threshold=args.area_threshold,
                        damp_strength=args.damp_strength,
                        temporal_decay=args.temporal_decay,
                        decay_start=args.decay_start,
                        decay_floor=args.decay_floor,
                        min_preserve=args.min_preserve,
                    )

                    guided_count += 1
                    mask_areas.append(area)
                    effective_scales.append(eff_ss)

                    if args.debug and step_idx % 10 == 0:
                        print(f"  [{step_idx:02d}] CAS={cas_val:.3f} GUIDED "
                              f"area={area:.3f} eff_ss={eff_ss:.3f}")

                    if args.save_maps and step_idx % 10 == 0:
                        prefix = f"{prompt_idx:04d}_{sample_idx:02d}_s{step_idx:03d}"
                        maps_dir = outdir / "maps"
                        cas_np = spatial_cas.float().cpu().numpy()
                        cas_np = np.nan_to_num(cas_np, nan=0.0)
                        cas_img = (np.clip((cas_np + 1) / 2, 0, 1) * 255).astype(np.uint8)
                        Image.fromarray(cas_img, 'L').save(
                            str(maps_dir / f"{prefix}_cas.png"))
                        m_np = soft_mask[0, 0].float().cpu().numpy()
                        Image.fromarray((np.clip(m_np, 0, 1) * 255).astype(np.uint8), 'L').save(
                            str(maps_dir / f"{prefix}_mask.png"))
                else:
                    eps_final = eps_cfg
                    if args.debug and step_idx % 10 == 0:
                        status = "CAS_ON" if should_trigger else "skip"
                        print(f"  [{step_idx:02d}] CAS={cas_val:.3f} {status}")

                latents_prev = latents.clone()
                latents = scheduler.step(eps_final, t, latents).prev_sample

                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    print(f"  WARNING: NaN/Inf at step {step_idx}, falling back")
                    eps_fallback = eps_null + args.cfg_scale * (eps_prompt_pred - eps_null)
                    latents = scheduler.step(eps_fallback, t, latents_prev).prev_sample

            # Decode
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
                "mean_mask_area": float(np.mean(mask_areas)) if mask_areas else 0,
                "mean_effective_ss": float(np.mean(effective_scales)) if effective_scales else 0,
            }
            all_stats.append(stats)

    # Save stats
    stats_path = outdir / "generation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    args_path = outdir / "args.json"
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    guided_imgs = sum(1 for s in all_stats if s["guided_steps"] > 0)
    avg_area = np.mean([s["mean_mask_area"] for s in all_stats if s["mean_mask_area"] > 0])
    avg_eff_ss = np.mean([s["mean_effective_ss"] for s in all_stats if s["mean_effective_ss"] > 0])
    print(f"\nDone! {len(all_stats)} images generated.")
    print(f"  Guided: {guided_imgs}/{len(all_stats)} "
          f"({100*guided_imgs/max(len(all_stats),1):.1f}%)")
    print(f"  Avg mask area: {avg_area:.3f}")
    print(f"  Avg effective ss: {avg_eff_ss:.3f}")
    print(f"  Stats: {stats_path}")


if __name__ == "__main__":
    main()
