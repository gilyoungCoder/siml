#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Direct Exemplar HOW v9: Training-Free Safe Generation

Architecture:
  WHEN (Global CAS): cos(d_prompt, d_target) > tau  — online, same as v4
  WHERE (Spatial CAS): per-pixel cosine similarity   — online, same as v4
  HOW (Direct Exemplar): use pre-computed exemplar directions for guidance
      - exemplar_hybrid:  eps_safe = eps_cfg - t_s * M * d_ex_target + a_s * M * d_ex_anchor
      - exemplar_sld:     eps_safe = eps_cfg - s * M * d_ex_target
      - exemplar_inpaint: eps_safe = eps_cfg * (1 - s*M) + eps_ex_anchor_cfg * (s*M)

Key idea:
  Unlike v8 which projects online d_target onto exemplar subspace,
  v9 directly substitutes exemplar directions into the guidance formula.
  This means HOW is entirely exemplar-driven — the correction direction
  is determined by what "nudity" and "clothed" look like on average,
  not what they look like for the current image.

  Advantage: Consistent concept removal regardless of adversarial prompt tricks
  Risk: May over-correct or under-correct since directions are not image-specific

UNet calls: 3 per guided step (null+prompt batched, target for CAS)
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
# Global CAS (WHEN) — same as v4
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
# Spatial CAS (WHERE) — same as v4
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
# Guidance Application (HOW) — v9 direct exemplar modes
# =============================================================================
def apply_guidance_v9(
    eps_cfg, eps_null, eps_prompt, eps_target,
    soft_mask, guide_mode, safety_scale, cfg_scale,
    d_exemplar_target=None, d_exemplar_anchor=None,
    **kwargs,
):
    """
    v9 guidance using direct exemplar directions.

    Modes:
        exemplar_hybrid: Use exemplar target/anchor directions directly.
            eps_safe = eps_cfg - t_s * M * d_ex_target + a_s * M * d_ex_anchor

        exemplar_sld: Use exemplar target direction (SLD-style removal).
            eps_safe = eps_cfg - s * M * d_ex_target

        exemplar_inpaint: Blend CFG with exemplar anchor CFG.
            eps_anchor_cfg = eps_null + cfg * d_ex_anchor
            eps_safe = eps_cfg * (1 - s*M) + eps_anchor_cfg * (s*M)

        online_hybrid: Fallback to v4-style online hybrid (4 UNet calls, but uses text anchor)
    """
    mask = soft_mask.to(eps_cfg.dtype)

    if guide_mode == "exemplar_hybrid":
        d_ex_tgt = d_exemplar_target.to(eps_cfg.dtype)
        d_ex_anc = d_exemplar_anchor.to(eps_cfg.dtype)
        t_scale = kwargs.get("target_scale", safety_scale)
        a_scale = kwargs.get("anchor_scale", safety_scale)
        eps_final = eps_cfg \
                    - t_scale * mask * d_ex_tgt \
                    + a_scale * mask * d_ex_anc

    elif guide_mode == "exemplar_sld":
        d_ex_tgt = d_exemplar_target.to(eps_cfg.dtype)
        eps_final = eps_cfg - safety_scale * mask * d_ex_tgt

    elif guide_mode == "exemplar_inpaint":
        d_ex_anc = d_exemplar_anchor.to(eps_cfg.dtype)
        eps_anchor_cfg = eps_null + cfg_scale * d_ex_anc
        eps_final = eps_cfg * (1.0 - safety_scale * mask) + eps_anchor_cfg * (safety_scale * mask)

    elif guide_mode == "exemplar_contrast":
        # Contrastive: nudity->clothed direction from exemplars
        # d_transform = d_anchor - d_target (direction from nude to clothed)
        d_ex_tgt = d_exemplar_target.to(eps_cfg.dtype)
        d_ex_anc = d_exemplar_anchor.to(eps_cfg.dtype)
        d_transform = d_ex_anc - d_ex_tgt
        eps_final = eps_cfg + safety_scale * mask * d_transform

    elif guide_mode == "sld":
        # Fallback: online SLD
        eps_final = eps_cfg - safety_scale * mask * (eps_target - eps_null)

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
            concept, padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        embeds = text_encoder(inputs.input_ids.to(device))[0]
        all_embeds.append(embeds)
    return torch.stack(all_embeds).mean(dim=0)


def parse_args():
    p = ArgumentParser(description="Direct Exemplar HOW v9")
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

    # Guidance (How) — v9 modes
    p.add_argument("--guide_mode", type=str, default="exemplar_hybrid",
                    choices=["exemplar_hybrid", "exemplar_sld", "exemplar_inpaint",
                             "exemplar_contrast", "sld"],
                    help="Guidance mode")
    p.add_argument("--safety_scale", type=float, default=1.0)
    p.add_argument("--target_scale", type=float, default=-1.0)
    p.add_argument("--anchor_scale", type=float, default=-1.0)
    p.add_argument("--guide_start_frac", type=float, default=0.0)

    # Concepts (for online target UNet call)
    p.add_argument("--target_concepts", type=str, nargs="+",
                    default=["nudity", "nude person", "naked body"])

    # Exemplar directions
    p.add_argument("--concept_dir_path", type=str, required=True,
                    help="Path to concept_directions.pt")

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
# Main
# =============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"Direct Exemplar HOW v9")
    print(f"{'='*70}")
    print(f"  WHEN:  Global CAS threshold={args.cas_threshold}, sticky={args.cas_sticky}")
    print(f"  WHERE: Spatial CAS, neighborhood={args.neighborhood_size}, "
          f"threshold={args.spatial_threshold}")
    print(f"  HOW:   {args.guide_mode}, safety_scale={args.safety_scale}")
    print(f"  Exemplar directions: {args.concept_dir_path}")
    print(f"  Model: {args.ckpt}")
    print(f"  Steps: {args.steps}, CFG: {args.cfg_scale}, Samples/prompt: {args.nsamples}")
    print(f"{'='*70}\n")

    # Load exemplar directions
    print(f"Loading concept directions from {args.concept_dir_path} ...")
    concept_data = torch.load(args.concept_dir_path, map_location="cpu")
    target_dirs = concept_data['target_directions']
    anchor_dirs = concept_data['anchor_directions']
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

    with torch.no_grad():
        target_embeds = encode_concepts(text_encoder, tokenizer,
                                        args.target_concepts, device)
        uncond_inputs = tokenizer(
            "", padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
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
                prompt_inputs = tokenizer(
                    prompt, padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt"
                )
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
                    eps_null, eps_prompt = raw.chunk(2)

                    # Online target (for CAS only)
                    eps_target = unet(lat_in, t,
                                      encoder_hidden_states=target_embeds).sample

                eps_cfg = eps_null + args.cfg_scale * (eps_prompt - eps_null)

                cas_val, should_trigger = cas.compute(eps_prompt, eps_null, eps_target)
                cas_values.append(cas_val)

                in_window = step_idx >= guide_start_step
                should_guide = should_trigger and in_window

                if should_guide:
                    d_ex_target = target_dirs[t_int].to(device, dtype=torch.float16).unsqueeze(0)
                    d_ex_anchor = anchor_dirs[t_int].to(device, dtype=torch.float16).unsqueeze(0)

                    # Spatial CAS — online
                    spatial_cas = compute_spatial_cas(
                        eps_prompt, eps_null, eps_target,
                        neighborhood_size=args.neighborhood_size,
                    )
                    soft_mask = compute_soft_mask(
                        spatial_cas,
                        spatial_threshold=args.spatial_threshold,
                        sigmoid_alpha=args.sigmoid_alpha,
                        blur_sigma=args.blur_sigma,
                        device=device,
                    )

                    eps_final = apply_guidance_v9(
                        eps_cfg=eps_cfg,
                        eps_null=eps_null,
                        eps_prompt=eps_prompt,
                        eps_target=eps_target,
                        soft_mask=soft_mask,
                        guide_mode=args.guide_mode,
                        safety_scale=args.safety_scale,
                        cfg_scale=args.cfg_scale,
                        d_exemplar_target=d_ex_target,
                        d_exemplar_anchor=d_ex_anchor,
                        target_scale=args.target_scale if args.target_scale > 0 else args.safety_scale,
                        anchor_scale=args.anchor_scale if args.anchor_scale > 0 else args.safety_scale,
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

                latents_prev = latents.clone()
                latents = scheduler.step(eps_final, t, latents).prev_sample

                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    print(f"  WARNING: NaN/Inf at step {step_idx}, reverting to standard CFG")
                    eps_fallback = eps_null + args.cfg_scale * (eps_prompt - eps_null)
                    latents = scheduler.step(eps_fallback, t, latents_prev).prev_sample

                if args.debug and step_idx % 10 == 0:
                    status = "GUIDED" if should_guide else ("CAS_ON" if should_trigger else "skip")
                    area_s = f" area={mask_areas[-1]:.3f}" if should_guide and mask_areas else ""
                    print(f"  [{step_idx:02d}] t={t.item():.0f} CAS={cas_val:.3f} {status}{area_s}")

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
            }
            all_stats.append(stats)

            if prompt_idx % 10 == 0 or args.debug:
                tqdm.write(
                    f"  [{prompt_idx:03d}_{sample_idx}] guided={guided_count}/{total_steps} "
                    f"CAS avg={stats['avg_cas']:.3f} max={stats['max_cas']:.3f} "
                    f"mask_area={stats['avg_mask_area']:.3f}"
                )

    n = len(all_stats)
    n_trig = sum(1 for s in all_stats if s["guided_steps"] > 0)
    summary = {
        "method": "Direct Exemplar HOW v9",
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
    print(f"  Guide mode: {args.guide_mode}, safety_scale={args.safety_scale}")
    print(f"  Avg guided steps: {summary['overall']['avg_guided_steps']:.1f}/{total_steps if n else 0}")
    print(f"  Avg mask area: {summary['overall']['avg_mask_area']:.3f}")
    print(f"  Output: {outdir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
