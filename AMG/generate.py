#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Activation Matching Guidance (AMG): Training-Free Safe Generation

Key Difference from CAS+SpatialCFG:
  - CAS compares NOISE OUTPUTS (ε) — global, no natural spatial info
  - AMG compares UNet INTERMEDIATE FEATURES (h-space) — naturally spatial

Architecture:
  WHEN: h-space global similarity — cos(h_prompt_global, h_target_global) > τ
  WHERE: per-pixel feature similarity at 16×16, 32×32 decoder layers
         score(x,y) = cos(h_prompt(x,y), h_target(x,y)) - cos(h_prompt(x,y), h_anchor(x,y))
  HOW: SLD-style safe CFG — ε_safe = ε_cfg - s * spatial_weight * (ε_target - ε_∅)

h-space (UNet bottleneck, 8×8) is known to carry the highest semantic info
(proven by SDID/Asyrp papers). Decoder layers (16×16, 32×32) add spatial detail.
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
# Feature Extractor: Hook into UNet intermediate layers
# =============================================================================
class UNetFeatureExtractor:
    """
    Extract intermediate features from UNet at multiple resolutions.

    Hooks into:
      - mid_block output (h-space, 8×8): highest semantic content
      - up_blocks[0] (16×16): semantic + some spatial
      - up_blocks[1] (32×32): balanced semantic-spatial
    """

    def __init__(self, unet):
        self.unet = unet
        self.features = {}
        self.hooks = []
        self._install_hooks()

    def _install_hooks(self):
        # Mid block (h-space, 8×8) — highest semantics
        self.hooks.append(
            self.unet.mid_block.register_forward_hook(
                self._make_hook("mid_8x8")
            )
        )

        # Up blocks — decoder layers with spatial info
        # up_blocks[0]: 16×16, up_blocks[1]: 32×32, up_blocks[2]: 64×64
        for i, block in enumerate(self.unet.up_blocks):
            if i <= 1:  # 16×16 and 32×32 only (skip 64×64, too noisy)
                res = [16, 32, 64][i]
                self.hooks.append(
                    block.register_forward_hook(
                        self._make_hook(f"up_{res}x{res}")
                    )
                )

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            # output can be a tuple; take the first element (hidden states)
            if isinstance(output, tuple):
                feat = output[0]
            else:
                feat = output
            self.features[name] = feat.detach()
        return hook_fn

    def clear(self):
        self.features.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_features(self) -> Dict[str, torch.Tensor]:
        """Return dict of {layer_name: (batch, channels, H, W)} features."""
        return dict(self.features)


# =============================================================================
# WHEN: Feature-based concept detection
# =============================================================================
class FeatureDetector:
    """
    Detect harmful concept alignment using h-space feature similarity.

    Unlike CAS (which compares noise outputs globally), this compares
    intermediate UNet features which are more semantically meaningful.
    """

    def __init__(self, threshold: float = 0.3, sticky: bool = True,
                 sticky_min_count: int = 1):
        self.threshold = threshold
        self.sticky = sticky
        self.sticky_min_count = sticky_min_count
        self.triggered = False
        self.consecutive_count = 0

    def reset(self):
        self.triggered = False
        self.consecutive_count = 0

    def compute(self, feat_prompt: Dict, feat_target: Dict, feat_uncond: Dict):
        """
        Compute feature-based concept alignment score.

        Uses h-space (8×8) for global detection — most semantic layer.
        """
        # Use mid_8x8 (h-space) for global detection
        h_prompt = feat_prompt.get("mid_8x8")
        h_target = feat_target.get("mid_8x8")
        h_uncond = feat_uncond.get("mid_8x8")

        if h_prompt is None or h_target is None or h_uncond is None:
            return 0.0, self.triggered if self.sticky else False

        # Compute directions in h-space (analogous to CAS but in feature space)
        d_prompt = (h_prompt - h_uncond).reshape(1, -1).float()
        d_target = (h_target - h_uncond).reshape(1, -1).float()

        score = F.cosine_similarity(d_prompt, d_target, dim=-1).item()

        if math.isnan(score) or math.isinf(score):
            return 0.0, self.triggered if self.sticky else False

        if self.sticky and self.triggered:
            return score, True

        exceeds = score > self.threshold
        if exceeds:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0

        should_guide = self.consecutive_count >= self.sticky_min_count

        if should_guide and self.sticky:
            self.triggered = True

        return score, should_guide


# =============================================================================
# WHERE: Spatial feature matching
# =============================================================================
def compute_feature_spatial_map(
    feat_prompt: Dict,
    feat_target: Dict,
    feat_anchor: Dict,
    feat_uncond: Dict,
    target_size: int = 64,
) -> torch.Tensor:
    """
    Compute spatial guidance map from UNet intermediate features.

    For each spatial position, computes:
      score(x,y) = cos(d_prompt(x,y), d_target(x,y)) - cos(d_prompt(x,y), d_anchor(x,y))

    Uses 16×16 and 32×32 features (semantic + spatial).

    Returns: (1, 1, target_size, target_size) in [0, 1]
    """
    maps = []

    for layer_name in ["up_16x16", "up_32x32"]:
        h_p = feat_prompt.get(layer_name)
        h_t = feat_target.get(layer_name)
        h_a = feat_anchor.get(layer_name)
        h_u = feat_uncond.get(layer_name)

        if h_p is None or h_t is None or h_a is None or h_u is None:
            continue

        # Directions in feature space
        d_prompt = (h_p - h_u).float()   # (1, C, H, W)
        d_target = (h_t - h_u).float()
        d_anchor = (h_a - h_u).float()

        # Per-pixel cosine similarity
        cos_target = F.cosine_similarity(d_prompt, d_target, dim=1).unsqueeze(1)  # (1, 1, H, W)
        cos_anchor = F.cosine_similarity(d_prompt, d_anchor, dim=1).unsqueeze(1)

        # Where target aligns more than anchor
        score = F.relu(cos_target - cos_anchor)

        # Interpolate to target size
        if score.shape[-1] != target_size:
            score = F.interpolate(score, size=(target_size, target_size),
                                  mode='bilinear', align_corners=False)

        maps.append(score)

    if not maps:
        return torch.zeros(1, 1, target_size, target_size,
                           device=next(iter(feat_prompt.values())).device)

    # Average across layers
    spatial = torch.stack(maps).mean(dim=0)

    # Normalize to [0, 1]
    smin, smax = spatial.min(), spatial.max()
    if smax - smin > 1e-8:
        spatial = (spatial - smin) / (smax - smin)
    else:
        spatial = torch.zeros_like(spatial)

    return spatial.clamp(0, 1)


# =============================================================================
# HOW: Safe CFG (same as CAS_SpatialCFG v3)
# =============================================================================
def apply_safe_cfg(
    noise_cfg, noise_uncond, noise_target, noise_anchor,
    mask, weight,
    mode="sld", safety_scale=3.0,
):
    d_target = noise_target - noise_uncond

    if mode == "sld":
        correction = safety_scale * weight * d_target
        noise_safe = noise_cfg - correction
    elif mode == "anchor_shift":
        d_shift = noise_target - noise_anchor
        correction = safety_scale * weight * d_shift
        noise_safe = noise_cfg - correction
    elif mode == "dual":
        correction = safety_scale * weight * (d_target - (noise_anchor - noise_uncond))
        noise_safe = noise_cfg - correction
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if torch.isnan(noise_safe).any() or torch.isinf(noise_safe).any():
        noise_safe = torch.where(torch.isfinite(noise_safe), noise_safe, noise_cfg)
    return noise_safe


# =============================================================================
# Utils
# =============================================================================
def load_prompts(f):
    f = Path(f)
    if f.suffix == ".csv":
        prompts = []
        with open(f, "r") as fp:
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
                prompts.append(row[prompt_col].strip())
        return prompts
    else:
        return [l.strip() for l in open(f) if l.strip()]


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def parse_args():
    p = ArgumentParser(description="Activation Matching Guidance (AMG)")
    p.add_argument("--ckpt", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--prompts", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--nsamples", type=int, default=4)
    p.add_argument("--cfg_scale", type=float, default=7.5)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    # Detection (When)
    p.add_argument("--det_threshold", type=float, default=0.3,
                   help="Feature-based detection threshold")
    p.add_argument("--det_sticky", action="store_true", default=True)
    p.add_argument("--det_no_sticky", action="store_true")
    p.add_argument("--det_min_count", type=int, default=1)

    # Concepts
    p.add_argument("--target", type=str, nargs="+",
                   default=["nudity", "nude person", "naked body"])
    p.add_argument("--anchor", type=str, nargs="+",
                   default=["clothed person", "person wearing clothes", "fully dressed person"])

    # Spatial (Where)
    p.add_argument("--spatial_threshold", type=float, default=0.3)

    # Guidance (How)
    p.add_argument("--guide_mode", type=str, default="sld",
                   choices=["sld", "anchor_shift", "dual"])
    p.add_argument("--safety_scale", type=float, default=3.0)
    p.add_argument("--warmup_steps", type=int, default=5)

    # Misc
    p.add_argument("--save_maps", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)

    args = p.parse_args()
    if args.det_no_sticky:
        args.det_sticky = False
    return args


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"Activation Matching Guidance (AMG)")
    print(f"{'='*70}")
    print(f"  WHEN: h-space similarity, threshold={args.det_threshold}, sticky={args.det_sticky}")
    print(f"  WHERE: feature spatial map (16×16 + 32×32), threshold={args.spatial_threshold}")
    print(f"  HOW: {args.guide_mode}, safety_scale={args.safety_scale}")
    print(f"  Target: {args.target}")
    print(f"  Anchor: {args.anchor}")
    print(f"{'='*70}\n")

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

    # Pre-encode concepts
    with torch.no_grad():
        def encode_text(text):
            inputs = tokenizer(text, padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True, return_tensors="pt")
            return text_encoder(inputs.input_ids.to(device))[0]

        target_embeds = encode_text(", ".join(args.target))
        anchor_embeds = encode_text(", ".join(args.anchor))
        uncond_embeds = encode_text("")

    # Feature extractor
    feat_extractor = UNetFeatureExtractor(unet)
    detector = FeatureDetector(
        threshold=args.det_threshold, sticky=args.det_sticky,
        sticky_min_count=args.det_min_count
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.save_maps:
        (outdir / "maps").mkdir(exist_ok=True)

    all_stats = []

    for prompt_idx, prompt in tqdm(prompts_with_idx, desc="Generating"):
        for sample_idx in range(args.nsamples):
            seed = args.seed + prompt_idx * args.nsamples + sample_idx
            set_seed(seed)
            detector.reset()

            guided_count = 0
            det_values = []
            areas = []

            with torch.no_grad():
                text_embeds = encode_text(prompt)

            set_seed(seed)
            latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
            latents = latents * scheduler.init_noise_sigma
            scheduler.set_timesteps(args.steps, device=device)
            total_steps = len(scheduler.timesteps)

            for step_idx, t in enumerate(scheduler.timesteps):
                # --- Pass 1: Standard CFG (uncond + text) ---
                lat_in = torch.cat([latents] * 2)
                lat_in = scheduler.scale_model_input(lat_in, t)

                feat_extractor.clear()
                with torch.no_grad():
                    raw = unet(lat_in, t,
                               encoder_hidden_states=torch.cat([uncond_embeds, text_embeds])
                               ).sample
                noise_uncond, noise_text = raw.chunk(2)
                noise_cfg = noise_uncond + args.cfg_scale * (noise_text - noise_uncond)

                # Capture features from the text-conditioned pass
                # The features are from the batched pass, split them
                feat_all = feat_extractor.get_features()
                feat_uncond_dict = {}
                feat_prompt_dict = {}
                for k, v in feat_all.items():
                    feat_uncond_dict[k] = v[:1]  # first half = uncond
                    feat_prompt_dict[k] = v[1:]  # second half = text

                # --- Pass 2: Target concept ---
                lat_single = scheduler.scale_model_input(latents, t)
                feat_extractor.clear()
                with torch.no_grad():
                    noise_target = unet(lat_single, t,
                                        encoder_hidden_states=target_embeds).sample
                feat_target_dict = feat_extractor.get_features()

                # --- WHEN: Feature-based detection ---
                det_score, det_triggered = detector.compute(
                    feat_prompt_dict, feat_target_dict, feat_uncond_dict
                )
                det_values.append(det_score)

                should_guide = det_triggered

                if should_guide:
                    # Warmup
                    if step_idx < args.warmup_steps:
                        current_scale = args.safety_scale * (step_idx + 1) / args.warmup_steps
                    else:
                        current_scale = args.safety_scale

                    # --- Pass 3: Anchor concept ---
                    feat_extractor.clear()
                    with torch.no_grad():
                        noise_anchor = unet(lat_single, t,
                                            encoder_hidden_states=anchor_embeds).sample
                    feat_anchor_dict = feat_extractor.get_features()

                    # --- WHERE: Feature spatial map ---
                    spatial_map = compute_feature_spatial_map(
                        feat_prompt_dict, feat_target_dict,
                        feat_anchor_dict, feat_uncond_dict,
                        target_size=64,
                    )

                    # Create mask and weight
                    mask = (spatial_map > args.spatial_threshold).float()
                    weight = spatial_map * mask
                    wmax = weight.max()
                    if wmax > 1e-8:
                        weight = weight / wmax

                    # --- HOW: Apply safe CFG ---
                    noise_cfg = apply_safe_cfg(
                        noise_cfg, noise_uncond, noise_target, noise_anchor,
                        mask.to(noise_cfg.dtype), weight.to(noise_cfg.dtype),
                        mode=args.guide_mode,
                        safety_scale=current_scale,
                    )

                    guided_count += 1
                    area_val = float(mask.mean().item())
                    areas.append(area_val)

                    if args.save_maps and step_idx % 10 == 0:
                        m = spatial_map[0, 0].float().cpu().numpy()
                        img_map = (np.clip(m, 0, 1) * 255).astype(np.uint8)
                        Image.fromarray(img_map, 'L').save(
                            str(outdir / "maps" / f"{prompt_idx:04d}_{sample_idx:02d}_s{step_idx:03d}.png"))

                # --- DDIM step ---
                latents_prev = latents.clone()
                latents = scheduler.step(noise_cfg, t, latents).prev_sample

                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    noise_fallback = noise_uncond + args.cfg_scale * (noise_text - noise_uncond)
                    latents = scheduler.step(noise_fallback, t, latents_prev).prev_sample

                if args.debug and step_idx % 10 == 0:
                    status = "GUIDED" if should_guide else ("DET_ON" if det_triggered else "skip")
                    area_s = f" area={areas[-1]:.3f}" if should_guide and areas else ""
                    print(f"  [{step_idx:02d}] t={t.item()} det={det_score:.3f} {status}{area_s}")

            # Decode
            with torch.no_grad():
                dec = vae.decode(latents.to(vae.dtype) / vae.config.scaling_factor).sample
                dec = (dec / 2 + 0.5).clamp(0, 1)
                img = (dec[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

            safe_name = "".join(c if c.isalnum() or c in ' -_' else '_' for c in prompt)[:50].replace(' ', '_')
            fname = f"{prompt_idx:04d}_{sample_idx:02d}_{safe_name}.png"
            Image.fromarray(img).resize((512, 512)).save(str(outdir / fname))

            stats = {
                "prompt_idx": prompt_idx, "sample_idx": sample_idx,
                "seed": seed, "prompt": prompt[:100], "filename": fname,
                "guided_steps": guided_count,
                "total_steps": total_steps,
                "guidance_ratio": guided_count / max(total_steps, 1),
                "det_triggered": detector.triggered,
                "avg_det": float(np.mean(det_values)) if det_values else 0,
                "max_det": float(np.max(det_values)) if det_values else 0,
                "avg_area": float(np.mean(areas)) if areas else 0,
            }
            all_stats.append(stats)

            if prompt_idx % 10 == 0 or args.debug:
                print(f"  [{prompt_idx:03d}_{sample_idx}] guided={guided_count}/{total_steps} "
                      f"det avg={stats['avg_det']:.3f} area={stats['avg_area']:.3f}")

    # Save summary
    n = len(all_stats)
    n_trig = sum(1 for s in all_stats if s["guided_steps"] > 0)
    summary = {
        "method": "Activation Matching Guidance (AMG)",
        "args": vars(args),
        "overall": {
            "total_images": n,
            "triggered": n_trig,
            "avg_guided": float(np.mean([s["guided_steps"] for s in all_stats])) if n else 0,
            "avg_det": float(np.mean([s["avg_det"] for s in all_stats if s["avg_det"] > 0])) if n else 0,
            "avg_area": float(np.mean([s["avg_area"] for s in all_stats if s["avg_area"] > 0])) if n else 0,
        },
        "per_image": all_stats,
    }
    with open(outdir / "stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"DONE! {n} images, {n_trig} triggered ({100*n_trig/max(n,1):.0f}%)")
    print(f"Output: {outdir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
