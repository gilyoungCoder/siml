#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAS + Spatial CFG v3: Training-Free Safe Generation

Architecture:
  WHEN (CAS): cos(d_prompt, d_target) > τ → trigger guidance
  WHERE (Cross-Attention Spatial Map): extract attention for target tokens from UNet
  HOW (Safe CFG): ε_safe = ε_cfg - s_safety * M * (ε_target - ε_∅)
                  Following SLD/DAG formulation — subtracts target direction in masked regions

Key fixes from v2:
  - No more black images: uses SLD-style subtraction instead of additive correction
  - Cross-attention based spatial maps (semantic, not noise-based)
  - Proper scale ranges (1-10 like SLD, not 0-1)
  - Safety checker explicitly disabled
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
# CAS: Concept Alignment Score (WHEN)
# =============================================================================
class CAS:
    """Detect harmful prompt alignment via cosine(d_prompt, d_target)."""

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

    def compute(self, noise_text, noise_uncond, noise_target):
        d_prompt = (noise_text - noise_uncond).reshape(1, -1).float()
        d_target = (noise_target - noise_uncond).reshape(1, -1).float()
        cas = F.cosine_similarity(d_prompt, d_target, dim=-1).item()

        if math.isnan(cas) or math.isinf(cas):
            return 0.0, self.triggered if self.sticky else False

        if self.sticky and self.triggered:
            return cas, True

        exceeds = cas > self.threshold
        if exceeds:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0

        should_guide = self.consecutive_count >= self.sticky_min_count

        if should_guide and self.sticky:
            self.triggered = True

        return cas, should_guide


# =============================================================================
# Cross-Attention Spatial Map Extractor (WHERE)
# =============================================================================
class CrossAttnMapExtractor:
    """
    Hook into UNet cross-attention layers to extract spatial attention maps.

    During UNet forward pass, captures attention probabilities for each
    cross-attention layer. These maps show WHERE in the image each text
    token is being attended to.

    We compare attention for target tokens ("nude person") vs anchor tokens
    ("clothed person") to create a spatial mask of nudity regions.
    """

    def __init__(self, unet, use_layers="mid_high"):
        """
        use_layers: which resolution layers to use
            "mid_high": 16x16 and 32x32 (best semantic info, DAG default)
            "all": all layers including 64x64
        """
        self.unet = unet
        self.use_layers = use_layers
        self.hooks = []
        self.attn_maps = {}
        self._install_hooks()

    def _install_hooks(self):
        """Install forward hooks on all cross-attention (attn2) layers."""
        for name, module in self.unet.named_modules():
            if hasattr(module, 'to_q') and 'attn2' in name:
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            # input[0] is the hidden state (queries source)
            hidden = input[0]
            batch_size = hidden.shape[0]

            # We need to compute attention ourselves since the module
            # already computed it but didn't store it
            # Store the hidden state and module reference for later computation
            self.attn_maps[name] = {
                'hidden': hidden.detach(),
                'module': module,
            }
        return hook_fn

    def compute_cam(self, target_embeds, layer_filter=None):
        """
        Compute cross-attention maps (CAM) for given text embeddings.

        Args:
            target_embeds: (1, seq_len, dim) text embeddings
            layer_filter: optional function to filter layers by resolution

        Returns:
            cam: (1, 1, H, W) aggregated attention map in [0, 1]
        """
        cams = []

        for name, data in self.attn_maps.items():
            hidden = data['hidden']
            module = data['module']

            # Get spatial resolution
            # hidden shape: (batch, h*w, dim)
            hw = hidden.shape[1]
            res = int(math.sqrt(hw))

            # Filter by resolution
            if self.use_layers == "mid_high" and res > 32:
                continue  # skip 64x64 (too noisy, as per DAG)

            # Compute Q and K
            q = module.to_q(hidden)  # (batch, h*w, inner_dim)
            k = module.to_k(target_embeds)  # (batch, seq_len, inner_dim)

            # Reshape for multi-head attention
            head_dim = module.heads
            inner_dim = q.shape[-1]
            dim_per_head = inner_dim // head_dim

            q = q.view(q.shape[0], q.shape[1], head_dim, dim_per_head).permute(0, 2, 1, 3)
            k = k.view(k.shape[0], k.shape[1], head_dim, dim_per_head).permute(0, 2, 1, 3)

            # Attention: softmax(Q @ K^T / sqrt(d))
            scale = dim_per_head ** -0.5
            attn = torch.bmm(
                q.reshape(-1, q.shape[2], q.shape[3]),
                k.reshape(-1, k.shape[2], k.shape[3]).transpose(-2, -1)
            ) * scale
            attn = attn.softmax(dim=-1)  # (batch*heads, h*w, seq_len)

            # Average over heads
            attn = attn.view(q.shape[0], head_dim, -1, attn.shape[-1])
            attn = attn.mean(dim=1)  # (batch, h*w, seq_len)

            # Average over text tokens (exclude BOS/EOS — take tokens 1:-1)
            # For "nude person" that's tokens 1 and 2
            n_tokens = target_embeds.shape[1]
            # Use tokens 1 to min(4, n_tokens-1) to focus on content tokens
            content_end = min(4, n_tokens - 1)
            attn_spatial = attn[:, :, 1:content_end].mean(dim=-1)  # (batch, h*w)

            # Reshape to spatial
            cam = attn_spatial.view(1, 1, res, res)
            cams.append(cam)

        if not cams:
            return torch.zeros(1, 1, 64, 64, device=target_embeds.device)

        # Aggregate: interpolate all to 64x64 and average
        aggregated = []
        for cam in cams:
            if cam.shape[-1] != 64:
                cam = F.interpolate(cam.float(), size=(64, 64), mode='bilinear',
                                    align_corners=False)
            aggregated.append(cam.float())

        result = torch.stack(aggregated).mean(dim=0)

        # Min-max normalize to [0, 1]
        rmin, rmax = result.min(), result.max()
        if rmax - rmin > 1e-8:
            result = (result - rmin) / (rmax - rmin)
        else:
            result = torch.zeros_like(result)

        return result.clamp(0, 1)

    def clear(self):
        """Clear stored attention maps."""
        self.attn_maps.clear()

    def remove_hooks(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def compute_spatial_mask(
    cam_target: torch.Tensor,   # (1, 1, H, W) cross-attention map for target
    cam_anchor: torch.Tensor,   # (1, 1, H, W) cross-attention map for anchor
    threshold: float = 0.3,
    mode: str = "diff",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spatial guidance mask from cross-attention maps.

    Returns:
        mask: (1, 1, H, W) binary mask where guidance should be applied
        weight: (1, 1, H, W) continuous weight for adaptive scaling
    """
    if mode == "diff":
        # Where target attention > anchor attention
        diff = (cam_target - cam_anchor).clamp(0, 1)
        # Normalize
        dmax = diff.max()
        if dmax > 1e-8:
            diff = diff / dmax
        mask = (diff > threshold).float()
        weight = diff * mask
    elif mode == "target_only":
        # Just use target attention magnitude
        mask = (cam_target > threshold).float()
        weight = cam_target * mask
    else:
        raise ValueError(f"Unknown spatial mode: {mode}")

    return mask, weight


# =============================================================================
# Noise-based Spatial Map (fallback when cross-attention is unavailable)
# =============================================================================
def compute_noise_spatial_map(
    noise_uncond, noise_target, noise_anchor, noise_text,
    method="weighted_cas",
) -> torch.Tensor:
    """Fallback: compute spatial map from noise predictions."""
    d_target = (noise_target - noise_uncond).float()
    d_anchor = (noise_anchor - noise_uncond).float()
    d_prompt = (noise_text - noise_uncond).float()

    if method == "weighted_cas":
        cos_target = F.cosine_similarity(d_prompt, d_target, dim=1).unsqueeze(1)
        cos_anchor = F.cosine_similarity(d_prompt, d_anchor, dim=1).unsqueeze(1)
        score = F.relu(cos_target - cos_anchor)
    elif method == "diff_norm":
        score = (d_target - d_anchor).norm(dim=1, keepdim=True)
    else:
        score = F.relu(F.cosine_similarity(d_prompt, d_target, dim=1).unsqueeze(1))

    # Percentile normalization
    flat = score.reshape(-1)
    if flat.max() - flat.min() < 1e-8:
        return torch.zeros(1, 1, score.shape[2], score.shape[3], device=score.device)
    p_lo = torch.quantile(flat, 0.02)
    p_hi = torch.quantile(flat, 0.98)
    if p_hi - p_lo < 1e-8:
        p_lo, p_hi = flat.min(), flat.max()
    normalized = ((score - p_lo) / (p_hi - p_lo + 1e-8)).clamp(0, 1)

    return torch.nan_to_num(normalized, nan=0.0).clamp(0, 1)


# =============================================================================
# Safe CFG Guidance (HOW) — SLD/DAG style
# =============================================================================
def apply_safe_cfg(
    noise_cfg: torch.Tensor,          # Standard CFG result: ε_∅ + s_g*(ε_p - ε_∅)
    noise_uncond: torch.Tensor,       # ε_∅
    noise_target: torch.Tensor,       # ε_target (nudity concept)
    noise_anchor: torch.Tensor,       # ε_anchor (clothed person)
    mask: torch.Tensor,               # (1, 1, H, W) binary mask
    weight: torch.Tensor,             # (1, 1, H, W) continuous weight [0,1]
    mode: str = "sld",
    safety_scale: float = 3.0,
    cfg_scale: float = 7.5,
) -> torch.Tensor:
    """
    Apply spatially-masked safety guidance following SLD/DAG formulation.

    SLD style:
        ε_safe = ε_cfg - s_safety * M * (ε_target - ε_∅)
        Subtracts the target concept direction ONLY in masked regions.

    anchor_shift style:
        ε_safe = ε_cfg - s_safety * M * (ε_target - ε_anchor)
        Pushes from target toward anchor direction in masked regions.

    DAG adaptive style:
        ε_safe = ε_cfg - s_safety * (sc * Area * T(A)) * (ε_target - ε_∅)
        With area-based scaling and magnitude-based per-pixel scaling.
    """
    d_target = noise_target - noise_uncond  # target concept direction

    if mode == "sld":
        # SLD: subtract target direction in masked regions
        correction = safety_scale * weight * d_target
        noise_safe = noise_cfg - correction

    elif mode == "anchor_shift":
        # Push from target toward anchor
        d_shift = noise_target - noise_anchor  # target - anchor direction
        correction = safety_scale * weight * d_shift
        noise_safe = noise_cfg - correction

    elif mode == "dag_adaptive":
        # DAG: area-based + magnitude-based adaptive scaling
        # Area scaling: larger detected region → stronger per-pixel guidance
        area = mask.sum() / mask.numel()
        area_scale = 5.0 / (mask.shape[-1] * mask.shape[-2])  # base weight
        area_factor = area_scale * area * mask.numel()  # = 5 * area_fraction

        # Magnitude scaling: rescale weight from [0,1] to [1, 5]
        mag_scale = 1.0 + 4.0 * weight  # pixels with higher attention get stronger guidance

        correction = safety_scale * area_factor * mag_scale * mask * d_target
        noise_safe = noise_cfg - correction

    elif mode == "dual":
        # Combine SLD + anchor: subtract target AND add anchor direction
        correction = safety_scale * weight * (d_target - (noise_anchor - noise_uncond))
        noise_safe = noise_cfg - correction

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # NaN guard
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
    p = ArgumentParser(description="CAS + Spatial CFG v3")
    p.add_argument("--ckpt", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--prompts", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--nsamples", type=int, default=4)
    p.add_argument("--cfg_scale", type=float, default=7.5)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    # CAS (When)
    p.add_argument("--cas_threshold", type=float, default=0.3)
    p.add_argument("--cas_sticky", action="store_true", default=True)
    p.add_argument("--cas_no_sticky", action="store_true")
    p.add_argument("--cas_min_count", type=int, default=1)

    # Concepts
    p.add_argument("--target", type=str, nargs="+",
                   default=["nudity", "nude person", "naked body"])
    p.add_argument("--anchor", type=str, nargs="+",
                   default=["clothed person", "person wearing clothes", "fully dressed person"])

    # Spatial (Where) — cross-attention vs noise-based
    p.add_argument("--spatial_mode", type=str, default="crossattn",
                   choices=["crossattn", "noise"],
                   help="crossattn: use UNet cross-attention maps (semantic), "
                        "noise: use noise prediction differences (fallback)")
    p.add_argument("--spatial_threshold", type=float, default=0.3,
                   help="Threshold for binary mask from spatial map")
    p.add_argument("--spatial_mask_mode", type=str, default="diff",
                   choices=["diff", "target_only"],
                   help="diff: target-anchor attention, target_only: just target attention")

    # Guidance (How) — SLD/DAG style
    p.add_argument("--guide_mode", type=str, default="sld",
                   choices=["sld", "anchor_shift", "dag_adaptive", "dual"])
    p.add_argument("--safety_scale", type=float, default=3.0,
                   help="Safety guidance scale (SLD uses 1-10, DAG uses ~5)")
    p.add_argument("--guide_start_frac", type=float, default=0.0,
                   help="Start guidance after this fraction of steps (0=all, 0.5=last half)")
    p.add_argument("--warmup_steps", type=int, default=5,
                   help="Number of warmup steps (DAG default=5)")

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
    print(f"CAS + Spatial CFG v3: Safe Generation (SLD/DAG style)")
    print(f"{'='*70}")
    print(f"  WHEN: CAS threshold={args.cas_threshold}, sticky={args.cas_sticky}")
    print(f"  WHERE: {args.spatial_mode}, threshold={args.spatial_threshold}, mask={args.spatial_mask_mode}")
    print(f"  HOW: {args.guide_mode}, safety_scale={args.safety_scale}, "
          f"start_frac={args.guide_start_frac}, warmup={args.warmup_steps}")
    print(f"  Target: {args.target}")
    print(f"  Anchor: {args.anchor}")
    print(f"{'='*70}\n")

    # Load prompts
    all_prompts = load_prompts(args.prompts)
    end_idx = args.end_idx if args.end_idx > 0 else len(all_prompts)
    prompts_with_idx = list(enumerate(all_prompts))[args.start_idx:end_idx]
    print(f"Loaded {len(all_prompts)} prompts, processing [{args.start_idx}:{end_idx}]")

    # Load pipeline — safety_checker explicitly None
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Also disable feature extractor (prevents any safety filtering)
    pipe.feature_extractor = None

    unet = pipe.unet
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    scheduler = pipe.scheduler

    # Pre-encode concept embeddings
    with torch.no_grad():
        def encode_text(text):
            inputs = tokenizer(text, padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True, return_tensors="pt")
            return text_encoder(inputs.input_ids.to(device))[0]

        target_embeds = encode_text(", ".join(args.target))
        anchor_embeds = encode_text(", ".join(args.anchor))
        uncond_embeds = encode_text("")

    # Cross-attention extractor (if using crossattn spatial mode)
    cam_extractor = None
    if args.spatial_mode == "crossattn":
        cam_extractor = CrossAttnMapExtractor(unet, use_layers="mid_high")
        print("Cross-attention hooks installed for spatial map extraction")

    # Init CAS
    cas = CAS(threshold=args.cas_threshold, sticky=args.cas_sticky,
              sticky_min_count=args.cas_min_count)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.save_maps:
        (outdir / "maps").mkdir(exist_ok=True)

    all_stats = []

    for prompt_idx, prompt in tqdm(prompts_with_idx, desc="Generating"):
        for sample_idx in range(args.nsamples):
            seed = args.seed + prompt_idx * args.nsamples + sample_idx
            set_seed(seed)
            cas.reset()

            guided_count = 0
            cas_values = []
            areas = []

            # Encode prompt
            with torch.no_grad():
                text_embeds = encode_text(prompt)

            # Init latents
            set_seed(seed)
            latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
            latents = latents * scheduler.init_noise_sigma
            scheduler.set_timesteps(args.steps, device=device)
            total_steps = len(scheduler.timesteps)

            guide_start_step = int(total_steps * args.guide_start_frac)
            warmup_end_step = guide_start_step + args.warmup_steps

            for step_idx, t in enumerate(scheduler.timesteps):
                # --- Standard CFG forward (2 passes: uncond + text) ---
                lat_in = torch.cat([latents] * 2)
                lat_in = scheduler.scale_model_input(lat_in, t)

                if cam_extractor:
                    cam_extractor.clear()

                with torch.no_grad():
                    raw = unet(lat_in, t,
                               encoder_hidden_states=torch.cat([uncond_embeds, text_embeds])
                               ).sample
                noise_uncond, noise_text = raw.chunk(2)
                noise_cfg = noise_uncond + args.cfg_scale * (noise_text - noise_uncond)

                # --- CAS check (WHEN) ---
                lat_single = scheduler.scale_model_input(latents, t)
                with torch.no_grad():
                    noise_target = unet(lat_single, t,
                                        encoder_hidden_states=target_embeds).sample

                cas_val, cas_triggered = cas.compute(noise_text, noise_uncond, noise_target)
                cas_values.append(cas_val)

                # Should we apply spatial guidance?
                in_window = step_idx >= guide_start_step
                should_guide = cas_triggered and in_window

                if should_guide:
                    # Warmup: linearly increase safety_scale during warmup
                    if step_idx < warmup_end_step and args.warmup_steps > 0:
                        warmup_ratio = (step_idx - guide_start_step + 1) / args.warmup_steps
                        current_scale = args.safety_scale * min(warmup_ratio, 1.0)
                    else:
                        current_scale = args.safety_scale

                    # Get anchor noise prediction
                    with torch.no_grad():
                        noise_anchor = unet(lat_single, t,
                                            encoder_hidden_states=anchor_embeds).sample

                    # --- WHERE: Compute spatial mask ---
                    if args.spatial_mode == "crossattn" and cam_extractor:
                        # Extract cross-attention maps for target and anchor
                        # We need to run forward with target/anchor embeds to get their CAMs
                        # But we already ran the forward for noise_target above,
                        # so cam_extractor has the maps from that forward pass.
                        # For target CAM:
                        cam_target = cam_extractor.compute_cam(target_embeds)

                        # For anchor CAM, use the anchor forward pass maps
                        cam_extractor.clear()
                        # The anchor noise was just computed, so hooks captured anchor maps
                        cam_anchor = cam_extractor.compute_cam(anchor_embeds)

                        # If anchor CAM is empty (hooks didn't fire for single pass),
                        # recompute with explicit forward
                        if cam_anchor.sum() < 1e-8:
                            cam_anchor = cam_target * 0  # fallback: no anchor info

                        mask, weight = compute_spatial_mask(
                            cam_target, cam_anchor,
                            threshold=args.spatial_threshold,
                            mode=args.spatial_mask_mode,
                        )
                    else:
                        # Fallback: noise-based spatial map
                        spatial_map = compute_noise_spatial_map(
                            noise_uncond, noise_target, noise_anchor, noise_text,
                        )
                        mask = (spatial_map > args.spatial_threshold).float()
                        weight = spatial_map * mask
                        # Normalize weight
                        wmax = weight.max()
                        if wmax > 1e-8:
                            weight = weight / wmax

                    # --- HOW: Apply safe CFG ---
                    noise_cfg = apply_safe_cfg(
                        noise_cfg, noise_uncond, noise_target, noise_anchor,
                        mask.to(noise_cfg.dtype), weight.to(noise_cfg.dtype),
                        mode=args.guide_mode,
                        safety_scale=current_scale,
                        cfg_scale=args.cfg_scale,
                    )

                    guided_count += 1
                    area_val = float(mask.mean().item())
                    areas.append(area_val)

                    # Save maps
                    if args.save_maps and step_idx % 10 == 0:
                        if args.spatial_mode == "crossattn":
                            m = cam_target[0, 0].float().cpu().numpy()
                        else:
                            m = spatial_map[0, 0].float().cpu().numpy()
                        m = np.nan_to_num(m, nan=0.0)
                        img_map = (np.clip(m, 0, 1) * 255).astype(np.uint8)
                        Image.fromarray(img_map, 'L').save(
                            str(outdir / "maps" / f"{prompt_idx:04d}_{sample_idx:02d}_s{step_idx:03d}.png"))

                # --- DDIM step ---
                latents_prev = latents.clone()
                latents = scheduler.step(noise_cfg, t, latents).prev_sample

                # NaN recovery
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    print(f"  WARNING: NaN at step {step_idx}, reverting")
                    noise_fallback = noise_uncond + args.cfg_scale * (noise_text - noise_uncond)
                    latents = scheduler.step(noise_fallback, t, latents_prev).prev_sample

                if args.debug and step_idx % 10 == 0:
                    status = "GUIDED" if should_guide else ("CAS_ON" if cas_triggered else "skip")
                    area_s = f" area={areas[-1]:.3f}" if should_guide and areas else ""
                    print(f"  [{step_idx:02d}] t={t.item()} CAS={cas_val:.3f} {status}{area_s}")

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
                "cas_triggered": cas.triggered,
                "avg_cas": float(np.mean(cas_values)) if cas_values else 0,
                "max_cas": float(np.max(cas_values)) if cas_values else 0,
                "avg_area": float(np.mean(areas)) if areas else 0,
            }
            all_stats.append(stats)

            if prompt_idx % 10 == 0 or args.debug:
                print(f"  [{prompt_idx:03d}_{sample_idx}] guided={guided_count}/{total_steps} "
                      f"CAS avg={stats['avg_cas']:.3f} area={stats['avg_area']:.3f}")

    # Save summary
    n = len(all_stats)
    n_trig = sum(1 for s in all_stats if s["guided_steps"] > 0)
    summary = {
        "method": "CAS + Spatial CFG v3 (SLD/DAG style)",
        "args": vars(args),
        "overall": {
            "total_images": n,
            "triggered": n_trig,
            "avg_guided": float(np.mean([s["guided_steps"] for s in all_stats])) if n else 0,
            "avg_cas": float(np.mean([s["avg_cas"] for s in all_stats if s["avg_cas"] > 0])) if n else 0,
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
