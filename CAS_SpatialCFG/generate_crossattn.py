#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAS + Cross-Attention Spatial Guidance: Training-Free Safe Generation

Key insight from DAG paper: Cross-attention maps in the UNet contain rich
semantic-spatial information. By comparing attention maps for target tokens
("nude") vs anchor tokens ("clothed"), we get precise spatial localization
of WHERE nudity is being generated.

Advantages over noise-based spatial map:
  - Cross-attention has 77 token dimensions (vs 4 latent channels)
  - Higher semantic meaning (token-to-pixel alignment)
  - Multi-resolution: 16×16, 32×32, 64×64 attention maps available
  - No need for patch-based tricks

Pipeline:
  WHEN: CAS (same as v2 — cosine of noise directions)
  WHERE: Cross-attention map difference (target tokens vs anchor tokens)
  HOW: Interpolation-based guidance in masked regions
"""

import os
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
# Cross-Attention Map Extractor
# =============================================================================
class CrossAttnMapExtractor:
    """
    Hook into UNet cross-attention layers to extract attention maps.

    During a UNet forward pass, captures the attention probability matrices
    from cross-attention layers. These maps show which spatial regions attend
    to which text tokens.
    """

    def __init__(self, unet, target_resolution=(16, 32)):
        """
        Args:
            unet: The UNet model
            target_resolution: Tuple of resolutions to capture (16=high semantic, 32=mid)
        """
        self.unet = unet
        self.target_resolution = target_resolution
        self.attn_maps = {}  # {resolution: list of attention maps}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on cross-attention layers (attn2 in diffusers)."""
        for name, module in self.unet.named_modules():
            # In diffusers SD v1.x, attn2 = cross-attention, attn1 = self-attention
            if name.endswith('.attn2') and hasattr(module, 'to_q'):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            # input[0] is the hidden states, input[1] is encoder_hidden_states
            if len(input) < 2 or input[1] is None:
                return  # Self-attention, skip

            hidden_states = input[0]  # (B, seq_len, dim)
            encoder_hidden_states = input[1]  # (B, 77, dim)

            # Compute attention map manually
            batch_size = hidden_states.shape[0]
            seq_len = hidden_states.shape[1]
            spatial_size = int(math.sqrt(seq_len))

            # Only capture target resolutions
            if spatial_size not in self.target_resolution:
                return

            # Compute Q, K
            query = module.to_q(hidden_states)
            key = module.to_k(encoder_hidden_states)

            # Reshape for multi-head attention
            num_heads = module.heads
            head_dim = query.shape[-1] // num_heads

            query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

            # Attention weights
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
            attn_probs = attn_weights.softmax(dim=-1)  # (B, heads, spatial, 77)

            # Average over heads
            attn_avg = attn_probs.mean(dim=1)  # (B, spatial, 77)

            # Store
            if spatial_size not in self.attn_maps:
                self.attn_maps[spatial_size] = []
            self.attn_maps[spatial_size].append(attn_avg.detach())

        return hook_fn

    def clear(self):
        """Clear stored attention maps."""
        self.attn_maps = {}

    def get_spatial_map_for_tokens(
        self,
        token_indices: List[int],
        resolution: int = 16,
    ) -> torch.Tensor:
        """
        Get spatial attention map for specific token indices.

        Args:
            token_indices: Which token positions to aggregate
            resolution: Which resolution attention maps to use

        Returns:
            (1, 1, res, res) attention map averaged over target tokens
        """
        if resolution not in self.attn_maps or not self.attn_maps[resolution]:
            return None

        # Average over all captured layers at this resolution
        all_maps = torch.stack(self.attn_maps[resolution])  # (num_layers, B, spatial, 77)
        avg_map = all_maps.mean(dim=0)  # (B, spatial, 77)

        # Select target token columns and average
        token_map = avg_map[:, :, token_indices].mean(dim=-1)  # (B, spatial)

        # Reshape to 2D
        B = token_map.shape[0]
        token_map = token_map.view(B, 1, resolution, resolution)

        return token_map

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# =============================================================================
# CAS (same as v2)
# =============================================================================
class CAS:
    def __init__(self, threshold=0.3, sticky=True, min_count=1):
        self.threshold = threshold
        self.sticky = sticky
        self.min_count = min_count
        self.triggered = False
        self.consec = 0

    def reset(self):
        self.triggered = False
        self.consec = 0

    def compute(self, noise_text, noise_uncond, noise_target):
        d_p = (noise_text - noise_uncond).reshape(1, -1)
        d_t = (noise_target - noise_uncond).reshape(1, -1)
        cas = F.cosine_similarity(d_p, d_t, dim=-1).item()
        if self.sticky and self.triggered:
            return cas, True
        if cas > self.threshold:
            self.consec += 1
        else:
            self.consec = 0
        go = self.consec >= self.min_count
        if go and self.sticky:
            self.triggered = True
        return cas, go


# =============================================================================
# Cross-Attention Spatial Guidance
# =============================================================================
def compute_crossattn_spatial_map(
    extractor: CrossAttnMapExtractor,
    target_token_ids: List[int],
    anchor_token_ids: List[int],
    resolution: int = 16,
    output_size: int = 64,
) -> torch.Tensor:
    """
    Compute spatial guidance map from cross-attention differences.

    target_map: where "nude" tokens attend → nudity regions
    anchor_map: where "clothed" tokens attend → clothing regions
    guidance_map = target_map - anchor_map → nudity-specific regions

    Returns: (1, 1, 64, 64) in [0, 1]
    """
    target_map = extractor.get_spatial_map_for_tokens(target_token_ids, resolution)
    anchor_map = extractor.get_spatial_map_for_tokens(anchor_token_ids, resolution)

    if target_map is None or anchor_map is None:
        return torch.zeros(1, 1, output_size, output_size)

    # Difference: where target attends more than anchor
    diff = F.relu(target_map - anchor_map)

    # Upscale to latent resolution
    diff = F.interpolate(diff.float(), size=(output_size, output_size),
                         mode='bilinear', align_corners=False)

    # Normalize to [0, 1] with percentile
    flat = diff.reshape(-1)
    if flat.max() - flat.min() < 1e-8:
        return torch.zeros(1, 1, output_size, output_size, device=diff.device)
    p_low = torch.quantile(flat, 0.05)
    p_high = torch.quantile(flat, 0.95)
    if p_high - p_low < 1e-8:
        p_low, p_high = flat.min(), flat.max()
    normalized = ((diff - p_low) / (p_high - p_low + 1e-8)).clamp(0, 1)

    # Gaussian blur for smoothness
    normalized = gaussian_blur_2d(normalized, sigma=2.0)
    flat2 = normalized.reshape(-1)
    n_min, n_max = flat2.min(), flat2.max()
    if n_max - n_min > 1e-8:
        normalized = (normalized - n_min) / (n_max - n_min)

    return normalized


def gaussian_blur_2d(x, sigma=2.0):
    if sigma <= 0:
        return x
    ks = int(2 * math.ceil(2 * sigma) + 1)
    if ks % 2 == 0:
        ks += 1
    coords = torch.arange(ks, device=x.device, dtype=x.dtype) - ks // 2
    k1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    k1d /= k1d.sum()
    pad = ks // 2
    kh, kw = k1d.view(1, 1, -1, 1), k1d.view(1, 1, 1, -1)
    out = F.pad(x, (0, 0, pad, pad), mode='reflect')
    out = F.conv2d(out, kh, groups=1)
    out = F.pad(out, (pad, pad, 0, 0), mode='reflect')
    return F.conv2d(out, kw, groups=1)


# =============================================================================
# Safety Guidance (same interpolation as v2)
# =============================================================================
def apply_guidance(noise_cfg, noise_uncond, noise_target, noise_anchor,
                   spatial_map, mode="interpolate", scale=1.0, threshold=0.3):
    weight = ((spatial_map - threshold) / (1.0 - threshold + 1e-8)).clamp(0, 1) * scale
    weight = weight.clamp(0, 1)
    if mode == "interpolate":
        correction = weight * ((noise_anchor - noise_uncond) - (noise_target - noise_uncond))
        return noise_cfg + correction
    elif mode == "negate":
        return noise_cfg - weight * (noise_target - noise_uncond)
    return noise_cfg


# =============================================================================
# Utils
# =============================================================================
def load_prompts(f):
    f = Path(f)
    if f.suffix == ".csv":
        prompts = []
        with open(f) as fp:
            reader = csv.DictReader(fp)
            for col in ['sensitive prompt', 'adv_prompt', 'prompt', 'target_prompt', 'text']:
                if col in reader.fieldnames:
                    prompt_col = col; break
            for row in reader:
                prompts.append(row[prompt_col].strip())
        return prompts
    return [l.strip() for l in open(f) if l.strip()]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_token_indices(tokenizer, text, target_words):
    """Find token indices in the tokenized text that match target words."""
    tokens = tokenizer.tokenize(text)
    indices = []
    for i, tok in enumerate(tokens):
        tok_clean = tok.replace('</w>', '').lower()
        if any(w.lower() in tok_clean for w in target_words):
            indices.append(i + 1)  # +1 for BOS token
    return indices if indices else [1]  # fallback to first real token


def parse_args():
    p = ArgumentParser(description="CAS + Cross-Attention Spatial Guidance")
    p.add_argument("--ckpt", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--prompts", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--nsamples", type=int, default=4)
    p.add_argument("--cfg_scale", type=float, default=7.5)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    # CAS
    p.add_argument("--cas_threshold", type=float, default=0.3)
    p.add_argument("--cas_sticky", action="store_true", default=True)
    # Concepts
    p.add_argument("--target", nargs="+", default=["nudity", "nude person", "nsfw person", "naked body"])
    p.add_argument("--anchor", nargs="+", default=["clothed person", "person wearing clothes", "fully dressed person"])
    p.add_argument("--target_words", nargs="+", default=["nude", "naked", "nudity", "nsfw", "bare", "exposed", "undress", "topless"])
    p.add_argument("--anchor_words", nargs="+", default=["clothed", "dressed", "wearing", "clothes"])
    # Guidance
    p.add_argument("--guide_mode", default="interpolate", choices=["interpolate", "negate"])
    p.add_argument("--guide_scale", type=float, default=0.8)
    p.add_argument("--guide_start_frac", type=float, default=0.4)
    p.add_argument("--spatial_threshold", type=float, default=0.3)
    p.add_argument("--attn_resolution", type=int, default=16, choices=[16, 32])
    # Misc
    p.add_argument("--save_maps", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda")

    print(f"\n{'='*70}")
    print(f"CAS + Cross-Attention Spatial Guidance")
    print(f"{'='*70}")
    print(f"  WHEN: CAS threshold={args.cas_threshold}")
    print(f"  WHERE: Cross-attention maps @ {args.attn_resolution}×{args.attn_resolution}")
    print(f"  HOW: {args.guide_mode}, scale={args.guide_scale}, start={args.guide_start_frac}")
    print(f"  Target words (for attn): {args.target_words}")
    print(f"  Anchor words (for attn): {args.anchor_words}")
    print(f"{'='*70}\n")

    prompts = load_prompts(args.prompts)
    end = args.end_idx if args.end_idx > 0 else len(prompts)
    prompts_with_idx = list(enumerate(prompts))[args.start_idx:end]

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, safety_checker=None).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    unet, vae, tokenizer, text_encoder = pipe.unet, pipe.vae, pipe.tokenizer, pipe.text_encoder
    scheduler = pipe.scheduler

    # Setup cross-attention extractor
    attn_extractor = CrossAttnMapExtractor(unet, target_resolution=(args.attn_resolution,))

    # Pre-encode concepts
    with torch.no_grad():
        def enc(text):
            inp = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length,
                            truncation=True, return_tensors="pt")
            return text_encoder(inp.input_ids.to(device))[0]
        target_emb = enc(", ".join(args.target))
        anchor_emb = enc(", ".join(args.anchor))
        uncond_emb = enc("")

    # Get token indices for target/anchor concepts
    target_text = ", ".join(args.target)
    anchor_text = ", ".join(args.anchor)
    target_tok_ids = get_token_indices(tokenizer, target_text, args.target_words)
    anchor_tok_ids = get_token_indices(tokenizer, anchor_text, args.anchor_words)
    print(f"Target token indices: {target_tok_ids}")
    print(f"Anchor token indices: {anchor_tok_ids}")

    cas = CAS(args.cas_threshold, args.cas_sticky)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if args.save_maps:
        (outdir / "maps").mkdir(exist_ok=True)

    all_stats = []

    for pi, prompt in tqdm(prompts_with_idx, desc="Generating"):
        for si in range(args.nsamples):
            seed = args.seed + pi * args.nsamples + si
            set_seed(seed)
            cas.reset()

            guided_count = 0
            cas_vals = []
            with torch.no_grad():
                prompt_emb = enc(prompt)

            set_seed(seed)
            latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
            latents *= scheduler.init_noise_sigma
            scheduler.set_timesteps(args.steps, device=device)
            total_steps = len(scheduler.timesteps)
            guide_start = int(total_steps * args.guide_start_frac)

            for step_idx, t in enumerate(scheduler.timesteps):
                # --- CFG forward ---
                lat_in = scheduler.scale_model_input(torch.cat([latents] * 2), t)
                with torch.no_grad():
                    raw = unet(lat_in, t,
                               encoder_hidden_states=torch.cat([uncond_emb, prompt_emb])).sample
                n_u, n_t = raw.chunk(2)
                n_cfg = n_u + args.cfg_scale * (n_t - n_u)

                # --- CAS ---
                lat_s = scheduler.scale_model_input(latents, t)
                with torch.no_grad():
                    n_target = unet(lat_s, t, encoder_hidden_states=target_emb).sample
                cas_val, cas_go = cas.compute(n_t, n_u, n_target)
                cas_vals.append(cas_val)

                should_guide = cas_go and step_idx >= guide_start

                if should_guide:
                    # --- Cross-Attention Spatial Map (WHERE) ---
                    # We need a forward pass with target embeddings to get attention maps
                    attn_extractor.clear()
                    with torch.no_grad():
                        # Forward with target concept to get target attention maps
                        _ = unet(lat_s, t, encoder_hidden_states=target_emb).sample
                    target_attn_maps = {k: list(v) for k, v in attn_extractor.attn_maps.items()}

                    attn_extractor.clear()
                    with torch.no_grad():
                        n_anchor = unet(lat_s, t, encoder_hidden_states=anchor_emb).sample
                    anchor_attn_maps = {k: list(v) for k, v in attn_extractor.attn_maps.items()}

                    # Compute spatial map from cross-attention difference
                    res = args.attn_resolution
                    if res in target_attn_maps and res in anchor_attn_maps:
                        # Stack all layers
                        t_maps = torch.stack(target_attn_maps[res]).mean(0)  # (B, spatial, 77)
                        a_maps = torch.stack(anchor_attn_maps[res]).mean(0)

                        # Sum attention for ALL tokens (target concepts activate broadly)
                        # Target tokens get high attention in nudity regions
                        t_spatial = t_maps.mean(dim=-1).view(1, 1, res, res)  # avg over all tokens
                        a_spatial = a_maps.mean(dim=-1).view(1, 1, res, res)

                        # Difference: where target attends more
                        diff = F.relu(t_spatial - a_spatial)
                        spatial_map = F.interpolate(diff.float(), size=(64, 64),
                                                     mode='bilinear', align_corners=False)

                        # Normalize
                        flat = spatial_map.reshape(-1)
                        if flat.max() - flat.min() > 1e-8:
                            p_lo = torch.quantile(flat, 0.05)
                            p_hi = torch.quantile(flat, 0.95)
                            if p_hi - p_lo < 1e-8:
                                p_lo, p_hi = flat.min(), flat.max()
                            spatial_map = ((spatial_map - p_lo) / (p_hi - p_lo + 1e-8)).clamp(0, 1)
                        else:
                            spatial_map = torch.zeros_like(spatial_map)

                        spatial_map = gaussian_blur_2d(spatial_map, 2.0)
                        flat2 = spatial_map.reshape(-1)
                        s_min, s_max = flat2.min(), flat2.max()
                        if s_max - s_min > 1e-8:
                            spatial_map = (spatial_map - s_min) / (s_max - s_min)
                    else:
                        spatial_map = torch.zeros(1, 1, 64, 64, device=device)
                        n_anchor = unet(lat_s, t, encoder_hidden_states=anchor_emb).sample

                    # Apply guidance
                    n_cfg = apply_guidance(n_cfg, n_u, n_target, n_anchor,
                                          spatial_map.to(n_cfg.dtype),
                                          args.guide_mode, args.guide_scale, args.spatial_threshold)
                    guided_count += 1

                    if args.save_maps and step_idx % 5 == 0:
                        m = spatial_map[0, 0].float().cpu().numpy()
                        m = np.nan_to_num(m, 0.0)
                        Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8), 'L').save(
                            str(outdir / "maps" / f"{pi:04d}_{si:02d}_s{step_idx:03d}.png"))

                # DDIM step
                latents = scheduler.step(n_cfg, t, latents).prev_sample

                if args.debug and step_idx % 10 == 0:
                    st = "GUIDED" if should_guide else ("CAS" if cas_go else "skip")
                    print(f"  [{step_idx:02d}] t={t.item()} CAS={cas_val:.3f} {st}")

            # Decode
            with torch.no_grad():
                dec = vae.decode(latents.to(vae.dtype) / vae.config.scaling_factor).sample
                dec = (dec / 2 + 0.5).clamp(0, 1)
                img = (dec[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

            name = "".join(c if c.isalnum() or c in ' -_' else '_' for c in prompt)[:50].replace(' ', '_')
            Image.fromarray(img).resize((512, 512)).save(str(outdir / f"{pi:04d}_{si:02d}_{name}.png"))

            stats = {
                "prompt_idx": pi, "sample_idx": si, "seed": seed,
                "prompt": prompt[:100], "guided": guided_count,
                "total": total_steps, "cas_triggered": cas.triggered,
                "avg_cas": float(np.mean(cas_vals)), "max_cas": float(np.max(cas_vals)),
            }
            all_stats.append(stats)
            print(f"  [{pi:03d}_{si}] guided={guided_count}/{total_steps} CAS={stats['avg_cas']:.3f}")

    n = len(all_stats)
    summary = {
        "method": "CAS + CrossAttn Spatial",
        "args": vars(args),
        "overall": {
            "total": n, "triggered": sum(1 for s in all_stats if s["guided"] > 0),
            "avg_guided": float(np.mean([s["guided"] for s in all_stats])) if n else 0,
        },
        "per_image": all_stats,
    }
    json.dump(summary, open(outdir / "stats.json", "w"), indent=2)
    print(f"\nDone! {n} images at {outdir}")


if __name__ == "__main__":
    main()
