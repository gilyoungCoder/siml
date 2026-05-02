#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v23: Dual-Direction Anchor Inpainting — Text + Image Target Directions

Key idea:
  SD1.4 uses CLIP as text encoder → CLIP image features are in the same 768-dim space.
  Feed CLIP(img) as UNet conditioning → get eps_img_target → pixel-level spatial CAS.
  This gives a proper image-informed WHERE mask, complementary to text-based WHERE.

Architecture:
  WHEN: Noise CAS with TEXT target (threshold=0.6, sticky) — unchanged
  WHERE (Dual-Direction):
    Direction 1 — Text: eps_text_target from text concepts ("nudity", "nude person")
      → spatial_cas(eps_prompt, eps_null, eps_text_target) → text_mask
    Direction 2 — Image: eps_img_target from CLIP image exemplar embeddings
      → spatial_cas(eps_prompt, eps_null, eps_img_target) → img_mask
    Fusion: union(text_mask, img_mask) or other modes
  HOW: Anchor inpainting (proven best)

Key difference from v20/v22:
  v20/v22: Used CLIP img for cross-attention PROBE (indirect, measures attention patterns)
  v23: Uses CLIP img as direct UNet conditioning → eps_img_target (direct noise prediction)
       This leverages the UNet itself to interpret image features spatially.

Cost: +1 UNet call per guided step (eps_img_target). Total: 4 calls
  (null+prompt batched, text_target, img_target, anchor)
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

# Block name → spatial resolution for SD1.4 UNet
_BLOCK_RES = {
    "down_blocks.0": 64, "down_blocks.1": 32, "down_blocks.2": 16,
    "mid_block": 8, "up_blocks.1": 16, "up_blocks.2": 32, "up_blocks.3": 64,
}

def _get_res(name):
    for prefix, r in _BLOCK_RES.items():
        if name.startswith(prefix): return r
    return 0


# =============================================================================
# Global CAS (WHEN) — text-based, same as v4
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
# Spatial CAS + Soft Mask (WHERE)
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
    kh = g.view(1, 1, kernel_size, 1)
    kw = g.view(1, 1, 1, kernel_size)
    p = kernel_size // 2
    x = F.pad(x, [0, 0, p, p], mode='reflect')
    x = F.conv2d(x, kh.expand(x.shape[1], -1, -1, -1), groups=x.shape[1])
    x = F.pad(x, [p, p, 0, 0], mode='reflect')
    x = F.conv2d(x, kw.expand(x.shape[1], -1, -1, -1), groups=x.shape[1])
    return x


def make_soft_mask(raw_map, threshold, sigmoid_alpha, blur_sigma):
    soft = torch.sigmoid(sigmoid_alpha * (raw_map - threshold))
    soft = soft.unsqueeze(0).unsqueeze(0)
    if blur_sigma > 0:
        soft = gaussian_blur_2d(soft, kernel_size=5, sigma=blur_sigma)
    return soft.clamp(0, 1)


# =============================================================================
# Dual-Direction Fusion
# =============================================================================
def capture_attn_maps(unet, embeds, lat_in, t, target_resolutions=[16, 32]):
    """Run UNet with embeds and capture cross-attention maps → spatial mask [H,W]."""
    attn_maps = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            # input[0] = hidden_states [B, seq, dim]
            # We need Q from hidden_states, K from encoder_hidden_states
            hidden = input[0]
            enc = input[1] if len(input) > 1 else None
            if enc is None:
                return
            B, seq_len, _ = hidden.shape
            q = module.to_q(hidden)
            k = module.to_k(enc)
            inner_dim = k.shape[-1]
            head_dim = inner_dim // module.heads
            q = q.view(B, -1, module.heads, head_dim).transpose(1, 2)
            k = k.view(B, -1, module.heads, head_dim).transpose(1, 2)
            scale = head_dim ** -0.5
            attn_w = torch.matmul(q * scale, k.transpose(-2, -1)).softmax(dim=-1)
            # Average over heads, sum over key tokens → [B, spatial]
            attn_avg = attn_w.mean(dim=1).mean(dim=-1)  # [B, spatial]
            attn_maps[name] = attn_avg.detach()
        return hook_fn

    # Register hooks on attn2 (cross-attention) layers
    for name, module in unet.named_modules():
        if not name.endswith(".attn2") or not hasattr(module, 'to_q'):
            continue
        res = _get_res(name)
        if res in target_resolutions:
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    # Forward pass
    with torch.no_grad():
        unet(lat_in, t, encoder_hidden_states=embeds)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Aggregate: group by resolution, upsample, average
    res_groups = {}
    for name, attn in attn_maps.items():
        res = _get_res(name)
        native = int(attn.shape[-1] ** 0.5)
        spatial = attn[0].view(1, 1, native, native)  # first sample only
        if res not in res_groups:
            res_groups[res] = []
        res_groups[res].append(spatial)

    if not res_groups:
        return torch.zeros(64, 64)

    all_up = []
    for res, maps in res_groups.items():
        avg = torch.stack(maps).mean(0)
        up = F.interpolate(avg.float(), size=(64, 64), mode='bilinear', align_corners=False)
        all_up.append(up)

    combined = torch.stack(all_up).mean(0).squeeze(0).squeeze(0)  # [64, 64]
    # Normalize to [0, 1]
    vmin, vmax = combined.min(), combined.max()
    if vmax - vmin > 1e-8:
        combined = (combined - vmin) / (vmax - vmin)
    return combined


def fuse_masks(text_mask, img_mask, mode="union"):
    if img_mask is None or mode == "text_only":
        return text_mask
    if mode == "img_only":
        return img_mask
    if img_mask.shape != text_mask.shape:
        img_mask = F.interpolate(img_mask, size=text_mask.shape[-2:],
                                 mode='bilinear', align_corners=False)
    if mode == "union":
        return torch.max(text_mask, img_mask)
    elif mode == "soft_union":
        return 1.0 - (1.0 - text_mask) * (1.0 - img_mask)
    elif mode == "mean":
        return (text_mask + img_mask) / 2.0
    elif mode == "multiply":
        return (text_mask * img_mask).clamp(0, 1)
    else:
        raise ValueError(f"Unknown fusion: {mode}")


# =============================================================================
# Utils
# =============================================================================
def load_prompts(filepath):
    filepath = Path(filepath)
    if filepath.suffix == ".csv":
        prompts = []
        with open(filepath, "r") as fp:
            reader = csv.DictReader(fp)
            prompt_col = None
            for col in ['sensitive prompt', 'adv_prompt', 'prompt', 'target_prompt',
                        'text', 'Prompt', 'Text']:
                if col in reader.fieldnames:
                    prompt_col = col
                    break
            if prompt_col is None:
                raise ValueError(f"No known prompt column in {reader.fieldnames}")
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
    p = ArgumentParser(description="v23: Dual-Direction (Text+Image) Anchor Inpainting")

    p.add_argument("--ckpt", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--prompts", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--nsamples", type=int, default=1)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg_scale", type=float, default=7.5)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)

    # WHEN
    p.add_argument("--cas_threshold", type=float, default=0.6)
    p.add_argument("--cas_sticky", action="store_true", default=True)
    p.add_argument("--cas_no_sticky", action="store_true")

    # WHERE — shared params
    p.add_argument("--spatial_threshold", type=float, default=0.1)
    p.add_argument("--sigmoid_alpha", type=float, default=10.0)
    p.add_argument("--neighborhood_size", type=int, default=3)
    p.add_argument("--blur_sigma", type=float, default=1.0)

    # WHERE — Image direction
    p.add_argument("--clip_embeddings", type=str, default=None,
                   help="CLIP exemplar .pt with target_clip_embeds [1,77,768]")
    p.add_argument("--img_where", type=str, default="noise",
                   choices=["noise", "attn"],
                   help="How to compute image WHERE mask: "
                        "noise = spatial CAS with eps_img_target (pixel-level), "
                        "attn = cross-attention map from img_target UNet call")
    p.add_argument("--img_spatial_threshold", type=float, default=None,
                   help="Spatial threshold for image mask (default: same as text)")
    p.add_argument("--attn_resolutions", type=int, nargs="+", default=[16, 32],
                   help="Resolutions to capture attention maps (attn mode only)")

    # WHERE — Fusion
    p.add_argument("--fusion", type=str, default="union",
                   choices=["union", "soft_union", "mean", "multiply",
                            "text_only", "img_only"])

    # HOW
    p.add_argument("--safety_scale", type=float, default=1.2)

    # Concepts
    p.add_argument("--target_concepts", type=str, nargs="+",
                   default=["nudity", "nude person", "naked body"])
    p.add_argument("--anchor_concepts", type=str, nargs="+",
                   default=["clothed person", "person wearing clothes"])

    p.add_argument("--save_maps", action="store_true")
    p.add_argument("--debug", action="store_true")

    args = p.parse_args()
    if args.cas_no_sticky:
        args.cas_sticky = False
    if args.img_spatial_threshold is None:
        args.img_spatial_threshold = args.spatial_threshold
    if args.clip_embeddings is None and args.fusion != "text_only":
        print("[v23] No --clip_embeddings: forcing fusion=text_only")
        args.fusion = "text_only"
    return args


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_img = args.clip_embeddings is not None and args.fusion != "text_only"

    print(f"\n{'='*70}")
    print(f"v23: Dual-Direction (Text+Image) Anchor Inpainting")
    print(f"{'='*70}")
    print(f"  WHEN:  CAS threshold={args.cas_threshold}")
    print(f"  WHERE: text(st={args.spatial_threshold}) "
          f"+ img(st={args.img_spatial_threshold}) → fusion={args.fusion}")
    print(f"  HOW:   anchor_inpaint, ss={args.safety_scale}")
    if use_img:
        print(f"  Image: {args.clip_embeddings}")
    print(f"{'='*70}\n")

    # Load prompts
    all_prompts = load_prompts(args.prompts)
    end_idx = args.end_idx if args.end_idx > 0 else len(all_prompts)
    prompts_with_idx = list(enumerate(all_prompts))[args.start_idx:end_idx]
    print(f"Loaded {len(all_prompts)} prompts, processing [{args.start_idx}:{end_idx}]")

    # Pipeline
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

    # Encode concepts
    with torch.no_grad():
        text_target_embeds = encode_concepts(text_encoder, tokenizer,
                                              args.target_concepts, device)
        anchor_embeds = encode_concepts(text_encoder, tokenizer,
                                        args.anchor_concepts, device)
        uncond_inputs = tokenizer("", padding="max_length",
                                  max_length=tokenizer.model_max_length,
                                  truncation=True, return_tensors="pt")
        uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]

    # Load CLIP image target embeddings
    img_target_embeds = None
    if use_img:
        clip_data = torch.load(args.clip_embeddings, map_location=device)
        img_target_embeds = clip_data["target_clip_embeds"].to(
            device=device, dtype=next(unet.parameters()).dtype)
        print(f"  Loaded image target embeds: {img_target_embeds.shape}")

    # CAS
    cas = GlobalCAS(threshold=args.cas_threshold, sticky=args.cas_sticky)

    # Output
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
                    truncation=True, return_tensors="pt")
                prompt_embeds = text_encoder(prompt_inputs.input_ids.to(device))[0]

            set_seed(seed)
            latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
            latents = latents * scheduler.init_noise_sigma
            scheduler.set_timesteps(args.steps, device=device)
            total_steps = len(scheduler.timesteps)

            for step_idx, t in enumerate(scheduler.timesteps):
                lat_in = scheduler.scale_model_input(latents, t)

                with torch.no_grad():
                    # Batched: null + prompt
                    raw = unet(torch.cat([lat_in, lat_in]), t,
                               encoder_hidden_states=torch.cat(
                                   [uncond_embeds, prompt_embeds])).sample
                    eps_null, eps_prompt_pred = raw.chunk(2)

                    # Text target
                    eps_text_target = unet(lat_in, t,
                                           encoder_hidden_states=text_target_embeds).sample

                # Standard CFG
                eps_cfg = eps_null + args.cfg_scale * (eps_prompt_pred - eps_null)

                # WHEN — text-based CAS
                cas_val, should_trigger = cas.compute(
                    eps_prompt_pred, eps_null, eps_text_target)
                cas_values.append(cas_val)

                if should_trigger:
                    with torch.no_grad():
                        eps_anchor = unet(lat_in, t,
                                          encoder_hidden_states=anchor_embeds).sample

                    # WHERE — Direction 1: Text spatial CAS
                    text_cas = compute_spatial_cas(
                        eps_prompt_pred, eps_null, eps_text_target,
                        neighborhood_size=args.neighborhood_size)
                    text_mask = make_soft_mask(
                        text_cas, args.spatial_threshold,
                        args.sigmoid_alpha, args.blur_sigma)

                    # WHERE — Direction 2: Image-based mask
                    img_mask = None
                    if use_img:
                        if args.img_where == "noise":
                            # Noise spatial CAS with eps_img_target
                            with torch.no_grad():
                                eps_img_target = unet(lat_in, t,
                                                       encoder_hidden_states=img_target_embeds).sample
                            img_cas = compute_spatial_cas(
                                eps_prompt_pred, eps_null, eps_img_target,
                                neighborhood_size=args.neighborhood_size)
                            img_mask = make_soft_mask(
                                img_cas, args.img_spatial_threshold,
                                args.sigmoid_alpha, args.blur_sigma)
                        elif args.img_where == "attn":
                            # Cross-attention map from img_target UNet call
                            attn_spatial = capture_attn_maps(
                                unet, img_target_embeds, lat_in, t,
                                target_resolutions=args.attn_resolutions)
                            img_mask = make_soft_mask(
                                attn_spatial.to(device), args.img_spatial_threshold,
                                args.sigmoid_alpha, args.blur_sigma)

                    # WHERE — Fusion
                    final_mask = fuse_masks(text_mask, img_mask, args.fusion)

                    # HOW — Anchor inpainting
                    mask = final_mask.to(eps_cfg.dtype)
                    blend = (args.safety_scale * mask).clamp(max=1.0)
                    eps_anchor_cfg = eps_null + args.cfg_scale * (eps_anchor - eps_null)
                    eps_final = eps_cfg * (1.0 - blend) + eps_anchor_cfg * blend

                    if torch.isnan(eps_final).any() or torch.isinf(eps_final).any():
                        eps_final = torch.where(torch.isfinite(eps_final), eps_final, eps_cfg)

                    guided_count += 1
                    mask_areas.append(float(final_mask.mean().item()))

                    if args.debug and step_idx % 10 == 0:
                        t_area = float(text_mask.mean())
                        i_area = float(img_mask.mean()) if img_mask is not None else 0
                        f_area = mask_areas[-1]
                        print(f"  [{step_idx:02d}] CAS={cas_val:.3f} "
                              f"text={t_area:.3f} img={i_area:.3f} final={f_area:.3f}")

                    if args.save_maps and step_idx % 10 == 0:
                        prefix = f"{prompt_idx:04d}_{sample_idx:02d}_s{step_idx:03d}"
                        md = outdir / "maps"
                        for name, m in [("text", text_mask), ("final", final_mask)]:
                            mn = m[0, 0].float().cpu().numpy()
                            Image.fromarray((np.clip(mn, 0, 1) * 255).astype(np.uint8), 'L').save(
                                str(md / f"{prefix}_{name}.png"))
                        if img_mask is not None:
                            mn = img_mask[0, 0].float().cpu().numpy()
                            Image.fromarray((np.clip(mn, 0, 1) * 255).astype(np.uint8), 'L').save(
                                str(md / f"{prefix}_img.png"))
                else:
                    eps_final = eps_cfg

                latents_prev = latents.clone()
                latents = scheduler.step(eps_final, t, latents).prev_sample

                if torch.isnan(latents).any() or torch.isinf(latents).any():
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

            all_stats.append({
                "prompt_idx": prompt_idx, "sample_idx": sample_idx,
                "seed": seed, "prompt": prompt[:100], "filename": fname,
                "guided_steps": guided_count, "total_steps": total_steps,
                "max_cas": max(cas_values) if cas_values else 0,
                "mean_mask_area": float(np.mean(mask_areas)) if mask_areas else 0,
            })

    with open(outdir / "generation_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    with open(outdir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    guided_imgs = sum(1 for s in all_stats if s["guided_steps"] > 0)
    print(f"\nDone! {len(all_stats)} images. Guided: {guided_imgs}/{len(all_stats)}")


if __name__ == "__main__":
    main()
