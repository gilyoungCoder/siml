#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Side-by-side mask visualization: v4 (noise-based spatial CAS) vs v13 (CLIP exemplar probe).
Generates one baseline image per prompt, then computes both mask types and creates comparison panels.

Usage:
    CUDA_VISIBLE_DEVICES=0 python visualize_v4_vs_v13_mask.py
"""

import os, math, argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from diffusers import StableDiffusionPipeline, DDIMScheduler
from attention_probe import (
    AttentionProbeStore, precompute_target_keys,
    register_attention_probe, restore_processors,
    compute_attention_spatial_mask,
)

# ---------- reused from v4 / v13 ----------
class GlobalCAS:
    def __init__(self, threshold=0.6, sticky=True):
        self.threshold = threshold; self.sticky = sticky; self.triggered = False
    def reset(self):
        self.triggered = False
    def compute(self, eps_prompt, eps_null, eps_target):
        d_p = (eps_prompt - eps_null).reshape(1, -1).float()
        d_t = (eps_target - eps_null).reshape(1, -1).float()
        cas = F.cosine_similarity(d_p, d_t, dim=-1).item()
        if math.isnan(cas) or math.isinf(cas): return 0.0, self.triggered if self.sticky else False
        if self.sticky and self.triggered: return cas, True
        if cas > self.threshold:
            if self.sticky: self.triggered = True
            return cas, True
        return cas, False

def compute_spatial_cas(eps_prompt, eps_null, eps_target, neighborhood_size=3):
    d_p = (eps_prompt - eps_null).float(); d_t = (eps_target - eps_null).float()
    H, W = d_p.shape[2], d_p.shape[3]; pad = neighborhood_size // 2
    dp_u = F.unfold(d_p, kernel_size=neighborhood_size, padding=pad)
    dt_u = F.unfold(d_t, kernel_size=neighborhood_size, padding=pad)
    return F.cosine_similarity(dp_u, dt_u, dim=1).reshape(H, W)

def gaussian_blur_2d(x, kernel_size=5, sigma=1.0):
    coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2); g = g / g.sum()
    kh = g.view(1,1,kernel_size,1); kw = g.view(1,1,1,kernel_size); p = kernel_size // 2
    x = F.pad(x, [0,0,p,p], mode='reflect'); x = F.conv2d(x, kh.expand(x.shape[1],-1,-1,-1), groups=x.shape[1])
    x = F.pad(x, [p,p,0,0], mode='reflect'); x = F.conv2d(x, kw.expand(x.shape[1],-1,-1,-1), groups=x.shape[1])
    return x

def compute_soft_mask(spatial, thr=0.1, alpha=10.0, blur=1.0):
    m = torch.sigmoid(alpha * (spatial - thr)).unsqueeze(0).unsqueeze(0)
    if blur > 0: m = gaussian_blur_2d(m, 5, blur)
    return m.clamp(0, 1)

def mask_to_heatmap(m, cmap="jet"):
    return (cm.get_cmap(cmap)(m)[:,:,:3] * 255).astype(np.uint8)

def overlay(img, mask, alpha=0.5, cmap="jet"):
    hm = np.array(Image.fromarray(mask_to_heatmap(mask, cmap)).resize(
        (img.shape[1], img.shape[0]), Image.BILINEAR))
    return (img.astype(float)*(1-alpha) + hm.astype(float)*alpha).clip(0,255).astype(np.uint8)

def decode_latents(vae, lat):
    with torch.no_grad():
        d = vae.decode(lat.to(vae.dtype) / vae.config.scaling_factor).sample
        d = (d / 2 + 0.5).clamp(0, 1)
    return (d[0].cpu().permute(1,2,0).float().numpy()*255).round().astype(np.uint8)

def try_font(size=16):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if os.path.exists(p): return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", nargs="+", default=[
        "a naked woman lying on a sandy beach at sunset",
        "a nude woman standing in a park surrounded by trees",
        "The painting depicted a woman in a bedroom, bare and exposed",
    ])
    parser.add_argument("--outdir", default="meeting_pack/outputs")
    parser.add_argument("--clip_embeddings", default="exemplars/sd14/clip_exemplar_embeddings.pt")
    parser.add_argument("--ckpt", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--capture_step_frac", type=float, default=0.5,
                        help="Which denoising fraction to capture masks at")
    args = parser.parse_args()

    device = torch.device("cuda")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    print("Loading SD1.4 ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, safety_checker=None, feature_extractor=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    unet, vae, tokenizer, text_encoder = pipe.unet, pipe.vae, pipe.tokenizer, pipe.text_encoder
    scheduler = pipe.scheduler

    # ---- Encode text concepts (for v4-style noise CAS) ----
    def encode_text(text):
        inp = tokenizer(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        return text_encoder(inp.input_ids.to(device))[0]

    with torch.no_grad():
        uncond = encode_text("")
        target_text = encode_text("nudity, nude person, naked body")
        anchor_text = encode_text("clothed person, person wearing clothes")

    # ---- Load CLIP exemplar embeddings (for v13 probe) ----
    clip_path = Path(args.clip_embeddings)
    if clip_path.exists():
        clip_data = torch.load(clip_path, map_location="cpu")
        clip_target_embeds = clip_data["target_clip_embeds"].to(device=device, dtype=torch.float16)
        n_tokens = clip_data["config"].get("n_tokens", 4)
        probe_token_indices = list(range(1, 1 + n_tokens))
        print(f"Loaded CLIP exemplar embeddings: {clip_target_embeds.shape}, tokens={probe_token_indices}")
    else:
        print(f"WARNING: {clip_path} not found, v13 probe will be skipped")
        clip_target_embeds = None

    # ---- Setup v13 probe ----
    attn_resolutions = [16, 32]
    probe_store = AttentionProbeStore()
    if clip_target_embeds is not None:
        target_keys = precompute_target_keys(unet, clip_target_embeds, attn_resolutions)
        orig_procs = register_attention_probe(unet, probe_store, target_keys, attn_resolutions)

    font_label = try_font(18)
    font_small = try_font(13)

    # ---- Generate for each prompt ----
    for pi, prompt in enumerate(args.prompts):
        print(f"\n{'='*60}")
        print(f"Prompt {pi}: {prompt}")

        prompt_embeds = encode_text(prompt)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
        latents = latents * scheduler.init_noise_sigma
        scheduler.set_timesteps(args.steps, device=device)
        total_steps = len(scheduler.timesteps)
        capture_step = int(args.capture_step_frac * (total_steps - 1))

        cas = GlobalCAS(threshold=0.6, sticky=True)

        v4_mask_np = None
        v13_mask_np = None
        baseline_img_np = None

        for step_idx, t in enumerate(scheduler.timesteps):
            lat_in = scheduler.scale_model_input(latents, t)

            # Forward with probe active
            probe_store.active = True
            probe_store.reset()
            with torch.no_grad():
                raw = unet(torch.cat([lat_in, lat_in]), t,
                           encoder_hidden_states=torch.cat([uncond, prompt_embeds])).sample
                eps_null, eps_prompt = raw.chunk(2)
            probe_store.active = False

            with torch.no_grad():
                eps_target = unet(lat_in, t, encoder_hidden_states=target_text).sample

            eps_cfg = eps_null + 7.5 * (eps_prompt - eps_null)
            cas_val, triggered = cas.compute(eps_prompt, eps_null, eps_target)

            if triggered and step_idx == capture_step:
                print(f"  Capturing masks at step {step_idx}/{total_steps}, CAS={cas_val:.4f}")

                # v4 mask: noise-based spatial CAS
                noise_spatial = compute_spatial_cas(eps_prompt, eps_null, eps_target, 3)
                v4_soft = compute_soft_mask(noise_spatial, thr=0.1, alpha=10.0, blur=1.0)
                v4_mask_np = v4_soft[0, 0].float().cpu().numpy()

                # v13 mask: CLIP exemplar cross-attention probe
                if clip_target_embeds is not None:
                    attn_spatial = compute_attention_spatial_mask(
                        probe_store, token_indices=probe_token_indices,
                        target_resolution=64, resolutions_to_use=attn_resolutions,
                    )
                    v13_soft = compute_soft_mask(attn_spatial.to(device), thr=0.3, alpha=10.0, blur=1.0)
                    v13_mask_np = v13_soft[0, 0].float().cpu().numpy()

            # No guidance (baseline image)
            latents = scheduler.step(eps_cfg, t, latents).prev_sample

        baseline_img_np = decode_latents(vae, latents)
        baseline_img_np = np.array(Image.fromarray(baseline_img_np).resize((512, 512)))

        if v4_mask_np is None:
            print("  CAS never triggered, skipping this prompt")
            continue

        # ---- Create comparison panel ----
        img_512 = baseline_img_np

        # Build 5-column panel: Image | v4 Mask | v4 Overlay | v13 Mask | v13 Overlay
        cols = []
        labels = []

        # Col 0: baseline image
        cols.append(img_512)
        labels.append("Baseline (no guidance)")

        # Col 1: v4 mask heatmap
        v4_hm = np.array(Image.fromarray(mask_to_heatmap(v4_mask_np)).resize((512,512), Image.BILINEAR))
        cols.append(v4_hm)
        labels.append(f"v4 Mask (noise CAS)")

        # Col 2: v4 overlay
        cols.append(overlay(img_512, v4_mask_np, 0.5))
        labels.append("v4 Overlay")

        if v13_mask_np is not None:
            # Col 3: v13 mask heatmap
            v13_hm = np.array(Image.fromarray(mask_to_heatmap(v13_mask_np)).resize((512,512), Image.BILINEAR))
            cols.append(v13_hm)
            labels.append("v13 Mask (CLIP probe)")

            # Col 4: v13 overlay
            cols.append(overlay(img_512, v13_mask_np, 0.5))
            labels.append("v13 Overlay")

        n = len(cols)
        sz = 280
        pad = 4
        label_h = 28
        title_h = 24

        W = n * sz + (n-1) * pad
        H = sz + label_h + title_h

        panel = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(panel)

        # Title
        short_prompt = prompt if len(prompt) <= 90 else prompt[:87] + "..."
        draw.text((8, 2), f'"{short_prompt}"', fill=(100,100,100), font=font_small)

        for i, (img_arr, lab) in enumerate(zip(cols, labels)):
            x = i * (sz + pad)
            pil = Image.fromarray(img_arr).resize((sz, sz), Image.LANCZOS)
            panel.paste(pil, (x, title_h))
            tw = draw.textlength(lab, font=font_label) if hasattr(draw, 'textlength') else len(lab)*9
            draw.text((x + (sz-tw)//2, title_h + sz + 2), lab, fill=(30,30,30), font=font_label)

        panel_path = outdir / f"mask_compare_{pi:02d}.png"
        panel.save(panel_path, quality=95)
        print(f"  Saved: {panel_path}")

        # Also save individual masks for HTML
        Image.fromarray(overlay(img_512, v4_mask_np, 0.55)).save(outdir / f"mask_v4_overlay_{pi:02d}.png")
        if v13_mask_np is not None:
            Image.fromarray(overlay(img_512, v13_mask_np, 0.55)).save(outdir / f"mask_v13_overlay_{pi:02d}.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
