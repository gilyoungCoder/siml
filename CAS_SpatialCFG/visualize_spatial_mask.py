#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spatial Mask Visualization — V4 anchor_inpaint best config
GradCAM-style overlay showing WHERE the spatial mask activates.

Usage:
    CUDA_VISIBLE_DEVICES=0 python visualize_spatial_mask.py \
        --prompt "standing nude woman" \
        --outdir outputs/mask_vis
"""

import os
import math
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from diffusers import StableDiffusionPipeline, DDIMScheduler

# =============================================================================
# Copy of core functions from generate_v4.py
# =============================================================================

class GlobalCAS:
    def __init__(self, threshold=0.3, sticky=True):
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


def compute_spatial_cas(eps_prompt, eps_null, eps_target, neighborhood_size=3):
    d_prompt = (eps_prompt - eps_null).float()
    d_target = (eps_target - eps_null).float()
    H, W = d_prompt.shape[2], d_prompt.shape[3]
    pad = neighborhood_size // 2
    d_p = F.unfold(d_prompt, kernel_size=neighborhood_size, padding=pad)
    d_t = F.unfold(d_target, kernel_size=neighborhood_size, padding=pad)
    spatial_cas = F.cosine_similarity(d_p, d_t, dim=1).reshape(H, W)
    return spatial_cas


def gaussian_blur_2d(x, kernel_size=5, sigma=1.0):
    coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    kernel_h = g.view(1, 1, kernel_size, 1)
    kernel_w = g.view(1, 1, 1, kernel_size)
    x = F.pad(x, [0, 0, kernel_size//2, kernel_size//2], mode='reflect')
    x = F.conv2d(x, kernel_h.expand(x.shape[1], -1, -1, -1), groups=x.shape[1])
    x = F.pad(x, [kernel_size//2, kernel_size//2, 0, 0], mode='reflect')
    x = F.conv2d(x, kernel_w.expand(x.shape[1], -1, -1, -1), groups=x.shape[1])
    return x


def compute_soft_mask(spatial_cas, spatial_threshold=0.1, sigmoid_alpha=10.0,
                      blur_sigma=1.0):
    soft_mask = torch.sigmoid(sigmoid_alpha * (spatial_cas - spatial_threshold))
    soft_mask = soft_mask.unsqueeze(0).unsqueeze(0)
    if blur_sigma > 0:
        soft_mask = gaussian_blur_2d(soft_mask, kernel_size=5, sigma=blur_sigma)
    return soft_mask.clamp(0, 1)


def encode_concepts(text_encoder, tokenizer, concepts, device):
    all_embeds = []
    for concept in concepts:
        inputs = tokenizer(concept, padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt")
        embeds = text_encoder(inputs.input_ids.to(device))[0]
        all_embeds.append(embeds)
    return torch.stack(all_embeds).mean(dim=0)


def decode_latents(vae, latents):
    """Decode latent tensor to uint8 numpy image [H, W, 3]."""
    with torch.no_grad():
        decoded = vae.decode(latents.to(vae.dtype) / vae.config.scaling_factor).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
    img = (decoded[0].cpu().permute(1, 2, 0).float().numpy() * 255).round().astype(np.uint8)
    return img


# =============================================================================
# Visualization helpers
# =============================================================================

def mask_to_heatmap(mask_np, colormap="jet"):
    """Convert float [0,1] mask [H,W] → RGB uint8 [H,W,3] heatmap."""
    cmap = cm.get_cmap(colormap)
    rgba = cmap(mask_np)           # [H, W, 4]
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def overlay_heatmap(img_np, mask_np, alpha=0.5, colormap="jet"):
    """Blend image and heatmap mask. Returns uint8 [H, W, 3]."""
    heatmap = mask_to_heatmap(mask_np, colormap)
    heatmap_resized = np.array(Image.fromarray(heatmap).resize(
        (img_np.shape[1], img_np.shape[0]), Image.BILINEAR))
    blended = (img_np.astype(float) * (1 - alpha) +
               heatmap_resized.astype(float) * alpha).clip(0, 255).astype(np.uint8)
    return blended


def save_panel(outdir, img_np, masks_at_steps, cas_map_last, soft_mask_last,
               prompt, step_indices, total_steps):
    """
    Save a comprehensive visualization panel:
    Row 1: final image | soft mask overlay (last step) | raw CAS map (last step)
    Row 2: soft mask evolution across denoising steps
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mask_last_np = soft_mask_last[0, 0].float().cpu().numpy()  # [H, W]
    cas_last_np  = cas_map_last.float().cpu().numpy()           # [H, W]

    # --- Save individual files ---
    Image.fromarray(img_np).save(str(outdir / "final_image.png"))

    # Soft mask (last step) as grayscale
    mask_img = (mask_last_np * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(mask_img, 'L').save(str(outdir / "soft_mask_last.png"))

    # Raw spatial CAS map (normalized to [0,1] from [-1,1])
    cas_norm = (cas_last_np.clip(-1, 1) + 1) / 2
    cas_img = (cas_norm * 255).astype(np.uint8)
    Image.fromarray(cas_img, 'L').save(str(outdir / "spatial_cas_map_last.png"))

    # Overlay: soft mask heatmap on final image
    overlay = overlay_heatmap(img_np, mask_last_np, alpha=0.55, colormap="jet")
    Image.fromarray(overlay).save(str(outdir / "overlay_soft_mask.png"))

    # Overlay: raw CAS heatmap on final image
    overlay_cas = overlay_heatmap(img_np, cas_norm, alpha=0.55, colormap="hot")
    Image.fromarray(overlay_cas).save(str(outdir / "overlay_cas_map.png"))

    # --- Main panel figure ---
    n_evo = len(masks_at_steps)
    fig_cols = max(5, n_evo + 1)
    fig, axes = plt.subplots(2, fig_cols, figsize=(3.5 * fig_cols, 7))
    fig.suptitle(f'Spatial Mask Visualization\nPrompt: "{prompt}"\n'
                 f'Config: V4 anchor_inpaint s=1.0 t=0.1 CAS_thr=0.6',
                 fontsize=11, y=1.01)

    # Row 0 — overview
    titles_row0 = ["Final Image", "Soft Mask Overlay\n(last step)",
                   "Soft Mask\n(grayscale)", "Spatial CAS\n(raw)", "CAS Overlay\n(hot)"]
    imgs_row0 = [
        img_np,
        overlay,
        np.stack([mask_img, mask_img, mask_img], axis=-1),
        np.stack([cas_img, cas_img, cas_img], axis=-1),
        overlay_cas,
    ]

    for col, (title, im) in enumerate(zip(titles_row0, imgs_row0)):
        if col < fig_cols:
            axes[0, col].imshow(im)
            axes[0, col].set_title(title, fontsize=9)
            axes[0, col].axis('off')

    # Hide extra cols in row0
    for col in range(len(titles_row0), fig_cols):
        axes[0, col].axis('off')

    # Row 1 — mask evolution
    axes[1, 0].imshow(img_np)
    axes[1, 0].set_title("Final Image", fontsize=9)
    axes[1, 0].axis('off')

    cmap_jet = cm.get_cmap("jet")
    for i, (step_i, m) in enumerate(zip(step_indices, masks_at_steps)):
        col = i + 1
        if col < fig_cols:
            m_np = m[0, 0].float().cpu().numpy()
            pct = int(100 * step_i / max(total_steps - 1, 1))
            axes[1, col].imshow(m_np, cmap='jet', vmin=0, vmax=1)
            axes[1, col].set_title(f"step {step_i}/{total_steps}\n({pct}%)", fontsize=9)
            axes[1, col].axis('off')

    for col in range(n_evo + 1, fig_cols):
        axes[1, col].axis('off')

    # Colorbar for heatmap
    sm = plt.cm.ScalarMappable(cmap=cmap_jet, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=axes[1, :], fraction=0.01, pad=0.01, label="Mask intensity")

    plt.tight_layout()
    panel_path = str(outdir / "panel.png")
    fig.savefig(panel_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved panel: {panel_path}")

    return panel_path


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="a standing nude woman",
                   help="Test prompt to visualize")
    p.add_argument("--outdir", type=str,
                   default="/mnt/home/yhgil99/unlearning/CAS_SpatialCFG/outputs/mask_vis")
    p.add_argument("--ckpt", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=7.5)
    # Best config: V4 anchor_inpaint s=1.0 t=0.1 CAS=0.6
    p.add_argument("--cas_threshold", type=float, default=0.6)
    p.add_argument("--spatial_threshold", type=float, default=0.1)
    p.add_argument("--safety_scale", type=float, default=1.0)
    p.add_argument("--sigmoid_alpha", type=float, default=10.0)
    p.add_argument("--blur_sigma", type=float, default=1.0)
    p.add_argument("--neighborhood_size", type=int, default=3)
    p.add_argument("--target_concepts", nargs="+",
                   default=["nudity", "nude person", "naked body"])
    p.add_argument("--anchor_concepts", nargs="+",
                   default=["clothed person", "person wearing clothes"])
    # Capture mask at these step fractions (0.0 = first, 1.0 = last)
    p.add_argument("--capture_fracs", nargs="+", type=float,
                   default=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                   help="Denoising step fractions at which to capture mask snapshots")
    p.add_argument("--no_guidance", action="store_true",
                   help="Run baseline (no guidance) for comparison")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Spatial Mask Visualization")
    print(f"  Prompt: {args.prompt}")
    print(f"  Config: anchor_inpaint, s={args.safety_scale}, "
          f"spatial_thr={args.spatial_threshold}, CAS_thr={args.cas_threshold}")
    print(f"  Output: {outdir}")
    print(f"{'='*60}\n")

    # ---- Load model ----
    print("Loading SD1.4 ...")
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
    print("Encoding concepts ...")
    with torch.no_grad():
        target_embeds = encode_concepts(text_encoder, tokenizer, args.target_concepts, device)
        anchor_embeds = encode_concepts(text_encoder, tokenizer, args.anchor_concepts, device)

        uncond_inputs = tokenizer("", padding="max_length",
                                  max_length=tokenizer.model_max_length,
                                  truncation=True, return_tensors="pt")
        uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]

        prompt_inputs = tokenizer(args.prompt, padding="max_length",
                                  max_length=tokenizer.model_max_length,
                                  truncation=True, return_tensors="pt")
        prompt_embeds = text_encoder(prompt_inputs.input_ids.to(device))[0]

    # ---- Init ----
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(args.steps, device=device)
    total_steps = len(scheduler.timesteps)

    cas_detector = GlobalCAS(threshold=args.cas_threshold, sticky=True)

    # Which steps to capture mask snapshots
    capture_steps = set(
        min(int(f * (total_steps - 1)), total_steps - 1)
        for f in args.capture_fracs
    )

    # Storage
    masks_at_steps = []
    step_indices_captured = []
    cas_map_last = None
    soft_mask_last = None
    cas_values = []
    triggered_at = None

    print(f"Running denoising ({total_steps} steps) ...")

    for step_idx, t in enumerate(scheduler.timesteps):
        lat_in = scheduler.scale_model_input(latents, t)

        with torch.no_grad():
            lat_batch = torch.cat([lat_in, lat_in])
            embed_batch = torch.cat([uncond_embeds, prompt_embeds])
            raw = unet(lat_batch, t, encoder_hidden_states=embed_batch).sample
            eps_null, eps_prompt = raw.chunk(2)

            eps_target = unet(lat_in, t,
                              encoder_hidden_states=target_embeds).sample

        eps_cfg = eps_null + args.cfg_scale * (eps_prompt - eps_null)

        cas_val, should_guide = cas_detector.compute(eps_prompt, eps_null, eps_target)
        cas_values.append(cas_val)
        if should_guide and triggered_at is None:
            triggered_at = step_idx

        if should_guide:
            # Always compute spatial mask when CAS triggers (for visualization)
            spatial_cas = compute_spatial_cas(eps_prompt, eps_null, eps_target,
                                              args.neighborhood_size)
            soft_mask = compute_soft_mask(
                spatial_cas,
                spatial_threshold=args.spatial_threshold,
                sigmoid_alpha=args.sigmoid_alpha,
                blur_sigma=args.blur_sigma,
            )

            # Store last
            cas_map_last = spatial_cas.detach().clone()
            soft_mask_last = soft_mask.detach().clone()

            # Capture at requested steps
            if step_idx in capture_steps:
                masks_at_steps.append(soft_mask.detach().clone())
                step_indices_captured.append(step_idx)
                mode_str = "NO_GUIDE" if args.no_guidance else "GUIDED"
                print(f"  [step {step_idx:02d}] CAS={cas_val:.3f} {mode_str} "
                      f"mask_area={float(soft_mask.mean()):.3f}")

            if not args.no_guidance:
                # Apply anchor_inpaint guidance
                with torch.no_grad():
                    eps_anchor = unet(lat_in, t,
                                      encoder_hidden_states=anchor_embeds).sample
                eps_anchor_cfg = eps_null + args.cfg_scale * (eps_anchor - eps_null)
                mask_s = soft_mask.to(eps_cfg.dtype)
                eps_final = (eps_cfg * (1.0 - args.safety_scale * mask_s) +
                             eps_anchor_cfg * (args.safety_scale * mask_s))
            else:
                eps_final = eps_cfg  # baseline image, but mask still computed

        else:
            eps_final = eps_cfg
            if step_idx in capture_steps:
                print(f"  [step {step_idx:02d}] CAS={cas_val:.3f} skip (CAS not triggered)")

        latents = scheduler.step(eps_final, t, latents).prev_sample

    # ---- Decode ----
    print("Decoding final image ...")
    img_np = decode_latents(vae, latents)
    img_np = np.array(Image.fromarray(img_np).resize((512, 512)))

    # Handle case where CAS never triggered
    if soft_mask_last is None:
        print("  WARNING: CAS never triggered (prompt not detected as unsafe). "
              "Forcing spatial mask at last step for visualization...")
        lat_in = scheduler.scale_model_input(latents, scheduler.timesteps[-1])
        with torch.no_grad():
            lat_batch = torch.cat([lat_in, lat_in])
            embed_batch = torch.cat([uncond_embeds, prompt_embeds])
            raw = unet(lat_batch, scheduler.timesteps[-1],
                       encoder_hidden_states=embed_batch).sample
            eps_null, eps_prompt = raw.chunk(2)
            eps_target = unet(lat_in, scheduler.timesteps[-1],
                              encoder_hidden_states=target_embeds).sample
        spatial_cas = compute_spatial_cas(eps_prompt, eps_null, eps_target,
                                          args.neighborhood_size)
        soft_mask_last = compute_soft_mask(
            spatial_cas,
            spatial_threshold=args.spatial_threshold,
            sigmoid_alpha=args.sigmoid_alpha,
            blur_sigma=args.blur_sigma,
        )
        cas_map_last = spatial_cas
        if not masks_at_steps:
            masks_at_steps = [soft_mask_last]
            step_indices_captured = [total_steps - 1]

    # ---- Save visualization ----
    print("Saving visualizations ...")
    panel_path = save_panel(
        outdir=outdir,
        img_np=img_np,
        masks_at_steps=masks_at_steps,
        cas_map_last=cas_map_last,
        soft_mask_last=soft_mask_last,
        prompt=args.prompt,
        step_indices=step_indices_captured,
        total_steps=total_steps,
    )

    # ---- Print summary ----
    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"  CAS triggered at step: {triggered_at} / {total_steps} "
          f"({'triggered' if triggered_at is not None else 'NOT triggered'})")
    print(f"  Peak CAS value: {max(cas_values):.4f} (threshold={args.cas_threshold})")
    if soft_mask_last is not None:
        print(f"  Final mask area (mean): {float(soft_mask_last.mean()):.4f}")
        print(f"  Final mask area (>0.5): "
              f"{float((soft_mask_last > 0.5).float().mean()):.4f}")
    print(f"\n  Outputs:")
    print(f"    {outdir}/final_image.png        — generated image")
    print(f"    {outdir}/overlay_soft_mask.png  — jet heatmap overlay")
    print(f"    {outdir}/overlay_cas_map.png    — raw CAS heatmap overlay")
    print(f"    {outdir}/soft_mask_last.png     — grayscale mask")
    print(f"    {outdir}/panel.png              — full panel (evolution)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
