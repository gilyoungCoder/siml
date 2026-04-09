#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v22: Dual-Signal Anchor Inpainting — Text + Image Combined Erasing

Key insight (교수님):
  - Text (keyword-based): direct erasing of explicitly named unsafe concepts
  - Image (CLIP exemplar): catches ambiguous visual patterns text can't express
  - Union of both signals → comprehensive coverage

Architecture:
  WHEN: Noise CAS (threshold=0.6, sticky) — unchanged from v4
  WHERE (Dual-Signal):
    Signal 1 — Noise Spatial CAS (text-driven):
      Per-pixel cos(d_prompt, d_target) with 3x3 neighborhood pooling
      → text_mask: where the noise direction aligns with target concept
    Signal 2 — CLIP Image Cross-Attention Probe:
      Inject CLIP exemplar features as cross-attention keys
      → image_mask: where UNet attends to unsafe visual patterns
    Fusion: union(text_mask, image_mask) via max or soft-union
  HOW: Anchor inpainting (proven best from v4)

Multi-concept generalization:
  Each concept needs: target_keywords + exemplar_images (or generated ones)
  Same pipeline works for nudity, violence, harassment, etc.

Evidence:
  v4 (text only): SR 94.0% (cas=0.6)
  v20 (image only, noise_boost): SR 91.5% (cas=0.6, 4s)
  v22 hypothesis: text∪image > either alone
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

from attention_probe import (
    AttentionProbeStore,
    precompute_target_keys,
    register_attention_probe,
    restore_processors,
    compute_attention_spatial_mask,
    find_token_indices,
)


# =============================================================================
# Global CAS (WHEN)
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
# Signal 1: Noise Spatial CAS (text-driven WHERE)
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


def make_soft_mask(raw_map, threshold, sigmoid_alpha, blur_sigma):
    """Convert raw spatial map to soft [0,1] mask."""
    soft = torch.sigmoid(sigmoid_alpha * (raw_map - threshold))
    soft = soft.unsqueeze(0).unsqueeze(0)
    if blur_sigma > 0:
        soft = gaussian_blur_2d(soft, kernel_size=5, sigma=blur_sigma)
    return soft.clamp(0, 1)


# =============================================================================
# Signal 2: CLIP Image Probe Embedding Builders
# =============================================================================
def build_image_probe_embedding(clip_features, text_encoder, tokenizer, device,
                                pool_mode="cls_multi", max_tokens=16):
    """
    Build probe embedding from CLIP image features.

    pool_mode:
      cls_multi:  Each exemplar CLS as individual token (best from v20)
      cls_mean:   Mean-pool all CLS features → single token repeated
      cls_pool:   Mean-pool → single token (no repeat)
    """
    token_embedding = text_encoder.text_model.embeddings.token_embedding
    bos_embed = token_embedding(torch.tensor([tokenizer.bos_token_id], device=device))
    eos_embed = token_embedding(torch.tensor([tokenizer.eos_token_id], device=device))
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_embed = token_embedding(torch.tensor([pad_id], device=device))

    feats = clip_features.to(device).float()
    feats = F.normalize(feats, dim=-1)

    if pool_mode == "cls_multi":
        K = min(feats.shape[0], max_tokens, 75)
        tokens_list = [bos_embed]
        for i in range(K):
            tokens_list.append(feats[i:i+1])
        tokens_list.append(eos_embed)
        n_pad = 77 - len(tokens_list)
        for _ in range(n_pad):
            tokens_list.append(pad_embed)
        probe_indices = list(range(1, 1 + K))

    elif pool_mode in ("cls_mean", "cls_pool"):
        avg_feat = feats.mean(dim=0, keepdim=True)  # [1, 768]
        avg_feat = F.normalize(avg_feat, dim=-1)
        n_repeat = 4 if pool_mode == "cls_mean" else 1
        tokens_list = [bos_embed]
        for _ in range(n_repeat):
            tokens_list.append(avg_feat)
        tokens_list.append(eos_embed)
        n_pad = 77 - len(tokens_list)
        for _ in range(n_pad):
            tokens_list.append(pad_embed)
        probe_indices = list(range(1, 1 + n_repeat))

    else:
        raise ValueError(f"Unknown pool_mode: {pool_mode}")

    embeds = torch.cat(tokens_list, dim=0).unsqueeze(0)  # [1, 77, 768]
    return embeds, probe_indices


# =============================================================================
# Dual-Signal Mask Fusion
# =============================================================================
def fuse_dual_signal(text_mask, image_mask, fusion_mode="union",
                     union_temp=1.0, weight_text=0.5):
    """
    Combine text-driven noise mask and image-driven attention mask.

    fusion_mode:
      union:     max(text, image) — either signal triggers erasure
      soft_union: 1 - (1-text)*(1-image) — probabilistic union
      weighted:  w*text + (1-w)*image — linear blend
      text_only: ignore image signal
      image_only: ignore text signal
    """
    if image_mask is None or fusion_mode == "text_only":
        return text_mask
    if fusion_mode == "image_only":
        return image_mask

    # Ensure same shape
    if image_mask.shape != text_mask.shape:
        image_mask = F.interpolate(image_mask, size=text_mask.shape[-2:],
                                   mode='bilinear', align_corners=False)

    if fusion_mode == "union":
        return torch.max(text_mask, image_mask)

    elif fusion_mode == "soft_union":
        # P(A∪B) = 1 - (1-P(A))*(1-P(B))
        return 1.0 - (1.0 - text_mask) * (1.0 - image_mask)

    elif fusion_mode == "weighted":
        return weight_text * text_mask + (1.0 - weight_text) * image_mask

    elif fusion_mode == "boost":
        # Text as base, image boosts confidence
        return (text_mask * (1.0 + union_temp * image_mask)).clamp(0, 1)

    else:
        raise ValueError(f"Unknown fusion_mode: {fusion_mode}")


# =============================================================================
# Guidance (HOW) — Anchor Inpainting
# =============================================================================
def apply_anchor_inpaint(eps_cfg, eps_null, eps_anchor, soft_mask,
                         safety_scale, cfg_scale):
    mask = soft_mask.to(eps_cfg.dtype)
    eps_anchor_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
    blend = safety_scale * mask
    blend = blend.clamp(max=1.0)
    eps_final = eps_cfg * (1.0 - blend) + eps_anchor_cfg * blend

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
    p = ArgumentParser(description="v22: Dual-Signal (Text+Image) Anchor Inpainting")

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

    # WHERE — Signal 1: Text (noise spatial CAS)
    p.add_argument("--spatial_threshold", type=float, default=0.1)
    p.add_argument("--sigmoid_alpha", type=float, default=10.0)
    p.add_argument("--neighborhood_size", type=int, default=3)
    p.add_argument("--blur_sigma", type=float, default=1.0)

    # WHERE — Signal 2: Image (CLIP exemplar cross-attention probe)
    p.add_argument("--clip_embeddings", type=str, default=None,
                   help="CLIP exemplar .pt file. If None, text-only mode.")
    p.add_argument("--img_pool", type=str, default="cls_multi",
                   choices=["cls_multi", "cls_mean", "cls_pool"])
    p.add_argument("--max_exemplars", type=int, default=16)
    p.add_argument("--attn_resolutions", type=int, nargs="+", default=[16, 32])
    p.add_argument("--attn_threshold", type=float, default=0.3)
    p.add_argument("--attn_sigmoid_alpha", type=float, default=10.0)

    # WHERE — Dual-Signal Fusion
    p.add_argument("--fusion", type=str, default="union",
                   choices=["union", "soft_union", "weighted", "boost",
                            "text_only", "image_only"],
                   help="How to combine text and image masks")
    p.add_argument("--weight_text", type=float, default=0.5,
                   help="Text weight for weighted fusion")
    p.add_argument("--boost_temp", type=float, default=1.0,
                   help="Temperature for boost fusion")

    # HOW
    p.add_argument("--safety_scale", type=float, default=1.2)

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
    if args.clip_embeddings is None:
        if args.fusion != "text_only":
            print("[v22] No --clip_embeddings: forcing fusion=text_only")
            args.fusion = "text_only"
    return args


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_image = args.clip_embeddings is not None and args.fusion != "text_only"

    print(f"\n{'='*70}")
    print(f"v22: Dual-Signal (Text+Image) Anchor Inpainting")
    print(f"{'='*70}")
    print(f"  WHEN:    CAS threshold={args.cas_threshold}, sticky={args.cas_sticky}")
    print(f"  WHERE:")
    print(f"    Text:  noise spatial CAS (nbr={args.neighborhood_size}, "
          f"thr={args.spatial_threshold})")
    if use_image:
        print(f"    Image: CLIP probe ({args.img_pool}, "
              f"max_ex={args.max_exemplars}, attn_thr={args.attn_threshold})")
    else:
        print(f"    Image: disabled (text-only mode)")
    print(f"    Fusion: {args.fusion}")
    print(f"  HOW:     anchor_inpaint, ss={args.safety_scale}")
    print(f"  Targets: {args.target_concepts}")
    print(f"  Anchors: {args.anchor_concepts}")
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

    # ---- Setup image probe ----
    probe_store = None
    probe_token_indices = None
    original_processors = None

    if use_image:
        clip_data = torch.load(args.clip_embeddings, map_location="cpu")
        clip_features = clip_data["target_clip_features"].float()
        probe_embeds, probe_token_indices = build_image_probe_embedding(
            clip_features, text_encoder, tokenizer, device,
            pool_mode=args.img_pool, max_tokens=args.max_exemplars)
        print(f"  Image probe: {args.img_pool}, {len(probe_token_indices)} tokens")

        probe_store = AttentionProbeStore()
        target_keys = precompute_target_keys(
            unet, probe_embeds.to(dtype=next(unet.parameters()).dtype),
            args.attn_resolutions)
        original_processors = register_attention_probe(
            unet, probe_store, target_keys, args.attn_resolutions)

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

            for step_idx, t in enumerate(scheduler.timesteps):
                lat_in = scheduler.scale_model_input(latents, t)

                # ---- UNet: null + prompt (batched) ----
                with torch.no_grad():
                    if use_image:
                        probe_store.active = True
                        probe_store.reset()

                    lat_batch = torch.cat([lat_in, lat_in])
                    embed_batch = torch.cat([uncond_embeds, prompt_embeds])
                    raw = unet(lat_batch, t,
                               encoder_hidden_states=embed_batch).sample
                    eps_null, eps_prompt_pred = raw.chunk(2)

                    if use_image:
                        probe_store.active = False

                    # Target concept
                    eps_target = unet(lat_in, t,
                                      encoder_hidden_states=target_embeds).sample

                # Standard CFG
                eps_cfg = eps_null + args.cfg_scale * (eps_prompt_pred - eps_null)

                # ---- WHEN ----
                cas_val, should_trigger = cas.compute(
                    eps_prompt_pred, eps_null, eps_target)
                cas_values.append(cas_val)

                if should_trigger:
                    with torch.no_grad():
                        eps_anchor = unet(lat_in, t,
                                          encoder_hidden_states=anchor_embeds).sample

                    # ---- Signal 1: Text mask (noise spatial CAS) ----
                    spatial_cas = compute_spatial_cas(
                        eps_prompt_pred, eps_null, eps_target,
                        neighborhood_size=args.neighborhood_size)
                    text_mask = make_soft_mask(
                        spatial_cas, args.spatial_threshold,
                        args.sigmoid_alpha, args.blur_sigma)

                    # ---- Signal 2: Image mask (cross-attention probe) ----
                    image_mask = None
                    if use_image and probe_store.get_maps():
                        attn_spatial = compute_attention_spatial_mask(
                            probe_store,
                            token_indices=probe_token_indices,
                            target_resolution=64,
                            resolutions_to_use=args.attn_resolutions)
                        image_mask = make_soft_mask(
                            attn_spatial.to(device), args.attn_threshold,
                            args.attn_sigmoid_alpha, args.blur_sigma)

                    # ---- Dual-Signal Fusion ----
                    final_mask = fuse_dual_signal(
                        text_mask, image_mask,
                        fusion_mode=args.fusion,
                        union_temp=args.boost_temp,
                        weight_text=args.weight_text)

                    # ---- HOW: Anchor Inpainting ----
                    eps_final = apply_anchor_inpaint(
                        eps_cfg, eps_null, eps_anchor, final_mask,
                        args.safety_scale, args.cfg_scale)

                    guided_count += 1
                    mask_areas.append(float(final_mask.mean().item()))

                    if args.debug and step_idx % 10 == 0:
                        t_area = float(text_mask.mean())
                        i_area = float(image_mask.mean()) if image_mask is not None else 0
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
                        if image_mask is not None:
                            mn = image_mask[0, 0].float().cpu().numpy()
                            Image.fromarray((np.clip(mn, 0, 1) * 255).astype(np.uint8), 'L').save(
                                str(md / f"{prefix}_image.png"))
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

    # Save
    with open(outdir / "generation_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    with open(outdir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    guided_imgs = sum(1 for s in all_stats if s["guided_steps"] > 0)
    print(f"\nDone! {len(all_stats)} images generated.")
    print(f"  Guided: {guided_imgs}/{len(all_stats)}")

    if original_processors is not None:
        restore_processors(unet, original_processors)


if __name__ == "__main__":
    main()
