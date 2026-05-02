#!/usr/bin/env python
"""
SafeGen on SD3: Training-Free Safe Generation via Dual-Probe Spatial Guidance.

Adapts SafeGen's When-Where-How framework from SD1.4 UNet to SD3 MMDiT:
  WHEN:  Global CAS on velocity predictions (same cosine similarity logic)
  WHERE: Joint-attention probe (image→text sub-matrix from MMDiT blocks)
  HOW:   Anchor inpaint / hybrid / target subtraction on velocity space

Key SD3 differences:
  - MMDiT transformer instead of UNet (joint attention, not cross-attention)
  - Flow matching (velocity prediction) instead of DDPM (noise prediction)
  - Triple text encoders (CLIP-L + CLIP-G + T5-XXL)
  - 16-channel VAE, latent resolution depends on image size
  - CPU offload required on RTX 3090 (24GB)

Based on: SafeGen/safegen/generate_family.py (latest method)
"""

import os
import sys
import json
import math
import random
import csv
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler

from attention_probe_sd3 import (
    SD3AttentionProbeStore,
    register_sd3_attention_probe,
    restore_sd3_processors,
    compute_sd3_spatial_mask,
    compute_sd3_image_probe_mask,
    build_sd3_image_probe_embeds,
    build_grouped_sd3_image_probe_embeds,
)


SD3_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"


# ---------------------------------------------------------------------------
# WHEN: Global CAS (works on velocity predictions for SD3)
# ---------------------------------------------------------------------------
class GlobalCAS:
    """
    CAS = cos(v_prompt - v_null, v_target - v_null)
    Same logic as SD1.4 version — cosine similarity is direction-based,
    so it works for both noise (epsilon) and velocity (v) predictions.
    """

    def __init__(self, threshold=0.4, sticky=True):
        self.threshold = threshold
        self.sticky = sticky
        self.triggered = False

    def reset(self):
        self.triggered = False

    def compute(self, v_prompt, v_null, v_target):
        dp = (v_prompt - v_null).reshape(1, -1).float()
        dt = (v_target - v_null).reshape(1, -1).float()
        c = F.cosine_similarity(dp, dt, dim=-1).item()
        if math.isnan(c) or math.isinf(c):
            return 0.0, self.triggered if self.sticky else False
        if self.sticky and self.triggered:
            return c, True
        if c > self.threshold:
            if self.sticky:
                self.triggered = True
            return c, True
        return c, False


# ---------------------------------------------------------------------------
# WHERE: Mask utilities
# ---------------------------------------------------------------------------
def gaussian_blur_2d(x, kernel_size=5, sigma=1.0):
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g /= g.sum()
    kh, kw = g.view(1, 1, kernel_size, 1), g.view(1, 1, 1, kernel_size)
    p = kernel_size // 2
    x = F.pad(x, [0, 0, p, p], "reflect")
    x = F.conv2d(x, kh.expand(x.shape[1], -1, -1, -1), groups=x.shape[1])
    x = F.pad(x, [p, p, 0, 0], "reflect")
    return F.conv2d(x, kw.expand(x.shape[1], -1, -1, -1), groups=x.shape[1])


def make_probe_mask(attn_spatial, threshold, alpha=10.0, blur=1.0, device=None):
    m = torch.sigmoid(alpha * (attn_spatial.to(device) - threshold))
    m = m.unsqueeze(0).unsqueeze(0)
    if blur > 0:
        m = gaussian_blur_2d(m, sigma=blur)
    return m.clamp(0, 1)


# ---------------------------------------------------------------------------
# HOW: Guidance (identical math, operates on velocity space)
# ---------------------------------------------------------------------------
def apply_guidance(v_cfg, v_null, v_prompt, v_target, v_anchor,
                   mask, mode, safety_scale, cfg_scale):
    m = mask.to(v_cfg.dtype)
    s = safety_scale

    if mode == "anchor_inpaint":
        va_cfg = v_null + cfg_scale * (v_anchor - v_null)
        blend = (s * m).clamp(max=1.0)
        out = v_cfg * (1 - blend) + va_cfg * blend

    elif mode == "hybrid":
        out = (v_cfg
               - s * m * (v_target - v_null)
               + s * m * (v_anchor - v_null))

    elif mode == "target_sub":
        out = v_cfg - s * m * (v_target - v_null)

    else:
        raise ValueError(f"Unknown guidance mode: {mode}")

    if torch.isnan(out).any() or torch.isinf(out).any():
        out = torch.where(torch.isfinite(out), out, v_cfg)
    return out


def apply_family_guidance(v_cfg, v_null, family_masks, family_targets, family_anchors,
                          mode, safety_scale, cfg_scale):
    """Per-family guidance with region-specific masks."""
    out = v_cfg.clone()
    s = safety_scale

    for mask, v_target, v_anchor in zip(family_masks, family_targets, family_anchors):
        m = mask.to(v_cfg.dtype)

        if mode == "anchor_inpaint":
            va_cfg = v_null + cfg_scale * (v_anchor - v_null)
            blend = (s * m).clamp(max=1.0)
            out = out * (1 - blend) + va_cfg * blend
        elif mode == "hybrid":
            out = (out
                   - s * m * (v_target - v_null)
                   + s * m * (v_anchor - v_null))
        elif mode == "target_sub":
            out = out - s * m * (v_target - v_null)
        else:
            raise ValueError(f"Unknown guidance mode: {mode}")

    if torch.isnan(out).any() or torch.isinf(out).any():
        out = torch.where(torch.isfinite(out), out, v_cfg)
    return out


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def load_prompts(filepath):
    fp = Path(filepath)
    if fp.suffix == ".csv":
        prompts = []
        with open(fp) as f:
            reader = csv.DictReader(f)
            col = next(
                (c for c in ["sensitive prompt", "adv_prompt", "prompt",
                              "target_prompt", "text", "Prompt", "Text"]
                 if c in reader.fieldnames), None)
            if not col:
                raise ValueError(f"No prompt col in {reader.fieldnames}")
            for row in reader:
                p = row[col].strip()
                if p:
                    prompts.append(p)
        return prompts
    return [line.strip() for line in open(fp) if line.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# SD3 text encoding helper
# ---------------------------------------------------------------------------
def encode_sd3_prompt(pipe, prompt, device):
    """Encode a single prompt through SD3's triple text encoders."""
    result = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    # Returns (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled)
    # When do_classifier_free_guidance=False, negatives are None
    return result[0], result[2]  # prompt_embeds, pooled_projections


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = ArgumentParser(description="SafeGen on SD3 — Dual-Probe Spatial Guidance")
    p.add_argument("--model_id", default=SD3_MODEL_ID)
    p.add_argument("--prompts", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--nsamples", type=int, default=1)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg_scale", type=float, default=7.0)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)

    # WHEN
    p.add_argument("--cas_threshold", type=float, default=0.6,
                   help="CAS threshold (SD3 may need different from SD1.4's 0.6)")

    # WHERE — Probe
    p.add_argument("--probe_mode", default="text",
                   choices=["text", "image", "both", "none"],
                   help="text=joint attention text probe, image=CLIP-exemplar "
                        "image probe, both=union of text+image, none=global")
    p.add_argument("--probe_blocks", type=int, nargs="+", default=None,
                   help="Which transformer blocks to hook (None=middle third)")
    p.add_argument("--attn_threshold", type=float, default=0.3)
    p.add_argument("--img_attn_threshold", type=float, default=None,
                   help="Separate threshold for image probe (defaults to --attn_threshold)")
    p.add_argument("--attn_sigmoid_alpha", type=float, default=10.0)
    p.add_argument("--blur_sigma", type=float, default=1.0)
    p.add_argument("--clip_embeddings", default=None,
                   help="Path to .pt file with CLIP exemplar image features "
                        "(target_clip_features: [N,768]). Required for "
                        "probe_mode in {image, both}.")
    p.add_argument("--n_img_tokens", type=int, default=4,
                   help="Number of pseudo-text token slots to overwrite with "
                        "the CLIP exemplar feature (SafeGen image probe).")
    p.add_argument("--family_config", default=None,
                   help="Path to grouped exemplar .pt file with family metadata "
                        "and per-family CLIP features.")
    p.add_argument("--family_guidance", action="store_true",
                   help="Enable family-specific target/anchor guidance.")

    # HOW
    p.add_argument("--how_mode", default="anchor_inpaint",
                   choices=["anchor_inpaint", "hybrid", "target_sub"])
    p.add_argument("--safety_scale", type=float, default=1.0)

    # Concepts
    p.add_argument("--target_concepts", nargs="+",
                   default=["nudity", "nude person", "naked body"])
    p.add_argument("--anchor_concepts", nargs="+",
                   default=["clothed person", "person wearing clothes"])
    p.add_argument("--target_words", nargs="+", default=None,
                   help="Keywords for text probe (auto-extracted if None)")

    # CPU offload (required for RTX 3090)
    p.add_argument("--cpu_offload", action="store_true", default=True)
    p.add_argument("--no_cpu_offload", dest="cpu_offload", action="store_false")

    p.add_argument("--debug", action="store_true")
    p.add_argument("--save_maps", action="store_true")

    args = p.parse_args()

    if args.img_attn_threshold is None:
        args.img_attn_threshold = args.attn_threshold

    if args.probe_mode in ("image", "both") and not args.clip_embeddings:
        if not (args.family_guidance and args.family_config):
            raise ValueError(
                f"--probe_mode={args.probe_mode} requires --clip_embeddings "
                f"(path to .pt file with target_clip_features)")

    if args.family_guidance and not args.family_config:
        raise ValueError("--family_guidance requires --family_config")

    if args.target_words is None:
        words = []
        for concept in args.target_concepts:
            for w in concept.replace("_", " ").split():
                w_clean = w.strip().lower()
                if len(w_clean) >= 3 and w_clean not in words:
                    words.append(w_clean)
        args.target_words = words

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print(f"SafeGen-SD3: Dual-Probe Safe Generation")
    print(f"{'=' * 60}")
    print(f"  WHEN:  CAS threshold={args.cas_threshold}")
    print(f"  WHERE: probe={args.probe_mode}, blocks={args.probe_blocks}")
    print(f"  HOW:   {args.how_mode}, safety_scale={args.safety_scale}")
    print(f"  Target: {args.target_concepts}")
    print(f"  Anchor: {args.anchor_concepts}")
    print(f"  Words:  {args.target_words}")
    print(f"{'=' * 60}\n")

    # Load prompts
    prompts = load_prompts(args.prompts)
    end = args.end_idx if args.end_idx > 0 else len(prompts)
    prompt_work = list(enumerate(prompts))[args.start_idx:end]

    # Load SD3 pipeline
    print(f"Loading SD3 from {args.model_id} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
        print("  CPU offload enabled")
    else:
        pipe = pipe.to(device)

    transformer = pipe.transformer
    vae = pipe.vae
    scheduler = pipe.scheduler

    # Encode concept embeddings
    print("Encoding concepts ...")
    with torch.no_grad():
        tgt_embeds, tgt_pooled = encode_sd3_prompt(pipe, ", ".join(args.target_concepts), device)
        anc_embeds, anc_pooled = encode_sd3_prompt(pipe, ", ".join(args.anchor_concepts), device)
        unc_embeds, unc_pooled = encode_sd3_prompt(pipe, "", device)

    family_names = []
    family_target_embeds = {}
    family_target_pooled = {}
    family_anchor_embeds = {}
    family_anchor_pooled = {}
    family_target_words = {}

    if args.family_guidance:
        print(f"Loading family config: {args.family_config}")
        family_data = torch.load(args.family_config, map_location="cpu", weights_only=False)
        family_meta = family_data.get("family_metadata", {})
        family_names = family_data.get("family_names")
        if not family_names:
            family_names = list(family_data.get("family_token_map", {}).keys())
        if not family_names:
            family_names = list(family_meta.keys())

        for fname in family_names:
            meta = family_meta.get(fname, {})
            target_terms = (
                meta.get("target_words")
                or meta.get("target_prompts")
                or args.target_concepts
            )
            anchor_terms = (
                meta.get("anchor_words")
                or meta.get("anchor_prompts")
                or args.anchor_concepts
            )
            family_target_words[fname] = target_terms
            with torch.no_grad():
                ft_embeds, ft_pooled = encode_sd3_prompt(pipe, ", ".join(target_terms[:3]), device)
                fa_embeds, fa_pooled = encode_sd3_prompt(pipe, ", ".join(anchor_terms[:3]), device)
            family_target_embeds[fname] = ft_embeds
            family_target_pooled[fname] = ft_pooled
            family_anchor_embeds[fname] = fa_embeds
            family_anchor_pooled[fname] = fa_pooled
        print(f"  Families: {family_names}")

    # Setup probe
    use_probe = args.probe_mode in ("text", "image", "both")
    use_txt = args.probe_mode in ("text", "both")
    use_img = args.probe_mode in ("image", "both")

    txt_probe = None
    original_procs = None
    image_probe_embeds = None
    image_probe_token_indices = None
    family_image_token_map = {}

    if use_probe:
        txt_probe = SD3AttentionProbeStore()
        print(f"  Probe ready (mode={args.probe_mode}; will hook during generation)")

    if use_img:
        img_cfg_path = args.family_config if args.family_guidance else args.clip_embeddings
        print(f"  Loading CLIP exemplar embeddings: {img_cfg_path}")
        clip_data = torch.load(img_cfg_path, map_location="cpu", weights_only=False)

        if args.family_guidance:
            family_feats = clip_data.get("target_clip_features")
            if not isinstance(family_feats, dict) or not family_feats:
                raise ValueError(
                    f"family_config missing dict target_clip_features; "
                    f"keys: {list(clip_data.keys()) if isinstance(clip_data, dict) else type(clip_data)}")
            raw_probe_embeds, image_probe_token_indices, family_image_token_map = (
                build_grouped_sd3_image_probe_embeds(
                    family_feats,
                    baseline_encoder_hidden=unc_embeds.detach(),
                    max_tokens=args.n_img_tokens,
                )
            )
            print(f"    Family image tokens: {family_image_token_map}")
        else:
            clip_feats = clip_data.get(
                "target_clip_features", clip_data.get("target_cls"))
            if clip_feats is None:
                raise ValueError(
                    f"clip_embeddings file missing 'target_clip_features' key; "
                    f"keys: {list(clip_data.keys()) if isinstance(clip_data, dict) else type(clip_data)}")
            clip_feats = clip_feats.float()
            print(f"    CLIP features: {tuple(clip_feats.shape)}")

            # Build pseudo-text embedding using SD3's empty-prompt as baseline
            # (pre-context_embedder, 4096-d).
            raw_probe_embeds, image_probe_token_indices = build_sd3_image_probe_embeds(
                clip_feats,
                baseline_encoder_hidden=unc_embeds.detach(),
                n_tokens=args.n_img_tokens,
            )
        # Project through transformer.context_embedder (4096 -> inner model dim
        # ~1536) so it's in the same space that each block's add_k_proj reads.
        ctx_embedder = transformer.context_embedder
        ctx_dtype = next(ctx_embedder.parameters()).dtype
        ctx_device = next(ctx_embedder.parameters()).device
        with torch.no_grad():
            image_probe_embeds = ctx_embedder(
                raw_probe_embeds.to(device=ctx_device, dtype=ctx_dtype))
        print(f"    Image-probe pseudo-text (raw 4096-d): {tuple(raw_probe_embeds.shape)}")
        print(f"    Image-probe pseudo-text (ctx-embedded): "
              f"{tuple(image_probe_embeds.shape)} tokens={image_probe_token_indices}")

    # Generate
    cas = GlobalCAS(args.cas_threshold)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.save_maps:
        (outdir / "maps").mkdir(exist_ok=True)
    stats = []

    gen_device = "cpu" if args.cpu_offload else device

    for pi, prompt in tqdm(prompt_work, desc="SafeGen-SD3"):
        if not prompt.strip():
            continue
        for si in range(args.nsamples):
            seed = args.seed + pi * args.nsamples + si
            set_seed(seed)
            cas.reset()
            guided_count = 0
            cas_vals, mask_areas = [], []

            # Encode this prompt
            with torch.no_grad():
                prompt_embeds, pooled_proj = encode_sd3_prompt(pipe, prompt, device)

            # Prepare latents
            set_seed(seed)
            generator = torch.Generator(device=gen_device).manual_seed(seed)

            # Use pipeline's call to handle the full generation with our custom loop
            # For now, use the simple approach: generate with guidance via callback
            # Actually, for full control we need a manual denoising loop.

            # Get latent shape from pipeline
            num_channels = transformer.config.in_channels
            latent_h = args.resolution // (vae.config.scaling_factor if hasattr(vae.config, 'scaling_factor') else 8)
            latent_w = latent_h
            # SD3 uses patch_size=2, but latent resolution is resolution/8
            latent_h = args.resolution // 8
            latent_w = args.resolution // 8

            latents = torch.randn(
                1, num_channels, latent_h, latent_w,
                generator=generator, device=gen_device, dtype=torch.float16
            )

            # Setup scheduler
            scheduler.set_timesteps(args.steps, device=gen_device)
            timesteps = scheduler.timesteps

            # CFG: prepare doubled inputs
            for step_i, t in enumerate(timesteps):
                latent_model_input = latents

                # Expand for CFG: [uncond, cond]
                latent_input = torch.cat([latent_model_input] * 2)
                t_input = t.expand(2)

                prompt_embeds_cfg = torch.cat([unc_embeds, prompt_embeds])
                pooled_cfg = torch.cat([unc_pooled, pooled_proj])

                # Forward pass — with probe hooked on first pass (no extra forward!)
                if txt_probe is not None:
                    original_procs = register_sd3_attention_probe(
                        transformer, txt_probe, args.probe_blocks,
                        probe_mode=args.probe_mode,
                        image_probe_embeds=image_probe_embeds,
                        image_probe_token_indices=image_probe_token_indices)
                    txt_probe.active = True
                    txt_probe.reset()

                with torch.no_grad():
                    noise_pred = transformer(
                        hidden_states=latent_input.to(device),
                        timestep=t_input.to(device),
                        encoder_hidden_states=prompt_embeds_cfg.to(device),
                        pooled_projections=pooled_cfg.to(device),
                        return_dict=False,
                    )[0]

                if txt_probe is not None:
                    txt_probe.active = False
                    restore_sd3_processors(transformer, original_procs)

                v_null, v_prompt = noise_pred.chunk(2)

                # Target prediction (for CAS)
                with torch.no_grad():
                    v_target = transformer(
                        hidden_states=latent_model_input.to(device),
                        timestep=t.unsqueeze(0).to(device),
                        encoder_hidden_states=tgt_embeds.to(device),
                        pooled_projections=tgt_pooled.to(device),
                        return_dict=False,
                    )[0]

                # CFG
                v_cfg = v_null + args.cfg_scale * (v_prompt - v_null)

                # WHEN: CAS check
                cv, trig = cas.compute(v_prompt, v_null, v_target)
                cas_vals.append(cv)

                if trig:
                    have_txt = (txt_probe is not None
                                and use_txt and txt_probe.get_maps())
                    have_img = (txt_probe is not None
                                and use_img and txt_probe.get_image_maps())

                    if args.family_guidance and family_names:
                        # ── Per-family target/anchor calls + text probe capture (Patch 6) ──
                        fam_targets = []
                        fam_anchors = []
                        # Stash per-family text probe maps captured via probe reset+activate
                        family_txt_probe_maps = {}  # {fname: probe_maps dict snapshot}

                        with torch.no_grad():
                            for fname in family_names:
                                # Capture text probe for THIS family's target call
                                if txt_probe is not None and use_txt:
                                    original_procs_fam = register_sd3_attention_probe(
                                        transformer, txt_probe, args.probe_blocks,
                                        probe_mode="text",
                                        image_probe_embeds=None,
                                        image_probe_token_indices=None)
                                    txt_probe.reset()
                                    txt_probe.active = True

                                fam_targets.append(
                                    transformer(
                                        hidden_states=latent_model_input.to(device),
                                        timestep=t.unsqueeze(0).to(device),
                                        encoder_hidden_states=family_target_embeds[fname].to(device),
                                        pooled_projections=family_target_pooled[fname].to(device),
                                        return_dict=False,
                                    )[0]
                                )

                                if txt_probe is not None and use_txt:
                                    txt_probe.active = False
                                    restore_sd3_processors(transformer, original_procs_fam)
                                    # Snapshot this family's text probe maps
                                    family_txt_probe_maps[fname] = dict(txt_probe.get_maps())

                                fam_anchors.append(
                                    transformer(
                                        hidden_states=latent_model_input.to(device),
                                        timestep=t.unsqueeze(0).to(device),
                                        encoder_hidden_states=family_anchor_embeds[fname].to(device),
                                        pooled_projections=family_anchor_pooled[fname].to(device),
                                        return_dict=False,
                                    )[0]
                                )

                        # ── Build per-family spatial masks ──
                        fam_masks = []
                        for fname in family_names:
                            fm = torch.zeros(1, 1, latent_h, latent_w, device=v_cfg.device)

                            # Text probe: use family-specific maps captured above
                            if use_txt and fname in family_txt_probe_maps and family_txt_probe_maps[fname]:
                                # Temporarily swap probe maps to the family snapshot
                                saved_maps = txt_probe.probe_maps
                                txt_probe.probe_maps = family_txt_probe_maps[fname]
                                txt_attn = compute_sd3_spatial_mask(
                                    txt_probe, token_indices=None,
                                    latent_h=latent_h, latent_w=latent_w)
                                txt_probe.probe_maps = saved_maps
                                fm = torch.maximum(
                                    fm,
                                    make_probe_mask(
                                        txt_attn, args.attn_threshold,
                                        args.attn_sigmoid_alpha, args.blur_sigma, v_cfg.device,
                                    ),
                                )

                            # Image probe: use family-specific token slot from grouped embeds
                            if have_img and fname in family_image_token_map:
                                img_attn = compute_sd3_image_probe_mask(
                                    txt_probe,
                                    token_indices=[family_image_token_map[fname]],
                                    latent_h=latent_h,
                                    latent_w=latent_w,
                                )
                                fm = torch.maximum(
                                    fm,
                                    make_probe_mask(
                                        img_attn, args.img_attn_threshold,
                                        args.attn_sigmoid_alpha, args.blur_sigma, v_cfg.device,
                                    ),
                                )

                            fam_masks.append(fm)

                        # Winner-take-all across families
                        if len(fam_masks) > 1:
                            stacked = torch.cat(fam_masks, dim=0)  # [N,1,H,W]
                            winner = stacked.argmax(dim=0, keepdim=True)
                            for fi in range(len(fam_masks)):
                                fam_masks[fi] = fam_masks[fi] * (
                                    winner == fi
                                ).float().squeeze(0)

                        v_final = apply_family_guidance(
                            v_cfg, v_null, fam_masks, fam_targets, fam_anchors,
                            args.how_mode, args.safety_scale, args.cfg_scale,
                        )
                        guided_count += 1
                        mask_areas.append(float(sum(fm.mean().item() for fm in fam_masks)))
                    else:
                        # Anchor prediction
                        with torch.no_grad():
                            v_anchor = transformer(
                                hidden_states=latent_model_input.to(device),
                                timestep=t.unsqueeze(0).to(device),
                                encoder_hidden_states=anc_embeds.to(device),
                                pooled_projections=anc_pooled.to(device),
                                return_dict=False,
                            )[0]

                        # WHERE: Compute probe mask (from first forward pass — no extra cost!)
                        txt_mask = None
                        img_mask = None
                        if have_txt:
                            txt_attn = compute_sd3_spatial_mask(
                                txt_probe, token_indices=None,
                                latent_h=latent_h, latent_w=latent_w)
                            txt_mask = make_probe_mask(
                                txt_attn, args.attn_threshold,
                                args.attn_sigmoid_alpha, args.blur_sigma, v_cfg.device)
                        if have_img:
                            img_attn = compute_sd3_image_probe_mask(
                                txt_probe,
                                token_indices=image_probe_token_indices,
                                latent_h=latent_h, latent_w=latent_w)
                            img_mask = make_probe_mask(
                                img_attn, args.img_attn_threshold,
                                args.attn_sigmoid_alpha, args.blur_sigma, v_cfg.device)

                        if txt_mask is not None and img_mask is not None:
                            # Union via max (soft-OR), matches SafeGen SD1.4 fusion.
                            probe_mask = torch.maximum(txt_mask, img_mask).clamp(0, 1)
                        elif txt_mask is not None:
                            probe_mask = txt_mask
                        elif img_mask is not None:
                            probe_mask = img_mask
                        else:
                            # No probe — global guidance
                            probe_mask = torch.ones(
                                1, 1, latent_h, latent_w, device=v_cfg.device) * 0.5

                        # HOW: Apply guidance
                        v_final = apply_guidance(
                            v_cfg, v_null, v_prompt, v_target, v_anchor,
                            probe_mask, args.how_mode, args.safety_scale, args.cfg_scale)

                        guided_count += 1
                        mask_areas.append(float(probe_mask.mean()))

                    if args.debug and step_i % 5 == 0:
                        print(f"  [{step_i:02d}] CAS={cv:.3f} mask={mask_areas[-1]:.3f}")
                else:
                    v_final = v_cfg

                # Scheduler step — ensure all tensors on same device
                step_device = v_final.device
                latents = scheduler.step(
                    v_final, t.to(step_device), latents.to(step_device),
                    return_dict=False)[0]

                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    latents = scheduler.step(
                        v_cfg.to(step_device), t.to(step_device), latents.to(step_device),
                        return_dict=False)[0]

            # Decode
            with torch.no_grad():
                latents_scaled = (latents / vae.config.scaling_factor) + vae.config.shift_factor
                image = vae.decode(latents_scaled.to(vae.dtype), return_dict=False)[0]
                image = (image / 2 + 0.5).clamp(0, 1)
                img_np = (image[0].cpu().permute(1, 2, 0).float().numpy() * 255).round().astype(np.uint8)

            Image.fromarray(img_np).save(str(outdir / f"{pi:04d}_{si:02d}.png"))
            stats.append({
                "prompt_idx": pi, "sample_idx": si, "seed": seed,
                "guided_steps": guided_count,
                "max_cas": max(cas_vals) if cas_vals else 0,
                "mean_mask_area": float(np.mean(mask_areas)) if mask_areas else 0,
            })

    # Save metadata
    json.dump(stats, open(outdir / "generation_stats.json", "w"), indent=2)
    json.dump(vars(args), open(outdir / "args.json", "w"), indent=2)

    guided = sum(1 for s in stats if s["guided_steps"] > 0)
    print(f"\nDone! {len(stats)} images, guided {guided}/{len(stats)}")


if __name__ == "__main__":
    main()
