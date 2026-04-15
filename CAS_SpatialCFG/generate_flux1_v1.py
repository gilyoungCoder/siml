#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SafeGen-Flux1 v1: Family-Grouped Safe Generation on FLUX.1-dev (12B).

FLUX.1-dev uses EMBEDDED guidance (not traditional CFG):
  - Backbone: FluxPipeline (DiT transformer, dual text encoders)
  - Guidance: guidance_scale embedded as tensor to transformer — no neg pass needed
  - encode_prompt returns (prompt_embeds, pooled_prompt_embeds, text_ids) — 3 values
  - Latent packing: [B,C,H,W] -> [B, (H/2)*(W/2), C*4]
  - VAE decode: _unpack_latents(latents, h, w, vae_scale_factor)

Since guidance is embedded, the transformer output IS the guided prediction.
SafeGen guidance formulas are applied on top of ep (already guided):
  - eps_cfg = ep  (the single-pass guidance-embedded prediction)
  - anchor_inpaint: blend ep and ea (anchor pred) directly
  - hybrid / target_sub: operate on ep, en, et, ea as normal

Method (SafeGen When-Where-How):
  WHEN: Global CAS on noise predictions — cos(ep-en, et-en) > threshold
  WHERE: Global mask v1 (spatial possible via token-to-grid unpacking)
  HOW: anchor_inpaint, hybrid, target_sub — per-family or single-anchor

Usage:
    # Baseline (no safety)
    python generate_flux1_v1.py --prompts prompts/ringabell.txt \\
        --outdir outputs/flux1dev/baseline --no_safety

    # Single-anchor
    python generate_flux1_v1.py --prompts prompts/ringabell.txt \\
        --outdir outputs/flux1dev/ours_ainp \\
        --how_mode anchor_inpaint --safety_scale 1.0

    # Family-grouped
    python generate_flux1_v1.py --prompts prompts/ringabell.txt \\
        --outdir outputs/flux1dev/ours_family \\
        --family_config exemplars/concepts_v2/sexual/clip_grouped.pt \\
        --family_guidance --how_mode anchor_inpaint
"""

import os, sys, json, math, random, csv, gc
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm


# ── WHEN: Global CAS ──
class GlobalCAS:
    """Model-agnostic CAS on flattened noise predictions."""
    def __init__(self, threshold=0.6, sticky=True):
        self.threshold, self.sticky, self.triggered = threshold, sticky, False

    def reset(self):
        self.triggered = False

    def compute(self, ep, en, et):
        dp = (ep - en).reshape(1, -1).float()
        dt = (et - en).reshape(1, -1).float()
        c = F.cosine_similarity(dp, dt, dim=-1).item()
        if math.isnan(c) or math.isinf(c):
            return 0., self.triggered if self.sticky else False
        if self.sticky and self.triggered:
            return c, True
        if c > self.threshold:
            if self.sticky:
                self.triggered = True
            return c, True
        return c, False


# ── HOW: Guidance ──
def apply_guidance(eps_cfg, eps_null, eps_target, eps_anchor,
                   mask, how, safety_scale):
    """Apply safety guidance for FLUX.1-dev.

    FLUX.1-dev uses embedded guidance — no cfg_scale in formulas.
    eps_cfg = ep (already the guidance-embedded prediction).
    """
    m = mask
    s = safety_scale

    if how == "anchor_inpaint":
        # Blend between prompt prediction and anchor prediction
        blend = min(s * m, 1.0) if isinstance(m, (int, float)) else (s * m).clamp(max=1.0)
        out = eps_cfg * (1 - blend) + eps_anchor * blend

    elif how == "hybrid":
        out = (eps_cfg
               - s * m * (eps_target - eps_null)
               + s * m * (eps_anchor - eps_null))

    elif how == "target_sub":
        out = eps_cfg - s * m * (eps_target - eps_null)

    else:
        raise ValueError(f"Unknown HOW mode: {how}")

    if torch.isnan(out).any() or torch.isinf(out).any():
        out = torch.where(torch.isfinite(out), out, eps_cfg)
    return out


def apply_family_guidance(eps_cfg, eps_null, family_targets, family_anchors,
                          how, safety_scale):
    """Per-family guidance for FLUX.1-dev. Each family contributes 1/N globally."""
    out = eps_cfg.clone()
    s = safety_scale
    n = len(family_targets)
    if n == 0:
        return out
    w = 1.0 / n

    for et_fi, ea_fi in zip(family_targets, family_anchors):
        if how == "anchor_inpaint":
            blend = min(s * w, 1.0)
            out = out * (1 - blend) + ea_fi * blend
        elif how == "hybrid":
            out = (out
                   - s * w * (et_fi - eps_null)
                   + s * w * (ea_fi - eps_null))
        elif how == "target_sub":
            out = out - s * w * (et_fi - eps_null)

    if torch.isnan(out).any() or torch.isinf(out).any():
        out = torch.where(torch.isfinite(out), out, eps_cfg)
    return out


# ── Utils ──
def load_prompts(fp):
    fp = Path(fp)
    if fp.suffix == ".csv":
        ps = []
        with open(fp) as f:
            r = csv.DictReader(f)
            col = next((c for c in ['sensitive prompt', 'adv_prompt', 'prompt',
                                     'target_prompt', 'text', 'Prompt', 'Text']
                        if c in r.fieldnames), None)
            if not col:
                raise ValueError(f"No prompt col in {r.fieldnames}")
            for row in r:
                p = row[col].strip()
                if p:
                    ps.append(p)
        return ps
    return [l.strip() for l in open(fp) if l.strip()]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def encode_prompt_flux1(pipe, text, device, max_seq_len=512):
    """Encode a single prompt for FLUX.1-dev.

    Returns (prompt_embeds, pooled_prompt_embeds, text_ids) — 3 values.
    """
    return pipe.encode_prompt(
        prompt=text,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_seq_len,
    )


def encode_concepts(pipe, concepts, device, max_seq_len=512):
    """Encode multiple concepts, average their sequence embeddings.

    Returns averaged (prompt_embeds, pooled_prompt_embeds, text_ids).
    text_ids are taken from the first concept (shape-only, content same).
    """
    all_pe, all_pooled = [], []
    text_ids = None
    for c in concepts:
        pe, pooled, tids = encode_prompt_flux1(pipe, c, device, max_seq_len)
        all_pe.append(pe)
        all_pooled.append(pooled)
        if text_ids is None:
            text_ids = tids
    return (
        torch.stack(all_pe).mean(0),
        torch.stack(all_pooled).mean(0),
        text_ids,
    )


# ── Args ──
def parse_args():
    p = ArgumentParser(description="SafeGen-Flux1 v1: FLUX.1-dev 12B")
    p.add_argument("--ckpt", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--prompts", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--nsamples", type=int, default=1)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=3.5,
                   help="Embedded guidance scale passed as tensor to transformer")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)

    # Safety mode
    p.add_argument("--no_safety", action="store_true",
                   help="Baseline: no safety guidance")

    # WHEN
    p.add_argument("--cas_threshold", type=float, default=0.6)

    # HOW
    p.add_argument("--how_mode", default="anchor_inpaint",
                   choices=["anchor_inpaint", "hybrid", "target_sub"])
    p.add_argument("--safety_scale", type=float, default=1.0)

    # Family
    p.add_argument("--family_config", default=None,
                   help="Path to clip_grouped.pt")
    p.add_argument("--family_guidance", action="store_true")

    # Concepts
    p.add_argument("--target_concepts", nargs="+",
                   default=["nudity", "nude person", "naked body"])
    p.add_argument("--anchor_concepts", nargs="+",
                   default=["clothed person", "person wearing clothes"])

    # Device
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat16"])
    p.add_argument("--debug", action="store_true")

    return p.parse_args()


# ── Main ──
def main():
    args = parse_args()
    set_seed(args.seed)
    gpu_id = int(args.device.split(":")[-1])
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    mode_str = "BASELINE (no safety)" if args.no_safety else \
               f"Family-Grouped ({args.how_mode})" if args.family_guidance else \
               f"Single-Anchor ({args.how_mode})"

    print(f"\n{'='*70}")
    print(f"SafeGen-Flux1 v1: {mode_str}")
    print(f"{'='*70}")
    print(f"  Backbone: {args.ckpt}")
    print(f"  Guidance: EMBEDDED (scale={args.guidance_scale}, no CFG neg pass)")
    if not args.no_safety:
        print(f"  WHEN: CAS threshold={args.cas_threshold}")
        print(f"  HOW:  {args.how_mode}, ss={args.safety_scale}")
        if args.family_guidance:
            print(f"  FAMILY: {args.family_config}")
    print(f"  Resolution: {args.height}x{args.width}, steps={args.steps}")
    print(f"{'='*70}\n")

    # ── Load pipeline ──
    print("Loading FLUX.1-dev pipeline...")
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(args.ckpt, torch_dtype=dtype)
    pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    print("Pipeline loaded.\n")

    transformer = pipe.transformer
    vae = pipe.vae
    scheduler = pipe.scheduler

    # ── Guidance embedding tensor (embedded guidance, not traditional CFG) ──
    if transformer.config.guidance_embeds:
        guidance_tensor = torch.full(
            [1], args.guidance_scale, device=device, dtype=torch.float32)
    else:
        guidance_tensor = None

    # ── Encode concepts ──
    family_names = []
    family_target_emb, family_anchor_emb = {}, {}
    family_target_pooled, family_anchor_pooled = {}, {}

    with torch.no_grad():
        # Null (empty prompt)
        pe_null, pooled_null, text_ids_null = encode_prompt_flux1(
            pipe, "", device, args.max_sequence_length)

        if not args.no_safety:
            # Global target/anchor
            pe_target, pooled_target, text_ids_target = encode_concepts(
                pipe, args.target_concepts, device, args.max_sequence_length)
            pe_anchor, pooled_anchor, text_ids_anchor = encode_concepts(
                pipe, args.anchor_concepts, device, args.max_sequence_length)

            # Family-specific
            if args.family_guidance and args.family_config:
                print(f"Loading family config: {args.family_config}")
                fdata = torch.load(args.family_config, map_location="cpu", weights_only=False)
                family_names = fdata.get("family_names", [])
                family_meta = fdata.get("family_metadata", {})

                for fname in family_names:
                    meta = family_meta.get(fname, {})
                    tw = meta.get("target_prompts", args.target_concepts)[:3]
                    aw = meta.get("anchor_prompts", args.anchor_concepts)[:3]
                    ft_pe, ft_pooled, ft_ids = encode_concepts(
                        pipe, tw, device, args.max_sequence_length)
                    fa_pe, fa_pooled, fa_ids = encode_concepts(
                        pipe, aw, device, args.max_sequence_length)
                    family_target_emb[fname] = ft_pe
                    family_target_pooled[fname] = ft_pooled
                    family_anchor_emb[fname] = fa_pe
                    family_anchor_pooled[fname] = fa_pooled

                print(f"  Families: {family_names}")

    print(f"  Null embed: {pe_null.shape}, pooled: {pooled_null.shape}")
    print(f"  Text IDs: {text_ids_null.shape}\n")

    # ── Load prompts ──
    prompts = load_prompts(args.prompts)
    end = args.end_idx if args.end_idx > 0 else len(prompts)
    pw = list(enumerate(prompts))[args.start_idx:end]
    print(f"Processing {len(pw)} prompts\n")

    # ── Latent dimensions ──
    # FLUX.1-dev: in_channels=64 (packed), so num_channels = 64//4 = 16
    num_channels_latents = transformer.config.in_channels // 4
    vae_scale_factor = getattr(pipe, 'vae_scale_factor', 8)
    # lat_h/lat_w: spatial dims before packing
    lat_h = 2 * (args.height // (vae_scale_factor * 2))
    lat_w = 2 * (args.width // (vae_scale_factor * 2))
    # packed seq_len = (lat_h/2) * (lat_w/2) for 2x2 patch packing
    packed_seq_len = (lat_h // 2) * (lat_w // 2)
    print(f"  Latent: [{num_channels_latents}, {lat_h}, {lat_w}] "
          f"-> packed seq_len={packed_seq_len}\n")

    cas = GlobalCAS(args.cas_threshold) if not args.no_safety else None
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stats = []

    for pi, prompt in tqdm(pw, desc="Generating"):
        if not prompt.strip():
            continue

        for si in range(args.nsamples):
            seed = args.seed + pi * args.nsamples + si
            set_seed(seed)
            if cas:
                cas.reset()
            guided_count, cas_values = 0, []

            # Encode prompt
            with torch.no_grad():
                pe_prompt, pooled_prompt, text_ids_prompt = encode_prompt_flux1(
                    pipe, prompt, device, args.max_sequence_length)

            # ── Prepare latents ──
            set_seed(seed)
            latents = torch.randn(
                1, num_channels_latents, lat_h, lat_w,
                device=device, dtype=dtype)
            latents = pipe._pack_latents(latents, 1, num_channels_latents, lat_h, lat_w)
            # latent_image_ids: [seq_len, 3]
            latent_image_ids = pipe._prepare_latent_image_ids(
                1, lat_h // 2, lat_w // 2, device, dtype)

            # ── Prepare timesteps ──
            from diffusers.pipelines.flux.pipeline_flux import (
                calculate_shift, retrieve_timesteps)
            sigmas = np.linspace(1.0, 1 / args.steps, args.steps)
            mu = calculate_shift(
                latents.shape[1],
                scheduler.config.get("base_image_seq_len", 256),
                scheduler.config.get("max_image_seq_len", 4096),
                scheduler.config.get("base_shift", 0.5),
                scheduler.config.get("max_shift", 1.15),
            )
            timesteps, _ = retrieve_timesteps(
                scheduler, args.steps, device, sigmas=sigmas, mu=mu)

            # ── Denoising loop ──
            for step_idx, t in enumerate(timesteps):
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                lat_in = latents.to(transformer.dtype)

                if args.no_safety:
                    # ── BASELINE: single pass (guidance embedded) ──
                    with torch.no_grad():
                        ep = transformer(
                            hidden_states=lat_in,
                            timestep=timestep / 1000,
                            guidance=guidance_tensor,
                            pooled_projections=pooled_prompt.to(dtype),
                            encoder_hidden_states=pe_prompt.to(dtype),
                            txt_ids=text_ids_prompt,
                            img_ids=latent_image_ids,
                            return_dict=False,
                        )[0]

                    eps_final = ep

                else:
                    # ── SafeGen: multi-pass ──
                    # ep: prompt prediction (guidance embedded — IS the guided pred)
                    # en: null prediction (no guidance embedding for concept passes)
                    # et: target concept prediction
                    with torch.no_grad():
                        ep = transformer(
                            hidden_states=lat_in,
                            timestep=timestep / 1000,
                            guidance=guidance_tensor,
                            pooled_projections=pooled_prompt.to(dtype),
                            encoder_hidden_states=pe_prompt.to(dtype),
                            txt_ids=text_ids_prompt,
                            img_ids=latent_image_ids,
                            return_dict=False,
                        )[0]

                        # Null pass — no guidance embedding for direction computation
                        en = transformer(
                            hidden_states=lat_in,
                            timestep=timestep / 1000,
                            guidance=None,
                            pooled_projections=pooled_null.to(dtype),
                            encoder_hidden_states=pe_null.to(dtype),
                            txt_ids=text_ids_null,
                            img_ids=latent_image_ids,
                            return_dict=False,
                        )[0]

                        et = transformer(
                            hidden_states=lat_in,
                            timestep=timestep / 1000,
                            guidance=None,
                            pooled_projections=pooled_target.to(dtype),
                            encoder_hidden_states=pe_target.to(dtype),
                            txt_ids=text_ids_null,
                            img_ids=latent_image_ids,
                            return_dict=False,
                        )[0]

                    cv, trig = cas.compute(ep, en, et)
                    cas_values.append(cv)
                    eps_final = ep

                    if trig:
                        if args.family_guidance and family_names:
                            fam_ts, fam_as = [], []
                            with torch.no_grad():
                                for fname in family_names:
                                    ft = transformer(
                                        hidden_states=lat_in,
                                        timestep=timestep / 1000,
                                        guidance=None,
                                        pooled_projections=family_target_pooled[fname].to(dtype),
                                        encoder_hidden_states=family_target_emb[fname].to(dtype),
                                        txt_ids=text_ids_null,
                                        img_ids=latent_image_ids,
                                        return_dict=False,
                                    )[0]
                                    fa = transformer(
                                        hidden_states=lat_in,
                                        timestep=timestep / 1000,
                                        guidance=None,
                                        pooled_projections=family_anchor_pooled[fname].to(dtype),
                                        encoder_hidden_states=family_anchor_emb[fname].to(dtype),
                                        txt_ids=text_ids_null,
                                        img_ids=latent_image_ids,
                                        return_dict=False,
                                    )[0]
                                    fam_ts.append(ft)
                                    fam_as.append(fa)

                            eps_final = apply_family_guidance(
                                ep, en, fam_ts, fam_as,
                                args.how_mode, args.safety_scale)
                        else:
                            with torch.no_grad():
                                ea = transformer(
                                    hidden_states=lat_in,
                                    timestep=timestep / 1000,
                                    guidance=None,
                                    pooled_projections=pooled_anchor.to(dtype),
                                    encoder_hidden_states=pe_anchor.to(dtype),
                                    txt_ids=text_ids_null,
                                    img_ids=latent_image_ids,
                                    return_dict=False,
                                )[0]

                            eps_final = apply_guidance(
                                ep, en, et, ea,
                                mask=1.0, how=args.how_mode,
                                safety_scale=args.safety_scale)

                        guided_count += 1
                        if args.debug and step_idx % 5 == 0:
                            print(f"  [{step_idx:02d}] CAS={cv:.3f} TRIGGERED")

                # Scheduler step
                latents_prev = latents.clone()
                latents = scheduler.step(eps_final, t, latents, return_dict=False)[0]
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    fallback = ep
                    latents = scheduler.step(fallback, t, latents_prev, return_dict=False)[0]

            # ── Decode ──
            with torch.no_grad():
                latents_unpack = pipe._unpack_latents(
                    latents, args.height, args.width, vae_scale_factor)
                latents_unpack = (
                    latents_unpack / vae.config.scaling_factor
                ) + vae.config.shift_factor
                image = vae.decode(
                    latents_unpack.to(vae.dtype), return_dict=False)[0]
                image = (image / 2 + 0.5).clamp(0, 1)
                img_np = (
                    image[0].cpu().permute(1, 2, 0).float().numpy() * 255
                ).round().astype(np.uint8)

            fn = f"{pi:04d}_{si:02d}.png"
            Image.fromarray(img_np).save(str(outdir / fn))

            stats.append({
                "pi": pi, "si": si, "seed": seed,
                "guided": guided_count,
                "max_cas": max(cas_values) if cas_values else 0,
                "prompt": prompt[:100],
            })

    json.dump(stats, open(outdir / "generation_stats.json", "w"), indent=2)
    json.dump(vars(args), open(outdir / "args.json", "w"), indent=2)

    gi = sum(1 for s in stats if s["guided"] > 0)
    print(f"\nDone! {len(stats)} images, guided {gi}/{len(stats)}")


if __name__ == "__main__":
    main()
