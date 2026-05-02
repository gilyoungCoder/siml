#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SafeGen-Flux v1: Family-Grouped Safe Generation on FLUX.2-klein-4B.

FLUX.2-klein uses traditional CFG (like SD v1.4), making SafeGen adaptation clean:
  - Backbone: Flux2KleinPipeline (DiT transformer, Mistral text encoder)
  - CFG: external neg + scale*(cond - neg) — same as SD v1.4
  - Latent: [B, C, H, W] → packed [B, H*W, C] (simple reshape)
  - No pooled_projections, no guidance embedding

Method (SafeGen When-Where-How):
  WHEN: Global CAS on noise predictions — cos(ep-en, et-en) > threshold
  WHERE: Global mask (v1 — spatial possible via token-to-grid unpacking)
  HOW: anchor_inpaint, hybrid, target_sub — per-family or single-anchor

Usage:
    # Baseline (no safety)
    python generate_flux_v1.py --prompts prompts/ringabell.txt \\
        --outdir outputs/flux2klein/baseline --no_safety

    # Single-anchor
    python generate_flux_v1.py --prompts prompts/ringabell.txt \\
        --outdir outputs/flux2klein/ours_ainp \\
        --how_mode anchor_inpaint --safety_scale 1.0

    # Family-grouped
    python generate_flux_v1.py --prompts prompts/ringabell.txt \\
        --outdir outputs/flux2klein/ours_family \\
        --family_config exemplars/concepts_v2/sexual/clip_grouped.pt \\
        --family_guidance --how_mode anchor_inpaint
"""

import os, sys, json, math, random, csv, gc
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm

from attention_probe_flux import (
    FluxSpatialProbe,
    compute_flux_spatial_mask,
    mask_to_packed_seq,
)


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
                   mask, how, safety_scale, cfg_scale):
    """Apply safety guidance. mask is scalar or tensor."""
    m = mask
    s = safety_scale

    if how == "anchor_inpaint":
        ea_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
        blend = min(s * m, 1.0) if isinstance(m, (int, float)) else (s * m).clamp(max=1.0)
        out = eps_cfg * (1 - blend) + ea_cfg * blend

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
                          how, safety_scale, cfg_scale):
    """Per-family guidance. Each family contributes 1/N globally."""
    out = eps_cfg.clone()
    s = safety_scale
    n = len(family_targets)
    if n == 0:
        return out
    w = 1.0 / n

    for et_fi, ea_fi in zip(family_targets, family_anchors):
        if how == "anchor_inpaint":
            ea_cfg = eps_null + cfg_scale * (ea_fi - eps_null)
            blend = min(s * w, 1.0)
            out = out * (1 - blend) + ea_cfg * blend
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


def encode_prompt_flux2klein(pipe, text, device, max_seq_len=512):
    """Encode a single prompt. Returns (prompt_embeds, text_ids)."""
    return pipe.encode_prompt(
        prompt=text, device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_seq_len,
    )


def encode_concepts(pipe, concepts, device, max_seq_len=512):
    """Encode multiple concepts, average their embeddings."""
    all_pe = []
    for c in concepts:
        pe, _ = encode_prompt_flux2klein(pipe, c, device, max_seq_len)
        all_pe.append(pe)
    return torch.stack(all_pe).mean(0)


# ── Args ──
def parse_args():
    p = ArgumentParser(description="SafeGen-Flux v1: FLUX.2-klein-4B")
    p.add_argument("--ckpt", default="black-forest-labs/FLUX.2-klein-4B")
    p.add_argument("--prompts", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--nsamples", type=int, default=1)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg_scale", type=float, default=4.0,
                   help="CFG scale (traditional, like SD v1.4)")
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

    # WHERE: spatial probe
    p.add_argument("--probe_mode", default="none",
                   choices=["none", "text", "contrast"],
                   help="text: cos-sim vs pooled text stream (prompt pass). "
                        "contrast: prompt-vs-target patch contrast.")
    p.add_argument("--probe_block_idx", type=int, default=-1,
                   help="Which transformer block to hook (negative = from end).")
    p.add_argument("--attn_threshold", type=float, default=0.1,
                   help="Floor value for normalised spatial mask.")

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
    print(f"SafeGen-Flux v1: {mode_str}")
    print(f"{'='*70}")
    print(f"  Backbone: {args.ckpt}")
    print(f"  CFG: traditional, scale={args.cfg_scale}")
    if not args.no_safety:
        print(f"  WHEN: CAS threshold={args.cas_threshold}")
        print(f"  HOW:  {args.how_mode}, ss={args.safety_scale}")
        if args.family_guidance:
            print(f"  FAMILY: {args.family_config}")
    print(f"  Resolution: {args.height}x{args.width}, steps={args.steps}")
    print(f"{'='*70}\n")

    # ── Load pipeline ──
    print("Loading FLUX.2-klein-4B pipeline...")
    from diffusers import Flux2KleinPipeline
    pipe = Flux2KleinPipeline.from_pretrained(args.ckpt, torch_dtype=dtype)
    pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    print("Pipeline loaded.\n")

    transformer = pipe.transformer
    vae = pipe.vae
    scheduler = pipe.scheduler

    # ── Encode concepts ──
    family_names = []
    family_target_emb, family_anchor_emb = {}, {}

    with torch.no_grad():
        # Null (empty prompt)
        pe_null, text_ids_null = encode_prompt_flux2klein(
            pipe, "", device, args.max_sequence_length)

        if not args.no_safety:
            # Global target/anchor
            pe_target = encode_concepts(pipe, args.target_concepts, device, args.max_sequence_length)
            pe_anchor = encode_concepts(pipe, args.anchor_concepts, device, args.max_sequence_length)

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
                    family_target_emb[fname] = encode_concepts(
                        pipe, tw, device, args.max_sequence_length)
                    family_anchor_emb[fname] = encode_concepts(
                        pipe, aw, device, args.max_sequence_length)

                print(f"  Families: {family_names}")

    print(f"  Null embed: {pe_null.shape}")
    print(f"  Text IDs: {text_ids_null.shape}\n")

    # ── Load prompts ──
    prompts = load_prompts(args.prompts)
    end = args.end_idx if args.end_idx > 0 else len(prompts)
    pw = list(enumerate(prompts))[args.start_idx:end]
    print(f"Processing {len(pw)} prompts\n")

    # ── Latent dimensions ──
    num_ch = transformer.config.in_channels
    vae_scale_factor = getattr(pipe, 'vae_scale_factor', 8)
    lat_h = 2 * (args.height // (vae_scale_factor * 2))
    lat_w = 2 * (args.width // (vae_scale_factor * 2))
    seq_img_len = (lat_h // 2) * (lat_w // 2)
    print(f"  Latent: {lat_h}x{lat_w} → packed seq_len={seq_img_len}\n")

    # ── WHERE: register spatial probe ──
    probe = None
    if args.probe_mode != "none" and not args.no_safety:
        # Find a blocks container — prefer single-stream (output is full
        # joint sequence [txt; img]).  Dual-stream blocks return a tuple
        # which our forward_hook cannot easily disambiguate.
        blocks = None
        for attr in ("single_transformer_blocks", "transformer_blocks", "blocks"):
            if hasattr(transformer, attr):
                cand = getattr(transformer, attr)
                if cand is not None and len(cand) > 0:
                    blocks = cand
                    break
        if blocks is None:
            print("  [probe] WARN: no transformer blocks found; disabling probe")
        else:
            idx = args.probe_block_idx if args.probe_block_idx >= 0 \
                else len(blocks) + args.probe_block_idx
            idx = max(0, min(len(blocks) - 1, idx))
            probe = FluxSpatialProbe(blocks[idx], seq_img_len=seq_img_len)
            print(f"  [probe] mode={args.probe_mode} hooked block {idx}/{len(blocks)-1}")

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
                pe_prompt, text_ids_prompt = encode_prompt_flux2klein(
                    pipe, prompt, device, args.max_sequence_length)

            # Prepare latents: [B, C*4, H/2, W/2] → pack → [B, H*W/4, C*4]
            set_seed(seed)
            latents = torch.randn(1, num_ch, lat_h // 2, lat_w // 2,
                                  device=device, dtype=dtype)
            latent_ids = pipe._prepare_latent_ids(latents).to(device)
            latents = pipe._pack_latents(latents)  # [B, seq, C]

            # Prepare timesteps
            from diffusers.pipelines.flux2.pipeline_flux2_klein import retrieve_timesteps
            sigmas = np.linspace(1.0, 1 / args.steps, args.steps)
            mu = calculate_shift_klein(
                latents.shape[1], scheduler, args.steps)
            timesteps, _ = retrieve_timesteps(
                scheduler, args.steps, device, sigmas=sigmas, mu=mu)

            # ── Denoising loop ──
            for step_idx, t in enumerate(timesteps):
                ts = t.expand(latents.shape[0]).to(latents.dtype)
                lat_in = latents.to(transformer.dtype)

                if args.no_safety:
                    # ── BASELINE: standard CFG (2 passes) ──
                    with torch.no_grad():
                        en = transformer(
                            hidden_states=lat_in, timestep=ts / 1000,
                            guidance=None,
                            encoder_hidden_states=pe_null.to(dtype),
                            txt_ids=text_ids_null, img_ids=latent_ids,
                            return_dict=False,
                        )[0][:, :latents.shape[1]]

                        ep = transformer(
                            hidden_states=lat_in, timestep=ts / 1000,
                            guidance=None,
                            encoder_hidden_states=pe_prompt.to(dtype),
                            txt_ids=text_ids_prompt, img_ids=latent_ids,
                            return_dict=False,
                        )[0][:, :latents.shape[1]]

                    eps_final = en + args.cfg_scale * (ep - en)

                else:
                    # ── SafeGen: multi-pass ──
                    if probe is not None:
                        probe.reset()
                    with torch.no_grad():
                        en = transformer(
                            hidden_states=lat_in, timestep=ts / 1000,
                            guidance=None,
                            encoder_hidden_states=pe_null.to(dtype),
                            txt_ids=text_ids_null, img_ids=latent_ids,
                            return_dict=False,
                        )[0][:, :latents.shape[1]]

                        if probe is not None:
                            probe.active = True
                            probe.tag = "prompt"
                        ep = transformer(
                            hidden_states=lat_in, timestep=ts / 1000,
                            guidance=None,
                            encoder_hidden_states=pe_prompt.to(dtype),
                            txt_ids=text_ids_prompt, img_ids=latent_ids,
                            return_dict=False,
                        )[0][:, :latents.shape[1]]
                        if probe is not None:
                            probe.active = False

                        if probe is not None and args.probe_mode == "contrast":
                            probe.active = True
                            probe.tag = "target"
                        et = transformer(
                            hidden_states=lat_in, timestep=ts / 1000,
                            guidance=None,
                            encoder_hidden_states=pe_target.to(dtype),
                            txt_ids=text_ids_null, img_ids=latent_ids,
                            return_dict=False,
                        )[0][:, :latents.shape[1]]
                        if probe is not None:
                            probe.active = False

                    ec = en + args.cfg_scale * (ep - en)
                    cv, trig = cas.compute(ep, en, et)
                    cas_values.append(cv)
                    eps_final = ec

                    # Compute spatial mask from probe captures (if enabled)
                    spatial_mask_packed = None
                    if probe is not None and trig and "prompt" in probe.captures:
                        feat_p = probe.captures["prompt"]  # [B, seq_img, C]
                        if args.probe_mode == "contrast" and "target" in probe.captures:
                            mask2d = compute_flux_spatial_mask(
                                feat_p, target_feat=probe.captures["target"],
                                threshold=args.attn_threshold, mode="contrast")
                        else:
                            # text mode: pooled text-stream as target vector.
                            # pe_target is [B, seq_txt, C_text] from text encoder.
                            # Project via avg-pool then broadcast; fall back if
                            # C_text != C_feat (use self-energy via None vec).
                            tgt_vec = pe_target.to(feat_p.device, feat_p.dtype).mean(dim=1)
                            if tgt_vec.shape[-1] != feat_p.shape[-1]:
                                tgt_vec = None
                            mask2d = compute_flux_spatial_mask(
                                feat_p, target_vec=tgt_vec,
                                threshold=args.attn_threshold, mode="text")
                        # Pack to sequence [B, seq_img, 1] for broadcast over eps
                        spatial_mask_packed = mask_to_packed_seq(
                            mask2d, seq_img_len=feat_p.shape[1]
                        ).to(en.device, en.dtype)
                        if args.debug and step_idx % 10 == 0:
                            print(f"  [probe] step={step_idx} mask "
                                  f"min={mask2d.min().item():.3f} "
                                  f"max={mask2d.max().item():.3f} "
                                  f"mean={mask2d.mean().item():.3f}")

                    if trig:
                        if args.family_guidance and family_names:
                            fam_ts, fam_as = [], []
                            with torch.no_grad():
                                for fname in family_names:
                                    ft = transformer(
                                        hidden_states=lat_in, timestep=ts / 1000,
                                        guidance=None,
                                        encoder_hidden_states=family_target_emb[fname].to(dtype),
                                        txt_ids=text_ids_null, img_ids=latent_ids,
                                        return_dict=False,
                                    )[0][:, :latents.shape[1]]
                                    fa = transformer(
                                        hidden_states=lat_in, timestep=ts / 1000,
                                        guidance=None,
                                        encoder_hidden_states=family_anchor_emb[fname].to(dtype),
                                        txt_ids=text_ids_null, img_ids=latent_ids,
                                        return_dict=False,
                                    )[0][:, :latents.shape[1]]
                                    fam_ts.append(ft)
                                    fam_as.append(fa)

                            eps_final = apply_family_guidance(
                                ec, en, fam_ts, fam_as,
                                args.how_mode, args.safety_scale, args.cfg_scale)
                        else:
                            with torch.no_grad():
                                ea = transformer(
                                    hidden_states=lat_in, timestep=ts / 1000,
                                    guidance=None,
                                    encoder_hidden_states=pe_anchor.to(dtype),
                                    txt_ids=text_ids_null, img_ids=latent_ids,
                                    return_dict=False,
                                )[0][:, :latents.shape[1]]

                            m_use = spatial_mask_packed if spatial_mask_packed is not None else 1.0
                            eps_final = apply_guidance(
                                ec, en, et, ea,
                                mask=m_use, how=args.how_mode,
                                safety_scale=args.safety_scale,
                                cfg_scale=args.cfg_scale)

                        guided_count += 1
                        if args.debug and step_idx % 5 == 0:
                            print(f"  [{step_idx:02d}] CAS={cv:.3f} TRIGGERED")

                # Scheduler step
                latents_prev = latents.clone()
                latents = scheduler.step(eps_final, t, latents, return_dict=False)[0]
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    fallback = en + args.cfg_scale * (ep - en) if not args.no_safety else eps_final
                    latents = scheduler.step(fallback, t, latents_prev, return_dict=False)[0]

            # ── Decode ──
            with torch.no_grad():
                lat_out = pipe._unpack_latents_with_ids(latents, latent_ids)
                # Batch norm denormalization
                bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(lat_out.device, lat_out.dtype)
                bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
                    lat_out.device, lat_out.dtype)
                lat_out = lat_out * bn_std + bn_mean
                lat_out = pipe._unpatchify_latents(lat_out)
                image = vae.decode(lat_out.to(vae.dtype), return_dict=False)[0]
                image = (image / 2 + 0.5).clamp(0, 1)
                img_np = (image[0].cpu().permute(1, 2, 0).float().numpy() * 255).round().astype(np.uint8)

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


def calculate_shift_klein(seq_len, scheduler, num_steps):
    """Calculate mu for timestep shifting."""
    mu = None
    if hasattr(scheduler.config, 'base_image_seq_len'):
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift
        mu = calculate_shift(
            seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
    return mu


if __name__ == "__main__":
    main()
