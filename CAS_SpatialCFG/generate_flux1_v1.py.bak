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

from attention_probe_flux import (
    FluxSpatialProbe,
    compute_flux_spatial_mask,
    mask_to_packed_seq,
    build_grouped_flux_image_probe_embeds,
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


def apply_family_guidance(eps_cfg, eps_null, family_masks, family_targets, family_anchors,
                          how, safety_scale):
    """Per-family guidance for FLUX.1-dev with spatial masks.

    Args:
        eps_cfg: guided noise prediction [B, seq, C]
        eps_null: null prediction [B, seq, C]
        family_masks: list of [B, seq, 1] packed spatial masks (one per family)
        family_targets: list of [B, seq, C] per-family target predictions
        family_anchors: list of [B, seq, C] per-family anchor predictions
        how: "anchor_inpaint" | "hybrid" | "target_sub"
        safety_scale: guidance strength scalar

    Returns:
        eps_safe: [B, seq, C]
    """
    out = eps_cfg.clone()
    s = safety_scale

    for mask_fi, et_fi, ea_fi in zip(family_masks, family_targets, family_anchors):
        m = mask_fi.to(eps_cfg.dtype)

        if how == "anchor_inpaint":
            # FLUX uses embedded guidance — anchor pred IS already guided
            blend = (s * m).clamp(max=1.0)
            out = out * (1 - blend) + ea_fi * blend

        elif how == "hybrid":
            out = (out
                   - s * m * (et_fi - eps_null)
                   + s * m * (ea_fi - eps_null))

        elif how == "target_sub":
            out = out - s * m * (et_fi - eps_null)

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


def build_flux_image_probe_embeds(
    clip_features: torch.Tensor,
    baseline_encoder_hidden: torch.Tensor,
    n_tokens: int = 4,
):
    """Construct FLUX pseudo-text embeddings from CLIP exemplar features.

    FLUX.1-dev `encoder_hidden_states` are raw T5 hidden states in the
    transformer's `joint_attention_dim` (4096) and are later projected by
    `transformer.context_embedder` into the joint inner dim (3072). To build an
    image-probe target in the same space as image tokens, inject the averaged
    CLIP exemplar vector into a few raw text-token slots, then let
    `context_embedder` project it.
    """
    if clip_features.dim() != 2:
        raise ValueError(f"clip_features must be [N,D], got {clip_features.shape}")

    avg = F.normalize(clip_features.float().mean(dim=0), dim=-1)
    baseline = baseline_encoder_hidden.clone()
    _, seq_len, hidden_dim = baseline.shape
    device = baseline.device
    dtype = baseline.dtype

    target_vec = torch.zeros(hidden_dim, device=device, dtype=dtype)
    fill = min(avg.shape[0], hidden_dim)
    target_vec[:fill] = avg[:fill].to(device=device, dtype=dtype)
    norm = target_vec.norm()
    if norm > 1e-8:
        target_vec = target_vec / norm

    n_tokens = min(n_tokens, seq_len - 1)
    token_indices = list(range(1, 1 + n_tokens))
    for idx in token_indices:
        baseline[0, idx] = target_vec

    return baseline, token_indices


def build_flux_image_probe_pooled(
    clip_features: torch.Tensor,
    baseline_pooled_embeds: torch.Tensor,
):
    """Construct pooled CLIP-style conditioning for the FLUX image probe pass."""
    if clip_features.dim() != 2:
        raise ValueError(f"clip_features must be [N,D], got {clip_features.shape}")

    avg = F.normalize(clip_features.float().mean(dim=0), dim=-1)
    pooled = torch.zeros_like(baseline_pooled_embeds)
    fill = min(avg.shape[0], pooled.shape[-1])
    pooled[..., :fill] = avg[:fill].to(device=pooled.device, dtype=pooled.dtype)
    norm = pooled.norm(dim=-1, keepdim=True)
    pooled = torch.where(norm > 1e-8, pooled / norm.clamp_min(1e-8), baseline_pooled_embeds.clone())
    return pooled


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

    # WHERE: spatial probe
    p.add_argument("--probe_mode", default="none",
                   choices=["none", "text", "image", "both", "contrast"],
                   help="text: CLIP text embedding of target concept; "
                        "image: CLIP vision exemplar embeddings; "
                        "both: union (max) of text and image masks; "
                        "contrast: prompt-vs-target patch contrast.")
    p.add_argument("--probe_block_idx", type=int, default=-1,
                   help="Which transformer block to hook (negative = from end).")
    p.add_argument("--attn_threshold", type=float, default=0.1,
                   help="Floor value for normalised spatial mask.")
    p.add_argument("--clip_embeddings", default=None,
                   help="Path to CLIP exemplar embeddings .pt (for image probe).")
    p.add_argument("--n_img_tokens", type=int, default=4,
                   help="Number of pseudo-text token slots used for image probe.")

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
    # Per-family CLIP image features (Patch 2)
    family_clip_features = {}
    # Per-family CLIP text vectors (Patch 2)
    family_clip_text_vec = {}

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

                # Load per-family CLIP image features (Patch 2)
                raw_family_clip = fdata.get("target_clip_features", {})
                if isinstance(raw_family_clip, dict):
                    family_clip_features = {k: v.float() for k, v in raw_family_clip.items()
                                            if k in family_names}

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

                    # Build per-family CLIP text vec (Patch 2)
                    if args.probe_mode in ("text", "both"):
                        try:
                            fam_concepts = meta.get("target_prompts", args.target_concepts)[:3]
                            tok = pipe.tokenizer(
                                fam_concepts,
                                padding="max_length",
                                max_length=pipe.tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt",
                            )
                            clip_out = pipe.text_encoder(
                                tok.input_ids.to(device),
                                attention_mask=tok.attention_mask.to(device)
                                if "attention_mask" in tok else None,
                                output_hidden_states=False,
                            )
                            pooled = getattr(clip_out, "pooler_output", None)
                            if pooled is None:
                                pooled = clip_out[1] if len(clip_out) > 1 else clip_out[0].mean(dim=1)
                            family_clip_text_vec[fname] = pooled.mean(dim=0)  # [768]
                        except Exception as e:
                            print(f"  [probe] WARN: CLIP text encode failed for {fname} ({e})")
                            family_clip_text_vec[fname] = None

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

    # ── WHERE: register spatial probe ──
    probe = None
    clip_text_tgt_vec = None   # [C_clip] target vector from CLIP text encoder
    clip_img_tgt_vec = None    # [C_inner] target vector after FLUX context projection
    image_probe_prompt_embeds = None
    image_probe_pooled = None
    # Per-family image target vectors (Patch 3)
    clip_img_tgt_vec_by_family = {}
    # Per-family grouped probe embeds (Patch 3)
    family_image_token_map = {}
    family_grouped_probe_embeds = None
    family_grouped_probe_pooled = None

    if args.probe_mode != "none" and not args.no_safety:
        # Prefer single-stream blocks: output is the full joint [txt; img] sequence,
        # so our hook can simply take the LAST seq_img tokens as image features.
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
            probe = FluxSpatialProbe(blocks[idx], seq_img_len=packed_seq_len)
            print(f"  [probe] mode={args.probe_mode} hooked block {idx}/{len(blocks)-1}")

        # ── Build target vectors for text / image probes ──
        # Text probe: CLIP-L text embedding of target concept (pooled 768-d)
        if args.probe_mode in ("text", "both"):
            try:
                with torch.no_grad():
                    tok = pipe.tokenizer(
                        args.target_concepts,
                        padding="max_length",
                        max_length=pipe.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    clip_out = pipe.text_encoder(
                        tok.input_ids.to(device),
                        attention_mask=tok.attention_mask.to(device)
                        if "attention_mask" in tok else None,
                        output_hidden_states=False,
                    )
                    # pooler_output: [N, 768]
                    pooled = getattr(clip_out, "pooler_output", None)
                    if pooled is None:
                        pooled = clip_out[1] if len(clip_out) > 1 else clip_out[0].mean(dim=1)
                    clip_text_tgt_vec = pooled.mean(dim=0)  # [768]
                    print(f"  [probe] CLIP text target vec: shape={tuple(clip_text_tgt_vec.shape)}")
            except Exception as e:
                print(f"  [probe] WARN: CLIP text encode failed ({e}); text probe disabled")
                clip_text_tgt_vec = None

        # Image probe: load CLIP vision exemplar features
        if args.probe_mode in ("image", "both"):
            # Non-family path: load from --clip_embeddings
            if not (args.family_guidance and family_clip_features):
                if args.clip_embeddings is None:
                    print("  [probe] WARN: --clip_embeddings not given; image probe disabled")
                else:
                    try:
                        cdata = torch.load(args.clip_embeddings, map_location="cpu",
                                           weights_only=False)
                        feats = cdata.get("target_clip_features", None)
                        if feats is None:
                            feats = cdata.get("target_clip_embeds", None)
                            if feats is not None and feats.dim() == 3:
                                feats = feats.mean(dim=1)  # [N, C]
                        if feats is None:
                            print("  [probe] WARN: no target CLIP features in file; image probe disabled")
                        else:
                            raw_probe_embeds, img_probe_token_idx = build_flux_image_probe_embeds(
                                feats.float(), pe_null.detach(), args.n_img_tokens)
                            image_probe_prompt_embeds = raw_probe_embeds.detach()
                            image_probe_pooled = build_flux_image_probe_pooled(
                                feats.float(), pooled_null.detach()
                            ).detach()
                            ctx_embedder = transformer.context_embedder
                            ctx_dtype = next(ctx_embedder.parameters()).dtype
                            ctx_device = next(ctx_embedder.parameters()).device
                            with torch.no_grad():
                                probe_ctx = ctx_embedder(
                                    raw_probe_embeds.to(device=ctx_device, dtype=ctx_dtype))
                            clip_img_tgt_vec = F.normalize(
                                probe_ctx[0, img_probe_token_idx].mean(dim=0).float(), dim=-1
                            ).to(device)
                            print(
                                f"  [probe] CLIP image pseudo-text: raw={tuple(raw_probe_embeds.shape)} "
                                f"ctx={tuple(probe_ctx.shape)} vec={tuple(clip_img_tgt_vec.shape)} "
                                f"pooled={tuple(image_probe_pooled.shape)} tokens={img_probe_token_idx}"
                            )
                    except Exception as e:
                        print(f"  [probe] WARN: load CLIP embeddings failed ({e})")
                        clip_img_tgt_vec = None
                        image_probe_prompt_embeds = None
                        image_probe_pooled = None

            # Family path: build grouped probe embeds (Patch 3)
            if args.family_guidance and family_clip_features:
                try:
                    raw_grouped_embeds, grouped_token_indices, family_image_token_map = (
                        build_grouped_flux_image_probe_embeds(
                            family_clip_features,
                            pe_null.detach(),
                            max_tokens=args.n_img_tokens,
                        )
                    )
                    family_grouped_probe_embeds = raw_grouped_embeds.detach()

                    # Build combined pooled: mean of per-family pooled CLIP features
                    all_family_pooled = []
                    for fname in family_names:
                        if fname in family_clip_features:
                            fp_pooled = build_flux_image_probe_pooled(
                                family_clip_features[fname], pooled_null.detach()
                            )
                            all_family_pooled.append(fp_pooled)
                    if all_family_pooled:
                        family_grouped_probe_pooled = torch.stack(all_family_pooled).mean(0).detach()
                    else:
                        family_grouped_probe_pooled = pooled_null.detach()

                    # Project through context_embedder to get per-family target vecs (Patch 3)
                    ctx_embedder = transformer.context_embedder
                    ctx_dtype = next(ctx_embedder.parameters()).dtype
                    ctx_device = next(ctx_embedder.parameters()).device
                    with torch.no_grad():
                        grouped_ctx = ctx_embedder(
                            raw_grouped_embeds.to(device=ctx_device, dtype=ctx_dtype))
                    for fname, tok_pos in family_image_token_map.items():
                        clip_img_tgt_vec_by_family[fname] = F.normalize(
                            grouped_ctx[0, tok_pos].float(), dim=-1
                        ).to(device)
                    print(
                        f"  [probe] Family grouped image probe: "
                        f"token_map={family_image_token_map} "
                        f"ctx={tuple(grouped_ctx.shape)}"
                    )
                except Exception as e:
                    print(f"  [probe] WARN: build grouped flux image probe failed ({e})")
                    family_grouped_probe_embeds = None
                    family_grouped_probe_pooled = None

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
                    if probe is not None:
                        probe.reset()
                    with torch.no_grad():
                        if probe is not None:
                            probe.active = True
                            probe.tag = "prompt"
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
                        if probe is not None:
                            probe.active = False

                        # Null pass — use guidance tensor (required by transformer API)
                        en = transformer(
                            hidden_states=lat_in,
                            timestep=timestep / 1000,
                            guidance=guidance_tensor,
                            pooled_projections=pooled_null.to(dtype),
                            encoder_hidden_states=pe_null.to(dtype),
                            txt_ids=text_ids_null,
                            img_ids=latent_image_ids,
                            return_dict=False,
                        )[0]

                        if probe is not None and args.probe_mode == "contrast":
                            probe.active = True
                            probe.tag = "target"
                        et = transformer(
                            hidden_states=lat_in,
                            timestep=timestep / 1000,
                            guidance=guidance_tensor,
                            pooled_projections=pooled_target.to(dtype),
                            encoder_hidden_states=pe_target.to(dtype),
                            txt_ids=text_ids_null,
                            img_ids=latent_image_ids,
                            return_dict=False,
                        )[0]
                        if probe is not None:
                            probe.active = False

                        if (
                            probe is not None
                            and args.probe_mode in ("image", "both")
                            and image_probe_prompt_embeds is not None
                            and image_probe_pooled is not None
                        ):
                            probe.active = True
                            probe.tag = "image"
                            _ = transformer(
                                hidden_states=lat_in,
                                timestep=timestep / 1000,
                                guidance=guidance_tensor,
                                pooled_projections=image_probe_pooled.to(dtype),
                                encoder_hidden_states=image_probe_prompt_embeds.to(dtype),
                                txt_ids=text_ids_null,
                                img_ids=latent_image_ids,
                                return_dict=False,
                            )[0]
                            probe.active = False

                    cv, trig = cas.compute(ep, en, et)
                    cas_values.append(cv)
                    eps_final = ep

                    # ── Compute spatial mask from probe captures (if enabled) ──
                    spatial_mask_packed = None
                    if probe is not None and trig and "prompt" in probe.captures:
                        feat_p = probe.captures["prompt"]  # [B, seq_img, C]
                        masks_2d = []

                        if args.probe_mode == "contrast" and "target" in probe.captures:
                            m = compute_flux_spatial_mask(
                                feat_p, target_feat=probe.captures["target"],
                                threshold=args.attn_threshold, mode="contrast")
                            masks_2d.append(m)

                        if args.probe_mode in ("text", "both"):
                            # Prefer CLIP-L text vector; fall back to T5-avg of
                            # pe_target (joint-text-stream mean) if dims mismatch.
                            tv = None
                            if clip_text_tgt_vec is not None and \
                               clip_text_tgt_vec.shape[-1] == feat_p.shape[-1]:
                                tv = clip_text_tgt_vec.to(feat_p.device, feat_p.dtype)
                            else:
                                tv_t5 = pe_target.to(feat_p.device, feat_p.dtype).mean(dim=1)
                                if tv_t5.shape[-1] == feat_p.shape[-1]:
                                    tv = tv_t5.squeeze(0) if tv_t5.dim() == 2 else tv_t5
                            m = compute_flux_spatial_mask(
                                feat_p, target_vec=tv,
                                threshold=args.attn_threshold, mode="text")
                            masks_2d.append(m)

                        if args.probe_mode in ("image", "both"):
                            if "image" in probe.captures:
                                m = compute_flux_spatial_mask(
                                    feat_p, target_feat=probe.captures["image"],
                                    threshold=args.attn_threshold, mode="contrast")
                                masks_2d.append(m)
                            elif clip_img_tgt_vec is not None:
                                tv = clip_img_tgt_vec.to(feat_p.device, feat_p.dtype)
                                m = compute_flux_spatial_mask(
                                    feat_p, target_vec=tv,
                                    threshold=args.attn_threshold, mode="text")
                                masks_2d.append(m)

                        if masks_2d:
                            mask2d = torch.stack(masks_2d, dim=0).max(dim=0)[0] \
                                if len(masks_2d) > 1 else masks_2d[0]
                            spatial_mask_packed = mask_to_packed_seq(
                                mask2d, seq_img_len=feat_p.shape[1]
                            ).to(ep.device, ep.dtype)
                            if args.debug and step_idx % 10 == 0:
                                print(f"  [probe] step={step_idx} mask "
                                      f"min={mask2d.min().item():.3f} "
                                      f"max={mask2d.max().item():.3f} "
                                      f"mean={mask2d.mean().item():.3f}")

                    if trig:
                        if args.family_guidance and family_names:
                            fam_ts, fam_as = [], []
                            fam_probe_captures = {}  # per-family target captures (Patch 4)

                            with torch.no_grad():
                                for fname in family_names:
                                    # Enable probe for per-family target pass (Patch 4)
                                    if probe is not None and args.probe_mode in ("image", "both"):
                                        probe.active = True
                                        probe.tag = f"target_{fname}"
                                    ft = transformer(
                                        hidden_states=lat_in,
                                        timestep=timestep / 1000,
                                        guidance=guidance_tensor,
                                        pooled_projections=family_target_pooled[fname].to(dtype),
                                        encoder_hidden_states=family_target_emb[fname].to(dtype),
                                        txt_ids=text_ids_null,
                                        img_ids=latent_image_ids,
                                        return_dict=False,
                                    )[0]
                                    if probe is not None:
                                        probe.active = False
                                        # Stash per-family target capture
                                        tag = f"target_{fname}"
                                        if tag in probe.captures:
                                            fam_probe_captures[fname] = probe.captures[tag]

                                    fa = transformer(
                                        hidden_states=lat_in,
                                        timestep=timestep / 1000,
                                        guidance=guidance_tensor,
                                        pooled_projections=family_anchor_pooled[fname].to(dtype),
                                        encoder_hidden_states=family_anchor_emb[fname].to(dtype),
                                        txt_ids=text_ids_null,
                                        img_ids=latent_image_ids,
                                        return_dict=False,
                                    )[0]
                                    fam_ts.append(ft)
                                    fam_as.append(fa)

                            # ── Build per-family spatial masks (Patch 4) ──
                            fam_masks = []
                            feat_p = probe.captures.get("prompt") if probe is not None else None

                            for fname in family_names:
                                fm_list = []

                                if feat_p is not None:
                                    # Text probe for this family
                                    if args.probe_mode in ("text", "both"):
                                        ftv = family_clip_text_vec.get(fname)
                                        if ftv is not None and ftv.shape[-1] == feat_p.shape[-1]:
                                            tv = ftv.to(feat_p.device, feat_p.dtype)
                                        else:
                                            # Fallback: T5 mean of family target embed
                                            tv_t5 = family_target_emb[fname].to(feat_p.device, feat_p.dtype).mean(dim=1)
                                            tv = tv_t5.squeeze(0) if tv_t5.dim() == 2 else tv_t5 \
                                                if tv_t5.shape[-1] == feat_p.shape[-1] else None
                                        m_txt = compute_flux_spatial_mask(
                                            feat_p, target_vec=tv,
                                            threshold=args.attn_threshold, mode="text")
                                        fm_list.append(m_txt)

                                    # Image probe for this family
                                    if args.probe_mode in ("image", "both"):
                                        if fname in fam_probe_captures:
                                            m_img = compute_flux_spatial_mask(
                                                feat_p,
                                                target_feat=fam_probe_captures[fname],
                                                threshold=args.attn_threshold, mode="contrast")
                                            fm_list.append(m_img)
                                        elif fname in clip_img_tgt_vec_by_family:
                                            tv = clip_img_tgt_vec_by_family[fname].to(
                                                feat_p.device, feat_p.dtype)
                                            m_img = compute_flux_spatial_mask(
                                                feat_p, target_vec=tv,
                                                threshold=args.attn_threshold, mode="text")
                                            fm_list.append(m_img)

                                if fm_list:
                                    # Max across text/image masks for this family
                                    fm_2d = torch.stack(fm_list, dim=0).max(dim=0)[0] \
                                        if len(fm_list) > 1 else fm_list[0]
                                    # Convert to packed sequence [B, seq, 1]
                                    fm_packed = mask_to_packed_seq(
                                        fm_2d, seq_img_len=feat_p.shape[1]
                                    ).to(ep.device, ep.dtype)
                                else:
                                    # No probe data: use spatial_mask_packed or uniform
                                    if spatial_mask_packed is not None:
                                        fm_packed = spatial_mask_packed
                                    else:
                                        fm_packed = torch.ones(
                                            ep.shape[0], ep.shape[1], 1,
                                            device=ep.device, dtype=ep.dtype)

                                fam_masks.append(fm_packed)

                            # ── Winner-take-all across families (Patch 4) ──
                            if len(fam_masks) > 1:
                                stacked = torch.cat(fam_masks, dim=-1)  # [B, seq, N]
                                winner = stacked.argmax(dim=-1, keepdim=True)  # [B, seq, 1]
                                for fi in range(len(fam_masks)):
                                    fam_masks[fi] = fam_masks[fi] * (winner == fi).float()

                            eps_final = apply_family_guidance(
                                ep, en, fam_masks, fam_ts, fam_as,
                                args.how_mode, args.safety_scale)
                        else:
                            with torch.no_grad():
                                ea = transformer(
                                    hidden_states=lat_in,
                                    timestep=timestep / 1000,
                                    guidance=guidance_tensor,
                                    pooled_projections=pooled_anchor.to(dtype),
                                    encoder_hidden_states=pe_anchor.to(dtype),
                                    txt_ids=text_ids_null,
                                    img_ids=latent_image_ids,
                                    return_dict=False,
                                )[0]

                            m_use = spatial_mask_packed if spatial_mask_packed is not None else 1.0
                            eps_final = apply_guidance(
                                ep, en, et, ea,
                                mask=m_use, how=args.how_mode,
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

    if probe is not None:
        probe.remove()

    gi = sum(1 for s in stats if s["guided"] > 0)
    print(f"\nDone! {len(stats)} images, guided {gi}/{len(stats)}")


if __name__ == "__main__":
    main()
