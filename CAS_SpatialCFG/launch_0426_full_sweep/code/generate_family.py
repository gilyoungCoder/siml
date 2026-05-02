#!/usr/bin/env python
"""
SafeGen Family-Guided Generation.

Extension of generate.py with family-specific guidance: each concept family
(e.g., weapon_threat, bodily_injury) gets its own probe mask AND its own
anchor/target direction, enabling semantically precise per-region guidance.

Ablation modes:
  --family_guidance       Enable family-specific guidance (vs single-anchor)
  --probe_mode text/image/both  Probe type ablation
  --how_mode anchor_inpaint/hybrid/target_sub  Guidance mode ablation

Usage:
    python -m safegen.generate_family \
        --prompts prompts/mja_violent.txt \
        --outdir outputs/violence_family \
        --family_config configs/exemplars/violence/clip_grouped.pt \
        --family_guidance \
        --how_mode anchor_inpaint
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
from diffusers import StableDiffusionPipeline, DDIMScheduler

from safegen.attention_probe import (
    AttentionProbeStore,
    precompute_target_keys,
    register_attention_probe,
    restore_processors,
    compute_attention_spatial_mask,
    find_token_indices,
)


# ---------------------------------------------------------------------------
# WHEN: Global CAS
# ---------------------------------------------------------------------------
class GlobalCAS:
    def __init__(self, threshold=0.6, sticky=True):
        self.threshold, self.sticky, self.triggered = threshold, sticky, False

    def reset(self):
        self.triggered = False

    def compute(self, eps_prompt, eps_null, eps_target):
        dp = (eps_prompt - eps_null).reshape(1, -1).float()
        dt = (eps_target - eps_null).reshape(1, -1).float()
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
# HOW: Family-specific guidance
# ---------------------------------------------------------------------------
def apply_family_guidance(
    eps_cfg, eps_null, family_masks, family_targets, family_anchors,
    mode, safety_scale, cfg_scale, device,
):
    """
    Apply per-family guidance. Each family has its own mask, target, and anchor.

    Args:
        eps_cfg: [1, 4, 64, 64] CFG noise prediction
        eps_null: [1, 4, 64, 64] unconditional prediction
        family_masks: list of [1, 1, 64, 64] masks, one per family
        family_targets: list of [1, 4, 64, 64] target predictions
        family_anchors: list of [1, 4, 64, 64] anchor predictions
        mode: "anchor_inpaint", "hybrid", "target_sub"
        safety_scale: guidance strength
        cfg_scale: CFG scale for anchor_inpaint mode

    Returns:
        eps_safe: [1, 4, 64, 64]
    """
    out = eps_cfg.clone()
    s = safety_scale

    for M_fi, et_fi, ea_fi in zip(family_masks, family_targets, family_anchors):
        m = M_fi.to(eps_cfg.dtype)

        if mode == "anchor_inpaint":
            ea_cfg = eps_null + cfg_scale * (ea_fi - eps_null)
            blend = (s * m).clamp(max=1.0)
            out = out * (1 - blend) + ea_cfg * blend

        elif mode == "hybrid":
            out = (out
                   - s * m * (et_fi - eps_null)
                   + s * m * (ea_fi - eps_null))

        elif mode == "target_sub":
            out = out - s * m * (et_fi - eps_null)

    if torch.isnan(out).any() or torch.isinf(out).any():
        out = torch.where(torch.isfinite(out), out, eps_cfg)
    return out


def apply_single_guidance(
    eps_cfg, eps_null, eps_prompt, eps_target, eps_anchor,
    mask, mode, safety_scale, cfg_scale,
):
    """Single-anchor guidance (baseline for ablation)."""
    m = mask.to(eps_cfg.dtype)
    s = safety_scale

    if mode == "anchor_inpaint":
        ea_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
        blend = (s * m).clamp(max=1.0)
        out = eps_cfg * (1 - blend) + ea_cfg * blend
    elif mode == "hybrid":
        out = (eps_cfg
               - s * m * (eps_target - eps_null)
               + s * m * (eps_anchor - eps_null))
    elif mode == "target_sub":
        out = eps_cfg - s * m * (eps_target - eps_null)
    else:
        raise ValueError(mode)

    if torch.isnan(out).any() or torch.isinf(out).any():
        out = torch.where(torch.isfinite(out), out, eps_cfg)
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
                raise ValueError(f"No prompt column in {reader.fieldnames}")
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


def encode_concepts(text_encoder, tokenizer, concepts, device):
    embeds = []
    for c in concepts:
        inp = tokenizer(c, padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
        embeds.append(text_encoder(inp.input_ids.to(device))[0])
    return torch.stack(embeds).mean(0)


def build_image_probe_embeds(clip_features, text_encoder, tokenizer, device, n_tokens=4):
    avg = F.normalize(clip_features.mean(dim=0), dim=-1)
    dtype = next(text_encoder.parameters()).dtype
    with torch.no_grad():
        empty_ids = tokenizer("", padding="max_length", max_length=77,
                              truncation=True, return_tensors="pt").input_ids.to(device)
        baseline = text_encoder(empty_ids)[0]
    result = baseline.clone()
    concept = avg.to(device=device, dtype=dtype)
    for i in range(1, 1 + n_tokens):
        result[0, i] = concept
    return result, list(range(1, 1 + n_tokens))


def build_grouped_probe_embeds(family_features_dict, text_encoder, tokenizer, device, max_tokens=4):
    """
    Build probe embedding with one token per family.
    Returns embeds [1,77,768], token_indices, family_names_ordered.
    """
    dtype = next(text_encoder.parameters()).dtype
    with torch.no_grad():
        empty_ids = tokenizer("", padding="max_length", max_length=77,
                              truncation=True, return_tensors="pt").input_ids.to(device)
        baseline = text_encoder(empty_ids)[0]

    result = baseline.clone()
    family_names = list(family_features_dict.keys())[:max_tokens]
    token_indices = []

    for i, fname in enumerate(family_names):
        feats = family_features_dict[fname]
        avg = F.normalize(feats.float().mean(dim=0), dim=-1)
        result[0, i + 1] = avg.to(device=device, dtype=dtype)
        token_indices.append(i + 1)

    # Pad remaining tokens with last family
    if family_names:
        last_avg = F.normalize(family_features_dict[family_names[-1]].float().mean(dim=0), dim=-1)
        for i in range(len(family_names), max_tokens):
            result[0, i + 1] = last_avg.to(device=device, dtype=dtype)

    return result, token_indices, family_names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = ArgumentParser(description="SafeGen: Family-Guided Generation")
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

    # WHERE — Probe
    p.add_argument("--probe_mode", default="both", choices=["text", "image", "both"])
    p.add_argument("--family_config", default=None,
                   help="Path to grouped .pt file from prepare_grouped_exemplar.py")
    p.add_argument("--clip_embeddings", default=None,
                   help="Path to single .pt file (non-family mode)")
    p.add_argument("--attn_resolutions", type=int, nargs="+", default=[16, 32])
    p.add_argument("--attn_threshold", type=float, default=0.3)
    p.add_argument("--img_attn_threshold", type=float, default=None)
    p.add_argument("--attn_sigmoid_alpha", type=float, default=10.0)
    p.add_argument("--blur_sigma", type=float, default=1.0)
    p.add_argument("--probe_fusion", default="union", choices=["union", "soft_union", "mean"])
    p.add_argument("--target_words", nargs="+", default=None)
    p.add_argument("--n_img_tokens", type=int, default=4)

    # HOW
    p.add_argument("--how_mode", default="anchor_inpaint",
                   choices=["anchor_inpaint", "hybrid", "target_sub"])
    p.add_argument("--safety_scale", type=float, default=1.0)
    p.add_argument("--family_guidance", action="store_true",
                   help="Enable per-family guidance (vs single-anchor)")

    # Concepts (concept-level descriptor for CAS direction).
    # NO default — must come from pack metadata (`concept_keywords`) OR be passed explicitly.
    # The previous nudity default was a silent footgun: non-nudity packs would compute CAS
    # against a nudity direction. Now we error out if neither source provides a descriptor.
    p.add_argument("--target_concepts", nargs="+", default=None,
                   help="Optional explicit override of concept-level descriptor. "
                        "Default behavior reads pack's `concept_keywords` field. "
                        "Required only if pack lacks concept_keywords.")
    p.add_argument("--anchor_concepts", nargs="+",
                   default=["clothed person", "person wearing clothes"],
                   help="Concept-level anchor descriptor (single-anchor fallback only). "
                        "Family-mode uses per-family anchor_words from the pack.")

    p.add_argument("--save_maps", action="store_true")

    args = p.parse_args()
    if args.img_attn_threshold is None:
        args.img_attn_threshold = args.attn_threshold

    # Auto-extract target_words from --target_concepts only if both given via CLI;
    # otherwise pack-loading path will derive it from concept_keywords.
    if args.target_words is None and args.target_concepts is not None:
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
    print(f"SafeGen: Family-Guided Generation")
    print(f"{'=' * 60}")
    print(f"  WHEN:   CAS threshold={args.cas_threshold}")
    print(f"  WHERE:  probe={args.probe_mode}, res={args.attn_resolutions}")
    print(f"  HOW:    {args.how_mode}, ss={args.safety_scale}")
    print(f"  FAMILY: {args.family_guidance}")
    print(f"{'=' * 60}\n")

    # Load prompts
    prompts = load_prompts(args.prompts)
    end = args.end_idx if args.end_idx > 0 else len(prompts)
    prompt_work = list(enumerate(prompts))[args.start_idx:end]

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, safety_checker=None,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.feature_extractor = None
    unet, vae, tok, te, sched = pipe.unet, pipe.vae, pipe.tokenizer, pipe.text_encoder, pipe.scheduler
    unet_dtype = next(unet.parameters()).dtype

    # Encode anchor + null. text_tgt (concept-level CAS direction) is computed lazily after
    # pack-load: the pack's `concept_keywords` field is the canonical source. Falling back
    # to args.target_concepts only if pack lacks the field; if neither, raise.
    with torch.no_grad():
        anchor_emb = encode_concepts(te, tok, args.anchor_concepts, device)
        unc = te(tok("", padding="max_length", max_length=tok.model_max_length,
                     truncation=True, return_tensors="pt").input_ids.to(device))[0]
        text_tgt = None
        if args.target_concepts is not None:
            text_tgt = encode_concepts(te, tok, args.target_concepts, device)

    # ── Load family config ──
    family_names = []
    family_target_embeds = {}  # {fname: [1, 77, 768]}
    family_anchor_embeds = {}
    family_target_words = {}   # {fname: [word1, word2, ...]}

    if args.family_config and os.path.exists(args.family_config):
        print(f"Loading family config: {args.family_config}")
        fdata = torch.load(args.family_config, map_location="cpu", weights_only=False)
        family_token_map = fdata.get("family_token_map", {})
        family_meta = fdata.get("family_metadata", {})
        family_names = list(family_token_map.keys())
        if not family_names:
            family_names = fdata.get("family_names", []) or list(family_meta.keys())

        # Concept-level CAS descriptor — pack's `concept_keywords` is the authority.
        # No silent nudity fallback: error if neither pack nor CLI provides it.
        ckw = fdata.get("concept_keywords")
        if not ckw:
            if args.target_concepts is None:
                raise ValueError(
                    f"Pack {args.family_config} has no `concept_keywords` metadata "
                    f"and --target_concepts was not provided. "
                    f"Cannot determine concept-level CAS descriptor. "
                    f"Either re-build the pack with concept_keywords, or pass "
                    f"--target_concepts <kw1> <kw2> ...  Refusing to silently fall "
                    f"back to nudity defaults (would compute CAS against the wrong "
                    f"direction for non-nudity concepts)."
                )
            ckw = list(args.target_concepts)
        ckw_clean = [str(w).replace("_", " ").strip() for w in ckw if str(w).strip()]
        if not ckw_clean:
            raise ValueError(
                f"Pack {args.family_config} resolved to empty concept descriptor. "
                f"Source ckw={ckw}."
            )
        # Override CLI args.target_concepts with pack-derived (or honor explicit CLI).
        if args.target_concepts is None:
            args.target_concepts = list(ckw_clean)
            # rederive target_words so legacy txt_probe registration (line 425) works
            words = []
            for concept in args.target_concepts:
                for w in concept.replace("_", " ").split():
                    w_clean = w.strip().lower()
                    if len(w_clean) >= 3 and w_clean not in words:
                        words.append(w_clean)
            args.target_words = words
        # Re-encode text_tgt with the resolved descriptor (pack-derived or CLI-provided).
        with torch.no_grad():
            text_tgt = encode_concepts(te, tok, ckw_clean, device)
        print(f"  Concept descriptor (CAS): {ckw_clean}")

        # Per-family target/anchor embeddings
        target_feats = fdata.get("target_clip_features", {})
        anchor_feats = fdata.get("anchor_clip_features", {})

        for fname in family_names:
            # Target: encode family-specific target words (use up to 5 keywords to align
            # with mask construction's `fw[:5]` token-index window)
            tw = family_meta.get(fname, {}).get("target_words") or family_meta.get(fname, {}).get("target_prompts") or args.target_concepts
            family_target_words[fname] = tw
            family_target_embeds[fname] = encode_concepts(te, tok, tw[:5], device)

            # Anchor: encode family-specific anchor words
            aw = family_meta.get(fname, {}).get("anchor_words") or family_meta.get(fname, {}).get("anchor_prompts") or args.anchor_concepts
            family_anchor_embeds[fname] = encode_concepts(te, tok, aw[:5], device)

        print(f"  Families: {family_names}")
        for fn in family_names:
            print(f"    {fn}: target={family_target_words[fn][:5]}, "
                  f"anchor={family_meta.get(fn, {}).get('anchor_words', [])[:5]}")
    else:
        # No pack loaded — must have explicit --target_concepts.
        if args.target_concepts is None:
            raise ValueError(
                "No --family_config provided AND --target_concepts not given. "
                "Cannot determine concept-level CAS descriptor. "
                "Either pass --family_config <pack.pt> or pass --target_concepts <kw1> <kw2> ..."
            )

    # ── Setup image probe ──
    img_probe = None
    img_tok_idx = None
    original_procs = None

    use_img = args.probe_mode in ("image", "both")
    use_txt = args.probe_mode in ("text", "both")

    if use_img:
        img_probe = AttentionProbeStore()

        if args.family_config and os.path.exists(args.family_config):
            fdata = torch.load(args.family_config, map_location="cpu", weights_only=False)
            target_feats = fdata.get("target_clip_features", {})
            img_embeds, img_tok_idx, _ = build_grouped_probe_embeds(
                target_feats, te, tok, device, args.n_img_tokens)
        elif args.clip_embeddings:
            clip_data = torch.load(args.clip_embeddings, map_location="cpu")
            clip_feats = clip_data.get("target_clip_features", clip_data.get("target_cls")).float()
            img_embeds, img_tok_idx = build_image_probe_embeds(
                clip_feats, te, tok, device, args.n_img_tokens)
        else:
            raise ValueError("Need --family_config or --clip_embeddings for image probe")

        img_keys = precompute_target_keys(unet, img_embeds.to(dtype=unet_dtype), args.attn_resolutions)
        original_procs = register_attention_probe(unet, img_probe, img_keys, args.attn_resolutions)

    # ── Setup text probe (uses global target keywords) ──
    txt_probe = None
    txt_tok_idx = None

    if use_txt:
        txt_probe = AttentionProbeStore()
        txt_keys = precompute_target_keys(unet, text_tgt.to(dtype=unet_dtype), args.attn_resolutions)
        target_text = ", ".join(args.target_concepts)
        txt_tok_idx = find_token_indices(target_text, args.target_words, tok)

        if not use_img:
            original_procs = register_attention_probe(unet, txt_probe, txt_keys, args.attn_resolutions)
        else:
            # Both: register text probe on same layers (dual store approach)
            from safegen.attention_probe import register_dual_attention_probe
            if original_procs:
                restore_processors(unet, original_procs)
            txt_probe_store = AttentionProbeStore()
            original_procs = register_dual_attention_probe(
                unet, img_probe, txt_probe_store,
                img_keys,
                precompute_target_keys(unet, text_tgt.to(dtype=unet_dtype), args.attn_resolutions),
                args.attn_resolutions)
            txt_probe = txt_probe_store

    # ── Generate ──
    cas = GlobalCAS(args.cas_threshold)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.save_maps:
        (outdir / "maps").mkdir(exist_ok=True)
    stats = []

    for pi, prompt in tqdm(prompt_work, desc="Generating"):
        if not prompt.strip():
            continue
        for si in range(args.nsamples):
            seed = args.seed + pi * args.nsamples + si
            set_seed(seed)
            cas.reset()
            guided_count = 0
            cas_vals, mask_areas = [], []

            with torch.no_grad():
                pemb = te(tok(prompt, padding="max_length", max_length=tok.model_max_length,
                              truncation=True, return_tensors="pt").input_ids.to(device))[0]

            set_seed(seed)
            lat = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
            lat = lat * sched.init_noise_sigma
            sched.set_timesteps(args.steps, device=device)

            for step_i, t in enumerate(sched.timesteps):
                li = sched.scale_model_input(lat, t)

                # Forward with probe active
                if img_probe:
                    img_probe.active = True; img_probe.reset()
                if txt_probe:
                    txt_probe.active = True; txt_probe.reset()

                with torch.no_grad():
                    raw = unet(torch.cat([li, li]), t,
                               encoder_hidden_states=torch.cat([unc, pemb])).sample
                    en, ep = raw.chunk(2)

                if img_probe:
                    img_probe.active = False
                if txt_probe:
                    txt_probe.active = False

                # Target for CAS
                with torch.no_grad():
                    et = unet(li, t, encoder_hidden_states=text_tgt).sample

                ec = en + args.cfg_scale * (ep - en)
                cv, trig = cas.compute(ep, en, et)
                cas_vals.append(cv)

                if trig:
                    # ── Family-specific guidance ──
                    if args.family_guidance and family_names:
                        # 1. Compute per-family probe masks
                        fam_masks = []
                        for fi, fname in enumerate(family_names):
                            fm = torch.zeros(1, 1, 64, 64, device=device)

                            # Image probe: use family-specific token
                            if img_probe and img_probe.get_maps():
                                fa = compute_attention_spatial_mask(
                                    img_probe, token_indices=[fi + 1],
                                    target_resolution=64,
                                    resolutions_to_use=args.attn_resolutions)
                                fm = torch.max(fm, make_probe_mask(
                                    fa, args.img_attn_threshold,
                                    args.attn_sigmoid_alpha, args.blur_sigma, device))

                            # Text probe: use family-specific keywords
                            if txt_probe and txt_probe.get_maps():
                                fw = family_target_words.get(fname, args.target_words)
                                target_text_fi = ", ".join(fw[:5])
                                fi_tok_idx = find_token_indices(target_text_fi, fw, tok)
                                if fi_tok_idx:
                                    ta = compute_attention_spatial_mask(
                                        txt_probe, token_indices=fi_tok_idx,
                                        target_resolution=64,
                                        resolutions_to_use=args.attn_resolutions)
                                    fm = torch.max(fm, make_probe_mask(
                                        ta, args.attn_threshold,
                                        args.attn_sigmoid_alpha, args.blur_sigma, device))

                            fam_masks.append(fm)

                        # 2. Resolve overlapping regions: argmax winner-take-all
                        if len(fam_masks) > 1:
                            stacked = torch.cat(fam_masks, dim=0)  # [N, 1, 64, 64]
                            winner = stacked.argmax(dim=0, keepdim=True)  # [1, 1, 64, 64]
                            for fi in range(len(fam_masks)):
                                fam_masks[fi] = fam_masks[fi] * (winner == fi).float().squeeze(0)

                        # 3. Per-family UNet calls
                        fam_targets = []
                        fam_anchors = []
                        with torch.no_grad():
                            for fname in family_names:
                                ft = unet(li, t, encoder_hidden_states=family_target_embeds[fname]).sample
                                fa = unet(li, t, encoder_hidden_states=family_anchor_embeds[fname]).sample
                                fam_targets.append(ft)
                                fam_anchors.append(fa)

                        # 4. Apply family guidance
                        eps_final = apply_family_guidance(
                            ec, en, fam_masks, fam_targets, fam_anchors,
                            args.how_mode, args.safety_scale, args.cfg_scale, device)

                        guided_count += 1
                        total_mask = sum(fm.mean().item() for fm in fam_masks)
                        mask_areas.append(total_mask)

                    else:
                        # ── Single-anchor fallback ──
                        with torch.no_grad():
                            ea = unet(li, t, encoder_hidden_states=anchor_emb).sample

                        # Compute unified probe mask
                        probe_mask = torch.zeros(1, 1, 64, 64, device=device)

                        if img_probe and img_probe.get_maps():
                            ia = compute_attention_spatial_mask(
                                img_probe, token_indices=img_tok_idx,
                                target_resolution=64,
                                resolutions_to_use=args.attn_resolutions)
                            probe_mask = torch.max(probe_mask, make_probe_mask(
                                ia, args.img_attn_threshold,
                                args.attn_sigmoid_alpha, args.blur_sigma, device))

                        if txt_probe and txt_probe.get_maps() and txt_tok_idx:
                            ta = compute_attention_spatial_mask(
                                txt_probe, token_indices=txt_tok_idx,
                                target_resolution=64,
                                resolutions_to_use=args.attn_resolutions)
                            probe_mask = torch.max(probe_mask, make_probe_mask(
                                ta, args.attn_threshold,
                                args.attn_sigmoid_alpha, args.blur_sigma, device))

                        if probe_mask.sum() == 0:
                            probe_mask = torch.ones(1, 1, 64, 64, device=device) * 0.5

                        eps_final = apply_single_guidance(
                            ec, en, ep, et, ea, probe_mask,
                            args.how_mode, args.safety_scale, args.cfg_scale)

                        guided_count += 1
                        mask_areas.append(float(probe_mask.mean()))

                    if args.save_maps and step_i % 10 == 0:
                        md = outdir / "maps"
                        pf = f"{pi:04d}_{si:02d}_s{step_i:03d}"
                        if args.family_guidance and family_names:
                            for fi, fname in enumerate(family_names):
                                mn = fam_masks[fi][0, 0].float().cpu().numpy()
                                Image.fromarray(
                                    (np.clip(mn, 0, 1) * 255).astype(np.uint8), "L"
                                ).save(str(md / f"{pf}_{fname}.png"))
                else:
                    eps_final = ec

                prev_lat = lat.clone()
                lat = sched.step(eps_final, t, lat).prev_sample
                if torch.isnan(lat).any() or torch.isinf(lat).any():
                    lat = sched.step(
                        en + args.cfg_scale * (ep - en), t, prev_lat
                    ).prev_sample

            # Decode
            with torch.no_grad():
                dec = vae.decode(lat.to(vae.dtype) / vae.config.scaling_factor).sample
                dec = (dec / 2 + 0.5).clamp(0, 1)
                img = (dec[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

            Image.fromarray(img).resize((512, 512)).save(str(outdir / f"{pi:04d}_{si:02d}.png"))
            stats.append({
                "prompt_idx": pi, "sample_idx": si, "seed": seed,
                "guided_steps": guided_count,
                "max_cas": max(cas_vals) if cas_vals else 0,
                "mean_mask_area": float(np.mean(mask_areas)) if mask_areas else 0,
                "family_guidance": args.family_guidance,
            })

    # Save
    json.dump(stats, open(outdir / "generation_stats.json", "w"), indent=2)
    json.dump(vars(args), open(outdir / "args.json", "w"), indent=2)

    if original_procs:
        restore_processors(unet, original_procs)

    guided = sum(1 for s in stats if s["guided_steps"] > 0)
    print(f"\nDone! {len(stats)} images, guided {guided}/{len(stats)}")
    if family_names:
        print(f"  Families used: {family_names}")


if __name__ == "__main__":
    main()
