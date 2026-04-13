#!/usr/bin/env python
"""
SafeGen: Training-Free Safe Image Generation via Dual-Probe Spatial Guidance.

WHEN: Global noise CAS (Concept Alignment Score) with sticky threshold
WHERE: Dual Cross-Attention Probe (text + image) — zero extra UNet calls
HOW:   Anchor inpaint / hybrid / target subtraction guidance

Usage:
    # Dual probe (text + image):
    python -m safegen.generate \
        --prompts prompts/i2p_sexual.txt \
        --outdir outputs/sexual \
        --probe_mode both \
        --clip_embeddings configs/exemplars/sexual/clip_exemplar_projected.pt \
        --how_mode anchor_inpaint --safety_scale 1.0

    # Text probe only:
    python -m safegen.generate \
        --prompts prompts/i2p_violence.txt \
        --outdir outputs/violence \
        --probe_mode text \
        --target_concepts "violence" "weapon" "blood" \
        --anchor_concepts "peaceful scene" "nature landscape"
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
    register_dual_attention_probe,
    register_attention_probe,
    restore_processors,
    compute_attention_spatial_mask,
    find_token_indices,
)


# ---------------------------------------------------------------------------
# WHEN: Global CAS (Concept Alignment Score)
# ---------------------------------------------------------------------------
class GlobalCAS:
    """
    Detect whether the current denoising step is generating unsafe content
    by measuring alignment between the prompt noise direction and the target
    concept noise direction.

    Args:
        threshold: CAS activation threshold (default 0.6)
        sticky: Once triggered, stay triggered for remaining steps
    """

    def __init__(self, threshold: float = 0.6, sticky: bool = True):
        self.threshold = threshold
        self.sticky = sticky
        self.triggered = False

    def reset(self):
        self.triggered = False

    def compute(self, eps_prompt, eps_null, eps_target):
        """Compute CAS = cos(d_prompt, d_target) where d = eps - eps_null."""
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
# WHERE: Spatial CAS (optional noise-space gate)
# ---------------------------------------------------------------------------
def compute_spatial_cas(eps_prompt, eps_null, eps_target, neighborhood: int = 3):
    """Per-pixel CAS using local neighborhoods for noise-space gating."""
    dp = (eps_prompt - eps_null).float()
    dt = (eps_target - eps_null).float()
    H, W = dp.shape[2], dp.shape[3]
    pad = neighborhood // 2
    pu = F.unfold(dp, neighborhood, padding=pad)
    tu = F.unfold(dt, neighborhood, padding=pad)
    return F.cosine_similarity(pu, tu, dim=1).reshape(H, W)


# ---------------------------------------------------------------------------
# WHERE: Mask utilities
# ---------------------------------------------------------------------------
def gaussian_blur_2d(x, kernel_size: int = 5, sigma: float = 1.0):
    """Apply separable Gaussian blur to a 4D tensor."""
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
    """Convert raw attention spatial map to soft binary mask via sigmoid."""
    m = torch.sigmoid(alpha * (attn_spatial.to(device) - threshold))
    m = m.unsqueeze(0).unsqueeze(0)
    if blur > 0:
        m = gaussian_blur_2d(m, sigma=blur)
    return m.clamp(0, 1)


# ---------------------------------------------------------------------------
# HOW: Guidance modes
# ---------------------------------------------------------------------------
def apply_guidance(
    eps_cfg, eps_null, eps_prompt, eps_target, eps_anchor,
    mask, mode, safety_scale, cfg_scale,
    target_scale=None, anchor_scale=None,
):
    """
    Apply spatial safety guidance to the CFG noise prediction.

    Modes:
        anchor_inpaint: eps_cfg * (1-s*M) + eps_anchor_cfg * (s*M)
        hybrid:         eps_cfg - s*M*(eps_target-eps_null) + s*M*(eps_anchor-eps_null)
        target_sub:     eps_cfg - s*M*(eps_target-eps_null)
    """
    m = mask.to(eps_cfg.dtype)
    ts = target_scale if target_scale is not None else safety_scale
    as_ = anchor_scale if anchor_scale is not None else safety_scale

    if mode == "anchor_inpaint":
        ea_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
        blend = (safety_scale * m).clamp(max=1.0)
        out = eps_cfg * (1 - blend) + ea_cfg * blend

    elif mode == "hybrid":
        out = (eps_cfg
               - ts * m * (eps_target - eps_null)
               + as_ * m * (eps_anchor - eps_null))

    elif mode == "target_sub":
        out = eps_cfg - ts * m * (eps_target - eps_null)

    else:
        raise ValueError(f"Unknown guidance mode: {mode}")

    if torch.isnan(out).any() or torch.isinf(out).any():
        out = torch.where(torch.isfinite(out), out, eps_cfg)
    return out


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_prompts(filepath):
    """Load prompts from .txt or .csv file."""
    fp = Path(filepath)
    if fp.suffix == ".csv":
        prompts = []
        with open(fp) as f:
            reader = csv.DictReader(f)
            col = next(
                (c for c in [
                    "sensitive prompt", "adv_prompt", "prompt",
                    "target_prompt", "text", "Prompt", "Text",
                ] if c in reader.fieldnames),
                None,
            )
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
    """Encode concept strings and return averaged text embeddings."""
    embeds = []
    for c in concepts:
        inp = tokenizer(
            c, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        embeds.append(text_encoder(inp.input_ids.to(device))[0])
    return torch.stack(embeds).mean(0)


def build_image_probe_embeds(clip_features, text_encoder, tokenizer, device, n_tokens=4):
    """
    Build [1, 77, 768] probe embedding from CLIP CLS features.
    Image features placed at token positions 1..n_tokens, with proper
    BOS/EOS/PAD from the text encoder's empty-string baseline.
    """
    avg = F.normalize(clip_features.mean(dim=0), dim=-1)
    dtype = next(text_encoder.parameters()).dtype
    with torch.no_grad():
        empty_ids = tokenizer(
            "", padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        baseline = text_encoder(empty_ids)[0]
    result = baseline.clone()
    concept = avg.to(device=device, dtype=dtype)
    for i in range(1, 1 + n_tokens):
        result[0, i] = concept
    return result, list(range(1, 1 + n_tokens))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = ArgumentParser(description="SafeGen: Dual-Probe Safe Generation")
    p.add_argument("--ckpt", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--prompts", required=True, help="Path to prompt .txt or .csv")
    p.add_argument("--outdir", required=True, help="Output directory for images")
    p.add_argument("--nsamples", type=int, default=1, help="Samples per prompt")
    p.add_argument("--steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg_scale", type=float, default=7.5)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)

    # WHEN
    p.add_argument("--cas_threshold", type=float, default=0.6,
                   help="CAS trigger threshold (default 0.6)")

    # WHERE — Probe
    p.add_argument("--probe_mode", default="both", choices=["text", "image", "both"],
                   help="Probe type: text, image, or both (dual)")
    p.add_argument("--clip_embeddings", default=None,
                   help="Path to CLIP .pt file for image probe")
    p.add_argument("--attn_resolutions", type=int, nargs="+", default=[16, 32],
                   help="UNet resolutions to probe")
    p.add_argument("--attn_threshold", type=float, default=0.3,
                   help="Sigmoid threshold for text probe mask")
    p.add_argument("--img_attn_threshold", type=float, default=None,
                   help="Separate threshold for image probe (default: same as attn_threshold)")
    p.add_argument("--attn_sigmoid_alpha", type=float, default=10.0)
    p.add_argument("--blur_sigma", type=float, default=1.0)
    p.add_argument("--probe_fusion", default="union",
                   choices=["union", "soft_union", "mean"])
    p.add_argument("--target_words", nargs="+", default=None,
                   help="Keywords for text probe token matching")
    p.add_argument("--n_img_tokens", type=int, default=4)

    # WHERE — Noise CAS gate
    p.add_argument("--noise_gate", action="store_true", default=False,
                   help="Use noise CAS as conservative gate on probe mask")
    p.add_argument("--noise_gate_threshold", type=float, default=0.1)
    p.add_argument("--neighborhood_size", type=int, default=3)

    # HOW
    p.add_argument("--how_mode", default="anchor_inpaint",
                   choices=["anchor_inpaint", "hybrid", "target_sub"])
    p.add_argument("--safety_scale", type=float, default=1.0)
    p.add_argument("--target_scale", type=float, default=None)
    p.add_argument("--anchor_scale", type=float, default=None)

    # Concepts
    p.add_argument("--target_concepts", nargs="+",
                   default=["nudity", "nude person", "naked body"])
    p.add_argument("--anchor_concepts", nargs="+",
                   default=["clothed person", "person wearing clothes"])

    p.add_argument("--save_maps", action="store_true",
                   help="Save intermediate probe masks as images")

    args = p.parse_args()

    if args.img_attn_threshold is None:
        args.img_attn_threshold = args.attn_threshold
    if args.target_scale is None:
        args.target_scale = args.safety_scale
    if args.anchor_scale is None:
        args.anchor_scale = args.safety_scale
    if args.probe_mode in ("image", "both") and args.clip_embeddings is None:
        p.error("--clip_embeddings required for image/both probe mode")

    # Auto-extract target_words from target_concepts
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
# Main generation loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_img = args.probe_mode in ("image", "both")
    use_txt = args.probe_mode in ("text", "both")

    print(f"\n{'=' * 60}")
    print(f"SafeGen: Dual-Probe Safe Generation")
    print(f"{'=' * 60}")
    print(f"  WHEN:  CAS threshold={args.cas_threshold}")
    print(f"  WHERE: probe={args.probe_mode}, fusion={args.probe_fusion}, "
          f"res={args.attn_resolutions}")
    print(f"  HOW:   {args.how_mode}, safety_scale={args.safety_scale}")
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
    unet = pipe.unet
    vae, tok, te, sched = pipe.vae, pipe.tokenizer, pipe.text_encoder, pipe.scheduler

    # Encode concept embeddings
    with torch.no_grad():
        text_tgt = encode_concepts(te, tok, args.target_concepts, device)
        anchor_emb = encode_concepts(te, tok, args.anchor_concepts, device)
        unc = te(tok(
            "", padding="max_length", max_length=tok.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device))[0]

    # Setup probes
    img_probe = txt_probe = None
    img_tok_idx = txt_tok_idx = None
    original_procs = None
    unet_dtype = next(unet.parameters()).dtype

    if args.probe_mode == "both":
        img_probe = AttentionProbeStore()
        txt_probe = AttentionProbeStore()

        clip_data = torch.load(args.clip_embeddings, map_location="cpu")
        clip_feats = clip_data.get("target_clip_features", clip_data.get("target_cls")).float()
        img_embeds, img_tok_idx = build_image_probe_embeds(clip_feats, te, tok, device, args.n_img_tokens)
        img_keys = precompute_target_keys(unet, img_embeds.to(dtype=unet_dtype), args.attn_resolutions)
        txt_keys = precompute_target_keys(unet, text_tgt.to(dtype=unet_dtype), args.attn_resolutions)

        target_text = ", ".join(args.target_concepts)
        txt_tok_idx = find_token_indices(target_text, args.target_words, tok)

        original_procs = register_dual_attention_probe(
            unet, img_probe, txt_probe, img_keys, txt_keys, args.attn_resolutions)

    elif args.probe_mode == "image":
        img_probe = AttentionProbeStore()
        clip_data = torch.load(args.clip_embeddings, map_location="cpu")
        clip_feats = clip_data.get("target_clip_features", clip_data.get("target_cls")).float()
        img_embeds, img_tok_idx = build_image_probe_embeds(clip_feats, te, tok, device, args.n_img_tokens)
        img_keys = precompute_target_keys(unet, img_embeds.to(dtype=unet_dtype), args.attn_resolutions)
        original_procs = register_attention_probe(unet, img_probe, img_keys, args.attn_resolutions)

    elif args.probe_mode == "text":
        txt_probe = AttentionProbeStore()
        txt_keys = precompute_target_keys(unet, text_tgt.to(dtype=unet_dtype), args.attn_resolutions)
        target_text = ", ".join(args.target_concepts)
        txt_tok_idx = find_token_indices(target_text, args.target_words, tok)
        original_procs = register_attention_probe(unet, txt_probe, txt_keys, args.attn_resolutions)

    # Generate
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
            guided_count, cas_vals, mask_areas = 0, [], []

            with torch.no_grad():
                pemb = te(tok(
                    prompt, padding="max_length", max_length=tok.model_max_length,
                    truncation=True, return_tensors="pt",
                ).input_ids.to(device))[0]

            set_seed(seed)
            lat = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
            lat = lat * sched.init_noise_sigma
            sched.set_timesteps(args.steps, device=device)

            for step_i, t in enumerate(sched.timesteps):
                li = sched.scale_model_input(lat, t)

                # Forward pass with probe active
                if img_probe:
                    img_probe.active = True
                    img_probe.reset()
                if txt_probe:
                    txt_probe.active = True
                    txt_probe.reset()

                with torch.no_grad():
                    raw = unet(
                        torch.cat([li, li]), t,
                        encoder_hidden_states=torch.cat([unc, pemb]),
                    ).sample
                    en, ep = raw.chunk(2)

                if img_probe:
                    img_probe.active = False
                if txt_probe:
                    txt_probe.active = False

                # Target prediction (for CAS + guidance)
                with torch.no_grad():
                    et = unet(li, t, encoder_hidden_states=text_tgt).sample

                ec = en + args.cfg_scale * (ep - en)
                cv, trig = cas.compute(ep, en, et)
                cas_vals.append(cv)

                if trig:
                    with torch.no_grad():
                        ea = unet(li, t, encoder_hidden_states=anchor_emb).sample

                    # Compute probe masks
                    img_mask = txt_mask = None

                    if img_probe and img_probe.get_maps():
                        img_attn = compute_attention_spatial_mask(
                            img_probe, token_indices=img_tok_idx,
                            target_resolution=64, resolutions_to_use=args.attn_resolutions)
                        img_mask = make_probe_mask(
                            img_attn, args.img_attn_threshold,
                            args.attn_sigmoid_alpha, args.blur_sigma, device)

                    if txt_probe and txt_probe.get_maps() and txt_tok_idx:
                        txt_attn = compute_attention_spatial_mask(
                            txt_probe, token_indices=txt_tok_idx,
                            target_resolution=64, resolutions_to_use=args.attn_resolutions)
                        txt_mask = make_probe_mask(
                            txt_attn, args.attn_threshold,
                            args.attn_sigmoid_alpha, args.blur_sigma, device)
                    elif txt_probe and txt_probe.get_maps():
                        txt_attn = compute_attention_spatial_mask(
                            txt_probe, token_indices=None,
                            target_resolution=64, resolutions_to_use=args.attn_resolutions)
                        txt_mask = make_probe_mask(
                            txt_attn, args.attn_threshold,
                            args.attn_sigmoid_alpha, args.blur_sigma, device)

                    # Fuse masks
                    if args.probe_mode == "both" and img_mask is not None and txt_mask is not None:
                        if args.probe_fusion == "union":
                            probe_mask = torch.max(img_mask, txt_mask)
                        elif args.probe_fusion == "soft_union":
                            probe_mask = 1 - (1 - img_mask) * (1 - txt_mask)
                        elif args.probe_fusion == "mean":
                            probe_mask = (img_mask + txt_mask) / 2
                    elif img_mask is not None:
                        probe_mask = img_mask
                    elif txt_mask is not None:
                        probe_mask = txt_mask
                    else:
                        probe_mask = torch.ones(1, 1, 64, 64, device=device) * 0.5

                    # Optional noise CAS gate
                    if args.noise_gate:
                        noise_cas = compute_spatial_cas(ep, en, et, args.neighborhood_size)
                        gate = (noise_cas.to(device) > args.noise_gate_threshold).float()
                        final_mask = probe_mask * gate.unsqueeze(0).unsqueeze(0)
                    else:
                        final_mask = probe_mask

                    # Apply guidance
                    eps_final = apply_guidance(
                        ec, en, ep, et, ea, final_mask,
                        args.how_mode, args.safety_scale, args.cfg_scale,
                        target_scale=args.target_scale,
                        anchor_scale=args.anchor_scale)

                    guided_count += 1
                    mask_areas.append(float(final_mask.mean()))

                    if args.save_maps and step_i % 10 == 0:
                        md = outdir / "maps"
                        pf = f"{pi:04d}_{si:02d}_s{step_i:03d}"
                        for name, mm in [("img", img_mask), ("txt", txt_mask), ("final", final_mask)]:
                            if mm is not None:
                                mn = mm[0, 0].float().cpu().numpy()
                                Image.fromarray(
                                    (np.clip(mn, 0, 1) * 255).astype(np.uint8), "L"
                                ).save(str(md / f"{pf}_{name}.png"))
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
            })

    # Save metadata
    json.dump(stats, open(outdir / "generation_stats.json", "w"), indent=2)
    json.dump(vars(args), open(outdir / "args.json", "w"), indent=2)

    if original_procs:
        restore_processors(unet, original_procs)

    guided = sum(1 for s in stats if s["guided_steps"] > 0)
    print(f"\nDone! {len(stats)} images, guided {guided}/{len(stats)}")


if __name__ == "__main__":
    main()
