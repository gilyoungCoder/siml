#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v27: Dual-Probe Safe Generation — Text + Image probes in attention space.

ML-researcher-validated architecture:
  WHEN: Global noise CAS (text-based, threshold=0.6, sticky)
  WHERE (Dual Cross-Attention Probe):
    - Text probe:  K_text = W_K · encode("nudity") → text attention map
    - Image probe: K_img = W_K · CLIP(exemplar images) → image attention map
    - Fusion: union/soft_union of text and image maps
    - Optional noise CAS gate (conservative, removes false positives)
    - Cost: ZERO extra UNet calls (hooks on existing forward pass)
  HOW (3 modes):
    - anchor_inpaint: eps_cfg*(1-s*M) + eps_anchor_cfg*(s*M)
    - hybrid: eps_cfg - s*M*(eps_target-eps_null) + s*M*(eps_anchor-eps_null)
    - target_sub: eps_cfg - s*M*(eps_target-eps_null)

Why probe > UNet conditioning:
  - Probe measures the CAUSAL mechanism (where UNet attends to concept)
  - UNet conditioning measures downstream effect (noise direction)
  - Probe is norm-insensitive (softmax normalizes)
  - Zero extra UNet calls vs +1 per step
  - Both text and image probes live in same attention space → clean fusion

Total UNet calls: 3 (null+prompt, target) or 4 (+anchor when triggered)
Same as v4 — probe is FREE.
"""

import os, sys, json, math, random, csv
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler

from attention_probe import (
    AttentionProbeStore,
    DualProbeCrossAttnProcessor,
    precompute_target_keys,
    register_dual_attention_probe,
    register_attention_probe,
    restore_processors,
    compute_attention_spatial_mask,
    find_token_indices,
)


# ── WHEN: Global CAS ──
class GlobalCAS:
    def __init__(self, threshold=0.6, sticky=True):
        self.threshold, self.sticky, self.triggered = threshold, sticky, False
    def reset(self): self.triggered = False
    def compute(self, ep, en, et):
        dp = (ep-en).reshape(1,-1).float(); dt = (et-en).reshape(1,-1).float()
        c = F.cosine_similarity(dp, dt, dim=-1).item()
        if math.isnan(c) or math.isinf(c): return 0., self.triggered if self.sticky else False
        if self.sticky and self.triggered: return c, True
        if c > self.threshold:
            if self.sticky: self.triggered = True
            return c, True
        return c, False


# ── WHERE: Spatial CAS (for noise gate) ──
def compute_spatial_cas(ep, en, et, nbr=3):
    dp=(ep-en).float(); dt=(et-en).float()
    H,W=dp.shape[2],dp.shape[3]; p=nbr//2
    pu=F.unfold(dp,nbr,padding=p); tu=F.unfold(dt,nbr,padding=p)
    return F.cosine_similarity(pu,tu,dim=1).reshape(H,W)


# ── Mask utilities ──
def gaussian_blur_2d(x, ks=5, sigma=1.):
    co=torch.arange(ks,dtype=x.dtype,device=x.device)-ks//2
    g=torch.exp(-.5*(co/sigma)**2); g/=g.sum()
    kh,kw=g.view(1,1,ks,1),g.view(1,1,1,ks); p=ks//2
    x=F.pad(x,[0,0,p,p],'reflect'); x=F.conv2d(x,kh.expand(x.shape[1],-1,-1,-1),groups=x.shape[1])
    x=F.pad(x,[p,p,0,0],'reflect'); return F.conv2d(x,kw.expand(x.shape[1],-1,-1,-1),groups=x.shape[1])

def make_probe_mask(attn_spatial, threshold, alpha=10., blur=1., device=None):
    m = torch.sigmoid(alpha * (attn_spatial.to(device) - threshold))
    m = m.unsqueeze(0).unsqueeze(0)
    if blur > 0: m = gaussian_blur_2d(m, sigma=blur)
    return m.clamp(0, 1)


# ── HOW: Guidance ──
def apply_guidance(eps_cfg, eps_null, eps_prompt, eps_target, eps_anchor,
                   mask, how, safety_scale, cfg_scale,
                   target_scale=None, anchor_scale=None, proj_scale=1.0):
    m = mask.to(eps_cfg.dtype)
    ts = target_scale if target_scale is not None else safety_scale
    as_ = anchor_scale if anchor_scale is not None else safety_scale

    if how == "anchor_inpaint":
        ea_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
        blend = (safety_scale * m).clamp(max=1.0)
        out = eps_cfg * (1 - blend) + ea_cfg * blend

    elif how == "hybrid":
        out = eps_cfg \
              - ts * m * (eps_target - eps_null) \
              + as_ * m * (eps_anchor - eps_null)

    elif how == "hybrid_proj":
        # Step 1: Project out nudity from prompt direction
        d_prompt = eps_prompt - eps_null
        d_target = eps_target - eps_null
        dot = (d_prompt * d_target).sum(dim=1, keepdim=True)
        norm_sq = (d_target * d_target).sum(dim=1, keepdim=True).clamp(min=1e-8)
        proj = (dot / norm_sq) * d_target
        d_safe = d_prompt - proj_scale * proj
        eps_safe_cfg = eps_null + cfg_scale * d_safe
        # Step 2: Blend safe CFG with anchor
        ea_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
        blend = (as_ * m).clamp(max=1.0)
        out = eps_safe_cfg * (1 - blend) + ea_cfg * blend

    elif how == "proj_replace":
        # Projection replacement: swap nudity component with anchor component
        # on the nudity axis, preserve everything else
        d_prompt = eps_prompt - eps_null
        d_target = eps_target - eps_null
        d_anchor = eps_anchor - eps_null

        # Per-pixel projection on nudity axis
        # dot and norm along channel dim (dim=1), keep spatial dims
        dot_pt = (d_prompt * d_target).sum(dim=1, keepdim=True)
        dot_tt = (d_target * d_target).sum(dim=1, keepdim=True).clamp(min=1e-8)
        d_target_unit = d_target / dot_tt.sqrt()

        # Prompt's nudity component
        prompt_on_target = (d_prompt * d_target_unit).sum(dim=1, keepdim=True)
        # Anchor's component on nudity axis
        anchor_on_target = (d_anchor * d_target_unit).sum(dim=1, keepdim=True)

        # Replace: remove prompt's nudity, add anchor's clothed
        d_safe = d_prompt \
                 - proj_scale * prompt_on_target * d_target_unit \
                 + as_ * anchor_on_target * d_target_unit

        eps_safe_cfg = eps_null + cfg_scale * d_safe
        out = eps_cfg * (1 - m) + eps_safe_cfg * m

    elif how == "target_sub":
        out = eps_cfg - ts * m * (eps_target - eps_null)

    else:
        raise ValueError(how)

    if torch.isnan(out).any() or torch.isinf(out).any():
        out = torch.where(torch.isfinite(out), out, eps_cfg)
    return out


# ── Utils ──
def load_prompts(fp):
    fp=Path(fp)
    if fp.suffix==".csv":
        ps=[]
        with open(fp) as f:
            r=csv.DictReader(f)
            col=next((c for c in ['sensitive prompt','adv_prompt','prompt','target_prompt','text','Prompt','Text'] if c in r.fieldnames),None)
            if not col: raise ValueError(f"No prompt col in {r.fieldnames}")
            for row in r:
                p=row[col].strip()
                if p: ps.append(p)
        return ps
    return [l.strip() for l in open(fp) if l.strip()]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def encode_concepts(te, tok, concepts, dev):
    es=[]
    for c in concepts:
        inp=tok(c,padding="max_length",max_length=tok.model_max_length,truncation=True,return_tensors="pt")
        es.append(te(inp.input_ids.to(dev))[0])
    return torch.stack(es).mean(0)


# ── Build image probe embedding ──
def build_image_probe_embeds(clip_features, text_encoder, tokenizer, device, n_tokens=4):
    """
    Build [1, 77, 768] probe embedding from CLIP CLS features.
    Uses text encoder empty-string baseline for BOS/EOS/PAD (proper norm ~28).
    Image features placed at token positions 1..n_tokens.

    For PROBE, norm doesn't matter much (softmax normalizes),
    but proper baseline prevents distribution artifacts.
    """
    avg = F.normalize(clip_features.mean(dim=0), dim=-1)  # [768]
    dtype = next(text_encoder.parameters()).dtype

    with torch.no_grad():
        empty_ids = tokenizer("", padding="max_length", max_length=77,
                              truncation=True, return_tensors="pt").input_ids.to(device)
        baseline = text_encoder(empty_ids)[0]  # [1, 77, 768] norm~28

    result = baseline.clone()
    concept = avg.to(device=device, dtype=dtype)
    for i in range(1, 1 + n_tokens):
        result[0, i] = concept

    return result, list(range(1, 1 + n_tokens))


# ── Args ──
def parse_args():
    p = ArgumentParser(description="v27: Dual-Probe (Text+Image) Safe Generation")
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
    p.add_argument("--probe_mode", default="both",
                   choices=["text", "image", "both"],
                   help="text=text probe only, image=image probe only, both=dual probe")
    p.add_argument("--clip_embeddings", default=None,
                   help="CLIP .pt file with target_clip_features/target_clip_embeds")
    p.add_argument("--attn_resolutions", type=int, nargs="+", default=[16, 32])
    p.add_argument("--attn_threshold", type=float, default=0.3,
                   help="Threshold for text probe mask")
    p.add_argument("--img_attn_threshold", type=float, default=None,
                   help="Separate threshold for image probe (default: same as attn_threshold)")
    p.add_argument("--attn_sigmoid_alpha", type=float, default=10.0)
    p.add_argument("--blur_sigma", type=float, default=1.0)
    p.add_argument("--probe_fusion", default="union",
                   choices=["union", "soft_union", "mean"])
    p.add_argument("--target_words", nargs="+", default=None,
                   help="Keywords for text probe token matching. "
                        "If None, auto-extracted from --target_concepts.")
    p.add_argument("--n_img_tokens", type=int, default=4)

    # WHERE — Noise CAS gate
    p.add_argument("--noise_gate", action="store_true", default=False,
                   help="Use noise CAS as conservative gate on probe mask")
    p.add_argument("--noise_gate_threshold", type=float, default=0.1)
    p.add_argument("--neighborhood_size", type=int, default=3)

    # HOW
    p.add_argument("--how_mode", default="anchor_inpaint",
                   choices=["anchor_inpaint", "hybrid", "hybrid_proj", "proj_replace", "target_sub"])
    p.add_argument("--safety_scale", type=float, default=1.0)
    p.add_argument("--target_scale", type=float, default=None,
                   help="Hybrid: target repulsion scale (default: same as safety_scale)")
    p.add_argument("--anchor_scale", type=float, default=None,
                   help="Hybrid: anchor attraction scale (default: same as safety_scale)")
    p.add_argument("--proj_scale", type=float, default=1.0,
                   help="hybrid_proj: projection removal strength")

    # Concepts
    p.add_argument("--target_concepts", nargs="+",
                   default=["nudity", "nude person", "naked body"])
    p.add_argument("--anchor_concepts", nargs="+",
                   default=["clothed person", "person wearing clothes"])

    p.add_argument("--debug", action="store_true")
    p.add_argument("--save_maps", action="store_true")

    a = p.parse_args()
    if a.img_attn_threshold is None:
        a.img_attn_threshold = a.attn_threshold
    if a.target_scale is None:
        a.target_scale = a.safety_scale
    if a.anchor_scale is None:
        a.anchor_scale = a.safety_scale
    if a.probe_mode in ("image", "both") and a.clip_embeddings is None:
        p.error("--clip_embeddings required for image/both probe mode")
    # Auto-extract target_words from target_concepts if not specified
    if a.target_words is None:
        words = []
        for concept in a.target_concepts:
            for w in concept.replace("_", " ").split():
                w_clean = w.strip().lower()
                if len(w_clean) >= 3 and w_clean not in words:
                    words.append(w_clean)
        a.target_words = words
        print(f"  Auto target_words: {a.target_words}")
    return a


# ── Main ──
def main():
    args = parse_args()
    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_img = args.probe_mode in ("image", "both")
    use_txt_probe = args.probe_mode in ("text", "both")

    print(f"\n{'='*70}")
    print(f"v27: Dual-Probe Safe Generation")
    print(f"{'='*70}")
    print(f"  WHEN:  CAS threshold={args.cas_threshold}")
    print(f"  WHERE: probe_mode={args.probe_mode}, fusion={args.probe_fusion}, "
          f"res={args.attn_resolutions}")
    print(f"         attn_thr={args.attn_threshold}, noise_gate={args.noise_gate}")
    print(f"  HOW:   {args.how_mode}, ss={args.safety_scale}")
    print(f"{'='*70}\n")

    prompts = load_prompts(args.prompts)
    end = args.end_idx if args.end_idx > 0 else len(prompts)
    pw = list(enumerate(prompts))[args.start_idx:end]

    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, safety_checker=None).to(dev)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.feature_extractor = None
    unet, vae, tok, te, sched = pipe.unet, pipe.vae, pipe.tokenizer, pipe.text_encoder, pipe.scheduler

    with torch.no_grad():
        text_tgt = encode_concepts(te, tok, args.target_concepts, dev)
        anchor_emb = encode_concepts(te, tok, args.anchor_concepts, dev)
        unc = te(tok("", padding="max_length", max_length=tok.model_max_length,
                     truncation=True, return_tensors="pt").input_ids.to(dev))[0]

    # ── Setup probes ──
    img_probe_store = None
    txt_probe_store = None
    img_token_indices = None
    txt_token_indices = None
    original_processors = None

    unet_dtype = next(unet.parameters()).dtype

    if args.probe_mode == "both":
        # Dual probe: image + text in one forward pass
        img_probe_store = AttentionProbeStore()
        txt_probe_store = AttentionProbeStore()

        # Image probe embedding
        clip_data = torch.load(args.clip_embeddings, map_location="cpu")
        if "target_clip_features" in clip_data:
            clip_features = clip_data["target_clip_features"].float()
        else:
            clip_features = clip_data["target_cls"].float()

        img_embeds, img_token_indices = build_image_probe_embeds(
            clip_features, te, tok, dev, n_tokens=args.n_img_tokens)
        img_keys = precompute_target_keys(
            unet, img_embeds.to(dtype=unet_dtype), args.attn_resolutions)

        # Text probe embedding
        txt_keys = precompute_target_keys(
            unet, text_tgt.to(dtype=unet_dtype), args.attn_resolutions)

        target_text = ", ".join(args.target_concepts)
        txt_token_indices = find_token_indices(target_text, args.target_words, tok)

        # Register dual probe (both in one forward pass)
        original_processors = register_dual_attention_probe(
            unet, img_probe_store, txt_probe_store,
            img_keys, txt_keys, args.attn_resolutions)

        print(f"  Dual probe: img_tokens={img_token_indices}, txt_tokens={txt_token_indices}")

    elif args.probe_mode == "image":
        img_probe_store = AttentionProbeStore()
        clip_data = torch.load(args.clip_embeddings, map_location="cpu")
        clip_features = clip_data.get("target_clip_features",
                                       clip_data.get("target_cls")).float()
        img_embeds, img_token_indices = build_image_probe_embeds(
            clip_features, te, tok, dev, n_tokens=args.n_img_tokens)
        img_keys = precompute_target_keys(
            unet, img_embeds.to(dtype=unet_dtype), args.attn_resolutions)
        original_processors = register_attention_probe(
            unet, img_probe_store, img_keys, args.attn_resolutions)
        print(f"  Image probe: tokens={img_token_indices}")

    elif args.probe_mode == "text":
        txt_probe_store = AttentionProbeStore()
        txt_keys = precompute_target_keys(
            unet, text_tgt.to(dtype=unet_dtype), args.attn_resolutions)
        target_text = ", ".join(args.target_concepts)
        txt_token_indices = find_token_indices(target_text, args.target_words, tok)
        original_processors = register_attention_probe(
            unet, txt_probe_store, txt_keys, args.attn_resolutions)
        print(f"  Text probe: tokens={txt_token_indices}")

    cas = GlobalCAS(args.cas_threshold)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if args.save_maps: (outdir / "maps").mkdir(exist_ok=True)
    stats = []

    for pi, prompt in tqdm(pw, desc="Generating"):
        if not prompt.strip(): continue
        for si in range(args.nsamples):
            seed = args.seed + pi * args.nsamples + si
            set_seed(seed); cas.reset()
            gc, cvs, mas = 0, [], []

            with torch.no_grad():
                pemb = te(tok(prompt, padding="max_length", max_length=tok.model_max_length,
                              truncation=True, return_tensors="pt").input_ids.to(dev))[0]

            set_seed(seed)
            lat = torch.randn(1, 4, 64, 64, device=dev, dtype=torch.float16)
            lat = lat * sched.init_noise_sigma
            sched.set_timesteps(args.steps, device=dev)

            for si_t, t in enumerate(sched.timesteps):
                li = sched.scale_model_input(lat, t)

                # ── UNet forward with probe active ──
                if img_probe_store: img_probe_store.active = True; img_probe_store.reset()
                if txt_probe_store: txt_probe_store.active = True; txt_probe_store.reset()

                with torch.no_grad():
                    raw = unet(torch.cat([li, li]), t,
                               encoder_hidden_states=torch.cat([unc, pemb])).sample
                    en, ep = raw.chunk(2)

                if img_probe_store: img_probe_store.active = False
                if txt_probe_store: txt_probe_store.active = False

                # Target (for WHEN CAS + HOW guidance)
                with torch.no_grad():
                    et = unet(li, t, encoder_hidden_states=text_tgt).sample

                ec = en + args.cfg_scale * (ep - en)
                cv, trig = cas.compute(ep, en, et)
                cvs.append(cv)

                if trig:
                    with torch.no_grad():
                        ea = unet(li, t, encoder_hidden_states=anchor_emb).sample

                    # ── WHERE: Compute probe masks ──
                    img_mask = None
                    txt_mask = None

                    if img_probe_store and img_probe_store.get_maps():
                        img_attn = compute_attention_spatial_mask(
                            img_probe_store, token_indices=img_token_indices,
                            target_resolution=64, resolutions_to_use=args.attn_resolutions)
                        img_mask = make_probe_mask(
                            img_attn, args.img_attn_threshold,
                            args.attn_sigmoid_alpha, args.blur_sigma, dev)

                    if txt_probe_store and txt_probe_store.get_maps() and txt_token_indices:
                        txt_attn = compute_attention_spatial_mask(
                            txt_probe_store, token_indices=txt_token_indices,
                            target_resolution=64, resolutions_to_use=args.attn_resolutions)
                        txt_mask = make_probe_mask(
                            txt_attn, args.attn_threshold,
                            args.attn_sigmoid_alpha, args.blur_sigma, dev)
                    elif txt_probe_store and txt_probe_store.get_maps():
                        # Fallback: use all non-special tokens if no keyword match
                        txt_attn = compute_attention_spatial_mask(
                            txt_probe_store, token_indices=None,
                            target_resolution=64, resolutions_to_use=args.attn_resolutions)
                        txt_mask = make_probe_mask(
                            txt_attn, args.attn_threshold,
                            args.attn_sigmoid_alpha, args.blur_sigma, dev)

                    # ── Fuse probe masks ──
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
                        probe_mask = torch.ones(1, 1, 64, 64, device=dev) * 0.5

                    # ── Optional noise CAS gate ──
                    if args.noise_gate:
                        noise_cas = compute_spatial_cas(
                            ep, en, et, nbr=args.neighborhood_size)
                        gate = (noise_cas.to(dev) > args.noise_gate_threshold).float()
                        gate = gate.unsqueeze(0).unsqueeze(0)
                        final_mask = probe_mask * gate
                    else:
                        final_mask = probe_mask

                    # ── HOW: Apply guidance ──
                    eps_final = apply_guidance(
                        ec, en, ep, et, ea, final_mask,
                        args.how_mode, args.safety_scale, args.cfg_scale,
                        target_scale=args.target_scale,
                        anchor_scale=args.anchor_scale,
                        proj_scale=args.proj_scale)

                    gc += 1
                    mas.append(float(final_mask.mean()))

                    if args.debug and si_t % 10 == 0:
                        im = float(img_mask.mean()) if img_mask is not None else 0
                        tm = float(txt_mask.mean()) if txt_mask is not None else 0
                        fm = mas[-1]
                        print(f"  [{si_t:02d}] CAS={cv:.3f} img={im:.3f} txt={tm:.3f} final={fm:.3f}")

                    if args.save_maps and si_t % 10 == 0:
                        md = outdir / "maps"
                        pf = f"{pi:04d}_{si:02d}_s{si_t:03d}"
                        for name, mm in [("img", img_mask), ("txt", txt_mask), ("final", final_mask)]:
                            if mm is not None:
                                mn = mm[0, 0].float().cpu().numpy()
                                Image.fromarray((np.clip(mn, 0, 1) * 255).astype(np.uint8), 'L').save(
                                    str(md / f"{pf}_{name}.png"))
                else:
                    eps_final = ec

                lp = lat.clone()
                lat = sched.step(eps_final, t, lat).prev_sample
                if torch.isnan(lat).any() or torch.isinf(lat).any():
                    lat = sched.step(en + args.cfg_scale * (ep - en), t, lp).prev_sample

            with torch.no_grad():
                dec = vae.decode(lat.to(vae.dtype) / vae.config.scaling_factor).sample
                dec = (dec / 2 + 0.5).clamp(0, 1)
                img = (dec[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

            fn = f"{pi:04d}_{si:02d}.png"
            Image.fromarray(img).resize((512, 512)).save(str(outdir / fn))
            stats.append({"pi": pi, "si": si, "seed": seed, "guided": gc,
                          "max_cas": max(cvs) if cvs else 0,
                          "mean_area": float(np.mean(mas)) if mas else 0})

    json.dump(stats, open(outdir / "generation_stats.json", "w"), indent=2)
    json.dump(vars(args), open(outdir / "args.json", "w"), indent=2)

    if original_processors:
        restore_processors(unet, original_processors)

    gi = sum(1 for s in stats if s["guided"] > 0)
    print(f"\nDone! {len(stats)} imgs, guided {gi}/{len(stats)}")


if __name__ == "__main__":
    main()
