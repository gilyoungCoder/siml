#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v24: Comprehensive Dual-Direction Safe Generation

All axes of variation in one script:

WHERE masking (--where_mode):
  noise:  pixel-level spatial CAS with UNet noise predictions
  attn:   cross-attention map from UNet forward pass

Example source (--example_mode):
  text:   text concept embeddings only (=v4)
  img:    CLIP image exemplar embeddings only
  both:   text + image, combined via --fusion

Image pooling (--img_pool):
  cls_mean:   mean of K exemplar CLS features → 1 token repeated N times
  cls_multi:  each exemplar CLS as separate token (union via max-across-tokens)
  patch_mean: mean of discriminative patch tokens (if available)

Fusion for 'both' mode (--fusion):
  union:      max(text_mask, img_mask)
  soft_union: 1 - (1-text)*(1-img)
  mean:       (text + img) / 2
  multiply:   text * img (intersection)

HOW guidance (--how_mode):
  anchor_inpaint:  eps_cfg*(1-s*M) + eps_anchor_cfg*(s*M)
  hybrid:          eps_cfg - s*M*(eps_target-eps_null) + s*M*(eps_anchor-eps_null)
  hybrid_cfg:      eps_cfg - s*M*(eps_target_cfg-eps_cfg) + a*M*(eps_anchor_cfg-eps_cfg)
  target_sub:      eps_cfg - s*M*(eps_target_cfg-eps_null)

WHEN: Noise CAS (text-based, threshold=0.6, sticky) — always fixed.
"""

import os, sys, json, math, random, csv
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler

# Block resolution map for SD1.4
_BLOCK_RES = {
    "down_blocks.0": 64, "down_blocks.1": 32, "down_blocks.2": 16,
    "mid_block": 8, "up_blocks.1": 16, "up_blocks.2": 32, "up_blocks.3": 64,
}
def _get_res(name):
    for prefix, r in _BLOCK_RES.items():
        if name.startswith(prefix): return r
    return 0

# ── Global CAS (WHEN) ──────────────────────────────────────────────
class GlobalCAS:
    def __init__(self, threshold=0.6, sticky=True):
        self.threshold, self.sticky, self.triggered = threshold, sticky, False
    def reset(self): self.triggered = False
    def compute(self, eps_prompt, eps_null, eps_target):
        d_p = (eps_prompt - eps_null).reshape(1,-1).float()
        d_t = (eps_target - eps_null).reshape(1,-1).float()
        cas = F.cosine_similarity(d_p, d_t, dim=-1).item()
        if math.isnan(cas) or math.isinf(cas):
            return 0.0, self.triggered if self.sticky else False
        if self.sticky and self.triggered: return cas, True
        if cas > self.threshold:
            if self.sticky: self.triggered = True
            return cas, True
        return cas, False

# ── Spatial CAS (noise WHERE) ──────────────────────────────────────
def compute_spatial_cas(eps_prompt, eps_null, eps_target, nbr=3):
    d_p = (eps_prompt - eps_null).float()
    d_t = (eps_target - eps_null).float()
    H, W = d_p.shape[2], d_p.shape[3]
    pad = nbr // 2
    pu = F.unfold(d_p, kernel_size=nbr, padding=pad)
    tu = F.unfold(d_t, kernel_size=nbr, padding=pad)
    return F.cosine_similarity(pu, tu, dim=1).reshape(H, W)

# ── Attention map capture (attn WHERE) ─────────────────────────────
def capture_attn_map(unet, embeds, lat_in, t, resolutions=[16,32]):
    maps = {}; hooks = []
    def make_hook(name):
        def fn(mod, inp, out):
            if len(inp) < 2 or inp[1] is None: return
            h, enc = inp[0], inp[1]
            B = h.shape[0]
            q = mod.to_q(h); k = mod.to_k(enc)
            dim = k.shape[-1]; hd = dim // mod.heads
            q = q.view(B,-1,mod.heads,hd).transpose(1,2)
            k = k.view(B,-1,mod.heads,hd).transpose(1,2)
            a = torch.matmul(q*(hd**-0.5), k.transpose(-2,-1)).softmax(-1)
            maps[name] = a.mean(1).mean(-1).detach()  # [B, spatial]
        return fn
    for name, mod in unet.named_modules():
        if name.endswith(".attn2") and hasattr(mod,'to_q') and _get_res(name) in resolutions:
            hooks.append(mod.register_forward_hook(make_hook(name)))
    with torch.no_grad(): unet(lat_in, t, encoder_hidden_states=embeds)
    for h in hooks: h.remove()
    groups = {}
    for name, a in maps.items():
        r = _get_res(name); n = int(a.shape[-1]**0.5)
        s = a[0].view(1,1,n,n)
        groups.setdefault(r,[]).append(s)
    if not groups: return torch.zeros(64,64)
    ups = [F.interpolate(torch.stack(ms).mean(0).float(), (64,64), mode='bilinear', align_corners=False)
           for ms in groups.values()]
    c = torch.stack(ups).mean(0).squeeze()
    mn, mx = c.min(), c.max()
    return (c - mn) / (mx - mn + 1e-8)

# ── Soft mask utilities ────────────────────────────────────────────
def gaussian_blur_2d(x, ks=5, sigma=1.0):
    co = torch.arange(ks, dtype=x.dtype, device=x.device) - ks//2
    g = torch.exp(-0.5*(co/sigma)**2); g = g/g.sum()
    kh, kw = g.view(1,1,ks,1), g.view(1,1,1,ks); p = ks//2
    x = F.pad(x,[0,0,p,p],'reflect')
    x = F.conv2d(x, kh.expand(x.shape[1],-1,-1,-1), groups=x.shape[1])
    x = F.pad(x,[p,p,0,0],'reflect')
    return F.conv2d(x, kw.expand(x.shape[1],-1,-1,-1), groups=x.shape[1])

def soft_mask(raw, thr, alpha=10.0, blur=1.0):
    m = torch.sigmoid(alpha * (raw - thr)).unsqueeze(0).unsqueeze(0)
    if blur > 0: m = gaussian_blur_2d(m, sigma=blur)
    return m.clamp(0,1)

def fuse(m1, m2, mode):
    if m2 is None or mode == "text_only": return m1
    if mode == "img_only": return m2
    if m2.shape != m1.shape:
        m2 = F.interpolate(m2, m1.shape[-2:], mode='bilinear', align_corners=False)
    if mode == "union": return torch.max(m1, m2)
    if mode == "soft_union": return 1-(1-m1)*(1-m2)
    if mode == "mean": return (m1+m2)/2
    if mode == "multiply": return (m1*m2).clamp(0,1)
    raise ValueError(mode)

# ── Image embedding builders ──────────────────────────────────────
def build_img_embeds(clip_data, text_encoder, tokenizer, device, pool="cls_mean", max_tok=16):
    te = text_encoder.text_model.embeddings.token_embedding
    bos = te(torch.tensor([tokenizer.bos_token_id], device=device))
    eos = te(torch.tensor([tokenizer.eos_token_id], device=device))
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad = te(torch.tensor([pad_id], device=device))

    if pool == "cls_mean":
        feats = clip_data["target_clip_features"].to(device).float()
        avg = F.normalize(feats.mean(0, keepdim=True), dim=-1)
        toks = [bos] + [avg]*4 + [eos] + [pad]*(77-6)
    elif pool == "cls_multi":
        feats = F.normalize(clip_data["target_clip_features"].to(device).float(), dim=-1)
        K = min(feats.shape[0], max_tok, 75)
        toks = [bos] + [feats[i:i+1] for i in range(K)] + [eos] + [pad]*(77-K-2)
    elif pool == "patch_mean":
        if "target_patches" in clip_data:
            patches = F.normalize(clip_data["target_patches"].to(device).float(), dim=-1)
        else:  # fallback to cls_mean
            feats = clip_data["target_clip_features"].to(device).float()
            patches = F.normalize(feats.mean(0, keepdim=True), dim=-1)
        K = min(patches.shape[0], 16, 75)
        toks = [bos] + [patches[i:i+1] for i in range(K)] + [eos] + [pad]*(77-K-2)
    else:
        raise ValueError(pool)

    return torch.cat(toks, dim=0).unsqueeze(0)  # [1, 77, 768]

# ── Guidance (HOW) ─────────────────────────────────────────────────
def apply_guidance(eps_cfg, eps_null, eps_prompt, eps_target, eps_anchor,
                   mask, how, ss, cfg_scale, anchor_scale=None):
    m = mask.to(eps_cfg.dtype)
    a_ss = anchor_scale if anchor_scale is not None else ss

    if how == "anchor_inpaint":
        ea_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
        blend = (ss * m).clamp(max=1.0)
        out = eps_cfg * (1 - blend) + ea_cfg * blend

    elif how == "hybrid":
        # Subtract target + add anchor (raw directions)
        out = eps_cfg - ss*m*(eps_target - eps_null) + a_ss*m*(eps_anchor - eps_null)

    elif how == "hybrid_cfg":
        # Operate in CFG space: subtract target_cfg deviation, add anchor_cfg deviation
        et_cfg = eps_null + cfg_scale * (eps_target - eps_null)
        ea_cfg = eps_null + cfg_scale * (eps_anchor - eps_null)
        out = eps_cfg - ss*m*(et_cfg - eps_cfg) + a_ss*m*(ea_cfg - eps_cfg)

    elif how == "target_sub":
        et_cfg = eps_null + cfg_scale * (eps_target - eps_null)
        out = eps_cfg - ss*m*(et_cfg - eps_null)

    else:
        raise ValueError(how)

    if torch.isnan(out).any() or torch.isinf(out).any():
        out = torch.where(torch.isfinite(out), out, eps_cfg)
    return out

# ── Utils ──────────────────────────────────────────────────────────
def load_prompts(fp):
    fp = Path(fp)
    if fp.suffix == ".csv":
        ps = []
        with open(fp) as f:
            r = csv.DictReader(f)
            col = next((c for c in ['sensitive prompt','adv_prompt','prompt','target_prompt',
                                    'text','Prompt','Text'] if c in r.fieldnames), None)
            if not col: raise ValueError(f"No prompt col in {r.fieldnames}")
            for row in r:
                p = row[col].strip()
                if p: ps.append(p)
        return ps
    return [l.strip() for l in open(fp) if l.strip()]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def safe_fn(p, ml=50):
    return "".join(c if c.isalnum() or c in ' -_' else '_' for c in p)[:ml].replace(' ','_')

def encode_concepts(te, tok, concepts, dev):
    es = []
    for c in concepts:
        inp = tok(c, padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt")
        es.append(te(inp.input_ids.to(dev))[0])
    return torch.stack(es).mean(0)

# ── Args ───────────────────────────────────────────────────────────
def parse_args():
    p = ArgumentParser(description="v24: Comprehensive dual-direction safe generation")
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
    # WHERE
    p.add_argument("--where_mode", default="noise", choices=["noise","attn"])
    p.add_argument("--example_mode", default="both", choices=["text","img","both"])
    p.add_argument("--img_pool", default="cls_multi", choices=["cls_mean","cls_multi","patch_mean"])
    p.add_argument("--fusion", default="union", choices=["union","soft_union","mean","multiply"])
    p.add_argument("--clip_embeddings", default=None)
    p.add_argument("--patch_embeddings", default=None, help="For patch_mean pool")
    p.add_argument("--spatial_threshold", type=float, default=0.1)
    p.add_argument("--img_spatial_threshold", type=float, default=None)
    p.add_argument("--sigmoid_alpha", type=float, default=10.0)
    p.add_argument("--neighborhood_size", type=int, default=3)
    p.add_argument("--blur_sigma", type=float, default=1.0)
    p.add_argument("--attn_resolutions", type=int, nargs="+", default=[16,32])
    p.add_argument("--max_exemplars", type=int, default=16)
    # HOW
    p.add_argument("--how_mode", default="anchor_inpaint",
                   choices=["anchor_inpaint","hybrid","hybrid_cfg","target_sub"])
    p.add_argument("--safety_scale", type=float, default=1.2)
    p.add_argument("--anchor_scale", type=float, default=None,
                   help="Separate anchor scale for hybrid modes (default: same as safety_scale)")
    # Concepts
    p.add_argument("--target_concepts", nargs="+", default=["nudity","nude person","naked body"])
    p.add_argument("--anchor_concepts", nargs="+", default=["clothed person","person wearing clothes"])
    p.add_argument("--save_maps", action="store_true")
    p.add_argument("--debug", action="store_true")

    a = p.parse_args()
    if a.img_spatial_threshold is None: a.img_spatial_threshold = a.spatial_threshold
    if a.example_mode in ("img","both") and a.clip_embeddings is None:
        p.error("--clip_embeddings required for example_mode=img/both")
    return a

# ── Main ───────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag = f"where={args.where_mode} ex={args.example_mode} pool={args.img_pool} " \
          f"fusion={args.fusion} how={args.how_mode} ss={args.safety_scale}"
    print(f"\n{'='*70}\nv24: {tag}\n{'='*70}")

    prompts = load_prompts(args.prompts)
    end = args.end_idx if args.end_idx > 0 else len(prompts)
    pw = list(enumerate(prompts))[args.start_idx:end]
    print(f"Prompts: {len(prompts)}, processing [{args.start_idx}:{end}]")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, safety_checker=None).to(dev)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.feature_extractor = None
    unet, vae, tok, te = pipe.unet, pipe.vae, pipe.tokenizer, pipe.text_encoder
    sched = pipe.scheduler

    with torch.no_grad():
        text_tgt = encode_concepts(te, tok, args.target_concepts, dev)
        anchor_emb = encode_concepts(te, tok, args.anchor_concepts, dev)
        unc_inp = tok("", padding="max_length", max_length=tok.model_max_length,
                      truncation=True, return_tensors="pt")
        unc_emb = te(unc_inp.input_ids.to(dev))[0]

    # Image embeddings
    img_tgt = None
    if args.example_mode in ("img", "both"):
        clip_data = torch.load(args.clip_embeddings, map_location="cpu")
        # Prefer projected embeddings (proper norm for UNet)
        if "target_clip_embeds_proj" in clip_data:
            img_tgt = clip_data["target_clip_embeds_proj"].to(device=dev, dtype=next(unet.parameters()).dtype)
            print(f"  Image embeds: PROJECTED, shape={img_tgt.shape}, "
                  f"norm={img_tgt.float().norm(dim=-1).mean():.1f}")
        else:
            if args.img_pool == "patch_mean" and args.patch_embeddings:
                patch_data = torch.load(args.patch_embeddings, map_location="cpu")
                clip_data["target_patches"] = patch_data.get("target_patches",
                                                              clip_data.get("target_clip_features"))
            img_tgt = build_img_embeds(clip_data, te, tok, dev,
                                        pool=args.img_pool, max_tok=args.max_exemplars)
            img_tgt = img_tgt.to(dtype=next(unet.parameters()).dtype)
            print(f"  Image embeds: {args.img_pool}, shape={img_tgt.shape}, "
                  f"norm={img_tgt.float().norm(dim=-1).mean():.1f}")

    cas = GlobalCAS(threshold=args.cas_threshold, sticky=args.cas_sticky)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if args.save_maps: (outdir/"maps").mkdir(exist_ok=True)
    stats = []

    for pi, prompt in tqdm(pw, desc="Generating"):
        if not prompt.strip(): continue
        for si in range(args.nsamples):
            seed = args.seed + pi * args.nsamples + si
            set_seed(seed); cas.reset()
            gc, cvs, mas = 0, [], []

            with torch.no_grad():
                pi_tok = tok(prompt, padding="max_length", max_length=tok.model_max_length,
                             truncation=True, return_tensors="pt")
                p_emb = te(pi_tok.input_ids.to(dev))[0]

            set_seed(seed)
            lat = torch.randn(1,4,64,64, device=dev, dtype=torch.float16) * sched.init_noise_sigma
            sched.set_timesteps(args.steps, device=dev)
            ts = len(sched.timesteps)

            for si_t, t in enumerate(sched.timesteps):
                li = sched.scale_model_input(lat, t)
                with torch.no_grad():
                    raw = unet(torch.cat([li,li]), t,
                               encoder_hidden_states=torch.cat([unc_emb, p_emb])).sample
                    en, ep = raw.chunk(2)
                    # Text target (always needed for WHEN + text WHERE)
                    et = unet(li, t, encoder_hidden_states=text_tgt).sample

                ec = en + args.cfg_scale * (ep - en)
                cv, trig = cas.compute(ep, en, et)
                cvs.append(cv)

                if trig:
                    with torch.no_grad():
                        ea = unet(li, t, encoder_hidden_states=anchor_emb).sample

                    # ── WHERE: compute masks ──
                    # Text mask
                    if args.example_mode in ("text", "both"):
                        if args.where_mode == "noise":
                            raw_t = compute_spatial_cas(ep, en, et, args.neighborhood_size)
                        else:  # attn
                            raw_t = capture_attn_map(unet, text_tgt, li, t, args.attn_resolutions)
                        text_mask = soft_mask(raw_t.to(dev), args.spatial_threshold,
                                              args.sigmoid_alpha, args.blur_sigma)
                    else:
                        text_mask = None

                    # Image mask
                    if args.example_mode in ("img", "both") and img_tgt is not None:
                        if args.where_mode == "noise":
                            with torch.no_grad():
                                ei = unet(li, t, encoder_hidden_states=img_tgt).sample
                            raw_i = compute_spatial_cas(ep, en, ei, args.neighborhood_size)
                        else:  # attn
                            raw_i = capture_attn_map(unet, img_tgt, li, t, args.attn_resolutions)
                        img_mask = soft_mask(raw_i.to(dev), args.img_spatial_threshold,
                                             args.sigmoid_alpha, args.blur_sigma)
                    else:
                        img_mask = None

                    # Fuse
                    if args.example_mode == "text":
                        final_mask = text_mask
                    elif args.example_mode == "img":
                        final_mask = img_mask
                    else:  # both
                        final_mask = fuse(text_mask, img_mask, args.fusion)

                    # ── HOW: apply guidance ──
                    eps_final = apply_guidance(ec, en, ep, et, ea, final_mask,
                                               args.how_mode, args.safety_scale,
                                               args.cfg_scale, args.anchor_scale)
                    gc += 1
                    mas.append(float(final_mask.mean()))

                    if args.debug and si_t % 10 == 0:
                        ta = float(text_mask.mean()) if text_mask is not None else 0
                        ia = float(img_mask.mean()) if img_mask is not None else 0
                        fa = mas[-1]
                        print(f"  [{si_t:02d}] CAS={cv:.3f} t={ta:.3f} i={ia:.3f} f={fa:.3f}")

                    if args.save_maps and si_t % 10 == 0:
                        md = outdir/"maps"
                        pf = f"{pi:04d}_{si:02d}_s{si_t:03d}"
                        for nm, mm in [("text",text_mask),("img",img_mask),("final",final_mask)]:
                            if mm is not None:
                                mn = mm[0,0].float().cpu().numpy()
                                Image.fromarray((np.clip(mn,0,1)*255).astype(np.uint8),'L').save(
                                    str(md/f"{pf}_{nm}.png"))
                else:
                    eps_final = ec

                lp = lat.clone()
                lat = sched.step(eps_final, t, lat).prev_sample
                if torch.isnan(lat).any() or torch.isinf(lat).any():
                    lat = sched.step(en + args.cfg_scale*(ep-en), t, lp).prev_sample

            with torch.no_grad():
                dec = vae.decode(lat.to(vae.dtype)/vae.config.scaling_factor).sample
                dec = (dec/2+0.5).clamp(0,1)
                img = (dec[0].cpu().permute(1,2,0).numpy()*255).round().astype(np.uint8)

            fn = f"{pi:04d}_{si:02d}_{safe_fn(prompt)}.png"
            Image.fromarray(img).resize((512,512)).save(str(outdir/fn))
            stats.append({"prompt_idx":pi,"sample_idx":si,"seed":seed,
                          "prompt":prompt[:100],"filename":fn,
                          "guided_steps":gc,"total_steps":ts,
                          "max_cas":max(cvs) if cvs else 0,
                          "mean_mask_area":float(np.mean(mas)) if mas else 0})

    json.dump(stats, open(outdir/"generation_stats.json","w"), indent=2)
    json.dump(vars(args), open(outdir/"args.json","w"), indent=2)
    gi = sum(1 for s in stats if s["guided_steps"]>0)
    print(f"\nDone! {len(stats)} imgs. Guided: {gi}/{len(stats)}")

if __name__ == "__main__":
    main()
