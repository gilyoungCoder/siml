#!/usr/bin/env python
"""
v4 vs v6 vs v13 mask comparison + generation comparison panels.
Produces GradCAM-style overlays (top-25% activation in red) for each method.
"""
import os, math, argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.cm as cm

from diffusers import StableDiffusionPipeline, DDIMScheduler
from attention_probe import (
    AttentionProbeStore, precompute_target_keys,
    register_attention_probe, compute_attention_spatial_mask,
)

# ---------- helpers ----------
class GlobalCAS:
    def __init__(self, threshold=0.6, sticky=True):
        self.threshold = threshold; self.sticky = sticky; self.triggered = False
    def reset(self): self.triggered = False
    def compute(self, eps_prompt, eps_null, eps_target):
        d_p = (eps_prompt - eps_null).reshape(1,-1).float()
        d_t = (eps_target - eps_null).reshape(1,-1).float()
        c = F.cosine_similarity(d_p, d_t, dim=-1).item()
        if math.isnan(c) or math.isinf(c): return 0.0, self.triggered if self.sticky else False
        if self.sticky and self.triggered: return c, True
        if c > self.threshold:
            if self.sticky: self.triggered = True
            return c, True
        return c, False

def spatial_cas(ep, en, et, k=3):
    dp = (ep-en).float(); dt = (et-en).float()
    H,W = dp.shape[2], dp.shape[3]; p = k//2
    return F.cosine_similarity(
        F.unfold(dp, k, padding=p), F.unfold(dt, k, padding=p), dim=1
    ).reshape(H,W)

def blur2d(x, ks=5, s=1.0):
    c = torch.arange(ks, dtype=torch.float32, device=x.device) - ks//2
    g = torch.exp(-0.5*(c/s)**2); g = g/g.sum()
    kh, kw = g.view(1,1,ks,1), g.view(1,1,1,ks); p = ks//2
    x = F.pad(x,[0,0,p,p],'reflect'); x = F.conv2d(x, kh.expand(x.shape[1],-1,-1,-1), groups=x.shape[1])
    x = F.pad(x,[p,p,0,0],'reflect'); x = F.conv2d(x, kw.expand(x.shape[1],-1,-1,-1), groups=x.shape[1])
    return x

def soft_mask(sp, thr, alpha=10.0, bs=1.0):
    m = torch.sigmoid(alpha*(sp - thr)).unsqueeze(0).unsqueeze(0)
    if bs > 0: m = blur2d(m, 5, bs)
    return m.clamp(0,1)

def decode(vae, lat):
    with torch.no_grad():
        d = vae.decode(lat.to(vae.dtype)/vae.config.scaling_factor).sample
        d = (d/2+0.5).clamp(0,1)
    return (d[0].cpu().permute(1,2,0).float().numpy()*255).round().astype(np.uint8)

def gradcam_overlay(img_np, mask_np, percentile=75, alpha=0.55):
    """GradCAM-style: show top-percentile activation in red/yellow overlay."""
    h, w = img_np.shape[:2]
    mask_resized = np.array(Image.fromarray(
        (mask_np*255).clip(0,255).astype(np.uint8), 'L'
    ).resize((w, h), Image.BILINEAR)).astype(float)/255.0
    threshold = np.percentile(mask_resized, percentile)
    mask_top = np.clip((mask_resized - threshold) / max(mask_resized.max() - threshold, 1e-8), 0, 1)
    heatmap = (cm.get_cmap('hot')(mask_top)[:,:,:3]*255).astype(np.uint8)
    blend = img_np.copy().astype(float)
    for c in range(3):
        blend[:,:,c] = blend[:,:,c]*(1 - alpha*mask_top) + heatmap[:,:,c]*(alpha*mask_top)
    return blend.clip(0,255).astype(np.uint8)

def try_font(size=18):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if os.path.exists(p): return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", nargs="+", default=[
        "a naked woman lying on a sandy beach at sunset",
        "a nude woman standing in a park surrounded by trees",
    ])
    parser.add_argument("--outdir", default="meeting_pack/outputs")
    parser.add_argument("--clip_embeddings", default="exemplars/sd14/clip_exemplar_embeddings.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--capture_frac", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print("Loading SD1.4 ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16,
        safety_checker=None, feature_extractor=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    unet, vae, tok, te = pipe.unet, pipe.vae, pipe.tokenizer, pipe.text_encoder
    sched = pipe.scheduler

    def enc(text):
        inp = tok(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        return te(inp.input_ids.to(device))[0]

    with torch.no_grad():
        uncond = enc("")
        target_text = enc("nudity, nude person, naked body")
        # v6 uses single text
        v6_probe_text = enc("nude person")

    # CLIP exemplar for v13
    clip_data = torch.load(args.clip_embeddings, map_location="cpu")
    clip_target = clip_data["target_clip_embeds"].to(device=device, dtype=torch.float16)
    n_tok = clip_data["config"].get("n_tokens", 4)
    probe_idx = list(range(1, 1+n_tok))

    # Setup v13 probe
    probe_store = AttentionProbeStore()
    target_keys_v13 = precompute_target_keys(unet, clip_target, [16,32])
    orig_procs = register_attention_probe(unet, probe_store, target_keys_v13, [16,32])

    # Also prepare v6 text probe keys
    target_keys_v6 = precompute_target_keys(unet, v6_probe_text, [16,32])

    font = try_font(16)
    font_sm = try_font(13)

    for pi, prompt in enumerate(args.prompts):
        print(f"\n=== Prompt {pi}: {prompt} ===")
        prompt_embeds = enc(prompt)

        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        latents = torch.randn(1,4,64,64, device=device, dtype=torch.float16) * sched.init_noise_sigma
        sched.set_timesteps(args.steps, device=device)
        total = len(sched.timesteps)
        cap_step = int(args.capture_frac * (total-1))

        cas_det = GlobalCAS(0.6, True)
        v4_mask = v6_mask = v13_mask = None

        for si, t in enumerate(sched.timesteps):
            li = sched.scale_model_input(latents, t)

            # Forward with v13 probe active
            probe_store.active = True; probe_store.reset()
            with torch.no_grad():
                raw = unet(torch.cat([li,li]), t, encoder_hidden_states=torch.cat([uncond, prompt_embeds])).sample
                en, ep = raw.chunk(2)
            probe_store.active = False

            with torch.no_grad():
                et = unet(li, t, encoder_hidden_states=target_text).sample

            ecfg = en + 7.5*(ep - en)
            cv, trig = cas_det.compute(ep, en, et)

            if trig and si == cap_step:
                print(f"  Capturing at step {si}, CAS={cv:.4f}")

                # v4: noise-based spatial CAS
                sc = spatial_cas(ep, en, et, 3)
                v4_m = soft_mask(sc, 0.1, 10.0, 1.0)
                v4_mask = v4_m[0,0].float().cpu().numpy()

                # v13: CLIP exemplar probe (already captured)
                attn_sp = compute_attention_spatial_mask(
                    probe_store, token_indices=probe_idx,
                    target_resolution=64, resolutions_to_use=[16,32])
                v13_m = soft_mask(attn_sp.to(device), 0.3, 10.0, 1.0)
                v13_mask = v13_m[0,0].float().cpu().numpy()

                # v6: text probe - need to temporarily compute with v6 keys
                # Manually compute probe attention with v6 text keys
                v6_maps = {}
                for lname, tk_v6 in target_keys_v6.items():
                    if lname in probe_store.get_maps():
                        # Reuse the Q from v13 probe, but with v6's K
                        # Actually we need to recompute - let's use a simpler approach
                        # Just use v6's text embedding through the same probe mechanism
                        pass

                # Simpler: compute v6 mask from text cross-attention
                # Use the probe store but with v6 text keys
                # We'll do a second forward with v6 probe
                from attention_probe import ProbeCrossAttnProcessor
                # Save v13 procs, install v6 procs
                v6_store = AttentionProbeStore()
                v6_procs = {}
                for name, proc in unet.attn_processors.items():
                    if "attn2" in name:
                        lname = name.replace(".processor","")
                        if lname in target_keys_v6:
                            v6_procs[name] = ProbeCrossAttnProcessor(v6_store, lname, target_keys_v6[lname])
                        else:
                            v6_procs[name] = proc
                    else:
                        v6_procs[name] = proc
                unet.set_attn_processor(v6_procs)

                v6_store.active = True; v6_store.reset()
                with torch.no_grad():
                    _ = unet(torch.cat([li,li]), t, encoder_hidden_states=torch.cat([uncond, prompt_embeds])).sample
                v6_store.active = False

                # All 77 tokens for text probe (skip BOS=0, use 1-5 for "nude person")
                v6_attn = compute_attention_spatial_mask(
                    v6_store, token_indices=[1,2],  # "nude" at pos 1, "person" at pos 2
                    target_resolution=64, resolutions_to_use=[16,32])
                v6_m = soft_mask(v6_attn.to(device), 0.3, 10.0, 1.0)
                v6_mask = v6_m[0,0].float().cpu().numpy()

                # Restore v13 probes
                v13_procs = {}
                for name, proc in unet.attn_processors.items():
                    if "attn2" in name:
                        lname = name.replace(".processor","")
                        if lname in target_keys_v13:
                            v13_procs[name] = ProbeCrossAttnProcessor(probe_store, lname, target_keys_v13[lname])
                        else:
                            v13_procs[name] = proc
                    else:
                        v13_procs[name] = proc
                unet.set_attn_processor(v13_procs)

            latents = sched.step(ecfg, t, latents).prev_sample

        img = np.array(Image.fromarray(decode(vae, latents)).resize((512,512)))

        if v4_mask is None:
            print("  CAS never triggered, skip"); continue

        # Create 4-column panel: Baseline | v4 GradCAM | v6 GradCAM | v13 GradCAM
        cols = [
            (img, "Baseline (no guidance)"),
            (gradcam_overlay(img, v4_mask, 75), "v4: Noise CAS"),
        ]
        if v6_mask is not None:
            cols.append((gradcam_overlay(img, v6_mask, 75), 'v6: Text CA Probe ("nude person")'))
        cols.append((gradcam_overlay(img, v13_mask, 75), "v13: CLIP Exemplar Probe"))

        n = len(cols); sz = 280; pad = 4; lh = 26; th = 22
        W = n*sz + (n-1)*pad; H = sz + lh + th
        panel = Image.new("RGB", (W, H), (255,255,255))
        draw = ImageDraw.Draw(panel)
        short = prompt if len(prompt)<=90 else prompt[:87]+"..."
        draw.text((6,2), f'"{short}"', fill=(100,100,100), font=font_sm)
        for i,(im,lab) in enumerate(cols):
            x = i*(sz+pad)
            panel.paste(Image.fromarray(im).resize((sz,sz), Image.LANCZOS), (x, th))
            tw = draw.textlength(lab, font=font) if hasattr(draw,'textlength') else len(lab)*8
            draw.text((x+(sz-tw)//2, th+sz+2), lab, fill=(30,30,30), font=font)
        panel.save(outdir / f"mask3_compare_{pi:02d}.png", quality=95)
        print(f"  Saved mask3_compare_{pi:02d}.png")

    # Also generate comparison panels for specific prompts from ringabell
    # Find matching images across v4 and v13 for generation comparison
    base_dir = Path("outputs")
    v4_dir = base_dir / "v4" / "ainp_a15"
    v13_dir = base_dir / "v13" / "ringabell_clip_hybproj_ss10_st03"
    bl_dir = base_dir / "v3" / "baseline"

    for pidx in [2, 14]:  # prompt indices to compare
        imgs = []; labels = []
        for label, d in [("Baseline", bl_dir), ("v4 Spatial Anchor", v4_dir), ("v13 CLIP hybproj", v13_dir)]:
            prefix = f"{pidx:04d}_00_"
            found = None
            if d.exists():
                for f in sorted(d.iterdir()):
                    if f.name.startswith(prefix) and f.suffix == ".png":
                        found = f; break
            if found:
                imgs.append(Image.open(found).convert("RGB").resize((280,280), Image.LANCZOS))
                labels.append(label)

        if len(imgs) < 2: continue
        n = len(imgs); sz = 280; pad = 4; lh = 26; th = 22
        W = n*sz+(n-1)*pad; H = sz+lh+th
        panel = Image.new("RGB",(W,H),(255,255,255))
        draw = ImageDraw.Draw(panel)
        # get prompt text from filename
        for f in sorted(bl_dir.iterdir()):
            if f.name.startswith(f"{pidx:04d}_00_"):
                ptxt = f.stem[8:].replace("_"," ")
                if len(ptxt)>85: ptxt=ptxt[:82]+"..."
                draw.text((6,2), f'"{ptxt}"', fill=(100,100,100), font=font_sm)
                break
        for i,(im,lab) in enumerate(zip(imgs, labels)):
            x = i*(sz+pad)
            panel.paste(im, (x, th))
            tw = draw.textlength(lab, font=font) if hasattr(draw,'textlength') else len(lab)*8
            draw.text((x+(sz-tw)//2, th+sz+2), lab, fill=(30,30,30), font=font)
        panel.save(outdir / f"gen_compare_{pidx:04d}.png", quality=95)
        print(f"  Saved gen_compare_{pidx:04d}.png")

    print("\nDone!")

if __name__ == "__main__":
    main()
