"""
Probe Visualization — Generate detailed mask overlays for paper figures.
Shows: original image, text mask, image mask, fused mask, guided result
at multiple timesteps, with and without sigmoid (soft vs hard).
"""
import torch, os, sys, argparse, json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, "/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG")


def visualize_single_prompt(args):
    from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    from generate_v27 import make_probe_mask, compute_attention_spatial_mask

    device = torch.device(args.device)
    dtype = torch.float16

    # Load model
    ckpt = "CompVis/stable-diffusion-v1-4"
    vae = AutoencoderKL.from_pretrained(ckpt, subfolder="vae", torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(ckpt, subfolder="unet", torch_dtype=dtype).to(device)
    tok = CLIPTokenizer.from_pretrained(ckpt, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(ckpt, subfolder="text_encoder", torch_dtype=dtype).to(device)
    sched = DDIMScheduler.from_pretrained(ckpt, subfolder="scheduler")
    sched.set_timesteps(args.steps)

    # Load CLIP exemplar embeddings
    clip_embs = torch.load(args.clip_embeddings, map_location=device) if args.clip_embeddings else None

    # Encode prompts
    def encode(text):
        ids = tok(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
        return te(ids)[0].to(dtype)

    prompt_emb = encode(args.prompt)
    uncond_emb = encode("")
    target_emb = encode(args.target_prompt)

    # Target word token indices
    target_words = args.target_words.split(",")
    prompt_tokens = tok.tokenize(args.prompt)
    target_indices = []
    for tw in target_words:
        tw_tokens = tok.tokenize(tw)
        for i, pt in enumerate(prompt_tokens):
            if pt in tw_tokens:
                target_indices.append(i + 1)  # +1 for BOS

    # Setup attention hooks
    attn_maps = {}
    hooks = []
    for name, module in unet.named_modules():
        if hasattr(module, 'processor') and 'attn2' in name and 'up_blocks' in name:
            def make_hook(n):
                def hook_fn(mod, inp, out):
                    if hasattr(mod, '_last_attn_map'):
                        attn_maps[n] = mod._last_attn_map
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Generate with mask extraction at every step
    gen = torch.Generator(device).manual_seed(args.seed)
    lat = torch.randn(1, 4, 64, 64, generator=gen, device=device, dtype=dtype)
    lat = lat * sched.init_noise_sigma

    os.makedirs(args.outdir, exist_ok=True)

    # Store masks at selected timesteps
    all_masks = {}

    for si_t, t in enumerate(sched.timesteps):
        with torch.no_grad():
            en = unet(lat, t, uncond_emb).sample
            ep = unet(lat, t, prompt_emb).sample
            et = unet(lat, t, target_emb).sample

        # CAS
        dp = ep - en
        dt = et - en
        cas = torch.nn.functional.cosine_similarity(dp.flatten(), dt.flatten(), dim=0).item()

        # Compute masks (always, for visualization)
        # Text probe
        txt_mask_raw = None
        txt_mask_soft = None
        if target_indices:
            txt_attn = compute_attention_spatial_mask(
                unet, target_indices, resolutions=[16, 32])
            txt_mask_raw = txt_attn.clone()  # Before sigmoid
            txt_mask_soft = make_probe_mask(txt_attn, args.attn_threshold, alpha=10., blur=1., device=device)

        # Image probe
        img_mask_raw = None
        img_mask_soft = None
        if clip_embs is not None:
            img_attn = compute_attention_spatial_mask(
                unet, list(range(1, 5)), resolutions=[16, 32], mode="image",
                clip_embeddings=clip_embs)
            img_mask_raw = img_attn.clone()
            img_mask_soft = make_probe_mask(img_attn, args.img_attn_threshold, alpha=10., blur=1., device=device)

        # Fused mask
        fused_raw = None
        fused_soft = None
        if txt_mask_soft is not None and img_mask_soft is not None:
            fused_soft = torch.clamp(txt_mask_soft + img_mask_soft, 0, 1)
            fused_raw = torch.clamp(txt_mask_raw + img_mask_raw, 0, 1) if txt_mask_raw is not None else None

        # Apply guidance
        if cas > args.cas_threshold:
            mask = fused_soft if fused_soft is not None else (txt_mask_soft if txt_mask_soft is not None else None)
            if mask is not None:
                mask_64 = torch.nn.functional.interpolate(mask, size=(64, 64), mode='bilinear')
                eps_final = (en + args.cfg_scale * dp) + args.target_scale * mask_64 * (et - en) * (-1) + args.anchor_scale * mask_64 * (en - et)
            else:
                eps_final = en + args.cfg_scale * dp
        else:
            eps_final = en + args.cfg_scale * dp

        # Save masks at selected steps
        if si_t in args.viz_steps:
            step_data = {"step": si_t, "t": int(t), "cas": cas}
            for name, m in [("txt_raw", txt_mask_raw), ("txt_soft", txt_mask_soft),
                            ("img_raw", img_mask_raw), ("img_soft", img_mask_soft),
                            ("fused_raw", fused_raw), ("fused_soft", fused_soft)]:
                if m is not None:
                    m_np = torch.nn.functional.interpolate(m, size=(512, 512), mode='bilinear')[0, 0].float().cpu().numpy()
                    step_data[name] = m_np
            all_masks[si_t] = step_data

        lat = sched.step(eps_final, t, lat).prev_sample

    # Decode final image
    with torch.no_grad():
        dec = vae.decode(lat.to(vae.dtype) / vae.config.scaling_factor).sample
        dec = (dec / 2 + 0.5).clamp(0, 1)
        result_img = (dec[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

    # Also generate baseline (no guidance) for comparison
    gen2 = torch.Generator(device).manual_seed(args.seed)
    lat2 = torch.randn(1, 4, 64, 64, generator=gen2, device=device, dtype=dtype)
    lat2 = lat2 * sched.init_noise_sigma
    for t in sched.timesteps:
        with torch.no_grad():
            en2 = unet(lat2, t, uncond_emb).sample
            ep2 = unet(lat2, t, prompt_emb).sample
        lat2 = sched.step(en2 + args.cfg_scale * (ep2 - en2), t, lat2).prev_sample
    with torch.no_grad():
        dec2 = vae.decode(lat2.to(vae.dtype) / vae.config.scaling_factor).sample
        dec2 = (dec2 / 2 + 0.5).clamp(0, 1)
        baseline_img = (dec2[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

    # Remove hooks
    for h in hooks:
        h.remove()

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    viz_steps = sorted(all_masks.keys())

    # Figure 1: Temporal evolution of masks
    fig, axes = plt.subplots(4, len(viz_steps) + 2, figsize=(3 * (len(viz_steps) + 2), 12))
    fig.suptitle(f'Probe Visualization: "{args.prompt[:60]}..."', fontsize=14, fontweight='bold')

    # Row labels
    row_labels = ["Text Probe\n(raw)", "Image Probe\n(raw)", "Fused Mask\n(soft)", "CAS-gated\nFinal Mask"]

    # First column: baseline
    for r in range(4):
        axes[r, 0].imshow(baseline_img)
        axes[r, 0].set_title("Baseline\n(no guidance)", fontsize=9)
        axes[r, 0].axis('off')
        axes[r, 0].set_ylabel(row_labels[r], fontsize=10, fontweight='bold')

    # Middle columns: masks at each timestep
    for col_idx, step in enumerate(viz_steps):
        data = all_masks[step]
        col = col_idx + 1

        # Row 0: Text probe raw
        if "txt_raw" in data:
            axes[0, col].imshow(baseline_img, alpha=0.4)
            im = axes[0, col].imshow(data["txt_raw"], cmap='hot', alpha=0.7, vmin=0, vmax=1)
            axes[0, col].set_title(f"Step {data['step']}\nCAS={data['cas']:.2f}", fontsize=9)
        axes[0, col].axis('off')

        # Row 1: Image probe raw
        if "img_raw" in data:
            axes[1, col].imshow(baseline_img, alpha=0.4)
            axes[1, col].imshow(data["img_raw"], cmap='hot', alpha=0.7, vmin=0, vmax=1)
        axes[1, col].axis('off')

        # Row 2: Fused soft
        if "fused_soft" in data:
            axes[2, col].imshow(baseline_img, alpha=0.4)
            axes[2, col].imshow(data["fused_soft"], cmap='hot', alpha=0.7, vmin=0, vmax=1)
        axes[2, col].axis('off')

        # Row 3: CAS-gated (fused_soft * (1 if cas > threshold else 0))
        if "fused_soft" in data:
            gated = data["fused_soft"] * (1.0 if data["cas"] > args.cas_threshold else 0.0)
            axes[3, col].imshow(baseline_img, alpha=0.4)
            axes[3, col].imshow(gated, cmap='hot', alpha=0.7, vmin=0, vmax=1)
            if data["cas"] <= args.cas_threshold:
                axes[3, col].text(256, 256, "CAS OFF", ha='center', va='center',
                                  fontsize=14, color='white', fontweight='bold',
                                  bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        axes[3, col].axis('off')

    # Last column: result
    for r in range(4):
        axes[r, -1].imshow(result_img)
        axes[r, -1].set_title("EBSG Result", fontsize=9)
        axes[r, -1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "probe_temporal.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # Figure 2: Side-by-side comparison (for paper)
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
    fig2.suptitle(f'Probe Masks at Peak CAS Step', fontsize=14, fontweight='bold')

    # Find peak CAS step
    peak_step = max(viz_steps, key=lambda s: all_masks[s]["cas"])
    peak = all_masks[peak_step]

    # Row 0: Raw masks (before sigmoid)
    axes2[0, 0].imshow(baseline_img); axes2[0, 0].set_title("Baseline", fontsize=11); axes2[0, 0].axis('off')
    if "txt_raw" in peak:
        axes2[0, 1].imshow(peak["txt_raw"], cmap='hot', vmin=0, vmax=1); axes2[0, 1].set_title("Text Probe (raw)", fontsize=11)
    axes2[0, 1].axis('off')
    if "img_raw" in peak:
        axes2[0, 2].imshow(peak["img_raw"], cmap='hot', vmin=0, vmax=1); axes2[0, 2].set_title("Image Probe (raw)", fontsize=11)
    axes2[0, 2].axis('off')
    if "fused_raw" in peak:
        axes2[0, 3].imshow(peak["fused_raw"], cmap='hot', vmin=0, vmax=1); axes2[0, 3].set_title("Fused (raw)", fontsize=11)
    axes2[0, 3].axis('off')

    # Row 1: Soft masks (after sigmoid) + result
    axes2[1, 0].imshow(baseline_img, alpha=0.5)
    if "fused_soft" in peak:
        axes2[1, 0].imshow(peak["fused_soft"], cmap='hot', alpha=0.6, vmin=0, vmax=1)
    axes2[1, 0].set_title(f"Overlay (step {peak_step}, CAS={peak['cas']:.2f})", fontsize=11); axes2[1, 0].axis('off')

    if "txt_soft" in peak:
        axes2[1, 1].imshow(peak["txt_soft"], cmap='hot', vmin=0, vmax=1); axes2[1, 1].set_title("Text Probe (soft)", fontsize=11)
    axes2[1, 1].axis('off')
    if "img_soft" in peak:
        axes2[1, 2].imshow(peak["img_soft"], cmap='hot', vmin=0, vmax=1); axes2[1, 2].set_title("Image Probe (soft)", fontsize=11)
    axes2[1, 2].axis('off')
    axes2[1, 3].imshow(result_img); axes2[1, 3].set_title("EBSG Result", fontsize=11); axes2[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "probe_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # Save individual images
    Image.fromarray(baseline_img).save(os.path.join(args.outdir, "baseline.png"))
    Image.fromarray(result_img).save(os.path.join(args.outdir, "result.png"))

    # Save individual masks as colormap images
    for step in viz_steps:
        data = all_masks[step]
        for name in ["txt_raw", "txt_soft", "img_raw", "img_soft", "fused_raw", "fused_soft"]:
            if name in data:
                plt.figure(figsize=(5, 5))
                plt.imshow(data[name], cmap='hot', vmin=0, vmax=1)
                plt.axis('off')
                plt.savefig(os.path.join(args.outdir, f"mask_s{step:02d}_{name}.png"),
                            dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close()

    print(f"Done! Saved to {args.outdir}")
    print(f"Peak CAS at step {peak_step}: {peak['cas']:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--target_prompt", type=str, default="a nude person, naked body, nudity")
    p.add_argument("--target_words", type=str, default="nude,naked,nudity,nsfw,bare,body")
    p.add_argument("--clip_embeddings", type=str,
                   default="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_embeddings.pt")
    p.add_argument("--outdir", type=str, default="/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/outputs/probe_viz")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=7.5)
    p.add_argument("--cas_threshold", type=float, default=0.6)
    p.add_argument("--attn_threshold", type=float, default=0.1)
    p.add_argument("--img_attn_threshold", type=float, default=0.4)
    p.add_argument("--target_scale", type=float, default=15.0)
    p.add_argument("--anchor_scale", type=float, default=15.0)
    p.add_argument("--viz_steps", type=int, nargs="+", default=[0, 5, 10, 15, 20, 30, 40, 49])
    main = visualize_single_prompt
    args = p.parse_args()
    main(args)
