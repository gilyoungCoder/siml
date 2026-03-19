#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GradCAM Spatial Mask Visualization for Z0 Classifier Guidance.

Generates an image with guidance and saves per-step GradCAM heatmaps
to verify that spatial masking is working correctly.

Saves:
  - generated image (with/without guidance)
  - per-step GradCAM heatmaps overlaid on decoded z0_hat
  - summary grid: image + selected step heatmaps
  - CSV of per-step stats (grad_mag, CDF values, mask_ratio, alpha_bar)

Usage:
    python visualize_gradcam.py \
        --prompt "a painting of a nude woman in a countryside landscape" \
        --classifier_ckpt ./work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth \
        --harmful_stats_path ./harmful_stats.pt \
        --output_dir ./gradcam_vis

    # With grad_wrt_z0 (to compare scale matching):
    python visualize_gradcam.py \
        --prompt "a painting of a nude woman in a countryside landscape" \
        --classifier_ckpt ./work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth \
        --harmful_stats_path ./harmful_stats.pt \
        --grad_wrt_z0 \
        --output_dir ./gradcam_vis_z0grad
"""

import csv
import math
import os
from argparse import ArgumentParser
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import DDIMScheduler
from geo_utils.custom_stable_diffusion import CustomStableDiffusionPipeline
from geo_utils.guidance_utils import Z0GuidanceModel


VAE_SCALE = 0.18215


def decode_z0_to_pil(vae, z0, device):
    """Decode latent z0 to PIL image."""
    with torch.no_grad():
        x0 = vae.decode(z0.detach() / VAE_SCALE, return_dict=False)[0]
        x0 = x0.clamp(-1, 1)
        x0 = (x0[0].permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255
        return Image.fromarray(x0.astype(np.uint8))


def make_heatmap(mask_2d, title="", vmin=0, vmax=1, bg_img=None, alpha=0.5):
    """Create a heatmap figure, optionally overlaid on a background image.

    Args:
        mask_2d: 2D numpy array for heatmap
        bg_img: PIL Image to use as background (overlaid with heatmap)
        alpha: heatmap opacity when overlaying (0=transparent, 1=opaque)
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    if bg_img is not None:
        # Resize background to match heatmap resolution for overlay
        bg_resized = np.array(bg_img.resize(
            (mask_2d.shape[1], mask_2d.shape[0]), Image.BILINEAR
        ))
        ax.imshow(bg_resized, interpolation="bilinear")
        im = ax.imshow(mask_2d, cmap="jet", vmin=vmin, vmax=vmax,
                       interpolation="nearest", alpha=alpha)
    else:
        im = ax.imshow(mask_2d, cmap="hot", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def overlay_heatmap_on_image(bg_img, mask_2d, colormap="jet", alpha=0.5):
    """Create a PIL image with heatmap overlaid on background.

    Args:
        bg_img: PIL Image (background)
        mask_2d: 2D numpy array [0, 1] for heatmap
        colormap: matplotlib colormap name
        alpha: heatmap opacity

    Returns:
        PIL Image with overlay
    """
    h, w = mask_2d.shape
    bg = np.array(bg_img.resize((w, h), Image.BILINEAR)).astype(np.float32) / 255.0

    cmap = plt.get_cmap(colormap)
    heatmap = cmap(np.clip(mask_2d, 0, 1))[:, :, :3]  # (H, W, 3), drop alpha

    blended = (1 - alpha) * bg + alpha * heatmap
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def fig_to_pil(fig):
    """Convert matplotlib figure to PIL Image."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)
    plt.close(fig)
    return Image.fromarray(img)


def visualize_callback(pipe, step, timestep, callback_kwargs,
                       guidance_model=None, guidance_scale=10.0,
                       target_class=1, step_records=None,
                       vae=None, device=None, save_every=5,
                       no_guidance=False):
    """
    Callback that applies guidance AND records per-step proper GradCAM info.
    Matches the CDF logic in guidance_utils.py for accurate visualization.
    """
    with torch.enable_grad():
        prev_latents = callback_kwargs["prev_latents"].clone().detach().requires_grad_(True)
        noise_pred = callback_kwargs["noise_pred"]
        noise_pred_uncond = callback_kwargs.get("noise_pred_uncond", noise_pred)

        alpha_bar = pipe.scheduler.alphas_cumprod.to(device)[timestep].item()

        # Tweedie z0_hat
        z0_hat = guidance_model.gradient_model.get_scaled_input(
            pipe, prev_latents, noise_pred_uncond, timestep
        )

        # Proper GradCAM from classifier intermediate features
        gradcam_map = guidance_model.gradient_model.model.compute_gradcam(
            z0_hat.detach(), target_class=guidance_model.gradcam_target,
            layer_name=guidance_model.gradcam_layer,
            ref_class=guidance_model.gradcam_ref_class,
        )  # (B, 1, H, W), per-sample normalized [0, 1]

        # Classifier gradient for guidance
        grad_input = z0_hat if guidance_model.grad_wrt_z0 else prev_latents
        diff_val, grad, output_for_log = guidance_model.gradient_model.get_grad(
            z0_hat, None, None, grad_input,
            target_class=target_class,
        )

    grad = grad.detach()

    # Apply CDF normalization if harmful_stats available (match guidance_utils.py)
    harmful_stats = guidance_model.harmful_stats
    if harmful_stats is not None and "gradcam_mu" in harmful_stats:
        mu = harmful_stats["gradcam_mu"]
        sigma = harmful_stats["gradcam_sigma"]
        z = (gradcam_map - mu) / (sigma + 1e-8)
        cdf_map = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    else:
        cdf_map = gradcam_map  # fallback: use raw GradCAM values

    # Spatial mask from CDF-normalized GradCAM
    if guidance_model.spatial_soft:
        spatial_mask = cdf_map
    else:
        spatial_mask = (cdf_map > guidance_model.spatial_threshold).float()
    mask_ratio = spatial_mask.mean().item()

    # GradCAM stats for logging (raw + CDF)
    gc_mean = gradcam_map.mean().item()
    gc_max = gradcam_map.max().item()
    cdf_mean = cdf_map.mean().item()

    record = {
        "step": step,
        "timestep": timestep.item(),
        "alpha_bar": alpha_bar,
        "inv_sqrt_alpha": 1.0 / math.sqrt(alpha_bar) if alpha_bar > 0 else float("inf"),
        "gradcam_mean": gc_mean,
        "gradcam_max": gc_max,
        "cdf_mean": cdf_mean,
        "mask_ratio": mask_ratio,
        "monitor": output_for_log.mean().item(),
    }

    # Save visualization at selected steps
    if step % save_every == 0 or step == 0:
        record["gradcam_2d"] = gradcam_map[0, 0].cpu().numpy()
        record["cdf_2d"] = cdf_map[0, 0].cpu().numpy()
        record["spatial_mask_2d"] = spatial_mask[0, 0].cpu().numpy()
        record["z0_hat_pil"] = decode_z0_to_pil(vae, z0_hat, device)

    step_records.append(record)

    if not no_guidance:
        # Apply actual guidance with spatial mask
        grad_masked = grad * spatial_mask
        latents = guidance_model.gradient_model.guide_samples(
            pipe, noise_pred, prev_latents,
            callback_kwargs["latents"], timestep, grad_masked, guidance_scale,
        )
        callback_kwargs["latents"] = latents

    return callback_kwargs


def parse_args():
    parser = ArgumentParser(description="GradCAM visualization for z0 classifier guidance")
    parser.add_argument("--prompt", type=str,
                        default="a painting of a nude woman in a countryside landscape")
    parser.add_argument("--ckpt_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--classifier_ckpt", type=str,
                        default="./work_dirs/z0_resnet18_classifier/checkpoint/step_7700/classifier.pth")
    parser.add_argument("--output_dir", type=str, default="./gradcam_vis")
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--spatial_threshold", type=float, default=0.3)
    parser.add_argument("--spatial_soft", action="store_true")
    parser.add_argument("--grad_wrt_z0", action="store_true",
                        help="Compute gradient w.r.t. z0_hat instead of z_t")
    parser.add_argument("--gradcam_layer", type=str, default="layer4",
                        choices=["layer1", "layer2", "layer3", "layer4"],
                        help="Which ResNet layer for GradCAM")
    parser.add_argument("--gradcam_ref_class", type=int, default=None,
                        help="Reference class for diff GradCAM (score=target-ref)")
    parser.add_argument("--guidance_mode", type=str, default="safe_minus_harm",
                        choices=["target", "safe_minus_harm"])
    parser.add_argument("--target_class", type=int, default=1)
    parser.add_argument("--safe_classes", type=int, nargs="+", default=[1])
    parser.add_argument("--harm_classes", type=int, nargs="+", default=[2])
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--harmful_stats_path", type=str, default=None,
                        help="Path to harmful_stats.pt for CDF normalization")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save heatmap every N steps")
    parser.add_argument("--no_guidance", action="store_true",
                        help="Record GradCAM per step without applying guidance")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load pipeline
    print("Loading SD pipeline...")
    pipe = CustomStableDiffusionPipeline.from_pretrained(
        args.ckpt_path, safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Setup guidance model
    model_config = {
        "architecture": "resnet18",
        "num_classes": 3,
        "space": "latent",
        "guidance_mode": args.guidance_mode,
        "safe_classes": args.safe_classes,
        "harm_classes": args.harm_classes,
        "spatial_mode": "gradcam",
        "spatial_threshold": args.spatial_threshold,
        "spatial_soft": args.spatial_soft,
        "grad_wrt_z0": args.grad_wrt_z0,
        "gradcam_layer": args.gradcam_layer,
        "gradcam_ref_class": args.gradcam_ref_class,
        "harmful_stats_path": args.harmful_stats_path,
    }
    guidance_model = Z0GuidanceModel(
        pipe, args.classifier_ckpt, model_config,
        target_class=args.target_class, device=device,
    )
    guidance_model.set_prompt(args.prompt, pipe.tokenizer)

    if not args.no_guidance:
        # ── Generate WITHOUT guidance (baseline) ──
        print(f"\nGenerating baseline (no guidance)...")
        gen = torch.Generator(device=device).manual_seed(args.seed)
        baseline = pipe(
            prompt=args.prompt, guidance_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            height=512, width=512, generator=gen,
            num_images_per_prompt=1,
        ).images[0]
        baseline.save(os.path.join(args.output_dir, "baseline_no_guidance.png"))

    # ── Generate with GradCAM recording ──
    mode_str = "observe only (no guidance)" if args.no_guidance else f"gs={args.guidance_scale}"
    print(f"\nGenerating ({mode_str}, "
          f"threshold={args.spatial_threshold}, "
          f"gradcam_layer={guidance_model.gradcam_layer})...")

    step_records = []
    gen = torch.Generator(device=device).manual_seed(args.seed)

    cb = partial(
        visualize_callback,
        guidance_model=guidance_model,
        guidance_scale=args.guidance_scale,
        target_class=args.target_class,
        step_records=step_records,
        vae=pipe.vae,
        device=device,
        save_every=args.save_every,
        no_guidance=args.no_guidance,
    )

    with torch.enable_grad():
        output = pipe(
            prompt=args.prompt, guidance_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            height=512, width=512, generator=gen,
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=[
                "latents", "noise_pred", "noise_pred_uncond", "prev_latents",
            ],
            num_images_per_prompt=1,
        )
    guided_img = output.images[0]
    out_name = "generated_no_guidance.png" if args.no_guidance else "guided.png"
    guided_img.save(os.path.join(args.output_dir, out_name))

    # ── Save per-step heatmaps ──
    heatmap_dir = os.path.join(args.output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    vis_records = [r for r in step_records if "gradcam_2d" in r]
    for r in vis_records:
        step = r["step"]
        t = r["timestep"]
        z0_img = r["z0_hat_pil"]

        # Save z0_hat decoded image
        z0_img.save(os.path.join(heatmap_dir, f"step{step:02d}_z0hat.png"))

        # Min-max rescaled GradCAM for better visualization
        gc = r["gradcam_2d"]
        p99 = np.percentile(gc, 99)
        gc_rescaled = np.clip(gc / (p99 + 1e-8), 0, 1) if p99 > 0 else gc

        # GradCAM heatmap overlaid on z0_hat (rescaled for visibility)
        fig = make_heatmap(
            gc_rescaled,
            title=f"step={step} t={t} GradCAM (mean={r['gradcam_mean']:.3f}, p99={p99:.3f})",
            vmin=0, vmax=1,
            bg_img=z0_img, alpha=0.5,
        )
        fig.savefig(os.path.join(heatmap_dir, f"step{step:02d}_gradcam.png"), dpi=100)
        plt.close(fig)

        # Spatial mask overlaid on z0_hat
        fig = make_heatmap(
            r["spatial_mask_2d"],
            title=f"step={step} mask (ratio={r['mask_ratio']:.1%}, thr={args.spatial_threshold})",
            vmin=0, vmax=1,
            bg_img=z0_img, alpha=0.4,
        )
        fig.savefig(os.path.join(heatmap_dir, f"step{step:02d}_spatial_mask.png"), dpi=100)
        plt.close(fig)

        # Clean overlay images (no axes/colorbar) — rescaled for visibility
        overlay_gc = overlay_heatmap_on_image(z0_img, gc_rescaled, alpha=0.45)
        overlay_gc.save(os.path.join(heatmap_dir, f"step{step:02d}_overlay_gradcam.png"))

        overlay_mask = overlay_heatmap_on_image(z0_img, r["spatial_mask_2d"], alpha=0.4)
        overlay_mask.save(os.path.join(heatmap_dir, f"step{step:02d}_overlay_mask.png"))

    # ── Save CSV stats ──
    csv_path = os.path.join(args.output_dir, "step_stats.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "timestep", "alpha_bar",
            "gradcam_mean", "gradcam_max", "cdf_mean", "mask_ratio", "monitor",
        ])
        for r in step_records:
            writer.writerow([
                r["step"], r["timestep"], f"{r['alpha_bar']:.6f}",
                f"{r['gradcam_mean']:.4f}", f"{r['gradcam_max']:.4f}",
                f"{r.get('cdf_mean', 0):.4f}",
                f"{r['mask_ratio']:.4f}", f"{r['monitor']:.4f}",
            ])
    print(f"  Step stats saved to: {csv_path}")

    # ── Summary grid ──
    selected_steps = [0, 10, 20, 30, 40, 49]
    selected = [r for r in vis_records if r["step"] in selected_steps]

    n_sel = len(selected)
    if n_sel > 0:
        fig = plt.figure(figsize=(4 * (n_sel + 2), 12))
        gs = gridspec.GridSpec(3, n_sel + 2, hspace=0.3, wspace=0.1)

        # Row 0: baseline + guided + z0_hat per step
        if not args.no_guidance:
            ax = fig.add_subplot(gs[0, 0])
            ax.imshow(np.array(baseline.resize((256, 256))))
            ax.set_title("Baseline\n(no guidance)", fontsize=9)
            ax.axis("off")

        ax = fig.add_subplot(gs[0, 1 if not args.no_guidance else 0])
        ax.imshow(np.array(guided_img.resize((256, 256))))
        title = "Generated\n(no guidance)" if args.no_guidance else f"Guided\n(gs={args.guidance_scale})"
        ax.set_title(title, fontsize=9)
        ax.axis("off")

        for i, r in enumerate(selected):
            ax = fig.add_subplot(gs[0, i + 2])
            ax.imshow(np.array(r["z0_hat_pil"].resize((256, 256))))
            ax.set_title(f"z0_hat step={r['step']}\nt={r['timestep']}", fontsize=8)
            ax.axis("off")

        # Row 1: GradCAM heatmaps overlaid on z0_hat
        for i, r in enumerate(selected):
            ax = fig.add_subplot(gs[1, i + 2])
            overlay = overlay_heatmap_on_image(
                r["z0_hat_pil"], r["gradcam_2d"], alpha=0.45
            ).resize((256, 256), Image.BILINEAR)
            ax.imshow(np.array(overlay))
            ax.set_title(f"GradCAM\nmean={r['gradcam_mean']:.3f}", fontsize=8)
            ax.axis("off")

        # Stats text in row 1, col 0-1
        ax = fig.add_subplot(gs[1, 0:2])
        ax.axis("off")
        stats_text = f"Prompt: {args.prompt[:60]}...\n"
        stats_text += f"Mode: {args.guidance_mode}\n"
        stats_text += f"GradCAM layer: {guidance_model.gradcam_layer}\n"
        stats_text += f"GradCAM target: class {guidance_model.gradcam_target}\n"
        stats_text += f"Threshold: {args.spatial_threshold}\n"
        stats_text += f"Soft: {args.spatial_soft}\n"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment="top", family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        # Row 2: Spatial masks overlaid on z0_hat
        for i, r in enumerate(selected):
            ax = fig.add_subplot(gs[2, i + 2])
            overlay = overlay_heatmap_on_image(
                r["z0_hat_pil"], r["spatial_mask_2d"], alpha=0.4
            ).resize((256, 256), Image.BILINEAR)
            ax.imshow(np.array(overlay))
            ratio = r["mask_ratio"]
            ax.set_title(f"Mask\nratio={ratio:.1%}", fontsize=8)
            ax.axis("off")

        # Mask ratio over time plot
        ax = fig.add_subplot(gs[2, 0:2])
        steps_all = [r["step"] for r in step_records]
        ratios_all = [r["mask_ratio"] for r in step_records]
        gc_mean_all = [r["gradcam_mean"] for r in step_records]
        ax.plot(steps_all, ratios_all, "r-o", markersize=3, label="mask_ratio")
        ax.set_xlabel("Step")
        ax.set_ylabel("Mask ratio", color="red")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=args.spatial_threshold, color="gray", linestyle="--", alpha=0.5)
        ax2 = ax.twinx()
        ax2.plot(steps_all, gc_mean_all, "b--", alpha=0.5, label="GradCAM mean")
        ax2.set_ylabel("GradCAM mean", color="blue")
        ax.set_title("Mask ratio & GradCAM over steps", fontsize=9)
        ax.legend(loc="upper left", fontsize=7)
        ax2.legend(loc="upper right", fontsize=7)

        grid_path = os.path.join(args.output_dir, "summary_grid.png")
        fig.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Summary grid saved to: {grid_path}")

    # ── Print summary table ──
    print(f"\n{'=' * 80}")
    print(f"  GRADCAM VISUALIZATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Prompt:         {args.prompt}")
    if args.no_guidance:
        print(f"  Mode:           OBSERVE ONLY (no guidance applied)")
    else:
        print(f"  Mode:           {args.guidance_mode}, gs={args.guidance_scale}")
    print(f"  GradCAM layer:  {guidance_model.gradcam_layer}")
    print(f"  GradCAM target: class {guidance_model.gradcam_target}")
    print(f"  Threshold:      {args.spatial_threshold} ({'soft' if args.spatial_soft else 'binary'})")
    print(f"{'=' * 80}")
    print(f"  {'step':>4s}  {'t':>4s}  {'ᾱ':>8s}  "
          f"{'GC_mean':>8s}  {'GC_max':>8s}  {'CDF_mean':>8s}  {'mask%':>6s}  {'monitor':>8s}")
    print(f"  {'-' * 72}")
    for r in step_records:
        if r["step"] % 5 == 0 or r["step"] == len(step_records) - 1:
            print(f"  {r['step']:4d}  {r['timestep']:4d}  "
                  f"{r['alpha_bar']:8.4f}  "
                  f"{r['gradcam_mean']:8.4f}  {r['gradcam_max']:8.4f}  "
                  f"{r.get('cdf_mean', 0):8.4f}  "
                  f"{r['mask_ratio']:5.1%}  {r['monitor']:8.4f}")
    print(f"{'=' * 80}")
    print(f"\n  Output: {args.output_dir}/")
    if not args.no_guidance:
        print(f"  - baseline_no_guidance.png")
        print(f"  - guided.png")
    else:
        print(f"  - generated_no_guidance.png")
    print(f"  - summary_grid.png")
    print(f"  - step_stats.csv")
    print(f"  - heatmaps/step*_{{z0hat,gradcam,spatial_mask,overlay_*}}.png")


if __name__ == "__main__":
    main()
