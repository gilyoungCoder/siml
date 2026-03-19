#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
True std_sgf + ASCG: Combines SGF's MMD repellency with ASCG's spatial classifier guidance.

Key differences from generate_sgf_window_unified.py:
  1. SGF MMD repellency gradient (closed-form RBF kernel gradient against unsafe reference set)
  2. x0-space guidance: predict x0_hat -> apply gradients -> re-noise (as in SGF paper)
  3. sigma(t)-proportional schedule from CBF theory (not hand-crafted cosine/linear)
  4. Gradient norm clipping for stability

Pipeline per timestep (when in guidance window):
  UNet -> CFG -> predict x0_hat -> ASCG spatial grad + SGF MMD grad -> re-noise -> DDIM step
"""

import os
import sys
import json
import math
import random
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional

from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torch.distributions import Normal

from geo_models.classifier.classifier import load_discriminator
from geo_utils.classifier_interpretability import ClassifierGradCAM


NUDITY_4CLASS_CONFIG = {
    "benign": 0, "safe_clothed": 1, "harm_nude": 2, "harm_color": 3,
    "safe_classes": [0, 1], "harm_classes": [2, 3], "guidance_target_safe": 1,
    "class_names": {0: "benign", 1: "safe_clothed", 2: "harm_nude", 3: "harm_color"}
}


def load_gradcam_stats(stats_dir: str) -> Dict:
    stats_dir = Path(stats_dir)
    mapping = {2: "gradcam_stats_harm_nude_class2.json", 3: "gradcam_stats_harm_color_class3.json"}
    stats_map = {}
    for cls, fname in mapping.items():
        path = stats_dir / fname
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            topk = d.get("topk", {})
            sample = d.get("sample_level", {})
            stats_map[cls] = {
                "topk_mean": float(topk.get("mean", d["mean"])),
                "topk_std": float(topk.get("std", d["std"])),
                "sample_mean": float(sample.get("mean", d["mean"])),
                "sample_std": float(sample.get("std", d["std"])),
            }
    return stats_map


# =============================================================================
# SGF MMD Repellency (ported from SGF codebase)
# =============================================================================
class SGFRepellency:
    """SGF's MMD repellency: closed-form RBF kernel gradient against unsafe references."""

    def __init__(self, proj_ref_path: str, scale: float = 0.03,
                 max_norm: float = 1.0, device: str = "cuda"):
        self.scale = scale
        self.max_norm = max_norm
        self.device = device

        # Load pre-computed reference embeddings (VAE-encoded unsafe images)
        # Shape: (N_ref, 4, 64, 64)
        self.proj_refs = torch.load(proj_ref_path, map_location=device).float()
        print(f"[SGF] Loaded {self.proj_refs.shape[0]} reference embeddings "
              f"from {proj_ref_path} (shape: {self.proj_refs.shape})")

    @torch.no_grad()
    def compute_repellency(self, x_0_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD repellency gradient: dK/dX where K is RBF kernel.
        This gradient points AWAY from the reference set (unsafe images).

        Args:
            x_0_hat: predicted clean sample, shape (1, 4, 64, 64)

        Returns:
            Repellency gradient, same shape as x_0_hat
        """
        X = x_0_hat.float()
        Y = self.proj_refs.to(X.device)

        # Flatten to (N, D)
        X_flat = X.view(X.size(0), -1)
        Y_flat = Y.view(Y.size(0), -1)

        # Squared Euclidean distances
        with torch.cuda.amp.autocast():
            dists = torch.cdist(X_flat, Y_flat, p=2) ** 2

        # Adaptive bandwidth via top-k (k=3)
        k = 3
        sorted_d, _ = torch.sort(dists, dim=1)
        r1_to_k = sorted_d[:, 3:k + 3].reshape(-1)
        r_k2 = r1_to_k.mean()
        eps = 0.05
        gamma = -torch.log(torch.tensor(eps, device=X.device)) / r_k2

        # RBF kernel: K[i,j] = exp(-gamma * ||x_i - y_j||^2)
        K = torch.exp(-gamma * dists)  # (1, M)

        # Gradient: dK/dX[i] = sum_j -2*gamma*K[i,j]*(X[i] - Y[j])
        diff = X_flat.unsqueeze(1) - Y_flat.unsqueeze(0)  # (1, M, D)
        dK = -2 * gamma * K.unsqueeze(-1) * diff  # (1, M, D)
        dK_dX = dK.sum(dim=1)  # (1, D)

        # Reshape back
        dK_dX = dK_dX.view(*X.shape)

        # Gradient norm clipping
        if self.max_norm is not None and self.max_norm > 0:
            nrm = dK_dX.flatten(1).norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
            clip_scale = (self.max_norm / nrm).clamp(max=1.0).view(-1, 1, 1, 1)
            dK_dX = dK_dX * clip_scale

        return dK_dX


# =============================================================================
# ASCG Spatial Classifier Guidance (from generate_sgf_window_unified.py)
# =============================================================================
class SpatialGuidance:
    """ASCG: Spatial CG using CDF-normalized GradCAM."""

    def __init__(self, classifier_model, stats_map: Dict, gradcam_layer: str, device: str = "cuda"):
        self.classifier = classifier_model.to(device)
        self.classifier.eval()
        self.device = device
        self.dtype = next(self.classifier.parameters()).dtype
        self.stats_map = stats_map
        self.gradcam = ClassifierGradCAM(classifier_model, gradcam_layer)
        self.normal = Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))

    def _pixel_cdf_normalize(self, heatmap: torch.Tensor, harm_class: int) -> torch.Tensor:
        mu = self.stats_map[harm_class]["topk_mean"]
        sigma = self.stats_map[harm_class]["topk_std"]
        z = (heatmap - mu) / (sigma + 1e-8)
        return self.normal.cdf(z)

    def _compute_gradcam_mask(self, latent, norm_t, active_harm_classes, spatial_threshold):
        masks = {}
        for hc in active_harm_classes:
            with torch.enable_grad():
                heatmap, _ = self.gradcam.generate_heatmap(
                    latent.to(dtype=self.dtype), norm_t, hc, normalize=False
                )
            heatmap_norm = self._pixel_cdf_normalize(heatmap, hc)
            mask = (heatmap_norm >= spatial_threshold).float()
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            masks[hc] = mask

        combined = None
        for hc in active_harm_classes:
            m = masks[hc]
            combined = m if combined is None else torch.max(combined, m)
        return combined if combined is not None else torch.zeros_like(latent[:, 0:1, :, :])

    def compute_gradient(self, latent, timestep, active_harm_classes, spatial_threshold,
                         guidance_scale=10.0, base_scale=0.0, spatial_mode="gradcam"):
        """Compute ASCG dual classifier gradient with spatial masking."""
        if not active_harm_classes:
            return torch.zeros_like(latent)

        lat = latent.to(dtype=self.dtype)
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        norm_t = timestep.float() / 1000.0

        # Dual classifier gradient: g_safe - g_harm
        with torch.enable_grad():
            l1 = latent.detach().to(dtype=self.dtype).requires_grad_(True)
            g_safe = torch.autograd.grad(self.classifier(l1, norm_t)[:, 1].sum(), l1)[0]

            g_harm = torch.zeros_like(g_safe)
            for hc in active_harm_classes:
                l2 = latent.detach().to(dtype=self.dtype).requires_grad_(True)
                g_harm += torch.autograd.grad(self.classifier(l2, norm_t)[:, hc].sum(), l2)[0]

            grad = g_safe - g_harm

        # Spatial mask (GradCAM)
        combined_mask = self._compute_gradcam_mask(latent, norm_t, active_harm_classes, spatial_threshold)

        # Apply spatial weighting
        weight = combined_mask * guidance_scale + (1 - combined_mask) * base_scale
        final_grad = (grad * weight).to(dtype=latent.dtype).detach()
        return final_grad


# =============================================================================
# Utils
# =============================================================================
def get_spatial_threshold(step, total, start, end, strategy="cosine"):
    t = step / max(total - 1, 1)
    if strategy == "constant":
        return start
    elif strategy == "linear":
        return start - (start - end) * t
    elif strategy == "cosine":
        return end + (start - end) * 0.5 * (1 + np.cos(np.pi * t))
    return start


def get_sigma_for_timestep(scheduler, timestep):
    """Get noise level sigma for a given timestep from the DDIM scheduler.

    For DDIM, sigma(t) = sqrt((1 - alpha_bar_t) / alpha_bar_t).
    This is proportional to the noise level at timestep t.
    """
    step_index = (scheduler.timesteps == timestep).nonzero(as_tuple=True)[0]
    if len(step_index) == 0:
        # Fallback: linear interpolation
        return timestep.float() / 1000.0
    alpha_bar = scheduler.alphas_cumprod[timestep.long()]
    sigma = torch.sqrt((1 - alpha_bar) / alpha_bar)
    return sigma.item()


def load_prompts(f):
    import csv
    f = Path(f)
    if f.suffix == ".csv":
        prompts = []
        with open(f, "r") as fp:
            reader = csv.DictReader(fp)
            fieldnames = reader.fieldnames
            column_priority = [
                'adv_prompt', 'sensitive prompt', 'prompt', 'target_prompt',
                'text', 'Prompt', 'Text'
            ]
            prompt_col = None
            for col in column_priority:
                if col in fieldnames:
                    prompt_col = col
                    break
            if prompt_col is None:
                raise ValueError(f"CSV has no recognizable prompt column. Available: {fieldnames}")
            print(f"[INFO] Using column '{prompt_col}' from {f}")
            for row in reader:
                prompts.append(row[prompt_col].strip())
        return prompts
    else:
        return [l.strip() for l in open(f) if l.strip()]


def save_image(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.resize((512, 512)).save(path)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# =============================================================================
# Args
# =============================================================================
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./scg_outputs/std_sgf_ascg")
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)

    # Classifier (ASCG)
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--gradcam_layer", type=str, default="encoder_model.middle_block.2")
    parser.add_argument("--gradcam_stats_dir", type=str, required=True)

    # SGF MMD repellency
    parser.add_argument("--sgf_ref_path", type=str, required=True,
                        help="Path to SGF reference embeddings (repellency_proj_ref.pt)")
    parser.add_argument("--sgf_scale", type=float, default=0.03,
                        help="SGF repellency scale (from SGF config, default 0.03)")
    parser.add_argument("--sgf_max_norm", type=float, default=1.0,
                        help="Gradient norm clipping threshold for SGF")

    # Guidance window (CBF-derived)
    parser.add_argument("--guidance_start_t", type=int, default=1000,
                        help="Start guidance at this timestep (high=early denoising)")
    parser.add_argument("--guidance_end_t", type=int, default=800,
                        help="Stop guidance at this timestep (low=later denoising)")
    parser.add_argument("--guidance_schedule", type=str, default="sigma",
                        choices=["sigma", "constant", "cosine", "linear"],
                        help="Guidance strength schedule. 'sigma' = CBF-derived sigma(t)-proportional")

    # ASCG spatial guidance
    parser.add_argument("--spatial_mode", type=str, default="gradcam",
                        choices=["gradcam", "none"])
    parser.add_argument("--ascg_scale", type=float, default=10.0,
                        help="ASCG classifier guidance scale (inside spatial mask)")
    parser.add_argument("--ascg_base_scale", type=float, default=2.0,
                        help="ASCG guidance scale outside spatial mask")
    parser.add_argument("--ascg_max_norm", type=float, default=0.0,
                        help="Gradient norm clipping for ASCG (0=disabled)")
    parser.add_argument("--spatial_threshold_start", type=float, default=0.3)
    parser.add_argument("--spatial_threshold_end", type=float, default=0.1)
    parser.add_argument("--spatial_threshold_strategy", type=str, default="cosine")

    # Mode flags
    parser.add_argument("--use_sgf", action="store_true", default=True,
                        help="Enable SGF MMD repellency")
    parser.add_argument("--no_sgf", action="store_true",
                        help="Disable SGF (ASCG only)")
    parser.add_argument("--use_ascg", action="store_true", default=True,
                        help="Enable ASCG classifier guidance")
    parser.add_argument("--no_ascg", action="store_true",
                        help="Disable ASCG (SGF only)")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    if args.no_sgf:
        args.use_sgf = False
    if args.no_ascg:
        args.use_ascg = False

    set_seed(args.seed)
    accelerator = Accelerator()
    device = accelerator.device

    print(f"\n{'='*60}")
    print(f"std_sgf + ASCG COMBINED PIPELINE")
    print(f"  SGF: {'ON' if args.use_sgf else 'OFF'} (scale={args.sgf_scale}, max_norm={args.sgf_max_norm})")
    print(f"  ASCG: {'ON' if args.use_ascg else 'OFF'} (scale={args.ascg_scale}, base={args.ascg_base_scale})")
    print(f"  Window: t=[{args.guidance_end_t}, {args.guidance_start_t}]")
    print(f"  Schedule: {args.guidance_schedule}")
    print(f"  HYBRID: SGF in x0-space (predict x0_hat -> MMD -> re-noise) + ASCG in z_t-space (after DDIM)")
    print(f"{'='*60}\n")

    # Load GradCAM stats
    stats_map = load_gradcam_stats(args.gradcam_stats_dir)
    if not stats_map and args.use_ascg:
        raise RuntimeError(f"No stats found in {args.gradcam_stats_dir}")

    # Load prompts
    all_prompts = load_prompts(args.prompt_file)
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx > 0 else len(all_prompts)
    prompts_with_idx = list(enumerate(all_prompts))[start_idx:end_idx]
    print(f"Loaded {len(all_prompts)} prompts, processing [{start_idx}:{end_idx}] = {len(prompts_with_idx)}")

    # Load SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    unet = pipe.unet
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    scheduler = pipe.scheduler

    # Load ASCG classifier
    if args.use_ascg:
        classifier = load_discriminator(
            ckpt_path=args.classifier_ckpt, condition=None, eval=True, channel=4, num_classes=4
        ).to(device)
        classifier.eval()
        guidance = SpatialGuidance(classifier, stats_map, args.gradcam_layer, device)
        print(f"[ASCG] Classifier loaded from {args.classifier_ckpt}")
    else:
        guidance = None

    # Load SGF repellency
    if args.use_sgf:
        sgf = SGFRepellency(
            proj_ref_path=args.sgf_ref_path,
            scale=args.sgf_scale,
            max_norm=args.sgf_max_norm,
            device=device,
        )
    else:
        sgf = None

    # Pre-compute sigma values for all timesteps (for sigma schedule)
    scheduler.set_timesteps(args.num_inference_steps, device=device)
    sigma_map = {}
    for t in scheduler.timesteps:
        t_val = t.item()
        alpha_bar = scheduler.alphas_cumprod[int(t_val)]
        sigma_map[t_val] = math.sqrt((1 - alpha_bar.item()) / alpha_bar.item())
    # Normalize sigma values to [0, 1] for schedule weighting
    sigma_max = max(sigma_map.values())
    sigma_min = min(sigma_map.values())
    print(f"[Schedule] sigma range: [{sigma_min:.4f}, {sigma_max:.4f}]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []

    for prompt_idx, prompt in tqdm(prompts_with_idx, desc="Generating"):
        for sample_idx in range(args.nsamples):
            current_seed = args.seed + prompt_idx * args.nsamples + sample_idx
            set_seed(current_seed)

            guided_steps_count = 0
            skipped_steps_count = 0
            step_history = []

            # Encode text for CFG
            with torch.no_grad():
                text_inputs = tokenizer(
                    prompt, padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt"
                )
                text_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
                uncond_inputs = tokenizer(
                    "", padding="max_length",
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt"
                )
                uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]
            prompt_embeds = torch.cat([uncond_embeds, text_embeds])

            # Initialize latents
            set_seed(current_seed)
            latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
            latents = latents * scheduler.init_noise_sigma
            scheduler.set_timesteps(args.num_inference_steps, device=device)

            # ====================================================
            # Denoising loop with true std_sgf + ASCG
            # ====================================================
            for step_idx, t in enumerate(scheduler.timesteps):
                t_val = t.item()

                # 1. UNet forward
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                with torch.no_grad():
                    noise_pred_raw = unet(
                        latent_model_input, t,
                        encoder_hidden_states=prompt_embeds
                    ).sample

                # 2. CFG
                noise_pred_uncond, noise_pred_text = noise_pred_raw.chunk(2)
                noise_pred = noise_pred_uncond + args.cfg_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # 3. Check if we're in guidance window
                in_window = (args.guidance_end_t <= t_val <= args.guidance_start_t)

                # Compute guidance weight based on schedule
                if in_window:
                    if args.guidance_schedule == "sigma":
                        # CBF-derived: weight proportional to sigma(t) / sigma_max
                        sigma_t = sigma_map.get(t_val, t_val / 1000.0)
                        window_weight = sigma_t / sigma_max
                    elif args.guidance_schedule == "constant":
                        window_weight = 1.0
                    elif args.guidance_schedule == "cosine":
                        window_size = args.guidance_start_t - args.guidance_end_t
                        progress = (args.guidance_start_t - t_val) / max(window_size, 1)
                        window_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
                    elif args.guidance_schedule == "linear":
                        window_size = args.guidance_start_t - args.guidance_end_t
                        progress = (args.guidance_start_t - t_val) / max(window_size, 1)
                        window_weight = 1.0 - progress
                    else:
                        window_weight = 1.0
                else:
                    window_weight = 0.0

                info = {
                    "step": step_idx,
                    "timestep": t_val,
                    "in_window": in_window,
                    "window_weight": round(window_weight, 4),
                }

                # 4. Apply guidance (if in window)
                # HYBRID: SGF in x0-space (as in SGF paper), ASCG in z_t-space (proven in v1)
                if in_window and window_weight > 0.01:
                    applied_any = False

                    # 4a. SGF MMD repellency in x0-space (predict x0_hat -> repel -> re-noise)
                    if sgf is not None:
                        step_output = scheduler.step(noise_pred, t, latents)
                        x_0_hat = step_output.pred_original_sample.float()

                        sgf_grad = sgf.compute_repellency(x_0_hat)
                        x_0_hat = x_0_hat + sgf.scale * window_weight * sgf_grad

                        # Re-noise modified x0_hat back to z_t
                        noise = torch.randn(x_0_hat.shape, device=device, dtype=torch.float32)
                        latents = scheduler.add_noise(
                            x_0_hat.to(latents.dtype), noise.to(latents.dtype), t
                        )
                        applied_any = True
                        info["sgf_applied"] = True
                        info["sgf_grad_norm"] = sgf_grad.flatten(1).norm(p=2).item()

                    if applied_any:
                        guided_steps_count += 1
                        info["guided"] = True
                    else:
                        skipped_steps_count += 1
                        info["guided"] = False
                else:
                    skipped_steps_count += 1
                    info["guided"] = False

                # 5. DDIM step (on current latents, possibly modified by SGF re-noising)
                latents = scheduler.step(noise_pred, t, latents).prev_sample

                # 6. ASCG classifier gradient in z_{t-1} space (after DDIM step)
                # Classifier was trained on noisy latents, so it works properly here
                if in_window and window_weight > 0.01 and guidance is not None:
                    active_classes = [2, 3]
                    spatial_thr = get_spatial_threshold(
                        step_idx, args.num_inference_steps,
                        args.spatial_threshold_start,
                        args.spatial_threshold_end,
                        args.spatial_threshold_strategy
                    )
                    effective_scale = args.ascg_scale * window_weight
                    effective_base = args.ascg_base_scale * window_weight

                    ascg_grad = guidance.compute_gradient(
                        latents, t, active_classes, spatial_thr,
                        effective_scale, effective_base,
                        spatial_mode=args.spatial_mode
                    )

                    # Optional norm clipping for ASCG
                    if args.ascg_max_norm > 0:
                        nrm = ascg_grad.flatten(1).norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
                        clip_scale = (args.ascg_max_norm / nrm).clamp(max=1.0).view(-1, 1, 1, 1)
                        ascg_grad = ascg_grad * clip_scale

                    latents = latents + ascg_grad
                    info["ascg_applied"] = True
                    info["ascg_grad_norm"] = ascg_grad.flatten(1).norm(p=2).item()
                    info["effective_ascg_scale"] = effective_scale

                step_history.append(info)

                if args.debug and step_idx % 5 == 0:
                    status = "GUIDED" if info.get("guided", False) else "skip"
                    print(f"  Step {step_idx}: t={t_val}, w={window_weight:.3f}, {status}")

            # Decode latents to image
            with torch.no_grad():
                latents_dec = 1.0 / vae.config.scaling_factor * latents
                image = vae.decode(latents_dec.to(vae.dtype)).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()
                image = (image[0] * 255).round().astype(np.uint8)

            safe_prompt = "".join(
                c if c.isalnum() or c in ' -_' else '_'
                for c in prompt
            )[:50].replace(' ', '_')
            img_filename = f"{prompt_idx:04d}_{sample_idx:02d}_{safe_prompt}.png"
            save_image(image, output_dir / img_filename)

            total_steps = guided_steps_count + skipped_steps_count

            img_stats = {
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "seed": current_seed,
                "prompt": prompt[:100],
                "filename": img_filename,
                "guidance_window": f"t=[{args.guidance_end_t},{args.guidance_start_t}]",
                "guidance_schedule": args.guidance_schedule,
                "use_sgf": args.use_sgf,
                "use_ascg": args.use_ascg,
                "guided_steps": guided_steps_count,
                "skipped_steps": skipped_steps_count,
                "total_steps": total_steps,
                "guidance_ratio": guided_steps_count / max(total_steps, 1),
            }
            if args.debug:
                img_stats["step_history"] = step_history
            all_stats.append(img_stats)

            print(
                f"  [{prompt_idx:03d}] Window t=[{args.guidance_end_t},{args.guidance_start_t}] "
                f"Guided: {guided_steps_count}/{total_steps} "
                f"({img_stats['guidance_ratio']*100:.1f}%)"
            )

    # Summary
    total_images = len(all_stats)
    avg_guided = np.mean([s["guided_steps"] for s in all_stats]) if all_stats else 0
    avg_ratio = np.mean([s["guidance_ratio"] for s in all_stats]) if all_stats else 0

    summary = {
        "args": vars(args),
        "overall": {
            "total_images": total_images,
            "avg_guided_steps": float(avg_guided),
            "avg_guidance_ratio": float(avg_ratio),
        },
        "per_image_stats": all_stats,
    }
    with open(output_dir / "generation_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Total images: {total_images}")
    print(f"SGF: {'ON' if args.use_sgf else 'OFF'}, ASCG: {'ON' if args.use_ascg else 'OFF'}")
    print(f"Window: t=[{args.guidance_end_t},{args.guidance_start_t}], schedule={args.guidance_schedule}")
    print(f"Avg guided steps: {avg_guided:.1f}/{args.num_inference_steps} ({avg_ratio*100:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
