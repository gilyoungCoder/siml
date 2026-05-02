#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MMD-Gated ASCG: SGF decides WHEN, ASCG decides WHERE.

Pipeline:
  1. UNet forward + CFG
  2. Predict x0_hat from noise prediction
  3. Compute MMD distance between x0_hat and unsafe reference set (SGF's "when")
  4. If MMD < threshold (close to unsafe): apply ASCG spatial guidance (our "where")
  5. DDIM step

Key idea: SGF's MMD acts as a DETECTOR (not gradient), ASCG provides spatial correction.
This should solve the COCO FP problem - benign images won't be close to unsafe refs.
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


# =============================================================================
# MMD Detector: computes distance to unsafe ref set (SGF's "when")
# =============================================================================
class MMDDetector:
    """Computes MMD distance between current x0_hat and unsafe reference embeddings.
    Used as a gating signal - NOT for gradient computation."""

    def __init__(self, proj_ref_path: str, device: str = "cuda"):
        self.device = device
        self.proj_refs = torch.load(proj_ref_path, map_location=device).float()
        self.proj_refs_flat = self.proj_refs.view(self.proj_refs.size(0), -1)
        print(f"[MMD Detector] Loaded {self.proj_refs.shape[0]} unsafe refs "
              f"(shape: {self.proj_refs.shape})")

    @torch.no_grad()
    def compute_mmd_distance(self, x0_hat: torch.Tensor) -> float:
        """Compute MMD^2 between x0_hat and unsafe reference set.

        Returns a scalar: lower = closer to unsafe distribution.
        We compute: MMD^2 = E[k(x,x)] - 2*E[k(x,y)] + E[k(y,y)]
        where x = x0_hat, y = unsafe refs.
        """
        X = x0_hat.float()
        Y = self.proj_refs.to(X.device)

        X_flat = X.view(X.size(0), -1)
        Y_flat = Y.view(Y.size(0), -1)

        # Pairwise distances
        dists_xy = torch.cdist(X_flat, Y_flat, p=2) ** 2  # (1, M)
        dists_yy = torch.cdist(Y_flat, Y_flat, p=2) ** 2  # (M, M)

        # Adaptive bandwidth (same as SGF)
        all_dists = torch.cat([dists_xy.reshape(-1), dists_yy.reshape(-1)])
        sorted_d = torch.sort(all_dists)[0]
        # Use median heuristic for bandwidth
        median_dist = sorted_d[len(sorted_d) // 2]
        gamma = 1.0 / (median_dist + 1e-8)

        # RBF kernel values
        K_xy = torch.exp(-gamma * dists_xy).mean()   # E[k(x,y)]
        K_yy = torch.exp(-gamma * dists_yy).mean()   # E[k(y,y)]
        # K_xx = 1.0 (single sample, k(x,x)=1 for RBF)

        # MMD^2 = 1 - 2*E[k(x,y)] + E[k(y,y)]
        mmd2 = 1.0 - 2.0 * K_xy + K_yy
        return mmd2.item()

    @torch.no_grad()
    def compute_mean_kernel_similarity(self, x0_hat: torch.Tensor) -> float:
        """Simpler metric: mean RBF kernel similarity to unsafe refs.
        Higher = more similar to unsafe images."""
        X = x0_hat.float()
        Y = self.proj_refs.to(X.device)

        X_flat = X.view(X.size(0), -1)
        Y_flat = Y.view(Y.size(0), -1)

        dists = torch.cdist(X_flat, Y_flat, p=2) ** 2  # (1, M)

        # Adaptive bandwidth (top-k, same as SGF)
        k = 3
        sorted_d, _ = torch.sort(dists, dim=1)
        r1_to_k = sorted_d[:, min(3, sorted_d.size(1)-1):min(k+3, sorted_d.size(1))].reshape(-1)
        r_k2 = r1_to_k.mean()
        eps = 0.05
        gamma = -torch.log(torch.tensor(eps, device=X.device)) / (r_k2 + 1e-8)

        K = torch.exp(-gamma * dists)  # (1, M)
        return K.mean().item()

    @torch.no_grad()
    def compute_min_distance(self, x0_hat: torch.Tensor) -> float:
        """Minimum L2 distance to any unsafe reference (simpler alternative)."""
        X = x0_hat.float().view(1, -1)
        Y = self.proj_refs_flat.to(X.device)
        dists = torch.cdist(X, Y, p=2)  # (1, M)
        return dists.min().item()


# =============================================================================
# ASCG Spatial Classifier Guidance (the "where")
# =============================================================================
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

    def get_classifier_probs(self, latent, timestep):
        """Get classifier probabilities (for logging)."""
        lat = latent.to(dtype=self.dtype)
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        norm_t = timestep.float() / 1000.0
        with torch.no_grad():
            logits = self.classifier(lat, norm_t)
            probs = F.softmax(logits, dim=1)[0]
        return probs

    def compute_gradient(self, latent, timestep, active_harm_classes, spatial_threshold,
                         guidance_scale=10.0, base_scale=0.0, spatial_mode="gradcam"):
        if not active_harm_classes:
            return torch.zeros_like(latent)

        lat = latent.to(dtype=self.dtype)
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        norm_t = timestep.float() / 1000.0

        with torch.enable_grad():
            l1 = latent.detach().to(dtype=self.dtype).requires_grad_(True)
            g_safe = torch.autograd.grad(self.classifier(l1, norm_t)[:, 1].sum(), l1)[0]

            g_harm = torch.zeros_like(g_safe)
            for hc in active_harm_classes:
                l2 = latent.detach().to(dtype=self.dtype).requires_grad_(True)
                g_harm += torch.autograd.grad(self.classifier(l2, norm_t)[:, hc].sum(), l2)[0]

            grad = g_safe - g_harm

        combined_mask = self._compute_gradcam_mask(latent, norm_t, active_harm_classes, spatial_threshold)
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


def load_prompts(f):
    import csv
    f = Path(f)
    if f.suffix == ".csv":
        prompts = []
        with open(f, "r") as fp:
            reader = csv.DictReader(fp)
            fieldnames = reader.fieldnames
            for col in ['adv_prompt', 'sensitive prompt', 'prompt', 'target_prompt',
                        'text', 'Prompt', 'Text']:
                if col in fieldnames:
                    prompt_col = col
                    break
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
    parser.add_argument("--output_dir", type=str, default="./scg_outputs/mmd_gated_ascg")
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # MMD Detector (SGF's "when")
    parser.add_argument("--sgf_ref_path", type=str, required=True,
                        help="Path to SGF reference embeddings (repellency_proj_ref.pt)")
    parser.add_argument("--mmd_metric", type=str, default="kernel_sim",
                        choices=["mmd2", "kernel_sim", "min_dist"],
                        help="Which MMD metric to use for gating")
    parser.add_argument("--mmd_threshold", type=float, default=0.1,
                        help="Threshold for MMD gating. Meaning depends on metric.")
    parser.add_argument("--mmd_window_start", type=int, default=1000,
                        help="Only check MMD within this timestep window")
    parser.add_argument("--mmd_window_end", type=int, default=400,
                        help="Only check MMD within this timestep window")
    parser.add_argument("--mmd_sticky", action="store_true",
                        help="Once triggered, keep guidance on for all remaining steps")
    parser.add_argument("--mmd_decision_step", type=int, default=-1,
                        help="If >= 0, make one-time decision at this step index and stick with it")

    # ASCG Classifier Guidance (the "where")
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--gradcam_layer", type=str, default="encoder_model.middle_block.2")
    parser.add_argument("--gradcam_stats_dir", type=str, required=True)
    parser.add_argument("--ascg_scale", type=float, default=20.0)
    parser.add_argument("--ascg_base_scale", type=float, default=3.0)
    parser.add_argument("--spatial_threshold_start", type=float, default=0.2)
    parser.add_argument("--spatial_threshold_end", type=float, default=0.3)
    parser.add_argument("--spatial_threshold_strategy", type=str, default="cosine")

    # Guidance schedule (for ASCG strength within window)
    parser.add_argument("--guidance_schedule", type=str, default="linear",
                        choices=["constant", "cosine", "linear"])

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator()
    device = accelerator.device

    print(f"\n{'='*60}")
    print(f"MMD-Gated ASCG: SGF decides WHEN, ASCG decides WHERE")
    print(f"  MMD metric: {args.mmd_metric}, threshold: {args.mmd_threshold}")
    print(f"  MMD window: t=[{args.mmd_window_end}, {args.mmd_window_start}]")
    print(f"  MMD sticky: {args.mmd_sticky}")
    print(f"  ASCG scale: {args.ascg_scale}, base: {args.ascg_base_scale}")
    print(f"  Spatial: {args.spatial_threshold_start}->{args.spatial_threshold_end} ({args.spatial_threshold_strategy})")
    print(f"  Schedule: {args.guidance_schedule}")
    print(f"{'='*60}\n")

    # Load GradCAM stats
    stats_map = load_gradcam_stats(args.gradcam_stats_dir)

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

    # Load MMD detector
    mmd_detector = MMDDetector(proj_ref_path=args.sgf_ref_path, device=str(device))

    # Load ASCG classifier
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt, condition=None, eval=True, channel=4, num_classes=4
    ).to(device)
    classifier.eval()
    guidance = SpatialGuidance(classifier, stats_map, args.gradcam_layer, str(device))
    print(f"[ASCG] Classifier loaded from {args.classifier_ckpt}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []

    for prompt_idx, prompt in tqdm(prompts_with_idx, desc="Generating"):
        for sample_idx in range(args.nsamples):
            current_seed = args.seed + prompt_idx * args.nsamples + sample_idx
            set_seed(current_seed)

            guided_steps_count = 0
            skipped_steps_count = 0
            mmd_triggered = False  # For sticky mode
            step_history = []

            # Encode text
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

                # 3. Check MMD gating (SGF's "when")
                in_window = (args.mmd_window_end <= t_val <= args.mmd_window_start)
                should_guide = False
                mmd_value = None

                if in_window:
                    if mmd_triggered and args.mmd_sticky:
                        # Already triggered in sticky mode
                        should_guide = True
                    else:
                        # Predict x0_hat for MMD computation
                        with torch.no_grad():
                            step_output = scheduler.step(noise_pred, t, latents)
                            x0_hat = step_output.pred_original_sample.float()

                        # Compute MMD metric
                        if args.mmd_metric == "kernel_sim":
                            mmd_value = mmd_detector.compute_mean_kernel_similarity(x0_hat)
                            should_guide = (mmd_value > args.mmd_threshold)
                        elif args.mmd_metric == "mmd2":
                            mmd_value = mmd_detector.compute_mmd_distance(x0_hat)
                            should_guide = (mmd_value < args.mmd_threshold)
                        elif args.mmd_metric == "min_dist":
                            mmd_value = mmd_detector.compute_min_distance(x0_hat)
                            should_guide = (mmd_value < args.mmd_threshold)

                        if should_guide and args.mmd_sticky:
                            mmd_triggered = True

                        # Decision step mode
                        if args.mmd_decision_step >= 0 and step_idx == args.mmd_decision_step:
                            mmd_triggered = should_guide
                        if args.mmd_decision_step >= 0 and step_idx > args.mmd_decision_step:
                            should_guide = mmd_triggered

                info = {
                    "step": step_idx,
                    "timestep": t_val,
                    "in_window": in_window,
                    "mmd_value": round(mmd_value, 6) if mmd_value is not None else None,
                    "should_guide": should_guide,
                }

                # 4. DDIM step
                latents = scheduler.step(noise_pred, t, latents).prev_sample

                # 5. Apply ASCG spatial guidance if MMD says so (the "where")
                if should_guide:
                    # Compute guidance weight from schedule
                    window_size = args.mmd_window_start - args.mmd_window_end
                    progress = (args.mmd_window_start - t_val) / max(window_size, 1)

                    if args.guidance_schedule == "linear":
                        window_weight = 1.0 - progress
                    elif args.guidance_schedule == "cosine":
                        window_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
                    else:
                        window_weight = 1.0

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
                        spatial_mode="gradcam"
                    )
                    latents = latents + ascg_grad

                    guided_steps_count += 1
                    info["guided"] = True
                    info["ascg_grad_norm"] = ascg_grad.flatten(1).norm(p=2).item()
                    info["window_weight"] = round(window_weight, 4)
                    info["effective_scale"] = round(effective_scale, 2)

                    # Also log classifier probs
                    probs = guidance.get_classifier_probs(latents, t)
                    info["p_harm"] = round((probs[2] + probs[3]).item(), 4)
                else:
                    skipped_steps_count += 1
                    info["guided"] = False

                step_history.append(info)

                if args.debug and step_idx % 5 == 0:
                    mmd_str = f"mmd={mmd_value:.4f}" if mmd_value is not None else "mmd=N/A"
                    status = "GUIDED" if info.get("guided", False) else "skip"
                    print(f"  Step {step_idx}: t={t_val}, {mmd_str}, {status}")

            # Decode
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

            # Collect MMD values for logging
            mmd_values = [s["mmd_value"] for s in step_history if s["mmd_value"] is not None]

            img_stats = {
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "seed": current_seed,
                "prompt": prompt[:100],
                "filename": img_filename,
                "guided_steps": guided_steps_count,
                "skipped_steps": skipped_steps_count,
                "total_steps": total_steps,
                "guidance_ratio": guided_steps_count / max(total_steps, 1),
                "mmd_triggered": mmd_triggered,
                "avg_mmd": float(np.mean(mmd_values)) if mmd_values else 0.0,
                "max_mmd": float(np.max(mmd_values)) if mmd_values else 0.0,
                "min_mmd": float(np.min(mmd_values)) if mmd_values else 0.0,
            }
            if args.debug:
                img_stats["step_history"] = step_history
            all_stats.append(img_stats)

            print(
                f"  [{prompt_idx:03d}] Guided: {guided_steps_count}/{total_steps} "
                f"({img_stats['guidance_ratio']*100:.1f}%) "
                f"mmd_avg={img_stats['avg_mmd']:.4f} triggered={mmd_triggered}"
            )

    # Summary
    total_images = len(all_stats)
    avg_guided = np.mean([s["guided_steps"] for s in all_stats]) if all_stats else 0
    avg_ratio = np.mean([s["guidance_ratio"] for s in all_stats]) if all_stats else 0
    n_triggered = sum(1 for s in all_stats if s["guided_steps"] > 0)

    summary = {
        "args": vars(args),
        "overall": {
            "total_images": total_images,
            "avg_guided_steps": float(avg_guided),
            "avg_skipped_steps": float(np.mean([s["skipped_steps"] for s in all_stats])) if all_stats else 0,
            "avg_guidance_ratio": float(avg_ratio),
            "no_guidance_count": sum(1 for s in all_stats if s["guided_steps"] == 0),
            "triggered_count": n_triggered,
        },
        "per_image_stats": all_stats,
    }
    with open(output_dir / "generation_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"MMD-GATED ASCG COMPLETE!")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Total images: {total_images}")
    print(f"Triggered: {n_triggered}/{total_images} ({100*n_triggered/max(total_images,1):.1f}%)")
    print(f"Avg guided steps: {avg_guided:.1f}/{args.num_inference_steps}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
