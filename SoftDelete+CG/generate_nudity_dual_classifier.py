#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nudity Dual Classifier: 3-class Monitoring + 4-class Guidance

The problem with single classifier:
  - Guidance gradient acts as adversarial attack on the classifier
  - Classifier gets "fooled" - thinks latent is safe when image is still harmful

Solution: Use two different classifiers
  - 3-class classifier for MONITORING (deciding whether to guide)
  - 4-class classifier for GUIDANCE (computing gradient)

Since 3-class wasn't trained with 4-class gradient, it won't be fooled.
"""

import os
import sys
import json
import random
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, List

from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torch.distributions import Normal

from geo_models.classifier.classifier import load_discriminator
from geo_utils.classifier_interpretability import ClassifierGradCAM


# 3-class config (for monitoring)
THREE_CLASS_CONFIG = {
    "benign": 0, "safe_clothed": 1, "harm_nude": 2,
    "safe_classes": [0, 1], "harm_classes": [2],
    "class_names": {0: "benign", 1: "safe_clothed", 2: "harm_nude"}
}

# 4-class config (for guidance)
FOUR_CLASS_CONFIG = {
    "benign": 0, "safe_clothed": 1, "harm_nude": 2, "harm_color": 3,
    "safe_classes": [0, 1], "harm_classes": [2, 3], "guidance_target_safe": 1,
    "class_names": {0: "benign", 1: "safe_clothed", 2: "harm_nude", 3: "harm_color"}
}


def load_gradcam_stats(stats_dir: str) -> Dict:
    """Load topk and sample-level statistics for 4-class (guidance)."""
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


class DualClassifierMonitor:
    """
    Monitoring using 3-class classifier with argmax-based decision.

    3-class: [benign, safe/clothed, harm/nude]
    - Class 2 is harmful
    - Guide if argmax(logits) == 2 (classifier predicts harm)
    """

    def __init__(self, classifier_3class, device: str = "cuda"):
        self.classifier = classifier_3class.to(device)
        self.classifier.eval()
        self.device = device
        self.dtype = next(self.classifier.parameters()).dtype

        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

        self.reset_state()

    def reset_state(self):
        """Reset state for a new sample."""
        self.stats = {
            "total_steps": 0,
            "guided_steps": 0,
            "skipped_steps": 0,
            "step_history": []
        }

    def compute_p_harm(self, latent: torch.Tensor, timestep: torch.Tensor) -> tuple:
        """
        Compute P(harm) using 3-class classifier softmax output.
        Returns: (p_harm, logits)
        """
        lat = latent.to(dtype=self.dtype)
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        norm_t = timestep.float() / 1000.0

        with torch.no_grad():
            logits = self.classifier(lat, norm_t)
            probs = F.softmax(logits, dim=-1)
            p_harm = probs[0, 2].item()  # Class 2 = harm_nude

        return p_harm, logits[0].cpu().tolist()

    def should_apply_guidance(self, latent: torch.Tensor, timestep: torch.Tensor,
                               step: int) -> tuple:
        """
        Determine if guidance should be applied based on 3-class argmax.
        If argmax(logits) == 2 (harm class), apply guidance.
        Returns: (should_guide, info_dict)
        """
        self.stats["total_steps"] += 1

        p_harm, logits = self.compute_p_harm(latent, timestep)
        pred_class = int(torch.tensor(logits).argmax().item())

        info = {
            "step": step,
            "p_harm": p_harm,
            "logits": logits,
            "pred_class": pred_class,
            "class_name": {0: "benign", 1: "safe", 2: "harm"}[pred_class]
        }

        # Guide if classifier predicts harm (class 2)
        should_guide = (pred_class == 2)

        if should_guide:
            self.stats["guided_steps"] += 1
            self.stats["step_history"].append({**info, "guided": True})
        else:
            self.stats["skipped_steps"] += 1
            self.stats["step_history"].append({**info, "guided": False})

        return should_guide, info


class SpatialGuidance4Class:
    """
    Spatial CG using 4-class classifier for gradient computation.

    4-class: [benign, safe_clothed, harm_nude, harm_color]
    - Classes 2 and 3 are harmful
    - Gradient: push towards class 1 (safe_clothed), away from classes 2, 3
    """

    def __init__(self, classifier_4class, stats_map_4class: Dict,
                 gradcam_layer: str, device: str = "cuda"):
        self.classifier = classifier_4class.to(device)
        self.classifier.eval()
        self.device = device
        self.dtype = next(self.classifier.parameters()).dtype
        self.stats_map = stats_map_4class
        self.gradcam = ClassifierGradCAM(classifier_4class, gradcam_layer)
        self.normal = Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))

        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

    def _pixel_cdf_normalize(self, heatmap: torch.Tensor, harm_class: int) -> torch.Tensor:
        """Apply CDF normalization using top-K pixel statistics."""
        mu = self.stats_map[harm_class]["topk_mean"]
        sigma = self.stats_map[harm_class]["topk_std"]
        z = (heatmap - mu) / (sigma + 1e-8)
        return self.normal.cdf(z)

    def compute_gradient(self, latent: torch.Tensor, timestep: torch.Tensor,
                         spatial_threshold: float,
                         guidance_scale: float = 5.0, base_scale: float = 0.0,
                         harmful_scale: float = 1.0) -> torch.Tensor:
        """
        Compute spatial guidance gradient using 4-class classifier.
        Always considers both harm classes (2 and 3).

        Gradient: g_safe - harmful_scale * g_harm
        Weight: mask * guidance_scale + (1-mask) * base_scale
        """
        lat = latent.to(dtype=self.dtype)
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        norm_t = timestep.float() / 1000.0

        # Generate masks for harm class (nude only, not color artifacts)
        harm_classes = [2]  # harm_nude only
        masks = {}
        for hc in harm_classes:
            if hc not in self.stats_map:
                continue
            with torch.enable_grad():
                heatmap, _ = self.gradcam.generate_heatmap(lat, norm_t, hc, normalize=False)
            heatmap_norm = self._pixel_cdf_normalize(heatmap, hc)
            mask = (heatmap_norm >= spatial_threshold).float()
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            masks[hc] = mask

        # Compute gradients: g_safe - harmful_scale * g_harm
        with torch.enable_grad():
            l1 = latent.detach().to(dtype=self.dtype).requires_grad_(True)
            g_safe = torch.autograd.grad(self.classifier(l1, norm_t)[:, 1].sum(), l1)[0]

            g_harm = torch.zeros_like(g_safe)
            for hc in harm_classes:
                if hc not in self.stats_map:
                    continue
                l2 = latent.detach().to(dtype=self.dtype).requires_grad_(True)
                g_harm += torch.autograd.grad(self.classifier(l2, norm_t)[:, hc].sum(), l2)[0]

            grad = g_safe - harmful_scale * g_harm

        # Combine masks from both harm classes
        combined_mask = None
        for hc in masks:
            m = masks[hc]
            combined_mask = m if combined_mask is None else torch.max(combined_mask, m)

        if combined_mask is None:
            combined_mask = torch.zeros_like(latent[:, 0:1, :, :])

        # Apply spatial weighting
        weight = combined_mask * guidance_scale + (1 - combined_mask) * base_scale
        return (grad * weight).to(dtype=latent.dtype).detach()


def get_spatial_threshold(step: int, total: int, start: float, end: float, strategy: str = "cosine") -> float:
    """Compute spatial threshold based on step."""
    t = step / max(total - 1, 1)
    if strategy == "constant":
        return start
    elif strategy == "linear":
        return start - (start - end) * t
    elif strategy == "cosine":
        return end + (start - end) * 0.5 * (1 + np.cos(np.pi * t))
    return start


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./scg_outputs/dual_classifier")
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # 3-class classifier (for monitoring) - uses softmax, no stats needed
    parser.add_argument("--classifier_3class_ckpt", type=str, required=True,
                        help="3-class classifier checkpoint for monitoring")

    # 4-class classifier (for guidance) - uses GradCAM for spatial masking
    parser.add_argument("--classifier_4class_ckpt", type=str, required=True,
                        help="4-class classifier checkpoint for guidance")
    parser.add_argument("--gradcam_stats_dir", type=str, required=True,
                        help="GradCAM stats directory for 4-class classifier")

    parser.add_argument("--gradcam_layer", type=str, default="encoder_model.middle_block.2")

    # Monitoring (using 3-class) - now argmax-based, threshold not used
    # Kept for backward compatibility but ignored
    parser.add_argument("--monitoring_threshold", type=float, default=0.5,
                        help="[DEPRECATED] Not used - monitoring uses argmax instead")

    # Guidance (using 4-class)
    parser.add_argument("--guidance_scale", type=float, default=10.0,
                        help="Scale for spatial weighting in harm regions")
    parser.add_argument("--base_guidance_scale", type=float, default=2.0,
                        help="Scale for spatial weighting in non-harm regions")
    parser.add_argument("--harmful_scale", type=float, default=1.0,
                        help="Multiplier for harm gradient (grad = g_safe - harmful_scale * g_harm)")
    parser.add_argument("--spatial_threshold_start", type=float, default=0.5)
    parser.add_argument("--spatial_threshold_end", type=float, default=0.1)
    parser.add_argument("--spatial_threshold_strategy", type=str, default="cosine")

    # Step range
    parser.add_argument("--guidance_start_step", type=int, default=0)
    parser.add_argument("--guidance_end_step", type=int, default=50)

    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_prompts(f):
    """Load prompts from txt or CSV file.

    For CSV files: tries columns in priority order (matches SAFREE):
    - adv_prompt (MMA-Diffusion adversarial)
    - sensitive prompt (Ring-A-Bell)
    - prompt (I2P, UnlearnDiff, etc.)
    - target_prompt (MMA-Diffusion original)
    """
    import csv
    f = Path(f)
    if f.suffix == ".csv":
        prompts = []
        with open(f, "r") as fp:
            reader = csv.DictReader(fp)
            fieldnames = reader.fieldnames

            # Column priority (matches SAFREE: generate_safree.py)
            column_priority = [
                'adv_prompt',       # MMA-Diffusion adversarial
                'sensitive prompt', # Ring-A-Bell
                'prompt',           # I2P, UnlearnDiff, etc.
                'target_prompt',    # MMA-Diffusion original
                'text', 'Prompt', 'Text'  # fallbacks
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


def main():
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator()
    device = accelerator.device

    print(f"\n{'='*60}")
    print(f"DUAL CLASSIFIER: 3-class Monitoring + 4-class Guidance")
    print(f"{'='*60}")
    print(f"3-class (monitoring): {args.classifier_3class_ckpt}")
    print(f"4-class (guidance):   {args.classifier_4class_ckpt}")
    print(f"Monitoring: argmax-based (guide if pred == harm)")
    print(f"Guidance scale: {args.guidance_scale} (base: {args.base_guidance_scale})")
    print(f"Harmful scale: {args.harmful_scale}")
    print(f"Spatial threshold: {args.spatial_threshold_start} -> {args.spatial_threshold_end}")
    print(f"{'='*60}\n")

    # Load stats for 4-class (guidance) - 3-class uses softmax, no stats needed
    stats_4class = load_gradcam_stats(args.gradcam_stats_dir)

    if not stats_4class:
        raise RuntimeError(f"No 4-class stats found in {args.gradcam_stats_dir}")

    print("3-class monitoring: argmax-based (guide when pred == harm)")
    print("\nLoaded 4-class GradCAM stats (for guidance):")
    for cls, s in stats_4class.items():
        print(f"  Class {cls}: topk_mean={s['topk_mean']:.4f}, topk_std={s['topk_std']:.4f}")

    # Load prompts
    prompts = load_prompts(args.prompt_file)
    print(f"\nLoaded {len(prompts)} prompts")

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Load 3-class classifier (monitoring)
    classifier_3class = load_discriminator(
        ckpt_path=args.classifier_3class_ckpt, condition=None, eval=True,
        channel=4, num_classes=3
    ).to(device)
    classifier_3class.eval()
    print(f"Loaded 3-class classifier for monitoring")

    # Load 4-class classifier (guidance)
    classifier_4class = load_discriminator(
        ckpt_path=args.classifier_4class_ckpt, condition=None, eval=True,
        channel=4, num_classes=4
    ).to(device)
    classifier_4class.eval()
    print(f"Loaded 4-class classifier for guidance")

    # Initialize modules
    monitor = DualClassifierMonitor(classifier_3class, device)
    guidance = SpatialGuidance4Class(classifier_4class, stats_4class, args.gradcam_layer, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []

    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Generating")):
        for sample_idx in range(args.nsamples):
            current_seed = args.seed + prompt_idx * args.nsamples + sample_idx
            set_seed(current_seed)

            monitor.reset_state()

            def callback_on_step_end(pipe, step, timestep, callback_kwargs):
                latents = callback_kwargs["latents"]

                if args.guidance_start_step <= step <= args.guidance_end_step:
                    # Monitoring with 3-class classifier (argmax-based)
                    should_guide, info = monitor.should_apply_guidance(latents, timestep, step)

                    if args.debug and step % 10 == 0:
                        pred = info["class_name"]
                        p_harm = info["p_harm"]
                        print(f"  Step {step:2d}: pred={pred}, P(harm)={p_harm:.3f} -> {'GUIDE' if should_guide else 'skip'}")

                    if should_guide:
                        # Guidance with 4-class classifier
                        spatial_thr = get_spatial_threshold(
                            step, args.num_inference_steps,
                            args.spatial_threshold_start, args.spatial_threshold_end,
                            args.spatial_threshold_strategy
                        )
                        grad = guidance.compute_gradient(
                            latents, timestep, spatial_thr,
                            args.guidance_scale, args.base_guidance_scale,
                            args.harmful_scale
                        )
                        callback_kwargs["latents"] = latents + grad

                return callback_kwargs

            with torch.no_grad():
                output = pipe(
                    prompt=prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.cfg_scale,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=["latents"]
                )

            safe_prompt = "".join(c if c.isalnum() or c in ' -_' else '_' for c in prompt)[:50].replace(' ', '_')
            img_filename = f"{prompt_idx:04d}_{sample_idx:02d}_{safe_prompt}.png"
            save_image(output.images[0], output_dir / img_filename)

            img_stats = {
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "seed": current_seed,
                "prompt": prompt[:100],
                "filename": img_filename,
                "guided_steps": monitor.stats["guided_steps"],
                "skipped_steps": monitor.stats["skipped_steps"],
                "total_steps": monitor.stats["total_steps"],
                "guidance_ratio": monitor.stats["guided_steps"] / max(monitor.stats["total_steps"], 1),
            }
            if args.debug:
                img_stats["step_history"] = monitor.stats["step_history"]
            all_stats.append(img_stats)

            print(f"  [{prompt_idx:03d}] Guided: {img_stats['guided_steps']}/{img_stats['total_steps']} "
                  f"({img_stats['guidance_ratio']*100:.1f}%)")

    # Compute overall summary
    total_images = len(all_stats)
    avg_guided = np.mean([s["guided_steps"] for s in all_stats]) if all_stats else 0
    avg_skipped = np.mean([s["skipped_steps"] for s in all_stats]) if all_stats else 0
    avg_ratio = np.mean([s["guidance_ratio"] for s in all_stats]) if all_stats else 0

    no_guidance = sum(1 for s in all_stats if s["guided_steps"] == 0)
    light_guidance = sum(1 for s in all_stats if 0 < s["guidance_ratio"] <= 0.3)
    medium_guidance = sum(1 for s in all_stats if 0.3 < s["guidance_ratio"] <= 0.7)
    heavy_guidance = sum(1 for s in all_stats if s["guidance_ratio"] > 0.7)

    # Save summary stats
    summary = {
        "args": vars(args),
        "overall": {
            "total_images": total_images,
            "avg_guided_steps": float(avg_guided),
            "avg_skipped_steps": float(avg_skipped),
            "avg_guidance_ratio": float(avg_ratio),
            "no_guidance_count": no_guidance,
            "light_guidance_count": light_guidance,
            "medium_guidance_count": medium_guidance,
            "heavy_guidance_count": heavy_guidance,
        },
        "per_image_stats": all_stats,
    }
    with open(output_dir / "generation_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Total images: {total_images}")
    print(f"\n[Dual Classifier Mode]")
    print(f"  Monitoring: 3-class classifier")
    print(f"  Guidance:   4-class classifier")
    print(f"\n[Guidance Statistics]")
    print(f"  Avg guided steps: {avg_guided:.1f}/{args.num_inference_steps} ({avg_ratio*100:.1f}%)")
    print(f"  Avg skipped steps: {avg_skipped:.1f}/{args.num_inference_steps}")
    print(f"\n[Guidance Distribution]")
    print(f"  No guidance (0%):      {no_guidance:3d} images ({no_guidance/max(total_images,1)*100:.1f}%)")
    print(f"  Light (1-30%):         {light_guidance:3d} images ({light_guidance/max(total_images,1)*100:.1f}%)")
    print(f"  Medium (31-70%):       {medium_guidance:3d} images ({medium_guidance/max(total_images,1)*100:.1f}%)")
    print(f"  Heavy (71-100%):       {heavy_guidance:3d} images ({heavy_guidance/max(total_images,1)*100:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
