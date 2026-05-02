#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nudity 4-Class Adaptive Spatial Classifier Guidance (3-WAY RESTRICTED VERSION)

4 classes:
  0: benign (no people)
  1: safe_clothed (person with clothes)
  2: harm_nude (nudity)
  3: harm_color (color artifacts/distortions)

Guidance Logic (3-WAY RESTRICTED version):
  When both harm classes exceed threshold:
  1. Compute gradients for safe, harm_nude, harm_color
  2. Apply pairwise restricted gradient to ALL THREE:
     - Remove harm_nude direction from safe and harm_color
     - Remove harm_color direction from safe and harm_nude
     - Remove safe direction from harm_nude and harm_color
  3. Final: delta_safe - harmful_scale * (delta_nude + delta_color)

Key Features:
  - All 3 gradients are orthogonalized (no conflicts between any pair)
  - safe-harm_nude, safe-harm_color, harm_nude-harm_color all restricted

Based on: generate_nudity_4class_spatial_cg_always_restricted.py
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
from typing import List, Optional, Dict, Tuple

from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDIMScheduler

from geo_models.classifier.classifier import load_discriminator
from geo_utils.classifier_interpretability import ClassifierGradCAM


# =========================
# Nudity 4-Class Configuration
# =========================
NUDITY_4CLASS_CONFIG = {
    "benign": 0,
    "safe_clothed": 1,
    "harm_nude": 2,
    "harm_color": 3,
    "safe_classes": [0, 1],
    "harm_classes": [2, 3],
    "guidance_target_safe": 1,
    "class_names": {
        0: "benign",
        1: "safe_clothed",
        2: "harm_nude",
        3: "harm_color"
    }
}


# =========================
# Adaptive Spatial Threshold Scheduler
# =========================
class AdaptiveSpatialThresholdScheduler:
    def __init__(self, strategy: str = "linear_decrease", start_value: float = 0.7,
                 end_value: float = 0.3, total_steps: int = 50):
        self.strategy = strategy
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps

    def get_threshold(self, current_step: int) -> float:
        if self.strategy == "constant":
            return self.start_value
        t = current_step / max(self.total_steps - 1, 1)
        if self.strategy == "linear_decrease":
            return self.start_value - (self.start_value - self.end_value) * t
        elif self.strategy == "cosine_anneal":
            return self.end_value + (self.start_value - self.end_value) * 0.5 * (1 + np.cos(np.pi * t))
        return self.start_value


# =========================
# Multi-Harm Class Detector
# =========================
class MultiHarmClassDetector:
    def __init__(self, classifier_model, config: Dict = NUDITY_4CLASS_CONFIG, device: str = "cuda"):
        self.classifier = classifier_model
        self.config = config
        self.device = device
        self.classifier_dtype = next(self.classifier.parameters()).dtype
        self.safe_classes = config["safe_classes"]
        self.harm_classes = config["harm_classes"]
        self.target_safe = config["guidance_target_safe"]
        self.class_names = config["class_names"]

    def detect_harm(self, latent: torch.Tensor, timestep: torch.Tensor) -> Tuple[Dict[int, float], int, Dict]:
        with torch.no_grad():
            latent_input = latent.to(dtype=self.classifier_dtype)
            if not isinstance(timestep, torch.Tensor):
                timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
            elif timestep.dim() == 0:
                timestep = timestep.unsqueeze(0)
            B = latent_input.shape[0]
            if timestep.shape[0] != B:
                timestep = timestep.expand(B).to(latent.device)
            norm_timestep = timestep.float() / 1000.0
            logits = self.classifier(latent_input, norm_timestep)
            harm_logits_dict = {hc: logits[0, hc].item() for hc in self.harm_classes}
            max_class = logits.argmax(dim=1)[0].item()
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
            info = {
                "all_logits": logits[0].detach().cpu().numpy(),
                "all_probs": probs,
                "max_class": max_class,
                "harm_logits": harm_logits_dict,
                "safe_class": self.target_safe
            }
        return harm_logits_dict, self.target_safe, info


# =========================
# GradCAM stats loader
# =========================
def load_gradcam_stats_map(stats_dir: str) -> Dict[int, Dict[str, float]]:
    stats_dir = Path(stats_dir)
    mapping = {
        2: "gradcam_stats_harm_nude_class2.json",
        3: "gradcam_stats_harm_color_class3.json",
    }
    stats_map = {}
    for cls, fname in mapping.items():
        path = stats_dir / fname
        if path.exists():
            with open(path, "r") as f:
                d = json.load(f)
            stats_map[cls] = {"mean": float(d["mean"]), "std": float(d["std"])}
    return stats_map


# =========================
# Multi-Harm Spatial Mask Generator
# =========================
class MultiHarmSpatialMaskGenerator:
    def __init__(self, classifier_model, harm_detector: MultiHarmClassDetector,
                 gradcam_layer: str = "encoder_model.middle_block.2", device: str = "cuda",
                 debug: bool = False, gradcam_stats_map: Optional[Dict] = None):
        self.classifier = classifier_model
        self.harm_detector = harm_detector
        self.device = device
        self.debug = debug
        self.classifier_dtype = next(self.classifier.parameters()).dtype
        self.gradcam_stats_map = gradcam_stats_map
        self.gradcam = ClassifierGradCAM(classifier_model=classifier_model, target_layer_name=gradcam_layer)
        self.classifier = self.classifier.to(device)
        self.classifier.eval()
        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)
        self.stats = {'total_steps': 0, 'both_harm_steps': 0, 'single_harm_steps': 0, 'step_history': []}

    def _apply_cdf_normalization(self, heatmap: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        z = (heatmap - mean) / (std + 1e-8)
        from torch.distributions import Normal
        normal = Normal(torch.tensor(0.0, device=heatmap.device), torch.tensor(1.0, device=heatmap.device))
        return normal.cdf(z)

    def _generate_heatmap_for_class(self, latent: torch.Tensor, norm_timestep: torch.Tensor, target_class: int) -> torch.Tensor:
        use_abs = (self.gradcam_stats_map is not None) and (target_class in self.gradcam_stats_map)
        with torch.enable_grad():
            heatmap, _ = self.gradcam.generate_heatmap(latent=latent, timestep=norm_timestep,
                                                        target_class=target_class, normalize=not use_abs)
        if use_abs:
            stats = self.gradcam_stats_map[target_class]
            heatmap = self._apply_cdf_normalization(heatmap, stats["mean"], stats["std"])
        return heatmap

    def generate_masks(self, latent: torch.Tensor, timestep: torch.Tensor, spatial_threshold: float,
                       current_step: Optional[int] = None):
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        B = latent.shape[0]
        if timestep.shape[0] != B:
            timestep = timestep.expand(B)

        harm_logits_dict, safe_class, detection_info = self.harm_detector.detect_harm(latent, timestep)
        latent_input = latent.to(dtype=self.classifier_dtype)
        norm_timestep = timestep.float() / 1000.0

        heatmaps_dict = {}
        masks_dict = {}
        active_harm_classes = []

        for harm_class in self.harm_detector.harm_classes:
            heatmap = self._generate_heatmap_for_class(latent_input, norm_timestep, harm_class)
            heatmaps_dict[harm_class] = heatmap
            mask = (heatmap >= spatial_threshold).float()
            if mask.mean().item() > 0:
                masks_dict[harm_class] = mask
                active_harm_classes.append(harm_class)

        self.stats['total_steps'] += 1
        if len(active_harm_classes) == 2:
            self.stats['both_harm_steps'] += 1
        elif len(active_harm_classes) == 1:
            self.stats['single_harm_steps'] += 1

        self.stats['step_history'].append({
            'step': current_step,
            'active_harm_classes': active_harm_classes,
            'num_active': len(active_harm_classes)
        })

        if self.debug:
            print(f"  [Step {current_step}] Active: {active_harm_classes}")

        return masks_dict, heatmaps_dict, active_harm_classes, safe_class, detection_info

    def get_statistics(self): return self.stats.copy()
    def reset_statistics(self):
        self.stats = {'total_steps': 0, 'both_harm_steps': 0, 'single_harm_steps': 0, 'step_history': []}


# =========================
# 3-Way Restricted Gradient Guidance
# =========================
class ThreeWayRestrictedGradientGuidance:
    """
    Applies 3-way Restricted Gradient: safe, harm_nude, harm_color
    All three gradients are orthogonalized pairwise.
    """

    def __init__(self, classifier_model, config: Dict = NUDITY_4CLASS_CONFIG, device: str = "cuda"):
        self.classifier = classifier_model
        self.config = config
        self.device = device
        self.classifier_dtype = next(self.classifier.parameters()).dtype
        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

    def _compute_class_gradient(self, latent: torch.Tensor, norm_timestep: torch.Tensor, target_class: int) -> torch.Tensor:
        latent_input = latent.detach().to(dtype=self.classifier_dtype).requires_grad_(True)
        logits = self.classifier(latent_input, norm_timestep)
        class_logit = logits[:, target_class].sum()
        grad = torch.autograd.grad(class_logit, latent_input)[0]
        return grad

    def _project_out(self, g_a: torch.Tensor, g_b: torch.Tensor) -> torch.Tensor:
        """Project out g_b's direction from g_a."""
        g_a_flat = g_a.view(-1)
        g_b_flat = g_b.view(-1)
        g_b_norm_sq = torch.dot(g_b_flat, g_b_flat) + 1e-8
        proj_coef = torch.dot(g_a_flat, g_b_flat) / g_b_norm_sq
        return g_a - proj_coef * g_b

    def _orthogonalize_three(self, g_safe: torch.Tensor, g_nude: torch.Tensor, g_color: torch.Tensor):
        """
        Orthogonalize all 3 gradients pairwise using sequential Gram-Schmidt-like process.
        """
        # Step 1: Orthogonalize nude and color (harm-harm)
        delta_nude = self._project_out(g_nude, g_color)
        delta_color = self._project_out(g_color, g_nude)

        # Step 2: Orthogonalize safe w.r.t. both harms
        delta_safe = self._project_out(g_safe, g_nude)
        delta_safe = self._project_out(delta_safe, g_color)

        return delta_safe, delta_nude, delta_color

    def compute_gradient(self, latent: torch.Tensor, timestep: torch.Tensor,
                         active_harm_classes: List[int], masks_dict: Dict[int, torch.Tensor],
                         safe_class: int, guidance_scale: float = 5.0, harmful_scale: float = 1.0,
                         base_guidance_scale: float = 0.0) -> torch.Tensor:
        with torch.enable_grad():
            if not isinstance(timestep, torch.Tensor):
                timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
            elif timestep.dim() == 0:
                timestep = timestep.unsqueeze(0).to(latent.device)
            else:
                timestep = timestep.to(latent.device)
            B = latent.shape[0]
            if timestep.shape[0] != B:
                timestep = timestep.expand(B).to(latent.device)
            norm_timestep = timestep.float() / 1000.0

            # Compute safe gradient
            grad_safe = self._compute_class_gradient(latent, norm_timestep, safe_class)

            if len(active_harm_classes) == 0:
                return (base_guidance_scale * grad_safe).to(dtype=latent.dtype).detach()

            elif len(active_harm_classes) == 1:
                # Single harm: safe-harm restricted
                harm_class = active_harm_classes[0]
                grad_harm = self._compute_class_gradient(latent, norm_timestep, harm_class)

                delta_safe = self._project_out(grad_safe, grad_harm)
                delta_harm = self._project_out(grad_harm, grad_safe)
                grad = delta_safe - harmful_scale * delta_harm

                mask = masks_dict[harm_class].unsqueeze(1)
                weight_map = mask * guidance_scale + (1 - mask) * base_guidance_scale
                return (grad * weight_map).to(dtype=latent.dtype).detach()

            else:
                # BOTH harm classes active: 3-way restricted
                grad_nude = self._compute_class_gradient(latent, norm_timestep, 2)
                grad_color = self._compute_class_gradient(latent, norm_timestep, 3)

                # 3-way orthogonalization
                delta_safe, delta_nude, delta_color = self._orthogonalize_three(grad_safe, grad_nude, grad_color)

                # Combined: delta_safe - harmful_scale * (delta_nude + delta_color)
                grad = delta_safe - harmful_scale * (delta_nude + delta_color)

                # Union mask
                mask_nude = masks_dict.get(2, torch.zeros_like(latent[:, 0:1, :, :]))
                mask_color = masks_dict.get(3, torch.zeros_like(latent[:, 0:1, :, :]))
                if mask_nude.dim() == 3:
                    mask_nude = mask_nude.unsqueeze(1)
                if mask_color.dim() == 3:
                    mask_color = mask_color.unsqueeze(1)
                combined_mask = torch.max(mask_nude, mask_color)

                weight_map = combined_mask * guidance_scale + (1 - combined_mask) * base_guidance_scale
                return (grad * weight_map).to(dtype=latent.dtype).detach()

    def apply_guidance(self, latent: torch.Tensor, timestep: torch.Tensor,
                       active_harm_classes: List[int], masks_dict: Dict[int, torch.Tensor],
                       safe_class: int, guidance_scale: float = 5.0, harmful_scale: float = 1.0,
                       base_guidance_scale: float = 0.0) -> torch.Tensor:
        weighted_grad = self.compute_gradient(latent, timestep, active_harm_classes, masks_dict,
                                               safe_class, guidance_scale, harmful_scale, base_guidance_scale)
        return latent + weighted_grad


# =========================
# Arguments
# =========================
def parse_args():
    parser = ArgumentParser(description="Nudity 4-Class 3-Way Restricted CG")
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_img/nudity_4class_3way_restricted")
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--gradcam_layer", type=str, default="encoder_model.middle_block.2")
    parser.add_argument("--gradcam_stats_dir", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--spatial_threshold_start", type=float, default=0.7)
    parser.add_argument("--spatial_threshold_end", type=float, default=0.3)
    parser.add_argument("--threshold_strategy", type=str, default="linear_decrease")
    parser.add_argument("--harmful_scale", type=float, default=1.0)
    parser.add_argument("--base_guidance_scale", type=float, default=0.0)
    parser.add_argument("--guidance_start_step", type=int, default=0)
    parser.add_argument("--guidance_end_step", type=int, default=50)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


# =========================
# Utilities
# =========================
def load_prompts(prompt_file: str) -> List[str]:
    with open(prompt_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def save_image(image, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.resize((512, 512)).save(filepath)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Main Generation
# =========================
def generate_with_3way_restricted(pipe, prompts, mask_generator, guidance_module, threshold_scheduler, args, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("NUDITY 4-CLASS 3-WAY RESTRICTED GRADIENT GUIDANCE")
    print(f"{'='*60}")

    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        for sample_idx in range(args.nsamples):
            mask_generator.reset_statistics()

            def callback_on_step_end(pipe, step, timestep, callback_kwargs):
                latents = callback_kwargs["latents"]
                if args.guidance_start_step <= step <= args.guidance_end_step:
                    spatial_threshold = threshold_scheduler.get_threshold(step)
                    masks_dict, _, active_harm_classes, safe_class, _ = mask_generator.generate_masks(
                        latent=latents, timestep=timestep, spatial_threshold=spatial_threshold, current_step=step)
                    if len(active_harm_classes) > 0:
                        guided_latents = guidance_module.apply_guidance(
                            latent=latents, timestep=timestep, active_harm_classes=active_harm_classes,
                            masks_dict=masks_dict, safe_class=safe_class, guidance_scale=args.guidance_scale,
                            harmful_scale=args.harmful_scale, base_guidance_scale=args.base_guidance_scale)
                        callback_kwargs["latents"] = guided_latents
                return callback_kwargs

            with torch.no_grad():
                output = pipe(prompt=prompt, num_inference_steps=args.num_inference_steps,
                              guidance_scale=args.cfg_scale, callback_on_step_end=callback_on_step_end,
                              callback_on_step_end_tensor_inputs=["latents"])

            safe_prompt = "".join(c if c.isalnum() or c in ' -_' else '_' for c in prompt)[:50].replace(' ', '_')
            save_image(output.images[0], output_dir / f"{prompt_idx:04d}_{sample_idx:02d}_{safe_prompt}.png")

    print(f"\nGeneration complete. Output: {output_dir}")


# =========================
# Main
# =========================
def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator()
    device = accelerator.device

    print(f"\n{'='*60}")
    print("NUDITY 4-CLASS 3-WAY RESTRICTED CG - INIT")
    print(f"{'='*60}")

    gradcam_stats_map = load_gradcam_stats_map(args.gradcam_stats_dir) if args.gradcam_stats_dir else None
    prompts = load_prompts(args.prompt_file)

    pipe = StableDiffusionPipeline.from_pretrained(args.ckpt_path, torch_dtype=torch.float16, safety_checker=None).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    classifier = load_discriminator(ckpt_path=args.classifier_ckpt, condition=None, eval=True, channel=4, num_classes=4).to(device)
    classifier.eval()

    harm_detector = MultiHarmClassDetector(classifier_model=classifier, config=NUDITY_4CLASS_CONFIG, device=device)
    threshold_scheduler = AdaptiveSpatialThresholdScheduler(strategy=args.threshold_strategy,
        start_value=args.spatial_threshold_start, end_value=args.spatial_threshold_end, total_steps=args.num_inference_steps)
    mask_generator = MultiHarmSpatialMaskGenerator(classifier_model=classifier, harm_detector=harm_detector,
        gradcam_layer=args.gradcam_layer, device=device, debug=args.debug, gradcam_stats_map=gradcam_stats_map)
    guidance_module = ThreeWayRestrictedGradientGuidance(classifier_model=classifier, config=NUDITY_4CLASS_CONFIG, device=device)

    generate_with_3way_restricted(pipe, prompts, mask_generator, guidance_module, threshold_scheduler, args, Path(args.output_dir))
    print("\nAll done!")


if __name__ == "__main__":
    main()
