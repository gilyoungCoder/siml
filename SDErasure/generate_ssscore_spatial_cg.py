#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SSScore-guided Spatial Classifier Guidance (SSScore + SCG)

Combines SDErasure's "when" (SSScore critical timestep selection) with
Spatial CG's "where/how" (GradCAM + bidirectional classifier guidance).

Key idea:
  - SDErasure's SSScore identifies which denoising timesteps are most critical
    for nudity concept formation
  - Spatial CG applies classifier guidance with GradCAM-based spatial masks
  - Instead of applying CG at ALL steps, we apply it only (or more strongly)
    at SSScore-identified critical timesteps

Modes:
  1. "ssscore_only": Apply spatial CG ONLY at critical timesteps
  2. "ssscore_boost": Apply CG at all steps, but BOOST guidance scale at critical timesteps
  3. "ssscore_adaptive": Scale guidance proportionally to SSScore at each step

Usage:
  python generate_ssscore_spatial_cg.py \
    --ckpt_path "CompVis/stable-diffusion-v1-4" \
    --prompt_file /path/to/ringabell.txt \
    --classifier_ckpt /path/to/classifier.pth \
    --output_dir ./outputs/ssscore_scg \
    --ssscore_mode ssscore_boost \
    --cg_guidance_scale 5.0 \
    --ssscore_boost_factor 3.0
"""

import os
import sys
import json
import random
import math
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDPMScheduler, DDIMScheduler
from diffusers.utils import logging

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "SoftDelete+CG"))

from geo_models.classifier.classifier import load_discriminator
from geo_utils.classifier_interpretability import ClassifierGradCAM

logger = logging.get_logger(__name__)


# =============================================================================
# SSScore Computation (from SDErasure)
# =============================================================================

@torch.no_grad()
def compute_ssscore_for_inference(
    unet,
    scheduler,
    tokenizer,
    text_encoder,
    target_concept: str = "nudity",
    anchor_concept: str = "",
    n_eval_timesteps: int = 50,
    n_samples: int = 8,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SSScore at each denoising timestep.

    Returns:
        scores: (n_eval_timesteps,) SSScore per timestep
        timestep_indices: (n_eval_timesteps,) DDPM timestep values
    """
    # Encode concepts
    def encode_text(text):
        tokens = tokenizer(
            [text] if text else [""],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return text_encoder(tokens.input_ids.to(device))[0]

    emb_target = encode_text(target_concept)
    emb_anchor = encode_text(anchor_concept)

    T = scheduler.config.num_train_timesteps  # 1000
    timestep_indices = torch.linspace(0, T - 1, n_eval_timesteps, dtype=torch.long)

    emb_t = emb_target.expand(n_samples, -1, -1)
    emb_a = emb_anchor.expand(n_samples, -1, -1)

    scores = []
    unet_dtype = next(unet.parameters()).dtype

    for t_val in tqdm(timestep_indices, desc="Computing SSScore"):
        t = t_val.to(device).unsqueeze(0).expand(n_samples)
        x0 = torch.randn(n_samples, 4, 64, 64, device=device)
        eps = torch.randn_like(x0)
        xt = scheduler.add_noise(x0, eps, t)

        pred_target = unet(xt.to(unet_dtype), t, emb_t.to(unet_dtype)).sample.float()
        L_c = F.mse_loss(pred_target, eps).item()

        pred_anchor = unet(xt.to(unet_dtype), t, emb_a.to(unet_dtype)).sample.float()
        L_a = F.mse_loss(pred_anchor, eps).item()

        delta = L_c - L_a
        S_t = 1.0 / (1.0 + np.exp(np.clip(delta, -50, 50)))
        scores.append(S_t)

    return np.array(scores), timestep_indices.numpy()


def build_ssscore_schedule(
    scores: np.ndarray,
    timestep_indices: np.ndarray,
    scheduler_timesteps: torch.Tensor,
    mode: str = "ssscore_boost",
    lambda_threshold: float = 0.5,
    boost_factor: float = 3.0,
) -> Dict[int, float]:
    """
    Build per-step guidance scale multiplier based on SSScore.

    Args:
        scores: SSScore array from compute_ssscore
        timestep_indices: corresponding DDPM timestep values
        scheduler_timesteps: actual timesteps used in inference scheduler
        mode: "ssscore_only", "ssscore_boost", or "ssscore_adaptive"
        lambda_threshold: SSScore threshold for critical timesteps
        boost_factor: multiplier for critical timesteps in boost mode

    Returns:
        Dict mapping inference step index -> guidance scale multiplier
    """
    # Interpolate SSScore to all DDPM timesteps
    # scores are at timestep_indices; we need values at scheduler_timesteps
    from scipy.interpolate import interp1d

    interp_fn = interp1d(
        timestep_indices.astype(float),
        scores,
        kind="linear",
        fill_value="extrapolate",
    )

    schedule = {}
    for step_idx, t in enumerate(scheduler_timesteps):
        t_val = t.item()
        s = float(interp_fn(t_val))

        if mode == "ssscore_only":
            # Apply CG only at critical timesteps (SSScore > threshold)
            schedule[step_idx] = 1.0 if s > lambda_threshold else 0.0

        elif mode == "ssscore_boost":
            # Always apply CG, but boost at critical timesteps
            schedule[step_idx] = boost_factor if s > lambda_threshold else 1.0

        elif mode == "ssscore_adaptive":
            # Scale guidance proportionally to SSScore
            # Normalize: higher SSScore = model more sensitive = more guidance needed
            schedule[step_idx] = s * boost_factor

        else:
            schedule[step_idx] = 1.0

    return schedule


# =============================================================================
# 4-Class Nudity Config
# =============================================================================

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


# =============================================================================
# Harm Detection
# =============================================================================

class HarmClassDetector:
    def __init__(self, classifier_model, config=NUDITY_4CLASS_CONFIG, device="cuda"):
        self.classifier = classifier_model
        self.config = config
        self.device = device
        self.classifier_dtype = next(self.classifier.parameters()).dtype
        self.safe_classes = config["safe_classes"]
        self.harm_classes = config["harm_classes"]
        self.target_safe = config["guidance_target_safe"]
        self.class_names = config["class_names"]

    def detect_harm(self, latent, timestep):
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
            max_class = logits.argmax(dim=1)[0].item()
            is_harmful = max_class in self.harm_classes
            harm_class = max_class if is_harmful else None
            safe_class = self.target_safe
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
            info = {
                "max_class": max_class,
                "max_class_name": self.class_names[max_class],
                "is_harmful": is_harmful,
                "harm_class": harm_class,
                "safe_class": safe_class,
                "probs": probs,
            }
        return is_harmful, harm_class, safe_class, info


# =============================================================================
# GradCAM Stats
# =============================================================================

def load_gradcam_stats_map(stats_dir: str) -> Dict[int, Dict[str, float]]:
    stats_dir = Path(stats_dir)
    mapping = {
        2: "gradcam_stats_harm_nude_class2.json",
        3: "gradcam_stats_harm_color_class3.json",
    }
    stats_map = {}
    for cls, fname in mapping.items():
        path = stats_dir / fname
        if not path.exists():
            print(f"[Warning] GradCAM stats file not found: {path}")
            continue
        with open(path, "r") as f:
            d = json.load(f)
        stats_map[cls] = {"mean": float(d["mean"]), "std": float(d["std"])}
    return stats_map


# =============================================================================
# Spatial Mask Generator
# =============================================================================

class SpatialMaskGenerator:
    def __init__(
        self,
        classifier_model,
        harm_detector: HarmClassDetector,
        gradcam_layer: str = "encoder_model.middle_block.2",
        device: str = "cuda",
        gradcam_stats_map=None,
    ):
        self.classifier = classifier_model
        self.harm_detector = harm_detector
        self.device = device
        self.classifier_dtype = next(self.classifier.parameters()).dtype
        self.gradcam_stats_map = gradcam_stats_map

        self.gradcam = ClassifierGradCAM(
            classifier_model=classifier_model,
            target_layer_name=gradcam_layer
        )
        self.classifier = self.classifier.to(device)
        self.classifier.eval()
        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

        self.stats = {
            'total_steps': 0, 'harmful_steps': 0, 'guidance_applied': 0,
            'ssscore_skipped': 0, 'step_history': [],
        }

    def _apply_cdf_normalization(self, heatmap, mean, std):
        z = (heatmap - mean) / (std + 1e-8)
        from torch.distributions import Normal
        normal = Normal(
            torch.tensor(0.0, device=heatmap.device, dtype=heatmap.dtype),
            torch.tensor(1.0, device=heatmap.device, dtype=heatmap.dtype)
        )
        return normal.cdf(z)

    def generate_mask(self, latent, timestep, spatial_threshold, current_step=None):
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        B = latent.shape[0]
        if timestep.shape[0] != B:
            timestep = timestep.expand(B)

        is_harmful, harm_class, safe_class, detection_info = self.harm_detector.detect_harm(
            latent=latent, timestep=timestep
        )
        self.stats['total_steps'] += 1
        if is_harmful:
            self.stats['harmful_steps'] += 1

        if not is_harmful:
            return False, None, harm_class, safe_class, detection_info

        self.stats['guidance_applied'] += 1
        latent_input = latent.to(dtype=self.classifier_dtype)
        norm_timestep = timestep.float() / 1000.0

        use_abs = (self.gradcam_stats_map is not None) and (harm_class in self.gradcam_stats_map)
        with torch.enable_grad():
            heatmap, info = self.gradcam.generate_heatmap(
                latent=latent_input,
                timestep=norm_timestep,
                target_class=harm_class,
                normalize=not use_abs
            )
        if use_abs:
            stats = self.gradcam_stats_map[harm_class]
            heatmap = self._apply_cdf_normalization(heatmap, stats["mean"], stats["std"])

        mask = (heatmap >= spatial_threshold).float()
        return True, mask, harm_class, safe_class, detection_info

    def get_statistics(self):
        return self.stats.copy()

    def reset_statistics(self):
        self.stats = {
            'total_steps': 0, 'harmful_steps': 0, 'guidance_applied': 0,
            'ssscore_skipped': 0, 'step_history': [],
        }


# =============================================================================
# Spatial Guidance
# =============================================================================

class SpatialGuidance:
    def __init__(self, classifier_model, config=NUDITY_4CLASS_CONFIG, device="cuda"):
        self.classifier = classifier_model
        self.config = config
        self.device = device
        self.classifier_dtype = next(self.classifier.parameters()).dtype
        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

    def apply_guidance(
        self, latent, timestep, spatial_mask, harm_class, safe_class,
        guidance_scale=5.0, harmful_scale=1.0, base_guidance_scale=0.0,
    ):
        with torch.enable_grad():
            latent_input = latent.detach().to(dtype=self.classifier_dtype).requires_grad_(True)
            if not isinstance(timestep, torch.Tensor):
                timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
            elif timestep.dim() == 0:
                timestep = timestep.unsqueeze(0).to(latent.device)
            else:
                timestep = timestep.to(latent.device)
            B = latent_input.shape[0]
            if timestep.shape[0] != B:
                timestep = timestep.expand(B).to(latent.device)
            norm_timestep = timestep.float() / 1000.0

            # Bidirectional: safe - harmful
            latent_for_safe = latent_input.detach().requires_grad_(True)
            logits_safe = self.classifier(latent_for_safe, norm_timestep)
            safe_logit = logits_safe[:, safe_class].sum()
            grad_safe = torch.autograd.grad(safe_logit, latent_for_safe)[0]

            latent_for_harmful = latent_input.detach().requires_grad_(True)
            logits_harmful = self.classifier(latent_for_harmful, norm_timestep)
            harmful_logit = logits_harmful[:, harm_class].sum()
            grad_harmful = torch.autograd.grad(harmful_logit, latent_for_harmful)[0]

            grad = grad_safe - harmful_scale * grad_harmful

        mask_expanded = spatial_mask.unsqueeze(1)
        weight_map = mask_expanded * guidance_scale + (1 - mask_expanded) * base_guidance_scale
        weighted_grad = (grad * weight_map).to(dtype=latent.dtype)
        return latent + weighted_grad.detach()


# =============================================================================
# Threshold Scheduler
# =============================================================================

class ThresholdScheduler:
    def __init__(self, strategy="linear_decrease", start_value=0.7,
                 end_value=0.3, total_steps=50):
        self.strategy = strategy
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps

    def get_threshold(self, current_step):
        if self.strategy == "constant":
            return self.start_value
        t = current_step / max(self.total_steps - 1, 1)
        if self.strategy == "linear_decrease":
            return self.start_value - (self.start_value - self.end_value) * t
        elif self.strategy == "cosine_anneal":
            return self.end_value + (self.start_value - self.end_value) * 0.5 * (1 + np.cos(np.pi * t))
        return self.start_value


# =============================================================================
# SSScore-guided Spatial CG Pipeline
# =============================================================================

class SSSCoreGuidedSpatialCGPipeline:
    """
    SSScore + Spatial CG combined pipeline.

    Uses SDErasure's SSScore to determine WHEN to apply guidance,
    and Spatial CG's GradCAM to determine WHERE to apply guidance.
    """

    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        classifier_model,
        mask_generator: SpatialMaskGenerator,
        guidance_module: SpatialGuidance,
        threshold_scheduler: ThresholdScheduler,
        ssscore_schedule: Dict[int, float],
        device: str = "cuda",
        debug: bool = False,
    ):
        self.pipe = pipe
        self.classifier = classifier_model
        self.mask_generator = mask_generator
        self.guidance_module = guidance_module
        self.threshold_scheduler = threshold_scheduler
        self.ssscore_schedule = ssscore_schedule
        self.device = device
        self.debug = debug

        self.unet = pipe.unet
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler

    def __call__(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        generator=None,
        # Spatial CG
        cg_guidance_scale=5.0,
        harmful_scale=1.0,
        base_guidance_scale=0.0,
        # SSScore mode
        ssscore_mode="ssscore_boost",
    ):
        batch_size = 1
        do_cfg = guidance_scale > 1.0

        # Encode prompt
        text_inputs = self.pipe.tokenizer(
            [prompt] if isinstance(prompt, str) else prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.pipe.text_encoder(
            text_inputs.input_ids.to(self.device)
        )[0]

        if do_cfg:
            neg_prompt = negative_prompt or ""
            neg_inputs = self.pipe.tokenizer(
                [neg_prompt],
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.pipe.text_encoder(
                neg_inputs.input_ids.to(self.device)
            )[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Prepare latents
        latent_shape = (batch_size, self.unet.config.in_channels, height // 8, width // 8)
        latents = torch.randn(
            latent_shape, generator=generator, device=self.device,
            dtype=text_embeddings.dtype
        )

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        latents = latents * self.scheduler.init_noise_sigma

        self.mask_generator.reset_statistics()

        # Stats tracking
        steps_guided = 0
        steps_ssscore_skipped = 0
        steps_safe_skipped = 0

        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="SSScore+SCG")):
            # UNet forward under no_grad to save memory
            with torch.no_grad():
                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

                if do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # SSScore-guided Spatial CG
            ssscore_mult = self.ssscore_schedule.get(i, 1.0)

            # Skip if SSScore says this step is not critical
            if ssscore_mult <= 0:
                steps_ssscore_skipped += 1
                if self.debug:
                    print(f"  [Step {i}] SSScore skip (mult={ssscore_mult:.3f})")
                continue

            # Compute effective guidance scale
            effective_cg_scale = cg_guidance_scale * ssscore_mult

            # Get spatial threshold
            spatial_threshold = self.threshold_scheduler.get_threshold(i)

            # Generate mask
            should_guide, spatial_mask, harm_class, safe_class, detection_info = \
                self.mask_generator.generate_mask(
                    latent=latents,
                    timestep=t,
                    spatial_threshold=spatial_threshold,
                    current_step=i
                )

            if should_guide and spatial_mask is not None:
                latents = self.guidance_module.apply_guidance(
                    latent=latents,
                    timestep=t,
                    spatial_mask=spatial_mask,
                    harm_class=harm_class,
                    safe_class=safe_class,
                    guidance_scale=effective_cg_scale,
                    harmful_scale=harmful_scale,
                    base_guidance_scale=base_guidance_scale
                )
                steps_guided += 1
                if self.debug:
                    print(f"  [Step {i}] GUIDED (SSScore mult={ssscore_mult:.2f}, "
                          f"eff_scale={effective_cg_scale:.1f}, "
                          f"harm={detection_info['max_class_name']})")
            else:
                steps_safe_skipped += 1

            # Free gradient computation memory
            torch.cuda.empty_cache()

        # Decode
        with torch.no_grad():
            latents = 1 / 0.18215 * latents.detach()
            image = self.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = self.pipe.numpy_to_pil(image)

        stats = self.mask_generator.get_statistics()
        print(f"  [Stats] Guided: {steps_guided}, SSScore-skipped: {steps_ssscore_skipped}, "
              f"Safe-skipped: {steps_safe_skipped}, Harmful detected: {stats['harmful_steps']}")

        return image


# =============================================================================
# Arguments
# =============================================================================

def parse_args():
    parser = ArgumentParser(description="SSScore-guided Spatial Classifier Guidance")

    # Model
    parser.add_argument("--ckpt_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/ssscore_scg")

    # Generation
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # Classifier
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--gradcam_layer", type=str, default="encoder_model.middle_block.2")
    parser.add_argument("--gradcam_stats_dir", type=str, default=None)

    # Spatial CG
    parser.add_argument("--cg_guidance_scale", type=float, default=5.0)
    parser.add_argument("--spatial_threshold_start", type=float, default=0.7)
    parser.add_argument("--spatial_threshold_end", type=float, default=0.3)
    parser.add_argument("--threshold_strategy", type=str, default="linear_decrease",
                        choices=["constant", "linear_decrease", "cosine_anneal"])
    parser.add_argument("--harmful_scale", type=float, default=1.0)
    parser.add_argument("--base_guidance_scale", type=float, default=0.0)

    # SSScore settings
    parser.add_argument("--ssscore_mode", type=str, default="ssscore_boost",
                        choices=["ssscore_only", "ssscore_boost", "ssscore_adaptive"],
                        help="How to use SSScore: only=CG at critical steps, "
                             "boost=CG everywhere but stronger at critical, "
                             "adaptive=scale proportional to SSScore")
    parser.add_argument("--ssscore_lambda", type=float, default=0.5,
                        help="SSScore threshold for critical timesteps")
    parser.add_argument("--ssscore_boost_factor", type=float, default=3.0,
                        help="Guidance boost multiplier at critical timesteps")
    parser.add_argument("--ssscore_target", type=str, default="nudity",
                        help="Target concept for SSScore computation")
    parser.add_argument("--ssscore_anchor", type=str, default="",
                        help="Anchor concept for SSScore ('', 'a person wearing clothes', etc.)")
    parser.add_argument("--ssscore_n_eval", type=int, default=50,
                        help="Number of timesteps to evaluate for SSScore")
    parser.add_argument("--ssscore_n_samples", type=int, default=8,
                        help="Number of samples for SSScore averaging")
    parser.add_argument("--ssscore_cache", type=str, default=None,
                        help="Path to cached SSScore JSON (skip recomputation)")

    # Debug
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_prompts(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device

    print("\n" + "=" * 80)
    print("SSScore-guided Spatial Classifier Guidance")
    print("=" * 80)
    print(f"Model     : {args.ckpt_path}")
    print(f"SSScore   : mode={args.ssscore_mode}, λ={args.ssscore_lambda}, "
          f"boost={args.ssscore_boost_factor}")
    print(f"Spatial CG: scale={args.cg_guidance_scale}, threshold={args.spatial_threshold_start}"
          f"→{args.spatial_threshold_end}")
    print("=" * 80 + "\n")

    # Load prompts
    prompts = load_prompts(args.prompt_file)
    print(f"[1/5] Loaded {len(prompts)} prompts")

    # Load pipeline
    print("\n[2/5] Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    print("  Pipeline loaded")

    # NOTE: Classifier loaded AFTER SSScore to avoid GPU OOM
    # Compute or load SSScore first (uses UNet only)
    print("\n[4/5] Computing SSScore schedule...")

    ssscore_cache_path = args.ssscore_cache or os.path.join(args.output_dir, "ssscore_cache.json")
    os.makedirs(os.path.dirname(ssscore_cache_path) or ".", exist_ok=True)

    if args.ssscore_cache and os.path.exists(args.ssscore_cache):
        print(f"  Loading cached SSScore from {args.ssscore_cache}")
        with open(args.ssscore_cache, "r") as f:
            cache = json.load(f)
        scores = np.array(cache["scores"])
        timestep_indices = np.array(cache["timestep_indices"])
    elif os.path.exists(ssscore_cache_path):
        print(f"  Loading cached SSScore from {ssscore_cache_path}")
        with open(ssscore_cache_path, "r") as f:
            cache = json.load(f)
        scores = np.array(cache["scores"])
        timestep_indices = np.array(cache["timestep_indices"])
    else:
        # Compute SSScore using the pipeline's UNet (before classifier is loaded)
        ddpm_scheduler = DDPMScheduler.from_pretrained(
            args.ckpt_path, subfolder="scheduler"
        )
        scores, timestep_indices = compute_ssscore_for_inference(
            unet=pipe.unet,
            scheduler=ddpm_scheduler,
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            target_concept=args.ssscore_target,
            anchor_concept=args.ssscore_anchor,
            n_eval_timesteps=args.ssscore_n_eval,
            n_samples=args.ssscore_n_samples,
            device=device,
        )
        del ddpm_scheduler
        torch.cuda.empty_cache()

        # Cache SSScore
        with open(ssscore_cache_path, "w") as f:
            json.dump({
                "target_concept": args.ssscore_target,
                "anchor_concept": args.ssscore_anchor,
                "scores": scores.tolist(),
                "timestep_indices": timestep_indices.tolist(),
            }, f, indent=2)
        print(f"  SSScore cached to {ssscore_cache_path}")

    print(f"  SSScore — min: {scores.min():.4f}  max: {scores.max():.4f}  "
          f"mean: {scores.mean():.4f}")
    n_critical = (scores > args.ssscore_lambda).sum()
    print(f"  Critical timesteps (>{args.ssscore_lambda}): {n_critical}/{len(scores)}")

    # Build per-step schedule
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    ssscore_schedule = build_ssscore_schedule(
        scores=scores,
        timestep_indices=timestep_indices,
        scheduler_timesteps=pipe.scheduler.timesteps,
        mode=args.ssscore_mode,
        lambda_threshold=args.ssscore_lambda,
        boost_factor=args.ssscore_boost_factor,
    )

    # Print schedule summary
    active_steps = sum(1 for v in ssscore_schedule.values() if v > 0)
    boosted_steps = sum(1 for v in ssscore_schedule.values() if v > 1.0)
    print(f"  Schedule: {active_steps} active steps, {boosted_steps} boosted steps")

    # Load classifier (after SSScore to save GPU memory)
    print("\n[5/6] Loading 4-class classifier...")
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt,
        condition=None,
        eval=True,
        channel=4,
        num_classes=4,
    ).to(device)
    classifier.eval()
    print("  Classifier loaded")

    gradcam_stats_map = None
    if args.gradcam_stats_dir:
        gradcam_stats_map = load_gradcam_stats_map(args.gradcam_stats_dir)
        print(f"  GradCAM stats loaded from {args.gradcam_stats_dir}")

    # Initialize components
    print("\n[6/6] Initializing components...")

    harm_detector = HarmClassDetector(
        classifier_model=classifier, config=NUDITY_4CLASS_CONFIG, device=device
    )
    threshold_scheduler = ThresholdScheduler(
        strategy=args.threshold_strategy,
        start_value=args.spatial_threshold_start,
        end_value=args.spatial_threshold_end,
        total_steps=args.num_inference_steps,
    )
    mask_generator = SpatialMaskGenerator(
        classifier_model=classifier,
        harm_detector=harm_detector,
        gradcam_layer=args.gradcam_layer,
        device=device,
        gradcam_stats_map=gradcam_stats_map,
    )
    guidance_module = SpatialGuidance(
        classifier_model=classifier, config=NUDITY_4CLASS_CONFIG, device=device
    )

    pipeline = SSSCoreGuidedSpatialCGPipeline(
        pipe=pipe,
        classifier_model=classifier,
        mask_generator=mask_generator,
        guidance_module=guidance_module,
        threshold_scheduler=threshold_scheduler,
        ssscore_schedule=ssscore_schedule,
        device=device,
        debug=args.debug,
    )

    # Generate images
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + "=" * 80)
    print("GENERATION START")
    print("=" * 80)

    generator = torch.Generator(device=device)
    total_images = 0

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n[Prompt {prompt_idx + 1}/{len(prompts)}] {prompt}")

        for sample_idx in range(args.nsamples):
            seed = args.seed + prompt_idx * args.nsamples + sample_idx
            generator.manual_seed(seed)

            images = pipeline(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.cfg_scale,
                generator=generator,
                cg_guidance_scale=args.cg_guidance_scale,
                harmful_scale=args.harmful_scale,
                base_guidance_scale=args.base_guidance_scale,
                ssscore_mode=args.ssscore_mode,
            )

            safe_prompt = "".join(
                c if c.isalnum() or c in [' ', '-', '_'] else '_' for c in prompt
            )[:50].strip().replace(' ', '_')
            filename = f"{prompt_idx:04d}_{sample_idx:02d}_{safe_prompt}.png"
            save_path = output_dir / filename
            images[0].save(save_path)
            total_images += 1

    print("\n" + "=" * 80)
    print(f"COMPLETE: {total_images} images → {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
