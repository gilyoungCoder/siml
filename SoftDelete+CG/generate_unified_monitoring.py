#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified Monitoring + Spatial CG for SoftDelete+CG.

Monitoring modes:
  - gradcam: GradCAM-based P(harm) = CDF(z-score) > threshold  (existing)
  - z0: z0-trigger P(harm) = softmax(classifier(z0, t=0)) > threshold  (new)
  - z0_sticky: z0-trigger + sticky (once triggered, guide all remaining)  (new)
  - ssscore: SDErasure SSScore-based "when to erase" detection  (new)
  - ssscore_sticky: SSScore trigger + sticky (once triggered, guide all remaining)  (new)

Spatial guidance modes:
  - gradcam: GradCAM heatmap → CDF → binary mask  (existing)
  - attention: cross-attention token map → spatial mask  (new)
  - attention_gradcam: GradCAM × attention mask product  (new)
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
    """Load topk (for spatial) and sample-level (for monitoring) statistics."""
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
# Monitoring: GradCAM-based (existing)
# =============================================================================
class SampleLevelMonitor:
    """GradCAM-based monitoring: P(harm) = CDF((heatmap.mean() - mu) / sigma)."""

    def __init__(self, classifier_model, stats_map: Dict, gradcam_layer: str, device: str = "cuda"):
        self.classifier = classifier_model.to(device)
        self.classifier.eval()
        self.device = device
        self.dtype = next(self.classifier.parameters()).dtype
        self.stats_map = stats_map
        self.gradcam = ClassifierGradCAM(classifier_model, gradcam_layer)
        self.normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
        self.stats = {"total_steps": 0, "guided_steps": 0, "skipped_steps": 0, "step_history": []}

    def compute_p_harm(self, latent: torch.Tensor, timestep: torch.Tensor, harm_class: int) -> float:
        if harm_class not in self.stats_map:
            return 0.0
        lat = latent.to(dtype=self.dtype)
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        norm_t = timestep.float() / 1000.0

        with torch.enable_grad():
            heatmap, info = self.gradcam.generate_heatmap(lat, norm_t, harm_class, normalize=False)

        heatmap_mean = heatmap.mean().item()
        mu = self.stats_map[harm_class]["sample_mean"]
        sigma = self.stats_map[harm_class]["sample_std"]
        z = (heatmap_mean - mu) / (sigma + 1e-8)
        if math.isnan(z) or math.isinf(z):
            return 0.0
        return self.normal.cdf(torch.tensor(z)).item()

    def should_apply_guidance(self, latent, timestep, monitoring_threshold, step):
        self.stats["total_steps"] += 1
        active_classes = []
        info = {"step": step, "p_harm": {}, "monitoring_mode": "gradcam"}

        for harm_class in [2, 3]:
            if harm_class not in self.stats_map:
                continue
            p_harm = self.compute_p_harm(latent, timestep, harm_class)
            info["p_harm"][harm_class] = p_harm
            if p_harm > monitoring_threshold:
                active_classes.append(harm_class)

        info["active_classes"] = active_classes
        guided = len(active_classes) > 0
        if guided:
            self.stats["guided_steps"] += 1
        else:
            self.stats["skipped_steps"] += 1
        self.stats["step_history"].append({**info, "guided": guided})
        return guided, active_classes, info

    def reset_stats(self):
        self.stats = {"total_steps": 0, "guided_steps": 0, "skipped_steps": 0, "step_history": []}


# =============================================================================
# Monitoring: z0-trigger (new)
# =============================================================================
class Z0TriggerMonitor:
    """z0-trigger monitoring: predict z0 from zt, classify with t=0."""

    def __init__(self, classifier_model, alphas_cumprod: torch.Tensor, device: str = "cuda"):
        self.classifier = classifier_model
        self.classifier.eval()
        self.device = device
        self.dtype = next(self.classifier.parameters()).dtype
        self.alphas_cumprod = alphas_cumprod
        self.stats = {"total_steps": 0, "guided_steps": 0, "skipped_steps": 0, "step_history": []}

    @staticmethod
    def predict_z0(zt, noise_pred, alpha_bar):
        return (zt - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)

    def should_apply_guidance(self, prev_latents, noise_pred, timestep, monitoring_threshold, step):
        self.stats["total_steps"] += 1

        alpha_bar = self.alphas_cumprod[timestep]
        z0_hat = self.predict_z0(prev_latents, noise_pred, alpha_bar)

        with torch.no_grad():
            z0_input = z0_hat.to(dtype=self.dtype)
            logits = self.classifier(z0_input, torch.zeros(1, device=self.device))
            probs = F.softmax(logits, dim=-1)

        p_nude = probs[0, 2].item()
        p_color = probs[0, 3].item()
        p_harm = p_nude + p_color

        active_classes = [2, 3] if p_harm > monitoring_threshold else []
        guided = len(active_classes) > 0

        info = {
            "step": step,
            "p_harm": {2: p_nude, 3: p_color},
            "z0_p_harm": p_harm,
            "monitoring_mode": "z0",
        }

        if guided:
            self.stats["guided_steps"] += 1
        else:
            self.stats["skipped_steps"] += 1
        self.stats["step_history"].append({**info, "guided": guided})
        return guided, active_classes, info

    def reset_stats(self):
        self.stats = {"total_steps": 0, "guided_steps": 0, "skipped_steps": 0, "step_history": []}


# =============================================================================
# Monitoring: SSScore-based (SDErasure "when to erase")
# =============================================================================
class SSSCoreMonitor:
    """
    SSScore-based monitoring: pre-computes SSScore per timestep, then uses it
    as a per-step schedule during denoising.

    SSScore = exp(-L_c) / (exp(-L_c) + exp(-L_a)) = 1 / (1 + exp(L_c - L_a))
    where L_c = MSE(eps_theta(xt, ct, t), eps), L_a = MSE(eps_theta(xt, ca, t), eps)

    Modes:
      - ssscore: guide only when SSScore > threshold at that timestep
      - ssscore_sticky: once SSScore > threshold at any step, guide all remaining
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.schedule = {}  # step_idx -> ssscore value
        self.stats = {"total_steps": 0, "guided_steps": 0, "skipped_steps": 0, "step_history": []}

    @staticmethod
    def compute_ssscore(
        unet,
        scheduler,
        tokenizer,
        text_encoder,
        target_concept: str = "nudity",
        anchor_concept: str = "",
        n_eval_timesteps: int = 50,
        n_samples: int = 8,
        device: str = "cuda",
    ):
        """Compute SSScore at evenly-spaced DDPM timesteps."""
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

            with torch.no_grad():
                pred_target = unet(xt.to(unet_dtype), t, emb_t.to(unet_dtype)).sample.float()
                L_c = F.mse_loss(pred_target, eps).item()

                pred_anchor = unet(xt.to(unet_dtype), t, emb_a.to(unet_dtype)).sample.float()
                L_a = F.mse_loss(pred_anchor, eps).item()

            delta = L_c - L_a
            S_t = 1.0 / (1.0 + np.exp(np.clip(delta, -50, 50)))
            scores.append(S_t)

        return np.array(scores), timestep_indices.numpy()

    def build_schedule(self, scores, timestep_indices, scheduler_timesteps):
        """Interpolate SSScore to inference scheduler timesteps."""
        from scipy.interpolate import interp1d
        interp_fn = interp1d(
            timestep_indices.astype(float), scores,
            kind="linear", fill_value="extrapolate",
        )
        self.schedule = {}
        for step_idx, t in enumerate(scheduler_timesteps):
            self.schedule[step_idx] = float(interp_fn(t.item()))

    def should_apply_guidance(self, step_idx, monitoring_threshold):
        """Check if SSScore at this step exceeds threshold."""
        self.stats["total_steps"] += 1
        ssscore = self.schedule.get(step_idx, 0.0)

        guided = ssscore > monitoring_threshold
        info = {
            "step": step_idx,
            "ssscore": ssscore,
            "monitoring_mode": "ssscore",
            "threshold": monitoring_threshold,
        }

        # Active classes: always [2, 3] when triggered (nudity)
        active_classes = [2, 3] if guided else []
        info["active_classes"] = active_classes

        if guided:
            self.stats["guided_steps"] += 1
        else:
            self.stats["skipped_steps"] += 1
        self.stats["step_history"].append({**info, "guided": guided})

        return guided, active_classes, info

    def reset_stats(self):
        self.stats = {"total_steps": 0, "guided_steps": 0, "skipped_steps": 0, "step_history": []}

    @staticmethod
    def load_cache(cache_path):
        """Load cached SSScore results."""
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                data = json.load(f)
            return np.array(data["scores"]), np.array(data["timestep_indices"])
        return None, None

    @staticmethod
    def save_cache(cache_path, scores, timestep_indices):
        """Save SSScore results to cache."""
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({
                "scores": scores.tolist(),
                "timestep_indices": timestep_indices.tolist(),
            }, f, indent=2)


# =============================================================================
# Monitoring: Noise Divergence (online, image-specific)
# =============================================================================
class NoiseDivMonitor:
    """
    Online noise divergence monitoring: at each step, measure how much
    the prompt's noise prediction diverges from a safe prompt's prediction.

    divergence = ||eps_theta(zt, prompt, t) - eps_theta(zt, safe_prompt, t)||_2

    If divergence is high, the image is heading away from safe content → guide.
    Requires one extra UNet forward pass per step with safe_prompt.

    'noise_div_free' variant: uses ||noise_text - noise_uncond||_2 from CFG
    as a free proxy (no extra UNet pass needed).
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.stats = {"total_steps": 0, "guided_steps": 0, "skipped_steps": 0, "step_history": []}
        # Running stats for adaptive thresholding
        self.div_history = []

    def compute_divergence_full(self, noise_pred_text, noise_pred_safe):
        """Full divergence: ||eps(prompt) - eps(safe_prompt)||"""
        return (noise_pred_text - noise_pred_safe).float().norm().item()

    def compute_divergence_free(self, noise_pred_text, noise_pred_uncond):
        """Free proxy: ||eps(prompt) - eps(uncond)|| — already available from CFG."""
        return (noise_pred_text - noise_pred_uncond).float().norm().item()

    def should_apply_guidance(self, divergence, monitoring_threshold, step_idx):
        self.stats["total_steps"] += 1
        self.div_history.append(divergence)

        guided = divergence > monitoring_threshold
        active_classes = [2, 3] if guided else []

        info = {
            "step": step_idx,
            "divergence": divergence,
            "monitoring_mode": "noise_div",
            "threshold": monitoring_threshold,
            "active_classes": active_classes,
        }

        if guided:
            self.stats["guided_steps"] += 1
        else:
            self.stats["skipped_steps"] += 1
        self.stats["step_history"].append({**info, "guided": guided})
        return guided, active_classes, info

    def reset_stats(self):
        self.stats = {"total_steps": 0, "guided_steps": 0, "skipped_steps": 0, "step_history": []}
        self.div_history = []


# =============================================================================
# Monitoring: Gradient Norm (classifier sensitivity)
# =============================================================================
class GradNormMonitor:
    """
    Classifier gradient norm monitoring: ||∇_z log p(harm|z)||_2

    If the classifier gradient is large, the latent is in a region sensitive
    to harmful content → should guide. This reuses the classifier we already
    have, and the gradient is computed anyway during guidance.
    """

    def __init__(self, classifier_model, device: str = "cuda"):
        self.classifier = classifier_model
        self.classifier.eval()
        self.device = device
        self.dtype = next(self.classifier.parameters()).dtype
        self.stats = {"total_steps": 0, "guided_steps": 0, "skipped_steps": 0, "step_history": []}

    def compute_grad_norm(self, latent, timestep):
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        norm_t = timestep.float() / 1000.0

        with torch.enable_grad():
            lat = latent.detach().to(dtype=self.dtype).requires_grad_(True)
            logits = self.classifier(lat, norm_t)
            # Sum of harm class logits (nude + color)
            harm_score = logits[:, 2].sum() + logits[:, 3].sum()
            grad = torch.autograd.grad(harm_score, lat)[0]

        return grad.float().norm().item()

    def should_apply_guidance(self, latent, timestep, monitoring_threshold, step_idx):
        self.stats["total_steps"] += 1
        grad_norm = self.compute_grad_norm(latent, timestep)

        guided = grad_norm > monitoring_threshold
        active_classes = [2, 3] if guided else []

        info = {
            "step": step_idx,
            "grad_norm": grad_norm,
            "monitoring_mode": "grad_norm",
            "threshold": monitoring_threshold,
            "active_classes": active_classes,
        }

        if guided:
            self.stats["guided_steps"] += 1
        else:
            self.stats["skipped_steps"] += 1
        self.stats["step_history"].append({**info, "guided": guided})
        return guided, active_classes, info

    def reset_stats(self):
        self.stats = {"total_steps": 0, "guided_steps": 0, "skipped_steps": 0, "step_history": []}


# =============================================================================
# Spatial Guidance (unified: gradcam / attention / attention_gradcam)
# =============================================================================
class SpatialGuidance:
    """Spatial CG using CDF-normalized GradCAM, attention maps, or both."""

    def __init__(self, classifier_model, stats_map: Dict, gradcam_layer: str, device: str = "cuda"):
        self.classifier = classifier_model.to(device)
        self.classifier.eval()
        self.device = device
        self.dtype = next(self.classifier.parameters()).dtype
        self.stats_map = stats_map
        self.gradcam = ClassifierGradCAM(classifier_model, gradcam_layer)
        self.normal = Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))

        # Attention fields (set externally when spatial_mode is attention/attention_gradcam)
        self.attention_store = None
        self.token_indices = []

    def _pixel_cdf_normalize(self, heatmap: torch.Tensor, harm_class: int) -> torch.Tensor:
        mu = self.stats_map[harm_class]["topk_mean"]
        sigma = self.stats_map[harm_class]["topk_std"]
        z = (heatmap - mu) / (sigma + 1e-8)
        return self.normal.cdf(z)

    def _compute_gradcam_mask(self, latent, norm_t, active_harm_classes, spatial_threshold):
        """Compute GradCAM-based spatial mask (existing logic)."""
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

    def _compute_attention_mask(self, spatial_threshold):
        """Compute attention-based spatial mask."""
        from geo_utils.attention_utils import compute_attention_mask
        if self.attention_store is None or not self.token_indices:
            return None
        mask = compute_attention_mask(
            self.attention_store, self.token_indices,
            target_resolution=64, threshold=spatial_threshold, soft=False,
        )
        return mask

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

        # Compute classifier gradient (same for all spatial modes)
        with torch.enable_grad():
            l1 = latent.detach().to(dtype=self.dtype).requires_grad_(True)
            g_safe = torch.autograd.grad(self.classifier(l1, norm_t)[:, 1].sum(), l1)[0]

            g_harm = torch.zeros_like(g_safe)
            for hc in active_harm_classes:
                l2 = latent.detach().to(dtype=self.dtype).requires_grad_(True)
                g_harm += torch.autograd.grad(self.classifier(l2, norm_t)[:, hc].sum(), l2)[0]

            grad = g_safe - g_harm

        # Compute spatial mask based on mode
        if spatial_mode == "gradcam":
            combined_mask = self._compute_gradcam_mask(latent, norm_t, active_harm_classes, spatial_threshold)
        elif spatial_mode == "attention":
            attn_mask = self._compute_attention_mask(spatial_threshold)
            if attn_mask is not None:
                combined_mask = attn_mask.to(grad.device)
                if combined_mask.dim() == 3:
                    combined_mask = combined_mask.unsqueeze(1)
            else:
                # Fallback to all-ones if no attention info
                combined_mask = torch.ones_like(latent[:, 0:1, :, :])
        elif spatial_mode == "attention_gradcam":
            gc_mask = self._compute_gradcam_mask(latent, norm_t, active_harm_classes, spatial_threshold)
            attn_mask = self._compute_attention_mask(0.0)  # soft attention
            if attn_mask is not None:
                attn_mask = attn_mask.to(grad.device)
                if attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(1)
                combined_mask = gc_mask * attn_mask
                max_val = combined_mask.amax(dim=(2, 3), keepdim=True) + 1e-8
                combined_mask = combined_mask / max_val
                combined_mask = (combined_mask > spatial_threshold).float()
            else:
                combined_mask = gc_mask
        else:
            combined_mask = torch.ones_like(latent[:, 0:1, :, :])

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
    parser.add_argument("--output_dir", type=str, default="./scg_outputs/unified_monitoring")
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)

    # Classifier
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--gradcam_layer", type=str, default="encoder_model.middle_block.2")
    parser.add_argument("--gradcam_stats_dir", type=str, required=True)

    # Monitoring mode
    parser.add_argument("--monitoring_mode", type=str, default="gradcam",
                        choices=["gradcam", "z0", "z0_sticky", "dual", "z0_trigger_cdf", "dual_path",
                                 "ssscore", "ssscore_sticky",
                                 "noise_div", "noise_div_sticky",
                                 "noise_div_free", "noise_div_free_sticky",
                                 "ssscore_weighted",
                                 "grad_norm", "grad_norm_sticky",
                                 "online_ssscore", "online_ssscore_sticky"],
                        help="Monitoring trigger mode. "
                             "'noise_div': online noise divergence (extra UNet fwd with safe prompt). "
                             "'noise_div_free': CFG magnitude as proxy (no extra UNet pass). "
                             "'ssscore_weighted': SSScore as continuous guidance multiplier. "
                             "'grad_norm': classifier gradient norm as monitoring signal.")
    parser.add_argument("--monitoring_threshold", type=float, default=0.05,
                        help="P(harm) threshold for monitoring trigger "
                             "(z0 softmax threshold in dual mode)")
    parser.add_argument("--monitoring_start_step", type=int, default=0,
                        help="Only check monitoring from this step onward")
    parser.add_argument("--cdf_threshold", type=float, default=0.05,
                        help="GradCAM CDF per-step threshold (used in dual/z0_trigger_cdf mode)")
    parser.add_argument("--z0_decision_step", type=int, default=7,
                        help="Step at which to make z0 decision (z0_trigger_cdf/dual_path mode)")

    # SSScore parameters
    parser.add_argument("--ssscore_target", type=str, default="nudity",
                        help="Target concept for SSScore computation")
    parser.add_argument("--ssscore_anchor", type=str, default="",
                        help="Anchor concept for SSScore ('' = unconditional)")
    parser.add_argument("--ssscore_n_samples", type=int, default=8,
                        help="Number of random latent samples for SSScore averaging")
    parser.add_argument("--ssscore_cache", type=str, default="",
                        help="Path to SSScore cache JSON (avoids recomputation)")

    # Noise divergence parameters
    parser.add_argument("--noise_div_safe_prompt", type=str,
                        default="a person wearing clothes",
                        help="Safe prompt for noise divergence computation")

    # SSScore weighted parameters
    parser.add_argument("--ssscore_weight_min", type=float, default=0.0,
                        help="Minimum SSScore to start applying weighted guidance")
    parser.add_argument("--ssscore_weight_power", type=float, default=1.0,
                        help="Power to raise SSScore weight (1=linear, 2=quadratic)")

    # Spatial guidance mode
    parser.add_argument("--spatial_mode", type=str, default="gradcam",
                        choices=["gradcam", "attention", "attention_gradcam"],
                        help="Spatial masking mode for guidance")

    # Attention config
    parser.add_argument("--harmful_keywords", nargs="+",
                        default=["nude", "naked", "breast", "topless", "bare", "undress", "nsfw"],
                        help="Keywords for attention token matching")
    parser.add_argument("--attn_resolutions", nargs="+", type=int, default=[16, 32],
                        help="UNet resolutions to hook for attention")

    # Guidance (spatial)
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--base_guidance_scale", type=float, default=2.0)
    parser.add_argument("--spatial_threshold_start", type=float, default=0.5)
    parser.add_argument("--spatial_threshold_end", type=float, default=0.1)
    parser.add_argument("--spatial_threshold_strategy", type=str, default="cosine")

    # Step range
    parser.add_argument("--guidance_start_step", type=int, default=0)
    parser.add_argument("--guidance_end_step", type=int, default=50)

    parser.add_argument("--debug", action="store_true")

    # Prompt slicing for multi-GPU
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
    print(f"UNIFIED MONITORING + SPATIAL CG")
    print(f"  Monitoring mode:      {args.monitoring_mode}")
    print(f"  Monitoring threshold: {args.monitoring_threshold}")
    if args.monitoring_mode == "dual":
        print(f"  CDF threshold:        {args.cdf_threshold}")
    if args.monitoring_mode == "z0_trigger_cdf":
        print(f"  z0 decision step:     {args.z0_decision_step}")
        print(f"  CDF threshold:        {args.cdf_threshold}")
    if args.monitoring_mode == "dual_path":
        print(f"  z0 decision step:     {args.z0_decision_step}")
        print(f"  CDF threshold:        {args.cdf_threshold}")
        print(f"  Mode: dual-path (guided + unguided until decision step)")
    if args.monitoring_mode in ("ssscore", "ssscore_sticky", "ssscore_weighted"):
        print(f"  SSScore target:       {args.ssscore_target}")
        print(f"  SSScore anchor:       '{args.ssscore_anchor}'")
        print(f"  SSScore n_samples:    {args.ssscore_n_samples}")
        print(f"  SSScore cache:        {args.ssscore_cache or '(auto)'}")
    if args.monitoring_mode in ("noise_div", "noise_div_sticky"):
        print(f"  Safe prompt:          '{args.noise_div_safe_prompt}'")
    if args.monitoring_mode in ("noise_div_free", "noise_div_free_sticky"):
        print(f"  Mode:                 CFG magnitude proxy (no extra UNet)")
    if args.monitoring_mode in ("grad_norm", "grad_norm_sticky"):
        print(f"  Mode:                 Classifier gradient norm")
    print(f"  Spatial mode:         {args.spatial_mode}")
    print(f"  Guidance scale:       {args.guidance_scale} (base: {args.base_guidance_scale})")
    print(f"  Spatial threshold:    {args.spatial_threshold_start} -> {args.spatial_threshold_end}")
    if args.spatial_mode in ("attention", "attention_gradcam"):
        print(f"  Harmful keywords:     {args.harmful_keywords}")
        print(f"  Attn resolutions:     {args.attn_resolutions}")
    print(f"{'='*60}\n")

    # Load stats
    stats_map = load_gradcam_stats(args.gradcam_stats_dir)
    if not stats_map:
        raise RuntimeError(f"No stats found in {args.gradcam_stats_dir}")

    print("Loaded GradCAM stats:")
    for cls, s in stats_map.items():
        print(f"  Class {cls}: sample_mean={s['sample_mean']:.4f}, topk_mean={s['topk_mean']:.4f}")

    # Load prompts
    all_prompts = load_prompts(args.prompt_file)
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx > 0 else len(all_prompts)
    prompts_with_idx = list(enumerate(all_prompts))[start_idx:end_idx]
    print(f"Loaded {len(all_prompts)} prompts, processing [{start_idx}:{end_idx}] = {len(prompts_with_idx)}")

    # Load pipeline for its components (unet, vae, tokenizer, text_encoder, scheduler)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    unet = pipe.unet
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    scheduler = pipe.scheduler

    # Load classifier
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt, condition=None, eval=True, channel=4, num_classes=4
    ).to(device)
    classifier.eval()

    # Initialize monitoring
    monitor = None
    z0_monitor = None
    ssscore_monitor = None

    if args.monitoring_mode == "gradcam":
        monitor = SampleLevelMonitor(classifier, stats_map, args.gradcam_layer, device)
    elif args.monitoring_mode in ("z0", "z0_sticky"):
        alphas_cumprod = scheduler.alphas_cumprod.to(device)
        z0_monitor = Z0TriggerMonitor(classifier, alphas_cumprod, device)
    elif args.monitoring_mode == "dual":
        # Dual: z0_sticky (phase 1) + GradCAM CDF (phase 2), AND logic
        alphas_cumprod = scheduler.alphas_cumprod.to(device)
        z0_monitor = Z0TriggerMonitor(classifier, alphas_cumprod, device)
        monitor = SampleLevelMonitor(classifier, stats_map, args.gradcam_layer, device)
    elif args.monitoring_mode == "z0_trigger_cdf":
        # z0_trigger_cdf: single-step z0 decision + GradCAM CDF per-step
        alphas_cumprod = scheduler.alphas_cumprod.to(device)
        z0_monitor = Z0TriggerMonitor(classifier, alphas_cumprod, device)
        monitor = SampleLevelMonitor(classifier, stats_map, args.gradcam_layer, device)
    elif args.monitoring_mode == "dual_path":
        # dual_path: two parallel latent paths, z0 decides which to keep
        alphas_cumprod = scheduler.alphas_cumprod.to(device)
        z0_monitor = Z0TriggerMonitor(classifier, alphas_cumprod, device)
        monitor = SampleLevelMonitor(classifier, stats_map, args.gradcam_layer, device)
    elif args.monitoring_mode in ("ssscore", "ssscore_sticky"):
        ssscore_monitor = SSSCoreMonitor(device=str(device))
        # Load or compute SSScore
        cache_path = args.ssscore_cache or os.path.join(args.output_dir, "ssscore_cache.json")
        scores, timestep_indices = SSSCoreMonitor.load_cache(cache_path)
        if scores is not None:
            print(f"[SSScore] Loaded cached scores from {cache_path}")
        else:
            print(f"[SSScore] Computing SSScore (target='{args.ssscore_target}', "
                  f"anchor='{args.ssscore_anchor}', n_samples={args.ssscore_n_samples})...")
            scores, timestep_indices = SSSCoreMonitor.compute_ssscore(
                unet, scheduler, tokenizer, text_encoder,
                target_concept=args.ssscore_target,
                anchor_concept=args.ssscore_anchor,
                n_eval_timesteps=args.num_inference_steps,
                n_samples=args.ssscore_n_samples,
                device=str(device),
            )
            SSSCoreMonitor.save_cache(cache_path, scores, timestep_indices)
            print(f"[SSScore] Saved cache to {cache_path}")
        # Build schedule
        scheduler.set_timesteps(args.num_inference_steps, device=device)
        ssscore_monitor.build_schedule(scores, timestep_indices, scheduler.timesteps)
        print(f"[SSScore] Schedule built: min={min(ssscore_monitor.schedule.values()):.4f}, "
              f"max={max(ssscore_monitor.schedule.values()):.4f}, "
              f"threshold={args.monitoring_threshold}")
        critical_count = sum(1 for v in ssscore_monitor.schedule.values() if v > args.monitoring_threshold)
        print(f"[SSScore] Critical timesteps (>{args.monitoring_threshold}): "
              f"{critical_count}/{len(ssscore_monitor.schedule)}")

    # --- New monitoring modes ---
    noise_div_monitor = None
    grad_norm_monitor = None
    safe_prompt_embeds = None  # for noise_div

    if args.monitoring_mode in ("noise_div", "noise_div_sticky",
                                "noise_div_free", "noise_div_free_sticky"):
        noise_div_monitor = NoiseDivMonitor(device=str(device))
        # Pre-encode safe prompt for noise_div (full mode)
        if args.monitoring_mode in ("noise_div", "noise_div_sticky"):
            with torch.no_grad():
                safe_inputs = tokenizer(
                    args.noise_div_safe_prompt, padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt"
                )
                safe_prompt_embeds = text_encoder(safe_inputs.input_ids.to(device))[0]
            print(f"[NoiseDivMon] Safe prompt encoded: '{args.noise_div_safe_prompt}'")

    if args.monitoring_mode in ("grad_norm", "grad_norm_sticky"):
        grad_norm_monitor = GradNormMonitor(classifier, device=str(device))
        print(f"[GradNormMon] Using classifier gradient norm as monitoring signal")

    # --- Online SSScore (Concept Alignment): cos(d_prompt, d_target) on actual z_t ---
    target_embeds = None
    if args.monitoring_mode in ("online_ssscore", "online_ssscore_sticky"):
        with torch.no_grad():
            target_inputs = tokenizer(
                args.ssscore_target, padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            )
            target_embeds = text_encoder(target_inputs.input_ids.to(device))[0]
        print(f"[ConceptAlign] target='{args.ssscore_target}'")
        print(f"[ConceptAlign] 1 extra UNet forward per timestep (prompt-dependent)")

    if args.monitoring_mode == "ssscore_weighted":
        # Need SSScore for weighting
        ssscore_monitor = SSSCoreMonitor(device=str(device))
        cache_path = args.ssscore_cache or os.path.join(args.output_dir, "ssscore_cache.json")
        scores, timestep_indices = SSSCoreMonitor.load_cache(cache_path)
        if scores is not None:
            print(f"[SSScoreWeighted] Loaded cached scores from {cache_path}")
        else:
            print(f"[SSScoreWeighted] Computing SSScore (target='{args.ssscore_target}', "
                  f"anchor='{args.ssscore_anchor}')...")
            scores, timestep_indices = SSSCoreMonitor.compute_ssscore(
                unet, scheduler, tokenizer, text_encoder,
                target_concept=args.ssscore_target,
                anchor_concept=args.ssscore_anchor,
                n_eval_timesteps=args.num_inference_steps,
                n_samples=args.ssscore_n_samples,
                device=str(device),
            )
            SSSCoreMonitor.save_cache(cache_path, scores, timestep_indices)
        scheduler.set_timesteps(args.num_inference_steps, device=device)
        ssscore_monitor.build_schedule(scores, timestep_indices, scheduler.timesteps)
        print(f"[SSScoreWeighted] Schedule: min={min(ssscore_monitor.schedule.values()):.4f}, "
              f"max={max(ssscore_monitor.schedule.values()):.4f}")

    # Initialize spatial guidance
    guidance = SpatialGuidance(classifier, stats_map, args.gradcam_layer, device)

    # Attention setup
    attention_store = None
    if args.spatial_mode in ("attention", "attention_gradcam"):
        from geo_utils.attention_utils import (
            AttentionStore, register_attention_store, find_token_indices,
        )
        attention_store = AttentionStore()
        register_attention_store(unet, attention_store, args.attn_resolutions)
        guidance.attention_store = attention_store

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []

    for prompt_idx, prompt in tqdm(prompts_with_idx, desc="Generating"):
        # Attention: find token indices per prompt
        if attention_store is not None:
            from geo_utils.attention_utils import find_token_indices
            token_indices = find_token_indices(prompt, args.harmful_keywords, tokenizer)
            guidance.token_indices = token_indices
            if args.debug:
                print(f"  [{prompt_idx:03d}] Token indices: {token_indices} "
                      f"for keywords={args.harmful_keywords}")

        for sample_idx in range(args.nsamples):
            current_seed = args.seed + prompt_idx * args.nsamples + sample_idx
            set_seed(current_seed)

            # Reset monitors
            if monitor:
                monitor.reset_stats()
            if z0_monitor:
                z0_monitor.reset_stats()
            if ssscore_monitor:
                ssscore_monitor.reset_stats()
            if noise_div_monitor:
                noise_div_monitor.reset_stats()
            if grad_norm_monitor:
                grad_norm_monitor.reset_stats()
            if attention_store:
                attention_store.reset()

            sticky_triggered = False
            sticky_classes = []
            z0_trigger_cdf_triggered = False
            dual_path_decided = False
            dual_path_use_guided = False
            dual_path_z0_p_harm = 0.0
            guided_steps_count = 0
            skipped_steps_count = 0
            step_history = []

            # --- Encode text for CFG ---
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
            prompt_embeds = torch.cat([uncond_embeds, text_embeds])  # [2, 77, 768]

            # --- Initialize latents ---
            set_seed(current_seed)
            latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
            latents = latents * scheduler.init_noise_sigma
            scheduler.set_timesteps(args.num_inference_steps, device=device)

            # dual_path: initialize guided path with same initial latents
            latents_guided = None
            if args.monitoring_mode == "dual_path":
                latents_guided = latents.clone()

            # --- Manual DDIM denoising loop ---
            for step_idx, t in enumerate(scheduler.timesteps):

                # ============================================================
                # DUAL_PATH: before decision, run two parallel UNet forwards
                # ============================================================
                if (args.monitoring_mode == "dual_path"
                        and not dual_path_decided):

                    # --- Path B (unguided) ---
                    latent_input_b = torch.cat([latents] * 2)
                    latent_input_b = scheduler.scale_model_input(
                        latent_input_b, t)
                    with torch.no_grad():
                        noise_raw_b = unet(
                            latent_input_b, t,
                            encoder_hidden_states=prompt_embeds
                        ).sample
                    noise_uncond_b, noise_text_b = noise_raw_b.chunk(2)
                    noise_pred_b = noise_uncond_b + args.cfg_scale * (
                        noise_text_b - noise_uncond_b)
                    prev_latents_b = latents.clone()
                    latents = scheduler.step(
                        noise_pred_b, t, latents).prev_sample

                    # --- Path A (guided) ---
                    latent_input_a = torch.cat([latents_guided] * 2)
                    latent_input_a = scheduler.scale_model_input(
                        latent_input_a, t)
                    with torch.no_grad():
                        noise_raw_a = unet(
                            latent_input_a, t,
                            encoder_hidden_states=prompt_embeds
                        ).sample
                    noise_uncond_a, noise_text_a = noise_raw_a.chunk(2)
                    noise_pred_a = noise_uncond_a + args.cfg_scale * (
                        noise_text_a - noise_uncond_a)
                    latents_guided = scheduler.step(
                        noise_pred_a, t, latents_guided).prev_sample

                    # Path A: CDF monitoring + guidance
                    gc_guide_a = False
                    gc_classes_a = []
                    gc_info_a = {}
                    if args.guidance_start_step <= step_idx <= args.guidance_end_step:
                        gc_guide_a, gc_classes_a, gc_info_a = (
                            monitor.should_apply_guidance(
                                latents_guided, t, args.cdf_threshold,
                                step_idx
                            )
                        )
                        if gc_guide_a:
                            spatial_thr = get_spatial_threshold(
                                step_idx, args.num_inference_steps,
                                args.spatial_threshold_start,
                                args.spatial_threshold_end,
                                args.spatial_threshold_strategy
                            )
                            grad = guidance.compute_gradient(
                                latents_guided, t, gc_classes_a,
                                spatial_thr,
                                args.guidance_scale,
                                args.base_guidance_scale,
                                spatial_mode=args.spatial_mode
                            )
                            latents_guided = latents_guided + grad

                    # Decision step: z0 check on Path B
                    if step_idx == args.z0_decision_step:
                        z0_guide, z0_classes, z0_info = (
                            z0_monitor.should_apply_guidance(
                                prev_latents_b, noise_pred_b, t,
                                args.monitoring_threshold, step_idx
                            )
                        )
                        dual_path_z0_p_harm = z0_info.get(
                            "z0_p_harm", 0)
                        dual_path_decided = True
                        if z0_guide:
                            # Use Path A (guided)
                            latents = latents_guided
                            dual_path_use_guided = True
                        # Free guided path memory
                        latents_guided = None

                    info = {
                        "step": step_idx,
                        "monitoring_mode": "dual_path",
                        "phase": ("decision" if step_idx == args.z0_decision_step
                                  else "dual"),
                        "path_a_guided": gc_guide_a,
                        "path_a_cdf": gc_info_a.get("p_harm", {}),
                    }
                    if step_idx == args.z0_decision_step:
                        info["z0_p_harm"] = dual_path_z0_p_harm
                        info["z0_triggered"] = dual_path_use_guided

                    if gc_guide_a:
                        guided_steps_count += 1
                    else:
                        skipped_steps_count += 1

                    step_history.append({
                        **info, "guided": gc_guide_a
                    })
                    continue  # skip the normal path below

                # ============================================================
                # Normal single-path: UNet forward + CFG + DDIM step
                # ============================================================
                # 1. UNet forward (CFG: uncond + cond)
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

                # 3. Save pre-step latents for z0-trigger
                prev_latents = latents.clone()

                # 4. DDIM step
                latents = scheduler.step(noise_pred, t, latents).prev_sample

                # 5. Monitoring + guidance (on post-step latents)
                if args.guidance_start_step <= step_idx <= args.guidance_end_step:
                    should_guide = False
                    active_classes = []
                    info = {"step": step_idx, "p_harm": {}}

                    if args.monitoring_mode == "z0_sticky" and sticky_triggered:
                        should_guide = True
                        active_classes = sticky_classes
                        # Still record z0 stats for logging
                        _, _, info = z0_monitor.should_apply_guidance(
                            prev_latents, noise_pred, t,
                            args.monitoring_threshold, step_idx
                        )
                        info["sticky"] = True
                    elif args.monitoring_mode == "dual" and sticky_triggered:
                        # Phase 2: GradCAM CDF per-step decision
                        gc_guide, gc_classes, gc_info = (
                            monitor.should_apply_guidance(
                                latents, t, args.cdf_threshold, step_idx
                            )
                        )
                        # Also record z0 for logging
                        _, _, z0_info = z0_monitor.should_apply_guidance(
                            prev_latents, noise_pred, t,
                            args.monitoring_threshold, step_idx
                        )
                        should_guide = gc_guide  # only if GradCAM CDF also agrees
                        active_classes = gc_classes if gc_guide else []
                        info = {
                            "step": step_idx,
                            "monitoring_mode": "dual",
                            "sticky": True,
                            "p_harm_z0": z0_info.get("z0_p_harm", 0),
                            "p_harm_cdf": gc_info.get("p_harm", {}),
                            "cdf_guided": gc_guide,
                        }
                    elif step_idx < args.monitoring_start_step:
                        info = {"step": step_idx, "p_harm": {},
                                "monitoring_mode": "skip"}
                    elif args.monitoring_mode == "gradcam":
                        should_guide, active_classes, info = (
                            monitor.should_apply_guidance(
                                latents, t, args.monitoring_threshold,
                                step_idx
                            )
                        )
                    elif args.monitoring_mode in ("z0", "z0_sticky"):
                        should_guide, active_classes, info = (
                            z0_monitor.should_apply_guidance(
                                prev_latents, noise_pred, t,
                                args.monitoring_threshold, step_idx
                            )
                        )
                        if should_guide and args.monitoring_mode == "z0_sticky":
                            sticky_triggered = True
                            sticky_classes = list(
                                set(sticky_classes + active_classes)
                            )
                    elif args.monitoring_mode == "dual":
                        # Phase 1: z0 sticky trigger
                        z0_guide, z0_classes, z0_info = (
                            z0_monitor.should_apply_guidance(
                                prev_latents, noise_pred, t,
                                args.monitoring_threshold, step_idx
                            )
                        )
                        info = {
                            "step": step_idx,
                            "monitoring_mode": "dual",
                            "p_harm_z0": z0_info.get("z0_p_harm", 0),
                            "sticky": False,
                        }
                        if z0_guide:
                            sticky_triggered = True
                            sticky_classes = list(
                                set(sticky_classes + z0_classes)
                            )
                            # Also check GradCAM CDF this step
                            gc_guide, gc_classes, gc_info = (
                                monitor.should_apply_guidance(
                                    latents, t, args.cdf_threshold,
                                    step_idx
                                )
                            )
                            should_guide = gc_guide
                            active_classes = gc_classes if gc_guide else []
                            info["p_harm_cdf"] = gc_info.get("p_harm", {})
                            info["cdf_guided"] = gc_guide
                            info["sticky"] = True
                    elif args.monitoring_mode == "z0_trigger_cdf":
                        if z0_trigger_cdf_triggered:
                            # Post-trigger: GradCAM CDF per-step
                            gc_guide, gc_classes, gc_info = (
                                monitor.should_apply_guidance(
                                    latents, t, args.cdf_threshold,
                                    step_idx
                                )
                            )
                            should_guide = gc_guide
                            active_classes = gc_classes if gc_guide else []
                            info = {
                                "step": step_idx,
                                "monitoring_mode": "z0_trigger_cdf",
                                "phase": "cdf",
                                "cdf_guided": gc_guide,
                                "p_harm_cdf": gc_info.get("p_harm", {}),
                            }
                        elif step_idx == args.z0_decision_step:
                            # Decision step: single z0 check
                            z0_guide, z0_classes, z0_info = (
                                z0_monitor.should_apply_guidance(
                                    prev_latents, noise_pred, t,
                                    args.monitoring_threshold, step_idx
                                )
                            )
                            info = {
                                "step": step_idx,
                                "monitoring_mode": "z0_trigger_cdf",
                                "phase": "z0_decision",
                                "z0_p_harm": z0_info.get("z0_p_harm", 0),
                            }
                            if z0_guide:
                                z0_trigger_cdf_triggered = True
                                # Also check GradCAM CDF this step
                                gc_guide, gc_classes, gc_info = (
                                    monitor.should_apply_guidance(
                                        latents, t, args.cdf_threshold,
                                        step_idx
                                    )
                                )
                                should_guide = gc_guide
                                active_classes = (
                                    gc_classes if gc_guide else []
                                )
                                info["cdf_guided"] = gc_guide
                                info["p_harm_cdf"] = gc_info.get(
                                    "p_harm", {}
                                )
                        else:
                            # Before decision step: skip
                            info = {
                                "step": step_idx,
                                "monitoring_mode": "z0_trigger_cdf",
                                "phase": "skip",
                            }
                    elif args.monitoring_mode == "dual_path":
                        # Post-decision: single path
                        if dual_path_use_guided:
                            # z0 triggered → continue CDF guidance
                            gc_guide, gc_classes, gc_info = (
                                monitor.should_apply_guidance(
                                    latents, t, args.cdf_threshold,
                                    step_idx
                                )
                            )
                            should_guide = gc_guide
                            active_classes = gc_classes if gc_guide else []
                            info = {
                                "step": step_idx,
                                "monitoring_mode": "dual_path",
                                "phase": "post_guided",
                                "cdf_guided": gc_guide,
                                "p_harm_cdf": gc_info.get("p_harm", {}),
                            }
                        else:
                            # z0 not triggered → no guidance
                            info = {
                                "step": step_idx,
                                "monitoring_mode": "dual_path",
                                "phase": "post_unguided",
                            }
                    elif args.monitoring_mode == "ssscore":
                        should_guide, active_classes, info = (
                            ssscore_monitor.should_apply_guidance(
                                step_idx, args.monitoring_threshold
                            )
                        )
                    elif args.monitoring_mode == "ssscore_sticky":
                        if sticky_triggered:
                            # Already triggered → always guide
                            should_guide = True
                            active_classes = sticky_classes
                            ssscore_val = ssscore_monitor.schedule.get(step_idx, 0.0)
                            info = {
                                "step": step_idx,
                                "ssscore": ssscore_val,
                                "monitoring_mode": "ssscore_sticky",
                                "sticky": True,
                            }
                            ssscore_monitor.stats["total_steps"] += 1
                            ssscore_monitor.stats["guided_steps"] += 1
                        else:
                            should_guide, active_classes, info = (
                                ssscore_monitor.should_apply_guidance(
                                    step_idx, args.monitoring_threshold
                                )
                            )
                            info["monitoring_mode"] = "ssscore_sticky"
                            if should_guide:
                                sticky_triggered = True
                                sticky_classes = list(
                                    set(sticky_classes + active_classes)
                                )
                                info["sticky_trigger"] = True

                    # --- noise_div: online noise divergence ---
                    elif args.monitoring_mode == "noise_div":
                        # Extra UNet forward with safe prompt
                        with torch.no_grad():
                            safe_embeds_batch = torch.cat([uncond_embeds, safe_prompt_embeds])
                            safe_input = scheduler.scale_model_input(
                                torch.cat([prev_latents] * 2), t)
                            noise_safe_raw = unet(
                                safe_input, t,
                                encoder_hidden_states=safe_embeds_batch
                            ).sample
                            _, noise_safe_text = noise_safe_raw.chunk(2)
                        div = noise_div_monitor.compute_divergence_full(
                            noise_pred_text, noise_safe_text)
                        should_guide, active_classes, info = (
                            noise_div_monitor.should_apply_guidance(
                                div, args.monitoring_threshold, step_idx))

                    elif args.monitoring_mode == "noise_div_sticky":
                        if sticky_triggered:
                            should_guide = True
                            active_classes = sticky_classes
                            info = {"step": step_idx,
                                    "monitoring_mode": "noise_div_sticky",
                                    "sticky": True}
                        else:
                            with torch.no_grad():
                                safe_embeds_batch = torch.cat([uncond_embeds, safe_prompt_embeds])
                                safe_input = scheduler.scale_model_input(
                                    torch.cat([prev_latents] * 2), t)
                                noise_safe_raw = unet(
                                    safe_input, t,
                                    encoder_hidden_states=safe_embeds_batch
                                ).sample
                                _, noise_safe_text = noise_safe_raw.chunk(2)
                            div = noise_div_monitor.compute_divergence_full(
                                noise_pred_text, noise_safe_text)
                            should_guide, active_classes, info = (
                                noise_div_monitor.should_apply_guidance(
                                    div, args.monitoring_threshold, step_idx))
                            info["monitoring_mode"] = "noise_div_sticky"
                            if should_guide:
                                sticky_triggered = True
                                sticky_classes = [2, 3]
                                info["sticky_trigger"] = True

                    # --- noise_div_free: CFG magnitude proxy ---
                    elif args.monitoring_mode == "noise_div_free":
                        div = noise_div_monitor.compute_divergence_free(
                            noise_pred_text, noise_pred_uncond)
                        should_guide, active_classes, info = (
                            noise_div_monitor.should_apply_guidance(
                                div, args.monitoring_threshold, step_idx))
                        info["monitoring_mode"] = "noise_div_free"

                    elif args.monitoring_mode == "noise_div_free_sticky":
                        if sticky_triggered:
                            should_guide = True
                            active_classes = sticky_classes
                            info = {"step": step_idx,
                                    "monitoring_mode": "noise_div_free_sticky",
                                    "sticky": True}
                        else:
                            div = noise_div_monitor.compute_divergence_free(
                                noise_pred_text, noise_pred_uncond)
                            should_guide, active_classes, info = (
                                noise_div_monitor.should_apply_guidance(
                                    div, args.monitoring_threshold, step_idx))
                            info["monitoring_mode"] = "noise_div_free_sticky"
                            if should_guide:
                                sticky_triggered = True
                                sticky_classes = [2, 3]
                                info["sticky_trigger"] = True

                    # --- ssscore_weighted: SSScore as continuous multiplier ---
                    elif args.monitoring_mode == "ssscore_weighted":
                        ssscore_val = ssscore_monitor.schedule.get(step_idx, 0.0)
                        # Always guide, but scale by SSScore
                        weight = max(0.0, (ssscore_val - args.ssscore_weight_min)) / \
                                 max(1.0 - args.ssscore_weight_min, 1e-8)
                        weight = min(weight, 1.0) ** args.ssscore_weight_power
                        should_guide = weight > 0.01  # skip if negligible
                        active_classes = [2, 3] if should_guide else []
                        info = {
                            "step": step_idx,
                            "ssscore": ssscore_val,
                            "weight": weight,
                            "monitoring_mode": "ssscore_weighted",
                            "active_classes": active_classes,
                        }
                        if should_guide:
                            ssscore_monitor.stats["guided_steps"] = \
                                ssscore_monitor.stats.get("guided_steps", 0) + 1
                        else:
                            ssscore_monitor.stats["skipped_steps"] = \
                                ssscore_monitor.stats.get("skipped_steps", 0) + 1

                    # --- grad_norm: classifier gradient norm ---
                    elif args.monitoring_mode == "grad_norm":
                        should_guide, active_classes, info = (
                            grad_norm_monitor.should_apply_guidance(
                                latents, t, args.monitoring_threshold,
                                step_idx))

                    elif args.monitoring_mode == "grad_norm_sticky":
                        if sticky_triggered:
                            should_guide = True
                            active_classes = sticky_classes
                            info = {"step": step_idx,
                                    "monitoring_mode": "grad_norm_sticky",
                                    "sticky": True}
                        else:
                            should_guide, active_classes, info = (
                                grad_norm_monitor.should_apply_guidance(
                                    latents, t, args.monitoring_threshold,
                                    step_idx))
                            info["monitoring_mode"] = "grad_norm_sticky"
                            if should_guide:
                                sticky_triggered = True
                                sticky_classes = [2, 3]
                                info["sticky_trigger"] = True

                    # --- online_ssscore: concept alignment on actual z_t ---
                    elif args.monitoring_mode in ("online_ssscore", "online_ssscore_sticky"):
                        if args.monitoring_mode == "online_ssscore_sticky" and sticky_triggered:
                            should_guide = True
                            active_classes = sticky_classes
                            info = {"step": step_idx,
                                    "monitoring_mode": "online_ssscore_sticky",
                                    "sticky": True}
                        else:
                            # Concept alignment: cos(d_prompt, d_target)
                            # d_prompt = ε(prompt) - ε(∅) — already from CFG
                            # d_target = ε(target) - ε(∅) — 1 extra UNet forward
                            with torch.no_grad():
                                scaled_latent = scheduler.scale_model_input(prev_latents, t)
                                noise_target = unet(
                                    scaled_latent.to(unet.dtype), t,
                                    encoder_hidden_states=target_embeds.to(unet.dtype)
                                ).sample.float()

                            d_prompt = (noise_pred_text - noise_pred_uncond).float().flatten()
                            d_target = (noise_target - noise_pred_uncond.float()).flatten()
                            cas = torch.nn.functional.cosine_similarity(
                                d_prompt.unsqueeze(0), d_target.unsqueeze(0)
                            ).item()

                            should_guide = cas > args.monitoring_threshold
                            active_classes = [2, 3] if should_guide else []
                            info = {
                                "step": step_idx,
                                "concept_alignment": cas,
                                "monitoring_mode": args.monitoring_mode,
                            }
                            if should_guide and args.monitoring_mode == "online_ssscore_sticky":
                                sticky_triggered = True
                                sticky_classes = [2, 3]
                                info["sticky_trigger"] = True

                    # Ensure active_classes is in info for tracking
                    info["active_classes"] = active_classes

                    if args.debug and step_idx % 5 == 0:
                        p_str = ", ".join(
                            [f"c{k}:{v:.3f}"
                             for k, v in info.get("p_harm", {}).items()]
                        )
                        z0_str = (f" z0={info.get('z0_p_harm', 'N/A')}"
                                  if "z0_p_harm" in info else "")
                        ss_str = (f" ss={info.get('ssscore', 'N/A'):.4f}"
                                  if "ssscore" in info else "")
                        sticky_str = " [STICKY]" if sticky_triggered else ""
                        mode_str = info.get("monitoring_mode",
                                            args.monitoring_mode)
                        print(f"  Step {step_idx}: [{mode_str}] "
                              f"P(harm)=[{p_str}]{z0_str}{ss_str}"
                              f" -> {'GUIDE' if should_guide else 'skip'}"
                              f"{sticky_str}")

                    # Apply guidance
                    if should_guide:
                        spatial_thr = get_spatial_threshold(
                            step_idx, args.num_inference_steps,
                            args.spatial_threshold_start,
                            args.spatial_threshold_end,
                            args.spatial_threshold_strategy
                        )
                        # ssscore_weighted: scale guidance by SSScore weight
                        eff_gs = args.guidance_scale
                        eff_bs = args.base_guidance_scale
                        if args.monitoring_mode == "ssscore_weighted":
                            w = info.get("weight", 1.0)
                            eff_gs = args.guidance_scale * w
                            eff_bs = args.base_guidance_scale * w

                        grad = guidance.compute_gradient(
                            latents, t, active_classes, spatial_thr,
                            eff_gs, eff_bs,
                            spatial_mode=args.spatial_mode
                        )
                        latents = latents + grad
                        guided_steps_count += 1
                    else:
                        skipped_steps_count += 1

                    step_history.append({
                        **info, "guided": should_guide
                    })

            # --- Decode latents to image ---
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
            img_filename = (
                f"{prompt_idx:04d}_{sample_idx:02d}_{safe_prompt}.png"
            )
            save_image(image, output_dir / img_filename)

            total_steps = guided_steps_count + skipped_steps_count

            # Per-class counts
            class_guidance_counts = {2: 0, 3: 0}
            for step_info in step_history:
                if step_info.get("guided", False):
                    for cls in step_info.get("active_classes", []):
                        class_guidance_counts[cls] = (
                            class_guidance_counts.get(cls, 0) + 1
                        )

            img_stats = {
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "seed": current_seed,
                "prompt": prompt[:100],
                "filename": img_filename,
                "monitoring_mode": args.monitoring_mode,
                "spatial_mode": args.spatial_mode,
                "guided_steps": guided_steps_count,
                "skipped_steps": skipped_steps_count,
                "total_steps": total_steps,
                "guidance_ratio": (
                    guided_steps_count / max(total_steps, 1)
                ),
                "class_guidance_counts": class_guidance_counts,
                "sticky_triggered": sticky_triggered,
                "z0_trigger_cdf_triggered": z0_trigger_cdf_triggered,
                "dual_path_use_guided": dual_path_use_guided,
                "dual_path_z0_p_harm": dual_path_z0_p_harm,
            }
            if args.debug:
                img_stats["step_history"] = step_history
            all_stats.append(img_stats)

            print(
                f"  [{prompt_idx:03d}] Guided: "
                f"{guided_steps_count}/{total_steps} "
                f"({img_stats['guidance_ratio']*100:.1f}%) | "
                f"class2: {class_guidance_counts[2]}, "
                f"class3: {class_guidance_counts[3]}"
                f"{' [STICKY]' if sticky_triggered else ''}"
            )

    # Summary
    total_images = len(all_stats)
    avg_guided = (np.mean([s["guided_steps"] for s in all_stats])
                  if all_stats else 0)
    avg_skipped = (np.mean([s["skipped_steps"] for s in all_stats])
                   if all_stats else 0)
    avg_ratio = (np.mean([s["guidance_ratio"] for s in all_stats])
                 if all_stats else 0)

    no_guidance = sum(1 for s in all_stats if s["guided_steps"] == 0)
    light_guidance = sum(
        1 for s in all_stats if 0 < s["guidance_ratio"] <= 0.3
    )
    medium_guidance = sum(
        1 for s in all_stats if 0.3 < s["guidance_ratio"] <= 0.7
    )
    heavy_guidance = sum(
        1 for s in all_stats if s["guidance_ratio"] > 0.7
    )

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
    print(f"Monitoring: {args.monitoring_mode}, Spatial: {args.spatial_mode}")
    print(f"\n[Guidance Statistics]")
    print(f"  Avg guided steps: {avg_guided:.1f}/{args.num_inference_steps}"
          f" ({avg_ratio*100:.1f}%)")
    print(f"\n[Guidance Distribution]")
    print(f"  No guidance (0%):      {no_guidance:3d} images"
          f" ({no_guidance/max(total_images,1)*100:.1f}%)")
    print(f"  Light (1-30%):         {light_guidance:3d} images"
          f" ({light_guidance/max(total_images,1)*100:.1f}%)")
    print(f"  Medium (31-70%):       {medium_guidance:3d} images"
          f" ({medium_guidance/max(total_images,1)*100:.1f}%)")
    print(f"  Heavy (71-100%):       {heavy_guidance:3d} images"
          f" ({heavy_guidance/max(total_images,1)*100:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
