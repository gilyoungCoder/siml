#!/usr/bin/env python
"""
Compare safe_harm_restricted vs prob_threshold methods.
Track how many steps guidance was applied.
"""

import json
import random
from argparse import ArgumentParser
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDIMScheduler
from geo_models.classifier.classifier import load_discriminator
from geo_utils.classifier_interpretability import ClassifierGradCAM


def load_gradcam_stats_map(stats_dir):
    stats_dir = Path(stats_dir)
    mapping = {2: "gradcam_stats_harm_nude_class2.json", 3: "gradcam_stats_harm_color_class3.json"}
    stats_map = {}
    for cls, fname in mapping.items():
        path = stats_dir / fname
        if path.exists():
            with open(path) as f: d = json.load(f)
            stats_map[cls] = {"mean": float(d["mean"]), "std": float(d["std"])}
    return stats_map


class GuidanceTracker:
    def __init__(self, classifier, gradcam_layer, device, gradcam_stats_map):
        self.classifier = classifier.to(device)
        self.classifier.eval()
        self.device = device
        self.dtype = next(self.classifier.parameters()).dtype
        self.gradcam_stats_map = gradcam_stats_map
        self.gradcam = ClassifierGradCAM(classifier, gradcam_layer)
        if hasattr(self.classifier, "encoder_model"):
            self.classifier.encoder_model = self.classifier.encoder_model.to(device)

    def _cdf_norm(self, heatmap, mean, std):
        z = (heatmap - mean) / (std + 1e-8)
        from torch.distributions import Normal
        n = Normal(torch.tensor(0.0, device=heatmap.device), torch.tensor(1.0, device=heatmap.device))
        return n.cdf(z)

    def _heatmap(self, latent, norm_t, cls):
        use_abs = self.gradcam_stats_map and cls in self.gradcam_stats_map
        with torch.enable_grad():
            hm, _ = self.gradcam.generate_heatmap(latent, norm_t, cls, normalize=not use_abs)
        if use_abs:
            hm = self._cdf_norm(hm, self.gradcam_stats_map[cls]["mean"], self.gradcam_stats_map[cls]["std"])
        return hm

    def get_spatial_threshold(self, step, total=50):
        t = step / max(total - 1, 1)
        return 0.1 + (0.5 - 0.1) * 0.5 * (1 + np.cos(np.pi * t))

    def safe_harm_restricted_grad(self, latent, timestep, step):
        """Always applies guidance."""
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        norm_t = timestep.float() / 1000.0
        lat = latent.to(dtype=self.dtype)
        thr = self.get_spatial_threshold(step)

        with torch.no_grad():
            logits = self.classifier(lat, norm_t)
            harm_cls = 2 if logits[0, 2] > logits[0, 3] else 3

        hm = self._heatmap(lat, norm_t, harm_cls)
        mask = (hm >= thr).float()
        if mask.dim() == 3: mask = mask.unsqueeze(1)

        with torch.enable_grad():
            l1 = latent.detach().to(dtype=self.dtype).requires_grad_(True)
            g_safe = torch.autograd.grad(self.classifier(l1, norm_t)[:, 1].sum(), l1)[0]
            l2 = latent.detach().to(dtype=self.dtype).requires_grad_(True)
            g_harm = torch.autograd.grad(self.classifier(l2, norm_t)[:, harm_cls].sum(), l2)[0]

            def proj_out(a, b):
                af, bf = a.view(-1), b.view(-1)
                return a - torch.dot(af, bf) / (torch.dot(bf, bf) + 1e-8) * b

            d_safe = proj_out(g_safe, g_harm)
            d_harm = proj_out(g_harm, g_safe)
            grad = d_safe - d_harm

        weight = mask * 10.0 + (1 - mask) * 2.0
        return (grad * weight).to(dtype=latent.dtype).detach(), True

    def prob_threshold_grad(self, latent, timestep, step, prob_thr=0.2):
        """Only applies if prob > threshold."""
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        norm_t = timestep.float() / 1000.0
        lat = latent.to(dtype=self.dtype)
        thr = self.get_spatial_threshold(step)

        with torch.no_grad():
            logits = self.classifier(lat, norm_t)
            probs = F.softmax(logits, dim=1)[0]

        active = [c for c in [2, 3] if probs[c].item() > prob_thr]
        if not active:
            return torch.zeros_like(latent), False

        masks = {}
        for c in active:
            hm = self._heatmap(lat, norm_t, c)
            m = (hm >= thr).float()
            if m.dim() == 3: m = m.unsqueeze(1)
            masks[c] = m

        with torch.enable_grad():
            l1 = latent.detach().to(dtype=self.dtype).requires_grad_(True)
            g_safe = torch.autograd.grad(self.classifier(l1, norm_t)[:, 1].sum(), l1)[0]
            g_harm = torch.zeros_like(g_safe)
            for c in active:
                l2 = latent.detach().to(dtype=self.dtype).requires_grad_(True)
                g_harm += torch.autograd.grad(self.classifier(l2, norm_t)[:, c].sum(), l2)[0]
            grad = g_safe - g_harm

        combined = None
        for c in active:
            combined = masks[c] if combined is None else torch.max(combined, masks[c])
        weight = combined * 10.0 + (1 - combined) * 2.0
        return (grad * weight).to(dtype=latent.dtype).detach(), True

    def gradcam_threshold_grad(self, latent, timestep, step, gradcam_thr=0.3):
        """Only applies if mean(CDF-normalized GradCAM heatmap) > threshold."""
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=latent.device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        norm_t = timestep.float() / 1000.0
        lat = latent.to(dtype=self.dtype)
        spatial_thr = self.get_spatial_threshold(step)

        # Check GradCAM MEAN value for each harm class (CDF normalized)
        active = []
        heatmaps = {}
        for c in [2, 3]:
            hm = self._heatmap(lat, norm_t, c)  # Already CDF normalized
            heatmaps[c] = hm
            mean_val = hm.mean().item()
            if mean_val > gradcam_thr:
                active.append(c)

        if not active:
            return torch.zeros_like(latent), False, {}

        masks = {}
        for c in active:
            m = (heatmaps[c] >= spatial_thr).float()
            if m.dim() == 3: m = m.unsqueeze(1)
            masks[c] = m

        with torch.enable_grad():
            l1 = latent.detach().to(dtype=self.dtype).requires_grad_(True)
            g_safe = torch.autograd.grad(self.classifier(l1, norm_t)[:, 1].sum(), l1)[0]
            l2 = latent.detach().to(dtype=self.dtype).requires_grad_(True)
            g_harm = torch.autograd.grad(self.classifier(l2, norm_t)[:, active[0]].sum(), l2)[0]

            def proj_out(a, b):
                af, bf = a.view(-1), b.view(-1)
                return a - torch.dot(af, bf) / (torch.dot(bf, bf) + 1e-8) * b

            d_safe = proj_out(g_safe, g_harm)
            d_harm = proj_out(g_harm, g_safe)
            grad = d_safe - d_harm

        combined = None
        for c in active:
            combined = masks[c] if combined is None else torch.max(combined, masks[c])
        weight = combined * 10.0 + (1 - combined) * 2.0

        info = {c: {"mean": heatmaps[c].mean().item(), "max": heatmaps[c].max().item()} for c in [2, 3]}
        return (grad * weight).to(dtype=latent.dtype).detach(), True, info


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompt_file", default="./prompts/sexual_50.txt")
    parser.add_argument("--output_dir", default="./scg_outputs/guidance_comparison")
    parser.add_argument("--classifier_ckpt", default="./work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth")
    parser.add_argument("--gradcam_stats_dir", default="./gradcam_stats/nudity_4class")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prob_thr", type=float, default=0.1, help="Prob threshold for harm detection")
    parser.add_argument("--gradcam_thr", type=float, default=0.3, help="GradCAM CDF threshold for harm detection")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    accelerator = Accelerator()
    device = accelerator.device

    print(f"\n{'='*60}\nCOMPARING: safe_harm_restricted vs gradcam_threshold\n{'='*60}\n")

    stats_map = load_gradcam_stats_map(args.gradcam_stats_dir)
    prompts = [l.strip() for l in open(args.prompt_file) if l.strip()][:args.num_images]

    pipe = StableDiffusionPipeline.from_pretrained(args.ckpt_path, torch_dtype=torch.float16, safety_checker=None).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    classifier = load_discriminator(args.classifier_ckpt, None, True, 4, 4).to(device)
    classifier.eval()

    tracker = GuidanceTracker(classifier, "encoder_model.middle_block.2", device, stats_map)

    out = Path(args.output_dir)
    (out / "safe_harm_restricted").mkdir(parents=True, exist_ok=True)
    (out / "gradcam_threshold").mkdir(parents=True, exist_ok=True)

    stats = {
        "safe_harm_restricted": [],
        "gradcam_threshold": {"guided": [], "skipped": [], "max_heatmaps": []}
    }

    for i, prompt in enumerate(tqdm(prompts)):
        print(f"\n[{i+1}/{len(prompts)}] {prompt[:50]}...")
        safe_name = "".join(c if c.isalnum() or c in ' -_' else '_' for c in prompt)[:50].replace(' ', '_')

        # Method 1: safe_harm_restricted
        cnt1 = 0
        def cb1(pipe, step, ts, kwargs):
            nonlocal cnt1
            if 0 <= step <= 50:
                g, applied = tracker.safe_harm_restricted_grad(kwargs["latents"], ts, step)
                if applied: cnt1 += 1
                kwargs["latents"] = kwargs["latents"] + g
            return kwargs

        with torch.no_grad():
            o1 = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, callback_on_step_end=cb1, callback_on_step_end_tensor_inputs=["latents"])
        o1.images[0].resize((512,512)).save(out / "safe_harm_restricted" / f"{i:04d}_{safe_name}.png")
        stats["safe_harm_restricted"].append(cnt1)

        # Method 2: gradcam_threshold
        cnt2, skip2 = 0, 0
        max_hms = []
        def cb2(pipe, step, ts, kwargs):
            nonlocal cnt2, skip2, max_hms
            if 0 <= step <= 50:
                g, applied, info = tracker.gradcam_threshold_grad(kwargs["latents"], ts, step, args.gradcam_thr)
                if info:
                    max_hms.append(info)
                if applied:
                    cnt2 += 1
                    kwargs["latents"] = kwargs["latents"] + g
                else:
                    skip2 += 1
            return kwargs

        with torch.no_grad():
            o2 = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, callback_on_step_end=cb2, callback_on_step_end_tensor_inputs=["latents"])
        o2.images[0].resize((512,512)).save(out / "gradcam_threshold" / f"{i:04d}_{safe_name}.png")
        stats["gradcam_threshold"]["guided"].append(cnt2)
        stats["gradcam_threshold"]["skipped"].append(skip2)
        stats["gradcam_threshold"]["max_heatmaps"].append(max_hms)

        print(f"  safe_harm_restricted: {cnt1}/50 guided")
        print(f"  gradcam_threshold: {cnt2}/50 guided, {skip2} skipped")

    # Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"\nsafe_harm_restricted: avg {np.mean(stats['safe_harm_restricted']):.1f}/50 steps")
    print(f"  per-image: {stats['safe_harm_restricted']}")
    print(f"\ngradcam_threshold (thr={args.gradcam_thr}): avg {np.mean(stats['gradcam_threshold']['guided']):.1f}/50 guided, {np.mean(stats['gradcam_threshold']['skipped']):.1f} skipped")
    print(f"  guided: {stats['gradcam_threshold']['guided']}")
    print(f"  skipped: {stats['gradcam_threshold']['skipped']}")

    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
