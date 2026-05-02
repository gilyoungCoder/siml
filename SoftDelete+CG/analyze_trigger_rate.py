#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze monitoring trigger rates on Ring-A-Bell (harmful) vs COCO (safe) prompts.
No image saving — only logs trigger statistics per prompt.

Usage:
  python analyze_trigger_rate.py --gpu 0 \
    --monitoring_modes grad_norm_sticky online_ssscore_sticky noise_div_free_sticky \
    --thresholds 0.3 0.45 0.5 \
    --coco_n 100
"""

import os
import sys
import json
import csv
import random
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDIMScheduler
from geo_models.classifier.classifier import load_discriminator


# ── Monitors (copied from generate_unified_monitoring.py) ──

class GradNormMonitor:
    def __init__(self, classifier, device="cuda"):
        self.classifier = classifier
        self.device = device

    def compute_grad_norm(self, latents, t):
        z = latents.detach().float().clone().requires_grad_(True)
        t_input = t.unsqueeze(0) if t.dim() == 0 else t
        logits = self.classifier(z, t_input)
        harm_logits = logits[:, [2, 3]]  # harm_nude, harm_color
        loss = harm_logits.sum()
        loss.backward()
        grad_norm = z.grad.norm().item()
        return grad_norm


class NoiseDivMonitor:
    def compute_divergence_free(self, noise_text, noise_uncond):
        diff = (noise_text - noise_uncond).float().flatten()
        return diff.norm().item()


def load_prompts_csv(path, col="prompt", col_alt="sensitive prompt"):
    """Load prompts from CSV (Ring-A-Bell or COCO format)."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if col in row:
                prompts.append(row[col])
            elif col_alt in row:
                prompts.append(row[col_alt])
    return prompts


def analyze_single_prompt(
    prompt, unet, text_encoder, tokenizer, scheduler, classifier, device,
    monitoring_mode, threshold, seed=42,
    target_embeds=None, num_steps=50
):
    """Run denoising loop for one prompt, return trigger info (no image save)."""
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

    # Init latents
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16, generator=generator)
    scheduler.set_timesteps(num_steps, device=device)
    latents = latents * scheduler.init_noise_sigma

    grad_norm_monitor = GradNormMonitor(classifier, device) if "grad_norm" in monitoring_mode else None
    noise_div_monitor = NoiseDivMonitor() if "noise_div" in monitoring_mode else None

    sticky_triggered = False
    trigger_step = -1
    step_values = []  # raw signal values per step
    guided_steps = 0

    for step_idx, t in enumerate(scheduler.timesteps):
        # UNet forward (CFG)
        latent_input = torch.cat([latents] * 2)
        latent_input = scheduler.scale_model_input(latent_input, t)
        with torch.no_grad():
            noise_raw = unet(latent_input, t, encoder_hidden_states=prompt_embeds).sample
        noise_pred_uncond, noise_pred_text = noise_raw.chunk(2)
        noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

        prev_latents = latents.clone()
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Monitoring
        should_guide = False
        signal_val = 0.0

        if "sticky" in monitoring_mode and sticky_triggered:
            should_guide = True
        else:
            if "grad_norm" in monitoring_mode:
                signal_val = grad_norm_monitor.compute_grad_norm(latents, t)
                should_guide = signal_val > threshold

            elif "online_ssscore" in monitoring_mode:
                # Concept alignment: cos(d_prompt, d_target)
                with torch.no_grad():
                    scaled = scheduler.scale_model_input(prev_latents, t)
                    noise_target = unet(
                        scaled.to(unet.dtype), t,
                        encoder_hidden_states=target_embeds.to(unet.dtype)
                    ).sample.float()
                d_prompt = (noise_pred_text - noise_pred_uncond).float().flatten()
                d_target = (noise_target - noise_pred_uncond.float()).flatten()
                signal_val = F.cosine_similarity(
                    d_prompt.unsqueeze(0), d_target.unsqueeze(0)
                ).item()
                should_guide = signal_val > threshold

            elif "noise_div_free" in monitoring_mode:
                signal_val = noise_div_monitor.compute_divergence_free(
                    noise_pred_text, noise_pred_uncond)
                should_guide = signal_val > threshold

            if should_guide and "sticky" in monitoring_mode and not sticky_triggered:
                sticky_triggered = True
                trigger_step = step_idx

        if should_guide:
            guided_steps += 1

        step_values.append(signal_val)

    return {
        "triggered": sticky_triggered if "sticky" in monitoring_mode else guided_steps > 0,
        "trigger_step": trigger_step,
        "guided_steps": guided_steps,
        "total_steps": num_steps,
        "signal_mean": float(np.mean(step_values)),
        "signal_max": float(np.max(step_values)),
        "signal_min": float(np.min(step_values)),
        "step_values": step_values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--monitoring_modes", nargs="+",
                        default=["grad_norm_sticky", "online_ssscore_sticky", "noise_div_free_sticky"])
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.45])
    parser.add_argument("--rab_file", type=str,
                        default="/mnt/home/yhgil99/unlearning/SAFREE/datasets/nudity-ring-a-bell.csv")
    parser.add_argument("--coco_file", type=str,
                        default="/mnt/home/yhgil99/unlearning/SAFREE/datasets/coco_30k.csv")
    parser.add_argument("--coco_n", type=int, default=100,
                        help="Number of COCO prompts to sample")
    parser.add_argument("--ckpt_path", type=str,
                        default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--classifier_ckpt", type=str,
                        default="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_ringabell/checkpoint/step_19200/classifier.pth")
    parser.add_argument("--ssscore_target", type=str, default="nudity")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str,
                        default="scg_outputs/trigger_analysis")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load prompts
    rab_prompts = load_prompts_csv(args.rab_file, col="prompt", col_alt="sensitive prompt")
    coco_all = load_prompts_csv(args.coco_file, col="prompt", col_alt="prompt")

    random.seed(args.seed)
    coco_prompts = random.sample(coco_all, min(args.coco_n, len(coco_all)))

    print(f"Ring-A-Bell prompts: {len(rab_prompts)}")
    print(f"COCO prompts (sampled): {len(coco_prompts)}")

    # Load model
    print("Loading SD pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    unet = pipe.unet
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    scheduler = pipe.scheduler

    # Load classifier
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt, condition=None, eval=True, channel=4, num_classes=4
    ).to(device)
    classifier.eval()

    # Pre-encode target for online_ssscore
    with torch.no_grad():
        target_inputs = tokenizer(
            args.ssscore_target, padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        target_embeds = text_encoder(target_inputs.input_ids.to(device))[0]

    # Run analysis
    for mode in args.monitoring_modes:
        for thr in args.thresholds:
            print(f"\n{'='*60}")
            print(f"MODE: {mode}, THRESHOLD: {thr}")
            print(f"{'='*60}")

            results = {"mode": mode, "threshold": thr, "rab": [], "coco": []}

            # Ring-A-Bell (harmful)
            print(f"\n--- Ring-A-Bell ({len(rab_prompts)} prompts) ---")
            rab_triggered = 0
            for i, prompt in enumerate(tqdm(rab_prompts, desc="RAB")):
                r = analyze_single_prompt(
                    prompt, unet, text_encoder, tokenizer, scheduler,
                    classifier, device, mode, thr,
                    seed=args.seed, target_embeds=target_embeds,
                    num_steps=args.num_steps
                )
                r["prompt_idx"] = i
                r["prompt"] = prompt[:80]
                del r["step_values"]  # save space
                results["rab"].append(r)
                if r["triggered"]:
                    rab_triggered += 1

            tp_rate = rab_triggered / len(rab_prompts) * 100
            print(f"  TP (triggered on harmful): {rab_triggered}/{len(rab_prompts)} = {tp_rate:.1f}%")

            # COCO (safe)
            print(f"\n--- COCO ({len(coco_prompts)} prompts) ---")
            coco_triggered = 0
            for i, prompt in enumerate(tqdm(coco_prompts, desc="COCO")):
                r = analyze_single_prompt(
                    prompt, unet, text_encoder, tokenizer, scheduler,
                    classifier, device, mode, thr,
                    seed=args.seed, target_embeds=target_embeds,
                    num_steps=args.num_steps
                )
                r["prompt_idx"] = i
                r["prompt"] = prompt[:80]
                del r["step_values"]
                results["coco"].append(r)
                if r["triggered"]:
                    coco_triggered += 1

            fp_rate = coco_triggered / len(coco_prompts) * 100
            print(f"  FP (triggered on safe): {coco_triggered}/{len(coco_prompts)} = {fp_rate:.1f}%")

            print(f"\n  >>> SUMMARY: TP={tp_rate:.1f}%, FP={fp_rate:.1f}%")
            print(f"  >>> Selectivity: TP-FP = {tp_rate - fp_rate:.1f}%p")

            # Save
            out_file = os.path.join(
                args.output_dir,
                f"trigger_{mode}_thr{thr}.json"
            )
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved to {out_file}")


if __name__ == "__main__":
    main()
