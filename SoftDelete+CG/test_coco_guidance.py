#!/usr/bin/env python
"""
Toy test: Run ASCG pipeline on 50 COCO prompts and check if guidance is actually triggered.
Logs per-step classifier probabilities and gradient norms to see if benign prompts get modified.
"""
import os
import sys
import json
import math
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


NUDITY_4CLASS_CONFIG = {
    "benign": 0, "safe_clothed": 1, "harm_nude": 2, "harm_color": 3,
    "safe_classes": [0, 1], "harm_classes": [2, 3], "guidance_target_safe": 1,
    "class_names": {0: "benign", 1: "safe_clothed", 2: "harm_nude", 3: "harm_color"}
}


def load_prompts(f):
    import csv
    f = Path(f)
    prompts = []
    with open(f, "r") as fp:
        reader = csv.DictReader(fp)
        fieldnames = reader.fieldnames
        for col in ['adv_prompt', 'sensitive prompt', 'prompt', 'target_prompt', 'text', 'Prompt', 'Text']:
            if col in fieldnames:
                prompt_col = col
                break
        for row in reader:
            prompts.append(row[prompt_col].strip())
    return prompts


def main():
    parser = ArgumentParser()
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--gradcam_stats_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--n_prompts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_start_t", type=int, default=1000)
    parser.add_argument("--guidance_end_t", type=int, default=400)
    parser.add_argument("--spatial_threshold", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16

    # Load model
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=dtype
    ).to(device)
    vae, unet, text_encoder, tokenizer = pipe.vae, pipe.unet, pipe.text_encoder, pipe.tokenizer
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(50)

    # Load classifier
    classifier = load_discriminator(
        ckpt_path=args.classifier_ckpt, condition=None, eval=True, channel=4, num_classes=4
    ).to(device)
    classifier.eval()

    # Load prompts
    prompts = load_prompts(args.prompt_file)[:args.n_prompts]
    print(f"Testing {len(prompts)} prompts from {args.prompt_file}")

    results = []
    generator = torch.Generator(device=device).manual_seed(args.seed)

    for idx, prompt in enumerate(tqdm(prompts, desc="Testing")):
        # Encode prompt
        tok = tokenizer(prompt, padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            prompt_embeds = text_encoder(tok.input_ids)[0]
        uncond_tok = tokenizer("", padding="max_length", max_length=77,
                               return_tensors="pt").to(device)
        with torch.no_grad():
            uncond_embeds = text_encoder(uncond_tok.input_ids)[0]
        text_embeds = torch.cat([uncond_embeds, prompt_embeds])

        # Init latents
        latents = torch.randn(1, 4, 64, 64, device=device, dtype=dtype, generator=generator)
        latents = latents * scheduler.init_noise_sigma

        prompt_info = {
            "idx": idx,
            "prompt": prompt[:80],
            "steps": [],
            "guided_steps": 0,
            "total_grad_norm": 0.0,
        }

        for step_idx, t in enumerate(scheduler.timesteps):
            t_val = t.item()
            with torch.no_grad():
                noise_pred_raw = unet(
                    torch.cat([latents] * 2),
                    t,
                    encoder_hidden_states=text_embeds
                ).sample
            noise_pred_uncond, noise_pred_text = noise_pred_raw.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            # DDIM step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            # Check if in guidance window
            in_window = args.guidance_end_t <= t_val <= args.guidance_start_t

            if in_window:
                # Get classifier predictions
                lat = latents.detach().to(dtype=dtype)
                norm_t = torch.tensor([t_val / 1000.0], device=device)

                with torch.no_grad():
                    logits = classifier(lat.float(), norm_t)
                    probs = F.softmax(logits, dim=1)[0]

                # Compute gradient norm (without applying)
                with torch.enable_grad():
                    l1 = latents.detach().float().requires_grad_(True)
                    out = classifier(l1, norm_t)
                    g_safe = torch.autograd.grad(out[:, 1].sum(), l1)[0]

                    l2 = latents.detach().float().requires_grad_(True)
                    out2 = classifier(l2, norm_t)
                    g_harm = torch.autograd.grad((out2[:, 2] + out2[:, 3]).sum(), l2)[0]

                grad = g_safe - g_harm
                grad_norm = grad.flatten().norm(p=2).item()

                # After applying scale (20.0) and weight
                window_weight = (args.guidance_start_t - t_val) / (args.guidance_start_t - args.guidance_end_t)
                window_weight = max(0, min(1, 1 - window_weight))  # linear
                effective_grad_norm = grad_norm * 20.0 * window_weight

                step_info = {
                    "step": step_idx,
                    "t": t_val,
                    "probs": {
                        "benign": round(probs[0].item(), 4),
                        "safe": round(probs[1].item(), 4),
                        "harm_nude": round(probs[2].item(), 4),
                        "harm_color": round(probs[3].item(), 4),
                    },
                    "predicted_class": NUDITY_4CLASS_CONFIG["class_names"][probs.argmax().item()],
                    "p_harm": round((probs[2] + probs[3]).item(), 4),
                    "grad_norm_raw": round(grad_norm, 6),
                    "grad_norm_effective": round(effective_grad_norm, 4),
                    "window_weight": round(window_weight, 4),
                }
                prompt_info["steps"].append(step_info)
                prompt_info["total_grad_norm"] += effective_grad_norm
                if effective_grad_norm > 0.01:
                    prompt_info["guided_steps"] += 1

        results.append(prompt_info)

        if idx < 5 or idx % 10 == 0:
            max_harm = max(s["p_harm"] for s in prompt_info["steps"]) if prompt_info["steps"] else 0
            max_grad = max(s["grad_norm_effective"] for s in prompt_info["steps"]) if prompt_info["steps"] else 0
            print(f"  [{idx}] {prompt[:50]:50s} | guided_steps={prompt_info['guided_steps']:2d} "
                  f"| max_p_harm={max_harm:.4f} | max_grad={max_grad:.4f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    guided_counts = [r["guided_steps"] for r in results]
    total_grads = [r["total_grad_norm"] for r in results]
    print(f"Prompts with ANY guidance: {sum(1 for g in guided_counts if g > 0)}/{len(results)}")
    print(f"Avg guided steps: {np.mean(guided_counts):.1f}")
    print(f"Avg total grad norm: {np.mean(total_grads):.4f}")
    print(f"Max total grad norm: {np.max(total_grads):.4f}")

    # Per-prompt summary
    print(f"\n{'Idx':>4s} {'Prompt':50s} {'Guide':>6s} {'TotGrad':>8s} {'MaxHarm':>8s}")
    print("-" * 80)
    for r in results:
        max_harm = max(s["p_harm"] for s in r["steps"]) if r["steps"] else 0
        print(f"{r['idx']:4d} {r['prompt'][:50]:50s} {r['guided_steps']:6d} {r['total_grad_norm']:8.4f} {max_harm:8.4f}")

    # Save
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
