#!/usr/bin/env python
"""
Score analysis: z0-trigger + GradCAM CDF for dual monitoring analysis.

At each diffusion step (guidance 적용 없이):
  1. Run UNet → noise prediction → DDIM step
  2. z0 score: predict z0 from zt → classifier(z0, t=0) → softmax → p_harm
  3. GradCAM CDF: classifier GradCAM on (zt, t) → heatmap mean → CDF(z-score)

Output: per-prompt, per-step z0 + CDF scores for offline threshold analysis.

Usage:
    # Ringabell (79 prompts)
    CUDA_VISIBLE_DEVICES=0 python score_analysis_dual.py \
        --prompt_file /path/to/nudity-ring-a-bell.csv \
        --output_path scg_outputs/score_analysis_dual/ringabell_scores.json

    # COCO benign (50 prompts)
    CUDA_VISIBLE_DEVICES=1 python score_analysis_dual.py \
        --prompt_file /path/to/coco_30k.csv \
        --output_path scg_outputs/score_analysis_dual/coco_scores.json \
        --end_idx 50
"""
import csv
import json
import math
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDIMScheduler
from geo_models.classifier.classifier import load_discriminator
from geo_utils.classifier_interpretability import ClassifierGradCAM


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict_z0(zt, noise_pred, alpha_bar):
    """Predict z0 from zt using DDIM formula."""
    return (zt - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)


def load_gradcam_stats(stats_dir: str):
    """Load sample-level GradCAM statistics for CDF computation."""
    stats_dir = Path(stats_dir)
    mapping = {
        2: "gradcam_stats_harm_nude_class2.json",
        3: "gradcam_stats_harm_color_class3.json",
    }
    stats_map = {}
    for cls, fname in mapping.items():
        path = stats_dir / fname
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            sample = d.get("sample_level", {})
            stats_map[cls] = {
                "sample_mean": float(sample.get("mean", d["mean"])),
                "sample_std": float(sample.get("std", d["std"])),
            }
    return stats_map


def compute_gradcam_cdf(classifier, gradcam, stats_map, normal_dist,
                        latent, timestep, harm_class, clf_dtype):
    """Compute GradCAM CDF P(harm) for a single class."""
    if harm_class not in stats_map:
        return 0.0

    lat = latent.to(dtype=clf_dtype)
    if not isinstance(timestep, torch.Tensor):
        timestep = torch.tensor([timestep], device=latent.device)
    elif timestep.dim() == 0:
        timestep = timestep.unsqueeze(0)
    norm_t = timestep.float() / 1000.0

    with torch.enable_grad():
        heatmap, _ = gradcam.generate_heatmap(lat, norm_t, harm_class, normalize=False)

    heatmap_mean = heatmap.mean().item()
    mu = stats_map[harm_class]["sample_mean"]
    sigma = stats_map[harm_class]["sample_std"]
    z = (heatmap_mean - mu) / (sigma + 1e-8)
    if math.isnan(z) or math.isinf(z):
        return 0.0
    return normal_dist.cdf(torch.tensor(z)).item()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str,
                        default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--classifier_ckpt", type=str,
                        default="/mnt/home/yhgil99/unlearning/SoftDelete+CG/"
                                "work_dirs/nudity_4class_safe_combined/checkpoint/"
                                "step_17100/classifier.pth")
    parser.add_argument("--gradcam_stats_dir", type=str,
                        default="/mnt/home/yhgil99/unlearning/SoftDelete+CG/"
                                "gradcam_stats/nudity_4class")
    parser.add_argument("--gradcam_layer", type=str,
                        default="encoder_model.middle_block.2")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load prompts
    prompts = []
    with open(args.prompt_file) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        prompt_col = 0
        if header:
            for i, col in enumerate(header):
                if col.lower() in ["prompt", "text", "caption"]:
                    prompt_col = i
                    break
        for row in reader:
            if row and len(row) > prompt_col:
                prompts.append(row[prompt_col].strip())

    end_idx = args.end_idx if args.end_idx > 0 else len(prompts)
    prompts_with_idx = list(enumerate(prompts))[args.start_idx:end_idx]
    print(f"Loaded {len(prompts_with_idx)} prompts "
          f"(indices {args.start_idx} to {end_idx-1})")

    # Load SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt_path, torch_dtype=torch.float32
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None

    unet = pipe.unet
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    scheduler = pipe.scheduler

    # Load classifier
    classifier = load_discriminator(
        args.classifier_ckpt, condition="", eval=True,
        channel=4, num_classes=4
    ).to(device)
    classifier.eval()
    clf_dtype = next(classifier.parameters()).dtype

    # z0 trigger setup
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # GradCAM CDF setup
    stats_map = load_gradcam_stats(args.gradcam_stats_dir)
    gradcam = ClassifierGradCAM(classifier, args.gradcam_layer)
    normal_dist = Normal(torch.tensor(0.0), torch.tensor(1.0))

    print(f"GradCAM stats loaded for classes: {list(stats_map.keys())}")

    all_results = []

    for prompt_idx, prompt in tqdm(prompts_with_idx, desc="Analyzing"):
        current_seed = args.seed + prompt_idx
        set_seed(current_seed)

        # Encode prompt for CFG
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

        # Initialize noise
        set_seed(current_seed)
        latents = torch.randn(1, 4, 64, 64, device=device)
        latents = latents * scheduler.init_noise_sigma
        scheduler.set_timesteps(args.num_inference_steps, device=device)

        scores = {}
        for i, t in enumerate(scheduler.timesteps):
            # UNet forward (CFG)
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(
                latent_model_input, t
            )

            with torch.no_grad():
                noise_pred = unet(
                    latent_model_input, t,
                    encoder_hidden_states=prompt_embeds
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + args.cfg_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # --- z0 score ---
            alpha_bar = alphas_cumprod[t]
            z0_pred = predict_z0(latents, noise_pred_cfg, alpha_bar)

            with torch.no_grad():
                z0_input = z0_pred.to(dtype=clf_dtype)
                logits = classifier(
                    z0_input, torch.zeros(1, device=device)
                )
                probs = F.softmax(logits, dim=-1)

            z0_p_benign = probs[0, 0].item()
            z0_p_safe = probs[0, 1].item()
            z0_p_nude = probs[0, 2].item()
            z0_p_color = probs[0, 3].item()
            z0_p_harm = z0_p_nude + z0_p_color

            # --- DDIM step (before GradCAM, so we have post-step latents) ---
            prev_latents = latents.clone()
            latents = scheduler.step(noise_pred_cfg, t, latents).prev_sample

            # --- GradCAM CDF score (on post-step latents, same as monitoring) ---
            cdf_p_nude = compute_gradcam_cdf(
                classifier, gradcam, stats_map, normal_dist,
                latents, t, harm_class=2, clf_dtype=clf_dtype
            )
            cdf_p_color = compute_gradcam_cdf(
                classifier, gradcam, stats_map, normal_dist,
                latents, t, harm_class=3, clf_dtype=clf_dtype
            )

            scores[i] = {
                "step": i,
                "timestep": t.item(),
                "z0_p_benign": round(z0_p_benign, 4),
                "z0_p_safe": round(z0_p_safe, 4),
                "z0_p_nude": round(z0_p_nude, 4),
                "z0_p_color": round(z0_p_color, 4),
                "z0_p_harm": round(z0_p_harm, 4),
                "cdf_p_nude": round(cdf_p_nude, 4),
                "cdf_p_color": round(cdf_p_color, 4),
            }

        # Summary
        s7 = scores.get(7, {})
        s10 = scores.get(10, {})
        tqdm.write(
            f"  [{prompt_idx:03d}] step7: z0={s7.get('z0_p_harm', 'N/A'):.3f} "
            f"cdf_nude={s7.get('cdf_p_nude', 'N/A'):.3f} | "
            f"step10: z0={s10.get('z0_p_harm', 'N/A'):.3f} "
            f"cdf_nude={s10.get('cdf_p_nude', 'N/A'):.3f}"
        )

        all_results.append({
            "prompt_idx": prompt_idx,
            "prompt": prompt[:100],
            "seed": current_seed,
            "scores": scores,
        })

    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)

    print(f"\nSaved {len(all_results)} results to {output_path}")


if __name__ == "__main__":
    main()
