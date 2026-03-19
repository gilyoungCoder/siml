#!/usr/bin/env python
"""
Score analysis: z0-trigger approach for SoftDelete+CG monitoring.

At each diffusion step:
  1. Run UNet to get noise prediction
  2. Predict z0 from zt: z0 = (zt - sqrt(1-αt) * ε) / sqrt(αt)
  3. Feed (z0, t=0) to the time-conditioned classifier
  4. Record softmax probabilities for monitoring analysis

Usage:
    CUDA_VISIBLE_DEVICES=0 python score_analysis_z0_trigger.py \
        --prompt_file <path> --output_path <path> [--end_idx 50]
"""
import json
import random
from argparse import ArgumentParser
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDIMScheduler
from geo_models.classifier.classifier import load_discriminator


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict_z0(zt, noise_pred, alpha_bar):
    """Predict z0 from zt using DDIM formula."""
    return (zt - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--classifier_ckpt", type=str,
                        default="/mnt/home/yhgil99/unlearning/SoftDelete+CG/work_dirs/nudity_4class_safe_combined/checkpoint/step_17100/classifier.pth")
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
    import csv
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
    print(f"Loaded {len(prompts_with_idx)} prompts (indices {args.start_idx} to {end_idx-1})")

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
        args.classifier_ckpt, condition="", eval=True, channel=4, num_classes=4
    ).to(device)
    classifier.eval()
    clf_dtype = next(classifier.parameters()).dtype

    # Get alpha_bar schedule
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    all_results = []

    for prompt_idx, prompt in tqdm(prompts_with_idx, desc="Analyzing"):
        current_seed = args.seed + prompt_idx
        set_seed(current_seed)

        # Encode prompt for CFG (negative + positive concatenated)
        with torch.no_grad():
            text_inputs = tokenizer(prompt, padding="max_length",
                                    max_length=tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt")
            text_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
            uncond_inputs = tokenizer("", padding="max_length",
                                      max_length=tokenizer.model_max_length,
                                      return_tensors="pt")
            uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]
        prompt_embeds = torch.cat([uncond_embeds, text_embeds])  # [2, 77, 768]

        # Initialize noise
        set_seed(current_seed)
        latents = torch.randn(1, 4, 64, 64, device=device)
        latents = latents * scheduler.init_noise_sigma

        scheduler.set_timesteps(args.num_inference_steps, device=device)

        z0_scores = {}
        for i, t in enumerate(scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                noise_pred = unet(latent_model_input, t,
                                  encoder_hidden_states=prompt_embeds).sample

            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + args.cfg_scale * (noise_pred_text - noise_pred_uncond)

            # Predict z0
            alpha_bar = alphas_cumprod[t]
            z0_pred = predict_z0(latents, noise_pred_cfg, alpha_bar)

            # Classify z0 with t=0
            with torch.no_grad():
                z0_input = z0_pred.to(dtype=clf_dtype)
                logits = classifier(z0_input, torch.zeros(1, device=device))
                probs = F.softmax(logits, dim=-1)

            p_benign = probs[0, 0].item()
            p_safe = probs[0, 1].item()
            p_nude = probs[0, 2].item()
            p_color = probs[0, 3].item()
            p_harm = p_nude + p_color

            z0_scores[i] = {
                "step": i,
                "timestep": t.item(),
                "p_benign": round(p_benign, 4),
                "p_safe": round(p_safe, 4),
                "p_nude": round(p_nude, 4),
                "p_color": round(p_color, 4),
                "p_harm": round(p_harm, 4),
            }

            # DDIM step
            scheduler_output = scheduler.step(noise_pred_cfg, t, latents)
            latents = scheduler_output.prev_sample

        # Summary for this prompt
        step13 = z0_scores.get(13, {})
        step20 = z0_scores.get(20, {})
        tqdm.write(f"  [{prompt_idx:03d}] step13: p_harm={step13.get('p_harm', 'N/A')}, "
                   f"step20: p_harm={step20.get('p_harm', 'N/A')}")

        all_results.append({
            "prompt_idx": prompt_idx,
            "prompt": prompt[:100],
            "seed": current_seed,
            "z0_scores": z0_scores,
        })

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)

    print(f"\nSaved {len(all_results)} results to {output_path}")


if __name__ == "__main__":
    main()
