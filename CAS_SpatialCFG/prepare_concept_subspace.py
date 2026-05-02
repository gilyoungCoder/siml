#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Offline Concept Subspace Preparation for Exemplar-Based Guidance (v6)

Generates diverse nudity and clothed exemplar z0 latents, then computes
per-timestep concept directions by forward-diffusing each exemplar to z_t
and measuring UNet(z_t, concept) - UNet(z_t, "").

Output: a .pt file containing per-timestep target/anchor directions,
global (flattened) directions, exemplar z0s, and config metadata.

Usage:
    CUDA_VISIBLE_DEVICES=0 python prepare_concept_subspace.py
    CUDA_VISIBLE_DEVICES=0 python prepare_concept_subspace.py --use_glass --save_images
"""

import os
import sys
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# =========================================================================
# Exemplar prompts
# =========================================================================
NUDITY_PROMPTS = [
    "adult nude standing front view, plain white backdrop, centered",
    "adult nude standing back view, plain white backdrop, centered",
    "adult nude side profile, plain white backdrop, centered",
    "adult nude sitting on stool, plain white backdrop, centered",
    "adult nude reclining pose, plain white backdrop, centered",
    "adult nude crouching pose, plain white backdrop, centered",
    "adult nude with arms raised above head, plain white backdrop, centered",
    "adult nude kneeling pose, plain white backdrop, centered",
    "male nude standing front view, plain white backdrop, centered",
    "male nude standing back view, plain white backdrop, centered",
    "male nude side profile, plain white backdrop, centered",
    "male nude sitting on stool, plain white backdrop, centered",
    "fine-art nude adult in contrapposto pose, plain white backdrop, centered",
    "classical sculpture style nude adult, plain white backdrop, centered",
    "adult nude figure study, dramatic lighting, plain dark backdrop, centered",
    "adult nude with body turned three-quarter view, plain white backdrop, centered",
]

CLOTHED_PROMPTS = [
    "adult in casual t-shirt and jeans standing front view, plain white backdrop, centered",
    "adult in casual t-shirt and jeans standing back view, plain white backdrop, centered",
    "adult in casual t-shirt and jeans side profile, plain white backdrop, centered",
    "adult in casual t-shirt and jeans sitting on stool, plain white backdrop, centered",
    "adult in long dress reclining pose, plain white backdrop, centered",
    "adult in sweater and pants crouching pose, plain white backdrop, centered",
    "adult in fitted shirt with arms raised above head, plain white backdrop, centered",
    "adult in blouse and skirt kneeling pose, plain white backdrop, centered",
    "male in casual t-shirt and jeans standing front view, plain white backdrop, centered",
    "male in casual t-shirt and jeans standing back view, plain white backdrop, centered",
    "male in casual t-shirt and jeans side profile, plain white backdrop, centered",
    "male in casual t-shirt and jeans sitting on stool, plain white backdrop, centered",
    "adult in elegant dress in contrapposto pose, plain white backdrop, centered",
    "classical sculpture style adult in draped toga, plain white backdrop, centered",
    "adult in dark clothing, dramatic lighting, plain dark backdrop, centered",
    "adult in fitted clothing turned three-quarter view, plain white backdrop, centered",
]


def read_prompt_lines(path: str | None) -> list[str]:
    if not path:
        return []
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# =========================================================================
# Utilities
# =========================================================================
def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_concepts(text_encoder, tokenizer, concepts, device):
    """Encode a list of concept strings and average their embeddings."""
    all_embeds = []
    for concept in concepts:
        inputs = tokenizer(
            concept,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        embeds = text_encoder(inputs.input_ids.to(device))[0]
        all_embeds.append(embeds)
    return torch.stack(all_embeds).mean(dim=0)


def decode_latent(vae, z0, device):
    """Decode a latent to a PIL image."""
    z0 = z0.to(device=device, dtype=torch.float16)
    with torch.no_grad():
        img = vae.decode(z0 / vae.config.scaling_factor).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    img = img[0].permute(1, 2, 0).cpu().numpy()
    img = (img * 255).round().astype(np.uint8)
    return Image.fromarray(img)


# =========================================================================
# Phase 1: Generate exemplar z0 latents via DDIM (or GLASS)
# =========================================================================
@torch.no_grad()
def generate_exemplar_z0_ddim(
    prompt: str,
    unet,
    vae,
    text_encoder,
    tokenizer,
    scheduler,
    device,
    seed: int = 42,
    steps: int = 50,
    cfg_scale: float = 7.5,
):
    """Generate a single exemplar z0 via standard DDIM."""
    set_seed(seed)

    # Encode prompt
    prompt_inputs = tokenizer(
        prompt, padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    prompt_embeds = text_encoder(prompt_inputs.input_ids.to(device))[0]

    # Unconditional
    uncond_inputs = tokenizer(
        "", padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]

    # Init latents
    set_seed(seed)
    latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(steps, device=device)

    # DDIM denoising loop
    for t in scheduler.timesteps:
        lat_in = scheduler.scale_model_input(latents, t)
        # Batched: [uncond, cond]
        lat_batch = torch.cat([lat_in, lat_in])
        embed_batch = torch.cat([uncond_embeds, prompt_embeds])
        raw = unet(lat_batch, t, encoder_hidden_states=embed_batch).sample
        eps_null, eps_prompt = raw.chunk(2)
        # CFG
        eps_cfg = eps_null + cfg_scale * (eps_prompt - eps_null)
        # DDIM step
        latents = scheduler.step(eps_cfg, t, latents).prev_sample

    return latents  # [1, 4, 64, 64] in fp16


@torch.no_grad()
def generate_exemplar_z0_glass(
    prompt: str,
    unet,
    vae,
    text_encoder,
    tokenizer,
    scheduler,
    device,
    seed: int = 42,
    steps: int = 50,
    cfg_scale: float = 7.5,
    glass_rho: float = 0.4,
    glass_inner_steps: int = 4,
):
    """Generate a single exemplar z0 via GLASS transitions."""
    from glass_sampler import GlassSampler

    set_seed(seed)

    # Encode prompt
    prompt_inputs = tokenizer(
        prompt, padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    prompt_embeds = text_encoder(prompt_inputs.input_ids.to(device))[0]

    # Unconditional
    uncond_inputs = tokenizer(
        "", padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]

    # For GLASS with CFG, we use a wrapper that does CFG inside the denoiser.
    # We create a simple CFG-aware UNet wrapper.
    class CFGUNetWrapper:
        """Wraps UNet to do CFG internally for GLASS sampler."""
        def __init__(self, unet, uncond_embeds, prompt_embeds, cfg_scale):
            self.unet = unet
            self.uncond_embeds = uncond_embeds
            self.prompt_embeds = prompt_embeds
            self.cfg_scale = cfg_scale

        def __call__(self, x, t, encoder_hidden_states=None):
            """GLASS sampler calls unet(x, t, encoder_hidden_states=...).
            We intercept and do CFG."""
            # Expand for batched call
            lat_batch = torch.cat([x.half(), x.half()])
            embed_batch = torch.cat([self.uncond_embeds, self.prompt_embeds])
            t_tensor = t if isinstance(t, torch.Tensor) else torch.tensor([t], device=x.device, dtype=torch.long)
            if t_tensor.dim() == 0:
                t_tensor = t_tensor.unsqueeze(0)
            t_batch = t_tensor.expand(2)
            raw = self.unet(lat_batch, t_batch, encoder_hidden_states=embed_batch).sample
            eps_null, eps_prompt = raw.chunk(2)
            eps_cfg = eps_null + self.cfg_scale * (eps_prompt - eps_null)

            class Result:
                def __init__(self, sample):
                    self.sample = sample
            return Result(eps_cfg)

        def parameters(self):
            return self.unet.parameters()

    cfg_unet = CFGUNetWrapper(unet, uncond_embeds, prompt_embeds, cfg_scale)
    sampler = GlassSampler(
        scheduler, cfg_unet,
        rho=glass_rho, inner_steps=glass_inner_steps, device=device,
    )

    # Init noise
    set_seed(seed)
    z_T = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
    z_T = z_T * scheduler.init_noise_sigma
    scheduler.set_timesteps(steps, device=device)

    # GLASS full trajectory — pass prompt_embeds as conditioning
    # (the CFG wrapper will ignore it and use its own)
    z0 = sampler.sample_full_trajectory(
        z_T, scheduler.timesteps, prompt_embeds, rho=glass_rho,
    )

    return z0  # [1, 4, 64, 64]


# =========================================================================
# Phase 2: Compute per-timestep concept directions
# =========================================================================
@torch.no_grad()
def compute_concept_directions(
    exemplar_z0s: list,
    unet,
    scheduler,
    concept_embeds: torch.Tensor,
    uncond_embeds: torch.Tensor,
    device,
    steps: int = 50,
    batch_size: int = 8,
    seed: int = 42,
    label: str = "concept",
):
    """
    Compute per-timestep concept directions from exemplar z0s.

    For each timestep t:
      - Forward-diffuse each z0_k to z_t_k using scheduler.add_noise
      - Compute eps_concept = UNet(z_t_k, t, concept_embeds)
      - Compute eps_null = UNet(z_t_k, t, uncond_embeds)
      - d_k = eps_concept - eps_null
      - Average across K exemplars

    Returns:
        directions: dict {t_int: tensor[4, 64, 64]} per-timestep mean directions
        global_dirs: dict {t_int: tensor[16384]} flattened directions
    """
    scheduler.set_timesteps(steps, device=device)
    timesteps = scheduler.timesteps
    K = len(exemplar_z0s)

    directions = {}
    global_dirs = {}

    for t in tqdm(timesteps, desc=f"Computing {label} directions"):
        t_int = t.item()

        # Forward-diffuse all exemplars to this timestep
        all_d = []
        # Process in batches
        for batch_start in range(0, K, batch_size):
            batch_end = min(batch_start + batch_size, K)
            batch_z0s = []

            for k in range(batch_start, batch_end):
                z0_k = exemplar_z0s[k].to(device=device, dtype=torch.float16)
                if z0_k.dim() == 3:
                    z0_k = z0_k.unsqueeze(0)

                # Generate noise with deterministic seed per exemplar
                set_seed(seed + k)
                noise_k = torch.randn_like(z0_k)

                # Forward diffuse: z_t = alpha_t * z0 + sigma_t * noise
                z_t_k = scheduler.add_noise(z0_k, noise_k, t)
                batch_z0s.append(z_t_k)

            # Stack into batch
            z_t_batch = torch.cat(batch_z0s, dim=0)  # [B, 4, 64, 64]
            B = z_t_batch.shape[0]

            # Expand embeddings for batch
            concept_batch = concept_embeds.expand(B, -1, -1)
            uncond_batch = uncond_embeds.expand(B, -1, -1)

            # UNet forward: concept conditioning
            eps_concept = unet(
                z_t_batch, t,
                encoder_hidden_states=concept_batch,
            ).sample

            # UNet forward: unconditional
            eps_null = unet(
                z_t_batch, t,
                encoder_hidden_states=uncond_batch,
            ).sample

            # d = eps_concept - eps_null
            d_batch = (eps_concept - eps_null).float()  # [B, 4, 64, 64]
            all_d.append(d_batch)

        # Concatenate all batches and average
        all_d = torch.cat(all_d, dim=0)  # [K, 4, 64, 64]
        d_mean = all_d.mean(dim=0)  # [4, 64, 64]

        directions[t_int] = d_mean.half().cpu()
        global_dirs[t_int] = d_mean.reshape(-1).half().cpu()  # [16384]

    return directions, global_dirs


# =========================================================================
# CLI
# =========================================================================
def parse_args():
    p = ArgumentParser(description="Offline Concept Subspace Preparation")
    p.add_argument("--output", type=str,
                   default="exemplars/sd14/concept_directions.pt",
                   help="Output .pt file path")
    p.add_argument("--steps", type=int, default=50,
                   help="DDIM steps for exemplar generation and direction computation")
    p.add_argument("--cfg_scale", type=float, default=7.5,
                   help="CFG scale for exemplar generation")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for UNet calls during direction computation")
    p.add_argument("--save_images", action="store_true",
                   help="Decode and save exemplar images for inspection")
    p.add_argument("--use_glass", action="store_true",
                   help="Use GLASS transitions instead of DDIM for exemplar generation")
    p.add_argument("--glass_rho", type=float, default=0.4,
                   help="GLASS correlation parameter")
    p.add_argument("--glass_inner_steps", type=int, default=4,
                   help="GLASS inner ODE steps per transition")
    p.add_argument("--target_concepts", type=str, nargs="+",
                   default=["nudity", "nude person", "naked body"],
                   help="Target concepts for direction computation")
    p.add_argument("--anchor_concepts", type=str, nargs="+",
                   default=["clothed person", "person wearing clothes"],
                   help="Anchor concepts for direction computation")
    p.add_argument("--target_prompt_file", type=str, default=None,
                   help="Optional text file with one target exemplar prompt per line")
    p.add_argument("--anchor_prompt_file", type=str, default=None,
                   help="Optional text file with one anchor exemplar prompt per line")
    p.add_argument("--target_image_prefix", type=str, default="nudity_",
                   help="Filename prefix used when saving target exemplar images")
    p.add_argument("--anchor_image_prefix", type=str, default="clothed_",
                   help="Filename prefix used when saving anchor exemplar images")
    return p.parse_args()


# =========================================================================
# Main
# =========================================================================
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from diffusers import StableDiffusionPipeline, DDIMScheduler

    print(f"\n{'='*70}")
    print(f"Concept Subspace Preparation")
    print(f"{'='*70}")
    print(f"  Mode:    {'GLASS' if args.use_glass else 'DDIM'}")
    print(f"  Steps:   {args.steps}, CFG: {args.cfg_scale}, Seed: {args.seed}")
    print(f"  Batch:   {args.batch_size}")
    print(f"  Target:  {args.target_concepts}")
    print(f"  Anchor:  {args.anchor_concepts}")
    print(f"  Output:  {args.output}")
    if args.use_glass:
        print(f"  GLASS:   rho={args.glass_rho}, inner_steps={args.glass_inner_steps}")
    print(f"{'='*70}\n")

    # ---- Load pipeline ----
    print("Loading model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.feature_extractor = None

    unet = pipe.unet
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    scheduler = pipe.scheduler

    # ---- Pre-encode concept embeddings for direction computation ----
    print("Encoding concept embeddings...")
    with torch.no_grad():
        target_embeds = encode_concepts(
            text_encoder, tokenizer, args.target_concepts, device
        )
        anchor_embeds = encode_concepts(
            text_encoder, tokenizer, args.anchor_concepts, device
        )
        uncond_inputs = tokenizer(
            "", padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]

    # ---- Output directory ----
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        img_dir = out_path.parent / "exemplar_images"
        img_dir.mkdir(parents=True, exist_ok=True)

    target_prompts = read_prompt_lines(args.target_prompt_file) or NUDITY_PROMPTS
    anchor_prompts = read_prompt_lines(args.anchor_prompt_file) or CLOTHED_PROMPTS

    # ==================================================================
    # Phase 1: Generate exemplar z0 latents
    # ==================================================================
    print(f"\n--- Phase 1: Generating exemplar z0 latents ---")

    generate_fn = generate_exemplar_z0_glass if args.use_glass else generate_exemplar_z0_ddim

    # Extra kwargs for GLASS
    glass_kwargs = {}
    if args.use_glass:
        glass_kwargs = {
            "glass_rho": args.glass_rho,
            "glass_inner_steps": args.glass_inner_steps,
        }

    # Generate nudity exemplars
    target_z0s = []
    print(f"\nGenerating {len(target_prompts)} target exemplar z0s...")
    for i, prompt in enumerate(tqdm(target_prompts, desc="Target exemplars")):
        seed_i = args.seed + i
        z0 = generate_fn(
            prompt=prompt,
            unet=unet, vae=vae,
            text_encoder=text_encoder, tokenizer=tokenizer,
            scheduler=scheduler, device=device,
            seed=seed_i, steps=args.steps, cfg_scale=args.cfg_scale,
            **glass_kwargs,
        )
        target_z0s.append(z0.squeeze(0).cpu())  # [4, 64, 64]

        if args.save_images:
            img = decode_latent(vae, z0, device)
            img.save(str(img_dir / f"{args.target_image_prefix}{i:02d}.png"))

    # Generate clothed exemplars
    anchor_z0s = []
    print(f"\nGenerating {len(anchor_prompts)} anchor exemplar z0s...")
    for i, prompt in enumerate(tqdm(anchor_prompts, desc="Anchor exemplars")):
        seed_i = args.seed + i
        z0 = generate_fn(
            prompt=prompt,
            unet=unet, vae=vae,
            text_encoder=text_encoder, tokenizer=tokenizer,
            scheduler=scheduler, device=device,
            seed=seed_i, steps=args.steps, cfg_scale=args.cfg_scale,
            **glass_kwargs,
        )
        anchor_z0s.append(z0.squeeze(0).cpu())  # [4, 64, 64]

        if args.save_images:
            img = decode_latent(vae, z0, device)
            img.save(str(img_dir / f"{args.anchor_image_prefix}{i:02d}.png"))

    print(f"\nGenerated {len(target_z0s)} target + {len(anchor_z0s)} anchor exemplar z0s")

    # ==================================================================
    # Phase 2: Compute per-timestep concept directions
    # ==================================================================
    print(f"\n--- Phase 2: Computing per-timestep concept directions ---")

    # Target (nudity) directions
    print(f"\nComputing target directions from {len(target_z0s)} exemplars...")
    target_directions, target_global = compute_concept_directions(
        exemplar_z0s=target_z0s,
        unet=unet, scheduler=scheduler,
        concept_embeds=target_embeds,
        uncond_embeds=uncond_embeds,
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        label="target",
    )

    # Anchor (clothed) directions
    print(f"\nComputing anchor directions from {len(anchor_z0s)} exemplars...")
    anchor_directions, anchor_global = compute_concept_directions(
        exemplar_z0s=anchor_z0s,
        unet=unet, scheduler=scheduler,
        concept_embeds=anchor_embeds,
        uncond_embeds=uncond_embeds,
        device=device,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        label="anchor",
    )

    # ==================================================================
    # Save
    # ==================================================================
    print(f"\n--- Saving results to {args.output} ---")

    result = {
        "target_directions": target_directions,
        "anchor_directions": anchor_directions,
        "target_global": target_global,
        "anchor_global": anchor_global,
        "config": {
            "num_target_exemplars": len(target_z0s),
            "num_anchor_exemplars": len(anchor_z0s),
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "target_concepts": args.target_concepts,
            "anchor_concepts": args.anchor_concepts,
            "use_glass": args.use_glass,
            "glass_rho": args.glass_rho if args.use_glass else None,
            "glass_inner_steps": args.glass_inner_steps if args.use_glass else None,
            "seed": args.seed,
            "target_prompts": target_prompts,
            "anchor_prompts": anchor_prompts,
            "target_image_prefix": args.target_image_prefix,
            "anchor_image_prefix": args.anchor_image_prefix,
        },
        "exemplar_z0s": {
            "target": target_z0s,   # list of [4, 64, 64] tensors
            "anchor": anchor_z0s,   # list of [4, 64, 64] tensors
        },
    }

    torch.save(result, str(out_path))

    # Print summary
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {args.output} ({file_size_mb:.1f} MB)")
    print(f"  target_directions: {len(target_directions)} timesteps, each [4, 64, 64] fp16")
    print(f"  anchor_directions: {len(anchor_directions)} timesteps, each [4, 64, 64] fp16")
    print(f"  target_global:     {len(target_global)} timesteps, each [16384] fp16")
    print(f"  anchor_global:     {len(anchor_global)} timesteps, each [16384] fp16")
    print(f"  exemplar_z0s:      {len(target_z0s)} target + {len(anchor_z0s)} anchor")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
