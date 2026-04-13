#!/usr/bin/env python
"""
Offline Concept Subspace Preparation.

Generates exemplar images via DDIM, then computes per-timestep concept
directions by forward-diffusing each exemplar to z_t and measuring
UNet(z_t, concept) - UNet(z_t, "").

Output: a .pt file containing per-timestep target/anchor directions,
global directions, exemplar z0s, and config metadata.

Usage:
    python -m safegen.prepare_concept_subspace \
        --output configs/exemplars/sexual/concept_directions.pt \
        --target_concepts "nudity" "nude person" "naked body" \
        --anchor_concepts "clothed person" "person wearing clothes"
"""

import os
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler


DEFAULT_TARGET_PROMPTS = [
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

DEFAULT_ANCHOR_PROMPTS = [
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_concepts(text_encoder, tokenizer, concepts, device):
    embeds = []
    for c in concepts:
        inp = tokenizer(c, padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
        embeds.append(text_encoder(inp.input_ids.to(device))[0])
    return torch.stack(embeds).mean(0)


@torch.no_grad()
def generate_exemplar_z0(prompt, unet, text_encoder, tokenizer, scheduler,
                         device, seed=42, steps=50, cfg_scale=7.5):
    """Generate a single exemplar z0 via DDIM."""
    set_seed(seed)
    p_inp = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                      truncation=True, return_tensors="pt")
    p_emb = text_encoder(p_inp.input_ids.to(device))[0]
    u_inp = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length,
                      truncation=True, return_tensors="pt")
    u_emb = text_encoder(u_inp.input_ids.to(device))[0]

    set_seed(seed)
    lat = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
    lat = lat * scheduler.init_noise_sigma
    scheduler.set_timesteps(steps, device=device)

    for t in scheduler.timesteps:
        li = scheduler.scale_model_input(lat, t)
        raw = unet(torch.cat([li, li]), t,
                   encoder_hidden_states=torch.cat([u_emb, p_emb])).sample
        en, ep = raw.chunk(2)
        ec = en + cfg_scale * (ep - en)
        lat = scheduler.step(ec, t, lat).prev_sample

    return lat


@torch.no_grad()
def compute_concept_directions(exemplar_z0s, unet, scheduler, concept_embeds,
                               uncond_embeds, device, steps=50, batch_size=8, seed=42, label="concept"):
    """
    Compute per-timestep concept directions from exemplar z0s.

    For each timestep t and exemplar z0_k:
        d_k = UNet(z_t_k, concept) - UNet(z_t_k, "")
    Then average across exemplars.
    """
    scheduler.set_timesteps(steps, device=device)
    K = len(exemplar_z0s)
    directions, global_dirs = {}, {}

    for t in tqdm(scheduler.timesteps, desc=f"Computing {label} directions"):
        t_int = t.item()
        all_d = []

        for bs in range(0, K, batch_size):
            be = min(bs + batch_size, K)
            batch = []
            for k in range(bs, be):
                z0 = exemplar_z0s[k].to(device=device, dtype=torch.float16)
                if z0.dim() == 3:
                    z0 = z0.unsqueeze(0)
                set_seed(seed + k)
                noise = torch.randn_like(z0)
                batch.append(scheduler.add_noise(z0, noise, t))

            z_t = torch.cat(batch, dim=0)
            B = z_t.shape[0]
            eps_c = unet(z_t, t, encoder_hidden_states=concept_embeds.expand(B, -1, -1)).sample
            eps_u = unet(z_t, t, encoder_hidden_states=uncond_embeds.expand(B, -1, -1)).sample
            all_d.append((eps_c - eps_u).float())

        d_mean = torch.cat(all_d, dim=0).mean(dim=0)
        directions[t_int] = d_mean.half().cpu()
        global_dirs[t_int] = d_mean.reshape(-1).half().cpu()

    return directions, global_dirs


def main():
    p = ArgumentParser(description="Offline Concept Subspace Preparation")
    p.add_argument("--output", default="configs/exemplars/concept_directions.pt")
    p.add_argument("--ckpt", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--save_images", action="store_true")
    p.add_argument("--target_concepts", nargs="+", default=["nudity", "nude person", "naked body"])
    p.add_argument("--anchor_concepts", nargs="+", default=["clothed person", "person wearing clothes"])
    p.add_argument("--target_prompt_file", default=None)
    p.add_argument("--anchor_prompt_file", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print(f"Concept Subspace Preparation")
    print(f"  Target: {args.target_concepts}")
    print(f"  Anchor: {args.anchor_concepts}")
    print(f"  Output: {args.output}")
    print(f"{'=' * 60}\n")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, safety_checker=None,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.feature_extractor = None
    unet, te, tok, sched = pipe.unet, pipe.text_encoder, pipe.tokenizer, pipe.scheduler

    with torch.no_grad():
        tgt_emb = encode_concepts(te, tok, args.target_concepts, device)
        anc_emb = encode_concepts(te, tok, args.anchor_concepts, device)
        unc_emb = te(tok("", padding="max_length", max_length=tok.model_max_length,
                         truncation=True, return_tensors="pt").input_ids.to(device))[0]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def read_prompts(path):
        if path and Path(path).exists():
            return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
        return None

    target_prompts = read_prompts(args.target_prompt_file) or DEFAULT_TARGET_PROMPTS
    anchor_prompts = read_prompts(args.anchor_prompt_file) or DEFAULT_ANCHOR_PROMPTS

    # Phase 1: Generate exemplar z0s
    print(f"Generating {len(target_prompts)} target exemplars...")
    target_z0s = []
    for i, pr in enumerate(tqdm(target_prompts, desc="Target")):
        z0 = generate_exemplar_z0(pr, unet, te, tok, sched, device, seed=args.seed + i,
                                  steps=args.steps, cfg_scale=args.cfg_scale)
        target_z0s.append(z0.squeeze(0).cpu())

    print(f"Generating {len(anchor_prompts)} anchor exemplars...")
    anchor_z0s = []
    for i, pr in enumerate(tqdm(anchor_prompts, desc="Anchor")):
        z0 = generate_exemplar_z0(pr, unet, te, tok, sched, device, seed=args.seed + i,
                                  steps=args.steps, cfg_scale=args.cfg_scale)
        anchor_z0s.append(z0.squeeze(0).cpu())

    # Phase 2: Compute directions
    print("\nComputing target directions...")
    tgt_dirs, tgt_global = compute_concept_directions(
        target_z0s, unet, sched, tgt_emb, unc_emb, device,
        steps=args.steps, batch_size=args.batch_size, seed=args.seed, label="target")

    print("Computing anchor directions...")
    anc_dirs, anc_global = compute_concept_directions(
        anchor_z0s, unet, sched, anc_emb, unc_emb, device,
        steps=args.steps, batch_size=args.batch_size, seed=args.seed, label="anchor")

    # Save
    result = {
        "target_directions": tgt_dirs,
        "anchor_directions": anc_dirs,
        "target_global": tgt_global,
        "anchor_global": anc_global,
        "config": {
            "target_concepts": args.target_concepts,
            "anchor_concepts": args.anchor_concepts,
            "num_target_exemplars": len(target_z0s),
            "num_anchor_exemplars": len(anchor_z0s),
            "steps": args.steps, "cfg_scale": args.cfg_scale, "seed": args.seed,
        },
        "exemplar_z0s": {"target": target_z0s, "anchor": anchor_z0s},
    }
    torch.save(result, str(out_path))
    mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {args.output} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
