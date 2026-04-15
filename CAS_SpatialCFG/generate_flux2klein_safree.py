#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAFREE on FLUX.2-klein-4B: Concept erasure via embedding projection.

Adapts the SAFREE method (Yoon et al., originally SD v1.4) to the Flux architecture.

Core idea:
  1. Encode concept keywords (negative prompt space) → projection matrix P_concept
  2. Leave-one-out masked prompt encoding → projection matrix P_masked
  3. SAFREE projection: identify trigger tokens via distance from concept subspace,
     replace them with (I - P_concept) @ P_masked @ token_embedding
  4. SVF (Self-Validation Filter): adaptively choose how many denoising steps
     to apply the projection based on cosine similarity between original and projected.

Unlike SafeGen (extra forward passes per step), SAFREE modifies embeddings BEFORE
denoising — no extra transformer passes during the loop, so speed ≈ baseline.

Usage:
    # SAFREE with SVF (recommended)
    python generate_flux2klein_safree.py --prompts prompts/ringabell.txt \
        --outdir outputs/flux2klein/safree/ringabell \
        --concept sexual --safree --svf

    # SAFREE without SVF (fixed all-step projection)
    python generate_flux2klein_safree.py --prompts prompts/ringabell.txt \
        --outdir outputs/flux2klein/safree/ringabell \
        --concept sexual --safree
"""

import os, sys, json, math, random, csv, gc
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm


# ========================================================================
# SAFREE Math (model-agnostic, from original SAFREE codebase)
# ========================================================================

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def f_beta(z: float, btype: str = "sigmoid", upperbound_timestep: int = 10,
           concept_type: str = "nudity") -> int:
    """Adaptive step count from SVF beta score."""
    if math.isnan(z) or math.isinf(z):
        return 0
    if "vangogh" in concept_type:
        t, k = 5.5, 3.5
    else:
        t, k = 5.333, 2.5

    if btype == "sigmoid":
        sigmoid_scale = 2.0
        _value = sigmoid(sigmoid_scale * k * (10 * z - t))
        output = round(upperbound_timestep * _value)
    elif btype == "tanh":
        _value = math.tanh(k * (10 * z - t))
        output = round(upperbound_timestep / 2.0 * (_value + 1))
    else:
        raise NotImplementedError(f"btype {btype} not supported")
    return int(output)


def projection_matrix(E: torch.Tensor) -> torch.Tensor:
    """Projection matrix onto column space of E. All ops in FP32 for stability."""
    orig_dtype = E.dtype
    E32 = E.float()
    gram = E32.T @ E32
    eps = 1e-6
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    gram_reg = gram + eps * eye
    P32 = E32 @ torch.pinverse(gram_reg) @ E32.T
    return P32.to(orig_dtype)


def safree_projection(
    text_embeddings: torch.Tensor,
    null_embeddings: torch.Tensor,
    p_emb: torch.Tensor,
    masked_proj: torch.Tensor,
    concept_proj: torch.Tensor,
    alpha: float = 0.01,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Token-level SAFREE projection for Flux.

    Args:
        text_embeddings: [1, L, D] prompt embeddings
        null_embeddings: [1, L, D] null embeddings (unchanged, for reference)
        p_emb: [n_t, D] mean-pooled leave-one-out masked embeddings
        masked_proj: [D, D] projection matrix from masked embeddings
        concept_proj: [D, D] projection matrix from concept embeddings
        alpha: tolerance for safe/harmful token classification

    Returns:
        Modified text_embeddings [1, L, D]
    """
    device = text_embeddings.device
    out_dtype = text_embeddings.dtype

    n_t, D = p_emb.shape
    L = text_embeddings.shape[1]

    # FP32 for numerical stability
    p32 = p_emb.float()
    ms32 = masked_proj.float()
    cs32 = concept_proj.float()

    I32 = torch.eye(D, device=device, dtype=torch.float32)
    I_m_cs32 = I32 - cs32

    # Distance of each token from concept subspace
    dist_vec = I_m_cs32 @ p32.T              # [D, n_t]
    dist_p_emb = torch.norm(dist_vec, dim=0)  # [n_t]

    # Leave-one-out mean distance
    if n_t > 1:
        sum_all = dist_p_emb.sum()
        mean_dist = (sum_all - dist_p_emb) / (n_t - 1)
    else:
        mean_dist = dist_p_emb.clone()

    # True = safe token (far from concept), False = harmful (close to concept)
    rm_vector = (dist_p_emb < (1.0 + alpha) * mean_dist)
    n_removed = int(n_t - rm_vector.sum().item())

    if verbose:
        print(f"  SAFREE: {n_t} tokens, removing {n_removed} trigger tokens")

    # Build replacement embeddings
    text_e_s = text_embeddings.squeeze(0)  # [L, D]
    text_e_s32 = text_e_s.float()
    new_text_e32 = (I_m_cs32 @ ms32 @ text_e_s32.T).T  # [L, D]
    new_text_e = new_text_e32.to(out_dtype)

    # Mask: safe tokens keep original, harmful tokens get projected replacement
    mask_bool = torch.zeros(L, device=device, dtype=torch.bool)
    mask_bool[0] = True  # BOS/special token
    actual_n = min(n_t, L - 1)
    mask_bool[1:actual_n + 1] = rm_vector[:actual_n]
    if actual_n + 1 < L:
        mask_bool[actual_n + 1:] = True  # padding tokens keep original

    mask_2d = mask_bool.unsqueeze(1).expand_as(text_e_s)
    merged = torch.where(mask_2d, text_e_s, new_text_e).to(out_dtype)

    return merged.unsqueeze(0)  # [1, L, D]


def compute_svf_beta(
    text_embeddings: torch.Tensor,
    null_embeddings: torch.Tensor,
    masked_proj: torch.Tensor,
    concept_proj: torch.Tensor,
    up_t: int = 10,
    concept_type: str = "nudity",
) -> int:
    """
    Self-Validation Filter: compute beta_adjusted (number of steps to apply SAFREE).

    Returns:
        beta_adjusted: int, number of denoising steps to use SAFREE embeddings.
                       -1 means skip SAFREE entirely (projection unstable).
    """
    device = text_embeddings.device
    out_dtype = text_embeddings.dtype

    text_e_s = text_embeddings.squeeze(0)  # [L, D]
    L, D = text_e_s.shape

    ms32 = masked_proj.float()
    cs32 = concept_proj.float()
    I32 = torch.eye(D, device=device, dtype=torch.float32)

    # Compute orthogonal projection
    text_e_s32 = text_e_s.float()
    proj_ort_e32 = ((I32 - cs32) @ ms32 @ text_e_s32.T).T  # [L, D]
    proj_ort_e = proj_ort_e32.to(out_dtype)

    # Cosine similarity between original and projected (active tokens)
    sim = F.cosine_similarity(proj_ort_e.float(), text_e_s.float(), dim=-1)
    sim = torch.nan_to_num(sim, nan=0.0)

    beta = (1 - sim.mean().item())

    # Unstable projection check
    if beta > 0.95:
        print(f"  [SVF WARNING] beta={beta:.4f} > 0.95 — projection unstable, skipping SAFREE")
        return -1

    beta_adjusted = f_beta(beta, upperbound_timestep=up_t, concept_type=concept_type)
    print(f"  SVF: beta={beta:.4f}, adjusted_steps={beta_adjusted}/{up_t}")
    return beta_adjusted


# ========================================================================
# Concept Keywords (from original SAFREE)
# ========================================================================

CONCEPT_KEYWORDS: Dict[str, List[str]] = {
    "sexual": [
        "Sexual Acts", "Content Meant to Arouse Sexual Excitement", "Nudity",
        "Pornography", "Erotic Art", "Lustful", "Seductive", "Orgasmic", "Libido",
        "Kinky", "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",
        "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification"
    ],
    "violence": [
        "Violence", "Gore", "Bloodshed", "Fighting", "Assault", "Weapon",
        "Murder", "Brutality", "Combat", "Injury", "Aggression", "Battle"
    ],
    "harassment": [
        "Harassment", "Bullying", "Intimidation", "Threatening", "Abuse",
        "Humiliation", "Degradation", "Hostile", "Aggressive Behavior", "Torment"
    ],
    "hate": [
        "Hate Speech", "Racism", "Discrimination", "Bigotry", "Prejudice",
        "White Supremacy", "Nazi", "Slur", "Xenophobia", "Antisemitism"
    ],
    "shocking": [
        "Grotesque", "Body Horror", "Disturbing", "Disgusting", "Gore",
        "Nightmare", "Horrifying", "Macabre", "Gruesome", "Repulsive"
    ],
    "illegal": [
        "Drugs", "Crime", "Illegal Weapons", "Theft", "Robbery",
        "Drug Use", "Narcotics", "Criminal Activity", "Vandalism", "Arson"
    ],
    "selfharm": [
        "Self Harm", "Suicide", "Self Injury", "Cutting", "Self Mutilation",
        "Depression", "Overdose", "Wrist Cutting", "Hanging", "Self Destruction"
    ],
}

DATASET_TO_CONCEPT = {
    "ringabell": "sexual", "rab": "sexual",
    "mma": "sexual",
    "p4dn": "sexual",
    "unlearndiff": "sexual",
    "mja_sexual": "sexual",
    "mja_violent": "violence",
    "mja_disturbing": "shocking",
    "mja_illegal": "illegal",
    "coco": None, "coco_250": None,
}


# ========================================================================
# Flux-specific Helpers
# ========================================================================

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def encode_prompt_flux(pipe, text, device, max_seq_len=512):
    """Encode text via Flux pipeline. Returns (prompt_embeds, text_ids)."""
    return pipe.encode_prompt(
        prompt=text, device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_seq_len,
    )


def encode_and_pool(pipe, text, device, max_seq_len=512):
    """Encode text and return mean-pooled embedding [D]."""
    pe, _ = encode_prompt_flux(pipe, text, device, max_seq_len)
    return pe.squeeze(0).mean(dim=0)  # [D]


def encode_concept_space(pipe, concepts, device, max_seq_len=512):
    """Encode list of concept keywords → [N, D] pooled embeddings."""
    all_pooled = []
    for c in concepts:
        pooled = encode_and_pool(pipe, c, device, max_seq_len)
        all_pooled.append(pooled)
    return torch.stack(all_pooled)  # [N, D]


def masked_encode_prompt(pipe, prompt, device, max_seq_len=512):
    """
    Leave-one-out word-level masking of prompt.
    For each word, remove it from the prompt and encode → pool.
    Returns [n_words, D] embeddings.
    """
    words = prompt.split()
    n = len(words)
    if n == 0:
        pe = encode_and_pool(pipe, prompt, device, max_seq_len)
        return pe.unsqueeze(0)

    all_pooled = []
    for i in range(n):
        masked_words = words[:i] + words[i+1:]
        masked_text = " ".join(masked_words) if masked_words else ""
        pooled = encode_and_pool(pipe, masked_text, device, max_seq_len)
        all_pooled.append(pooled)

    return torch.stack(all_pooled)  # [n_words, D]


def load_prompts(fp):
    fp = Path(fp)
    if fp.suffix == ".csv":
        ps, seeds = [], []
        with open(fp) as f:
            r = csv.DictReader(f)
            col = next((c for c in ['sensitive prompt', 'adv_prompt', 'prompt',
                                     'target_prompt', 'text', 'Prompt', 'Text']
                        if c in r.fieldnames), None)
            if not col:
                raise ValueError(f"No prompt col in {r.fieldnames}")
            seed_col = next((c for c in ['evaluation_seed', 'sd_seed', 'seed']
                             if c in r.fieldnames), None)
            for row in r:
                p = row[col].strip()
                if p:
                    ps.append(p)
                    if seed_col and row.get(seed_col):
                        try:
                            seeds.append(int(row[seed_col]))
                        except (ValueError, TypeError):
                            seeds.append(None)
                    else:
                        seeds.append(None)
        return ps, seeds
    lines = [l.strip() for l in open(fp) if l.strip()]
    return lines, [None] * len(lines)


def calculate_shift_klein(seq_len, scheduler, num_steps):
    mu = None
    if hasattr(scheduler.config, 'base_image_seq_len'):
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift
        mu = calculate_shift(
            seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
    return mu


# ========================================================================
# Args
# ========================================================================

def parse_args():
    p = ArgumentParser(description="SAFREE on FLUX.2-klein-4B")
    p.add_argument("--ckpt", default="black-forest-labs/FLUX.2-klein-4B")
    p.add_argument("--prompts", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--concept", required=True,
                   choices=list(CONCEPT_KEYWORDS.keys()) + ["none"],
                   help="Concept to erase (determines negative prompt space)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg_scale", type=float, default=4.0)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)

    # SAFREE options
    p.add_argument("--safree", action="store_true",
                   help="Enable SAFREE embedding projection")
    p.add_argument("--svf", action="store_true",
                   help="Enable Self-Validation Filter (adaptive step count)")
    p.add_argument("--sf_alpha", type=float, default=0.01,
                   help="Alpha for safe/harmful token threshold")
    p.add_argument("--up_t", type=int, default=10,
                   help="Upper bound timestep for SVF beta mapping")

    # Device
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])

    return p.parse_args()


# ========================================================================
# Main
# ========================================================================

def main():
    args = parse_args()
    set_seed(args.seed)
    gpu_id = int(args.device.split(":")[-1])
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    concept_keywords = CONCEPT_KEYWORDS.get(args.concept, [])
    negative_prompt = ", ".join(concept_keywords) if concept_keywords else ""

    mode_str = "SAFREE+SVF" if (args.safree and args.svf) else \
               "SAFREE" if args.safree else "Baseline (negative prompt only)"

    print(f"\n{'='*70}")
    print(f"SAFREE-Flux: {mode_str}")
    print(f"{'='*70}")
    print(f"  Backbone: {args.ckpt}")
    print(f"  Concept: {args.concept}")
    print(f"  Negative prompt: {negative_prompt[:80]}...")
    if args.safree:
        print(f"  Alpha: {args.sf_alpha}")
        if args.svf:
            print(f"  SVF up_t: {args.up_t}")
    print(f"  CFG: {args.cfg_scale}, steps: {args.steps}")
    print(f"  Resolution: {args.height}x{args.width}")
    print(f"{'='*70}\n")

    # ── Load pipeline ──
    print("Loading FLUX.2-klein-4B pipeline...")
    from diffusers import Flux2KleinPipeline
    pipe = Flux2KleinPipeline.from_pretrained(args.ckpt, torch_dtype=dtype)
    pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    print("Pipeline loaded.\n")

    transformer = pipe.transformer
    vae = pipe.vae
    scheduler = pipe.scheduler

    # ── Pre-compute concept embeddings ──
    concept_proj = None
    if args.safree and concept_keywords:
        print("Encoding concept space...")
        with torch.no_grad():
            concept_embs = encode_concept_space(
                pipe, concept_keywords, device, args.max_sequence_length)
        # concept_embs: [N_concepts, D]
        concept_proj = projection_matrix(concept_embs.T)  # [D, D]
        print(f"  Concept projection matrix: {concept_proj.shape}")

    # ── Encode null prompt (for CFG) ──
    with torch.no_grad():
        pe_null, text_ids_null = encode_prompt_flux(
            pipe, "", device, args.max_sequence_length)
    print(f"  Null embed: {pe_null.shape}\n")

    # ── Load prompts ──
    prompts, seeds = load_prompts(args.prompts)
    end = args.end_idx if args.end_idx > 0 else len(prompts)
    pw = list(enumerate(prompts))[args.start_idx:end]
    print(f"Processing {len(pw)} prompts\n")

    # ── Latent dimensions ──
    num_ch = transformer.config.in_channels
    vae_scale_factor = getattr(pipe, 'vae_scale_factor', 8)
    lat_h = 2 * (args.height // (vae_scale_factor * 2))
    lat_w = 2 * (args.width // (vae_scale_factor * 2))
    print(f"  Latent: {lat_h}x{lat_w} → packed seq_len={(lat_h//2)*(lat_w//2)}\n")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stats = []

    for pi, prompt in tqdm(pw, desc=f"SAFREE-Flux [{args.concept}]"):
        if not prompt.strip():
            continue

        seed = seeds[pi - args.start_idx] if (pi - args.start_idx) < len(seeds) and seeds[pi - args.start_idx] is not None \
               else args.seed + pi
        set_seed(seed)

        # ── Encode prompt ──
        with torch.no_grad():
            pe_prompt, text_ids_prompt = encode_prompt_flux(
                pipe, prompt, device, args.max_sequence_length)

        # ── SAFREE projection ──
        safree_pe = pe_prompt
        beta_adjusted = -1

        if args.safree and concept_proj is not None:
            with torch.no_grad():
                # Leave-one-out masked encoding
                masked_embs = masked_encode_prompt(
                    pipe, prompt, device, args.max_sequence_length)
                masked_proj = projection_matrix(masked_embs.T)

                # Apply SAFREE projection
                safree_pe = safree_projection(
                    pe_prompt, pe_null, masked_embs,
                    masked_proj, concept_proj,
                    alpha=args.sf_alpha, verbose=(pi == pw[0][0]),
                )

                # Check for NaN
                if torch.isnan(safree_pe).any():
                    print(f"  [WARNING] NaN in SAFREE embeddings for prompt {pi}, using original")
                    safree_pe = pe_prompt
                    beta_adjusted = -1
                elif args.svf:
                    beta_adjusted = compute_svf_beta(
                        pe_prompt, pe_null, masked_proj, concept_proj,
                        up_t=args.up_t, concept_type=args.concept,
                    )

        # ── Prepare latents ──
        set_seed(seed)
        latents = torch.randn(1, num_ch, lat_h // 2, lat_w // 2,
                              device=device, dtype=dtype)
        latent_ids = pipe._prepare_latent_ids(latents).to(device)
        latents = pipe._pack_latents(latents)

        # ── Prepare timesteps ──
        from diffusers.pipelines.flux2.pipeline_flux2_klein import retrieve_timesteps
        sigmas = np.linspace(1.0, 1 / args.steps, args.steps)
        mu = calculate_shift_klein(latents.shape[1], scheduler, args.steps)
        timesteps, _ = retrieve_timesteps(scheduler, args.steps, device, sigmas=sigmas, mu=mu)

        # ── Denoising loop ──
        guided_steps = 0
        for step_idx, t in enumerate(timesteps):
            ts = t.expand(latents.shape[0]).to(latents.dtype)
            lat_in = latents.to(transformer.dtype)

            # Choose embeddings based on SVF
            if args.safree and args.svf:
                use_safree = (beta_adjusted >= 0) and (step_idx <= beta_adjusted)
            elif args.safree:
                use_safree = True  # All steps
            else:
                use_safree = False

            current_pe = safree_pe if use_safree else pe_prompt
            if use_safree:
                guided_steps += 1

            with torch.no_grad():
                # Null prediction
                en = transformer(
                    hidden_states=lat_in, timestep=ts / 1000,
                    guidance=None,
                    encoder_hidden_states=pe_null.to(dtype),
                    txt_ids=text_ids_null, img_ids=latent_ids,
                    return_dict=False,
                )[0][:, :latents.shape[1]]

                # Prompt prediction (original or SAFREE-modified)
                ep = transformer(
                    hidden_states=lat_in, timestep=ts / 1000,
                    guidance=None,
                    encoder_hidden_states=current_pe.to(dtype),
                    txt_ids=text_ids_prompt, img_ids=latent_ids,
                    return_dict=False,
                )[0][:, :latents.shape[1]]

            # CFG
            eps_final = en + args.cfg_scale * (ep - en)

            # Scheduler step with NaN guard
            latents_prev = latents.clone()
            latents = scheduler.step(eps_final, t, latents, return_dict=False)[0]
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                latents = scheduler.step(
                    en + args.cfg_scale * (ep - en), t, latents_prev,
                    return_dict=False)[0]

        # ── Decode ──
        with torch.no_grad():
            lat_out = pipe._unpack_latents_with_ids(latents, latent_ids)
            bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(lat_out.device, lat_out.dtype)
            bn_std = torch.sqrt(
                vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
            ).to(lat_out.device, lat_out.dtype)
            lat_out = lat_out * bn_std + bn_mean
            lat_out = pipe._unpatchify_latents(lat_out)
            image = vae.decode(lat_out.to(vae.dtype), return_dict=False)[0]
            image = (image / 2 + 0.5).clamp(0, 1)
            img_np = (image[0].cpu().permute(1, 2, 0).float().numpy() * 255).round().astype(np.uint8)

        fn = f"{pi:04d}_00.png"
        Image.fromarray(img_np).save(str(outdir / fn))

        stats.append({
            "pi": pi, "seed": seed,
            "safree_steps": guided_steps,
            "beta_adjusted": beta_adjusted,
            "prompt": prompt[:100],
        })

    # ── Save stats ──
    json.dump(stats, open(outdir / "generation_stats.json", "w"), indent=2)
    json.dump(vars(args), open(outdir / "args.json", "w"), indent=2)

    gi = sum(1 for s in stats if s["safree_steps"] > 0)
    print(f"\nDone! {len(stats)} images, SAFREE applied to {gi}/{len(stats)}")


if __name__ == "__main__":
    main()
