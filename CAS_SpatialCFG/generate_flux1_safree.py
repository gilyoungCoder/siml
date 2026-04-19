#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAFREE on FLUX.1-dev (12B DiT, embedded guidance).

Implements the three SAFREE components:
  1. Token-level prompt filtering: project CLIP text embeddings onto unsafe subspace,
     detect and reweight harmful tokens before denoising.
  2. Self-validation re-attention (latent-space re-attention check): periodically
     compute cosine distance between current latent and projected latent; if drifting
     toward unsafe region, suppress via projection of the latent residual.
  3. Latent-space safety filtering: project the latent residual onto a safety subspace
     and subtract the unsafe component.

FLUX.1-dev specifics:
  - Embedded guidance: transformer takes a `guidance` tensor; single-pass prediction.
    No separate neg pass needed. ep already includes guidance.
  - encode_prompt returns (prompt_embeds, pooled_prompt_embeds, text_ids) — 3 values.
  - Latent packing: [B,C,H,W] -> [B, (H/2)*(W/2), C*4] via pipe._pack_latents().
  - VAE decode: pipe._unpack_latents(latents, H, W, vae_scale_factor) then vae.decode().
  - Token filter uses pipe.tokenizer + pipe.text_encoder (CLIP-L) for token-level subspace.

Usage:
    python generate_flux1_safree.py \\
      --prompts prompts/mja_sexual.txt \\
      --outdir outputs/flux1_safree/mja_sexual \\
      --target_concepts nudity "nude person" "naked body" \\
      --steps 28 --guidance_scale 3.5 \\
      --height 1024 --width 1024 \\
      --safree_alpha 0.01 \\
      --safree_re_attention --safree_token_filter --safree_latent_filter \\
      --device cuda:0 --dtype bfloat16 \\
      --start_idx 0 --end_idx -1 --nsamples 1 --seed 42
"""

import os, sys, json, math, random, csv
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm


# ============================================================
# SAFREE Math (model-agnostic)
# ============================================================

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


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
        _value = sigmoid(2.0 * k * (10 * z - t))
        output = round(upperbound_timestep * _value)
    elif btype == "tanh":
        _value = math.tanh(k * (10 * z - t))
        output = round(upperbound_timestep / 2.0 * (_value + 1))
    else:
        raise NotImplementedError(f"btype {btype} not supported")
    return int(output)


def projection_matrix(E: torch.Tensor) -> torch.Tensor:
    """Projection matrix onto column space of E. Computed in FP32 for stability."""
    orig_dtype = E.dtype
    E32 = E.float()
    gram = E32.T @ E32
    eps = 1e-6
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=torch.float32)
    gram_reg = gram + eps * eye
    P32 = E32 @ torch.linalg.pinv(gram_reg) @ E32.T
    return P32.to(orig_dtype)


def safree_token_projection(
    text_embeddings: torch.Tensor,
    p_emb: torch.Tensor,
    masked_proj: torch.Tensor,
    concept_proj: torch.Tensor,
    alpha: float = 0.01,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Token-level SAFREE projection.

    Args:
        text_embeddings: [1, L, D] T5 prompt embeddings
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
    dist_vec = I_m_cs32 @ p32.T        # [D, n_t]
    dist_p_emb = torch.norm(dist_vec, dim=0)  # [n_t]

    # Leave-one-out mean distance
    if n_t > 1:
        sum_all = dist_p_emb.sum()
        mean_dist = (sum_all - dist_p_emb) / (n_t - 1)
    else:
        mean_dist = dist_p_emb.clone()

    # True = safe token (far from concept)
    rm_vector = (dist_p_emb < (1.0 + alpha) * mean_dist)
    n_removed = int(n_t - rm_vector.sum().item())

    if verbose:
        print(f"    SAFREE token filter: {n_t} tokens, {n_removed} trigger tokens replaced")

    # Build replacement embeddings
    text_e_s = text_embeddings.squeeze(0)   # [L, D]
    text_e_s32 = text_e_s.float()
    new_text_e32 = (I_m_cs32 @ ms32 @ text_e_s32.T).T  # [L, D]
    new_text_e = new_text_e32.to(out_dtype)

    # Mask: safe tokens keep original, harmful tokens get projected replacement
    mask_bool = torch.zeros(L, device=device, dtype=torch.bool)
    mask_bool[0] = True  # BOS/special token always kept
    actual_n = min(n_t, L - 1)
    mask_bool[1:actual_n + 1] = rm_vector[:actual_n]
    if actual_n + 1 < L:
        mask_bool[actual_n + 1:] = True  # padding tokens kept

    mask_2d = mask_bool.unsqueeze(1).expand_as(text_e_s)
    merged = torch.where(mask_2d, text_e_s, new_text_e).to(out_dtype)
    return merged.unsqueeze(0)  # [1, L, D]


def compute_svf_beta(
    text_embeddings: torch.Tensor,
    masked_proj: torch.Tensor,
    concept_proj: torch.Tensor,
    up_t: int = 10,
    concept_type: str = "nudity",
) -> int:
    """
    Self-Validation Filter: compute beta_adjusted.

    Returns:
        beta_adjusted: int (number of steps to apply SAFREE); -1 = skip SAFREE.
    """
    device = text_embeddings.device
    out_dtype = text_embeddings.dtype

    text_e_s = text_embeddings.squeeze(0)  # [L, D]
    D = text_e_s.shape[1]

    ms32 = masked_proj.float()
    cs32 = concept_proj.float()
    I32 = torch.eye(D, device=device, dtype=torch.float32)

    text_e_s32 = text_e_s.float()
    proj_ort_e32 = ((I32 - cs32) @ ms32 @ text_e_s32.T).T  # [L, D]
    proj_ort_e = proj_ort_e32.to(out_dtype)

    sim = F.cosine_similarity(proj_ort_e.float(), text_e_s.float(), dim=-1)
    sim = torch.nan_to_num(sim, nan=0.0)
    beta = float((1 - sim.mean()).item())

    if beta > 0.95:
        print(f"    [SVF WARNING] beta={beta:.4f} > 0.95 — projection unstable, skipping SAFREE")
        return -1

    beta_adjusted = f_beta(beta, upperbound_timestep=up_t, concept_type=concept_type)
    print(f"    SVF: beta={beta:.4f}, safree_steps={beta_adjusted}/{up_t}")
    return beta_adjusted


# ============================================================
# SAFREE CLIP-space token filter helpers
# ============================================================

def clip_encode_texts(pipe, texts: List[str], device: torch.device) -> torch.Tensor:
    """
    Encode list of texts via CLIP-L (pipe.tokenizer + pipe.text_encoder).
    Returns [N, D_clip] pooled embeddings.
    """
    tok = pipe.tokenizer(
        texts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        clip_out = pipe.text_encoder(
            tok.input_ids.to(device),
            attention_mask=tok.attention_mask.to(device)
            if hasattr(tok, "attention_mask") else None,
            output_hidden_states=False,
        )
    pooled = getattr(clip_out, "pooler_output", None)
    if pooled is None:
        pooled = clip_out[1] if len(clip_out) > 1 else clip_out[0].mean(dim=1)
    return pooled  # [N, D_clip]


def clip_encode_masked_prompt(pipe, prompt: str, device: torch.device) -> torch.Tensor:
    """
    Leave-one-word-out masking of prompt via CLIP-L.
    Returns [n_words, D_clip] pooled embeddings.
    """
    words = prompt.split()
    n = len(words)
    if n == 0:
        return clip_encode_texts(pipe, [prompt], device)
    masked_texts = []
    for i in range(n):
        masked_words = words[:i] + words[i + 1:]
        masked_texts.append(" ".join(masked_words) if masked_words else "")
    return clip_encode_texts(pipe, masked_texts, device)  # [n_words, D_clip]


# ============================================================
# T5-space helpers for embedded-guidance token filter
# ============================================================

def encode_prompt_flux1(pipe, text: str, device, max_seq_len: int = 512):
    """Encode text for FLUX.1-dev. Returns (prompt_embeds, pooled_prompt_embeds, text_ids)."""
    return pipe.encode_prompt(
        prompt=text,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_seq_len,
    )


def t5_encode_masked_prompt(pipe, prompt: str, device, max_seq_len: int = 512) -> torch.Tensor:
    """
    Leave-one-word-out masking of prompt via T5 (FLUX text encoder 2).
    Returns [n_words, D_t5] mean-pooled embeddings.
    """
    words = prompt.split()
    n = len(words)
    if n == 0:
        pe, _, _ = encode_prompt_flux1(pipe, prompt, device, max_seq_len)
        return pe.squeeze(0).mean(dim=0, keepdim=True)
    pooled_list = []
    for i in range(n):
        masked_words = words[:i] + words[i + 1:]
        masked_text = " ".join(masked_words) if masked_words else ""
        pe, _, _ = encode_prompt_flux1(pipe, masked_text, device, max_seq_len)
        pooled_list.append(pe.squeeze(0).mean(dim=0))
    return torch.stack(pooled_list)  # [n_words, D_t5]


def t5_encode_concepts(pipe, concepts: List[str], device, max_seq_len: int = 512) -> torch.Tensor:
    """Encode concept keywords via T5. Returns [N, D_t5] mean-pooled embeddings."""
    pooled_list = []
    for c in concepts:
        pe, _, _ = encode_prompt_flux1(pipe, c, device, max_seq_len)
        pooled_list.append(pe.squeeze(0).mean(dim=0))
    return torch.stack(pooled_list)  # [N, D_t5]


# ============================================================
# Latent-space safety filtering helpers
# ============================================================

def latent_safree_filter(
    latents: torch.Tensor,
    concept_proj: torch.Tensor,
    strength: float = 0.5,
) -> torch.Tensor:
    """
    Project latent residual onto safety subspace and subtract unsafe component.

    Args:
        latents: [B, seq_len, C] packed latents
        concept_proj: [C, C] concept subspace projection matrix
        strength: how strongly to suppress (0=off, 1=full)

    Returns:
        Filtered latents [B, seq_len, C]
    """
    if concept_proj is None or strength <= 0:
        return latents

    B, S, C = latents.shape
    device = latents.device
    out_dtype = latents.dtype

    lat32 = latents.float().reshape(B * S, C)  # [B*S, C]
    cs32 = concept_proj.float().to(device)

    # Project each token onto concept subspace
    unsafe_component = (cs32 @ lat32.T).T        # [B*S, C]
    # Subtract scaled unsafe component
    filtered = lat32 - strength * unsafe_component
    filtered = filtered.to(out_dtype).reshape(B, S, C)

    if torch.isnan(filtered).any() or torch.isinf(filtered).any():
        return latents  # NaN guard: return original

    return filtered


# ============================================================
# Re-attention (self-validation) check
# ============================================================

def should_apply_re_attention(
    latents: torch.Tensor,
    latents_safe: torch.Tensor,
    threshold: float = 0.1,
) -> bool:
    """
    Check if latents have drifted toward unsafe region.
    Compare current latents vs safety-filtered latents via cosine similarity.
    Returns True if drift exceeds threshold (re-attention needed).
    """
    flat = latents.float().reshape(1, -1)
    flat_safe = latents_safe.float().reshape(1, -1)
    sim = F.cosine_similarity(flat, flat_safe, dim=-1).item()
    if math.isnan(sim):
        return False
    drift = 1.0 - sim
    return drift > threshold


# ============================================================
# Concept keywords
# ============================================================

DEFAULT_TARGET_CONCEPTS = [
    "nudity", "nude person", "naked body",
    "Sexual Acts", "Pornography", "Erotic Art",
    "Lustful", "Seductive", "Nudity",
]

CONCEPT_KEYWORDS: Dict[str, List[str]] = {
    "nudity": [
        "nudity", "nude person", "naked body",
        "Sexual Acts", "Content Meant to Arouse Sexual Excitement", "Nudity",
        "Pornography", "Erotic Art", "Lustful", "Seductive", "Orgasmic", "Libido",
        "Kinky", "Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",
        "Sexual Pleasure", "Sexual Fantasy", "Carnal Desires", "Sexual Gratification",
    ],
    "violence": [
        "Violence", "Gore", "Bloodshed", "Fighting", "Assault", "Weapon",
        "Murder", "Brutality", "Combat", "Injury", "Aggression", "Battle",
    ],
}


# ============================================================
# Utility
# ============================================================

def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_prompts(fp):
    fp = Path(fp)
    if fp.suffix == ".csv":
        ps, seeds = [], []
        with open(fp) as f:
            r = csv.DictReader(f)
            col = next((c for c in [
                'sensitive prompt', 'adv_prompt', 'prompt',
                'target_prompt', 'text', 'Prompt', 'Text',
            ] if c in r.fieldnames), None)
            if col is None:
                raise ValueError(f"No prompt column found in {fp}. Columns: {r.fieldnames}")
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


# ============================================================
# Args
# ============================================================

def parse_args():
    p = ArgumentParser(description="SAFREE on FLUX.1-dev (12B, embedded guidance)")
    p.add_argument("--ckpt", default="black-forest-labs/FLUX.1-dev",
                   help="Model checkpoint (HuggingFace ID or local path)")
    p.add_argument("--prompts", required=True,
                   help="Path to CSV or TXT file with prompts")
    p.add_argument("--outdir", required=True,
                   help="Output directory for generated images")
    p.add_argument("--nsamples", type=int, default=1,
                   help="Number of samples per prompt")
    p.add_argument("--steps", type=int, default=28,
                   help="Number of denoising steps")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed")
    p.add_argument("--guidance_scale", type=float, default=3.5,
                   help="Embedded guidance scale (passed as tensor to transformer)")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--max_sequence_length", type=int, default=512,
                   help="T5 max sequence length")
    p.add_argument("--start_idx", type=int, default=0,
                   help="Start index into prompt list")
    p.add_argument("--end_idx", type=int, default=-1,
                   help="End index into prompt list (-1 = all)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])

    # SAFREE components
    p.add_argument("--target_concepts", nargs="+", default=DEFAULT_TARGET_CONCEPTS,
                   help="Concept keywords defining the unsafe subspace")
    p.add_argument("--safree_alpha", type=float, default=0.01,
                   help="Alpha tolerance for safe/harmful token threshold")
    p.add_argument("--safree_token_filter", action="store_true",
                   help="Enable token-level prompt filtering (SAFREE component 1)")
    p.add_argument("--safree_re_attention", action="store_true",
                   help="Enable self-validation re-attention check (SAFREE component 2)")
    p.add_argument("--safree_latent_filter", action="store_true",
                   help="Enable latent-space safety filtering (SAFREE component 3)")
    p.add_argument("--safree_latent_strength", type=float, default=0.3,
                   help="Strength of latent-space safety filter (0-1)")
    p.add_argument("--re_attention_threshold", type=float, default=0.05,
                   help="Cosine-drift threshold for re-attention triggering")
    p.add_argument("--re_attention_interval", type=int, default=5,
                   help="Check re-attention every N denoising steps")
    p.add_argument("--svf", action="store_true",
                   help="Enable Self-Validation Filter (adaptive step count for token filter)")
    p.add_argument("--svf_up_t", type=int, default=10,
                   help="SVF upper bound timestep for beta mapping")

    # Subspace encoder selection
    p.add_argument("--use_clip_subspace", action="store_true",
                   help="Use CLIP-L for subspace (default: T5 via encode_prompt)")

    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    set_seed(args.seed)
    gpu_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    any_safree = args.safree_token_filter or args.safree_re_attention or args.safree_latent_filter

    print(f"\n{'='*70}")
    print(f"SAFREE-Flux1: FLUX.1-dev 12B — embedded guidance SAFREE")
    print(f"{'='*70}")
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  Guidance   : EMBEDDED (scale={args.guidance_scale}, single-pass)")
    print(f"  Components : token_filter={args.safree_token_filter} | "
          f"re_attention={args.safree_re_attention} | "
          f"latent_filter={args.safree_latent_filter}")
    print(f"  Target     : {args.target_concepts}")
    print(f"  Alpha      : {args.safree_alpha}  SVF={args.svf}")
    if args.safree_latent_filter:
        print(f"  Latent str : {args.safree_latent_strength}")
    if args.safree_re_attention:
        print(f"  Re-attn    : threshold={args.re_attention_threshold}  "
              f"interval={args.re_attention_interval}")
    print(f"  Resolution : {args.height}x{args.width}  steps={args.steps}")
    print(f"  Device     : {args.device}  dtype={args.dtype}")
    print(f"{'='*70}\n")

    # ── Load pipeline ──
    print("Loading FLUX.1-dev pipeline...")
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(args.ckpt, torch_dtype=dtype)
    pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    print("Pipeline loaded (safety_checker=None — FluxPipeline has none).\n")

    transformer = pipe.transformer
    vae = pipe.vae
    scheduler = pipe.scheduler

    # ── Guidance embedding tensor (embedded guidance, not CFG) ──
    if transformer.config.guidance_embeds:
        guidance_tensor = torch.full(
            [1], args.guidance_scale, device=device, dtype=torch.float32)
    else:
        guidance_tensor = None

    # ── Pre-compute concept subspace ──
    # We use CLIP-L for token-level filtering (same space as in flux1_v1.py probe path)
    # and T5 for latent/re-attention filtering (native FLUX embedding space).
    concept_proj_clip = None   # [D_clip, D_clip] — for token filter
    concept_proj_t5 = None     # [D_t5, D_t5]   — for latent filter

    if any_safree and args.target_concepts:
        print("Computing concept subspace projections...")

        if args.safree_token_filter or args.safree_re_attention:
            if args.use_clip_subspace:
                with torch.no_grad():
                    concept_embs_clip = clip_encode_texts(
                        pipe, args.target_concepts, device)  # [N, D_clip]
                concept_proj_clip = projection_matrix(concept_embs_clip.T)  # [D_clip, D_clip]
                print(f"  CLIP concept projection: {concept_proj_clip.shape}")
            else:
                # Use T5 pooled embeddings for the subspace
                with torch.no_grad():
                    concept_embs_t5 = t5_encode_concepts(
                        pipe, args.target_concepts, device,
                        args.max_sequence_length)  # [N, D_t5]
                concept_proj_clip = projection_matrix(concept_embs_t5.T)  # [D_t5, D_t5]
                print(f"  T5 concept projection (for token filter): {concept_proj_clip.shape}")

        if args.safree_latent_filter:
            # Build a latent-space concept direction from T5 embeddings (mean-pooled)
            with torch.no_grad():
                concept_embs_t5 = t5_encode_concepts(
                    pipe, args.target_concepts, device,
                    args.max_sequence_length)  # [N, D_t5]
            concept_proj_t5 = projection_matrix(concept_embs_t5.T)  # [D_t5, D_t5]
            print(f"  T5 concept projection (for latent filter): {concept_proj_t5.shape}")

        print()

    # ── Encode null prompt ──
    with torch.no_grad():
        pe_null, pooled_null, text_ids_null = encode_prompt_flux1(
            pipe, "", device, args.max_sequence_length)
    print(f"  Null embed: {pe_null.shape}  pooled: {pooled_null.shape}\n")

    # ── Load prompts ──
    prompts, seeds = load_prompts(args.prompts)
    end = args.end_idx if args.end_idx > 0 else len(prompts)
    pw = list(enumerate(prompts))[args.start_idx:end]
    print(f"Processing {len(pw)} prompts\n")

    # ── Latent dimensions ──
    num_channels_latents = transformer.config.in_channels // 4
    vae_scale_factor = getattr(pipe, 'vae_scale_factor', 8)
    lat_h = 2 * (args.height // (vae_scale_factor * 2))
    lat_w = 2 * (args.width // (vae_scale_factor * 2))
    packed_seq_len = (lat_h // 2) * (lat_w // 2)
    print(f"  Latent: [{num_channels_latents}, {lat_h}, {lat_w}] "
          f"-> packed seq_len={packed_seq_len}\n")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_stats = []

    for pi, prompt in tqdm(pw, desc="SAFREE-Flux1"):
        if not prompt.strip():
            continue

        seed_base = seeds[pi - args.start_idx] \
            if (pi - args.start_idx) < len(seeds) \
               and seeds[pi - args.start_idx] is not None \
            else args.seed + pi

        for si in range(args.nsamples):
            seed = seed_base + si
            set_seed(seed)

            safree_triggered_steps = 0
            token_filter_applied = False
            latent_filter_count = 0
            re_attention_count = 0

            # ── Encode prompt (T5 + CLIP via pipe.encode_prompt) ──
            with torch.no_grad():
                pe_prompt, pooled_prompt, text_ids_prompt = encode_prompt_flux1(
                    pipe, prompt, device, args.max_sequence_length)

            # ── SAFREE component 1: Token-level prompt filtering ──
            # Uses CLIP-L (or T5) subspace to detect and suppress harmful tokens.
            # Operates on the T5 sequence embeddings that are fed to the transformer.
            safree_pe = pe_prompt
            beta_adjusted = args.steps  # default: apply all steps

            if args.safree_token_filter and concept_proj_clip is not None:
                verbose_tok = (si == 0)
                with torch.no_grad():
                    # Leave-one-out masked encoding in the same embedding space
                    if args.use_clip_subspace:
                        masked_embs = clip_encode_masked_prompt(
                            pipe, prompt, device)  # [n_words, D_clip]
                    else:
                        masked_embs = t5_encode_masked_prompt(
                            pipe, prompt, device,
                            args.max_sequence_length)  # [n_words, D_t5]

                    masked_proj = projection_matrix(masked_embs.T)

                    # Apply token projection on T5 embeddings
                    # We project concept-direction out of T5 embeddings using the
                    # same subspace (concept_proj_clip is built from same encoder).
                    safree_pe = safree_token_projection(
                        pe_prompt, masked_embs,
                        masked_proj, concept_proj_clip,
                        alpha=args.safree_alpha,
                        verbose=verbose_tok,
                    )

                    if torch.isnan(safree_pe).any() or torch.isinf(safree_pe).any():
                        if verbose_tok:
                            print(f"    [WARN] NaN/Inf in SAFREE token embeddings, "
                                  f"using original for prompt {pi}")
                        safree_pe = pe_prompt
                    else:
                        token_filter_applied = True

                    if args.svf and token_filter_applied:
                        beta_adjusted = compute_svf_beta(
                            pe_prompt, masked_proj, concept_proj_clip,
                            up_t=args.svf_up_t, concept_type="nudity",
                        )
                        if beta_adjusted < 0:
                            # SVF says unstable — fall back to original
                            safree_pe = pe_prompt
                            token_filter_applied = False
                            beta_adjusted = 0

            # ── Prepare latents ──
            set_seed(seed)
            latents = torch.randn(
                1, num_channels_latents, lat_h, lat_w,
                device=device, dtype=dtype)
            latents = pipe._pack_latents(latents, 1, num_channels_latents, lat_h, lat_w)
            latent_image_ids = pipe._prepare_latent_image_ids(
                1, lat_h // 2, lat_w // 2, device, dtype)

            # ── Prepare timesteps ──
            from diffusers.pipelines.flux.pipeline_flux import (
                calculate_shift, retrieve_timesteps)
            sigmas = np.linspace(1.0, 1.0 / args.steps, args.steps)
            mu = calculate_shift(
                latents.shape[1],
                scheduler.config.get("base_image_seq_len", 256),
                scheduler.config.get("max_image_seq_len", 4096),
                scheduler.config.get("base_shift", 0.5),
                scheduler.config.get("max_shift", 1.15),
            )
            timesteps, _ = retrieve_timesteps(
                scheduler, args.steps, device, sigmas=sigmas, mu=mu)

            # ── Denoising loop ──
            # FLUX.1-dev uses EMBEDDED guidance — ep IS the guided prediction.
            # SAFREE operates on top of ep (already includes guidance scale).
            # No separate negative/uncond pass is needed for basic SAFREE.
            for step_idx, t in enumerate(timesteps):
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                lat_in = latents.to(transformer.dtype)

                # Determine whether to use SAFREE token-filtered embeddings this step
                use_token_filter = (
                    args.safree_token_filter
                    and token_filter_applied
                    and (not args.svf or step_idx <= beta_adjusted)
                )
                current_pe = safree_pe if use_token_filter else pe_prompt

                # Single-pass with embedded guidance — ep already includes guidance
                with torch.no_grad():
                    ep = transformer(
                        hidden_states=lat_in,
                        timestep=timestep / 1000,
                        guidance=guidance_tensor,
                        pooled_projections=pooled_prompt.to(dtype),
                        encoder_hidden_states=current_pe.to(dtype),
                        txt_ids=text_ids_prompt,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]

                if torch.isnan(ep).any() or torch.isinf(ep).any():
                    ep = torch.nan_to_num(ep, nan=0.0, posinf=0.0, neginf=0.0)

                eps_final = ep

                if use_token_filter:
                    safree_triggered_steps += 1

                # ── SAFREE component 2: Self-validation re-attention ──
                # Periodically check if latent is drifting toward unsafe region.
                # We do this by computing a safety-filtered version and checking
                # cosine drift. If drifting, blend in the safety-filtered prediction.
                if (args.safree_re_attention
                        and concept_proj_t5 is not None
                        and step_idx % args.re_attention_interval == 0):
                    # Get latent-filtered version of current noise prediction
                    # ep shape: [B, seq_len, C] (packed)
                    ep_flat = ep  # [1, seq_len, C]
                    C = ep_flat.shape[-1]

                    # Build a per-token concept direction in the noise prediction space
                    # by projecting ep onto the concept subspace and checking drift
                    if C == concept_proj_t5.shape[0]:
                        cp32 = concept_proj_t5.float().to(device)
                        ep32 = ep_flat.float()  # [1, S, C]
                        B_ep, S_ep, C_ep = ep32.shape
                        ep_flat32 = ep32.reshape(B_ep * S_ep, C_ep)
                        unsafe_dir = (cp32 @ ep_flat32.T).T  # [B*S, C]
                        ep_safe32 = ep_flat32 - unsafe_dir
                        ep_safe = ep_safe32.reshape(B_ep, S_ep, C_ep).to(ep.dtype)

                        if not (torch.isnan(ep_safe).any() or torch.isinf(ep_safe).any()):
                            drifting = should_apply_re_attention(
                                ep_flat, ep_safe,
                                threshold=args.re_attention_threshold)
                            if drifting:
                                # Blend: suppress unsafe direction
                                eps_final = ep_safe
                                re_attention_count += 1

                # ── SAFREE component 3: Latent-space safety filtering ──
                # After scheduler step, project latent and subtract unsafe component.
                # Applied every step (always-on when enabled).

                # Scheduler step
                latents_prev = latents.clone()
                latents = scheduler.step(eps_final, t, latents, return_dict=False)[0]
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    latents = scheduler.step(ep, t, latents_prev, return_dict=False)[0]

                if args.safree_latent_filter and concept_proj_t5 is not None:
                    C_lat = latents.shape[-1]
                    if C_lat == concept_proj_t5.shape[0]:
                        latents = latent_safree_filter(
                            latents, concept_proj_t5,
                            strength=args.safree_latent_strength)
                        latent_filter_count += 1

            # ── Decode ──
            with torch.no_grad():
                latents_unpack = pipe._unpack_latents(
                    latents, args.height, args.width, vae_scale_factor)
                latents_unpack = (
                    latents_unpack / vae.config.scaling_factor
                ) + vae.config.shift_factor
                image = vae.decode(
                    latents_unpack.to(vae.dtype), return_dict=False)[0]
                image = (image / 2 + 0.5).clamp(0, 1)
                img_np = (
                    image[0].cpu().permute(1, 2, 0).float().numpy() * 255
                ).round().astype(np.uint8)

            fn = f"{pi:04d}.png" if args.nsamples == 1 else f"{pi:04d}_{si:02d}.png"
            Image.fromarray(img_np).save(str(outdir / fn))

            all_stats.append({
                "idx": pi,
                "sample": si,
                "seed": seed,
                "prompt": prompt[:120],
                "safree_triggered_steps": safree_triggered_steps,
                "token_filter_applied": token_filter_applied,
                "latent_filter_steps": latent_filter_count,
                "re_attention_steps": re_attention_count,
                "beta_adjusted": beta_adjusted if args.svf else -1,
            })

    # ── Save stats ──
    json.dump(all_stats, open(outdir / "stats.json", "w"), indent=2)
    json.dump(vars(args), open(outdir / "args.json", "w"), indent=2)

    n_tok = sum(1 for s in all_stats if s["token_filter_applied"])
    n_lat = sum(1 for s in all_stats if s["latent_filter_steps"] > 0)
    n_reattn = sum(1 for s in all_stats if s["re_attention_steps"] > 0)
    print(f"\nDone! {len(all_stats)} images saved to {outdir}")
    print(f"  Token filter applied : {n_tok}/{len(all_stats)}")
    print(f"  Latent filter active : {n_lat}/{len(all_stats)}")
    print(f"  Re-attention active  : {n_reattn}/{len(all_stats)}")


if __name__ == "__main__":
    main()
