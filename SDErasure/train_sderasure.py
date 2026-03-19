#!/usr/bin/env python3
"""
SDErasure: Concept-Specific Trajectory Shifting for Concept Erasure
via Adaptive Diffusion Classifier

ICLR 2026 Implementation

Components:
  1. Step Selection  - Compute SSScore at each timestep, select t where SSScore > λ
  2. Score Rematching loss (Le) - Shift denoising trajectory from target → anchor
  3. Early-Preserve loss (Lp)  - Keep predictions stable at high-noise (early) steps
  4. Concept-Retain loss (Lr)  - Preserve unrelated concept generations
  5. Combined objective: Lo = Le + β1·Lr + β2·Lp

Usage:
  python train_sderasure.py \
    --target_concept "nudity" \
    --anchor_concept "" \
    --retain_concepts "a photo of a person" "a landscape" \
    --output_dir ./outputs/sderasure_nudity

  # Object erasure (anchor-based)
  python train_sderasure.py \
    --target_concept "cat" \
    --anchor_concept "dog" \
    --retain_concepts "airplane" "automobile" "bird" \
    --output_dir ./outputs/sderasure_cat

  # Celebrity erasure
  python train_sderasure.py \
    --target_concept "Elon Musk" \
    --anchor_concept "" \
    --retain_concepts "a man" "a person" \
    --output_dir ./outputs/sderasure_elon_musk
"""

import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from diffusers import DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer


# ============================================================================
# Model Loading
# ============================================================================

def load_models(model_id: str, device: torch.device, train_attn_only: bool = False):
    """
    Load frozen reference model and trainable fine-tuned model.

    Returns:
        tokenizer, text_encoder, vae, unet_frozen, unet, scheduler
    """
    print(f"Loading models from {model_id} ...")

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Frozen reference model (θ*) — kept in fp16 to save memory, never updated
    unet_frozen = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float16
    ).to(device)
    unet_frozen.eval()
    for p in unet_frozen.parameters():
        p.requires_grad = False

    # Trainable model (θ) — fp32 for stable gradient computation
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)

    if train_attn_only:
        # Fine-tune only cross-attention layers (more conservative, less memory)
        for name, param in unet.named_parameters():
            if "attn2" in name:   # cross-attention
                param.requires_grad = True
            else:
                param.requires_grad = False
        n_trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        print(f"Training cross-attention only: {n_trainable:,} parameters")
    else:
        # Full UNet fine-tuning (ESD-u style)
        n_trainable = sum(p.numel() for p in unet.parameters())
        print(f"Training full UNet: {n_trainable:,} parameters")

    # Enable gradient checkpointing to reduce activation memory
    unet.enable_gradient_checkpointing()

    return tokenizer, text_encoder, vae, unet_frozen, unet, scheduler


# ============================================================================
# Text Embedding Utilities
# ============================================================================

@torch.no_grad()
def get_text_embeddings(tokenizer, text_encoder, texts, device: torch.device):
    """Encode text prompt(s) to CLIP embeddings. Returns (B, 77, 768)."""
    if isinstance(texts, str):
        texts = [texts]
    tokens = tokenizer(
        texts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(tokens.input_ids.to(device))[0]


# ============================================================================
# Step Selection — SSScore Computation  (Section 3.3)
# ============================================================================

@torch.no_grad()
def compute_ssscore(
    unet_frozen,
    scheduler,
    emb_target: torch.Tensor,
    emb_anchor: torch.Tensor,
    n_eval_timesteps: int = 50,
    n_samples: int = 4,
    latent_shape: tuple = (4, 64, 64),
    device: torch.device = torch.device("cuda"),
    image_latents: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute SSScore for every evaluated timestep (Eq. 8 of the paper).

    St = exp(-L_t^c) / (exp(-L_t^c) + exp(-L_t^a))

    where:
      L_t^c = MSE(ε_θ*(x_t, ct, t), ε)
      L_t^a = MSE(ε_θ*(x_t, ca, t), ε)

    Args:
        unet_frozen:      frozen reference UNet (θ*)
        scheduler:        DDPMScheduler
        emb_target:       text embedding for target concept  (1, 77, D)
        emb_anchor:       text embedding for anchor concept  (1, 77, D)
        n_eval_timesteps: number of timesteps to evaluate (evenly spaced)
        n_samples:        number of latent samples to average over
        latent_shape:     (C, H, W) — (4, 64, 64) for SD v1.4
        image_latents:    optional pre-encoded latents of target concept images

    Returns:
        scores:            (n_eval_timesteps,) SSScore per evaluated timestep
        timestep_indices:  (n_eval_timesteps,) DDPM timestep values (0 … T-1)
    """
    T = scheduler.config.num_train_timesteps  # 1000 for DDPM
    # Evenly spaced timesteps: 0 to T-1
    timestep_indices = torch.linspace(0, T - 1, n_eval_timesteps, dtype=torch.long)

    scores = []
    unet_frozen.eval()

    # Expand text embeddings for batch
    emb_t = emb_target.expand(n_samples, -1, -1)  # (B, 77, D)
    emb_a = emb_anchor.expand(n_samples, -1, -1)

    for t_val in tqdm(timestep_indices, desc="Computing SSScore"):
        t = t_val.to(device).unsqueeze(0).expand(n_samples)

        # Use provided latents or sample random ones
        if image_latents is not None:
            idx = torch.randint(0, image_latents.shape[0], (n_samples,))
            x0 = image_latents[idx].float().to(device)
        else:
            x0 = torch.randn(n_samples, *latent_shape, device=device)

        eps = torch.randn_like(x0)
        xt = scheduler.add_noise(x0, eps, t)

        # Noise prediction under target concept (fp16 frozen model)
        pred_target = unet_frozen(xt.half(), t, emb_t.half()).sample.float()
        L_c = F.mse_loss(pred_target, eps).item()

        # Noise prediction under anchor concept
        pred_anchor = unet_frozen(xt.half(), t, emb_a.half()).sample.float()
        L_a = F.mse_loss(pred_anchor, eps).item()

        # SSScore (Eq. 8): instantaneous posterior probability of target concept
        # S_t = exp(-L_c) / (exp(-L_c) + exp(-L_a)) = 1 / (1 + exp(L_c - L_a))
        delta = L_c - L_a
        S_t = 1.0 / (1.0 + np.exp(np.clip(delta, -50, 50)))
        scores.append(S_t)

    return torch.tensor(scores, dtype=torch.float32), timestep_indices


def select_critical_timesteps(
    scores: torch.Tensor,
    timestep_indices: torch.Tensor,
    threshold: float = 0.8,
) -> list[int]:
    """
    Select timesteps where SSScore > λ (threshold).
    Falls back to the top-5 highest scoring timesteps if none pass the threshold.
    """
    mask = scores > threshold
    selected = timestep_indices[mask].tolist()

    if len(selected) == 0:
        print(f"  [warn] No timesteps with SSScore > {threshold:.2f}. "
              f"Falling back to top-5 timesteps.")
        top5 = torch.topk(scores, k=min(5, len(scores))).indices
        selected = timestep_indices[top5].tolist()

    return [int(t) for t in selected]


# ============================================================================
# Loss Functions  (Section 3.4)
# ============================================================================

def _frozen_pred(unet_frozen, xt, t, emb):
    """Run frozen (fp16) UNet and return fp32 prediction."""
    with torch.no_grad():
        return unet_frozen(xt.half(), t, emb.half()).sample.float()


def score_rematching_loss(
    unet: torch.nn.Module,
    unet_frozen: torch.nn.Module,
    xt: torch.Tensor,
    t: torch.Tensor,
    emb_target: torch.Tensor,
    emb_anchor: torch.Tensor,
    eta: float = 1.0,
) -> torch.Tensor:
    """
    Score Rematching Loss (Eq. 11):

      Le = ||ε_θ(xt, ct, t) - [ε_θ*(xt, ca, t) - η·σ(xt, ct, ca, t)]||²

    where σ = ε_θ*(xt, ct, t) - ε_θ*(xt, ca, t)  (concept trajectory shift)

    When anchor is empty (""), this becomes anchor-free erasure.
    """
    pred_target_frozen = _frozen_pred(unet_frozen, xt, t, emb_target)
    pred_anchor_frozen = _frozen_pred(unet_frozen, xt, t, emb_anchor)

    # Conceptual trajectory shift σ (Eq. 10)
    sigma = pred_target_frozen - pred_anchor_frozen
    # Rematching target: steer toward anchor, away from concept (Eq. 11)
    rematch_target = pred_anchor_frozen - eta * sigma

    # Fine-tuned model prediction
    pred = unet(xt, t, emb_target).sample
    return F.mse_loss(pred, rematch_target)


def early_preserve_loss(
    unet: torch.nn.Module,
    unet_frozen: torch.nn.Module,
    xt_early: torch.Tensor,
    t_early: torch.Tensor,
    emb_target: torch.Tensor,
) -> torch.Tensor:
    """
    Early-Preserve Loss (Eq. 12):

      Lp = ||ε_θ(xt*, ct, t*) - ε_θ*(xt*, ct, t*)||²

    Prevents disruption at early denoising steps (high-noise phase)
    where all concepts converge toward the natural image manifold.
    These steps have low SSScore and should not be perturbed.
    """
    ref = _frozen_pred(unet_frozen, xt_early, t_early, emb_target)
    pred = unet(xt_early, t_early, emb_target).sample
    return F.mse_loss(pred, ref)


def concept_retain_loss(
    unet: torch.nn.Module,
    unet_frozen: torch.nn.Module,
    xt_retain: torch.Tensor,
    t_retain: torch.Tensor,
    emb_retain: torch.Tensor,
) -> torch.Tensor:
    """
    Concept-Retain Loss (Eq. 13):

      Lr = ||ε_θ(xt, cr, t) - ε_θ*(xt, cr, t)||²

    Prevents the fine-tuning from degrading unrelated concept generation.
    """
    ref = _frozen_pred(unet_frozen, xt_retain, t_retain, emb_retain)
    pred = unet(xt_retain, t_retain, emb_retain).sample
    return F.mse_loss(pred, ref)


# ============================================================================
# Optional: Encode real images with VAE for better SSScore computation
# ============================================================================

@torch.no_grad()
def encode_images_to_latents(vae, image_paths: list[str], device: torch.device) -> torch.Tensor:
    """Encode real images to VAE latent space for SSScore computation."""
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    latents_list = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        pixel = transform(img).unsqueeze(0).to(device)
        lat = vae.encode(pixel).latent_dist.sample() * vae.config.scaling_factor
        latents_list.append(lat)
    return torch.cat(latents_list, dim=0)


@torch.no_grad()
def generate_concept_images_as_latents(
    unet_frozen: UNet2DConditionModel,
    scheduler,
    tokenizer,
    text_encoder,
    target_concept: str,
    n_images: int = 8,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    Generate clean latents of the target concept using DDIM sampling with the
    frozen model. These latents are used for SSScore computation.

    Uses a minimal DDIM loop to avoid loading a second copy of the pipeline.
    Returns latents: (n_images, 4, 64, 64)
    """
    print(f"  Generating {n_images} concept latents for SSScore ('{target_concept}')...")

    # Build a DDIM scheduler from the DDPM scheduler config
    ddim = DDIMScheduler(
        num_train_timesteps=scheduler.config.num_train_timesteps,
        beta_start=scheduler.config.beta_start,
        beta_end=scheduler.config.beta_end,
        beta_schedule=scheduler.config.beta_schedule,
        clip_sample=False,
    )
    ddim.set_timesteps(num_inference_steps, device=device)

    # Encode the text prompt
    emb_cond = get_text_embeddings(tokenizer, text_encoder, target_concept, device)  # (1,77,D)
    emb_uncond = get_text_embeddings(tokenizer, text_encoder, "", device)             # (1,77,D)
    # Batch: [uncond, cond] for classifier-free guidance
    emb = torch.cat([emb_uncond, emb_cond])  # (2, 77, D)

    emb_half = emb.half()  # match frozen unet's dtype

    all_latents = []
    for i in range(n_images):
        lat = torch.randn(1, 4, 64, 64, device=device)
        for t in ddim.timesteps:
            lat_in = torch.cat([lat, lat]).half()  # (2, 4, 64, 64) fp16
            t_in = t.unsqueeze(0).expand(2)
            noise_pred = unet_frozen(lat_in, t_in, emb_half).sample.float()
            # Classifier-free guidance
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            lat = ddim.step(noise_pred, t, lat).prev_sample

        all_latents.append(lat)
        if (i + 1) % 4 == 0:
            print(f"    [{i+1}/{n_images}]")

    return torch.cat(all_latents, dim=0)  # (n_images, 4, 64, 64)


# ============================================================================
# Main Training
# ============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------ Load
    tokenizer, text_encoder, vae, unet_frozen, unet, scheduler = load_models(
        args.model_id, device, train_attn_only=args.train_attn_only
    )

    T = scheduler.config.num_train_timesteps  # 1000

    # ------------------------------------------------------------------ Embeddings
    print(f"\nTarget concept : '{args.target_concept}'")
    print(f"Anchor concept : '{args.anchor_concept}' (empty = anchor-free erasure)")
    print(f"Retain concepts: {args.retain_concepts}\n")

    emb_target = get_text_embeddings(tokenizer, text_encoder, args.target_concept, device)
    emb_anchor = get_text_embeddings(tokenizer, text_encoder,
                                     args.anchor_concept if args.anchor_concept else "", device)

    emb_retain_list = []
    for c in args.retain_concepts:
        emb_retain_list.append(get_text_embeddings(tokenizer, text_encoder, c, device))

    # ------------------------------------------------------------------ SSScore Step Selection
    print("=" * 60)
    print("STEP 1: Computing SSScore for step selection...")
    print("=" * 60)

    # Obtain concept latents for SSScore computation:
    # Priority: (1) user-supplied image dir, (2) generate from model, (3) random noise
    image_latents = None
    if args.image_dir and os.path.isdir(args.image_dir):
        exts = (".png", ".jpg", ".jpeg", ".webp")
        img_paths = [str(p) for p in Path(args.image_dir).iterdir()
                     if p.suffix.lower() in exts][:args.n_ssscore_images]
        if img_paths:
            print(f"Using {len(img_paths)} real images from {args.image_dir} for SSScore.")
            image_latents = encode_images_to_latents(vae, img_paths, device)

    if image_latents is None and not args.random_latents_for_ssscore:
        # Generate concept images from the frozen model for more accurate SSScore
        image_latents = generate_concept_images_as_latents(
            unet_frozen=unet_frozen,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            target_concept=args.target_concept,
            n_images=args.n_ssscore_images,
            device=device,
        )
        print(f"  Generated latents shape: {image_latents.shape}")
    elif image_latents is None:
        print("  Using random latents for SSScore (less accurate).")

    scores, timestep_indices = compute_ssscore(
        unet_frozen=unet_frozen,
        scheduler=scheduler,
        emb_target=emb_target,
        emb_anchor=emb_anchor,
        n_eval_timesteps=args.n_eval_timesteps,
        n_samples=args.ssscore_batch_size,
        latent_shape=(4, 64, 64),
        device=device,
        image_latents=image_latents,
    )

    selected_timesteps = select_critical_timesteps(scores, timestep_indices, args.lambda_threshold)
    print(f"\nSSScore — min: {scores.min():.3f}  max: {scores.max():.3f}  mean: {scores.mean():.3f}")
    print(f"Selected {len(selected_timesteps)} critical timesteps (λ={args.lambda_threshold}): "
          f"{sorted(selected_timesteps)}")

    # Early timestep range: high-noise phase (t close to T)
    # Paper: "45 < t < 50" of DDIM-50 ≈ 900-999 in DDPM-1000 scale
    early_t_lo = int(args.early_t_fraction_lo * T)  # e.g. 900
    early_t_hi = int(args.early_t_fraction_hi * T)  # e.g. 1000
    early_t_hi = min(early_t_hi, T)
    print(f"Early-preserve timestep range: [{early_t_lo}, {early_t_hi}]")

    # ------------------------------------------------------------------ Save SSScore
    os.makedirs(args.output_dir, exist_ok=True)
    ssscore_path = os.path.join(args.output_dir, "ssscore.json")
    with open(ssscore_path, "w") as f:
        json.dump({
            "target_concept": args.target_concept,
            "anchor_concept": args.anchor_concept,
            "lambda_threshold": args.lambda_threshold,
            "scores": scores.tolist(),
            "timestep_indices": timestep_indices.tolist(),
            "selected_timesteps": selected_timesteps,
        }, f, indent=2)
    print(f"SSScore saved to {ssscore_path}")

    # ------------------------------------------------------------------ Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ------------------------------------------------------------------ Training loop
    print("\n" + "=" * 60)
    print(f"STEP 2: Training SDErasure for {args.num_steps} steps...")
    print("=" * 60)

    unet.train()
    log_history = []
    selected_arr = np.array(selected_timesteps)

    for step in tqdm(range(1, args.num_steps + 1), desc="Training"):
        optimizer.zero_grad()

        B = args.batch_size
        x0 = torch.randn(B, 4, 64, 64, device=device)

        # ---- Score Rematching Loss (Le) --------------------------------
        # Sample t from selected critical timesteps
        t_idx = np.random.choice(selected_arr, size=B, replace=True)
        t = torch.tensor(t_idx, device=device, dtype=torch.long)
        eps = torch.randn_like(x0)
        xt = scheduler.add_noise(x0, eps, t)

        emb_t = emb_target.expand(B, -1, -1)
        emb_a = emb_anchor.expand(B, -1, -1)

        Le = score_rematching_loss(unet, unet_frozen, xt, t, emb_t, emb_a, eta=args.eta)

        # ---- Early-Preserve Loss (Lp) ----------------------------------
        t_early = torch.randint(early_t_lo, early_t_hi, (B,), device=device)
        x0_early = torch.randn(B, 4, 64, 64, device=device)
        eps_early = torch.randn_like(x0_early)
        xt_early = scheduler.add_noise(x0_early, eps_early, t_early)
        emb_t_b = emb_target.expand(B, -1, -1)

        Lp = early_preserve_loss(unet, unet_frozen, xt_early, t_early, emb_t_b)

        # ---- Concept-Retain Loss (Lr) ----------------------------------
        if emb_retain_list:
            # Cycle through retain concepts
            emb_r_1 = emb_retain_list[step % len(emb_retain_list)]
            emb_r = emb_r_1.expand(B, -1, -1)

            t_retain = torch.randint(0, T, (B,), device=device)
            x0_retain = torch.randn(B, 4, 64, 64, device=device)
            eps_retain = torch.randn_like(x0_retain)
            xt_retain = scheduler.add_noise(x0_retain, eps_retain, t_retain)

            Lr = concept_retain_loss(unet, unet_frozen, xt_retain, t_retain, emb_r)
        else:
            Lr = torch.tensor(0.0, device=device)

        # ---- Combined objective (Eq. 14) --------------------------------
        loss = Le + args.beta1 * Lr + args.beta2 * Lp

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in unet.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        # ---- Logging ---------------------------------------------------
        if step % args.log_every == 0 or step == 1:
            rec = {
                "step": step,
                "loss": loss.item(),
                "Le": Le.item(),
                "Lr": Lr.item() if isinstance(Lr, torch.Tensor) else Lr,
                "Lp": Lp.item(),
            }
            log_history.append(rec)
            tqdm.write(
                f"[step {step:5d}] loss={loss.item():.4f}  "
                f"Le={Le.item():.4f}  Lr={Lr.item() if isinstance(Lr, torch.Tensor) else Lr:.4f}  "
                f"Lp={Lp.item():.4f}"
            )

        # ---- Checkpoint ------------------------------------------------
        if step % args.save_every == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            unet.save_pretrained(ckpt_dir)
            print(f"Checkpoint saved: {ckpt_dir}")

    # ------------------------------------------------------------------ Final Save
    final_unet_dir = os.path.join(args.output_dir, "unet")
    unet.save_pretrained(final_unet_dir)
    print(f"\nFinal UNet saved to: {final_unet_dir}")

    # Save training log
    log_path = os.path.join(args.output_dir, "train_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "args": vars(args),
            "selected_timesteps": selected_timesteps,
            "history": log_history,
        }, f, indent=2)
    print(f"Training log saved to: {log_path}")

    # ------------------------------------------------------------------ Summary
    print("\n" + "=" * 60)
    print("SDErasure Training Complete")
    print(f"  Target concept : {args.target_concept}")
    print(f"  Anchor concept : {args.anchor_concept or '(none — anchor-free)'}")
    print(f"  Steps trained  : {args.num_steps}")
    print(f"  Output dir     : {args.output_dir}")
    print("=" * 60)


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="SDErasure: Concept Erasure via Adaptive Trajectory Shifting")

    # ---- Model ----
    p.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4",
                   help="HuggingFace model ID or local path")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save fine-tuned UNet and logs")
    p.add_argument("--train_attn_only", action="store_true",
                   help="Fine-tune only cross-attention layers (conservative; default: full UNet)")

    # ---- Concepts ----
    p.add_argument("--target_concept", type=str, required=True,
                   help="Concept to erase (e.g., 'nudity', 'cat', 'Elon Musk')")
    p.add_argument("--anchor_concept", type=str, default="",
                   help="Replacement concept for anchor-based altering. "
                        "Leave empty for anchor-free erasure.")
    p.add_argument("--retain_concepts", type=str, nargs="*", default=[],
                   help="Concepts whose generation quality should be preserved")

    # ---- SSScore / Step Selection ----
    p.add_argument("--lambda_threshold", type=float, default=0.8,
                   help="SSScore threshold λ for step selection (paper: best at 0.8)")
    p.add_argument("--n_eval_timesteps", type=int, default=50,
                   help="Number of timesteps to evaluate SSScore over (0..T-1)")
    p.add_argument("--ssscore_batch_size", type=int, default=4,
                   help="Batch size for SSScore computation")
    p.add_argument("--image_dir", type=str, default=None,
                   help="(Optional) Directory of real target concept images for SSScore")
    p.add_argument("--n_ssscore_images", type=int, default=8,
                   help="Number of concept images to generate (or load) for SSScore")
    p.add_argument("--random_latents_for_ssscore", action="store_true",
                   help="Use random noise latents for SSScore instead of generated images (faster but less accurate)")

    # ---- Training ----
    p.add_argument("--num_steps", type=int, default=500,
                   help="Number of fine-tuning gradient steps")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5, help="AdamW learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--eta", type=float, default=1.0,
                   help="Erasure strength η in Score Rematching loss (Eq. 11)")
    p.add_argument("--beta1", type=float, default=0.1,
                   help="Weight β1 for Concept-Retain loss Lr (Eq. 14)")
    p.add_argument("--beta2", type=float, default=0.1,
                   help="Weight β2 for Early-Preserve loss Lp (Eq. 14)")

    # ---- Early-Preserve timestep range ----
    # "45 < t < 50" in DDIM-50 ≈ 0.90 * T to T in DDPM-1000 scale
    p.add_argument("--early_t_fraction_lo", type=float, default=0.90,
                   help="Lower bound of early timestep range as fraction of T (default 0.90 → t=900)")
    p.add_argument("--early_t_fraction_hi", type=float, default=1.00,
                   help="Upper bound of early timestep range as fraction of T (default 1.00 → t=1000)")

    # ---- Misc ----
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=250)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
