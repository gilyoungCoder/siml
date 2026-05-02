#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare Contrastive CLIP Image Direction for v16.

Computes a "nudity direction" in CLIP image space by contrasting nude vs clothed
exemplar images:
    d_concept = normalize(mean(CLIP(nude_images)) - mean(CLIP(clothed_images)))

This difference vector isolates nudity-specific features, removing shared features
(pose, background, person). More concept-specific than either raw nude CLS or
text "nudity".

Two levels of features are extracted:
  1. CLS token [768] — global image-level direction
  2. Patch tokens [256, 768] — spatial, per-patch direction

Probe embeddings are built in three variants:
  A. cls_contrastive: CLS direction repeated as concept tokens
  B. patch_contrastive: top-K most discriminative patch directions
  C. mixed: CLS + top-(K-1) patch directions

Usage:
    CUDA_VISIBLE_DEVICES=0 python prepare_contrastive_direction.py
    CUDA_VISIBLE_DEVICES=0 python prepare_contrastive_direction.py \
        --exemplar_dir exemplars/sd14/exemplar_images \
        --output exemplars/sd14/contrastive_embeddings.pt \
        --n_tokens 4 --top_k 8

Output:
    contrastive_embeddings.pt containing:
      - cls_direction: [768]
      - patch_directions: [256, 768]
      - top_k_patch_dirs: [K, 768]
      - top_k_patch_norms: [K] (direction magnitudes)
      - top_k_patch_indices: [K] (which patches were selected)
      - target_embeds_cls: [1, 77, 768] (option A)
      - target_embeds_patch: [1, 77, 768] (option B)
      - target_embeds_mixed: [1, 77, 768] (option C)
      - cosine_sim: float (nude vs clothed avg similarity, for reference)
      - config: metadata dict
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_exemplar_images(exemplar_dir, prefix, max_images=16):
    """Load exemplar images with given prefix (e.g., 'nudity_' or 'clothed_')."""
    images = []
    img_dir = Path(exemplar_dir)
    files = sorted([f for f in img_dir.iterdir()
                    if f.name.startswith(prefix) and f.suffix == '.png'])
    for f in files[:max_images]:
        img = Image.open(f).convert('RGB')
        images.append(img)
    print(f"  Loaded {len(images)} images with prefix '{prefix}' from {exemplar_dir}")
    return images


def extract_clip_features_with_patches(images, device):
    """
    Extract both CLS token and patch tokens from CLIP ViT-L/14.

    SD1.4 uses openai/clip-vit-large-patch14 (768-dim).
    ViT-L/14 processes 224x224 images into 16x16 patches = 256 patches.

    Returns:
        cls_features: [N, 768] CLS token per image (normalized)
        patch_features: [N, 256, 768] patch tokens per image (normalized)
    """
    from transformers import CLIPModel, CLIPProcessor

    model_name = "openai/clip-vit-large-patch14"
    print(f"  Loading CLIP model: {model_name}")
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.eval()

    cls_features = []
    patch_features = []

    with torch.no_grad():
        for img in tqdm(images, desc="  Extracting CLIP features"):
            inputs = clip_processor(images=img, return_tensors="pt").to(device)

            # Get full vision model output (not just pooled)
            vision_outputs = clip_model.vision_model(
                pixel_values=inputs["pixel_values"]
            )
            # last_hidden_state: [1, 257, 1024] for ViT-L/14
            # but after visual projection it becomes 768-dim
            hidden = vision_outputs.last_hidden_state  # [1, 257, 1024]

            # Apply visual projection to map 1024 -> 768
            # The visual_projection maps pooled output; for patches we use it too
            cls_token = hidden[:, 0, :]  # [1, 1024]
            patches = hidden[:, 1:, :]   # [1, 256, 1024]

            # Project CLS through the visual projection layer
            cls_projected = clip_model.visual_projection(cls_token)  # [1, 768]
            cls_projected = cls_projected / cls_projected.norm(dim=-1, keepdim=True)

            # Project each patch through the same projection
            B, N, D = patches.shape
            patches_flat = patches.reshape(B * N, D)  # [256, 1024]
            patches_projected = clip_model.visual_projection(patches_flat)  # [256, 768]
            patches_projected = patches_projected.reshape(B, N, -1)  # [1, 256, 768]
            patches_projected = patches_projected / patches_projected.norm(
                dim=-1, keepdim=True
            )

            cls_features.append(cls_projected.cpu())
            patch_features.append(patches_projected.cpu())

    cls_features = torch.cat(cls_features, dim=0)      # [N, 768]
    patch_features = torch.cat(patch_features, dim=0)   # [N, 256, 768]

    del clip_model
    torch.cuda.empty_cache()
    return cls_features, patch_features


def compute_contrastive_directions(nude_cls, nude_patches, clothed_cls, clothed_patches,
                                   top_k=8):
    """
    Compute contrastive directions between nude and clothed features.

    Args:
        nude_cls: [N_nude, 768] CLS tokens from nude images
        nude_patches: [N_nude, 256, 768] patch tokens from nude images
        clothed_cls: [N_clothed, 768] CLS tokens from clothed images
        clothed_patches: [N_clothed, 256, 768] patch tokens from clothed images
        top_k: number of most discriminative patches to select

    Returns:
        cls_direction: [768] normalized CLS contrastive direction
        patch_directions: [256, 768] per-patch contrastive directions (normalized)
        top_k_dirs: [K, 768] top-K most discriminative patch directions
        top_k_norms: [K] direction magnitudes before normalization
        top_k_indices: [K] which patches were selected
        cosine_sim: float, similarity between nude and clothed means
    """
    # CLS direction
    mean_nude_cls = nude_cls.mean(dim=0)      # [768]
    mean_clothed_cls = clothed_cls.mean(dim=0)  # [768]
    cls_diff = mean_nude_cls - mean_clothed_cls  # [768]
    cls_direction = F.normalize(cls_diff, dim=0)  # [768]

    # Cosine similarity for reference
    cosine_sim = F.cosine_similarity(
        mean_nude_cls.unsqueeze(0), mean_clothed_cls.unsqueeze(0)
    ).item()

    # Patch directions
    mean_nude_patches = nude_patches.mean(dim=0)      # [256, 768]
    mean_clothed_patches = clothed_patches.mean(dim=0)  # [256, 768]
    patch_diff = mean_nude_patches - mean_clothed_patches  # [256, 768]

    # Compute norms before normalizing (magnitude = discriminativeness)
    patch_norms = patch_diff.norm(dim=-1)  # [256]
    patch_directions = F.normalize(patch_diff, dim=-1)  # [256, 768]

    # Select top-K patches by direction magnitude (most discriminative)
    top_k_actual = min(top_k, patch_norms.shape[0])
    top_k_norms, top_k_indices = torch.topk(patch_norms, top_k_actual)
    top_k_dirs = patch_directions[top_k_indices]  # [K, 768]

    print(f"  CLS direction norm (before normalize): {cls_diff.norm():.4f}")
    print(f"  Cosine sim (nude vs clothed): {cosine_sim:.4f}")
    print(f"  Patch direction norms: min={patch_norms.min():.4f}, "
          f"max={patch_norms.max():.4f}, mean={patch_norms.mean():.4f}")
    print(f"  Top-{top_k_actual} patch indices: {top_k_indices.tolist()}")
    print(f"  Top-{top_k_actual} patch norms: "
          f"{[f'{n:.4f}' for n in top_k_norms.tolist()]}")

    return cls_direction, patch_directions, top_k_dirs, top_k_norms, top_k_indices, cosine_sim


def build_probe_embeddings(cls_direction, top_k_dirs, text_encoder, tokenizer,
                           device, n_tokens=4):
    """
    Build 77-token probe embeddings from contrastive directions.

    Three options:
      A. cls_contrastive: [BOS, cls_dir, cls_dir, ..., EOS, PAD...]
      B. patch_contrastive: [BOS, top_patch_1, ..., top_patch_K, EOS, PAD...]
      C. mixed: [BOS, cls_dir, top_patch_1, ..., top_patch_{K-1}, EOS, PAD...]

    Args:
        cls_direction: [768] normalized CLS contrastive direction
        top_k_dirs: [K, 768] top-K patch directions
        text_encoder: SD's CLIPTextModel
        tokenizer: SD's CLIPTokenizer
        device: torch device
        n_tokens: number of concept tokens per embedding

    Returns:
        embeds_cls: [1, 77, 768] option A
        embeds_patch: [1, 77, 768] option B
        embeds_mixed: [1, 77, 768] option C
    """
    token_embedding = text_encoder.text_model.embeddings.token_embedding

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    with torch.no_grad():
        bos_embed = token_embedding(torch.tensor([bos_id], device=device))  # [1, 768]
        eos_embed = token_embedding(torch.tensor([eos_id], device=device))
        pad_embed = token_embedding(torch.tensor([pad_id], device=device))

    def _build_sequence(concept_tokens):
        """Build 77-token sequence: BOS + concept_tokens + EOS + padding."""
        tokens_list = [bos_embed]
        for tok in concept_tokens:
            tokens_list.append(tok.unsqueeze(0).to(device))  # [1, 768]
        tokens_list.append(eos_embed)
        n_pad = 77 - len(tokens_list)
        for _ in range(n_pad):
            tokens_list.append(pad_embed)
        return torch.cat(tokens_list, dim=0).unsqueeze(0)  # [1, 77, 768]

    cls_dir = cls_direction.to(device)
    top_dirs = top_k_dirs.to(device)

    # Option A: CLS direction repeated n_tokens times
    concept_a = [cls_dir for _ in range(n_tokens)]
    embeds_cls = _build_sequence(concept_a)

    # Option B: Top-K patch directions (use up to n_tokens)
    n_patch = min(n_tokens, top_dirs.shape[0])
    concept_b = [top_dirs[i] for i in range(n_patch)]
    # Pad with CLS if fewer patches than n_tokens
    while len(concept_b) < n_tokens:
        concept_b.append(cls_dir)
    embeds_patch = _build_sequence(concept_b)

    # Option C: Mixed — CLS + top-(n_tokens-1) patches
    concept_c = [cls_dir]
    n_mixed_patch = min(n_tokens - 1, top_dirs.shape[0])
    for i in range(n_mixed_patch):
        concept_c.append(top_dirs[i])
    while len(concept_c) < n_tokens:
        concept_c.append(cls_dir)
    embeds_mixed = _build_sequence(concept_c)

    return embeds_cls, embeds_patch, embeds_mixed


def main():
    parser = ArgumentParser(description="Prepare contrastive CLIP direction for v16")
    parser.add_argument("--exemplar_dir", type=str,
                        default="exemplars/sd14/exemplar_images",
                        help="Directory with nudity_XX.png and clothed_XX.png")
    parser.add_argument("--output", type=str,
                        default="exemplars/sd14/contrastive_embeddings.pt")
    parser.add_argument("--n_nudity", type=int, default=16)
    parser.add_argument("--n_clothed", type=int, default=16)
    parser.add_argument("--target_prefix", type=str, default="nudity_",
                        help="Filename prefix for target exemplar images")
    parser.add_argument("--anchor_prefix", type=str, default="clothed_",
                        help="Filename prefix for anchor exemplar images")
    parser.add_argument("--n_tokens", type=int, default=4,
                        help="Number of concept tokens in probe embeddings")
    parser.add_argument("--top_k", type=int, default=8,
                        help="Number of most discriminative patches to select")
    parser.add_argument("--ckpt", type=str, default="CompVis/stable-diffusion-v1-4",
                        help="SD checkpoint (for text encoder token embeddings)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load exemplar images ---
    print("\n=== Loading exemplar images ===")
    nude_images = load_exemplar_images(args.exemplar_dir, args.target_prefix, args.n_nudity)
    clothed_images = load_exemplar_images(args.exemplar_dir, args.anchor_prefix, args.n_clothed)

    if len(nude_images) == 0 or len(clothed_images) == 0:
        print("ERROR: No exemplar images found!")
        sys.exit(1)

    # --- Extract CLIP features (CLS + patches) ---
    print("\n=== Extracting CLIP image features (CLS + patches) ===")
    nude_cls, nude_patches = extract_clip_features_with_patches(nude_images, device)
    clothed_cls, clothed_patches = extract_clip_features_with_patches(clothed_images, device)

    print(f"  Nude: CLS={nude_cls.shape}, patches={nude_patches.shape}")
    print(f"  Clothed: CLS={clothed_cls.shape}, patches={clothed_patches.shape}")

    # --- Compute contrastive directions ---
    print("\n=== Computing contrastive directions ===")
    (cls_direction, patch_directions, top_k_dirs, top_k_norms,
     top_k_indices, cosine_sim) = compute_contrastive_directions(
        nude_cls, nude_patches, clothed_cls, clothed_patches, top_k=args.top_k
    )

    # --- Build probe embeddings ---
    print(f"\n=== Building probe embeddings (n_tokens={args.n_tokens}) ===")
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, safety_checker=None,
        feature_extractor=None,
    ).to(device)

    embeds_cls, embeds_patch, embeds_mixed = build_probe_embeddings(
        cls_direction, top_k_dirs,
        pipe.text_encoder, pipe.tokenizer, device, args.n_tokens
    )
    del pipe
    torch.cuda.empty_cache()

    print(f"  target_embeds_cls: {embeds_cls.shape}")
    print(f"  target_embeds_patch: {embeds_patch.shape}")
    print(f"  target_embeds_mixed: {embeds_mixed.shape}")

    # --- Save ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        # Raw directions
        "cls_direction": cls_direction.cpu().half(),                # [768]
        "patch_directions": patch_directions.cpu().half(),         # [256, 768]
        "top_k_patch_dirs": top_k_dirs.cpu().half(),               # [K, 768]
        "top_k_patch_norms": top_k_norms.cpu().half(),             # [K]
        "top_k_patch_indices": top_k_indices.cpu(),                # [K]
        # Probe embeddings (ready for cross-attention)
        "target_embeds_cls": embeds_cls.cpu().half(),              # [1, 77, 768]
        "target_embeds_patch": embeds_patch.cpu().half(),          # [1, 77, 768]
        "target_embeds_mixed": embeds_mixed.cpu().half(),          # [1, 77, 768]
        # Reference
        "cosine_sim": cosine_sim,
        # Config
        "config": {
            "n_tokens": args.n_tokens,
            "top_k": args.top_k,
            "n_nudity": len(nude_images),
            "n_clothed": len(clothed_images),
            "target_prefix": args.target_prefix,
            "anchor_prefix": args.anchor_prefix,
            "ckpt": args.ckpt,
            "exemplar_dir": args.exemplar_dir,
        }
    }

    torch.save(save_dict, output_path)
    print(f"\n=== Saved to {output_path} ===")
    print(f"  Keys: {list(save_dict.keys())}")
    print(f"  Cosine sim (nude vs clothed): {cosine_sim:.4f}")


if __name__ == "__main__":
    main()
