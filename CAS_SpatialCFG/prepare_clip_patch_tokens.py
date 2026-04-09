#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare CLIP Patch Token Embeddings for v15 Cross-Attention Probe.

Instead of v13's single CLS token (repeated 4x, causing representation collapse),
this script extracts all 256 CLIP ViT-L/14 patch tokens from exemplar images.
Each patch token encodes a different spatial region of the image, providing
richer spatial selectivity when used as cross-attention probe keys.

Pipeline:
  1. Load exemplar images (nudity + clothed)
  2. Extract all 256 patch tokens per image via CLIP ViT-L/14 vision model
     - last_hidden_state[:, 1:, :] skips CLS -> [1, 256, 1024]
     - visual_projection maps 1024 -> 768 (matching SD1.4 text encoder dim)
  3. Average across exemplar images per category (nude / clothed)
  4. Compute discriminative scores per patch:
     discriminative_score[i] = ||mean(nude_patches[:, i]) - mean(clothed_patches[:, i])||
  5. Select top-K most discriminative patches (body-relevant)
  6. Project selected patches into text-encoder format with BOS/EOS/PAD

Output: clip_patch_tokens.pt containing:
  - target_patches: [K, 768] top-K discriminative nude patches
  - anchor_patches: [K, 768] corresponding clothed patches
  - all_target_patches: [256, 768] full set (averaged across images)
  - all_anchor_patches: [256, 768] full set (averaged across images)
  - discriminative_scores: [256] per-patch L2 distance
  - selected_indices: [K] which patch positions were selected
  - target_embeds: [1, 77, 768] ready-to-use text-format embeddings (target)
  - anchor_embeds: [1, 77, 768] ready-to-use text-format embeddings (anchor)
  - config: metadata dict

Usage:
    CUDA_VISIBLE_DEVICES=0 python prepare_clip_patch_tokens.py
    CUDA_VISIBLE_DEVICES=0 python prepare_clip_patch_tokens.py --num_patches 32
    CUDA_VISIBLE_DEVICES=0 python prepare_clip_patch_tokens.py --selection random
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


def extract_patch_tokens(images, device):
    """
    Extract all 256 CLIP ViT-L/14 patch tokens per image.

    CLIP ViT-L/14 produces 257 tokens: [CLS, patch_1, ..., patch_256]
    where CLS is at index 0 and patches correspond to a 16x16 grid
    of the 224x224 input image. Each patch has hidden_size=1024.

    We then project through CLIP's visual_projection (1024 -> 768) to match
    SD1.4's text encoder dimension.

    Args:
        images: list of PIL Images
        device: torch device

    Returns:
        all_patches: [N, 256, 768] projected patch tokens for all images
    """
    from transformers import CLIPModel, CLIPProcessor

    model_name = "openai/clip-vit-large-patch14"
    print(f"  Loading CLIP model: {model_name}")
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.eval()

    # Get the visual projection matrix for 1024 -> 768 mapping
    visual_projection = clip_model.visual_projection  # Linear(1024, 768, bias=False)

    all_patches = []
    with torch.no_grad():
        for img in tqdm(images, desc="  Extracting patch tokens"):
            inputs = clip_processor(images=img, return_tensors="pt").to(device)

            # Use vision_model directly to get all hidden states
            vision_outputs = clip_model.vision_model(**inputs)
            # last_hidden_state: [1, 257, 1024] = [CLS, patch_1, ..., patch_256]
            last_hidden = vision_outputs.last_hidden_state

            # Skip CLS token (index 0), keep all 256 patch tokens
            patch_tokens = last_hidden[:, 1:, :]  # [1, 256, 1024]

            # Project to 768-dim via CLIP's visual_projection
            patch_tokens_proj = visual_projection(patch_tokens)  # [1, 256, 768]

            # Normalize each patch token
            patch_tokens_proj = F.normalize(patch_tokens_proj, dim=-1)

            all_patches.append(patch_tokens_proj.cpu())

    all_patches = torch.cat(all_patches, dim=0)  # [N, 256, 768]
    print(f"  Extracted patch tokens: {all_patches.shape}")

    del clip_model
    torch.cuda.empty_cache()
    return all_patches


def compute_discriminative_scores(target_patches, anchor_patches):
    """
    Compute per-patch discriminative scores.

    For each of the 256 patch positions, compute the L2 distance between
    the mean target (nude) patch and mean anchor (clothed) patch. Higher
    scores indicate patches that differ most between nude and clothed
    exemplars -- these correspond to body-relevant spatial regions.

    Args:
        target_patches: [N_t, 256, 768] target patch tokens
        anchor_patches: [N_a, 256, 768] anchor patch tokens

    Returns:
        scores: [256] per-patch discriminative scores
        mean_target: [256, 768] averaged target patches
        mean_anchor: [256, 768] averaged anchor patches
    """
    mean_target = target_patches.mean(dim=0)  # [256, 768]
    mean_anchor = anchor_patches.mean(dim=0)  # [256, 768]

    # L2 distance per patch position
    diff = mean_target - mean_anchor  # [256, 768]
    scores = torch.norm(diff, dim=-1)  # [256]

    return scores, mean_target, mean_anchor


def select_patches(scores, mean_target, mean_anchor, K, method="discriminative"):
    """
    Select top-K patches based on selection method.

    Args:
        scores: [256] discriminative scores
        mean_target: [256, 768] averaged target patches
        mean_anchor: [256, 768] averaged anchor patches
        K: number of patches to select
        method: 'discriminative' (top-K by score), 'random', or 'all'

    Returns:
        target_selected: [K, 768]
        anchor_selected: [K, 768]
        selected_indices: [K] indices of selected patches
    """
    K = min(K, 256)

    if method == "discriminative":
        # Top-K patches with highest discriminative scores
        _, indices = torch.topk(scores, K)
        indices = indices.sort()[0]  # Sort for deterministic ordering
    elif method == "random":
        indices = torch.randperm(256)[:K].sort()[0]
    elif method == "all":
        K = 256
        indices = torch.arange(256)
    else:
        raise ValueError(f"Unknown selection method: {method}")

    target_selected = mean_target[indices]  # [K, 768]
    anchor_selected = mean_anchor[indices]  # [K, 768]

    return target_selected, anchor_selected, indices


def build_text_format_embeddings(patch_tokens, text_encoder, tokenizer, device):
    """
    Build 77-token text-format embeddings from patch tokens.

    Format: [BOS, patch_1, patch_2, ..., patch_K, EOS, PAD, PAD, ...]
    where K = patch_tokens.shape[0] (must be <= 75 to fit in 77 with BOS+EOS).

    Args:
        patch_tokens: [K, 768] selected patch token embeddings
        text_encoder: SD's CLIPTextModel (for BOS/EOS/PAD embeddings)
        tokenizer: SD's CLIPTokenizer
        device: torch device

    Returns:
        embeds: [1, 77, 768] text-format embeddings
    """
    K = patch_tokens.shape[0]
    assert K <= 75, f"Too many patches ({K}) to fit in 77-token sequence with BOS+EOS"

    token_embedding = text_encoder.text_model.embeddings.token_embedding

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    with torch.no_grad():
        bos_embed = token_embedding(torch.tensor([bos_id], device=device))  # [1, 768]
        eos_embed = token_embedding(torch.tensor([eos_id], device=device))
        pad_embed = token_embedding(torch.tensor([pad_id], device=device))

    # Build sequence: BOS + K patch tokens + EOS + padding
    tokens_list = [bos_embed]
    for i in range(K):
        tokens_list.append(patch_tokens[i:i+1].to(device))  # [1, 768]
    tokens_list.append(eos_embed)

    # Pad to 77
    n_pad = 77 - len(tokens_list)
    for _ in range(n_pad):
        tokens_list.append(pad_embed)

    token_embeds = torch.cat(tokens_list, dim=0).unsqueeze(0)  # [1, 77, 768]
    return token_embeds


def main():
    parser = ArgumentParser(description="Prepare CLIP patch token embeddings for v15")
    parser.add_argument("--exemplar_dir", type=str,
                        default="exemplars/sd14/exemplar_images",
                        help="Directory with nudity_XX.png and clothed_XX.png")
    parser.add_argument("--output", type=str,
                        default="exemplars/sd14/clip_patch_tokens.pt")
    parser.add_argument("--n_nudity", type=int, default=16,
                        help="Max number of nudity exemplar images")
    parser.add_argument("--n_clothed", type=int, default=16,
                        help="Max number of clothed exemplar images")
    parser.add_argument("--target_prefix", type=str, default="nudity_",
                        help="Filename prefix for target exemplar images")
    parser.add_argument("--anchor_prefix", type=str, default="clothed_",
                        help="Filename prefix for anchor exemplar images")
    parser.add_argument("--num_patches", type=int, default=16,
                        help="Number of top-K discriminative patches to select")
    parser.add_argument("--selection", type=str, default="discriminative",
                        choices=["discriminative", "random", "all"],
                        help="Patch selection method")
    parser.add_argument("--ckpt", type=str, default="CompVis/stable-diffusion-v1-4",
                        help="SD checkpoint (for text encoder BOS/EOS/PAD tokens)")
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

    # --- Extract CLIP patch tokens ---
    print("\n=== Extracting CLIP patch tokens ===")
    target_patches = extract_patch_tokens(nude_images, device)    # [N_t, 256, 768]
    anchor_patches = extract_patch_tokens(clothed_images, device)  # [N_a, 256, 768]

    # --- Compute discriminative scores ---
    print("\n=== Computing discriminative scores ===")
    scores, mean_target, mean_anchor = compute_discriminative_scores(
        target_patches, anchor_patches
    )
    print(f"  Discriminative scores: min={scores.min():.4f}, "
          f"max={scores.max():.4f}, mean={scores.mean():.4f}")

    # Show top patches (which spatial positions are most discriminative)
    top_k_indices = torch.topk(scores, min(args.num_patches, 256))[1].sort()[0]
    # Convert to 16x16 grid positions for interpretability
    grid_positions = [(idx.item() // 16, idx.item() % 16) for idx in top_k_indices]
    print(f"  Top-{args.num_patches} discriminative patch positions (row, col):")
    print(f"    {grid_positions}")

    # --- Select patches ---
    print(f"\n=== Selecting patches (method={args.selection}, K={args.num_patches}) ===")
    target_selected, anchor_selected, selected_indices = select_patches(
        scores, mean_target, mean_anchor, args.num_patches, args.selection
    )
    print(f"  Selected target patches: {target_selected.shape}")
    print(f"  Selected anchor patches: {anchor_selected.shape}")
    print(f"  Selected indices: {selected_indices.tolist()}")

    # --- Cosine similarity analysis ---
    cos_sim = F.cosine_similarity(
        target_selected.mean(0, keepdim=True),
        anchor_selected.mean(0, keepdim=True)
    ).item()
    print(f"  Cosine sim (selected target mean vs anchor mean): {cos_sim:.4f}")

    # --- Build text-format embeddings ---
    print(f"\n=== Building text-format embeddings ===")
    K = target_selected.shape[0]
    if K > 75:
        print(f"  WARNING: K={K} > 75, clamping to 75 for text-format compatibility")
        target_selected = target_selected[:75]
        anchor_selected = anchor_selected[:75]
        selected_indices = selected_indices[:75]
        K = 75

    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, safety_checker=None,
        feature_extractor=None,
    ).to(device)

    target_embeds = build_text_format_embeddings(
        target_selected, pipe.text_encoder, pipe.tokenizer, device
    )
    anchor_embeds = build_text_format_embeddings(
        anchor_selected, pipe.text_encoder, pipe.tokenizer, device
    )
    print(f"  Target embeds: {target_embeds.shape}")
    print(f"  Anchor embeds: {anchor_embeds.shape}")

    del pipe
    torch.cuda.empty_cache()

    # --- Save ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        # Selected patches (ready to use)
        "target_patches": target_selected.cpu().half(),         # [K, 768]
        "anchor_patches": anchor_selected.cpu().half(),         # [K, 768]
        # Full patch sets (for analysis / ablation)
        "all_target_patches": mean_target.cpu().half(),         # [256, 768]
        "all_anchor_patches": mean_anchor.cpu().half(),         # [256, 768]
        # Scores and indices
        "discriminative_scores": scores.cpu().half(),           # [256]
        "selected_indices": selected_indices.cpu(),             # [K]
        # Text-format embeddings (ready for precompute_target_keys)
        "target_embeds": target_embeds.cpu().half(),            # [1, 77, 768]
        "anchor_embeds": anchor_embeds.cpu().half(),            # [1, 77, 768]
        # Metadata
        "config": {
            "num_patches": K,
            "selection": args.selection,
            "n_nudity": len(nude_images),
            "n_clothed": len(clothed_images),
            "target_prefix": args.target_prefix,
            "anchor_prefix": args.anchor_prefix,
            "ckpt": args.ckpt,
            "clip_model": "openai/clip-vit-large-patch14",
            "total_patches": 256,
            "grid_size": 16,
            "hidden_dim": 768,
        }
    }

    torch.save(save_dict, output_path)
    print(f"\n=== Saved to {output_path} ===")
    print(f"  Keys: {list(save_dict.keys())}")
    print(f"  Config: {save_dict['config']}")


if __name__ == "__main__":
    main()
