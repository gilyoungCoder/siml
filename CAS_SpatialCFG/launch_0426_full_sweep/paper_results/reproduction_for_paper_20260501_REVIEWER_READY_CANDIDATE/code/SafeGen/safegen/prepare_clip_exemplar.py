#!/usr/bin/env python
"""
Prepare CLIP Image Exemplar Embeddings for cross-attention image probe.

Takes exemplar images (e.g., nude + clothed) and extracts CLIP image features,
then projects them into the SD text encoder embedding space for use as
cross-attention probe keys.

Output: .pt file with target_clip_features, anchor_clip_features, and projected
embeddings ready for the image probe.

Usage:
    python -m safegen.prepare_clip_exemplar \
        --exemplar_dir configs/exemplars/sexual/images \
        --output configs/exemplars/sexual/clip_exemplar_projected.pt
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def load_exemplar_images(exemplar_dir, prefix, max_images=16):
    """Load exemplar images with given filename prefix."""
    img_dir = Path(exemplar_dir)
    files = sorted(f for f in img_dir.iterdir() if f.name.startswith(prefix) and f.suffix == ".png")
    images = [Image.open(f).convert("RGB") for f in files[:max_images]]
    print(f"  Loaded {len(images)} images with prefix '{prefix}'")
    return images


def extract_clip_features(images, device):
    """Extract CLIP ViT-L/14 image features (same model used by SD1.4)."""
    from transformers import CLIPModel, CLIPProcessor

    model_name = "openai/clip-vit-large-patch14"
    print(f"  Loading CLIP: {model_name}")
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.eval()

    features = []
    with torch.no_grad():
        for img in tqdm(images, desc="  Extracting CLIP features"):
            inputs = clip_processor(images=img, return_tensors="pt").to(device)
            feats = clip_model.get_image_features(**inputs).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            features.append(feats.cpu())

    del clip_model
    torch.cuda.empty_cache()
    return torch.cat(features, dim=0)


def project_simple(clip_features, text_encoder, tokenizer, device, n_tokens=4):
    """
    Simple projection: place averaged CLIP image features as raw token
    embeddings with proper BOS/EOS/PAD structure. Preserves raw image
    semantics for probing.
    """
    avg = F.normalize(clip_features.mean(dim=0), dim=-1)
    token_emb = text_encoder.text_model.embeddings.token_embedding

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    with torch.no_grad():
        bos = token_emb(torch.tensor([bos_id], device=device))
        eos = token_emb(torch.tensor([eos_id], device=device))
        pad = token_emb(torch.tensor([pad_id], device=device))

    concept = avg.unsqueeze(0).to(device)
    tokens = [bos] + [concept] * n_tokens + [eos] + [pad] * (77 - n_tokens - 2)
    return torch.cat(tokens, dim=0).unsqueeze(0)


def main():
    p = ArgumentParser(description="Prepare CLIP Exemplar Embeddings")
    p.add_argument("--exemplar_dir", required=True, help="Directory with exemplar images")
    p.add_argument("--output", required=True, help="Output .pt file path")
    p.add_argument("--target_prefix", default="nudity_", help="Filename prefix for target images")
    p.add_argument("--anchor_prefix", default="clothed_", help="Filename prefix for anchor images")
    p.add_argument("--max_images", type=int, default=16)
    p.add_argument("--n_tokens", type=int, default=4)
    p.add_argument("--ckpt", default="CompVis/stable-diffusion-v1-4")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load images
    target_imgs = load_exemplar_images(args.exemplar_dir, args.target_prefix, args.max_images)
    anchor_imgs = load_exemplar_images(args.exemplar_dir, args.anchor_prefix, args.max_images)

    if not target_imgs or not anchor_imgs:
        print("ERROR: No exemplar images found!")
        sys.exit(1)

    # Extract CLIP features
    print("\nExtracting CLIP features...")
    target_feats = extract_clip_features(target_imgs, device)
    anchor_feats = extract_clip_features(anchor_imgs, device)
    print(f"  Target: {target_feats.shape}, Anchor: {anchor_feats.shape}")
    print(f"  Cosine similarity: {F.cosine_similarity(target_feats.mean(0, keepdim=True), anchor_feats.mean(0, keepdim=True)).item():.4f}")

    # Project to text embedding space
    print("\nProjecting to text space...")
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, safety_checker=None,
    ).to(device)

    target_embeds = project_simple(target_feats, pipe.text_encoder, pipe.tokenizer, device, args.n_tokens)
    anchor_embeds = project_simple(anchor_feats, pipe.text_encoder, pipe.tokenizer, device, args.n_tokens)
    del pipe
    torch.cuda.empty_cache()

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "target_clip_embeds": target_embeds.cpu().half(),
        "anchor_clip_embeds": anchor_embeds.cpu().half(),
        "target_clip_features": target_feats.cpu().half(),
        "anchor_clip_features": anchor_feats.cpu().half(),
        "config": {
            "n_tokens": args.n_tokens,
            "n_target": len(target_imgs),
            "n_anchor": len(anchor_imgs),
        },
    }, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
