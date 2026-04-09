#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare CLIP Image Exemplar Embeddings for v13 Cross-Attention Probe.

Takes existing exemplar images (nude + clothed) and extracts CLIP image
embeddings, then projects them into the SD text encoder space for use
as cross-attention probe keys.

Two approaches for projection:
  1. "clip_class" — Use CLIP class token (1×768 via ViT-L/14 used by SD1.4)
     and repeat to K tokens. Simple, training-free.
  2. "textual_inversion" — Find pseudo-token embeddings that best reproduce
     the CLIP image embedding in text space. Requires optimization but more
     accurate.

We use approach 1 (clip_class) as default — completely training-free.

Usage:
    CUDA_VISIBLE_DEVICES=0 python prepare_clip_exemplar.py
    CUDA_VISIBLE_DEVICES=0 python prepare_clip_exemplar.py --exemplar_dir exemplars/sd14/exemplar_images --n_nudity 16 --n_clothed 16

Output:
    clip_exemplar_embeddings.pt containing:
      - target_clip_embeds: [1, K, 768] projected target embedding
      - anchor_clip_embeds: [1, K, 768] projected anchor embedding
      - target_clip_features: [N, 768] raw CLIP features per nudity image
      - anchor_clip_features: [N, 768] raw CLIP features per clothed image
      - config: metadata
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
    files = sorted([f for f in img_dir.iterdir() if f.name.startswith(prefix) and f.suffix == '.png'])
    for f in files[:max_images]:
        img = Image.open(f).convert('RGB')
        images.append(img)
    print(f"  Loaded {len(images)} images with prefix '{prefix}' from {exemplar_dir}")
    return images


def extract_clip_features_openclip(images, device):
    """
    Extract CLIP image features using the same CLIP model that SD1.4 uses.
    SD1.4 uses openai/clip-vit-large-patch14 which has 768-dim embeddings.
    We use the transformers CLIPModel for compatibility.
    """
    from transformers import CLIPModel, CLIPProcessor

    model_name = "openai/clip-vit-large-patch14"
    print(f"  Loading CLIP model: {model_name}")
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.eval()

    features = []
    with torch.no_grad():
        for img in tqdm(images, desc="  Extracting CLIP features"):
            inputs = clip_processor(images=img, return_tensors="pt").to(device)
            # get_image_features returns pooled output tensor [1, 768]
            outputs = clip_model.get_image_features(**inputs)
            # Handle case where outputs might be a model output object
            if hasattr(outputs, 'pooler_output'):
                outputs = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                outputs = outputs.last_hidden_state[:, 0, :]  # CLS token
            # Normalize
            feats = outputs.float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            features.append(feats.cpu())

    features = torch.cat(features, dim=0)  # [N, 768]
    del clip_model
    torch.cuda.empty_cache()
    return features


def extract_clip_features_from_generated(
    prompts, pipe, clip_model, clip_processor, device,
    n_images=16, seed=42, steps=50, cfg_scale=7.5
):
    """
    Generate images from prompts and extract CLIP features.
    Used when no pre-existing exemplar images are available.
    """
    import random
    random.seed(seed)
    torch.manual_seed(seed)

    features = []
    with torch.no_grad():
        for i, prompt in enumerate(tqdm(prompts[:n_images], desc="  Generating & extracting")):
            # Generate image
            result = pipe(
                prompt, num_inference_steps=steps, guidance_scale=cfg_scale,
                generator=torch.Generator(device=device).manual_seed(seed + i),
            )
            img = result.images[0]

            # Extract CLIP feature
            inputs = clip_processor(images=img, return_tensors="pt").to(device)
            outputs = clip_model.get_image_features(**inputs)
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
            features.append(outputs.cpu())

    return torch.cat(features, dim=0)


def project_to_text_space(clip_features, text_encoder, tokenizer, device, n_tokens=4):
    """
    Project CLIP image features into SD's text encoder embedding space.

    Approach: Since CLIP ViT-L/14 (used by SD1.4) shares the same 768-dim
    embedding space for both text and image, we can directly use the averaged
    CLIP image feature as a pseudo text token embedding.

    We create a sequence of n_tokens copies, padded with BOS/EOS/PAD to
    match the expected 77-token format.

    Args:
        clip_features: [N, 768] CLIP image features
        text_encoder: SD's CLIPTextModel
        tokenizer: SD's CLIPTokenizer
        device: torch device
        n_tokens: number of concept tokens to use (default 4)

    Returns:
        embeds: [1, 77, 768] text-space embeddings ready for to_k() projection
    """
    # Average features across all exemplar images
    avg_feature = clip_features.mean(dim=0)  # [768]
    avg_feature = avg_feature / avg_feature.norm()

    # Get the token embedding layer
    token_embedding = text_encoder.text_model.embeddings.token_embedding

    # Get BOS and EOS token embeddings
    bos_id = tokenizer.bos_token_id  # typically 49406
    eos_id = tokenizer.eos_token_id  # typically 49407
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    with torch.no_grad():
        bos_embed = token_embedding(torch.tensor([bos_id], device=device))  # [1, 768]
        eos_embed = token_embedding(torch.tensor([eos_id], device=device))
        pad_embed = token_embedding(torch.tensor([pad_id], device=device))

    # Build 77-token sequence: BOS + n_tokens * concept + EOS + padding
    concept_embed = avg_feature.unsqueeze(0).to(device)  # [1, 768]
    tokens_list = [bos_embed]  # BOS
    for _ in range(n_tokens):
        tokens_list.append(concept_embed)  # concept tokens
    tokens_list.append(eos_embed)  # EOS
    # Pad to 77
    n_pad = 77 - len(tokens_list)
    for _ in range(n_pad):
        tokens_list.append(pad_embed)

    token_embeds = torch.cat(tokens_list, dim=0).unsqueeze(0)  # [1, 77, 768]

    # Pass through text encoder's transformer (position embeddings + transformer layers)
    # to get proper contextualized embeddings
    with torch.no_grad():
        # Add position embeddings
        position_ids = torch.arange(77, device=device).unsqueeze(0)
        position_embeds = text_encoder.text_model.embeddings.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds

        # Pass through encoder layers
        # Use the text encoder's internal transformer
        encoder = text_encoder.text_model.encoder
        causal_mask = text_encoder.text_model._build_causal_attention_mask(
            1, 77, hidden_states.dtype
        ).to(device) if hasattr(text_encoder.text_model, '_build_causal_attention_mask') else None

        for layer in encoder.layers:
            layer_out = layer(hidden_states, attention_mask=causal_mask)
            if isinstance(layer_out, tuple):
                hidden_states = layer_out[0]
            else:
                hidden_states = layer_out

        # Final layer norm
        hidden_states = text_encoder.text_model.final_layer_norm(hidden_states)

    return hidden_states  # [1, 77, 768]


def project_simple(clip_features, text_encoder, tokenizer, device, n_tokens=4):
    """
    Simpler projection: just place CLIP image features as raw token embeddings
    without running through the text transformer. This preserves the raw
    image semantics better for probing purposes.

    Returns: [1, 77, 768] embeddings
    """
    avg_feature = clip_features.mean(dim=0)  # [768]
    avg_feature = avg_feature / avg_feature.norm()

    token_embedding = text_encoder.text_model.embeddings.token_embedding

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    with torch.no_grad():
        bos_embed = token_embedding(torch.tensor([bos_id], device=device))
        eos_embed = token_embedding(torch.tensor([eos_id], device=device))
        pad_embed = token_embedding(torch.tensor([pad_id], device=device))

    concept_embed = avg_feature.unsqueeze(0).to(device)
    tokens_list = [bos_embed]
    for _ in range(n_tokens):
        tokens_list.append(concept_embed)
    tokens_list.append(eos_embed)
    n_pad = 77 - len(tokens_list)
    for _ in range(n_pad):
        tokens_list.append(pad_embed)

    token_embeds = torch.cat(tokens_list, dim=0).unsqueeze(0)  # [1, 77, 768]
    return token_embeds


def project_via_text_encoding(clip_features, pipe, device, concept_text="nude person"):
    """
    Fallback approach: just use text encoding of concept_text.
    This is what v6 already does - included for comparison.
    """
    text_inputs = pipe.tokenizer(
        concept_text, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        text_embeds = pipe.text_encoder(**text_inputs).last_hidden_state  # [1, 77, 768]
    return text_embeds


def main():
    parser = ArgumentParser()
    parser.add_argument("--exemplar_dir", type=str,
                        default="exemplars/sd14/exemplar_images",
                        help="Directory with nudity_XX.png and clothed_XX.png")
    parser.add_argument("--output", type=str,
                        default="exemplars/sd14/clip_exemplar_embeddings.pt")
    parser.add_argument("--n_nudity", type=int, default=16)
    parser.add_argument("--n_clothed", type=int, default=16)
    parser.add_argument("--n_tokens", type=int, default=4,
                        help="Number of concept tokens in projected embedding")
    parser.add_argument("--projection", type=str, default="simple",
                        choices=["simple", "transformer", "text_only"],
                        help="How to project CLIP image features to text space")
    parser.add_argument("--ckpt", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--use_full_prompts", action="store_true",
                        help="Generate exemplar images from full_nudity_exemplar_prompts.txt")
    parser.add_argument("--full_prompts_path", type=str,
                        default="prompts/full_nudity_exemplar_prompts.txt")
    parser.add_argument("--n_full_samples", type=int, default=32,
                        help="Number of exemplar images to generate from full prompts")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load exemplar images ---
    if args.use_full_prompts:
        print("\n=== Generating exemplar images from Full nudity prompts ===")
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        pipe = StableDiffusionPipeline.from_pretrained(
            args.ckpt, torch_dtype=torch.float16, safety_checker=None,
            feature_extractor=None,
        ).to(device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        # Load prompts
        with open(args.full_prompts_path) as f:
            prompts = [l.strip() for l in f if l.strip()]

        # Sample subset
        import random
        random.seed(42)
        selected = random.sample(prompts, min(args.n_full_samples, len(prompts)))

        # Generate and save
        gen_dir = Path(args.exemplar_dir).parent / "exemplar_images_full"
        gen_dir.mkdir(parents=True, exist_ok=True)

        nude_images = []
        for i, prompt in enumerate(tqdm(selected, desc="Generating exemplar images")):
            result = pipe(
                prompt, num_inference_steps=50, guidance_scale=7.5,
                generator=torch.Generator(device=device).manual_seed(42 + i),
            )
            img = result.images[0]
            img.save(gen_dir / f"full_nudity_{i:03d}.png")
            nude_images.append(img)

        del pipe
        torch.cuda.empty_cache()

        # For clothed, use existing exemplar images
        clothed_images = load_exemplar_images(args.exemplar_dir, "clothed_", args.n_clothed)
    else:
        print("\n=== Loading existing exemplar images ===")
        nude_images = load_exemplar_images(args.exemplar_dir, "nudity_", args.n_nudity)
        clothed_images = load_exemplar_images(args.exemplar_dir, "clothed_", args.n_clothed)

    if len(nude_images) == 0 or len(clothed_images) == 0:
        print("ERROR: No exemplar images found!")
        sys.exit(1)

    # --- Extract CLIP features ---
    print("\n=== Extracting CLIP image features ===")
    target_features = extract_clip_features_openclip(nude_images, device)  # [N, 768]
    anchor_features = extract_clip_features_openclip(clothed_images, device)

    print(f"  Target features: {target_features.shape}")
    print(f"  Anchor features: {anchor_features.shape}")
    print(f"  Cosine sim (target mean vs anchor mean): "
          f"{F.cosine_similarity(target_features.mean(0, keepdim=True), anchor_features.mean(0, keepdim=True)).item():.4f}")

    # --- Project to text embedding space ---
    print(f"\n=== Projecting to text space (method: {args.projection}) ===")

    if args.projection == "text_only":
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            args.ckpt, torch_dtype=torch.float16, safety_checker=None,
            feature_extractor=None,
        ).to(device)
        target_embeds = project_via_text_encoding(target_features, pipe, device, "nude person")
        anchor_embeds = project_via_text_encoding(anchor_features, pipe, device, "clothed person")
        del pipe
    else:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            args.ckpt, torch_dtype=torch.float16, safety_checker=None,
            feature_extractor=None,
        ).to(device)

        # Also generate text-based embeddings for comparison
        target_text_embeds = project_via_text_encoding(target_features, pipe, device, "nude person")
        anchor_text_embeds = project_via_text_encoding(anchor_features, pipe, device, "clothed person")

        if args.projection == "simple":
            target_embeds = project_simple(
                target_features, pipe.text_encoder, pipe.tokenizer, device, args.n_tokens)
            anchor_embeds = project_simple(
                anchor_features, pipe.text_encoder, pipe.tokenizer, device, args.n_tokens)
        elif args.projection == "transformer":
            target_embeds = project_to_text_space(
                target_features, pipe.text_encoder, pipe.tokenizer, device, args.n_tokens)
            anchor_embeds = project_to_text_space(
                anchor_features, pipe.text_encoder, pipe.tokenizer, device, args.n_tokens)

        del pipe

    torch.cuda.empty_cache()

    print(f"  Target embeds: {target_embeds.shape}")
    print(f"  Anchor embeds: {anchor_embeds.shape}")

    # --- Save ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "target_clip_embeds": target_embeds.cpu().half(),      # [1, 77, 768]
        "anchor_clip_embeds": anchor_embeds.cpu().half(),
        "target_clip_features": target_features.cpu().half(),  # [N, 768]
        "anchor_clip_features": anchor_features.cpu().half(),
        "target_text_embeds": target_text_embeds.cpu().half() if args.projection != "text_only" else target_embeds.cpu().half(),
        "anchor_text_embeds": anchor_text_embeds.cpu().half() if args.projection != "text_only" else anchor_embeds.cpu().half(),
        "config": {
            "projection": args.projection,
            "n_tokens": args.n_tokens,
            "n_nudity": len(nude_images),
            "n_clothed": len(clothed_images),
            "ckpt": args.ckpt,
            "use_full_prompts": args.use_full_prompts,
        }
    }

    torch.save(save_dict, output_path)
    print(f"\n=== Saved to {output_path} ===")
    print(f"  Keys: {list(save_dict.keys())}")


if __name__ == "__main__":
    main()
