#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare CLIP exemplar embeddings for ANY concept.

For each concept:
1. Generate exemplar images from target/anchor prompts using SD1.4
2. Extract CLIP ViT-L/14 CLS features
3. Project through text encoder transformer (norm ~28-35)
4. Save as .pt file ready for v24

Usage:
  python prepare_concept_exemplars.py --concept violence
  python prepare_concept_exemplars.py --concept harassment --n_images 16
  python prepare_concept_exemplars.py --all  # prepare all concepts
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm


CONCEPT_PACKS_DIR = Path(__file__).parent.parent / "docs/neurips_plan/multi_concept/concept_packs"
OUTPUT_DIR = Path(__file__).parent / "exemplars/concepts"

CONCEPTS = ["violence", "harassment", "hate", "shocking", "illegal_activity", "self-harm"]


def load_prompts(filepath):
    with open(filepath) as f:
        return [l.strip() for l in f if l.strip()]


def generate_images(prompts, pipe, device, n_images=16, seed=42):
    """Generate images from prompts using SD pipeline."""
    import random
    random.seed(seed)
    selected = random.sample(prompts, min(n_images, len(prompts)))

    images = []
    for i, prompt in enumerate(tqdm(selected, desc="  Generating")):
        result = pipe(
            prompt, num_inference_steps=50, guidance_scale=7.5,
            generator=torch.Generator(device=device).manual_seed(seed + i),
        )
        images.append(result.images[0])
    return images


def extract_clip_features(images, device):
    """Extract CLIP ViT-L/14 CLS features."""
    from transformers import CLIPModel, CLIPProcessor

    model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.eval()

    features = []
    with torch.no_grad():
        for img in tqdm(images, desc="  CLIP features"):
            inputs = clip_processor(images=img, return_tensors="pt").to(device)
            outputs = clip_model.get_image_features(**inputs)
            if hasattr(outputs, 'pooler_output'):
                feats = outputs.pooler_output.float()
            elif hasattr(outputs, 'last_hidden_state'):
                feats = outputs.last_hidden_state[:, 0, :].float()
            elif isinstance(outputs, torch.Tensor):
                feats = outputs.float()
            else:
                feats = outputs[0].float() if isinstance(outputs, tuple) else outputs.float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            features.append(feats.cpu())

    del clip_model
    torch.cuda.empty_cache()
    return torch.cat(features, dim=0)  # [N, 768]


def project_through_transformer(clip_features, text_encoder, tokenizer, device, n_tokens=4):
    """Project CLIP features through text encoder transformer for proper norm."""
    avg = F.normalize(clip_features.mean(dim=0, keepdim=True), dim=-1)
    te_dtype = next(text_encoder.parameters()).dtype

    temb = text_encoder.text_model.embeddings.token_embedding
    bos = temb(torch.tensor([tokenizer.bos_token_id], device=device))
    eos = temb(torch.tensor([tokenizer.eos_token_id], device=device))
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad = temb(torch.tensor([pad_id], device=device))

    concept = avg.to(device=device, dtype=te_dtype)
    toks = [bos] + [concept] * n_tokens + [eos] + [pad] * (77 - n_tokens - 2)
    h = torch.cat(toks, dim=0).unsqueeze(0)

    with torch.no_grad():
        pos = text_encoder.text_model.embeddings.position_embedding(
            torch.arange(77, device=device).unsqueeze(0))
        h = h + pos
        causal = torch.full((77, 77), float("-inf"), device=device, dtype=te_dtype)
        causal = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0)
        for layer in text_encoder.text_model.encoder.layers:
            o = layer(h, attention_mask=causal, causal_attention_mask=causal)
            h = o[0] if isinstance(o, tuple) else o
        h = text_encoder.text_model.final_layer_norm(h)

    return h  # [1, 77, 768]


def prepare_concept(concept, n_images=16, ckpt="CompVis/stable-diffusion-v1-4"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pack_dir = CONCEPT_PACKS_DIR / concept
    if not pack_dir.exists():
        print(f"ERROR: concept pack not found: {pack_dir}")
        return

    out_dir = OUTPUT_DIR / concept
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "clip_exemplar_projected.pt"
    if output_file.exists():
        print(f"SKIP: {output_file} already exists")
        return

    print(f"\n{'='*60}")
    print(f"Preparing exemplars for: {concept}")
    print(f"{'='*60}")

    # Load prompts
    target_prompts = load_prompts(pack_dir / "target_prompts.txt")
    anchor_prompts = load_prompts(pack_dir / "anchor_prompts.txt")
    print(f"  Target prompts: {len(target_prompts)}")
    print(f"  Anchor prompts: {len(anchor_prompts)}")

    # Load SD pipeline
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        ckpt, torch_dtype=torch.float16, safety_checker=None,
        feature_extractor=None,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Generate images
    print(f"\n  Generating {n_images} target images...")
    target_imgs = generate_images(target_prompts, pipe, device, n_images=n_images)
    print(f"  Generating {n_images} anchor images...")
    anchor_imgs = generate_images(anchor_prompts, pipe, device, n_images=n_images)

    # Save generated images
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    for i, img in enumerate(target_imgs):
        img.save(img_dir / f"target_{i:03d}.png")
    for i, img in enumerate(anchor_imgs):
        img.save(img_dir / f"anchor_{i:03d}.png")

    te = pipe.text_encoder
    tok = pipe.tokenizer

    # Extract CLIP features
    print(f"\n  Extracting CLIP features...")
    target_features = extract_clip_features(target_imgs, device)
    anchor_features = extract_clip_features(anchor_imgs, device)
    print(f"  Target: {target_features.shape}, Anchor: {anchor_features.shape}")

    # Project through transformer
    print(f"\n  Projecting through text encoder transformer...")
    with torch.no_grad():
        target_proj = project_through_transformer(target_features, te, tok, device)
        anchor_proj = project_through_transformer(anchor_features, te, tok, device)

    print(f"  Target proj norm: {target_proj.float().norm(dim=-1).mean():.1f}")
    print(f"  Anchor proj norm: {anchor_proj.float().norm(dim=-1).mean():.1f}")

    # Save
    save_dict = {
        "target_clip_features": target_features.cpu().half(),
        "anchor_clip_features": anchor_features.cpu().half(),
        "target_clip_embeds_proj": target_proj.cpu().half(),
        "anchor_clip_embeds_proj": anchor_proj.cpu().half(),
        "config": {
            "concept": concept,
            "n_target": len(target_imgs),
            "n_anchor": len(anchor_imgs),
            "n_tokens": 4,
            "ckpt": ckpt,
        }
    }
    torch.save(save_dict, output_file)
    print(f"\n  Saved: {output_file}")

    del pipe
    torch.cuda.empty_cache()


def main():
    parser = ArgumentParser()
    parser.add_argument("--concept", type=str, default=None, choices=CONCEPTS)
    parser.add_argument("--all", action="store_true", help="Prepare all concepts")
    parser.add_argument("--n_images", type=int, default=16)
    parser.add_argument("--ckpt", default="CompVis/stable-diffusion-v1-4")
    args = parser.parse_args()

    if args.all:
        for concept in CONCEPTS:
            prepare_concept(concept, args.n_images, args.ckpt)
    elif args.concept:
        prepare_concept(args.concept, args.n_images, args.ckpt)
    else:
        parser.error("Specify --concept or --all")


if __name__ == "__main__":
    main()
