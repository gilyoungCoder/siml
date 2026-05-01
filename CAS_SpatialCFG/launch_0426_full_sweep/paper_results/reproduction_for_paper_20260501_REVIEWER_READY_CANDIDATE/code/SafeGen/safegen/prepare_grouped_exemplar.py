#!/usr/bin/env python
"""
Family-Grouped CLIP Exemplar Preparation.

Instead of mean-pooling ALL exemplar images into one vector, this script:
1. Groups exemplar images by concept family (e.g., weapon_threat, bodily_injury)
2. Mean-pools within each family separately
3. Assigns each family's embedding to a distinct token position

Result: BOS + [family1_avg, family2_avg, family3_avg, family4_avg] + EOS + PAD
Each token captures a distinct sub-aspect of the concept.

Usage:
    python -m safegen.prepare_grouped_exemplar \
        --concept violence \
        --concept_pack configs/concept_packs/violence \
        --output configs/exemplars/violence/clip_grouped.pt
"""

import os
import sys
import json
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_families(concept_pack_dir: str) -> List[dict]:
    """Load families from concept pack."""
    families_path = Path(concept_pack_dir) / "families.json"
    with open(families_path) as f:
        data = json.load(f)
    # Only use pilot families (or all if fewer than max_families)
    families = data.get("families", [])
    pilot = [f for f in families if f.get("pilot", False)]
    return pilot if pilot else families


def generate_family_exemplars(
    family: dict,
    target_prompts: List[str],
    anchor_prompts: List[str],
    pipe,
    device,
    n_images: int = 4,
    seed: int = 42,
    steps: int = 50,
    cfg_scale: float = 7.5,
) -> Tuple[List[Image.Image], List[Image.Image]]:
    """Generate exemplar images for one family."""
    target_imgs = []
    anchor_imgs = []

    for i, (tp, ap) in enumerate(zip(target_prompts[:n_images], anchor_prompts[:n_images])):
        gen = torch.Generator(device).manual_seed(seed + i)
        timg = pipe(tp, num_inference_steps=steps, guidance_scale=cfg_scale, generator=gen).images[0]
        target_imgs.append(timg)

        gen = torch.Generator(device).manual_seed(seed + i)
        aimg = pipe(ap, num_inference_steps=steps, guidance_scale=cfg_scale, generator=gen).images[0]
        anchor_imgs.append(aimg)

    return target_imgs, anchor_imgs


def extract_clip_features(images: List[Image.Image], clip_model, clip_processor, device) -> torch.Tensor:
    """Extract and normalize CLIP features from images."""
    features = []
    with torch.no_grad():
        for img in images:
            inputs = clip_processor(images=img, return_tensors="pt").to(device)
            feats = clip_model.get_image_features(**inputs).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            features.append(feats.cpu())
    return torch.cat(features, dim=0)  # [N, 768]


def build_grouped_probe_embeds(
    family_features: Dict[str, torch.Tensor],  # {family_name: [N, 768]}
    text_encoder,
    tokenizer,
    device,
    max_tokens: int = 4,
) -> Tuple[torch.Tensor, List[int], Dict[str, int]]:
    """
    Build probe embedding with family-grouped tokens.

    Each family gets its own token position. If more families than max_tokens,
    select the top families by strength (pilot families first).

    Returns:
        embeds: [1, 77, 768]
        token_indices: list of active token positions
        family_token_map: {family_name: token_position}
    """
    token_emb = text_encoder.text_model.embeddings.token_embedding

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    with torch.no_grad():
        bos = token_emb(torch.tensor([bos_id], device=device))
        eos = token_emb(torch.tensor([eos_id], device=device))
        pad = token_emb(torch.tensor([pad_id], device=device))

    # Mean-pool within each family
    family_names = list(family_features.keys())[:max_tokens]
    family_avgs = []
    family_token_map = {}

    for i, fname in enumerate(family_names):
        feats = family_features[fname]  # [N, 768]
        avg = F.normalize(feats.mean(dim=0), dim=-1)  # [768]
        family_avgs.append(avg.unsqueeze(0).to(device))
        family_token_map[fname] = i + 1  # token position (1-indexed, after BOS)

    # Build token sequence: BOS + [fam1, fam2, ...] + EOS + PAD...
    tokens = [bos]
    for avg in family_avgs:
        tokens.append(avg)

    # If fewer families than max_tokens, pad with last family's embedding
    while len(tokens) - 1 < max_tokens:
        tokens.append(family_avgs[-1])

    tokens.append(eos)

    # Pad to 77
    n_pad = 77 - len(tokens)
    for _ in range(n_pad):
        tokens.append(pad)

    embeds = torch.cat(tokens, dim=0).unsqueeze(0)  # [1, 77, 768]
    token_indices = list(range(1, 1 + min(len(family_names), max_tokens)))

    return embeds, token_indices, family_token_map


def main():
    p = ArgumentParser(description="Family-Grouped CLIP Exemplar Preparation")
    p.add_argument("--concept", required=True, help="Concept name")
    p.add_argument("--concept_pack", required=True, help="Path to concept pack directory")
    p.add_argument("--output", required=True, help="Output .pt file")
    p.add_argument("--ckpt", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--n_images_per_family", type=int, default=4,
                   help="Number of exemplar images per family")
    p.add_argument("--max_tokens", type=int, default=4,
                   help="Max token positions for families")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--use_existing_images", default=None,
                   help="Path to existing exemplar images dir (skip generation)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # Load families
    families = load_families(args.concept_pack)
    print(f"\n{'=' * 60}")
    print(f"Family-Grouped Exemplar: {args.concept}")
    print(f"Families: {[f['name'] for f in families]}")
    print(f"Images per family: {args.n_images_per_family}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"{'=' * 60}\n")

    # Load CLIP model
    from transformers import CLIPModel, CLIPProcessor
    clip_model_name = "openai/clip-vit-large-patch14"
    print(f"Loading CLIP: {clip_model_name}")
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model.eval()

    # Load SD for generation (if needed) and text encoder for projection
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    print(f"Loading SD: {args.ckpt}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.ckpt, torch_dtype=torch.float16, safety_checker=None,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Read target/anchor prompts from concept pack
    pack_dir = Path(args.concept_pack)
    target_prompts_all = [l.strip() for l in (pack_dir / "target_prompts.txt").read_text().splitlines() if l.strip()]
    anchor_prompts_all = [l.strip() for l in (pack_dir / "anchor_prompts.txt").read_text().splitlines() if l.strip()]

    # Map prompts to families based on family keywords
    # Simple heuristic: distribute prompts evenly across families
    n_fam = len(families)
    prompts_per_family = len(target_prompts_all) // n_fam

    target_family_features = {}
    anchor_family_features = {}
    family_metadata = {}

    out_dir = Path(args.output).parent / "family_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fi, family in enumerate(families[:args.max_tokens]):
        fname = family["name"]
        print(f"\n--- Family: {fname} ---")

        # Get this family's prompts
        start = fi * prompts_per_family
        end = start + args.n_images_per_family
        fam_target_prompts = target_prompts_all[start:end]
        fam_anchor_prompts = anchor_prompts_all[start:end]

        # Pad if not enough prompts
        while len(fam_target_prompts) < args.n_images_per_family:
            fam_target_prompts.append(fam_target_prompts[-1])
            fam_anchor_prompts.append(fam_anchor_prompts[-1])

        print(f"  Target prompts: {len(fam_target_prompts)}")
        for tp in fam_target_prompts:
            print(f"    - {tp[:80]}")

        # Generate images
        print(f"  Generating {args.n_images_per_family} target + anchor images...")
        target_imgs, anchor_imgs = generate_family_exemplars(
            family, fam_target_prompts, fam_anchor_prompts,
            pipe, device, args.n_images_per_family, args.seed + fi * 100,
            args.steps,
        )

        # Save images
        fam_dir = out_dir / fname
        fam_dir.mkdir(exist_ok=True)
        for i, (timg, aimg) in enumerate(zip(target_imgs, anchor_imgs)):
            timg.save(str(fam_dir / f"target_{i:02d}.png"))
            aimg.save(str(fam_dir / f"anchor_{i:02d}.png"))

        # Extract CLIP features
        print(f"  Extracting CLIP features...")
        t_feats = extract_clip_features(target_imgs, clip_model, clip_processor, device)
        a_feats = extract_clip_features(anchor_imgs, clip_model, clip_processor, device)

        target_family_features[fname] = t_feats
        anchor_family_features[fname] = a_feats
        family_metadata[fname] = {
            "target_words": family.get("target_words", []),
            "anchor_words": family.get("anchor_words", []),
            "mapping_strength": family.get("mapping_strength", "medium"),
            "n_images": len(target_imgs),
        }

        # Report cosine similarity
        t_avg = F.normalize(t_feats.mean(0, keepdim=True))
        a_avg = F.normalize(a_feats.mean(0, keepdim=True))
        sim = F.cosine_similarity(t_avg, a_avg).item()
        print(f"  Target-Anchor cosine sim: {sim:.4f}")

    # Build grouped embeddings
    print(f"\nBuilding grouped probe embeddings...")
    target_embeds, target_tok_idx, target_map = build_grouped_probe_embeds(
        target_family_features, pipe.text_encoder, pipe.tokenizer, device, args.max_tokens)
    anchor_embeds, anchor_tok_idx, anchor_map = build_grouped_probe_embeds(
        anchor_family_features, pipe.text_encoder, pipe.tokenizer, device, args.max_tokens)

    print(f"  Target token map: {target_map}")
    print(f"  Token indices: {target_tok_idx}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "target_clip_embeds": target_embeds.cpu().half(),
        "anchor_clip_embeds": anchor_embeds.cpu().half(),
        "target_clip_features": {k: v.cpu().half() for k, v in target_family_features.items()},
        "anchor_clip_features": {k: v.cpu().half() for k, v in anchor_family_features.items()},
        "target_token_indices": target_tok_idx,
        "anchor_token_indices": anchor_tok_idx,
        "family_token_map": target_map,
        "family_metadata": family_metadata,
        "config": {
            "concept": args.concept,
            "n_families": len(target_family_features),
            "n_images_per_family": args.n_images_per_family,
            "max_tokens": args.max_tokens,
            "grouped": True,
        },
    }
    torch.save(save_dict, out_path)
    print(f"\nSaved: {out_path}")
    print(f"  Families: {list(target_family_features.keys())}")
    print(f"  Token mapping: {target_map}")


if __name__ == "__main__":
    main()
